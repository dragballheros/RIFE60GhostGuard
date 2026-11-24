//! Overlapped decode, Vulkan interpolation, guards, and encode.
//! Bounded channels keep a couple of frames in flight so the GPU is not idle
//! while the CPU guards or libx265 run.

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use anyhow::{anyhow, Result};
use rife_guards::{clean_source_outlier, polish_inbetween_with, Frame};

use crate::ffmpeg::{tool_path, Decoder, Encoder};
use crate::rife::Engine;
use crate::schedule::{output_count, output_rate, slot_at, slots_from_pts, Rational, Slot};
use crate::settings::ProcessingSettings;

#[derive(Clone, Debug)]
pub struct Options {
    pub input: PathBuf,
    pub output: PathBuf,
    pub multi: u32,
    pub fps: Option<f64>,
    pub model: PathBuf,
    pub gpu: i32,
    pub tile: i32,
    pub sensitivity: f64,
    pub ffmpeg: Option<PathBuf>,
    pub settings: ProcessingSettings,
    /// Original file whose audio is copied. Empty means `input`.
    pub audio_source: PathBuf,
}

#[derive(Clone, Debug)]
pub struct Progress {
    pub output_frame: u64,
    pub output_total: u64,
    pub fps: f64,
    pub source_frame: u64,
    pub stage: String,
    pub eta_seconds: f64,
    pub ram_mb: u64,
    pub seam: f32,
    pub fraction: f32,
    pub decode_ms: f32,
    pub rife_ms: f32,
    pub guard_ms: f32,
    pub encode_ms: f32,
}

#[derive(Clone, Debug)]
pub struct Summary {
    pub output_frames: u64,
    pub seconds: f64,
    pub fps: f64,
    pub scene_boundaries: u64,
    pub stroke_repeats: u64,
    pub source_outliers: u64,
    pub gpu_name: String,
    pub notes: Vec<String>,
    pub diagnostics_path: Option<PathBuf>,
    pub output_path: PathBuf,
    pub decode_ms: f32,
    pub rife_ms: f32,
    pub guard_ms: f32,
    pub encode_ms: f32,
}

struct StageClock {
    decode_us: std::sync::atomic::AtomicU64,
    decode_n: std::sync::atomic::AtomicU64,
    rife_us: std::sync::atomic::AtomicU64,
    rife_n: std::sync::atomic::AtomicU64,
    guard_us: std::sync::atomic::AtomicU64,
    guard_n: std::sync::atomic::AtomicU64,
    encode_us: std::sync::atomic::AtomicU64,
    encode_n: std::sync::atomic::AtomicU64,
}

impl StageClock {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            decode_us: std::sync::atomic::AtomicU64::new(0),
            decode_n: std::sync::atomic::AtomicU64::new(0),
            rife_us: std::sync::atomic::AtomicU64::new(0),
            rife_n: std::sync::atomic::AtomicU64::new(0),
            guard_us: std::sync::atomic::AtomicU64::new(0),
            guard_n: std::sync::atomic::AtomicU64::new(0),
            encode_us: std::sync::atomic::AtomicU64::new(0),
            encode_n: std::sync::atomic::AtomicU64::new(0),
        })
    }

    fn add(total: &std::sync::atomic::AtomicU64, count: &std::sync::atomic::AtomicU64, started: Instant) {
        total.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        count.fetch_add(1, Ordering::Relaxed);
    }

    fn avg(total: &std::sync::atomic::AtomicU64, count: &std::sync::atomic::AtomicU64) -> f32 {
        let count = count.load(Ordering::Relaxed);
        if count == 0 {
            0.0
        } else {
            total.load(Ordering::Relaxed) as f32 / count as f32 / 1000.0
        }
    }

    fn apply(&self, progress: &mut Progress) {
        progress.decode_ms = Self::avg(&self.decode_us, &self.decode_n);
        progress.rife_ms = Self::avg(&self.rife_us, &self.rife_n);
        progress.guard_ms = Self::avg(&self.guard_us, &self.guard_n);
        progress.encode_ms = Self::avg(&self.encode_us, &self.encode_n);
    }
}

enum GuardJob {
    Real(Frame),
    Synth {
        previous: Frame,
        mid: Frame,
        next: Frame,
        timestep: f64,
        context: Option<Frame>,
    },
}

struct CleanWindow {
    previous: Option<(usize, Frame)>,
    pending: Option<(usize, Frame)>,
    next_index: usize,
}

impl CleanWindow {
    fn new() -> Self {
        Self {
            previous: None,
            pending: None,
            next_index: 0,
        }
    }

    fn push(&mut self, frame: Frame, sensitivity: f64, outliers: &mut u64, clean: bool) -> Vec<(usize, Frame)> {
        let index = self.next_index;
        self.next_index += 1;
        if self.pending.is_none() {
            self.pending = Some((index, frame));
            return Vec::new();
        }
        if self.previous.is_none() {
            self.previous = self.pending.take();
            self.pending = Some((index, frame));
            return Vec::new();
        }
        let (a_index, a) = self.previous.take().unwrap();
        let (b_index, b) = self.pending.take().unwrap();
        let before = b.data.clone();
        let corrected = if clean {
            clean_source_outlier(&a, &b, &frame, 0.5, sensitivity)
        } else {
            b
        };
        if clean && corrected.data != before {
            *outliers += 1;
        }
        self.previous = Some((b_index, corrected));
        self.pending = Some((index, frame));
        vec![(a_index, a)]
    }

    fn finish(&mut self) -> Vec<(usize, Frame)> {
        let mut out = Vec::new();
        if let Some(previous) = self.previous.take() {
            out.push(previous);
        }
        if let Some(pending) = self.pending.take() {
            out.push(pending);
        }
        out
    }
}

struct RifeWork {
    sequence: u64,
    previous: Frame,
    next: Frame,
    context: Option<Frame>,
    timestep: f32,
    tile: i32,
}

enum RifeMessage {
    Run(RifeWork),
    Stop,
}

struct RifeResult {
    sequence: u64,
    job: Result<GuardJob>,
}

/// Ordered, bounded worker pool. Inference may finish out of order, but jobs
/// are emitted to GhostGuard in the exact output-timestamp order.
struct RifeWorkerPool {
    task_tx: SyncSender<RifeMessage>,
    result_rx: Receiver<RifeResult>,
    workers: Vec<thread::JoinHandle<()>>,
    outstanding: usize,
    max_outstanding: usize,
    next_sequence: u64,
    next_emit: u64,
    ready: BTreeMap<u64, Result<GuardJob>>,
}

impl RifeWorkerPool {
    fn new(engine: Engine, worker_count: usize, clock: Arc<StageClock>) -> Self {
        let worker_count = worker_count.clamp(1, 2);
        let max_outstanding = (worker_count * 2).max(2);
        let (task_tx, task_rx) = sync_channel::<RifeMessage>(max_outstanding);
        let (result_tx, result_rx) = sync_channel::<RifeResult>(max_outstanding);
        let shared_rx = Arc::new(Mutex::new(task_rx));
        let engine = Arc::new(engine);
        let mut workers = Vec::with_capacity(worker_count);

        for _ in 0..worker_count {
            let rx = Arc::clone(&shared_rx);
            let tx = result_tx.clone();
            let engine = Arc::clone(&engine);
            let clock = Arc::clone(&clock);
            workers.push(thread::spawn(move || loop {
                let message = match rx.lock() {
                    Ok(receiver) => receiver.recv(),
                    Err(_) => break,
                };
                let work = match message {
                    Ok(RifeMessage::Run(work)) => work,
                    Ok(RifeMessage::Stop) | Err(_) => break,
                };
                let sequence = work.sequence;
                let started = Instant::now();
                let job = engine
                    .interpolate(&work.previous, &work.next, work.timestep, work.tile)
                    .map(|mid| GuardJob::Synth {
                        previous: work.previous,
                        mid,
                        next: work.next,
                        timestep: work.timestep as f64,
                        context: work.context,
                    });
                StageClock::add(&clock.rife_us, &clock.rife_n, started);
                if tx.send(RifeResult { sequence, job }).is_err() {
                    break;
                }
            }));
        }
        drop(result_tx);

        Self {
            task_tx,
            result_rx,
            workers,
            outstanding: 0,
            max_outstanding,
            next_sequence: 0,
            next_emit: 0,
            ready: BTreeMap::new(),
        }
    }

    fn submit_real(&mut self, frame: Frame, tx: &SyncSender<GuardJob>) -> Result<()> {
        self.collect_available()?;
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        self.ready.insert(sequence, Ok(GuardJob::Real(frame)));
        self.flush_ready(tx)
    }

    fn submit_interpolation(
        &mut self,
        previous: Frame,
        next: Frame,
        context: Option<Frame>,
        timestep: f32,
        tile: i32,
        tx: &SyncSender<GuardJob>,
    ) -> Result<()> {
        while self.outstanding >= self.max_outstanding {
            self.collect_one()?;
            self.flush_ready(tx)?;
        }

        let sequence = self.next_sequence;
        self.next_sequence += 1;
        self.task_tx
            .send(RifeMessage::Run(RifeWork {
                sequence,
                previous,
                next,
                context,
                timestep,
                tile,
            }))
            .map_err(|_| anyhow!("RIFE worker queue stopped"))?;
        self.outstanding += 1;
        self.collect_available()?;
        self.flush_ready(tx)
    }

    fn collect_available(&mut self) -> Result<()> {
        loop {
            match self.result_rx.try_recv() {
                Ok(result) => self.accept_result(result)?,
                Err(TryRecvError::Empty) => return Ok(()),
                Err(TryRecvError::Disconnected) => {
                    if self.outstanding != 0 {
                        return Err(anyhow!("RIFE workers stopped before returning all frames"));
                    }
                    return Ok(());
                }
            }
        }
    }

    fn collect_one(&mut self) -> Result<()> {
        let result = self
            .result_rx
            .recv()
            .map_err(|_| anyhow!("RIFE worker result channel closed unexpectedly"))?;
        self.accept_result(result)
    }

    fn accept_result(&mut self, result: RifeResult) -> Result<()> {
        self.outstanding = self
            .outstanding
            .checked_sub(1)
            .ok_or_else(|| anyhow!("received an unexpected RIFE worker result"))?;
        if self.ready.insert(result.sequence, result.job).is_some() {
            return Err(anyhow!("duplicate RIFE worker result sequence"));
        }
        Ok(())
    }

    fn flush_ready(&mut self, tx: &SyncSender<GuardJob>) -> Result<()> {
        while let Some(result) = self.ready.remove(&self.next_emit) {
            let job = result?;
            tx.send(job)
                .map_err(|_| anyhow!("guard stage stopped"))?;
            self.next_emit += 1;
        }
        Ok(())
    }

    fn finish(&mut self, tx: &SyncSender<GuardJob>) -> Result<()> {
        while self.outstanding > 0 {
            self.collect_one()?;
            self.flush_ready(tx)?;
        }
        self.flush_ready(tx)?;
        if self.next_emit != self.next_sequence {
            return Err(anyhow!("RIFE output ordering gap at sequence {}", self.next_emit));
        }

        for _ in 0..self.workers.len() {
            self.task_tx
                .send(RifeMessage::Stop)
                .map_err(|_| anyhow!("could not stop RIFE worker"))?;
        }
        for worker in self.workers.drain(..) {
            if worker.join().is_err() {
                return Err(anyhow!("RIFE worker thread panicked"));
            }
        }
        Ok(())
    }
}

impl Drop for RifeWorkerPool {
    fn drop(&mut self) {
        // Also shut down cleanly on cancellation or an inference error. The
        // result queue is sized to the maximum outstanding work, so workers
        // can finish already-submitted calls while this owner joins them.
        for _ in 0..self.workers.len() {
            let _ = self.task_tx.send(RifeMessage::Stop);
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn effective_rife_workers(requested: u8, tile: i32, gpu_memory_mb: u64) -> usize {
    // A 1080p whole-frame RIFE inference needs enough Vulkan device-local
    // memory for the model, activations, command buffers, and the second
    // concurrent inference. On a ~4 GB Radeon this is not safe: two workers
    // can reach VK_ERROR_OUT_OF_DEVICE_MEMORY even though one worker is fine.
    // Treat "2" as a maximum request, not a promise to oversubscribe the GPU.
    let can_run_two = (tile > 0 && tile <= 512) || gpu_memory_mb >= 7 * 1024;
    match requested {
        1 => 1,
        2 if can_run_two => 2,
        2 => 1,
        _ if can_run_two => 2,
        _ => 1,
    }
}

pub fn run(
    options: &Options,
    cancel: Arc<AtomicBool>,
    progress: impl Fn(Progress) + Send + Sync + 'static,
) -> Result<Summary> {
    crate::passes::orchestrate(options, cancel, progress)
}

pub(crate) fn interpolate(
    options: &Options,
    cancel: Arc<AtomicBool>,
    progress: impl Fn(Progress) + Send + 'static,
    apply_pre: bool,
    apply_post: bool,
    lossless: bool,
) -> Result<Summary> {
    if options.input == options.output {
        return Err(anyhow!("input and output are the same path"));
    }
    if options.multi == 0 || options.multi > 8 {
        return Err(anyhow!("--multi must be from 1 to 8"));
    }
    if !options.model.join("flownet.param").is_file()
        || !options.model.join("flownet.bin").is_file()
    {
        return Err(anyhow!(
            "RIFE model not found in {}. Download rife-v4.6 (flownet.param and flownet.bin). See rife-pc/README.md.",
            options.model.display()
        ));
    }
    let ffmpeg = tool_path(options.ffmpeg.as_deref(), bin_name("ffmpeg"));
    let ffprobe = tool_path(
        options
            .ffmpeg
            .as_deref()
            .and_then(|p| p.parent())
            .map(|d| d.join(bin_name("ffprobe")))
            .as_deref(),
        bin_name("ffprobe"),
    );
    let info = crate::ffmpeg::probe(&ffprobe, &options.input)?;
    let out_rate = output_rate(info.rate, options.multi, options.fps);
    let engine = Engine::open(&options.model, options.gpu)?;
    let gpu_name = engine.gpu_name.clone();
    let rife_tile = crate::perf::rife_tile(
        options.settings.performance_tier,
        options.tile,
        info.width,
        info.height,
        engine.gpu_memory_mb,
    );
    let rife_workers = effective_rife_workers(
        options.settings.rife_workers,
        rife_tile,
        engine.gpu_memory_mb,
    );
    let target_fps = out_rate.num as f64 / out_rate.den.max(1) as f64;
    let slots = match crate::ffmpeg::frame_pts(&ffprobe, &options.input) {
        Ok(pts) if pts.len() >= 2 && pts.windows(2).all(|pair| pair[1] > pair[0] + 1e-5) => {
            Some(Arc::new(slots_from_pts(&pts, target_fps)))
        }
        _ => None,
    };
    let slot_count = slots.as_ref().map(|slots| slots.len());
    eprintln!(
        "rife-pc: ncnn Vulkan on \"{gpu_name}\"  {}x{}  {}/{} -> {}/{}  model {}  Vulkan VRAM {} MB  RIFE tile {}  workers {} (setting {})",
        info.width,
        info.height,
        info.rate.num,
        info.rate.den,
        out_rate.num,
        out_rate.den,
        options.model.display(),
        engine.gpu_memory_mb,
        if rife_tile <= 0 { "full".to_string() } else { rife_tile.to_string() },
        rife_workers,
        options.settings.rife_workers
    );

    let (decoded_tx, decoded_rx) = sync_channel::<Frame>(4);
    let (guard_tx, guard_rx) = sync_channel::<GuardJob>(2);
    let (encode_tx, encode_rx) = sync_channel::<Frame>(2);
    let clock = StageClock::new();
    let failure: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let started = Instant::now();
    let stats_scene = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let stats_stroke = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let produced = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let input = options.input.clone();
    let settings_decode = options.settings.clone();
    let notes_decode: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let notes_guard = Arc::clone(&notes_decode);
    let width = info.width & !1;
    let height = info.height & !1;
    let source_width = info.width;
    let source_height = info.height;
    let ffmpeg_decode = ffmpeg.clone();
    let cancel_decode = Arc::clone(&cancel);
    let fail_decode = Arc::clone(&failure);
    let notes_for_decode = Arc::clone(&notes_decode);
    let clock_decode = Arc::clone(&clock);
    let decode_thread = thread::spawn(move || {
        if let Err(err) = decode_loop(
            &ffmpeg_decode,
            &input,
            width,
            height,
            source_width,
            source_height,
            decoded_tx,
            &cancel_decode,
            &settings_decode,
            apply_pre,
            &notes_for_decode,
            &clock_decode,
        ) {
            *fail_decode.lock().unwrap() = Some(err.to_string());
        }
    });

    let sensitivity = options.sensitivity;
    let rate = info.rate;
    let cancel_gpu = Arc::clone(&cancel);
    let fail_gpu = Arc::clone(&failure);
    let outliers = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let outliers_gpu = Arc::clone(&outliers);
    let source_seen = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let source_seen_gpu = Arc::clone(&source_seen);
    let clean_outliers = options.settings.source_outlier;
    let clock_gpu = Arc::clone(&clock);
    let gpu_thread = thread::spawn(move || {
        if let Err(err) = gpu_loop(
            engine,
            decoded_rx,
            guard_tx,
            rate,
            out_rate,
            sensitivity,
            rife_tile,
            &cancel_gpu,
            &outliers_gpu,
            &source_seen_gpu,
            clean_outliers,
            rife_workers,
            slots,
            clock_gpu,
        ) {
            *fail_gpu.lock().unwrap() = Some(err.to_string());
        }
    });

    let sensitivity_guard = options.sensitivity;
    let cancel_guard = Arc::clone(&cancel);
    let fail_guard = Arc::clone(&failure);
    let scene = Arc::clone(&stats_scene);
    let stroke = Arc::clone(&stats_stroke);
    let settings_guard = options.settings.clone();
    let clock_guard = Arc::clone(&clock);
    let guard_thread = thread::spawn(move || {
        if let Err(err) = guard_loop(
            guard_rx,
            encode_tx,
            sensitivity_guard,
            &cancel_guard,
            &scene,
            &stroke,
            &settings_guard,
            apply_post,
            &notes_guard,
            &clock_guard,
        ) {
            *fail_guard.lock().unwrap() = Some(err.to_string());
        }
    });

    let output = options.output.clone();
    let source = if options.audio_source.as_os_str().is_empty() {
        options.input.clone()
    } else {
        options.audio_source.clone()
    };
    let preserve_audio = options.settings.preserve_audio && !lossless;
    let ffmpeg_enc = ffmpeg.clone();
    let color = info.color.clone();
    let expected = if let Some(count) = slot_count {
        count as u64
    } else if info.frame_count > 0 {
        output_count(info.frame_count, info.rate, out_rate) as u64
    } else if info.duration > 0.0 {
        (info.duration * out_rate.num as f64 / out_rate.den as f64)
            .round()
            .max(1.0) as u64
    } else {
        0
    };
    let cancel_enc = Arc::clone(&cancel);
    let fail_enc = Arc::clone(&failure);
    let produced_enc = Arc::clone(&produced);
    let source_for_progress = Arc::clone(&source_seen);
    let clock_encode = Arc::clone(&clock);
    let encode_thread = thread::spawn(move || {
        let result = encode_loop(
            &ffmpeg_enc,
            &source,
            &output,
            width,
            height,
            out_rate,
            &color,
            encode_rx,
            expected,
            &cancel_enc,
            &produced_enc,
            &source_for_progress,
            started,
            preserve_audio,
            lossless,
            progress,
            &clock_encode,
        );
        if let Err(err) = result {
            *fail_enc.lock().unwrap() = Some(err.to_string());
        }
    });

    let _ = decode_thread.join();
    let _ = gpu_thread.join();
    let _ = guard_thread.join();
    let _ = encode_thread.join();
    if let Some(err) = failure.lock().unwrap().take() {
        return Err(anyhow!(err));
    }
    let frames = produced.load(Ordering::Relaxed);
    let seconds = started.elapsed().as_secs_f64().max(0.001);
    let notes = notes_decode.lock().unwrap().clone();
    Ok(Summary {
        output_frames: frames,
        seconds,
        fps: frames as f64 / seconds,
        scene_boundaries: stats_scene.load(Ordering::Relaxed),
        stroke_repeats: stats_stroke.load(Ordering::Relaxed),
        source_outliers: outliers.load(Ordering::Relaxed),
        gpu_name,
        notes,
        diagnostics_path: None,
        output_path: options.output.clone(),
        decode_ms: StageClock::avg(&clock.decode_us, &clock.decode_n),
        rife_ms: StageClock::avg(&clock.rife_us, &clock.rife_n),
        guard_ms: StageClock::avg(&clock.guard_us, &clock.guard_n),
        encode_ms: StageClock::avg(&clock.encode_us, &clock.encode_n),
    })
}

fn decode_loop(
    ffmpeg: &std::path::Path,
    input: &std::path::Path,
    width: i32,
    height: i32,
    source_width: i32,
    source_height: i32,
    tx: SyncSender<Frame>,
    cancel: &AtomicBool,
    settings: &ProcessingSettings,
    apply_pre: bool,
    notes: &Mutex<Vec<String>>,
    clock: &StageClock,
) -> Result<()> {
    let mut decoder = Decoder::spawn(ffmpeg, input, width, height, source_width, source_height)?;
    let mut filters = crate::filters::FilterState::default();
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }
        let started = Instant::now();
        match decoder.read_frame()? {
            Some(frame) => {
                let frame = if apply_pre {
                    crate::filters::pre_frame(&frame, settings, &mut filters)
                } else {
                    frame
                };
                StageClock::add(&clock.decode_us, &clock.decode_n, started);
                if tx.send(frame).is_err() {
                    notes.lock().unwrap().extend(filters.notes.drain(..));
                    return Ok(());
                }
            }
            None => break,
        }
    }
    notes.lock().unwrap().extend(filters.notes);
    decoder.finish()?;
    Ok(())
}

fn gpu_loop(
    engine: Engine,
    rx: Receiver<Frame>,
    tx: SyncSender<GuardJob>,
    source_rate: Rational,
    out_rate: Rational,
    sensitivity: f64,
    tile: i32,
    cancel: &AtomicBool,
    outliers: &std::sync::atomic::AtomicU64,
    source_seen: &std::sync::atomic::AtomicU64,
    clean_outliers: bool,
    worker_count: usize,
    slots: Option<Arc<Vec<Slot>>>,
    clock: Arc<StageClock>,
) -> Result<()> {
    let mut worker_pool = RifeWorkerPool::new(engine, worker_count, Arc::clone(&clock));
    let mut window = CleanWindow::new();
    let mut ready: VecDeque<(usize, Frame)> = VecDeque::new();
    let mut next_j = 0i64;
    let mut outlier_count = 0u64;
    while let Ok(frame) = rx.recv() {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }
        for cleaned in window.push(frame, sensitivity, &mut outlier_count, clean_outliers) {
            ready.push_back(cleaned);
            source_seen.store(
                ready.back().map(|(i, _)| *i as u64 + 1).unwrap_or(0),
                Ordering::Relaxed,
            );
            drain(
                &mut worker_pool,
                &mut ready,
                &mut next_j,
                source_rate,
                out_rate,
                tile,
                false,
                &tx,
                cancel,
                slots.as_ref().map(|slots| slots.as_slice()),
            )?;
        }
    }
    for cleaned in window.finish() {
        ready.push_back(cleaned);
    }
    let n = ready.back().map(|(i, _)| *i as i64 + 1).unwrap_or(0);
    source_seen.store(n as u64, Ordering::Relaxed);
    outliers.store(outlier_count, Ordering::Relaxed);
    if n == 0 {
        return Err(anyhow!("the input has no video frames"));
    }
    drain(
        &mut worker_pool,
        &mut ready,
        &mut next_j,
        source_rate,
        out_rate,
        tile,
        true,
        &tx,
        cancel,
        slots.as_ref().map(|slots| slots.as_slice()),
    )?;
    Ok(())
}

fn drain(
    worker_pool: &mut RifeWorkerPool,
    ready: &mut VecDeque<(usize, Frame)>,
    next_j: &mut i64,
    source_rate: Rational,
    out_rate: Rational,
    tile: i32,
    eof: bool,
    tx: &SyncSender<GuardJob>,
    cancel: &AtomicBool,
    pts_slots: Option<&[Slot]>,
) -> Result<()> {
    let Some(&(last_index, _)) = ready.back() else {
        return Ok(());
    };
    let have = last_index as i64 + 1;
    let limit = if let Some(pts_slots) = pts_slots {
        pts_limit(pts_slots, *next_j, have)
    } else if eof {
        output_count(have, source_rate, out_rate)
    } else {
        // Slots whose timestamp is at or before the newest cleaned source frame.
        slots_ready(last_index as i64, source_rate, out_rate)
    };
    while *next_j < limit {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }
        let slot = if let Some(pts_slots) = pts_slots {
            pts_slots
                .get(*next_j as usize)
                .copied()
                .ok_or_else(|| anyhow!("missing output slot {}", *next_j))?
        } else {
            slot_at(*next_j, have, source_rate, out_rate)
        };
        match slot {
            Slot::Source(index) => {
                let frame = frame_at(ready, index)
                    .ok_or_else(|| anyhow!("missing source frame {index}"))?;
                worker_pool.submit_real(frame, tx)?;
            }
            Slot::Interp { left, timestep } => {
                let previous =
                    frame_at(ready, left).ok_or_else(|| anyhow!("missing source frame {left}"))?;
                let next = frame_at(ready, left + 1)
                    .ok_or_else(|| anyhow!("missing source frame {}", left + 1))?;
                let context = if left > 0 {
                    frame_at(ready, left - 1)
                } else {
                    None
                };
                worker_pool.submit_interpolation(previous, next, context, timestep, tile, tx)?;
            }
        }
        *next_j += 1;
    }
    let upcoming = if let Some(pts_slots) = pts_slots {
        pts_slots
            .get(*next_j as usize)
            .copied()
            .unwrap_or(Slot::Source(have.max(1) as usize))
    } else {
        slot_at(*next_j, have.max(1), source_rate, out_rate)
    };
    let oldest_index = match upcoming {
        Slot::Source(index) => index.saturating_sub(1),
        Slot::Interp { left, .. } => left.saturating_sub(1),
    };
    while ready
        .front()
        .map(|(index, _)| *index < oldest_index)
        .unwrap_or(false)
    {
        ready.pop_front();
    }
    if eof {
        worker_pool.finish(tx)?;
    }
    Ok(())
}

fn pts_limit(slots: &[Slot], next_j: i64, have: i64) -> i64 {
    let mut index = next_j.max(0) as usize;
    while index < slots.len() {
        let ready = match slots[index] {
            Slot::Source(frame) => (frame as i64) < have,
            Slot::Interp { left, .. } => (left as i64) + 1 < have,
        };
        if !ready {
            break;
        }
        index += 1;
    }
    index as i64
}

fn slots_ready(last_index: i64, source_rate: Rational, out_rate: Rational) -> i64 {
    // Count of output slots with timestamp <= the newest cleaned source frame.
    let den = out_rate.den * source_rate.num;
    if den == 0 {
        return 0;
    }
    let num = last_index * source_rate.den * out_rate.num;
    num / den + 1
}

fn frame_at(ready: &VecDeque<(usize, Frame)>, index: usize) -> Option<Frame> {
    ready
        .iter()
        .find(|(i, _)| *i == index)
        .map(|(_, frame)| frame.clone())
}

fn guard_loop(
    rx: Receiver<GuardJob>,
    tx: SyncSender<Frame>,
    sensitivity: f64,
    cancel: &AtomicBool,
    scene: &std::sync::atomic::AtomicU64,
    stroke: &std::sync::atomic::AtomicU64,
    settings: &ProcessingSettings,
    apply_post: bool,
    notes: &Mutex<Vec<String>>,
    clock: &StageClock,
) -> Result<()> {
    let flags = settings.guard_flags();
    let mut filters = crate::filters::FilterState::default();
    let mut index = 0u64;
    let diag_dir = crate::models::data_dir().join("Diagnostics");
    let mut diag = crate::diagnostics::Recorder::new(diag_dir);
    while let Ok(job) = rx.recv() {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }
        let started = Instant::now();
        let frame = match job {
            GuardJob::Real(frame) => frame,
            GuardJob::Synth {
                previous,
                mid,
                next,
                timestep,
                context,
            } => {
                let (frame, stats) = polish_inbetween_with(
                    &previous,
                    &mid,
                    &next,
                    timestep,
                    sensitivity,
                    context.as_ref(),
                    &flags,
                );
                if stats.scene_boundary {
                    scene.fetch_add(1, Ordering::Relaxed);
                }
                if stats.stroke_repeat {
                    stroke.fetch_add(1, Ordering::Relaxed);
                }
                frame
            }
        };
        let frame = if apply_post {
            crate::filters::post_frame(&frame, settings, &mut filters)
        } else {
            frame
        };
        if settings.diagnostics {
            diag.observe("output", index, &frame);
        }
        StageClock::add(&clock.guard_us, &clock.guard_n, started);
        index += 1;
        if tx.send(frame).is_err() {
            break;
        }
    }
    notes.lock().unwrap().extend(filters.notes);
    if settings.diagnostics {
        if let Ok(path) = diag.finish(0.0) {
            notes.lock().unwrap().push(format!("Diagnostics: {}", path.display()));
        }
    }
    Ok(())
}

fn encode_loop(
    ffmpeg: &std::path::Path,
    source: &std::path::Path,
    output: &std::path::Path,
    width: i32,
    height: i32,
    out_rate: Rational,
    color: &crate::x265::ColorMeta,
    rx: Receiver<Frame>,
    expected: u64,
    cancel: &AtomicBool,
    produced: &std::sync::atomic::AtomicU64,
    source_seen: &std::sync::atomic::AtomicU64,
    started: Instant,
    preserve_audio: bool,
    lossless: bool,
    mut progress: impl FnMut(Progress),
    clock: &StageClock,
) -> Result<()> {
    let mut encoder = if lossless {
        Encoder::spawn_ffv1(ffmpeg, output, width, height, out_rate.num, out_rate.den)?
    } else {
        Encoder::spawn_audio(
            ffmpeg,
            source,
            output,
            width,
            height,
            out_rate.num,
            out_rate.den,
            color,
            preserve_audio,
        )?
    };
    let mut count = 0u64;
    let mut last_print = Instant::now();
    while let Ok(frame) = rx.recv() {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }
        let frame_started = Instant::now();
        encoder.write_frame(&frame)?;
        StageClock::add(&clock.encode_us, &clock.encode_n, frame_started);
        count += 1;
        produced.store(count, Ordering::Relaxed);
        if last_print.elapsed().as_millis() >= 500 {
            let seconds = started.elapsed().as_secs_f64().max(0.001);
            let fps = count as f64 / seconds;
            let remaining = expected.saturating_sub(count) as f64;
            let mut event = Progress {
                output_frame: count,
                output_total: expected.max(count),
                fps,
                source_frame: source_seen.load(Ordering::Relaxed),
                stage: "encode".into(),
                eta_seconds: if fps > 0.0 { remaining / fps } else { 0.0 },
                ram_mb: crate::perf::available_ram_mb(),
                seam: 0.0,
                fraction: count as f32 / expected.max(count) as f32,
                decode_ms: 0.0,
                rife_ms: 0.0,
                guard_ms: 0.0,
                encode_ms: 0.0,
            };
            clock.apply(&mut event);
            progress(event);
            last_print = Instant::now();
        }
    }
    encoder.finish()?;
    let seconds = started.elapsed().as_secs_f64().max(0.001);
    let mut event = Progress {
        output_frame: count,
        output_total: count,
        fps: count as f64 / seconds,
        source_frame: source_seen.load(Ordering::Relaxed),
        stage: "encode".into(),
        eta_seconds: 0.0,
        ram_mb: crate::perf::available_ram_mb(),
        seam: 0.0,
        fraction: 1.0,
        decode_ms: 0.0,
        rife_ms: 0.0,
        guard_ms: 0.0,
        encode_ms: 0.0,
    };
    clock.apply(&mut event);
    progress(event);
    Ok(())
}

fn bin_name(stem: &str) -> &'static str {
    match stem {
        "ffmpeg" if cfg!(windows) => "ffmpeg.exe",
        "ffprobe" if cfg!(windows) => "ffprobe.exe",
        "ffmpeg" => "ffmpeg",
        _ => "ffprobe",
    }
}


#[cfg(test)]
mod rife_worker_tests {
    use super::effective_rife_workers;

    #[test]
    fn auto_uses_two_workers_for_small_tiles() {
        assert_eq!(effective_rife_workers(0, 512, 4096), 2);
        assert_eq!(effective_rife_workers(0, 256, 4096), 2);
    }

    #[test]
    fn auto_keeps_whole_frame_and_large_tiles_single_worker_on_4gb_cards() {
        assert_eq!(effective_rife_workers(0, 0, 4096), 1);
        assert_eq!(effective_rife_workers(0, 768, 4096), 1);
    }

    #[test]
    fn auto_uses_two_workers_for_8gb_whole_frame_gpu() {
        assert_eq!(effective_rife_workers(0, 0, 8192), 2);
        assert_eq!(effective_rife_workers(0, 768, 8192), 2);
    }

    #[test]
    fn manual_worker_selection_is_respected_when_safe() {
        assert_eq!(effective_rife_workers(1, 512, 8192), 1);
        assert_eq!(effective_rife_workers(2, 0, 8192), 2);
    }

    #[test]
    fn manual_two_workers_are_capped_on_4gb_whole_frame_gpu() {
        assert_eq!(effective_rife_workers(2, 0, 4096), 1);
        assert_eq!(effective_rife_workers(2, 1080, 4096), 1);
    }

    #[test]
    fn manual_two_workers_remain_available_for_small_tiles() {
        assert_eq!(effective_rife_workers(2, 512, 4096), 2);
    }
}
