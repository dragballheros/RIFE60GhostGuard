//! Overlapped decode, Vulkan interpolation, guards, and encode.
//! Bounded channels keep a couple of frames in flight so the GPU is not idle
//! while the CPU guards or libx265 run.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use anyhow::{anyhow, Result};
use rife_guards::{clean_source_outlier, polish_inbetween_with, Frame};

use crate::ffmpeg::{tool_path, Decoder, Encoder};
use crate::rife::Engine;
use crate::schedule::{output_count, output_rate, slot_at, Rational, Slot};
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
    eprintln!(
        "rife-pc: ncnn Vulkan on \"{gpu_name}\"  {}x{}  {}/{} -> {}/{}  model {}",
        info.width,
        info.height,
        info.rate.num,
        info.rate.den,
        out_rate.num,
        out_rate.den,
        options.model.display()
    );

    let (decoded_tx, decoded_rx) = sync_channel::<Frame>(4);
    let (guard_tx, guard_rx) = sync_channel::<GuardJob>(2);
    let (encode_tx, encode_rx) = sync_channel::<Frame>(2);
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
        ) {
            *fail_decode.lock().unwrap() = Some(err.to_string());
        }
    });

    let sensitivity = options.sensitivity;
    let tile = options.tile;
    let rate = info.rate;
    let cancel_gpu = Arc::clone(&cancel);
    let fail_gpu = Arc::clone(&failure);
    let outliers = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let outliers_gpu = Arc::clone(&outliers);
    let source_seen = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let source_seen_gpu = Arc::clone(&source_seen);
    let clean_outliers = options.settings.source_outlier;
    let gpu_thread = thread::spawn(move || {
        if let Err(err) = gpu_loop(
            engine,
            decoded_rx,
            guard_tx,
            rate,
            out_rate,
            sensitivity,
            tile,
            &cancel_gpu,
            &outliers_gpu,
            &source_seen_gpu,
            clean_outliers,
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
    let expected = if info.frame_count > 0 {
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
) -> Result<()> {
    let mut decoder = Decoder::spawn(ffmpeg, input, width, height, source_width, source_height)?;
    let mut filters = crate::filters::FilterState::default();
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }
        match decoder.read_frame()? {
            Some(frame) => {
                let frame = if apply_pre {
                    crate::filters::pre_frame(&frame, settings, &mut filters)
                } else {
                    frame
                };
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
) -> Result<()> {
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
                &engine,
                &mut ready,
                &mut next_j,
                source_rate,
                out_rate,
                tile,
                false,
                &tx,
                cancel,
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
        &engine,
        &mut ready,
        &mut next_j,
        source_rate,
        out_rate,
        tile,
        true,
        &tx,
        cancel,
    )?;
    Ok(())
}

fn drain(
    engine: &Engine,
    ready: &mut VecDeque<(usize, Frame)>,
    next_j: &mut i64,
    source_rate: Rational,
    out_rate: Rational,
    tile: i32,
    eof: bool,
    tx: &SyncSender<GuardJob>,
    cancel: &AtomicBool,
) -> Result<()> {
    let Some(&(last_index, _)) = ready.back() else {
        return Ok(());
    };
    let have = last_index as i64 + 1;
    let limit = if eof {
        output_count(have, source_rate, out_rate)
    } else {
        // Slots whose timestamp is at or before the newest cleaned source frame.
        slots_ready(last_index as i64, source_rate, out_rate)
    };
    while *next_j < limit {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }
        let slot = slot_at(*next_j, have, source_rate, out_rate);
        let job = match slot {
            Slot::Source(index) => {
                let frame = frame_at(ready, index)
                    .ok_or_else(|| anyhow!("missing source frame {index}"))?;
                GuardJob::Real(frame)
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
                let mid = engine.interpolate(&previous, &next, timestep, tile)?;
                GuardJob::Synth {
                    previous,
                    mid,
                    next,
                    timestep: timestep as f64,
                    context,
                }
            }
        };
        if tx.send(job).is_err() {
            return Err(anyhow!("guard stage stopped"));
        }
        *next_j += 1;
    }
    let upcoming = slot_at(*next_j, have.max(1), source_rate, out_rate);
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
    Ok(())
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
    progress: impl Fn(Progress),
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
        encoder.write_frame(&frame)?;
        count += 1;
        produced.store(count, Ordering::Relaxed);
        if last_print.elapsed().as_millis() >= 500 {
            let seconds = started.elapsed().as_secs_f64().max(0.001);
            let fps = count as f64 / seconds;
            let remaining = expected.saturating_sub(count) as f64;
            progress(Progress {
                output_frame: count,
                output_total: expected.max(count),
                fps,
                source_frame: source_seen.load(Ordering::Relaxed),
                stage: "interpolate".into(),
                eta_seconds: if fps > 0.0 { remaining / fps } else { 0.0 },
                ram_mb: crate::perf::available_ram_mb(),
                seam: 0.0,
                fraction: count as f32 / expected.max(count) as f32,
            });
            last_print = Instant::now();
        }
    }
    encoder.finish()?;
    let seconds = started.elapsed().as_secs_f64().max(0.001);
    progress(Progress {
        output_frame: count,
        output_total: count,
        fps: count as f64 / seconds,
        source_frame: source_seen.load(Ordering::Relaxed),
        stage: "interpolate".into(),
        eta_seconds: 0.0,
        ram_mb: crate::perf::available_ram_mb(),
        seam: 0.0,
        fraction: 1.0,
    });
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
