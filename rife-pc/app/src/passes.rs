//! Stage order from the iOS two-pass processor, minus Wan VACE.
//! Anime Restore, watermark removal, and Compression Guard run on the source.
//! Real-CUGAN is either before RIFE (`upscale_first`) or after it.
//! Color Rebuild, Outline Rebuild, Sharpie, and Color Pop run on the last picture.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use rife_guards::Frame;

use crate::ffmpeg::{self, tool_path, Decoder, Encoder};
use crate::perf::{self, Awake};
use crate::pipeline::{self, Options, Progress, Summary};
use crate::scratch::Scratch;

pub fn orchestrate(
    options: &Options,
    cancel: Arc<AtomicBool>,
    progress: impl Fn(Progress) + Send + Sync + 'static,
) -> Result<Summary> {
    let _awake = Awake::acquire(options.settings.render_power);
    let mut options = options.clone();
    if options.audio_source.as_os_str().is_empty() {
        options.audio_source = options.input.clone();
    }
    options.settings.clamp_ranges();
    let ram = perf::available_ram_mb();
    let tier = perf::effective_tier(options.settings.performance_tier, options.settings.render_power, ram);
    options.tile = perf::tile_for(tier, options.tile);
    let power_note = perf::tier_note(tier, options.settings.render_power, ram);

    if options.input == options.output {
        return Err(anyhow!("input and output are the same path"));
    }
    if let Some(parent) = options.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let progress = Arc::new(progress);
    let ffmpeg = tool_path(options.ffmpeg.as_deref(), bin_name("ffmpeg"));
    let ffprobe = tool_path(
        options.ffmpeg.as_deref().and_then(|p| p.parent()).map(|d| d.join(bin_name("ffprobe"))).as_deref(),
        bin_name("ffprobe"),
    );

    if crate::export_media::is_still_path(&options.input) {
        options.output = crate::export_media::coerce_still_output(
            &options.input,
            &options.output,
            options.settings.reddit_mode,
        );
        return still(&options, &ffmpeg, &ffprobe, &power_note, cancel, progress);
    }

    let info = ffmpeg::probe(&ffprobe, &options.input)?;
    let scratch = Scratch::create("job")?;
    let mut current = options.input.clone();
    let mut notes = vec![power_note];
    let want_pre = options.settings.source_filters(info.width, info.height);
    let upscale = options.settings.upscale;
    let upscale_first = upscale && options.settings.upscale_first;
    let upscale_after = upscale && !options.settings.upscale_first;
    let want_post = options.settings.visual_polish();

    if want_pre && upscale_first {
        emit(progress.as_ref(), "prepare", 0.0);
        let prepared = scratch.path.join("prepared.mkv");
        let mut state = crate::filters::FilterState::default();
        let settings = options.settings.clone();
        map_video(&ffmpeg, &ffprobe, &current, &prepared, false, &options.input, true, cancel.clone(), |frame| {
            crate::filters::pre_frame(&frame, &settings, &mut state)
        }, |_, _| {})?;
        notes.extend(state.notes);
        current = prepared;
    }

    if upscale_first {
        check_cancel(&cancel)?;
        emit(progress.as_ref(), "upscale", 0.02);
        let up = scratch.path.join("upscaled.mkv");
        let probed = ffmpeg::probe(&ffprobe, &current)?;
        crate::cugan::upscale_file(
            &ffmpeg,
            &current,
            &up,
            options.settings.cugan_noise,
            crate::cugan::passes(probed.width, probed.height),
            options.tile.max(32),
            options.gpu,
        )?;
        notes.push(format!(
            "Real-CUGAN 2× noise {} ran before interpolation (sync gap 3). The iOS intensity 1.30 is inside the Core ML graph and is not a separate ncnn control.",
            options.settings.cugan_noise
        ));
        current = up;
    }

    let video_dest = if options.settings.video_to_gif {
        scratch.path.join("master.mp4")
    } else {
        options.output.clone()
    };
    let rife_is_final = !upscale_after;
    let rife_output = if rife_is_final {
        video_dest.clone()
    } else {
        scratch.path.join("rife.mkv")
    };
    let mut rife_options = options.clone();
    rife_options.input = current;
    rife_options.output = rife_output.clone();
    // Audio is copied on the encode that produces the file the user keeps.
    if !rife_is_final {
        rife_options.settings.preserve_audio = false;
    }
    let summary = pipeline::interpolate(
        &rife_options,
        Arc::clone(&cancel),
        {
            let progress = Arc::clone(&progress);
            move |mut event| {
                event.stage = "interpolate".into();
                progress(event);
            }
        },
        !(want_pre && upscale_first),
        want_post && !upscale_after,
        !rife_is_final,
    )?;
    notes.extend(summary.notes.clone());
    current = rife_output;

    if upscale_after {
        check_cancel(&cancel)?;
        emit(progress.as_ref(), "upscale", 0.8);
        let up = scratch.path.join("upscaled.mkv");
        let probed = ffmpeg::probe(&ffprobe, &current)?;
        crate::cugan::upscale_file(
            &ffmpeg,
            &current,
            &up,
            options.settings.cugan_noise,
            crate::cugan::passes(probed.width, probed.height),
            options.tile.max(32),
            options.gpu,
        )?;
        notes.push(format!(
            "Real-CUGAN 2× noise {} ran after interpolation (sync gap 3). Intensity 1.30 stays inside the iOS Core ML graph.",
            options.settings.cugan_noise
        ));
        current = up;
    }

    if upscale_after {
        check_cancel(&cancel)?;
        emit(progress.as_ref(), "encode", 0.9);
        let mut state = crate::filters::FilterState::default();
        let settings = options.settings.clone();
        let preserve = options.settings.preserve_audio;
        let source = options.input.clone();
        let progress_encode = Arc::clone(&progress);
        let encode_started = std::time::Instant::now();
        map_video(
            &ffmpeg,
            &ffprobe,
            &current,
            &video_dest,
            preserve,
            &source,
            false,
            cancel.clone(),
            |frame| {
                if want_post {
                    crate::filters::post_frame(&frame, &settings, &mut state)
                } else {
                    frame
                }
            },
            move |done, total| {
                let fraction = if total == 0 {
                    0.9
                } else {
                    0.9 + 0.1 * (done as f32 / total as f32)
                };
                let elapsed = encode_started.elapsed().as_secs_f64().max(0.001);
                let fps = done as f64 / elapsed;
                let remaining = total.saturating_sub(done) as f64;
                progress_encode(Progress {
                    output_frame: done,
                    output_total: total,
                    fps,
                    source_frame: 0,
                    stage: "encode".into(),
                    eta_seconds: if fps > 0.0 { remaining / fps } else { 0.0 },
                    ram_mb: perf::available_ram_mb(),
                    seam: 0.0,
                    fraction,
                    decode_ms: 0.0,
                    rife_ms: 0.0,
                    guard_ms: 0.0,
                    encode_ms: 0.0,
                });
            },
        )?;
        notes.extend(state.notes);
        current = video_dest.clone();
    } else if rife_is_final {
        current = video_dest.clone();
    }

    let saved = finish_delivery(&options, &ffmpeg, &ffprobe, &current, &mut notes, &cancel)?;
    notes.push(format!("Saved {}", saved.display()));
    let mut summary = summary;
    summary.notes = notes;
    summary.output_path = saved;
    if options.settings.diagnostics {
        let dir = crate::models::data_dir().join("Diagnostics");
        let mut recorder = crate::diagnostics::Recorder::new(dir);
        for note in &summary.notes {
            recorder.note(note.clone());
        }
        recorder.metric("output_frames", summary.output_frames as f64);
        recorder.metric("fps", summary.fps);
        recorder.metric("decode_ms", summary.decode_ms as f64);
        recorder.metric("rife_ms", summary.rife_ms as f64);
        recorder.metric("guard_ms", summary.guard_ms as f64);
        recorder.metric("encode_ms", summary.encode_ms as f64);
        if let Ok(path) = recorder.finish(summary.seconds) {
            summary.diagnostics_path = Some(path);
        }
    }
    Ok(summary)
}

fn still(
    options: &Options,
    ffmpeg: &Path,
    ffprobe: &Path,
    power_note: &str,
    cancel: Arc<AtomicBool>,
    progress: Arc<impl Fn(Progress)>,
) -> Result<Summary> {
    emit(progress.as_ref(), "still", 0.1);
    let (width, height) = ffmpeg::pixel_size(ffprobe, &options.input)?;
    let mut state = crate::filters::FilterState::default();
    let mut frame = ffmpeg::decode_image(ffmpeg, &options.input, width, height, false)?;
    frame = crate::filters::pre_frame(&frame, &options.settings, &mut state);
    let scratch = Scratch::create("still")?;
    if options.settings.upscale {
        let staged = scratch.path.join("still-src.png");
        ffmpeg::write_image(ffmpeg, &frame, &staged)?;
        let up = scratch.path.join("still-up.mkv");
        crate::cugan::upscale_file(
            ffmpeg,
            &staged,
            &up,
            options.settings.cugan_noise,
            crate::cugan::passes(frame.width, frame.height),
            options.tile.max(32),
            options.gpu,
        )?;
        let probed = ffmpeg::probe(ffprobe, &up)?;
        frame = decode_one(ffmpeg, &up, probed.width, probed.height)?;
    }
    frame = crate::filters::post_frame(&frame, &options.settings, &mut state);
    emit(progress.as_ref(), "save", 0.9);
    if options.settings.reddit_mode {
        let staged = scratch.path.join("still.png");
        ffmpeg::write_image(ffmpeg, &frame, &staged)?;
        crate::export_media::write_still_capped(
            ffmpeg,
            &staged,
            &options.output,
            frame.width.max(frame.height),
        )?;
    } else {
        ffmpeg::write_image(ffmpeg, &frame, &options.output)?;
    }
    if !options.output.is_file() {
        return Err(anyhow!("the still was not written to {}", options.output.display()));
    }
    let mut notes = state.notes;
    notes.insert(0, power_note.to_string());
    notes.push(format!("Saved {}", options.output.display()));
    if options.settings.reddit_mode {
        notes.push("Reddit Mode kept the still at or under 20 MB.".into());
    }
    check_cancel(&cancel)?;
    Ok(Summary {
        output_frames: 1,
        seconds: 0.0,
        fps: 0.0,
        scene_boundaries: 0,
        stroke_repeats: 0,
        source_outliers: 0,
        gpu_name: String::new(),
        notes,
        diagnostics_path: None,
        output_path: options.output.clone(),
        decode_ms: 0.0,
        rife_ms: 0.0,
        guard_ms: 0.0,
        encode_ms: 0.0,
    })
}

fn finish_delivery(
    options: &Options,
    ffmpeg: &Path,
    ffprobe: &Path,
    current: &Path,
    notes: &mut Vec<String>,
    cancel: &AtomicBool,
) -> Result<PathBuf> {
    check_cancel(cancel)?;
    if !options.settings.video_to_gif && current != options.output && current.is_file() {
        if std::fs::rename(current, &options.output).is_err() {
            std::fs::copy(current, &options.output)?;
        }
    }
    let delivered = if options.settings.video_to_gif {
        current.to_path_buf()
    } else {
        options.output.clone()
    };
    let saved = if options.settings.video_to_gif {
        let gif = if options.output.extension().and_then(|e| e.to_str()) == Some("gif") {
            options.output.clone()
        } else {
            options.output.with_extension("gif")
        };
        let info = ffmpeg::probe(ffprobe, &delivered)?;
        let fps = info.rate.num as f64 / info.rate.den.max(1) as f64;
        crate::export_media::write_gif(ffmpeg, &delivered, &gif, info.width, info.height, fps)?;
        notes.push(format!(
            "GIF written to {}. Audio is dropped. The iOS loop-seam search is approximated by ffmpeg's palette.",
            gif.display()
        ));
        gif
    } else if options.settings.reddit_mode {
        let info = ffmpeg::probe(ffprobe, &delivered)?;
        let scratch = Scratch::create("reddit")?;
        let tmp = scratch.path.join("reddit.mp4");
        crate::export_media::cap_video(ffmpeg, &delivered, &tmp, info.duration.max(0.1), options.settings.preserve_audio && info.has_audio)?;
        std::fs::copy(&tmp, &delivered)?;
        notes.push("Reddit Mode re-encoded the video toward 19.8 MB.".into());
        delivered
    } else {
        if options.settings.preserve_audio {
            notes.push("Original audio was stream-copied when the final encode ran.".into());
        }
        delivered
    };
    if !saved.is_file() {
        return Err(anyhow!("the export was not written to {}", saved.display()));
    }
    Ok(saved)
}

pub fn map_video(
    ffmpeg: &Path,
    ffprobe: &Path,
    input: &Path,
    output: &Path,
    preserve_audio: bool,
    audio_source: &Path,
    lossless: bool,
    cancel: Arc<AtomicBool>,
    mut map: impl FnMut(Frame) -> Frame,
    mut on_frame: impl FnMut(u64, u64),
) -> Result<()> {
    let info = ffmpeg::probe(ffprobe, input)?;
    let width = info.width & !1;
    let height = info.height & !1;
    let total = info.frame_count.max(0) as u64;
    let mut decoder = Decoder::spawn(ffmpeg, input, width, height, info.width, info.height)?;
    let mut written = 0u64;
    let encode_result = if lossless {
        let mut encoder = Encoder::spawn_ffv1(ffmpeg, output, width, height, info.rate.num, info.rate.den)?;
        let mut failed = None;
        while let Some(frame) = decoder.read_frame()? {
            if cancel.load(Ordering::Relaxed) {
                failed = Some(anyhow!("cancelled"));
                break;
            }
            if let Err(err) = encoder.write_frame(&map(frame)) {
                failed = Some(err);
                break;
            }
            written += 1;
            on_frame(written, total);
        }
        // Close the encoder stdin before waiting on the decoder so libx265
        // sees EOF even if the decoder process is still shutting down.
        let finished = failed.map(Err).unwrap_or_else(|| encoder.finish());
        finish_both(finished, decoder.finish())
    } else {
        let mut encoder = Encoder::spawn_audio(
            ffmpeg,
            audio_source,
            output,
            width,
            height,
            info.rate.num,
            info.rate.den,
            &info.color,
            preserve_audio,
        )?;
        let mut failed = None;
        while let Some(frame) = decoder.read_frame()? {
            if cancel.load(Ordering::Relaxed) {
                failed = Some(anyhow!("cancelled"));
                break;
            }
            if let Err(err) = encoder.write_frame(&map(frame)) {
                failed = Some(err);
                break;
            }
            written += 1;
            on_frame(written, total);
        }
        let finished = failed.map(Err).unwrap_or_else(|| encoder.finish());
        finish_both(finished, decoder.finish())
    };
    encode_result
}

fn finish_both(encoded: Result<()>, decoded: Result<()>) -> Result<()> {
    match (encoded, decoded) {
        (Ok(()), decoded) => decoded,
        (Err(err), _) => Err(err),
    }
}

fn decode_one(ffmpeg: &Path, input: &Path, width: i32, height: i32) -> Result<Frame> {
    let width = width & !1;
    let height = height & !1;
    let mut decoder = Decoder::spawn(ffmpeg, input, width, height, width, height)?;
    let frame = decoder.read_frame()?.ok_or_else(|| anyhow!("no frames in {}", input.display()))?;
    decoder.finish()?;
    Ok(frame)
}

fn emit(progress: &dyn Fn(Progress), stage: &str, fraction: f32) {
    progress(Progress {
        output_frame: 0,
        output_total: 1,
        fps: 0.0,
        source_frame: 0,
        stage: stage.into(),
        eta_seconds: 0.0,
        ram_mb: perf::available_ram_mb(),
        seam: 0.0,
        fraction,
        decode_ms: 0.0,
        rife_ms: 0.0,
        guard_ms: 0.0,
        encode_ms: 0.0,
    });
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(anyhow!("cancelled"))
    } else {
        Ok(())
    }
}

fn bin_name(stem: &str) -> &'static str {
    match stem {
        "ffmpeg" if cfg!(windows) => "ffmpeg.exe",
        "ffprobe" if cfg!(windows) => "ffprobe.exe",
        "ffmpeg" => "ffmpeg",
        _ => "ffprobe",
    }
}
