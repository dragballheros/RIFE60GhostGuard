mod anime_restore;
mod blush;
mod color_pop;
mod color_rebuild;
mod compression;
mod cugan;
mod diagnostics;
mod export_media;
mod ffmpeg;
mod filters;
mod h3;
mod models;
mod onnx_runtime;
mod outline;
mod passes;
mod perf;
mod pipeline;
mod pixels;
mod rife;
mod schedule;
mod scratch;
mod settings;
mod studio;
mod watermark;
mod x265;

#[cfg(feature = "gui")]
mod gui;

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use clap::Parser;

use crate::pipeline::Options;
use crate::settings::{
    AnimeQuality, CorrectionMode, PerformanceTier, ProcessingSettings, RifeModelKind, WatermarkModelKind,
};

#[derive(Parser, Debug)]
#[command(
    name = "rife-pc",
    about = "RIFE frame interpolation for Windows, with the iOS quality guards"
)]
struct Cli {
    /// Source video.
    input: PathBuf,
    /// Output mp4.
    #[arg(short, long)]
    output: PathBuf,
    /// Frames per source frame. 2 inserts one midpoint. Ignored when --fps is set.
    #[arg(long, default_value_t = 2)]
    multi: u32,
    /// Output frame rate. Overrides --multi.
    #[arg(long)]
    fps: Option<f64>,
    /// Directory containing flownet.param and flownet.bin. Defaults to models/rife-v4.6 beside the exe.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Vulkan GPU index. -1 picks an AMD Radeon if one is present.
    #[arg(long, default_value_t = -1)]
    gpu: i32,
    /// Tile size in pixels. 512 fits the 4 GB Radeon Pro 5300M/5500M. 0 runs the whole frame.
    #[arg(long, default_value_t = 512)]
    tile: i32,
    /// Same range as the iOS app: 0.65 through 1.35. Default 1.35.
    #[arg(long, default_value_t = 1.35)]
    sensitivity: f64,
    /// RIFE flownet. 4.6 is the previous default. 4.26 needs the ncnn model with padding 64.
    #[arg(long, default_value = "4.6")]
    rife: String,
    #[arg(long)]
    no_ghost: bool,
    #[arg(long)]
    no_scene_cuts: bool,
    #[arg(long)]
    no_thin_line: bool,
    #[arg(long)]
    no_line_stabilizers: bool,
    #[arg(long)]
    no_outlier: bool,
    #[arg(long)]
    no_audio: bool,
    #[arg(long)]
    render_power: bool,
    #[arg(long, default_value = "auto")]
    tier: String,
    #[arg(long)]
    reddit: bool,
    #[arg(long)]
    gif: bool,
    #[arg(long)]
    compression: bool,
    #[arg(long)]
    color_pop: bool,
    #[arg(long, default_value_t = 1.0)]
    color_pop_strength: f64,
    #[arg(long, default_value = "off")]
    warmth: String,
    #[arg(long, default_value = "processed")]
    skin_tone: String,
    #[arg(long)]
    color_rebuild: bool,
    #[arg(long, default_value_t = 0.45)]
    color_rebuild_strength: f64,
    #[arg(long)]
    outline_rebuild: bool,
    #[arg(long, default_value_t = 0.5)]
    outline_rebuild_strength: f64,
    #[arg(long)]
    sharpie: bool,
    #[arg(long)]
    anime_restore: bool,
    #[arg(long, default_value = "auto")]
    anime_quality: String,
    #[arg(long, default_value_t = 0.8)]
    anime_strength: f64,
    #[arg(long)]
    upscale: bool,
    #[arg(long)]
    upscale_first: bool,
    #[arg(long)]
    watermark: bool,
    #[arg(long, default_value = "lama")]
    watermark_model: String,
    #[arg(long)]
    diagnostics: bool,
    #[arg(long)]
    export_dir: Option<PathBuf>,
    /// Extra videos, processed with the same settings after the positional input.
    #[arg(long)]
    also: Vec<PathBuf>,
}

fn main() {
    if let Err(err) = dispatch() {
        eprintln!("rife-pc: {err:#}");
        std::process::exit(1);
    }
}

fn dispatch() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.first().map(String::as_str) == Some("gui") {
        return launch_gui();
    }
    if args.first().map(String::as_str) == Some("h3") {
        return h3_cli(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("studio") {
        return studio_cli(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("download") {
        let id = args.get(1).map(String::as_str).unwrap_or("");
        let path = models::download_id(id)?;
        eprintln!("rife-pc: downloaded {}", path.display());
        return Ok(());
    }
    let cli = Cli::parse();
    let settings = settings_from_cli(&cli)?;
    let model = cli.model.clone().unwrap_or_else(|| models::rife_dir(settings.rife_model));
    let mut inputs = vec![cli.input.clone()];
    inputs.extend(cli.also.iter().cloned());
    let cancel = Arc::new(AtomicBool::new(false));
    for (index, input) in inputs.into_iter().enumerate() {
        let export_dir = cli.export_dir.as_ref().map(|dir| dir.display().to_string()).unwrap_or_default();
        let override_path = if index == 0 {
            cli.output.display().to_string()
        } else {
            String::new()
        };
        let output = export_media::delivery_path(
            &input,
            &export_dir,
            &override_path,
            settings.video_to_gif,
            settings.reddit_mode,
        );
        let summary = pipeline::run(
        &Options {
            audio_source: input.clone(),
            input,
            output,
            multi: cli.multi,
            fps: cli.fps,
            model: model.clone(),
            gpu: cli.gpu,
            tile: cli.tile,
            sensitivity: cli.sensitivity,
            ffmpeg: None,
            settings: settings.clone(),
        },
        Arc::clone(&cancel),
        |progress| {
            eprintln!(
                "rife-pc: {}/{} frames  {:.1} fps  source {}",
                progress.output_frame, progress.output_total, progress.fps, progress.source_frame
            );
        },
    )?;
    eprintln!(
        "rife-pc: done {} frames in {:.1}s ({:.1} fps) on {}  scene-boundary {}  stroke-repeat {}  source-outliers {}",
        summary.output_frames,
        summary.seconds,
        summary.fps,
        summary.gpu_name,
        summary.scene_boundaries,
        summary.stroke_repeats,
        summary.source_outliers
    );
        for note in &summary.notes {
            eprintln!("rife-pc: {note}");
        }
    }
    Ok(())
}

fn settings_from_cli(cli: &Cli) -> Result<ProcessingSettings> {
    let mut settings = ProcessingSettings::cli();
    settings.rife_model = RifeModelKind::parse(&cli.rife).ok_or_else(|| anyhow!("--rife must be 4.6 or 4.26"))?;
    settings.ghost_protection = !cli.no_ghost;
    settings.scene_cut_protection = !cli.no_scene_cuts;
    settings.thin_line = !cli.no_thin_line;
    settings.line_stabilizers = !cli.no_line_stabilizers;
    settings.source_outlier = !cli.no_outlier;
    settings.preserve_audio = !cli.no_audio;
    settings.render_power = cli.render_power;
    settings.performance_tier = match cli.tier.as_str() {
        "auto" => PerformanceTier::Auto,
        "performance" => PerformanceTier::Performance,
        "balanced" => PerformanceTier::Balanced,
        "safe" => PerformanceTier::Safe,
        other => return Err(anyhow!("--tier {other} is not auto, performance, balanced, or safe")),
    };
    settings.reddit_mode = cli.reddit;
    settings.video_to_gif = cli.gif;
    settings.compression_guard = cli.compression;
    settings.color_pop = cli.color_pop;
    settings.color_pop_strength = cli.color_pop_strength;
    settings.warmth = CorrectionMode::parse(&cli.warmth).ok_or_else(|| anyhow!("--warmth must be off, original, processed, or full"))?;
    settings.skin_tone = CorrectionMode::parse(&cli.skin_tone).ok_or_else(|| anyhow!("--skin-tone must be off, original, processed, or full"))?;
    settings.color_rebuild = cli.color_rebuild;
    settings.color_rebuild_strength = cli.color_rebuild_strength;
    settings.outline_rebuild = cli.outline_rebuild;
    settings.outline_rebuild_strength = cli.outline_rebuild_strength;
    settings.sharpie = cli.sharpie;
    settings.anime_restore = cli.anime_restore;
    settings.anime_restore_suppress_auto = !cli.anime_restore;
    settings.anime_restore_quality = AnimeQuality::parse(&cli.anime_quality).ok_or_else(|| anyhow!("--anime-quality must be auto, apisr, or span"))?;
    settings.anime_restore_strength = cli.anime_strength;
    settings.upscale = cli.upscale || cli.upscale_first;
    settings.upscale_first = cli.upscale_first;
    settings.watermark = cli.watermark;
    settings.watermark_model = WatermarkModelKind::parse(&cli.watermark_model).ok_or_else(|| anyhow!("--watermark-model must be lama, bar, or mosaic"))?;
    settings.diagnostics = cli.diagnostics;
    if let Some(dir) = &cli.export_dir {
        settings.export_dir = dir.display().to_string();
    }
    settings.clamp_ranges();
    Ok(settings)
}

fn h3_cli(args: &[String]) -> Result<()> {
    let mut request = h3::Request {
        prompt: String::new(),
        width: 864,
        height: 480,
        frames: 56,
        fps: 24.0,
        cfg: 1.0,
        reference: None,
        keep_audio: true,
        vulkan: false,
        license_accepted: false,
        output: PathBuf::from("h3.mp4"),
    };
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let next = args.get(i + 1).cloned();
        match arg {
            "--prompt" | "-p" => request.prompt = next.ok_or_else(|| anyhow!("--prompt needs a value"))?,
            "--output" | "-o" => request.output = PathBuf::from(next.ok_or_else(|| anyhow!("--output needs a value"))?),
            "--width" => request.width = next.ok_or_else(|| anyhow!("--width needs a value"))?.parse()?,
            "--height" => request.height = next.ok_or_else(|| anyhow!("--height needs a value"))?.parse()?,
            "--frames" => request.frames = next.ok_or_else(|| anyhow!("--frames needs a value"))?.parse()?,
            "--reference" | "-r" => request.reference = Some(PathBuf::from(next.ok_or_else(|| anyhow!("--reference needs a value"))?)),
            "--vulkan" => {
                request.vulkan = true;
                i += 1;
                continue;
            }
            "--accept-license" => {
                request.license_accepted = true;
                i += 1;
                continue;
            }
            "--no-audio" => {
                request.keep_audio = false;
                i += 1;
                continue;
            }
            other => return Err(anyhow!("unknown h3 flag {other}")),
        }
        i += 2;
    }
    eprintln!("{}", request.estimate_label());
    h3::run(&request)
}

fn studio_cli(args: &[String]) -> Result<()> {
    let mut request = studio::Request {
        profile: studio::Profile::Animagine,
        prompt: String::new(),
        negative: String::new(),
        width: 832,
        height: 1216,
        steps: 28,
        cfg: 5.0,
        clip_skip: 2,
        sampler: "euler_a".into(),
        seed: 1,
        hires: false,
        hires_scale: 1.5,
        outline: false,
        outline_strength: 0.5,
        vulkan: false,
        output: PathBuf::from("studio.png"),
    };
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let next = args.get(i + 1).cloned();
        match arg {
            "--prompt" | "-p" => request.prompt = next.ok_or_else(|| anyhow!("--prompt needs a value"))?,
            "--negative" | "-n" => request.negative = next.ok_or_else(|| anyhow!("--negative needs a value"))?,
            "--output" | "-o" => request.output = PathBuf::from(next.ok_or_else(|| anyhow!("--output needs a value"))?),
            "--width" => request.width = next.ok_or_else(|| anyhow!("--width needs a value"))?.parse()?,
            "--height" => request.height = next.ok_or_else(|| anyhow!("--height needs a value"))?.parse()?,
            "--steps" => request.steps = next.ok_or_else(|| anyhow!("--steps needs a value"))?.parse()?,
            "--cfg" => request.cfg = next.ok_or_else(|| anyhow!("--cfg needs a value"))?.parse()?,
            "--seed" => request.seed = next.ok_or_else(|| anyhow!("--seed needs a value"))?.parse()?,
            "--outline" => {
                request.outline = true;
                i += 1;
                continue;
            }
            "--vulkan" => {
                request.vulkan = true;
                i += 1;
                continue;
            }
            "--anima" => {
                request.profile = studio::Profile::Anima;
                i += 1;
                continue;
            }
            other => return Err(anyhow!("unknown studio flag {other}")),
        }
        i += 2;
    }
    eprintln!("{}", request.estimate_label());
    let output = studio::run(&request)?;
    eprintln!("rife-pc: wrote {}", output.display());
    Ok(())
}

fn launch_gui() -> Result<()> {
    #[cfg(feature = "gui")]
    {
        crate::gui::run()
    }
    #[cfg(not(feature = "gui"))]
    {
        Err(anyhow!(
            "this build has no window. Use: rife-pc input.mp4 -o output.mp4 [--multi 2] [--fps 60]"
        ))
    }
}

