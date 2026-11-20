mod ffmpeg;
mod pipeline;
mod rife;
mod schedule;
mod x265;

#[cfg(feature = "gui")]
mod gui;

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use clap::Parser;

use crate::pipeline::Options;

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
    let cli = Cli::parse();
    let model = cli.model.clone().unwrap_or_else(default_model_dir);
    let cancel = Arc::new(AtomicBool::new(false));
    let summary = pipeline::run(
        &Options {
            input: cli.input,
            output: cli.output,
            multi: cli.multi,
            fps: cli.fps,
            model,
            gpu: cli.gpu,
            tile: cli.tile,
            sensitivity: cli.sensitivity,
            ffmpeg: None,
        },
        cancel,
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

fn default_model_dir() -> PathBuf {
    #[cfg(feature = "gui")]
    {
        return crate::gui::default_model();
    }
    #[cfg(not(feature = "gui"))]
    {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let bundled = dir.join("models").join("rife-v4.6");
                if bundled.join("flownet.param").is_file() {
                    return bundled;
                }
            }
        }
        PathBuf::from("models").join("rife-v4.6")
    }
}
