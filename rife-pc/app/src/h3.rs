//! MiniMax H3 video generation through stable-diffusion.cpp.
//! Weights stay on disk in the user data folder. The CPU build keeps the
//! model in system RAM (`--offload-to-cpu`). The optional Vulkan build is
//! the low-VRAM offload for the Radeon Pro.
//!
//! leejet/MiniMax-H3-GGUF publishes Q4_K_M and Q2_K_M. It does not publish
//! Q8_0. Q4_K_M is the default because it fits a 64 GB machine together with
//! the Qwen3-VL Q4_K_M encoder.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{anyhow, Result};

pub const LICENSE_URL: &str =
    "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE";

pub const TERRITORY_WARNING: &str = "\
Territory restriction: the MiniMax H3 Community License excludes the United States, \
the European Union, the United Kingdom, and South Korea from its standard applicable \
territory. If you are in an excluded territory, do not download or use these weights \
unless MiniMax has separately authorized you. Review the current license before proceeding.";

#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: String,
    pub width: i32,
    pub height: i32,
    pub frames: u32,
    pub fps: f64,
    pub cfg: f64,
    pub reference: Option<PathBuf>,
    pub keep_audio: bool,
    pub vulkan: bool,
    pub license_accepted: bool,
    pub output: PathBuf,
}

impl Request {
    pub fn estimate_seconds(&self) -> f64 {
        let pixels = (self.width.max(64) as f64 * self.height.max(64) as f64) / (864.0 * 480.0);
        let per_frame = 180.0 * pixels.max(0.25);
        let text_encoder = 15.0 * 60.0;
        let total = text_encoder + self.frames.max(1) as f64 * per_frame;
        if self.vulkan { total * 0.55 } else { total }
    }

    pub fn estimate_label(&self) -> String {
        let seconds = self.estimate_seconds();
        let hours = (seconds / 3600.0).floor() as u64;
        let minutes = ((seconds % 3600.0) / 60.0).round() as u64;
        format!(
            "About {hours} h {minutes} min on a 2019 laptop CPU with Q4_K_M, {frames} frames at {w}×{h}. \
             This is a rough estimate, not a benchmark. CPU generation is slow. \
             Vulkan offload can be shorter when the 4–8 GB Radeon can hold a tile, and it can be slower if it spills back to RAM.",
            frames = self.frames,
            w = self.width,
            h = self.height
        )
    }
}

pub fn command(sd_cli: &Path, request: &Request) -> Result<Command> {
    if !request.license_accepted {
        return Err(anyhow!(
            "Accept the MiniMax H3 license statement before downloading or generating. {TERRITORY_WARNING}"
        ));
    }
    let paths = crate::models::h3_paths();
    let missing = paths.missing();
    if !missing.is_empty() {
        return Err(anyhow!(
            "MiniMax H3 weights are not downloaded yet ({}). Use the model downloader. They are not in the installer.",
            missing.join(", ")
        ));
    }
    let mut cmd = Command::new(sd_cli);
    cmd.arg("-M")
        .arg("vid_gen")
        .arg("--diffusion-model")
        .arg(&paths.diffusion)
        .arg("--vae")
        .arg(&paths.vae)
        .arg("--llm")
        .arg(&paths.llm)
        .arg("-p")
        .arg(&request.prompt)
        .arg("--cfg-scale")
        .arg(format!("{:.2}", request.cfg))
        .arg("-W")
        .arg(request.width.to_string())
        .arg("-H")
        .arg(request.height.to_string())
        .arg("--fps")
        .arg(format!("{:.2}", request.fps))
        .arg("--video-frames")
        .arg(request.frames.to_string())
        .arg("--rng")
        .arg("cpu")
        .arg("--offload-to-cpu")
        .arg("-o")
        .arg(&request.output)
        .arg("-v");
    if request.keep_audio && paths.audio_vae.is_file() {
        cmd.arg("--audio-vae").arg(&paths.audio_vae);
    }
    if let Some(reference) = &request.reference {
        cmd.arg("-r").arg(reference);
    }
    cmd.stdin(Stdio::null());
    Ok(cmd)
}

pub fn run(request: &Request) -> Result<()> {
    let cli = crate::models::sd_cli(request.vulkan).ok_or_else(|| {
        anyhow!(
            "stable-diffusion.cpp is not downloaded. Use Download CPU build (or the Vulkan build) in the model panel. It is not in the installer."
        )
    })?;
    let mut cmd = command(&cli, request)?;
    let status = cmd.status()?;
    if !status.success() {
        return Err(anyhow!(
            "stable-diffusion.cpp exited {}. CPU H3 generation can take hours; the log above is the tool's own output.",
            status.code().unwrap_or(-1)
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_estimate_is_hours_for_a_short_clip() {
        let request = Request {
            prompt: "a cat".into(),
            width: 864,
            height: 480,
            frames: 56,
            fps: 24.0,
            cfg: 1.0,
            reference: None,
            keep_audio: true,
            vulkan: false,
            license_accepted: true,
            output: PathBuf::from("out.mp4"),
        };
        assert!(request.estimate_seconds() > 2.0 * 3600.0);
        assert!(request.estimate_label().contains("2019"));
    }

    #[test]
    fn generation_refuses_without_the_license_acknowledgement() {
        let request = Request {
            prompt: "a cat".into(),
            width: 448,
            height: 256,
            frames: 16,
            fps: 24.0,
            cfg: 1.0,
            reference: None,
            keep_audio: false,
            vulkan: false,
            license_accepted: false,
            output: PathBuf::from("out.mp4"),
        };
        let err = command(Path::new("sd-cli"), &request).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("United States"));
        assert!(text.contains("South Korea"));
    }
}
