//! AI Image Studio on the CPU build of stable-diffusion.cpp.
//! Animagine XL v3.1 is the SDXL profile the iOS app uses. Anima (Cosmos 2.5)
//! is shown so the picker matches, and it is not started: this runner does
//! not implement that flow-matching stack. Anima's upstream license is
//! non-commercial.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{anyhow, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Animagine,
    Anima,
}

impl Profile {
    pub fn label(self) -> &'static str {
        match self {
            Self::Animagine => "Animagine XL v3.1",
            Self::Anima => "Anima (not run on this CPU build)",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Request {
    pub profile: Profile,
    pub prompt: String,
    pub negative: String,
    pub width: i32,
    pub height: i32,
    pub steps: u32,
    pub cfg: f64,
    pub clip_skip: i32,
    pub sampler: String,
    pub seed: i64,
    pub hires: bool,
    pub hires_scale: f64,
    pub outline: bool,
    pub outline_strength: f64,
    pub vulkan: bool,
    pub output: PathBuf,
}

impl Request {
    pub fn estimate_seconds(&self) -> f64 {
        let pixels = (self.width.max(64) as f64 * self.height.max(64) as f64) / (832.0 * 1216.0);
        let mut seconds = self.steps.max(1) as f64 * 10.0 * pixels.max(0.25);
        if self.hires {
            seconds += seconds * 0.45 * self.hires_scale.max(1.0);
        }
        if self.vulkan { seconds * 0.45 } else { seconds }
    }

    pub fn estimate_label(&self) -> String {
        let seconds = self.estimate_seconds();
        let minutes = (seconds / 60.0).round().max(1.0) as u64;
        format!(
            "About {minutes} min on a 2019 laptop CPU for {}×{} and {} steps. \
             CPU sampling is slow. Q4 or a smaller canvas is the practical way to shorten it. \
             Vulkan offload uses the Radeon when the build is the Vulkan zip; the 4 GB 5300M and 5500M may still spill to the 64 GB of RAM.",
            self.width, self.height, self.steps
        )
    }
}

pub fn animagine_path() -> PathBuf {
    crate::models::data_dir()
        .join("models")
        .join("studio")
        .join("animagine-xl-3.1.safetensors")
}

pub fn command(sd_cli: &Path, request: &Request) -> Result<Command> {
    if request.profile == Profile::Anima {
        return Err(anyhow!(
            "Anima is the iOS Draw Things catalog model (Cosmos 2.5 2B). Its upstream license is non-commercial. \
             This Windows build runs Animagine XL through stable-diffusion.cpp and does not start the Anima flow-matching stack."
        ));
    }
    let model = animagine_path();
    if !model.is_file() {
        return Err(anyhow!(
            "Animagine XL v3.1 is not downloaded yet ({}). Use the model panel. It is not in the installer.",
            model.display()
        ));
    }
    let mut cmd = Command::new(sd_cli);
    cmd.arg("-m")
        .arg(&model)
        .arg("-p")
        .arg(&request.prompt)
        .arg("-n")
        .arg(&request.negative)
        .arg("-W")
        .arg(request.width.to_string())
        .arg("-H")
        .arg(request.height.to_string())
        .arg("--steps")
        .arg(request.steps.to_string())
        .arg("--cfg-scale")
        .arg(format!("{:.2}", request.cfg))
        .arg("--sampling-method")
        .arg(if request.sampler.is_empty() { "euler_a".to_string() } else { request.sampler.clone() })
        .arg("-s")
        .arg(request.seed.to_string())
        .arg("--clip-skip")
        .arg(request.clip_skip.clamp(1, 2).to_string())
        .arg("--offload-to-cpu")
        .arg("-o")
        .arg(&request.output);
    if request.hires {
        cmd.arg("--upscale-repeats")
            .arg("1");
    }
    cmd.stdin(Stdio::null());
    Ok(cmd)
}

pub fn run(request: &Request) -> Result<PathBuf> {
    let cli = crate::models::sd_cli(request.vulkan).ok_or_else(|| {
        anyhow!("stable-diffusion.cpp is not downloaded. Use the model panel. It is not in the installer.")
    })?;
    let mut cmd = command(&cli, request)?;
    let status = cmd.status()?;
    if !status.success() {
        return Err(anyhow!("stable-diffusion.cpp exited {}", status.code().unwrap_or(-1)));
    }
    let mut output = request.output.clone();
    if request.outline && output.is_file() {
        output = apply_outline(&output, request.outline_strength)?;
    }
    Ok(output)
}

fn apply_outline(path: &Path, strength: f64) -> Result<PathBuf> {
    let ffmpeg = crate::ffmpeg::tool_path(None, if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" });
    let ffprobe = ffmpeg.with_file_name(if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" });
    let info = crate::ffmpeg::probe(&ffprobe, path)?;
    let mut decoder = crate::ffmpeg::Decoder::spawn(&ffmpeg, path, info.width, info.height, info.width, info.height)?;
    let frame = decoder
        .read_frame()?
        .ok_or_else(|| anyhow!("the generated image had no frame"))?;
    decoder.finish()?;
    let rebuilt = crate::outline::rebuild(&frame, strength);
    let dest = path.with_file_name(format!(
        "{}-outline.png",
        path.file_stem().and_then(|s| s.to_str()).unwrap_or("studio")
    ));
    // Write a PNG through ffmpeg so the outline result is a normal image.
    let raw = path.with_extension("bgra");
    std::fs::write(&raw, &rebuilt.data)?;
    let status = Command::new(&ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra", "-s"])
        .arg(format!("{}x{}", rebuilt.width, rebuilt.height))
        .arg("-i")
        .arg(&raw)
        .arg(&dest)
        .status()?;
    let _ = std::fs::remove_file(&raw);
    if !status.success() {
        return Err(anyhow!("could not write the outlined still"));
    }
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anima_is_refused_and_the_estimate_mentions_the_cpu() {
        let request = Request {
            profile: Profile::Anima,
            prompt: "test".into(),
            negative: String::new(),
            width: 832,
            height: 1216,
            steps: 28,
            cfg: 4.0,
            clip_skip: 2,
            sampler: "euler_a".into(),
            seed: 1,
            hires: false,
            hires_scale: 1.5,
            outline: false,
            outline_strength: 0.5,
            vulkan: false,
            output: PathBuf::from("out.png"),
        };
        assert!(request.estimate_seconds() > 60.0);
        let err = command(Path::new("sd-cli"), &request).unwrap_err().to_string();
        assert!(err.contains("non-commercial"));
    }
}
