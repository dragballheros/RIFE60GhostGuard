//! Real-CUGAN 2× through the official `realcugan-ncnn-vulkan` executable.
//! Noise level 3 matches the iOS Real-CUGAN anime model. The iOS intensity
//! of 1.30 is an input of the Core ML graph and is not a separate ncnn flag.
//! Sync gap stays at 3. The directory worker pool is tuned to keep the GPU fed
//! without blindly multiplying VRAM use.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Result};

pub fn passes(width: i32, height: i32) -> u32 {
    let long = width.max(height);
    let pixels_4x = width as i64 * 4 * height as i64 * 4;
    if long < 1080 && pixels_4x <= 40_000_000 {
        2
    } else {
        1
    }
}

pub fn find_exe() -> Option<PathBuf> {
    let name = if cfg!(windows) {
        "realcugan-ncnn-vulkan.exe"
    } else {
        "realcugan-ncnn-vulkan"
    };
    let path = crate::models::find_tool(name);
    if path.is_file() {
        Some(path)
    } else {
        None
    }
}

pub fn model_dir() -> PathBuf {
    let bundled = crate::models::find_tool("models").join("models-se");
    if bundled.join("up2x-denoise3x.param").is_file() {
        return bundled;
    }
    crate::models::data_dir().join("models").join("models-se")
}

/// Upscale every frame of `input` by 2× `passes` times and write `output`.
/// The model directory's path must contain `models-se` so the tool selects
/// the SE prepadding of 18.
pub fn upscale_file(
    ffmpeg: &Path,
    input: &Path,
    output: &Path,
    noise: i32,
    passes_n: u32,
    tile: i32,
    gpu: i32,
) -> Result<()> {
    let exe = find_exe().ok_or_else(|| {
        anyhow!(
            "Real-CUGAN is on, but realcugan-ncnn-vulkan.exe was not found next to rife-pc.exe. \
             The Windows installer copies it. Turn upscale off to interpolate without it."
        )
    })?;
    let models = model_dir();
    if !models.join(format!("up2x-denoise{}x.param", noise.max(0))).is_file() && noise >= 0 {
        return Err(anyhow!(
            "Real-CUGAN models-se up2x-denoise{}x is not in {}",
            noise,
            models.display()
        ));
    }
    let scratch = crate::scratch::Scratch::create("cugan")?;
    let mut current = input.to_path_buf();
    for pass in 0..passes_n.max(1) {
        let frames = scratch.path.join(format!("in-{pass}"));
        let up = scratch.path.join(format!("out-{pass}"));
        std::fs::create_dir_all(&frames)?;
        std::fs::create_dir_all(&up)?;
        let status = Command::new(ffmpeg)
            .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
            .arg(&current)
            .args(["-fps_mode", "passthrough"])
            .arg(frames.join("frame_%06d.png"))
            .status()
            .map_err(|err| anyhow!("failed to start ffmpeg for Real-CUGAN frames: {err}"))?;
        if !status.success() {
            return Err(anyhow!("ffmpeg could not write PNG frames for Real-CUGAN"));
        }
        let mut cmd = Command::new(&exe);
        cmd.arg("-i")
            .arg(&frames)
            .arg("-o")
            .arg(&up)
            .arg("-n")
            .arg(noise.to_string())
            .arg("-s")
            .arg("2")
            .arg("-t")
            .arg(tile.max(32).to_string())
            .arg("-c")
            .arg("3")
            // Real-CUGAN supports independent load/proc/save worker counts.
            // Large frames use 2:2:2 to limit VRAM pressure; smaller frames can
            // safely keep four decode/inference/encode workers in flight.
            .arg("-j")
            .arg(if tile >= 768 { "2:2:2" } else if tile >= 512 { "3:3:3" } else { "4:4:4" })
            .arg("-m")
            .arg(&models);
        if gpu >= 0 {
            cmd.arg("-g").arg(gpu.to_string());
        }
        let status = cmd
            .status()
            .map_err(|err| anyhow!("failed to start {}: {err}", exe.display()))?;
        if !status.success() {
            return Err(anyhow!("realcugan-ncnn-vulkan failed (exit {})", status.code().unwrap_or(-1)));
        }
        let next = if pass + 1 == passes_n.max(1) {
            output.to_path_buf()
        } else {
            scratch.path.join(format!("mid-{pass}.mkv"))
        };
        let rate = probe_rate(ffmpeg, &current)?;
        let status = Command::new(ffmpeg)
            .args(["-y", "-hide_banner", "-loglevel", "error", "-framerate"])
            .arg(rate)
            .arg("-i")
            .arg(up.join("frame_%06d.png"))
            .args(["-c:v", "ffv1", "-pix_fmt", "gbrp", "-an"])
            .arg(&next)
            .status()?;
        if !status.success() {
            return Err(anyhow!("ffmpeg could not pack the Real-CUGAN frames"));
        }
        current = next;
    }
    Ok(())
}

fn probe_rate(ffmpeg: &Path, input: &Path) -> Result<String> {
    let ffprobe = ffmpeg.with_file_name(if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" });
    let probe = crate::ffmpeg::probe(&ffprobe, input).or_else(|_| crate::ffmpeg::probe(Path::new("ffprobe"), input))?;
    Ok(format!("{}/{}", probe.rate.num, probe.rate.den.max(1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_720p_frame_gets_two_passes_and_1080p_gets_one() {
        assert_eq!(passes(960, 540), 2);
        assert_eq!(passes(1280, 720), 1);
        assert_eq!(passes(1920, 1080), 1);
    }
}
