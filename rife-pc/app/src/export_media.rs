//! Reddit Mode and video-to-GIF, both capped at 20 MB.
//! iOS Reddit Mode re-encodes stills as AVIF. This port uses the same byte
//! budget and the same resolution/quality ladder, with ffmpeg's AVIF encoder
//! when it is present and JPEG when it is not. GIF uses ffmpeg's palette.

use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, Result};

pub const HARD_LIMIT: u64 = 20_000_000;
pub const TARGET: u64 = 19_800_000;

pub fn video_bitrate(seconds: f64, has_audio: bool) -> u64 {
    let total = (TARGET as f64 * 8.0 / seconds.max(0.1)) * 0.975;
    let audio = if has_audio { 96_000.0 } else { 0.0 };
    let reserve = 64_000.0;
    (total - audio - reserve).max(250_000.0) as u64
}

pub fn gif_candidates(width: i32, height: i32, fps: f64) -> Vec<(f64, i32)> {
    let long = width.max(height).max(2);
    let mut dims = vec![long.min(2560)];
    for edge in [1920, 1600, 1280, 960, 720, 640, 480, 360] {
        if edge < dims[0] && edge >= 240 {
            dims.push(edge);
        }
    }
    let mut rates = vec![fps.min(60.0)];
    for rate in [50.0, 40.0, 30.0, 24.0, 20.0, 15.0, 12.0, 10.0, 8.0, 6.0, 5.0] {
        if rate < rates[0] - 0.1 {
            rates.push(rate);
        }
    }
    let mut out = Vec::new();
    for rate in rates {
        for edge in &dims {
            out.push((rate, *edge));
        }
    }
    out
}

pub fn cap_video(ffmpeg: &Path, input: &Path, output: &Path, seconds: f64, has_audio: bool) -> Result<()> {
    if file_len(input) <= HARD_LIMIT && input == output {
        return Ok(());
    }
    let bitrate = video_bitrate(seconds, has_audio);
    let mut scale = 1.0f32;
    for _ in 0..4 {
        let status = if has_audio {
            Command::new(ffmpeg)
                .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
                .arg(input)
                .args([
                    "-vf",
                    &format!("scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2"),
                    "-c:v",
                    "libx265",
                    "-profile:v",
                    "main10",
                    "-pix_fmt",
                    "yuv420p10le",
                    "-b:v",
                    &bitrate.to_string(),
                    "-tag:v",
                    "hvc1",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "96k",
                ])
                .arg(output)
                .status()?
        } else {
            Command::new(ffmpeg)
                .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
                .arg(input)
                .args([
                    "-vf",
                    &format!("scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2"),
                    "-c:v",
                    "libx265",
                    "-profile:v",
                    "main10",
                    "-pix_fmt",
                    "yuv420p10le",
                    "-b:v",
                    &bitrate.to_string(),
                    "-tag:v",
                    "hvc1",
                    "-an",
                ])
                .arg(output)
                .status()?
        };
        if !status.success() {
            return Err(anyhow!("Reddit Mode re-encode failed"));
        }
        if file_len(output) <= HARD_LIMIT {
            return Ok(());
        }
        scale *= 0.75;
    }
    if file_len(output) > HARD_LIMIT {
        return Err(anyhow!(
            "Reddit Mode could not get the file under 20 MB (it is {} bytes)",
            file_len(output)
        ));
    }
    Ok(())
}

pub fn write_gif(ffmpeg: &Path, input: &Path, output: &Path, width: i32, height: i32, fps: f64) -> Result<()> {
    let mut last_err = anyhow!("GIF encode did not run");
    for (rate, edge) in gif_candidates(width, height, fps) {
        let filter = format!(
            "fps={rate},scale='if(gt(iw,ih),{edge},-2)':'if(gt(iw,ih),-2,{edge})':flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse"
        );
        let status = Command::new(ffmpeg)
            .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
            .arg(input)
            .args(["-vf", &filter, "-loop", "0", "-an"])
            .arg(output)
            .status()?;
        if !status.success() {
            last_err = anyhow!("ffmpeg GIF encode failed at {rate} fps / {edge}px");
            continue;
        }
        if file_len(output) <= TARGET {
            return Ok(());
        }
    }
    if output.is_file() && file_len(output) <= HARD_LIMIT {
        return Ok(());
    }
    Err(last_err)
}

pub fn write_still_capped(ffmpeg: &Path, input: &Path, output: &Path, long_edge: i32) -> Result<()> {
    let mut edges = vec![long_edge.min(8192)];
    for edge in [6144, 5120, 4096, 3200, 2560, 2048, 1600, 1280, 960, 640] {
        if edge < edges[0] {
            edges.push(edge);
        }
    }
    for edge in edges {
        for quality in [100, 95, 90, 85, 80, 70, 60, 50, 40] {
            if try_still(ffmpeg, input, output, edge, quality)? && file_len(output) <= TARGET {
                return Ok(());
            }
        }
    }
    if output.is_file() && file_len(output) <= HARD_LIMIT {
        return Ok(());
    }
    Err(anyhow!("could not fit the still under 20 MB"))
}

fn try_still(ffmpeg: &Path, input: &Path, output: &Path, edge: i32, quality: i32) -> Result<bool> {
    let scale = format!("scale='if(gt(iw,ih),{edge},-2)':'if(gt(iw,ih),-2,{edge})':flags=lanczos");
    let avif = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(input)
        .args(["-vf", &scale, "-c:v", "libaom-av1", "-still-picture", "1", "-q:v"])
        .arg((100 - quality).max(1).to_string())
        .arg(output)
        .status();
    if matches!(&avif, Ok(status) if status.success()) && output.is_file() {
        return Ok(true);
    }
    let jpeg = output.with_extension("jpg");
    let status = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(input)
        .args(["-vf", &scale, "-q:v"])
        .arg(((100 - quality) / 3).clamp(2, 20).to_string())
        .arg(&jpeg)
        .status()?;
    if status.success() {
        if output != jpeg.as_path() {
            std::fs::rename(&jpeg, output).ok();
        }
        return Ok(true);
    }
    Ok(false)
}

fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reddit_bitrate_stays_under_the_20mb_budget() {
        let bits = video_bitrate(10.0, true);
        let bytes = (bits as f64 + 96_000.0) * 10.0 / 8.0;
        assert!(bytes < HARD_LIMIT as f64);
    }

    #[test]
    fn gif_search_tries_the_source_size_before_it_shrinks() {
        let choices = gif_candidates(1280, 720, 24.0);
        assert_eq!(choices[0], (24.0, 1280));
        assert!(choices.iter().any(|c| c.0 < 24.0 && c.1 < 1280));
    }
}
