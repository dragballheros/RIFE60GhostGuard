//! Reddit Mode and video-to-GIF, both capped at 20 MB.
//! iOS Reddit Mode re-encodes stills as AVIF. This port uses the same byte
//! budget and the same resolution/quality ladder, with ffmpeg's AVIF encoder
//! when it is present and JPEG when it is not. GIF uses ffmpeg's palette.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Result};

pub fn is_still_path(path: &Path) -> bool {
    matches!(
        extension(path).as_str(),
        "png" | "jpg" | "jpeg" | "webp" | "bmp" | "tif" | "tiff"
    )
}

pub fn is_image_extension(ext: &str) -> bool {
    matches!(ext, "png" | "jpg" | "jpeg" | "webp" | "bmp")
}

/// Extension written for a still. JPEG stays JPEG. Everything else becomes PNG.
pub fn still_extension(path: &Path) -> &'static str {
    match extension(path).as_str() {
        "jpg" | "jpeg" => "jpg",
        "webp" => "webp",
        "bmp" => "bmp",
        _ => "png",
    }
}

/// Where a job should write. A still never inherits an mp4/mkv/mov name, and an
/// export folder wins over a video filename left in the output field.
pub fn delivery_path(
    input: &Path,
    export_dir: &str,
    output_override: &str,
    video_to_gif: bool,
    reddit_mode: bool,
) -> PathBuf {
    let still = is_still_path(input);
    let ext = if still {
        if reddit_mode { "jpg" } else { still_extension(input) }
    } else if video_to_gif {
        "gif"
    } else {
        "mp4"
    };
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let auto_name = format!("{stem}-60.{ext}");
    let override_trim = output_override.trim();
    if !override_trim.is_empty() {
        let chosen = PathBuf::from(override_trim);
        if chosen.is_dir() {
            return chosen.join(auto_name);
        }
        let chosen_ext = extension(&chosen);
        if still && !is_image_extension(&chosen_ext) {
            if !export_dir.trim().is_empty() {
                return PathBuf::from(export_dir.trim()).join(auto_name);
            }
            return replace_extension(&chosen, ext);
        }
        if !still && (chosen_ext.is_empty() || (video_to_gif && chosen_ext != "gif")) {
            if video_to_gif {
                return replace_extension(&chosen, "gif");
            }
            if chosen_ext.is_empty() {
                return replace_extension(&chosen, ext);
            }
        }
        return chosen;
    }
    if !export_dir.trim().is_empty() {
        return PathBuf::from(export_dir.trim()).join(auto_name);
    }
    input.with_file_name(auto_name)
}

/// Last-line guard so a still cannot enter libx265 because the caller passed an mp4 path.
pub fn coerce_still_output(input: &Path, output: &Path, reddit_mode: bool) -> PathBuf {
    let ext = if reddit_mode { "jpg" } else { still_extension(input) };
    let current = extension(output);
    if !reddit_mode && is_image_extension(&current) {
        return output.to_path_buf();
    }
    if reddit_mode && current == "jpg" {
        return output.to_path_buf();
    }
    replace_extension(output, ext)
}

/// H3 and AI Image Studio. An empty output field lands in the export folder.
pub fn placed_output(export_dir: &str, output_override: &str, fallback_name: &str) -> PathBuf {
    let override_trim = output_override.trim();
    if !override_trim.is_empty() {
        let chosen = PathBuf::from(override_trim);
        if chosen.is_dir() {
            return chosen.join(fallback_name);
        }
        let parent_missing = chosen.parent().map(|p| p.as_os_str().is_empty()).unwrap_or(true);
        if parent_missing && !export_dir.trim().is_empty() {
            return PathBuf::from(export_dir.trim()).join(chosen);
        }
        return chosen;
    }
    if !export_dir.trim().is_empty() {
        return PathBuf::from(export_dir.trim()).join(fallback_name);
    }
    PathBuf::from(fallback_name)
}

pub fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn replace_extension(path: &Path, ext: &str) -> PathBuf {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
    let name = format!("{stem}.{ext}");
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(name),
        _ => PathBuf::from(name),
    }
}

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
    if extension(output) == "avif" {
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

    #[test]
    fn a_png_in_the_export_folder_stays_a_png() {
        let path = delivery_path(
            Path::new("C:/clips/shot.PNG"),
            "D:/export",
            "D:/export/shot-60.mp4",
            false,
            false,
        );
        assert_eq!(path, PathBuf::from("D:/export/shot-60.png"));
        assert!(is_still_path(Path::new("shot.PNG")));
    }

    #[test]
    fn jpeg_keeps_its_extension_and_webp_stays_webp() {
        let jpeg = delivery_path(Path::new("/in/a.jpeg"), "", "", false, false);
        assert_eq!(jpeg.file_name().and_then(|s| s.to_str()), Some("a-60.jpg"));
        let webp = delivery_path(Path::new("/in/a.webp"), "/out", "", true, false);
        assert_eq!(webp, PathBuf::from("/out/a-60.webp"));
    }

    #[test]
    fn an_explicit_png_path_is_kept_and_reddit_stills_are_jpeg() {
        let kept = delivery_path(Path::new("a.png"), "/out", "/custom/pic.png", false, false);
        assert_eq!(kept, PathBuf::from("/custom/pic.png"));
        let reddit = delivery_path(Path::new("a.png"), "/out", "", false, true);
        assert_eq!(reddit, PathBuf::from("/out/a-60.jpg"));
    }

    #[test]
    fn video_gif_and_h3_use_the_export_folder() {
        let gif = delivery_path(Path::new("/in/clip.mp4"), "/out", "", true, false);
        assert_eq!(gif, PathBuf::from("/out/clip-60.gif"));
        let video = delivery_path(Path::new("/in/clip.mov"), "/out", "", false, false);
        assert_eq!(video, PathBuf::from("/out/clip-60.mp4"));
        assert_eq!(placed_output("/out", "", "h3.mp4"), PathBuf::from("/out/h3.mp4"));
        assert_eq!(
            placed_output("/out", "studio.png", "studio.png"),
            PathBuf::from("/out/studio.png")
        );
        let coerced = coerce_still_output(Path::new("a.png"), Path::new("/out/a-60.mp4"), false);
        assert_eq!(coerced, PathBuf::from("/out/a-60.png"));
    }
}
