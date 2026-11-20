//! ffmpeg subprocess pipes. Decode yields packed BGRA. Encode is libx265 Main10.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::{anyhow, Context, Result};
use rife_guards::Frame;

use crate::schedule::{parse_rate, Rational};
use crate::x265::{color_from_probe, ColorMeta};

#[derive(Clone, Debug)]
pub struct Probe {
    pub width: i32,
    pub height: i32,
    pub rate: Rational,
    pub frame_count: i64,
    pub duration: f64,
    pub color: ColorMeta,
    pub has_audio: bool,
}

pub fn probe(ffprobe: &Path, input: &Path) -> Result<Probe> {
    let output = Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
        ])
        .arg(input)
        .output()
        .with_context(|| format!("failed to run {}", ffprobe.display()))?;
    if !output.status.success() {
        return Err(anyhow!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).context("ffprobe json")?;
    parse_probe(&json)
}

pub fn parse_probe(json: &serde_json::Value) -> Result<Probe> {
    let streams = json
        .get("streams")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("ffprobe returned no streams"))?;
    let video = streams
        .iter()
        .find(|s| s.get("codec_type").and_then(|v| v.as_str()) == Some("video"))
        .ok_or_else(|| anyhow!("the input has no video stream"))?;
    let width = video.get("width").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
    let height = video.get("height").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
    if width < 8 || height < 8 {
        return Err(anyhow!("video size {width}x{height} is too small"));
    }
    let rate = video
        .get("avg_frame_rate")
        .and_then(|v| v.as_str())
        .and_then(parse_rate)
        .or_else(|| {
            video
                .get("r_frame_rate")
                .and_then(|v| v.as_str())
                .and_then(parse_rate)
        })
        .ok_or_else(|| anyhow!("could not read the source frame rate"))?;
    let nb = video
        .get("nb_frames")
        .and_then(|v| v.as_str())
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|n| *n > 0);
    let duration = video
        .get("duration")
        .and_then(|v| v.as_str())
        .and_then(|v| v.parse::<f64>().ok())
        .or_else(|| {
            json.get("format")
                .and_then(|f| f.get("duration"))
                .and_then(|v| v.as_str())
                .and_then(|v| v.parse::<f64>().ok())
        })
        .unwrap_or(0.0);
    let frame_count = nb.unwrap_or_else(|| {
        if duration > 0.0 {
            (duration * rate.num as f64 / rate.den as f64).round() as i64
        } else {
            0
        }
    });
    let color = color_from_probe(
        video
            .get("color_primaries")
            .and_then(|v| v.as_str())
            .unwrap_or(""),
        video
            .get("color_transfer")
            .and_then(|v| v.as_str())
            .unwrap_or(""),
        video
            .get("color_space")
            .and_then(|v| v.as_str())
            .unwrap_or(""),
        video
            .get("color_range")
            .and_then(|v| v.as_str())
            .unwrap_or(""),
    );
    let has_audio = streams
        .iter()
        .any(|s| s.get("codec_type").and_then(|v| v.as_str()) == Some("audio"));
    Ok(Probe {
        width,
        height,
        rate,
        frame_count,
        duration,
        color,
        has_audio,
    })
}

pub struct Decoder {
    child: Option<Child>,
    stderr: Arc<Mutex<String>>,
    width: i32,
    height: i32,
    frame_bytes: usize,
}

impl Decoder {
    pub fn spawn(
        ffmpeg: &Path,
        input: &Path,
        width: i32,
        height: i32,
        source_width: i32,
        source_height: i32,
    ) -> Result<Self> {
        let mut cmd = Command::new(ffmpeg);
        cmd.arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-i")
            .arg(input);
        if width != source_width || height != source_height {
            cmd.arg("-vf")
                .arg(format!("scale={width}:{height}:flags=lanczos"));
        }
        cmd.args([
            "-map", "0:v:0", "-an", "-f", "rawvideo", "-pix_fmt", "bgra", "-",
        ]);
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to start {}", ffmpeg.display()))?;
        let stderr = Arc::new(Mutex::new(String::new()));
        let pipe = child.stderr.take();
        let log = Arc::clone(&stderr);
        thread::spawn(move || {
            if let Some(mut pipe) = pipe {
                let mut buf = Vec::new();
                let _ = pipe.read_to_end(&mut buf);
                *log.lock().unwrap() = String::from_utf8_lossy(&buf).into_owned();
            }
        });
        Ok(Self {
            child: Some(child),
            stderr,
            width,
            height,
            frame_bytes: (width as usize) * (height as usize) * 4,
        })
    }

    pub fn read_frame(&mut self) -> Result<Option<Frame>> {
        let mut buf = vec![0u8; self.frame_bytes];
        let mut read = 0;
        let stdout = self
            .child
            .as_mut()
            .and_then(|child| child.stdout.as_mut())
            .ok_or_else(|| anyhow!("decoder stdout closed"))?;
        while read < buf.len() {
            match stdout.read(&mut buf[read..]) {
                Ok(0) if read == 0 => return Ok(None),
                Ok(0) => {
                    return Err(anyhow!(
                        "decoder ended in the middle of a frame ({read} of {} bytes)\n{}",
                        buf.len(),
                        self.stderr.lock().unwrap()
                    ));
                }
                Ok(n) => read += n,
                Err(err) => return Err(err).context("reading a decoded frame"),
            }
        }
        Ok(Some(Frame {
            width: self.width,
            height: self.height,
            rgba: false,
            data: buf,
        }))
    }

    pub fn finish(mut self) -> Result<()> {
        let Some(mut child) = self.child.take() else {
            return Ok(());
        };
        let status = child.wait().context("waiting for the decoder")?;
        if !status.success() {
            return Err(anyhow!(
                "decoder ffmpeg failed\n{}",
                self.stderr.lock().unwrap()
            ));
        }
        Ok(())
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub struct Encoder {
    child: Option<Child>,
    stderr: Arc<Mutex<String>>,
    frame_bytes: usize,
}

impl Encoder {
    pub fn spawn(
        ffmpeg: &Path,
        source: &Path,
        output: &Path,
        width: i32,
        height: i32,
        fps_num: i64,
        fps_den: i64,
        color: &ColorMeta,
    ) -> Result<Self> {
        Self::spawn_audio(ffmpeg, source, output, width, height, fps_num, fps_den, color, true)
    }

    pub fn spawn_audio(
        ffmpeg: &Path,
        source: &Path,
        output: &Path,
        width: i32,
        height: i32,
        fps_num: i64,
        fps_den: i64,
        color: &ColorMeta,
        preserve_audio: bool,
    ) -> Result<Self> {
        let args = crate::x265::encode_args_audio(
            &ffmpeg.display().to_string(),
            width,
            height,
            fps_num,
            fps_den,
            color,
            &source.display().to_string(),
            &output.display().to_string(),
            preserve_audio,
        );
        Self::from_args(ffmpeg, width, height, &args)
    }

    pub fn spawn_ffv1(
        ffmpeg: &Path,
        output: &Path,
        width: i32,
        height: i32,
        fps_num: i64,
        fps_den: i64,
    ) -> Result<Self> {
        let rate = format!("{fps_num}/{fps_den}");
        let args = vec![
            ffmpeg.display().to_string(),
            "-y".into(),
            "-hide_banner".into(),
            "-loglevel".into(),
            "error".into(),
            "-f".into(),
            "rawvideo".into(),
            "-pix_fmt".into(),
            "bgra".into(),
            "-s".into(),
            format!("{width}x{height}"),
            "-r".into(),
            rate,
            "-i".into(),
            "-".into(),
            "-an".into(),
            "-c:v".into(),
            "ffv1".into(),
            "-pix_fmt".into(),
            "gbrp".into(),
            output.display().to_string(),
        ];
        Self::from_args(ffmpeg, width, height, &args)
    }

    fn from_args(ffmpeg: &Path, width: i32, height: i32, args: &[String]) -> Result<Self> {
        let mut cmd = Command::new(&args[0]);
        cmd.args(&args[1..]);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to start {}", ffmpeg.display()))?;
        let stderr = Arc::new(Mutex::new(String::new()));
        let pipe = child.stderr.take();
        let log = Arc::clone(&stderr);
        thread::spawn(move || {
            if let Some(mut pipe) = pipe {
                let mut buf = Vec::new();
                let _ = pipe.read_to_end(&mut buf);
                *log.lock().unwrap() = String::from_utf8_lossy(&buf).into_owned();
            }
        });
        Ok(Self {
            child: Some(child),
            stderr,
            frame_bytes: (width as usize) * (height as usize) * 4,
        })
    }

    pub fn write_frame(&mut self, frame: &Frame) -> Result<()> {
        if frame.data.len() != self.frame_bytes {
            return Err(anyhow!(
                "encoder frame is {} bytes, expected {}",
                frame.data.len(),
                self.frame_bytes
            ));
        }
        let stdin = self
            .child
            .as_mut()
            .and_then(|child| child.stdin.as_mut())
            .ok_or_else(|| anyhow!("encoder stdin closed"))?;
        stdin
            .write_all(&frame.data)
            .context("writing a frame to libx265")?;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        let Some(mut child) = self.child.take() else {
            return Ok(());
        };
        drop(child.stdin.take());
        let status = child.wait().context("waiting for libx265")?;
        if !status.success() {
            return Err(anyhow!(
                "libx265 encode failed\n{}",
                self.stderr.lock().unwrap()
            ));
        }
        Ok(())
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub fn tool_path(explicit: Option<&Path>, name: &str) -> PathBuf {
    if let Some(path) = explicit {
        return path.to_path_buf();
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside = dir.join(name);
            if beside.exists() {
                return beside;
            }
        }
    }
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_probe_keeps_bt709_and_the_rate() {
        let json = serde_json::json!({
            "streams": [{
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "24000/1001",
                "nb_frames": "48",
                "duration": "2.002",
                "color_primaries": "bt709",
                "color_transfer": "bt709",
                "color_space": "bt709",
                "color_range": "tv"
            }, {
                "codec_type": "audio",
                "codec_name": "aac"
            }]
        });
        let probe = parse_probe(&json).unwrap();
        assert_eq!(probe.rate, Rational::new(24000, 1001));
        assert_eq!(probe.frame_count, 48);
        assert!(probe.has_audio);
        assert_eq!(probe.color.primaries.as_deref(), Some("bt709"));
        assert_eq!(probe.color.range.as_deref(), Some("limited"));
        assert_eq!(probe.color.ffmpeg_range.as_deref(), Some("tv"));
    }

    #[test]
    fn libx265_main10_roundtrip_accepts_the_preset() {
        let ffmpeg = PathBuf::from("ffmpeg");
        let ffprobe = PathBuf::from("ffprobe");
        if Command::new(&ffmpeg).arg("-version").output().is_err() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("rife-pc-x265-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let width = 64;
        let height = 64;
        let mut frames = Vec::new();
        for n in 0..4 {
            let mut frame = Frame::new(width, height, false);
            for y in 0..height {
                for x in 0..width {
                    let value = if (x - 20 - n).abs() <= 1 { 16 } else { 210 };
                    frame.set_rgb(x, y, value);
                }
            }
            frames.push(frame);
        }
        // A tiny silent source so the audio map is optional and still succeeds.
        let source = dir.join("src.mp4");
        let raw = dir.join("raw.bgra");
        let mut bytes = Vec::new();
        for frame in &frames {
            bytes.extend_from_slice(&frame.data);
        }
        std::fs::write(&raw, &bytes).unwrap();
        let status = Command::new(&ffmpeg)
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgra",
                "-s",
                "64x64",
                "-r",
                "24",
                "-i",
            ])
            .arg(&raw)
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-an"])
            .arg(&source)
            .status()
            .unwrap();
        assert!(status.success());
        let output = dir.join("out.mp4");
        let color = crate::x265::color_from_probe("bt709", "bt709", "bt709", "tv");
        let mut encoder =
            Encoder::spawn(&ffmpeg, &source, &output, width, height, 48, 1, &color).unwrap();
        for frame in &frames {
            encoder.write_frame(frame).unwrap();
        }
        encoder.finish().unwrap();
        let probed = probe(&ffprobe, &output).unwrap();
        assert_eq!(probed.width, 64);
        assert_eq!(probed.height, 64);
        assert_eq!(probed.color.primaries.as_deref(), Some("bt709"));
        assert_eq!(probed.color.ffmpeg_range.as_deref(), Some("tv"));
        let shown = Command::new(&ffprobe)
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,profile,pix_fmt",
                "-of",
                "csv=p=0",
            ])
            .arg(&output)
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&shown.stdout);
        assert!(text.contains("hevc"), "{text}");
        assert!(
            text.to_ascii_lowercase().contains("main 10") || text.contains("Main 10"),
            "{text}"
        );
        assert!(text.contains("yuv420p10le"), "{text}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
