//! ffmpeg subprocess pipes. Decode yields packed BGRA. Windows uses AMD AMF HEVC when available, with libx265 as a compatibility fallback.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use rife_guards::Frame;

use crate::schedule::{parse_rate, Rational};
use crate::x265::{color_from_probe, ColorMeta};

/// Returns true when the bundled FFmpeg exposes the AMD AMF HEVC encoder.
/// The driver is still validated by FFmpeg when the encoder process starts.
pub fn hevc_amf_available(ffmpeg: &Path) -> bool {
    if !cfg!(windows) {
        return false;
    }
    Command::new(ffmpeg)
        .args(["-hide_banner", "-encoders"])
        .output()
        .map(|output| {
            output.status.success()
                && String::from_utf8_lossy(&output.stdout).lines().any(|line| {
                    let fields: Vec<&str> = line.split_whitespace().collect();
                    fields.iter().any(|field| *field == "hevc_amf")
                })
        })
        .unwrap_or(false)
}

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
        if let Some(pipe) = child.stderr.take() {
            drain_pipe(pipe, Arc::clone(&stderr));
        }
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
    /// `None` asks the writer to drop stdin so ffmpeg sees EOF.
    writer_tx: Option<mpsc::SyncSender<Option<Vec<u8>>>>,
    writer: Option<thread::JoinHandle<std::io::Result<()>>>,
}

const WRITE_TIMEOUT: Duration = Duration::from_secs(120);
const FINISH_TIMEOUT: Duration = Duration::from_secs(180);

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
        let args = if hevc_amf_available(ffmpeg) {
            let profile = if width.max(height) >= 3840 { "Main 8-bit" } else { "Main10" };
            eprintln!(
                "rife-pc: using AMD AMF hardware HEVC {} encoder for {}x{}",
                profile, width, height
            );
            crate::x265::encode_args_amf_audio(
                &ffmpeg.display().to_string(),
                width,
                height,
                fps_num,
                fps_den,
                color,
                &source.display().to_string(),
                &output.display().to_string(),
                preserve_audio,
            )
        } else {
            eprintln!(
                "rife-pc: AMD AMF HEVC is unavailable; using libx265 software fallback for {}x{}",
                width, height
            );
            crate::x265::encode_args_audio(
                &ffmpeg.display().to_string(),
                width,
                height,
                fps_num,
                fps_den,
                color,
                &source.display().to_string(),
                &output.display().to_string(),
                preserve_audio,
            )
        };
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
            "-nostats".into(),
            "-loglevel".into(),
            "error".into(),
            "-thread_queue_size".into(),
            "1024".into(),
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
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to start {}", ffmpeg.display()))?;
        let stderr = Arc::new(Mutex::new(String::new()));
        if let Some(pipe) = child.stderr.take() {
            drain_pipe(pipe, Arc::clone(&stderr));
        }
        if let Some(pipe) = child.stdout.take() {
            drain_pipe(pipe, Arc::clone(&stderr));
        }
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("encoder stdin was not piped"))?;
        let (tx, rx) = mpsc::sync_channel::<Option<Vec<u8>>>(1);
        let writer = thread::spawn(move || {
            let mut stdin = stdin;
            while let Ok(Some(frame)) = rx.recv() {
                stdin.write_all(&frame)?;
                stdin.flush()?;
            }
            // Dropping stdin sends EOF to FFmpeg. Do it before the parent calls wait,
            // or the muxer can block forever.
            drop(stdin);
            Ok(())
        });
        Ok(Self {
            child: Some(child),
            stderr,
            frame_bytes: (width as usize) * (height as usize) * 4,
            writer_tx: Some(tx),
            writer: Some(writer),
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
        if let Some(message) = self.exited_early() {
            return Err(message);
        }
        if self.writer_tx.is_none() {
            return Err(anyhow!("encoder stdin is already closed"));
        }
        let mut payload = Some(frame.data.clone());
        let deadline = Instant::now() + WRITE_TIMEOUT;
        loop {
            let send_result = {
                let tx = self
                    .writer_tx
                    .as_ref()
                    .ok_or_else(|| anyhow!("encoder stdin is already closed"))?;
                payload.take().map(|bytes| tx.try_send(Some(bytes)))
            };
            match send_result {
                Some(Ok(())) => return Ok(()),
                Some(Err(TrySendError::Disconnected(_))) => {
                    return Err(self.stopped_error("libx265 closed its input while a frame was waiting"));
                }
                Some(Err(TrySendError::Full(returned))) => {
                    payload = returned;
                }
                None => {}
            }
            if let Some(message) = self.exited_early() {
                return Err(message);
            }
            if Instant::now() >= deadline {
                self.kill_child();
                return Err(self.stopped_error(&format!(
                    "libx265 stopped consuming frames for {}s",
                    WRITE_TIMEOUT.as_secs()
                )));
            }
            thread::sleep(Duration::from_millis(40));
        }
    }

    pub fn finish(mut self) -> Result<()> {
        // Closing the sender makes the writer drop stdin, which is EOF.
        self.writer_tx.take();
        let deadline = Instant::now() + FINISH_TIMEOUT;
        while self.writer.as_ref().is_some_and(|handle| !handle.is_finished()) {
            if Instant::now() >= deadline {
                self.kill_child();
                return Err(self.stopped_error("timed out closing encoder input"));
            }
            thread::sleep(Duration::from_millis(40));
        }
        if let Some(handle) = self.writer.take() {
            match handle.join() {
                Ok(Err(err)) => {
                    self.kill_child();
                    return Err(self.stopped_error(&format!("writing a frame to encoder: {err}")));
                }
                Err(_) => {
                    self.kill_child();
                    return Err(self.stopped_error("the encoder writer thread panicked"));
                }
                Ok(Ok(())) => {}
            }
        }
        let Some(mut child) = self.child.take() else {
            return Ok(());
        };
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    if status.success() {
                        return Ok(());
                    }
                    return Err(anyhow!(
                        "libx265 encode failed ({status})\n{}",
                        self.stderr.lock().unwrap()
                    ));
                }
                Ok(None) if Instant::now() >= deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(self.stopped_error("timed out waiting for encoder to exit after EOF"));
                }
                Ok(None) => thread::sleep(Duration::from_millis(40)),
                Err(err) => return Err(err).context("waiting for libx265"),
            }
        }
    }

    fn exited_early(&mut self) -> Option<anyhow::Error> {
        let status = self.child.as_mut().and_then(|child| child.try_wait().ok()).flatten()?;
        Some(anyhow!(
            "encoder exited with {status} before the frame was written\n{}",
            self.stderr.lock().unwrap()
        ))
    }

    fn stopped_error(&self, why: &str) -> anyhow::Error {
        anyhow!("{why}\n{}", self.stderr.lock().unwrap())
    }

    fn kill_child(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        self.writer_tx.take();
        self.kill_child();
        if let Some(handle) = self.writer.take() {
            let _ = handle.join();
        }
    }
}

fn drain_pipe(mut pipe: impl Read + Send + 'static, log: Arc<Mutex<String>>) {
    thread::spawn(move || {
        let mut buf = [0u8; 4096];
        loop {
            match pipe.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let mut guard = log.lock().unwrap();
                    guard.push_str(&String::from_utf8_lossy(&buf[..n]));
                    if guard.len() > 64_000 {
                        let drop_to = guard.len() - 32_000;
                        guard.drain(..drop_to);
                    }
                }
            }
        }
    });
}

pub struct RgbaImage {
    pub width: i32,
    pub height: i32,
    pub rgba: Vec<u8>,
}

pub fn pixel_size(ffprobe: &Path, input: &Path) -> Result<(i32, i32)> {
    let output = Command::new(ffprobe)
        .args(["-v", "error", "-print_format", "json", "-show_streams"])
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
    let streams = json
        .get("streams")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("ffprobe returned no streams for {}", input.display()))?;
    let video = streams
        .iter()
        .find(|s| s.get("codec_type").and_then(|v| v.as_str()) == Some("video"))
        .or_else(|| streams.first())
        .ok_or_else(|| anyhow!("{} has no picture stream", input.display()))?;
    let width = video.get("width").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
    let height = video.get("height").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
    if width < 1 || height < 1 {
        return Err(anyhow!("could not read the picture size of {}", input.display()));
    }
    Ok((width, height))
}

pub fn decode_image(ffmpeg: &Path, input: &Path, width: i32, height: i32, rgba: bool) -> Result<Frame> {
    if width < 1 || height < 1 {
        return Err(anyhow!("picture size {width}x{height} is empty"));
    }
    let pix = if rgba { "rgba" } else { "bgra" };
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(input)
        .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", pix, "-s"])
        .arg(format!("{width}x{height}"))
        .arg("-")
        .output()
        .with_context(|| format!("failed to start {}", ffmpeg.display()))?;
    if !output.status.success() {
        return Err(anyhow!(
            "could not read {}\n{}",
            input.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let expected = (width as usize) * (height as usize) * 4;
    if output.stdout.len() < expected {
        return Err(anyhow!(
            "read {} bytes from {}, expected {expected}\n{}",
            output.stdout.len(),
            input.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(Frame {
        width,
        height,
        rgba,
        data: output.stdout[..expected].to_vec(),
    })
}

pub fn preview_rgba(input: &Path) -> Result<RgbaImage> {
    let ffmpeg = tool_path(None, preview_bin("ffmpeg"));
    let ffprobe = tool_path(None, preview_bin("ffprobe"));
    let (width, height) = pixel_size(&ffprobe, input)?;
    let (width, height) = fit_edge(width, height, 1280);
    let frame = decode_image(&ffmpeg, input, width, height, true)?;
    Ok(RgbaImage {
        width: frame.width,
        height: frame.height,
        rgba: frame.data,
    })
}

pub fn write_image(ffmpeg: &Path, frame: &Frame, output: &Path) -> Result<()> {
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let expected = (frame.width as usize) * (frame.height as usize) * 4;
    if frame.data.len() < expected || frame.width < 1 || frame.height < 1 {
        return Err(anyhow!(
            "refusing to write {}x{} from {} bytes",
            frame.width,
            frame.height,
            frame.data.len()
        ));
    }
    let raw = output.with_extension("bgra");
    std::fs::write(&raw, &frame.data[..expected])?;
    let ext = crate::export_media::extension(output);
    let mut cmd = Command::new(ffmpeg);
    cmd.args([
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        if frame.rgba { "rgba" } else { "bgra" },
        "-s",
    ])
    .arg(format!("{}x{}", frame.width, frame.height))
    .arg("-i")
    .arg(&raw)
    .args(["-frames:v", "1"]);
    match ext.as_str() {
        "jpg" | "jpeg" => {
            cmd.args(["-c:v", "mjpeg", "-q:v", "2"]);
        }
        "webp" => {
            cmd.args(["-c:v", "libwebp", "-q:v", "90"]);
        }
        "bmp" => {
            cmd.args(["-c:v", "bmp"]);
        }
        _ => {
            cmd.args(["-c:v", "png"]);
        }
    }
    let result = cmd.arg(output).output();
    let _ = std::fs::remove_file(&raw);
    let result = result.with_context(|| format!("failed to start {}", ffmpeg.display()))?;
    if !result.status.success() {
        return Err(anyhow!(
            "could not write {}\n{}",
            output.display(),
            String::from_utf8_lossy(&result.stderr)
        ));
    }
    let len = std::fs::metadata(output).map(|meta| meta.len()).unwrap_or(0);
    if len == 0 {
        return Err(anyhow!("wrote an empty image at {}", output.display()));
    }
    Ok(())
}

fn fit_edge(width: i32, height: i32, edge: i32) -> (i32, i32) {
    let long = width.max(height).max(1);
    if long <= edge {
        return (width.max(1), height.max(1));
    }
    let scale = edge as f32 / long as f32;
    (
        ((width as f32) * scale).round().max(1.0) as i32,
        ((height as f32) * scale).round().max(1.0) as i32,
    )
}

fn preview_bin(stem: &str) -> &'static str {
    match stem {
        "ffmpeg" if cfg!(windows) => "ffmpeg.exe",
        "ffprobe" if cfg!(windows) => "ffprobe.exe",
        "ffmpeg" => "ffmpeg",
        _ => "ffprobe",
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
