//! DJATOM x265 preset from `X265PresetParams.h` / commit 126d4ed.
//! Output is 10-bit Main 10. `fps` and color tags are rewritten from the source
//! so the file keeps the real rate and the source color metadata.

use std::fmt::Write as _;

pub const PRESET_TEXT: &str = include_str!("x265_preset.txt");

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorMeta {
    pub primaries: Option<String>,
    pub transfer: Option<String>,
    pub matrix: Option<String>,
    /// x265 `range` value: `limited` or `full`.
    pub range: Option<String>,
    /// ffmpeg `-color_range` value: `tv` or `pc`.
    pub ffmpeg_range: Option<String>,
}

impl ColorMeta {
    pub fn unknown() -> Self {
        Self {
            primaries: None,
            transfer: None,
            matrix: None,
            range: None,
            ffmpeg_range: None,
        }
    }
}

/// Build an explicit RGB -> YUV conversion so the encoder cannot silently
/// choose a different range for the packed BGRA input. FFmpeg documents RGB
/// as full-range input and allows the output YUV range to be forced explicitly.
fn rgb_to_yuv_filter(color: &ColorMeta, output_format: &str) -> String {
    let out_range = color.ffmpeg_range.as_deref().unwrap_or("tv");
    let mut filter = format!(
        "scale=iw:ih:in_range=full:out_range={out_range}"
    );
    if let Some(matrix) = color.matrix.as_deref() {
        // The pipe is RGB/BGRA, so only the destination YUV matrix is
        // meaningful here. Supplying an input YUV matrix would be incorrect.
        filter.push_str(&format!(":out_color_matrix={matrix}"));
    }
    filter.push_str(&format!(",format={output_format}"));
    filter
}

fn output_range(color: &ColorMeta) -> &'static str {
    match color.ffmpeg_range.as_deref() {
        Some("pc") => "pc",
        _ => "tv",
    }
}

pub fn x265_params(
    width: i32,
    height: i32,
    fps_num: i64,
    fps_den: i64,
    color: &ColorMeta,
) -> String {
    let mut parts = split_params(PRESET_TEXT.trim());
    set_key(&mut parts, "fps", &format!("{fps_num}/{fps_den}"));
    // Never let the preset's baked-in range decide the output. The source
    // range is supplied by the caller and the RGB->YUV filter uses the same
    // range explicitly.
    set_key(&mut parts, "range", output_range(color));
    let side = width.min(height);
    if side < 32 {
        set_key(&mut parts, "ctu", "16");
    } else if side < 64 {
        set_key(&mut parts, "ctu", "32");
    }
    if let Some(value) = color.primaries.as_deref() {
        set_key(&mut parts, "colorprim", value);
    }
    if let Some(value) = color.transfer.as_deref() {
        set_key(&mut parts, "transfer", value);
    }
    if let Some(value) = color.matrix.as_deref() {
        set_key(&mut parts, "colormatrix", value);
    }
    if let Some(value) = color.range.as_deref() {
        set_key(&mut parts, "range", value);
    }
    parts.join(":")
}

/// ffmpeg argv for the libx265 Main10 encode. Audio is stream-copied.
pub fn encode_args(
    ffmpeg: &str,
    width: i32,
    height: i32,
    fps_num: i64,
    fps_den: i64,
    color: &ColorMeta,
    source_for_audio: &str,
    output: &str,
) -> Vec<String> {
    encode_args_audio(
        ffmpeg, width, height, fps_num, fps_den, color, source_for_audio, output, true,
    )
}

/// FFmpeg argv for AMD AMF HEVC Main10. This path keeps the raw BGRA pipe but
/// converts to 10-bit P010 for the hardware encoder. Audio is stream-copied.
pub fn encode_args_amf_audio(
    ffmpeg: &str,
    width: i32,
    height: i32,
    fps_num: i64,
    fps_den: i64,
    color: &ColorMeta,
    source_for_audio: &str,
    output: &str,
    preserve_audio: bool,
) -> Vec<String> {
    let rate = format!("{fps_num}/{fps_den}");
    let mut args = vec![
        ffmpeg.to_string(), "-y".into(), "-hide_banner".into(), "-nostats".into(),
        "-loglevel".into(), "error".into(), "-thread_queue_size".into(), "1024".into(),
        "-f".into(), "rawvideo".into(), "-pix_fmt".into(), "bgra".into(),
        "-s".into(), format!("{width}x{height}"), "-r".into(), rate.clone(), "-i".into(), "-".into(),
    ];
    if preserve_audio {
        args.extend(["-i".into(), source_for_audio.to_string(), "-map".into(), "0:v:0".into(), "-map".into(), "1:a?".into()]);
    } else {
        args.extend(["-map".into(), "0:v:0".into()]);
    }
    args.extend([
        "-vf".into(), rgb_to_yuv_filter(color, "p010le").into(),
        "-c:v".into(), "hevc_amf".into(),
        "-profile:v".into(), "main10".into(),
        "-quality".into(), "balanced".into(),
        "-rc".into(), "cqp".into(),
        "-qp_i".into(), "20".into(),
        "-qp_p".into(), "22".into(),
        "-tag:v".into(), "hvc1".into(),
        "-r".into(), rate,
    ]);
    if preserve_audio { args.extend(["-c:a".into(), "copy".into()]); } else { args.push("-an".into()); }
    if let Some(value) = color.primaries.as_deref() { args.extend(["-color_primaries".into(), value.to_string()]); }
    if let Some(value) = color.transfer.as_deref() { args.extend(["-color_trc".into(), value.to_string()]); }
    if let Some(value) = color.matrix.as_deref() { args.extend(["-colorspace".into(), value.to_string()]); }
    if let Some(value) = color.ffmpeg_range.as_deref() { args.extend(["-color_range".into(), value.to_string()]); }
    args.push(output.to_string());
    args
}

pub fn encode_args_audio(
    ffmpeg: &str,
    width: i32,
    height: i32,
    fps_num: i64,
    fps_den: i64,
    color: &ColorMeta,
    source_for_audio: &str,
    output: &str,
    preserve_audio: bool,
) -> Vec<String> {
    let params = x265_params(width, height, fps_num, fps_den, color);
    let rate = format!("{fps_num}/{fps_den}");
    let mut args = vec![
        ffmpeg.to_string(),
        "-y".into(),
        "-hide_banner".into(),
        "-nostats".into(),
        "-loglevel".into(),
        "error".into(),
        // The raw pipe and the audio file are read together. A short queue
        // deadlocks once libx265 stops pulling frames and the pipe fills.
        "-thread_queue_size".into(),
        "1024".into(),
        "-f".into(),
        "rawvideo".into(),
        "-pix_fmt".into(),
        "bgra".into(),
        "-s".into(),
        format!("{width}x{height}"),
        "-r".into(),
        rate.clone(),
        "-i".into(),
        "-".into(),
    ];
    if preserve_audio {
        args.extend([
            "-i".into(),
            source_for_audio.to_string(),
            "-map".into(),
            "0:v:0".into(),
            "-map".into(),
            "1:a?".into(),
        ]);
    } else {
        args.extend(["-map".into(), "0:v:0".into()]);
    }
    args.extend([
        "-c:v".into(),
        "libx265".into(),
        "-preset".into(),
        "medium".into(),
        "-profile:v".into(),
        "main10".into(),
        "-pix_fmt".into(),
        "yuv420p10le".into(),
        "-x265-params".into(),
        params,
        "-tag:v".into(),
        "hvc1".into(),
    ]);
    if preserve_audio {
        args.extend(["-c:a".into(), "copy".into()]);
    } else {
        args.push("-an".into());
    }
    if let Some(value) = color.primaries.as_deref() {
        args.push("-color_primaries".into());
        args.push(value.to_string());
    }
    if let Some(value) = color.transfer.as_deref() {
        args.push("-color_trc".into());
        args.push(value.to_string());
    }
    if let Some(value) = color.matrix.as_deref() {
        args.push("-colorspace".into());
        args.push(value.to_string());
    }
    args.push("-color_range".into());
    args.push(output_range(color).into());
    args.push(output.to_string());
    args
}

pub fn color_from_probe(primaries: &str, transfer: &str, matrix: &str, range: &str) -> ColorMeta {
    ColorMeta {
        primaries: map_name(primaries, PRIMARIES),
        transfer: map_name(transfer, TRANSFER),
        matrix: map_name(matrix, MATRIX),
        range: match range {
            "tv" | "mpeg" | "limited" => Some("limited".into()),
            "pc" | "jpeg" | "full" => Some("full".into()),
            _ => None,
        },
        ffmpeg_range: match range {
            "tv" | "mpeg" | "limited" => Some("tv".into()),
            "pc" | "jpeg" | "full" => Some("pc".into()),
            _ => None,
        },
    }
}

const PRIMARIES: &[&str] = &[
    "bt709",
    "bt470m",
    "bt470bg",
    "smpte170m",
    "smpte240m",
    "film",
    "bt2020",
    "smpte428",
    "smpte431",
    "smpte432",
];
const TRANSFER: &[&str] = &[
    "bt709",
    "bt470m",
    "bt470bg",
    "smpte170m",
    "smpte240m",
    "linear",
    "log100",
    "log316",
    "iec61966-2-4",
    "bt1361e",
    "iec61966-2-1",
    "bt2020-10",
    "bt2020-12",
    "smpte2084",
    "smpte428",
    "arib-std-b67",
];
const MATRIX: &[&str] = &[
    "gbr",
    "bt709",
    "fcc",
    "bt470bg",
    "smpte170m",
    "smpte240m",
    "ycgco",
    "bt2020nc",
    "bt2020c",
    "smpte2085",
];

fn map_name(value: &str, allowed: &[&str]) -> Option<String> {
    let value = value.trim();
    if value.is_empty() || value == "unknown" || value == "unspecified" {
        return None;
    }
    allowed
        .iter()
        .find(|item| **item == value)
        .map(|item| (*item).to_string())
}

fn split_params(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            cur.push(ch);
            if let Some(next) = chars.next() {
                cur.push(next);
            }
        } else if ch == ':' {
            parts.push(std::mem::take(&mut cur));
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    parts
}

fn set_key(parts: &mut [String], key: &str, value: &str) {
    let prefix = format!("{key}=");
    for part in parts.iter_mut() {
        if part.starts_with(&prefix) {
            part.clear();
            let _ = write!(part, "{key}={value}");
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_keeps_crf_and_rewrites_rate_and_color() {
        let color = color_from_probe("bt709", "bt709", "bt709", "tv");
        let params = x265_params(1920, 1080, 60, 1, &color);
        assert!(params.contains("crf=20.0"), "{params}");
        assert!(params.contains("psy-rd=2.00"), "{params}");
        assert!(params.contains("aq-mode=3"), "{params}");
        assert!(params.contains("fps=60/1"), "{params}");
        assert!(params.contains("colorprim=bt709"), "{params}");
        assert!(params.contains("transfer=bt709"), "{params}");
        assert!(params.contains("colormatrix=bt709"), "{params}");
        assert!(params.contains("range=limited"), "{params}");
        assert!(params.contains("deblock=0\\:0"), "{params}");
        assert!(params.contains("ctu=64"), "{params}");
        assert!(!params.contains("colorprim=unknown"));
    }

    #[test]
    fn encode_args_are_main10_and_copy_audio() {
        let color = color_from_probe("bt709", "smpte2084", "bt2020nc", "pc");
        let args = encode_args(
            "ffmpeg", 1920, 1080, 48000, 1001, &color, "in.mp4", "out.mp4",
        );
        let joined = args.join(" ");
        assert!(joined.contains("libx265"));
        assert!(joined.contains("-nostats"));
        assert!(joined.contains("-thread_queue_size 1024"));
        assert!(joined.contains("-profile:v main10"));
        assert!(joined.contains("yuv420p10le"));
        assert!(joined.contains("-preset medium"));
        assert!(joined.contains("-c:a copy"));
        assert!(joined.contains("-tag:v hvc1"));
        assert!(joined.contains("-color_primaries bt709"));
        assert!(joined.contains("-color_trc smpte2084"));
        assert!(joined.contains("-colorspace bt2020nc"));
        assert!(joined.contains("-color_range pc"));
        assert!(args.iter().any(|arg| arg.contains("crf=20.0")));
        assert_eq!(args.last().map(String::as_str), Some("out.mp4"));
    }

    #[test]
    fn rgb_to_yuv_filter_is_explicitly_range_preserving() {
        let limited = color_from_probe("bt709", "bt709", "bt709", "tv");
        let full = color_from_probe("bt709", "bt709", "bt709", "pc");
        let limited_filter = rgb_to_yuv_filter(&limited, "p010le");
        let full_filter = rgb_to_yuv_filter(&full, "yuv420p10le");
        assert!(limited_filter.contains("in_range=full:out_range=tv"));
        assert!(limited_filter.contains("out_color_matrix=bt709"));
        assert!(full_filter.contains("in_range=full:out_range=pc"));
        assert!(full_filter.contains("format=yuv420p10le"));
        assert!(!limited_filter.contains("in_color_matrix"));
    }

    #[test]
    fn amf_args_use_hevc_main10_and_p010() {
        let color = color_from_probe("bt709", "bt709", "bt709", "tv");
        let args = encode_args_amf_audio("ffmpeg", 1920, 1080, 60, 1, &color, "in.mp4", "out.mp4", true);
        let joined = args.join(" ");
        assert!(joined.contains("hevc_amf"));
        assert!(joined.contains("main10"));
        assert!(joined.contains("p010le"));
        assert!(joined.contains("-rc cqp"));
        assert!(joined.contains("-qp_i 20"));
        assert!(joined.contains("-qp_p 22"));
        assert!(joined.contains("-tag:v hvc1"));
        assert!(joined.contains("-c:a copy"));
        assert!(joined.contains("in_range=full:out_range=tv"));
        assert!(joined.contains("out_color_matrix=bt709"));
    }

    #[test]
    fn small_frames_drop_ctu_below_the_preset_64() {
        let params = x265_params(48, 48, 24, 1, &ColorMeta::unknown());
        assert!(params.contains("ctu=32"), "{params}");
    }
}
