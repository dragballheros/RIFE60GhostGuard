//! Processing options shared by the window and the command line.
//! GUI defaults match the iOS toggles. CLI defaults keep the previous
//! `rife-pc input -o output` behavior (guards and audio on, extra passes off).

use serde::{Deserialize, Serialize};

use rife_guards::GuardFlags;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CorrectionMode {
    Off,
    OriginalOnly,
    ProcessedOnly,
    Full,
}

impl CorrectionMode {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "original" | "originalonly" | "original-only" => Some(Self::OriginalOnly),
            "processed" | "processedonly" | "processed-only" => Some(Self::ProcessedOnly),
            "full" => Some(Self::Full),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::OriginalOnly => "Original only",
            Self::ProcessedOnly => "Processed only",
            Self::Full => "Full",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PerformanceTier {
    Auto,
    Performance,
    Balanced,
    Safe,
}

impl PerformanceTier {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Performance => "Performance",
            Self::Balanced => "Balanced",
            Self::Safe => "Safe",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AnimeQuality {
    Auto,
    Apisr,
    Span,
}

impl AnimeQuality {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "apisr" => Some(Self::Apisr),
            "span" => Some(Self::Span),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Apisr => "APISR",
            Self::Span => "SPAN",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum WatermarkModelKind {
    Lama,
    DeepCreamBar,
    DeepCreamMosaic,
}

impl WatermarkModelKind {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "lama" => Some(Self::Lama),
            "bar" | "deepcream" | "deepcream-bar" | "deepcreambar" => Some(Self::DeepCreamBar),
            "mosaic" | "deepcream-mosaic" | "deepcreammosaic" => Some(Self::DeepCreamMosaic),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Lama => "LaMa (anime inpaint)",
            Self::DeepCreamBar => "DeepCream bar",
            Self::DeepCreamMosaic => "DeepCream mosaic",
        }
    }

    pub fn file_stem(self) -> &'static str {
        match self {
            Self::Lama => "AnimeWatermarkLaMa512",
            Self::DeepCreamBar => "DeepCreamBar",
            Self::DeepCreamMosaic => "DeepCreamMosaic",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RifeModelKind {
    V46,
    V426,
}

impl RifeModelKind {
    pub fn dir_name(self) -> &'static str {
        match self {
            Self::V46 => "rife-v4.6",
            Self::V426 => "rife-v4.26",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::V46 => "RIFE 4.6",
            Self::V426 => "RIFE 4.26",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.trim() {
            "4.6" | "v4.6" | "rife-v4.6" => Some(Self::V46),
            "4.26" | "v4.26" | "rife-v4.26" => Some(Self::V426),
            _ => None,
        }
    }
}

/// Normalized strokes and rectangles. Coordinates are 0–1 from the top left.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MaskSpec {
    pub strokes: Vec<Vec<[f32; 2]>>,
    pub rects: Vec<[f32; 4]>,
}

impl MaskSpec {
    pub fn is_empty(&self) -> bool {
        self.strokes.iter().all(|s| s.len() < 2) && self.rects.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct QueueItem {
    pub path: String,
    pub remove_watermark: bool,
}

impl Default for QueueItem {
    fn default() -> Self {
        Self {
            path: String::new(),
            remove_watermark: true,
        }
    }
}

impl QueueItem {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            remove_watermark: true,
        }
    }
}

/// Trim whitespace and one pair of surrounding quotes from a pasted Windows path.
pub fn clean_path(text: &str) -> String {
    let mut text = text.trim();
    if text.len() >= 2 {
        let quote = text.as_bytes()[0];
        let end = text.as_bytes()[text.len() - 1];
        if (quote == b'"' || quote == b'\'') && end == quote {
            text = text[1..text.len() - 1].trim();
        }
    }
    text.to_string()
}

pub fn missing_input_message(failed: &str) -> String {
    let failed = failed.trim();
    if failed.is_empty() {
        "Pick an input file (video or image) that exists.".into()
    } else {
        format!("Pick an input file (video or image) that exists. {failed}")
    }
}

fn same_path(left: &str, right: &str) -> bool {
    let fold = |text: &str| text.replace('/', "\\");
    if cfg!(windows) {
        fold(left).eq_ignore_ascii_case(&fold(right))
    } else {
        fold(left) == fold(right)
    }
}

/// The Input box runs even when a saved queue is present. It is prepended when
/// it is not already queued. Missing files are returned separately and not run.
pub fn plan_jobs(
    input: &str,
    queue: &[QueueItem],
    mut exists: impl FnMut(&str) -> bool,
) -> (Vec<QueueItem>, Vec<String>) {
    let input = clean_path(input);
    let mut run = Vec::new();
    let mut missing = Vec::new();
    let already = !input.is_empty()
        && queue.iter().any(|item| same_path(&clean_path(&item.path), &input));
    if !input.is_empty() && !already {
        if exists(&input) {
            run.push(QueueItem::new(input.clone()));
        } else {
            missing.push(input.clone());
        }
    }
    for item in queue {
        let path = clean_path(&item.path);
        if path.is_empty() {
            continue;
        }
        if !input.is_empty() && same_path(&path, &input) && missing.iter().any(|item| same_path(item, &input)) {
            continue;
        }
        if exists(&path) {
            run.push(QueueItem {
                path,
                remove_watermark: item.remove_watermark,
            });
        } else {
            missing.push(path);
        }
    }
    (run, missing)
}

/// Drop queued files that are gone and strip quotes from the ones that remain.
pub fn prune_queue(queue: &mut Vec<QueueItem>, mut exists: impl FnMut(&str) -> bool) -> Vec<String> {
    let mut missing = Vec::new();
    let mut kept = Vec::new();
    for item in std::mem::take(queue) {
        let path = clean_path(&item.path);
        if path.is_empty() {
            continue;
        }
        if exists(&path) {
            kept.push(QueueItem {
                path,
                remove_watermark: item.remove_watermark,
            });
        } else {
            missing.push(path);
        }
    }
    *queue = kept;
    missing
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ProcessingSettings {
    pub ghost_protection: bool,
    pub scene_cut_protection: bool,
    pub thin_line: bool,
    pub line_stabilizers: bool,
    pub source_outlier: bool,
    pub render_power: bool,
    pub performance_tier: PerformanceTier,
    /// 0 = VRAM-aware Auto, 1 = single worker, 2 = two concurrent RIFE calls.
    pub rife_workers: u8,
    pub preserve_audio: bool,
    pub reddit_mode: bool,
    pub video_to_gif: bool,
    pub compression_guard: bool,
    pub color_pop: bool,
    pub color_pop_strength: f64,
    pub warmth: CorrectionMode,
    pub skin_tone: CorrectionMode,
    pub color_rebuild: bool,
    pub color_rebuild_strength: f64,
    pub outline_rebuild: bool,
    pub outline_rebuild_strength: f64,
    pub sharpie: bool,
    pub anime_restore: bool,
    pub anime_restore_suppress_auto: bool,
    pub anime_restore_quality: AnimeQuality,
    pub anime_restore_strength: f64,
    pub upscale: bool,
    pub upscale_first: bool,
    pub cugan_noise: i32,
    pub watermark: bool,
    pub watermark_model: WatermarkModelKind,
    pub watermark_padding: f32,
    pub watermark_mask: MaskSpec,
    pub diagnostics: bool,
    pub export_dir: String,
    pub rife_model: RifeModelKind,
    pub queue: Vec<QueueItem>,
}

impl Default for ProcessingSettings {
    fn default() -> Self {
        Self::gui()
    }
}

impl ProcessingSettings {
    /// iOS toggle defaults. Real-CUGAN, color, and Sharpie start on.
    pub fn gui() -> Self {
        Self {
            ghost_protection: true,
            scene_cut_protection: true,
            thin_line: true,
            line_stabilizers: true,
            source_outlier: true,
            render_power: true,
            performance_tier: PerformanceTier::Auto,
            rife_workers: 0,
            preserve_audio: true,
            reddit_mode: false,
            video_to_gif: false,
            compression_guard: true,
            color_pop: true,
            color_pop_strength: 1.0,
            warmth: CorrectionMode::Full,
            skin_tone: CorrectionMode::ProcessedOnly,
            color_rebuild: true,
            color_rebuild_strength: 0.45,
            outline_rebuild: true,
            outline_rebuild_strength: 0.5,
            sharpie: true,
            anime_restore: false,
            anime_restore_suppress_auto: false,
            anime_restore_quality: AnimeQuality::Auto,
            anime_restore_strength: 0.8,
            upscale: true,
            upscale_first: false,
            cugan_noise: 3,
            watermark: false,
            watermark_model: WatermarkModelKind::Lama,
            watermark_padding: 1.0,
            watermark_mask: MaskSpec::default(),
            diagnostics: true,
            export_dir: String::new(),
            rife_model: RifeModelKind::V426,
            queue: Vec::new(),
        }
    }

    /// Previous command-line behavior, plus flags the user can turn on.
    pub fn cli() -> Self {
        Self {
            ghost_protection: true,
            scene_cut_protection: true,
            thin_line: true,
            line_stabilizers: true,
            source_outlier: true,
            render_power: false,
            performance_tier: PerformanceTier::Auto,
            rife_workers: 0,
            preserve_audio: true,
            reddit_mode: false,
            video_to_gif: false,
            compression_guard: false,
            color_pop: false,
            color_pop_strength: 1.0,
            warmth: CorrectionMode::Off,
            skin_tone: CorrectionMode::ProcessedOnly,
            color_rebuild: false,
            color_rebuild_strength: 0.45,
            outline_rebuild: false,
            outline_rebuild_strength: 0.5,
            sharpie: false,
            anime_restore: false,
            anime_restore_suppress_auto: true,
            anime_restore_quality: AnimeQuality::Auto,
            anime_restore_strength: 0.8,
            upscale: false,
            upscale_first: false,
            cugan_noise: 3,
            watermark: false,
            watermark_model: WatermarkModelKind::Lama,
            watermark_padding: 1.0,
            watermark_mask: MaskSpec::default(),
            diagnostics: false,
            export_dir: String::new(),
            rife_model: RifeModelKind::V46,
            queue: Vec::new(),
        }
    }

    pub fn guard_flags(&self) -> GuardFlags {
        GuardFlags {
            ghost: self.ghost_protection,
            scene_cuts: self.scene_cut_protection,
            thin_line: self.thin_line,
            static_lock: self.line_stabilizers,
            small_motion: self.line_stabilizers,
            stroke: self.line_stabilizers,
        }
    }

    pub fn anime_restore_active(&self, width: i32, height: i32) -> bool {
        if self.anime_restore {
            return true;
        }
        if self.anime_restore_suppress_auto {
            return false;
        }
        width.min(height) <= 720
    }

    pub fn visual_polish(&self) -> bool {
        self.color_rebuild
            || self.outline_rebuild
            || self.sharpie
            || self.color_pop
            || self.warmth != CorrectionMode::Off
            || self.skin_tone != CorrectionMode::Off
    }

    pub fn source_filters(&self, width: i32, height: i32) -> bool {
        self.anime_restore_active(width, height) || self.watermark || self.compression_guard
    }

    pub fn clamp_ranges(&mut self) {
        self.rife_workers = self.rife_workers.min(2);
        self.color_pop_strength = self.color_pop_strength.clamp(0.0, 1.0);
        self.color_rebuild_strength = self.color_rebuild_strength.clamp(0.1, 1.0);
        self.outline_rebuild_strength = self.outline_rebuild_strength.clamp(0.1, 1.0);
        self.anime_restore_strength = self.anime_restore_strength.clamp(0.1, 1.0);
        self.watermark_padding = self.watermark_padding.clamp(0.0, 16.0);
        self.cugan_noise = self.cugan_noise.clamp(-1, 3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults_do_not_turn_on_upscale_or_color() {
        let cli = ProcessingSettings::cli();
        assert!(!cli.upscale);
        assert!(!cli.color_pop);
        assert!(cli.preserve_audio);
        assert!(cli.ghost_protection);
        assert_eq!(cli.rife_model, RifeModelKind::V46);
    }

    #[test]
    fn gui_defaults_match_the_ios_toggles() {
        let gui = ProcessingSettings::gui();
        assert!(gui.upscale);
        assert!(!gui.upscale_first);
        assert!(gui.compression_guard);
        assert!(gui.color_pop);
        assert!(gui.sharpie);
        assert_eq!(gui.warmth, CorrectionMode::Full);
        assert_eq!(gui.skin_tone, CorrectionMode::ProcessedOnly);
        assert_eq!(gui.rife_model, RifeModelKind::V426);
        assert!(gui.anime_restore_active(1280, 720));
        assert!(!gui.anime_restore_active(1920, 1080));
    }

    #[test]
    fn clean_path_strips_windows_copy_as_path_quotes() {
        assert_eq!(clean_path("  \"C:\\clips\\shot.png\"  "), "C:\\clips\\shot.png");
        assert_eq!(clean_path("'D:\\out.mp4'"), "D:\\out.mp4");
        assert_eq!(clean_path("  plain.png  "), "plain.png");
    }

    #[test]
    fn the_input_box_is_prepended_and_a_saved_queue_cannot_hide_it() {
        let queue = vec![
            QueueItem::new("C:\\old\\missing.mp4"),
            QueueItem::new("\"C:\\old\\still.mp4\""),
        ];
        let (run, missing) = plan_jobs("  \"C:\\new\\shot.png\"  ", &queue, |path| path.ends_with("shot.png") || path.ends_with("still.mp4"));
        assert_eq!(run[0].path, "C:\\new\\shot.png");
        assert_eq!(run[1].path, "C:\\old\\still.mp4");
        assert_eq!(missing, vec!["C:\\old\\missing.mp4".to_string()]);
        assert!(missing_input_message("C:\\new\\shot.png").contains("shot.png"));
        assert!(missing_input_message("").starts_with("Pick an input file (video or image) that exists."));
    }

    #[test]
    fn an_input_already_in_the_queue_is_not_duplicated() {
        let queue = vec![QueueItem {
            path: "C:\\clips\\shot.png".into(),
            remove_watermark: false,
        }];
        let (run, missing) = plan_jobs("C:\\clips\\shot.png", &queue, |_| true);
        assert!(missing.is_empty());
        assert_eq!(run.len(), 1);
        assert!(!run[0].remove_watermark);
    }

    #[test]
    fn startup_prune_drops_missing_files_and_keeps_quoted_ones_that_exist() {
        let mut queue = vec![
            QueueItem::new("\"C:\\keep.png\""),
            QueueItem::new("C:\\gone.mp4"),
        ];
        let missing = prune_queue(&mut queue, |path| path.ends_with("keep.png"));
        assert_eq!(queue.len(), 1);
        assert_eq!(queue[0].path, "C:\\keep.png");
        assert_eq!(missing, vec!["C:\\gone.mp4".to_string()]);
    }
}
