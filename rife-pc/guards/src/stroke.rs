//! Stroke-width consistency from commit 7d8c9be.
//!
//! The in-between's mean line width must stay within 20% of the nearer
//! motion-compensated source, and the stroke must not be darker than that
//! source. Otherwise the nearer real frame is repeated. Without a verified
//! motion field, a mismatch is ignored unless the source strokes sit inside
//! the in-between (a widened copy of the same line).

use crate::frame::{luma_plane_parallel, Frame};
use crate::small_motion::{global_shift, line_mask, lines_contained, mean_stroke_width};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrokeDecision {
    Keep,
    /// Repeat this endpoint. `true` means the next frame, `false` the previous.
    RepeatNearer {
        nearer_is_next: bool,
    },
}

pub fn stroke_width_decision(
    previous: &Frame,
    mid: &Frame,
    next: &Frame,
    progress: f64,
    sensitivity: f64,
) -> StrokeDecision {
    if previous.width != mid.width
        || next.width != mid.width
        || previous.height != mid.height
        || next.height != mid.height
        || previous.rgba != mid.rgba
        || next.rgba != mid.rgba
    {
        return StrokeDecision::Keep;
    }
    let sensitivity = crate::frame::clamp_sensitivity(sensitivity);
    let t = if progress.is_finite() {
        progress.clamp(0.0, 1.0)
    } else {
        0.5
    };
    let nearer_is_next = t > 0.5;
    let shift = global_shift(previous, next, sensitivity);
    let place_scale = if nearer_is_next { t - 1.0 } else { t };
    let (dx, dy) = if shift.explained {
        (
            (place_scale * shift.dx as f64).round() as i32,
            (place_scale * shift.dy as f64).round() as i32,
        )
    } else {
        (0, 0)
    };
    let contrast = (0.11 / sensitivity).max(0.055) as f32;
    let nearer = if nearer_is_next { next } else { previous };
    let source_luma = luma_plane_parallel(nearer);
    let mid_luma = luma_plane_parallel(mid);
    let width = mid.width;
    let height = mid.height;
    let source_lines = line_mask(&source_luma, width, height, contrast);
    let mid_lines = line_mask(&mid_luma, width, height, contrast);
    // 7d8c9be skips a zero-offset repair when the source strokes are not inside
    // the in-between. That is the "no verified motion field" case.
    if !shift.explained && dx == 0 && dy == 0 && !lines_contained(&source_lines, &mid_lines) {
        return StrokeDecision::Keep;
    }
    let _ = (dx, dy);
    let source_width = mean_stroke_width(&source_lines, width, height);
    let mid_width = mean_stroke_width(&mid_lines, width, height);
    if source_width <= 0.5 || mid_width <= 0.5 {
        return StrokeDecision::Keep;
    }
    // Width is taken on each frame's own mask. Motion compensation chooses the
    // nearer source (and whether the field is trusted); a pure translation does
    // not change mean run length.
    let wider_or_narrower = (mid_width - source_width).abs() / source_width > 0.20;
    let darker =
        mean_line_luma(&mid_luma, &mid_lines) + 0.01 < mean_line_luma(&source_luma, &source_lines);
    if wider_or_narrower || darker {
        StrokeDecision::RepeatNearer { nearer_is_next }
    } else {
        StrokeDecision::Keep
    }
}

fn mean_line_luma(luma: &[f32], lines: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    let mut n = 0.0f32;
    for index in 0..lines.len() {
        if lines[index] != 0 {
            sum += luma[index];
            n += 1.0;
        }
    }
    if n == 0.0 {
        1.0
    } else {
        sum / n
    }
}
