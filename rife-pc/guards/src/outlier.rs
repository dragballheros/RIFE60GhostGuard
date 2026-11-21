//! Detects a real source-frame outlier before interpolation.

use crate::frame::{sample_rgb, Frame};
use crate::structural;

#[derive(Clone, Debug)]
pub struct OutlierResult {
    pub is_outlier: bool,
    pub score: f64,
    pub reason: String,
}

pub struct SourceFrameOutlierGuard {
    pub sensitivity: f64,
}

impl Default for SourceFrameOutlierGuard {
    fn default() -> Self {
        Self { sensitivity: 1.35 }
    }
}

impl SourceFrameOutlierGuard {
    pub fn inspect(&self, previous: &Frame, candidate: &Frame, next: &Frame) -> OutlierResult {
        let (Some(a), Some(b), Some(c)) = (
            Sample::new(previous),
            Sample::new(candidate),
            Sample::new(next),
        ) else {
            return OutlierResult {
                is_outlier: false,
                score: 0.0,
                reason: "analysis unavailable".into(),
            };
        };
        let ab = a.mean_abs(&b);
        let bc = b.mean_abs(&c);
        let ac = a.mean_abs(&c);
        let ab_color = a.mean_abs_color(&b);
        let bc_color = b.mean_abs_color(&c);
        let ac_color = a.mean_abs_color(&c);

        let cut_ab = ab.max(ab_color);
        let cut_bc = bc.max(bc_color);
        let cut_ac = ac.max(ac_color);
        let structure = structural::assess(previous, next, self.sensitivity, Some(candidate));
        if structure.as_ref().map(|s| s.crossfade_cut).unwrap_or(false) {
            return OutlierResult {
                is_outlier: true,
                score: 0.90,
                reason: "stray frame at scene cut".into(),
            };
        }
        let nearer_cut = cut_ab.min(cut_bc);
        let stray_limit = 0.10_f64.max(0.6 * cut_ac);
        if (cut_ac > 0.12
            || structure
                .as_ref()
                .map(|s| s.structural_cut)
                .unwrap_or(false))
            && nearer_cut > stray_limit
        {
            let score = (nearer_cut / cut_ac.max(0.012)).clamp(0.0, 1.0);
            return OutlierResult {
                is_outlier: true,
                score: score.max(0.90),
                reason: "stray frame at scene cut".into(),
            };
        }

        if ab <= 0.025 || bc <= 0.025 {
            return OutlierResult {
                is_outlier: false,
                score: 0.0,
                reason: "candidate close to a neighbor".into(),
            };
        }

        let detour_ratio = (ab + bc) / ac.max(0.012);
        let midpoint_error = b.midpoint_error(&a, &c);
        let neighbor_agreement = ab.min(bc) / ac.max(0.012);
        let flash_like = ac < 0.055 && ab > 0.10 && bc > 0.10;
        let detour_like = detour_ratio > 1.42
            && midpoint_error > 0.075
            && neighbor_agreement > 0.34
            && ab > 0.055
            && bc > 0.055;
        let score = (0.55 * ((detour_ratio - 1.0) / 1.0).clamp(0.0, 1.0)
            + 0.25 * (midpoint_error / 0.16).clamp(0.0, 1.0)
            + 0.20 * (neighbor_agreement / 0.75).clamp(0.0, 1.0))
        .clamp(0.0, 1.0);

        if flash_like {
            return OutlierResult {
                is_outlier: true,
                score: score.max(0.92),
                reason: "isolated source-frame flash/ghost".into(),
            };
        }
        if detour_like {
            return OutlierResult {
                is_outlier: true,
                score,
                reason: "temporal source-frame outlier".into(),
            };
        }
        OutlierResult {
            is_outlier: false,
            score,
            reason: "normal temporal path".into(),
        }
    }

    /// Repeats the neighbor a cut-boundary stray frame is closer to.
    pub fn nearer_is_next(&self, previous: &Frame, candidate: &Frame, next: &Frame) -> bool {
        let (Some(a), Some(b), Some(c)) = (
            Sample::new(previous),
            Sample::new(candidate),
            Sample::new(next),
        ) else {
            return false;
        };
        let ab = a.mean_abs(&b).max(a.mean_abs_color(&b));
        let bc = b.mean_abs(&c).max(b.mean_abs_color(&c));
        bc < ab
    }

    /// Linear bridge used for a flash/detour, never for a cross-fade cut.
    pub fn bridge(&self, previous: &Frame, next: &Frame, progress: f64) -> Option<Frame> {
        if previous.rgba || next.rgba {
            return None;
        }
        let width = previous.width.min(next.width);
        let height = previous.height.min(next.height);
        if width <= 0 || height <= 0 {
            return None;
        }
        let t = progress.clamp(0.0, 1.0);
        let mut output = Frame::new(width, height, false);
        for y in 0..height {
            let p = previous.row(y);
            let n = next.row(y);
            let o = output.row_mut(y);
            let bytes = (width as usize) * 4;
            if t == 0.0 {
                o[..bytes].copy_from_slice(&p[..bytes]);
                continue;
            }
            if t == 1.0 {
                o[..bytes].copy_from_slice(&n[..bytes]);
                continue;
            }
            for x in 0..bytes {
                let mixed = (1.0 - t) * p[x] as f64 + t * n[x] as f64 + 0.5;
                o[x] = mixed.clamp(0.0, 255.0) as u8;
            }
        }
        Some(output)
    }
}

struct Sample {
    values: Vec<f32>,
    color: Vec<f32>,
}

impl Sample {
    fn new(frame: &Frame) -> Option<Self> {
        if frame.rgba || frame.width < 64 || frame.height < 64 {
            return None;
        }
        let width = frame.width;
        let height = frame.height;
        let x_step = 1.max(width / 64);
        let y_step = 1.max(height / 36);
        let x_offset = x_step / 2;
        let y_offset = y_step / 2;
        let mut values = Vec::with_capacity(64 * 36);
        let mut color = Vec::with_capacity(64 * 36 * 3);
        for row in 0..36 {
            let y = (height - 1).min(y_offset + row * y_step);
            let source = frame.row(y);
            for column in 0..64 {
                let x = (width - 1).min(x_offset + column * x_step);
                let i = (x as usize) * 4;
                let s = sample_rgb(&source[i..], false);
                values.push(s.y);
                color.push(s.r);
                color.push(s.g);
                color.push(s.b);
            }
        }
        Some(Self { values, color })
    }

    fn mean_abs(&self, other: &Sample) -> f64 {
        let n = self.values.len().min(other.values.len());
        if n == 0 {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += (self.values[i] - other.values[i]).abs();
        }
        (sum / n as f32) as f64
    }

    fn mean_abs_color(&self, other: &Sample) -> f64 {
        let n = self.color.len().min(other.color.len());
        if n == 0 {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += (self.color[i] - other.color[i]).abs();
        }
        (sum / n as f32) as f64
    }

    fn midpoint_error(&self, first: &Sample, last: &Sample) -> f64 {
        let n = self
            .values
            .len()
            .min(first.values.len())
            .min(last.values.len());
        if n == 0 {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for i in 0..n {
            let midpoint = (first.values[i] + last.values[i]) * 0.5;
            sum += (self.values[i] - midpoint).abs();
        }
        (sum / n as f32) as f64
    }
}
