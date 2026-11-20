//! GhostGuard and the endpoint fallback policy.
//! Source-frame substitution is reserved for explicit temporal boundaries.

use crate::frame::{sample_rgb, Frame};
use crate::structural;
use crate::warped;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Disposition {
    RetainInterpolation,
    UsePreviousSource,
    UseCurrentSource,
}

pub fn endpoint_disposition(reason: &str, progress: f64) -> Disposition {
    let t = if progress.is_finite() {
        progress.clamp(0.0, 1.0)
    } else {
        0.5
    };
    match reason {
        "scene cut / transition discontinuity" | "stray frame at scene cut" => {
            if t < 0.5 {
                Disposition::UsePreviousSource
            } else {
                Disposition::UseCurrentSource
            }
        }
        "isolated scene-end frame" => Disposition::UseCurrentSource,
        _ => Disposition::RetainInterpolation,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GhostGuardResult {
    pub reject: bool,
    pub reason: String,
}

pub struct GhostGuard {
    pub sensitivity: f64,
    pub enable_scene_cuts: bool,
}

impl Default for GhostGuard {
    fn default() -> Self {
        Self {
            sensitivity: 1.35,
            enable_scene_cuts: true,
        }
    }
}

impl GhostGuard {
    pub fn new(sensitivity: f64, enable_scene_cuts: bool) -> Self {
        Self {
            sensitivity,
            enable_scene_cuts,
        }
    }

    pub fn inspect(
        &self,
        previous: &Frame,
        generated: &Frame,
        current: &Frame,
        previous_context: Option<&Frame>,
    ) -> GhostGuardResult {
        let Some(a) = Sampled::new(previous) else {
            return GhostGuardResult {
                reject: false,
                reason: "analysis unavailable".into(),
            };
        };
        let Some(m) = Sampled::new(generated) else {
            return GhostGuardResult {
                reject: false,
                reason: "analysis unavailable".into(),
            };
        };
        let Some(b) = Sampled::new(current) else {
            return GhostGuardResult {
                reject: false,
                reason: "analysis unavailable".into(),
            };
        };
        let ab = a.mean_abs(&b);
        let ab_color = a.mean_abs_color(&b);

        if let Some(context_frame) = previous_context {
            if let Some(context) = Sampled::new(context_frame) {
                let context_to_previous = context.mean_abs(&a);
                let context_to_current = context.mean_abs(&b);
                if context_to_previous > 0.14 && ab > 0.14 && context_to_current < 0.10 {
                    return GhostGuardResult {
                        reject: true,
                        reason: "isolated scene-end frame".into(),
                    };
                }
            }
        }

        let structure = structural::assess(previous, current, self.sensitivity, Some(generated));
        if self.enable_scene_cuts
            && (ab.max(ab_color) > 0.20 / self.sensitivity
                || structure.as_ref().map(|s| s.is_cut()).unwrap_or(false))
        {
            return GhostGuardResult {
                reject: true,
                reason: "scene cut / transition discontinuity".into(),
            };
        }

        let am = a.mean_abs(&m);
        let mb = m.mean_abs(&b);
        if ab > 0.025 && am.min(mb) > ab * (1.18 / self.sensitivity) {
            return GhostGuardResult {
                reject: true,
                reason: "temporal outlier".into(),
            };
        }

        let ea = a.edge_energy();
        let em = m.edge_energy();
        let eb = b.edge_energy();
        let edge_limit = ea.max(eb) * (1.62 / self.sensitivity) + 0.008;
        if ab > 0.035 && em > edge_limit {
            return GhostGuardResult {
                reject: true,
                reason: "double-edge/ghost energy".into(),
            };
        }

        let edge_floor = ea.min(eb);
        let collapse_factor = 0.8_f64.min(0.5 * self.sensitivity);
        if ab > 0.03 && edge_floor > 0.01 && em < edge_floor * collapse_factor {
            return GhostGuardResult {
                reject: true,
                reason: "edge collapse / smeared midframe".into(),
            };
        }

        let blend = a.blend_difference(&m, &b);
        if ab > 0.04 && blend > 0.23 / self.sensitivity {
            return GhostGuardResult {
                reject: true,
                reason: "midframe inconsistency".into(),
            };
        }

        if warped::rejects(previous, generated, current, self.sensitivity) {
            return GhostGuardResult {
                reject: true,
                reason: "warped/doubled outline".into(),
            };
        }

        GhostGuardResult {
            reject: false,
            reason: "ok".into(),
        }
    }
}

struct Sampled {
    w: i32,
    h: i32,
    luma: Vec<f32>,
    color: Vec<f32>,
}

impl Sampled {
    fn new(frame: &Frame) -> Option<Self> {
        let src_w = frame.width;
        let src_h = frame.height;
        if src_w < 1 || src_h < 1 {
            return None;
        }
        let step = 1.max(src_w.max(src_h) / 160);
        let mut values = Vec::new();
        let mut colors = Vec::new();
        let mut y = 0;
        while y < src_h {
            let row = frame.row(y);
            let mut x = 0;
            while x < src_w {
                let i = (x as usize) * 4;
                let s = sample_rgb(&row[i..], frame.rgba);
                values.push(s.y);
                colors.push(s.r);
                colors.push(s.g);
                colors.push(s.b);
                x += step;
            }
            y += step;
        }
        let w = ((src_w as f64) / (step as f64)).ceil() as i32;
        let h = ((src_h as f64) / (step as f64)).ceil() as i32;
        Some(Self {
            w,
            h,
            luma: values,
            color: colors,
        })
    }

    fn mean_abs(&self, other: &Sampled) -> f64 {
        let n = self.luma.len().min(other.luma.len());
        if n == 0 {
            return 0.0;
        }
        let mut s = 0.0f32;
        for i in 0..n {
            s += (self.luma[i] - other.luma[i]).abs();
        }
        (s / n as f32) as f64
    }

    fn mean_abs_color(&self, other: &Sampled) -> f64 {
        let n = self.color.len().min(other.color.len());
        if n == 0 {
            return 0.0;
        }
        let mut s = 0.0f32;
        for i in 0..n {
            s += (self.color[i] - other.color[i]).abs();
        }
        (s / n as f32) as f64
    }

    fn edge_energy(&self) -> f64 {
        if self.w <= 2 || self.h <= 2 || self.luma.len() < (self.w * self.h) as usize {
            return 0.0;
        }
        let mut sum = 0.0f32;
        let mut n = 0.0f32;
        let w = self.w as usize;
        for y in 1..(self.h - 1) {
            for x in 1..(self.w - 1) {
                let i = y as usize * w + x as usize;
                let gx = self.luma[i + 1] - self.luma[i - 1];
                let gy = self.luma[i + w] - self.luma[i - w];
                sum += gx.abs() + gy.abs();
                n += 1.0;
            }
        }
        (sum / n.max(1.0)) as f64
    }

    fn blend_difference(&self, mid: &Sampled, other: &Sampled) -> f64 {
        let n = self.luma.len().min(mid.luma.len()).min(other.luma.len());
        if n == 0 {
            return 0.0;
        }
        let mut s = 0.0f32;
        for i in 0..n {
            let linear = (self.luma[i] + other.luma[i]) * 0.5;
            s += (mid.luma[i] - linear).abs();
        }
        (s / n as f32) as f64
    }
}
