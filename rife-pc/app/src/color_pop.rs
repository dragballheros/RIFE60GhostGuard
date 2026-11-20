//! Color Pop, automatic warmth, and the skin-tone pass from the iOS app.
//! The Core Image 33³ LUT is the same per-pixel formula applied directly.

use rife_guards::Frame;

use crate::pixels::{put_rgb, rgb_at, smoothstep};
use crate::settings::CorrectionMode;

#[derive(Clone, Debug)]
pub struct WarmthState {
    pub smoothed: f32,
    pub started: bool,
}

impl Default for WarmthState {
    fn default() -> Self {
        Self {
            smoothed: 0.0,
            started: false,
        }
    }
}

pub fn apply(
    frame: &Frame,
    strength: f32,
    warmth: CorrectionMode,
    skin: CorrectionMode,
    warmth_state: &mut WarmthState,
) -> Frame {
    let graded = if strength > 0.001 {
        grade(frame, strength)
    } else {
        frame.clone()
    };
    let warmed = apply_warmth(frame, &graded, warmth, warmth_state);
    apply_skin(frame, &warmed, skin)
}

fn grade(frame: &Frame, strength: f32) -> Frame {
    let mut out = Frame::new(frame.width, frame.height, frame.rgba);
    for y in 0..frame.height {
        for x in 0..frame.width {
            let rgb = rgb_at(frame, x, y);
            let graded = grade_pixel(
                rgb[0] as f32 / 255.0,
                rgb[1] as f32 / 255.0,
                rgb[2] as f32 / 255.0,
                strength,
            );
            put_rgb(
                &mut out,
                x,
                y,
                [
                    (graded[0] * 255.0).round().clamp(0.0, 255.0) as u8,
                    (graded[1] * 255.0).round().clamp(0.0, 255.0) as u8,
                    (graded[2] * 255.0).round().clamp(0.0, 255.0) as u8,
                ],
            );
        }
    }
    out
}

fn grade_pixel(r: f32, g: f32, b: f32, strength: f32) -> [f32; 3] {
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    if strength <= 0.0 || y <= 1e-5 {
        return [r, g, b];
    }
    let maxc = r.max(g).max(b);
    let minc = r.min(g).min(b);
    let chroma = maxc - minc;
    let chroma_weight = smoothstep(0.020, 0.095, chroma);
    if chroma_weight <= 0.0 {
        return [r, g, b];
    }
    let (hue, sat, _) = hsv(r, g, b);
    let skin = skin_weight(hue, sat, y, r, g, b);
    let shadow = smoothstep(0.015, 0.075, y);
    let highlight = 1.0 - smoothstep(0.84, 0.97, y);
    let usable = chroma_weight * shadow * highlight;
    let contrast = 0.035 * strength * usable * (1.0 - skin);
    let target_y = y + contrast * (y - 0.5) * 4.0 * y * (1.0 - y);
    let family = family_accent(hue);
    let proposed = 1.0 + strength * (0.28 * (1.0 - strength).powf(0.75) + family) * usable * (1.0 - 0.92 * skin);
    let tone = if y > 1e-5 { target_y / y } else { 1.0 };
    let tr = (r * tone).clamp(0.0, 1.0);
    let tg = (g * tone).clamp(0.0, 1.0);
    let tb = (b * tone).clamp(0.0, 1.0);
    let dr = tr - target_y;
    let dg = tg - target_y;
    let db = tb - target_y;
    let gamut = gamut_limit(target_y, dr, dg, db);
    let gain = proposed.min(gamut).max(0.0);
    [
        (target_y + dr * gain).clamp(0.0, 1.0),
        (target_y + dg * gain).clamp(0.0, 1.0),
        (target_y + db * gain).clamp(0.0, 1.0),
    ]
}

fn skin_weight(hue: f32, sat: f32, y: f32, r: f32, g: f32, b: f32) -> f32 {
    let hue_ok = smoothstep(2.0, 8.0, hue) * (1.0 - smoothstep(36.0, 43.0, hue))
        + smoothstep(334.0, 340.0, hue) * (1.0 - smoothstep(348.0, 354.0, hue));
    let sat_ok = smoothstep(0.045, 0.12, sat) * (1.0 - smoothstep(0.62, 0.82, sat));
    let y_ok = smoothstep(0.08, 0.16, y) * (1.0 - smoothstep(0.82, 0.95, y));
    let rg = if g > 1e-4 { r / g } else { 0.0 };
    let gb = if b > 1e-4 { g / b } else { 0.0 };
    let rg_ok = smoothstep(0.99, 1.08, rg) * (1.0 - smoothstep(1.45, 1.62, rg));
    let gb_ok = smoothstep(1.0, 1.15, gb) * (1.0 - smoothstep(2.2, 2.65, gb));
    (hue_ok * sat_ok * y_ok * rg_ok * gb_ok).clamp(0.0, 1.0)
}

fn family_accent(hue: f32) -> f32 {
    let band = |center: f32, width: f32, amount: f32| {
        let d = (hue - center).abs().min((hue - center + 360.0).abs());
        amount * (1.0 - smoothstep(width * 0.45, width, d))
    };
    band(0.0, 18.0, 0.10)
        + band(330.0, 24.0, 0.08)
        + band(190.0, 28.0, 0.12)
        + band(220.0, 26.0, 0.14)
        + band(120.0, 30.0, 0.08)
        + band(30.0, 18.0, 0.05)
}

fn gamut_limit(y: f32, dr: f32, dg: f32, db: f32) -> f32 {
    let mut limit = 8.0f32;
    for d in [dr, dg, db] {
        if d > 1e-5 {
            limit = limit.min((1.0 - y) / d);
        } else if d < -1e-5 {
            limit = limit.min((0.0 - y) / d);
        }
    }
    limit.max(0.0)
}

fn hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let maxc = r.max(g).max(b);
    let minc = r.min(g).min(b);
    let d = maxc - minc;
    let s = if maxc <= 1e-6 { 0.0 } else { d / maxc };
    let h = if d <= 1e-6 {
        0.0
    } else if (maxc - r).abs() < 1e-5 {
        60.0 * (((g - b) / d) % 6.0)
    } else if (maxc - g).abs() < 1e-5 {
        60.0 * (((b - r) / d) + 2.0)
    } else {
        60.0 * (((r - g) / d) + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    (h, s, maxc)
}

fn apply_warmth(source: &Frame, graded: &Frame, mode: CorrectionMode, state: &mut WarmthState) -> Frame {
    if mode == CorrectionMode::Off {
        state.smoothed *= 0.35;
        return graded.clone();
    }
    let estimate = warmth_estimate(source, graded);
    let added = (estimate.added_shift - 0.0010).max(0.0) * 0.98;
    let original = (estimate.source_bias - 0.006).max(0.0) * 0.78;
    let selected = match mode {
        CorrectionMode::Off => 0.0,
        CorrectionMode::OriginalOnly => original,
        CorrectionMode::ProcessedOnly => added,
        CorrectionMode::Full => added + original,
    };
    let target = selected.clamp(0.0, 0.095);
    let confidence = ((estimate.samples as f32 - 8.0) / 40.0).clamp(0.0, 1.0);
    let scaled = target * confidence;
    if scaled <= 0.0 {
        state.smoothed *= 0.35;
    } else if !state.started {
        state.smoothed = scaled;
        state.started = true;
    } else {
        state.smoothed = 0.35 * state.smoothed + 0.65 * scaled;
    }
    let amount = state.smoothed;
    if amount <= 0.0008 {
        return graded.clone();
    }
    let mut out = Frame::new(graded.width, graded.height, graded.rgba);
    for y in 0..graded.height {
        for x in 0..graded.width {
            let src = rgb_at(source, x, y);
            let pix = rgb_at(graded, x, y);
            let corrected = warmth_pixel(src, pix, amount);
            put_rgb(&mut out, x, y, corrected);
        }
    }
    out
}

struct WarmthEstimate {
    source_bias: f32,
    added_shift: f32,
    samples: usize,
}

fn warmth_estimate(source: &Frame, graded: &Frame) -> WarmthEstimate {
    let max_edge = source.width.max(source.height).max(1) as f32;
    let step = (max_edge / 64.0).ceil().max(1.0) as i32;
    let mut biases = Vec::new();
    let mut shifts = Vec::new();
    let mut y = 0;
    while y < source.height {
        let mut x = 0;
        while x < source.width {
            let s = rgb_at(source, x, y);
            let g = rgb_at(graded, x, y);
            let sr = s[0] as f32 / 255.0;
            let sg = s[1] as f32 / 255.0;
            let sb = s[2] as f32 / 255.0;
            let gr = g[0] as f32 / 255.0;
            let gb = g[2] as f32 / 255.0;
            let luma = 0.2126 * sr + 0.7152 * sg + 0.0722 * sb;
            let chroma = (sr - sg).abs().max((sg - sb).abs());
            let sat = {
                let maxc = sr.max(sg).max(sb);
                let minc = sr.min(sg).min(sb);
                if maxc <= 1e-6 { 0.0 } else { (maxc - minc) / maxc }
            };
            if (0.075..=0.94).contains(&luma)
                && chroma < 0.27
                && (sr - sg).abs() < 0.20
                && (sg - sb).abs() < 0.20
                && sat < 0.68
            {
                let bias = sr - sb;
                if bias > 0.0 {
                    biases.push(bias);
                }
                shifts.push((gr - gb) - (sr - sb));
            }
            x += step.max(4);
        }
        y += step.max(4);
    }
    WarmthEstimate {
        source_bias: median(&mut biases),
        added_shift: median_positive(&mut shifts),
        samples: biases.len(),
    }
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

fn median_positive(values: &mut Vec<f32>) -> f32 {
    values.retain(|v| *v > 0.0);
    median(values)
}

fn warmth_pixel(source: [u8; 3], graded: [u8; 3], amount: f32) -> [u8; 3] {
    let r = graded[0] as f32 / 255.0;
    let g = graded[1] as f32 / 255.0;
    let b = graded[2] as f32 / 255.0;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let chroma = (r - g).abs().max((g - b).abs());
    if y < 0.025 || y > 0.985 || chroma < 0.002 {
        return graded;
    }
    let (hue, sat, _) = hsv(r, g, b);
    let sr = source[0] as f32 / 255.0;
    let sg = source[1] as f32 / 255.0;
    let sb = source[2] as f32 / 255.0;
    let (sh, ss, _) = hsv(sr, sg, sb);
    let skin = if (3.0..=42.0).contains(&sh) && (0.025..=0.78).contains(&ss) {
        1.0
    } else {
        0.0
    };
    let neutral = 1.0 - smoothstep(0.045, 0.22, chroma);
    let warm = smoothstep(18.0, 28.0, hue)
        * (1.0 - smoothstep(52.0, 62.0, hue))
        * smoothstep(0.12, 0.22, sat)
        * (1.0 - smoothstep(0.75, 0.92, sat))
        * 0.62;
    let weight = neutral.max(warm) * (1.0 - 0.98 * skin);
    let correction = amount * weight;
    let rr = (r - 0.52 * correction).clamp(0.0, 1.0);
    let bb = (b + 0.48 * correction).clamp(0.0, 1.0);
    [
        (rr * 255.0).round() as u8,
        graded[1],
        (bb * 255.0).round() as u8,
    ]
}

fn apply_skin(source: &Frame, graded: &Frame, mode: CorrectionMode) -> Frame {
    if mode == CorrectionMode::Off {
        return graded.clone();
    }
    let original = matches!(mode, CorrectionMode::OriginalOnly | CorrectionMode::Full);
    let processed = matches!(mode, CorrectionMode::ProcessedOnly | CorrectionMode::Full);
    let mut out = Frame::new(graded.width, graded.height, graded.rgba);
    for y in 0..graded.height {
        for x in 0..graded.width {
            let s = rgb_at(source, x, y);
            let g = rgb_at(graded, x, y);
            put_rgb(&mut out, x, y, skin_pixel(s, g, original, processed));
        }
    }
    out
}

fn rec709(rgb: [u8; 3]) -> f32 {
    0.2126 * rgb[0] as f32 / 255.0 + 0.7152 * rgb[1] as f32 / 255.0 + 0.0722 * rgb[2] as f32 / 255.0
}

fn skin_pixel(source: [u8; 3], graded: [u8; 3], original: bool, processed: bool) -> [u8; 3] {
    let gr = graded[0] as f32 / 255.0;
    let gg = graded[1] as f32 / 255.0;
    let gb = graded[2] as f32 / 255.0;
    let sy = rec709(source);
    let gy = rec709(graded);
    let rg = (gr - gg).max(0.0);
    let gb_d = (gg - gb).max(0.0);
    let skin = smoothstep(0.02, 0.08, rg)
        * smoothstep(0.01, 0.06, gb_d)
        * (1.0 - smoothstep(0.28, 0.45, rg))
        * (1.0 - smoothstep(0.35, 0.55, gb_d))
        * smoothstep(0.16, 0.28, sy)
        * (1.0 - smoothstep(0.96, 0.995, sy));
    let source_excess = if original {
        ((sy - 0.76).max(0.0) * 0.14).min(0.025)
    } else {
        0.0
    };
    let processing_lift = if processed {
        ((gy - sy).max(0.0) * 0.85).min(0.035)
    } else {
        0.0
    };
    let correction = (source_excess + processing_lift) * skin;
    let scale = (gy - correction).max(0.0) / gy.max(0.001);
    [
        (gr * scale * 255.0).round().clamp(0.0, 255.0) as u8,
        (gg * scale * 255.0).round().clamp(0.0, 255.0) as u8,
        (gb * scale * 255.0).round().clamp(0.0, 255.0) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_is_left_alone() {
        let pixel = grade_pixel(0.4, 0.4, 0.4, 1.0);
        assert!((pixel[0] - 0.4).abs() < 1e-4);
        assert!((pixel[1] - 0.4).abs() < 1e-4);
    }

    #[test]
    fn a_saturated_red_does_not_leave_the_gamut() {
        let pixel = grade_pixel(0.9, 0.1, 0.1, 1.0);
        assert!(pixel.iter().all(|c| (0.0..=1.0).contains(c)));
        assert!(pixel[0] >= pixel[1]);
    }
}
