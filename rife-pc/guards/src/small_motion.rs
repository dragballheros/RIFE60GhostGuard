//! Keeps thin line art temporally aligned during a coherent global pan.
//! Ambiguous or local motion is left to RIFE. A detected cut returns nothing.

use rayon::prelude::*;

use crate::frame::{box_mean, byte_round, luma_at, Frame};
use crate::structural;

#[derive(Clone, Copy, Debug)]
pub struct Shift {
    pub dx: i32,
    pub dy: i32,
    #[allow(dead_code)]
    pub residual: f32,
    pub explained: bool,
}

pub struct SmallMotionLineStabilizer {
    pub sensitivity: f64,
}

impl SmallMotionLineStabilizer {
    pub fn new(sensitivity: f64) -> Self {
        Self {
            sensitivity: crate::frame::clamp_sensitivity(sensitivity),
        }
    }

    pub fn stabilize(
        &self,
        previous: &Frame,
        interpolated: &Frame,
        next: &Frame,
        progress: f64,
    ) -> Option<Frame> {
        if previous.rgba != interpolated.rgba || previous.rgba != next.rgba {
            return None;
        }
        let width = previous.width;
        let height = previous.height;
        if width < 32
            || height < 32
            || interpolated.width != width
            || next.width != width
            || interpolated.height != height
            || next.height != height
        {
            return None;
        }
        if let Some(cut) = structural::assess(previous, next, self.sensitivity, Some(interpolated))
        {
            if cut.is_cut() {
                return None;
            }
        }
        let t = if progress.is_finite() {
            progress.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let line_contrast = (0.11 / self.sensitivity).max(0.055) as f32;
        let shift = global_shift(previous, next, self.sensitivity);
        if !shift.explained {
            return None;
        }
        composite_temporal(
            interpolated,
            previous,
            next,
            t,
            shift.dx,
            shift.dy,
            line_contrast,
        )
    }
}

pub fn global_shift(a: &Frame, c: &Frame, sensitivity: f64) -> Shift {
    let width = a.width;
    let height = a.height;
    let rgba = a.rgba;
    let half_w = width / 2;
    let half_h = height / 2;
    if half_w < 8 || half_h < 8 {
        return Shift {
            dx: 0,
            dy: 0,
            residual: 1.0,
            explained: false,
        };
    }
    let mut half_a = vec![1.0f32; (half_w * half_h) as usize];
    let mut half_c = vec![1.0f32; (half_w * half_h) as usize];
    for y in 0..height {
        let a_line = a.row(y);
        let c_line = c.row(y);
        let hy = (half_h - 1).min(y / 2);
        for x in 0..width {
            let hx = (half_w - 1).min(x / 2);
            let index = (hy * half_w + hx) as usize;
            half_a[index] = half_a[index].min(luma_at(a_line, x, rgba));
            half_c[index] = half_c[index].min(luma_at(c_line, x, rgba));
        }
    }
    let mut best = (0i32, 0i32, f32::MAX);
    for sy in -6..=6 {
        for sx in -6..=6 {
            let score = half_sad(&half_a, &half_c, half_w, half_h, sx, sy);
            if score < best.2 {
                best = (sx * 2, sy * 2, score);
            }
        }
    }
    let mut refined = (best.0, best.1, f32::MAX);
    for dy in -2..=2 {
        for dx in -2..=2 {
            let sx = (best.0 + dx).clamp(-12, 12);
            let sy = (best.1 + dy).clamp(-12, 12);
            let score = full_sad(a, c, sx, sy);
            if score < refined.2 {
                refined = (sx, sy, score);
            }
        }
    }
    let zero = full_sad(a, c, 0, 0);
    let sad_limit = (0.034 + 0.014 * (sensitivity - 0.65) / 0.70) as f32;
    let moved = refined.0 != 0 || refined.1 != 0;
    let line_agreement = shifted_line_agreement(a, c, refined.0, refined.1);
    let explained = moved
        && refined.0.abs() <= 12
        && refined.1.abs() <= 12
        && refined.2 < sad_limit
        && refined.2 < zero * 0.55
        && line_agreement >= 0.90;
    Shift {
        dx: refined.0,
        dy: refined.1,
        residual: refined.2,
        explained,
    }
}

fn half_sad(a: &[f32], c: &[f32], width: i32, height: i32, dx: i32, dy: i32) -> f32 {
    let mut sum = 0.0f32;
    let mut n = 0.0f32;
    let w = width as usize;
    for y in 0..height {
        let yy = y + dy;
        if yy < 0 || yy >= height {
            continue;
        }
        for x in 0..width {
            let xx = x + dx;
            if xx < 0 || xx >= width {
                continue;
            }
            sum += (a[(y as usize) * w + x as usize] - c[(yy as usize) * w + xx as usize]).abs();
            n += 1.0;
        }
    }
    if n > 0.0 {
        sum / n
    } else {
        1.0
    }
}

fn full_sad(a: &Frame, c: &Frame, dx: i32, dy: i32) -> f32 {
    let mut sum = 0.0f32;
    let mut n = 0.0f32;
    let mut y = 0;
    while y < a.height {
        let yy = y + dy;
        if yy >= 0 && yy < c.height {
            let a_line = a.row(y);
            let c_line = c.row(yy);
            let mut x = 0;
            while x < a.width {
                let xx = x + dx;
                if xx >= 0 && xx < c.width {
                    sum += (luma_at(a_line, x, a.rgba) - luma_at(c_line, xx, c.rgba)).abs();
                    n += 1.0;
                }
                x += 2;
            }
        }
        y += 2;
    }
    if n > 0.0 {
        sum / n
    } else {
        1.0
    }
}

fn shifted_line_agreement(a: &Frame, c: &Frame, dx: i32, dy: i32) -> f32 {
    let mut hits = 0.0f32;
    let mut matched = 0.0f32;
    let mut y = 2;
    while y < a.height - 2 {
        let a_line = a.row(y);
        let up = a.row(y - 1);
        let down = a.row(y + 1);
        let mut x = 2;
        while x < a.width - 2 {
            let center = luma_at(a_line, x, a.rgba);
            let around = luma_at(a_line, x - 1, a.rgba)
                .max(luma_at(a_line, x + 1, a.rgba))
                .max(luma_at(up, x, a.rgba).max(luma_at(down, x, a.rgba)));
            if around - center >= 0.12 && center < 0.55 {
                hits += 1.0;
                let xx = x + dx;
                let yy = y + dy;
                if xx >= 1 && yy >= 1 && xx < c.width - 1 && yy < c.height - 1 {
                    let c_center = luma_at(c.row(yy), xx, c.rgba);
                    if c_center <= center + 0.08 {
                        matched += 1.0;
                    }
                }
            }
            x += 1;
        }
        y += 1;
    }
    if hits < 8.0 {
        1.0
    } else {
        matched / hits
    }
}

fn composite_temporal(
    mid: &Frame,
    previous: &Frame,
    next: &Frame,
    progress: f64,
    dx: i32,
    dy: i32,
    line_contrast: f32,
) -> Option<Frame> {
    let t = if progress.is_finite() {
        progress.clamp(0.0, 1.0) as f32
    } else {
        0.5
    };
    let width = mid.width;
    let height = mid.height;
    let rgba = mid.rgba;
    let previous_luma = crate::frame::luma_plane_parallel(previous);
    let next_luma = crate::frame::luma_plane_parallel(next);
    let previous_lines = expanded_line_mask(
        &line_mask(&previous_luma, width, height, line_contrast),
        width,
        height,
    );
    let next_lines = expanded_line_mask(
        &line_mask(&next_luma, width, height, line_contrast),
        width,
        height,
    );
    let mut output = mid.clone();
    let previous_x_offset = t * dx as f32;
    let previous_y_offset = t * dy as f32;
    let next_x_offset = (1.0 - t) * dx as f32;
    let next_y_offset = (1.0 - t) * dy as f32;
    let wrote = std::sync::atomic::AtomicBool::new(false);
    let stride = output.stride();
    output
        .data
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, dest)| {
            let y = y as i32;
            for x in 0..width {
                let ax = x as f32 - previous_x_offset;
                let ay = y as f32 - previous_y_offset;
                let cx = x as f32 + next_x_offset;
                let cy = y as f32 + next_y_offset;
                let previous_coverage = mask_coverage(&previous_lines, width, height, ax, ay);
                let next_coverage = mask_coverage(&next_lines, width, height, cx, cy);
                let weight = previous_coverage.max(next_coverage);
                if weight <= 0.001 {
                    continue;
                }
                let Some(a) = bilinear(previous, ax, ay) else {
                    continue;
                };
                let Some(c) = bilinear(next, cx, cy) else {
                    continue;
                };
                let source_b = a.0 * (1.0 - t) + c.0 * t;
                let source_g = a.1 * (1.0 - t) + c.1 * t;
                let source_r = a.2 * (1.0 - t) + c.2 * t;
                let keep = 1.0 - weight;
                let pixel = (x as usize) * 4;
                dest[pixel] = byte_round(keep * dest[pixel] as f32 + weight * source_b);
                dest[pixel + 1] = byte_round(keep * dest[pixel + 1] as f32 + weight * source_g);
                dest[pixel + 2] = byte_round(keep * dest[pixel + 2] as f32 + weight * source_r);
                wrote.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            let _ = rgba;
        });
    if wrote.load(std::sync::atomic::Ordering::Relaxed) {
        Some(output)
    } else {
        None
    }
}

fn expanded_line_mask(lines: &[u8], width: i32, height: i32) -> Vec<u8> {
    let mut expanded = vec![0u8; (width * height) as usize];
    let w = width as usize;
    for y in 0..height {
        for x in 0..width {
            let mut coverage = 0u8;
            for dy in -1..=1 {
                let yy = y + dy;
                if yy < 0 || yy >= height {
                    continue;
                }
                for dx in -1..=1 {
                    let xx = x + dx;
                    if xx < 0 || xx >= width || lines[(yy as usize) * w + xx as usize] == 0 {
                        continue;
                    }
                    let candidate: u8 = if dx == 0 && dy == 0 {
                        255
                    } else if dx == 0 || dy == 0 {
                        128
                    } else {
                        64
                    };
                    coverage = coverage.max(candidate);
                }
            }
            expanded[(y as usize) * w + x as usize] = coverage;
        }
    }
    expanded
}

fn mask_coverage(mask: &[u8], width: i32, height: i32, x: f32, y: f32) -> f32 {
    if x < 0.0 || y < 0.0 || x > (width - 1) as f32 || y > (height - 1) as f32 {
        return 0.0;
    }
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = (width - 1).min(x0 + 1);
    let y1 = (height - 1).min(y0 + 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let w = width as usize;
    let top = (mask[(y0 as usize) * w + x0 as usize] as f32 * (1.0 - tx)
        + mask[(y0 as usize) * w + x1 as usize] as f32 * tx)
        / 255.0;
    let bottom = (mask[(y1 as usize) * w + x0 as usize] as f32 * (1.0 - tx)
        + mask[(y1 as usize) * w + x1 as usize] as f32 * tx)
        / 255.0;
    top * (1.0 - ty) + bottom * ty
}

fn bilinear(pixels: &Frame, x: f32, y: f32) -> Option<(f32, f32, f32)> {
    let snapped_x = if (x - x.round()).abs() < 0.08 {
        x.round()
    } else {
        x
    };
    let snapped_y = if (y - y.round()).abs() < 0.08 {
        y.round()
    } else {
        y
    };
    let x0 = snapped_x.floor() as i32;
    let y0 = snapped_y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    if x0 < 0 || y0 < 0 || x1 >= pixels.width || y1 >= pixels.height {
        if x0 >= 0 && y0 >= 0 && x0 < pixels.width && y0 < pixels.height {
            return Some(pixel(pixels, x0, y0));
        }
        return None;
    }
    let tx = snapped_x - x0 as f32;
    let ty = snapped_y - y0 as f32;
    let p00 = pixel(pixels, x0, y0);
    let p10 = pixel(pixels, x1, y0);
    let p01 = pixel(pixels, x0, y1);
    let p11 = pixel(pixels, x1, y1);
    let mix = |a: f32, b: f32, c: f32, d: f32| {
        let top = a + (b - a) * tx;
        let bottom = c + (d - c) * tx;
        top + (bottom - top) * ty
    };
    Some((
        mix(p00.0, p10.0, p01.0, p11.0),
        mix(p00.1, p10.1, p01.1, p11.1),
        mix(p00.2, p10.2, p01.2, p11.2),
    ))
}

fn pixel(frame: &Frame, x: i32, y: i32) -> (f32, f32, f32) {
    let p = &frame.row(y)[(x as usize) * 4..];
    (p[0] as f32, p[1] as f32, p[2] as f32)
}

/// Thin-line mask used by the stabilizer and the stroke-width check.
/// A pixel counts when it is darker than 0.55 and darker than its 5×5 mean
/// by `contrast`.
pub fn line_mask(luma: &[f32], width: i32, height: i32, contrast: f32) -> Vec<u8> {
    let blur = box_mean(luma, width, height, 2);
    let mut mask = vec![0u8; luma.len()];
    if width <= 4 || height <= 4 {
        return mask;
    }
    for y in 2..(height - 2) {
        for x in 2..(width - 2) {
            let index = (y * width + x) as usize;
            let value = luma[index];
            if value < 0.55 && blur[index] - value >= contrast {
                mask[index] = 1;
            }
        }
    }
    mask
}

pub fn mean_stroke_width(lines: &[u8], width: i32, height: i32) -> f32 {
    let w = width as usize;
    let mut horizontal = 0i32;
    let mut horizontal_runs = 0i32;
    for y in 0..height {
        let mut run = 0i32;
        for x in 0..width {
            if lines[(y as usize) * w + x as usize] != 0 {
                run += 1;
            } else if run > 0 {
                horizontal += run;
                horizontal_runs += 1;
                run = 0;
            }
        }
        if run > 0 {
            horizontal += run;
            horizontal_runs += 1;
        }
    }
    let mut vertical = 0i32;
    let mut vertical_runs = 0i32;
    for x in 0..width {
        let mut run = 0i32;
        for y in 0..height {
            if lines[(y as usize) * w + x as usize] != 0 {
                run += 1;
            } else if run > 0 {
                vertical += run;
                vertical_runs += 1;
                run = 0;
            }
        }
        if run > 0 {
            vertical += run;
            vertical_runs += 1;
        }
    }
    let horizontal_mean = if horizontal_runs > 0 {
        horizontal as f32 / horizontal_runs as f32
    } else {
        0.0
    };
    let vertical_mean = if vertical_runs > 0 {
        vertical as f32 / vertical_runs as f32
    } else {
        0.0
    };
    if horizontal_mean == 0.0 {
        vertical_mean
    } else if vertical_mean == 0.0 {
        horizontal_mean
    } else {
        horizontal_mean.min(vertical_mean)
    }
}

pub fn lines_contained(source: &[u8], mid: &[u8]) -> bool {
    let mut hits = 0i32;
    let mut inside = 0i32;
    for index in 0..source.len() {
        if source[index] != 0 {
            hits += 1;
            if mid[index] != 0 {
                inside += 1;
            }
        }
    }
    hits >= 8 && (inside as f64) / (hits as f64) >= 0.6
}
