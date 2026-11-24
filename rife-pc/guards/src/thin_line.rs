//! Restores thin dark strokes that both real frames still share when an
//! in-between fades them out. Moving lines are left to RIFE.

use rayon::prelude::*;

use crate::frame::{box_mean, sample_rgb, Frame};

pub struct ThinLinePreservationPass {
    pub sensitivity: f64,
}

impl ThinLinePreservationPass {
    pub fn new(sensitivity: f64) -> Self {
        Self {
            sensitivity: crate::frame::clamp_sensitivity(sensitivity),
        }
    }

    pub fn preserve(
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
        if width < 8
            || height < 8
            || interpolated.width != width
            || next.width != width
            || interpolated.height != height
            || next.height != height
        {
            return None;
        }
        let rgba = previous.rgba;
        let t = if progress.is_finite() {
            progress.clamp(0.0, 1.0)
        } else {
            0.5
        };
        if discontinuity(previous, next, rgba) > 0.20 / self.sensitivity {
            return None;
        }
        let previous_luma = crate::frame::luma_plane_parallel(previous);
        let next_luma = crate::frame::luma_plane_parallel(next);
        let mid_luma = crate::frame::luma_plane_parallel(interpolated);
        let previous_blur = box_mean(&previous_luma, width, height, 2);
        let next_blur = box_mean(&next_luma, width, height, 2);
        let line_contrast = (0.12 / self.sensitivity).max(0.055) as f32;
        let previous_lines =
            line_mask(&previous_luma, &previous_blur, width, height, line_contrast);
        let next_lines = line_mask(&next_luma, &next_blur, width, height, line_contrast);
        let count = (width * height) as usize;
        let mut agreement = 0i32;
        for index in 0..count {
            if previous_lines[index] != 0 && next_lines[index] != 0 {
                agreement += 1;
            }
        }
        let fraction = agreement as f64 / count as f64;
        if agreement < 12 || fraction > 0.06 {
            return None;
        }
        let fade_gap = (0.10 / self.sensitivity).max(0.045) as f32;
        let prefer_previous = t <= 0.5;
        let mut hits = Vec::new();
        for y in 2..(height - 2) {
            let prev_row = previous.row(y);
            let next_row = next.row(y);
            for x in 2..(width - 2) {
                let index = (y * width + x) as usize;
                if previous_lines[index] == 0 || next_lines[index] == 0 {
                    continue;
                }
                let y_a = previous_luma[index];
                let y_b = next_luma[index];
                let y_m = mid_luma[index];
                let darker = y_a.min(y_b);
                let average = (y_a + y_b) * 0.5;
                if !(y_m > average + fade_gap && y_m > darker + fade_gap) {
                    continue;
                }
                let pixel = (x as usize) * 4;
                let near_y = if prefer_previous { y_a } else { y_b };
                let near = if prefer_previous { prev_row } else { next_row };
                let dark = if y_a <= y_b { prev_row } else { next_row };
                let source = if near_y <= darker + 0.04 { near } else { dark };
                hits.push(Hit {
                    x,
                    y,
                    c0: source[pixel],
                    c1: source[pixel + 1],
                    c2: source[pixel + 2],
                });
            }
        }
        if hits.is_empty() {
            return None;
        }
        let supported = connected_stroke_hits(&hits, width, height);
        if supported.is_empty() {
            return None;
        }
        let mut c0 = vec![0u8; count];
        let mut c1 = vec![0u8; count];
        let mut c2 = vec![0u8; count];
        let mut repair = vec![0u8; count];
        for hit in &supported {
            let index = (hit.y * width + hit.x) as usize;
            repair[index] = 1;
            c0[index] = hit.c0;
            c1[index] = hit.c1;
            c2[index] = hit.c2;
        }
        let mut output = interpolated.clone();
        let stride = output.stride();
        output
            .data
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(y, dest)| {
                let row_index = y * width as usize;
                for x in 0..width as usize {
                    let repaired = row_index + x;
                    if repair[repaired] == 0 {
                        continue;
                    }
                    let pixel = x * 4;
                    dest[pixel] = c0[repaired];
                    dest[pixel + 1] = c1[repaired];
                    dest[pixel + 2] = c2[repaired];
                }
            });
        Some(output)
    }
}

struct Hit {
    x: i32,
    y: i32,
    c0: u8,
    c1: u8,
    c2: u8,
}

fn connected_stroke_hits(hits: &[Hit], width: i32, height: i32) -> Vec<Hit> {
    let count = (width * height) as usize;
    let mut state = vec![0u8; count];
    for hit in hits {
        state[(hit.y * width + hit.x) as usize] = 1;
    }
    let mut queue = Vec::with_capacity(64);
    for hit in hits {
        let start = (hit.y * width + hit.x) as usize;
        if state[start] != 1 {
            continue;
        }
        queue.clear();
        queue.push(start);
        state[start] = 2;
        let mut cursor = 0usize;
        let mut min_x = hit.x;
        let mut max_x = hit.x;
        let mut min_y = hit.y;
        let mut max_y = hit.y;
        while cursor < queue.len() {
            let index = queue[cursor];
            cursor += 1;
            let x = (index as i32) % width;
            let y = (index as i32) / width;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            for dy in -1..=1 {
                let ny = y + dy;
                if ny < 0 || ny >= height {
                    continue;
                }
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x + dx;
                    if nx < 0 || nx >= width {
                        continue;
                    }
                    let neighbor = (ny * width + nx) as usize;
                    if state[neighbor] == 1 {
                        state[neighbor] = 2;
                        queue.push(neighbor);
                    }
                }
            }
        }
        let extent = (max_x - min_x).max(max_y - min_y);
        if queue.len() >= 3 && extent >= 2 {
            for index in &queue {
                state[*index] = 3;
            }
        }
    }
    hits.iter()
        .filter(|hit| state[(hit.y * width + hit.x) as usize] == 3)
        .cloned()
        .collect()
}

impl Clone for Hit {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for Hit {}

fn line_mask(luma: &[f32], blur: &[f32], width: i32, height: i32, contrast: f32) -> Vec<u8> {
    let mut mask = vec![0u8; luma.len()];
    if width <= 4 || height <= 4 {
        return mask;
    }
    for y in 2..(height - 2) {
        for x in 2..(width - 2) {
            let index = (y * width + x) as usize;
            let value = luma[index];
            if value < 0.58 && blur[index] - value >= contrast {
                mask[index] = 1;
            }
        }
    }
    mask
}

fn discontinuity(previous: &Frame, next: &Frame, rgba: bool) -> f64 {
    let width = previous.width.min(next.width);
    let height = previous.height.min(next.height);
    let step = 1.max(width.max(height) / 160);
    let mut luma_sum = 0.0f32;
    let mut color_sum = 0.0f32;
    let mut samples = 0.0f32;
    let mut y = 0;
    while y < height {
        let prev = previous.row(y);
        let nxt = next.row(y);
        let mut x = 0;
        while x < width {
            let pixel = (x as usize) * 4;
            let p = sample_rgb(&prev[pixel..], rgba);
            let n = sample_rgb(&nxt[pixel..], rgba);
            luma_sum += (p.y - n.y).abs();
            // Swift's discontinuity sums raw buffer channels, not RGB-ordered ones.
            let p0 = prev[pixel] as f32 / 255.0;
            let p1 = prev[pixel + 1] as f32 / 255.0;
            let p2 = prev[pixel + 2] as f32 / 255.0;
            let n0 = nxt[pixel] as f32 / 255.0;
            let n1 = nxt[pixel + 1] as f32 / 255.0;
            let n2 = nxt[pixel + 2] as f32 / 255.0;
            color_sum += ((p0 - n0).abs() + (p1 - n1).abs() + (p2 - n2).abs()) / 3.0;
            let _ = (p, n);
            samples += 1.0;
            x += step;
        }
        y += step;
    }
    if samples <= 0.0 {
        return 0.0;
    }
    (luma_sum.max(color_sum) / samples) as f64
}
