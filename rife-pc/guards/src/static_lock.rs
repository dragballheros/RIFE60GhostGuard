//! Pins background that did not move between the two sources.

use rayon::prelude::*;

use crate::frame::{box_mean, byte_floor, sample_rgb, Frame};
use crate::structural;

pub struct StaticRegionLock {
    pub sensitivity: f64,
}

impl StaticRegionLock {
    pub fn new(sensitivity: f64) -> Self {
        Self {
            sensitivity: crate::frame::clamp_sensitivity(sensitivity),
        }
    }

    pub fn lock(
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
        if let Some(cut) = structural::assess(previous, next, self.sensitivity, Some(interpolated))
        {
            if cut.is_cut() {
                return None;
            }
        }
        let rgba = previous.rgba;
        let count = (width * height) as usize;
        let mut metric = vec![0.0f32; count];
        let mut disagreement = vec![0.0f32; count];
        let w = width as usize;
        metric
            .par_chunks_mut(w)
            .zip(disagreement.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, (metric_row, disagree_row))| {
                let y = y as i32;
                let prev = previous.row(y);
                let nxt = next.row(y);
                let mid = interpolated.row(y);
                for x in 0..w {
                    let pixel = x * 4;
                    let a = sample_rgb(&prev[pixel..], rgba);
                    let b = sample_rgb(&nxt[pixel..], rgba);
                    let m = sample_rgb(&mid[pixel..], rgba);
                    let color = ((a.r - b.r).abs() + (a.g - b.g).abs() + (a.b - b.b).abs()) / 3.0;
                    metric_row[x] = (a.y - b.y).abs().max(color);
                    let mid_color =
                        ((m.r - a.r).abs() + (m.g - a.g).abs() + (m.b - a.b).abs()) / 3.0;
                    disagree_row[x] = if (m.y - a.y).abs().max(mid_color) > 0.15 {
                        1.0
                    } else {
                        0.0
                    };
                }
            });
        let pixel_limit = (0.034 - 0.006 * self.sensitivity) as f32;
        let neighborhood_limit = (0.026 - 0.004 * self.sensitivity) as f32;
        let local_agreement = box_mean(&metric, width, height, 2);
        let local_change = box_mean(&disagreement, width, height, 2);
        let mut mask = vec![0u8; count];
        let mut locked = 0i32;
        for index in 0..count {
            if metric[index] < pixel_limit
                && local_agreement[index] < neighborhood_limit
                && local_change[index] < 0.72
            {
                mask[index] = 1;
                locked += 1;
            }
        }
        if locked == 0 {
            return None;
        }
        let coverage = box_mean(
            &mask.iter().map(|v| *v as f32).collect::<Vec<_>>(),
            width,
            height,
            2,
        );
        let mut output = interpolated.clone();
        let t = if progress.is_finite() {
            progress.clamp(0.0, 1.0) as f32
        } else {
            0.5
        };
        let stride = width as usize * 4;
        output
            .data
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(y, dest)| {
                let prev = previous.row(y as i32);
                let nxt = next.row(y as i32);
                for x in 0..width as usize {
                    let index = y * width as usize + x;
                    if mask[index] == 0 {
                        continue;
                    }
                    let weight: f32 = if coverage[index] > 0.99 { 1.0 } else { 0.45 };
                    let keep = 1.0 - weight;
                    let pixel = x * 4;
                    for channel in 0..3 {
                        let source = prev[pixel + channel] as f32 * (1.0 - t)
                            + nxt[pixel + channel] as f32 * t;
                        dest[pixel + channel] =
                            byte_floor(keep * dest[pixel + channel] as f32 + weight * source);
                    }
                }
            });
        Some(output)
    }
}
