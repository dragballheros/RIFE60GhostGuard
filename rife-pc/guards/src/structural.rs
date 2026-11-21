//! Shot-boundary test for two frames that can share a palette.
//! Coarse color difference misses a dark-room or pink/beige cut, so this also
//! compares edge layout on a 64×36 luma grid.

#[derive(Clone, Debug)]
pub struct Assessment {
    pub appearance: f64,
    pub overlap: Option<f64>,
    pub blend: Option<f64>,
    /// Low edge overlap plus a modest color change.
    pub structural_cut: bool,
    /// `max(luma, RGB) > 0.20 / sensitivity`.
    pub classic_cut: bool,
    /// Mid frame sits on the A/C average while the shots do not share edges.
    pub crossfade_cut: bool,
}

impl Assessment {
    pub fn is_cut(&self) -> bool {
        self.classic_cut || self.structural_cut || self.crossfade_cut
    }
}

/// Dice overlap below this is a different layout. Higher sensitivity is stricter.
/// 0.22 at 0.65, 0.30 at the default 1.35.
pub fn overlap_limit(sensitivity: f64) -> f64 {
    let s = crate::frame::clamp_sensitivity(sensitivity);
    0.22 + 0.08 * (s - 0.65) / 0.70
}

/// Cuts with almost no appearance change are left to the existing rules.
/// 0.058 at 0.65, 0.050 at the default 1.35.
pub fn appearance_floor(sensitivity: f64) -> f64 {
    let s = crate::frame::clamp_sensitivity(sensitivity);
    0.058 - 0.008 * (s - 0.65) / 0.70
}

pub fn assess(
    previous: &crate::frame::Frame,
    next: &crate::frame::Frame,
    sensitivity: f64,
    mid: Option<&crate::frame::Frame>,
) -> Option<Assessment> {
    if previous.rgba != next.rgba {
        return None;
    }
    if let Some(mid) = mid {
        if mid.rgba != previous.rgba {
            return None;
        }
    }
    let mut width = previous.width.min(next.width);
    let mut height = previous.height.min(next.height);
    if let Some(mid) = mid {
        width = width.min(mid.width);
        height = height.min(mid.height);
    }
    if width < 8 || height < 8 {
        return None;
    }
    Some(assess_size(previous, next, width, height, sensitivity, mid))
}

fn assess_size(
    previous: &crate::frame::Frame,
    next: &crate::frame::Frame,
    width: i32,
    height: i32,
    sensitivity: f64,
    mid: Option<&crate::frame::Frame>,
) -> Assessment {
    let columns = 64.min(width);
    let rows = 36.min(height);
    let count = (columns * rows) as usize;
    let mut luma_a = vec![0.0f32; count];
    let mut luma_b = vec![0.0f32; count];
    let mut luma_sum = 0.0f32;
    let mut color_sum = 0.0f32;
    let mut blend_sum = 0.0f32;
    let rgba = previous.rgba;

    for row in 0..rows {
        let y = (height - 1).min((row * height + height / 2) / rows);
        let prev_line = previous.row(y);
        let next_line = next.row(y);
        let mid_line = mid.map(|m| m.row(y));
        for column in 0..columns {
            let x = (width - 1).min((column * width + width / 2) / columns);
            let pixel = (x as usize) * 4;
            let a = crate::frame::sample_rgb(&prev_line[pixel..], rgba);
            let b = crate::frame::sample_rgb(&next_line[pixel..], rgba);
            let index = (row * columns + column) as usize;
            luma_a[index] = a.y;
            luma_b[index] = b.y;
            luma_sum += (a.y - b.y).abs();
            color_sum += ((a.r - b.r).abs() + (a.g - b.g).abs() + (a.b - b.b).abs()) / 3.0;
            if let Some(mid_line) = mid_line {
                let m = crate::frame::sample_rgb(&mid_line[pixel..], rgba);
                blend_sum += (m.y - (a.y + b.y) * 0.5).abs();
            }
        }
    }

    let samples = (count.max(1)) as f32;
    let appearance = (luma_sum.max(color_sum) / samples) as f64;
    let blend = mid.map(|_| (blend_sum / samples) as f64);
    let tuned = crate::frame::clamp_sensitivity(sensitivity);
    let classic_cut = appearance > 0.20 / tuned;

    let edges = edge_masks(&luma_a, &luma_b, columns, rows);
    let overlap = if edges.fraction_a < 0.012 && edges.fraction_b < 0.012 {
        None
    } else if edges.edges_a == 0 || edges.edges_b == 0 {
        if edges.edges_a == 0 && edges.edges_b == 0 {
            None
        } else {
            Some(0.0)
        }
    } else {
        let recall_a = edges.covered_a as f64 / edges.edges_a as f64;
        let recall_b = edges.covered_b as f64 / edges.edges_b as f64;
        Some(0.5 * (recall_a + recall_b))
    };

    let floor = appearance_floor(tuned);
    let structural_cut = overlap
        .map(|value| value < overlap_limit(tuned) && appearance > floor)
        .unwrap_or(false);
    let crossfade_cut = if let (Some(overlap), Some(blend)) = (overlap, blend) {
        let crossfade_overlap = 0.42_f64.min(overlap_limit(tuned) + 0.08);
        appearance > floor
            && overlap < crossfade_overlap
            && blend < 0.018_f64.max(appearance * 0.28)
    } else {
        false
    };

    Assessment {
        appearance,
        overlap,
        blend,
        structural_cut,
        classic_cut,
        crossfade_cut,
    }
}

struct Edges {
    fraction_a: f64,
    fraction_b: f64,
    covered_a: i32,
    covered_b: i32,
    edges_a: i32,
    edges_b: i32,
}

fn edge_masks(luma_a: &[f32], luma_b: &[f32], columns: i32, rows: i32) -> Edges {
    let count = (columns * rows) as usize;
    let mut mask_a = vec![0u8; count];
    let mut mask_b = vec![0u8; count];
    let mut edges_a = 0i32;
    let mut edges_b = 0i32;
    if columns > 2 && rows > 2 {
        for y in 1..(rows - 1) {
            for x in 1..(columns - 1) {
                let index = (y * columns + x) as usize;
                let mag_a = (luma_a[index + 1] - luma_a[index - 1]).abs()
                    + (luma_a[index + columns as usize] - luma_a[index - columns as usize]).abs();
                let mag_b = (luma_b[index + 1] - luma_b[index - 1]).abs()
                    + (luma_b[index + columns as usize] - luma_b[index - columns as usize]).abs();
                if mag_a >= 0.12 {
                    mask_a[index] = 1;
                    edges_a += 1;
                }
                if mag_b >= 0.12 {
                    mask_b[index] = 1;
                    edges_b += 1;
                }
            }
        }
    }
    let raw_a = mask_a.clone();
    let raw_b = mask_b.clone();
    crate::frame::dilate(&mut mask_a, columns, rows, 1);
    crate::frame::dilate(&mut mask_b, columns, rows, 1);
    let mut covered_a = 0i32;
    let mut covered_b = 0i32;
    for index in 0..count {
        if raw_a[index] != 0 && mask_b[index] != 0 {
            covered_a += 1;
        }
        if raw_b[index] != 0 && mask_a[index] != 0 {
            covered_b += 1;
        }
    }
    Edges {
        fraction_a: edges_a as f64 / count.max(1) as f64,
        fraction_b: edges_b as f64 / count.max(1) as f64,
        covered_a,
        covered_b,
        edges_a,
        edges_b,
    }
}
