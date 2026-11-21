//! Tightly packed 8-bit frames. BGRA matches `kCVPixelFormatType_32BGRA`.

#[derive(Clone)]
pub struct Frame {
    pub width: i32,
    pub height: i32,
    /// `false` is BGRA (the iOS pipeline). `true` is RGBA.
    pub rgba: bool,
    pub data: Vec<u8>,
}

impl Frame {
    pub fn new(width: i32, height: i32, rgba: bool) -> Self {
        let n = (width as usize) * (height as usize) * 4;
        Self {
            width,
            height,
            rgba,
            data: vec![0; n],
        }
    }

    pub fn stride(&self) -> usize {
        self.width as usize * 4
    }

    pub fn gray(&self, x: i32, y: i32) -> u8 {
        self.data[(y as usize) * self.stride() + (x as usize) * 4]
    }

    pub fn set_rgb(&mut self, x: i32, y: i32, value: u8) {
        let i = (y as usize) * self.stride() + (x as usize) * 4;
        self.data[i] = value;
        self.data[i + 1] = value;
        self.data[i + 2] = value;
        self.data[i + 3] = 255;
    }

    pub fn row(&self, y: i32) -> &[u8] {
        let start = (y as usize) * self.stride();
        &self.data[start..start + self.stride()]
    }

    pub fn row_mut(&mut self, y: i32) -> &mut [u8] {
        let start = (y as usize) * self.stride();
        let stride = self.stride();
        &mut self.data[start..start + stride]
    }
}

#[derive(Clone, Copy)]
pub struct Rgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub y: f32,
}

pub fn sample_rgb(px: &[u8], rgba: bool) -> Rgb {
    let c0 = px[0] as f32 / 255.0;
    let c1 = px[1] as f32 / 255.0;
    let c2 = px[2] as f32 / 255.0;
    let (r, b) = if rgba { (c0, c2) } else { (c2, c0) };
    Rgb {
        r,
        g: c1,
        b,
        y: 0.2126 * r + 0.7152 * c1 + 0.0722 * b,
    }
}

pub fn luma_at(row: &[u8], x: i32, rgba: bool) -> f32 {
    sample_rgb(&row[(x as usize) * 4..], rgba).y
}

/// `UInt8(min(255, max(0, value + 0.5)))` — truncate after adding one half.
pub fn byte_floor(value: f32) -> u8 {
    let x = value + 0.5;
    if !x.is_finite() || x <= 0.0 {
        0
    } else if x >= 255.0 {
        255
    } else {
        x as u8
    }
}

/// Swift `Float.rounded()` (half away from zero), then clamped to a byte.
pub fn byte_round(value: f32) -> u8 {
    let x = value.round();
    if !x.is_finite() || x <= 0.0 {
        0
    } else if x >= 255.0 {
        255
    } else {
        x as u8
    }
}

pub fn clamp_sensitivity(sensitivity: f64) -> f64 {
    let value = if sensitivity.is_finite() {
        sensitivity
    } else {
        1.35
    };
    value.clamp(0.65, 1.35)
}

/// Integral-image box mean. `radius` 2 is the 5×5 window used by the guards.
pub fn box_mean(values: &[f32], width: i32, height: i32, radius: i32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let stride = w + 1;
    let mut integral = vec![0.0f32; stride * (h + 1)];
    for y in 0..h {
        let mut row_sum = 0.0f32;
        let current = (y + 1) * stride;
        let above = y * stride;
        let source = y * w;
        for x in 0..w {
            row_sum += values[source + x];
            integral[current + x + 1] = integral[above + x + 1] + row_sum;
        }
    }
    let mut mean = vec![0.0f32; w * h];
    for y in 0..height {
        let y0 = (y - radius).max(0);
        let y1 = (y + radius).min(height - 1);
        for x in 0..width {
            let x0 = (x - radius).max(0);
            let x1 = (x + radius).min(width - 1);
            let sum = integral[(y1 as usize + 1) * stride + (x1 as usize + 1)]
                - integral[y0 as usize * stride + (x1 as usize + 1)]
                - integral[(y1 as usize + 1) * stride + x0 as usize]
                + integral[y0 as usize * stride + x0 as usize];
            let area = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f32;
            mean[(y as usize) * w + x as usize] = sum / area;
        }
    }
    mean
}

pub fn dilate(mask: &mut [u8], columns: i32, rows: i32, radius: i32) {
    let mut dilated = mask.to_vec();
    let cols = columns as usize;
    for y in 0..rows {
        for x in 0..columns {
            if mask[(y as usize) * cols + x as usize] == 0 {
                continue;
            }
            for dy in -radius..=radius {
                let yy = y + dy;
                if yy < 0 || yy >= rows {
                    continue;
                }
                for dx in -radius..=radius {
                    let xx = x + dx;
                    if xx < 0 || xx >= columns {
                        continue;
                    }
                    dilated[(yy as usize) * cols + xx as usize] = 1;
                }
            }
        }
    }
    mask.copy_from_slice(&dilated);
}

pub fn luma_plane_parallel(frame: &Frame) -> Vec<f32> {
    use rayon::prelude::*;
    let w = frame.width as usize;
    let rgba = frame.rgba;
    let stride = frame.stride();
    let mut plane = vec![0.0f32; w * frame.height as usize];
    plane.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let src = &frame.data[y * stride..y * stride + stride];
        for x in 0..w {
            row[x] = luma_at(src, x as i32, rgba);
        }
    });
    plane
}
