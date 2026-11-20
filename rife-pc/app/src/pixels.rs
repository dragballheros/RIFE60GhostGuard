//! Shared BGRA helpers. Frames stay packed the way ffmpeg and the iOS pipeline do.

use rife_guards::Frame;

pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < 1e-8 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn idx(frame: &Frame, x: i32, y: i32) -> usize {
    (y as usize) * frame.stride() + (x as usize) * 4
}

pub fn rgb_at(frame: &Frame, x: i32, y: i32) -> [u8; 3] {
    let i = idx(frame, x, y);
    let p = &frame.data[i..i + 3];
    if frame.rgba {
        [p[0], p[1], p[2]]
    } else {
        [p[2], p[1], p[0]]
    }
}

pub fn put_rgb(frame: &mut Frame, x: i32, y: i32, rgb: [u8; 3]) {
    let i = idx(frame, x, y);
    if frame.rgba {
        frame.data[i] = rgb[0];
        frame.data[i + 1] = rgb[1];
        frame.data[i + 2] = rgb[2];
    } else {
        frame.data[i] = rgb[2];
        frame.data[i + 1] = rgb[1];
        frame.data[i + 2] = rgb[0];
    }
    frame.data[i + 3] = 255;
}

pub fn luma8(rgb: [u8; 3]) -> u8 {
    let y = (77 * rgb[0] as i32 + 150 * rgb[1] as i32 + 29 * rgb[2] as i32) >> 8;
    y.clamp(0, 255) as u8
}

pub fn ycbcr(rgb: [u8; 3]) -> (u8, i16, i16) {
    let r = rgb[0] as i32;
    let g = rgb[1] as i32;
    let b = rgb[2] as i32;
    let y = ((77 * r + 150 * g + 29 * b) >> 8).clamp(0, 255) as u8;
    let cb = (((-43 * r - 85 * g + 128 * b) >> 8) + 128) as i16;
    let cr = (((128 * r - 107 * g - 21 * b) >> 8) + 128) as i16;
    (y, cb, cr)
}

pub fn ycbcr_to_rgb(y: i32, cb: i32, cr: i32) -> [u8; 3] {
    let cbb = cb - 128;
    let crr = cr - 128;
    let r = y + ((359 * crr) >> 8);
    let g = y - ((88 * cbb + 183 * crr) >> 8);
    let b = y + ((454 * cbb) >> 8);
    [r.clamp(0, 255) as u8, g.clamp(0, 255) as u8, b.clamp(0, 255) as u8]
}

/// Separable gaussian blur. `radius` is the kernel half-width in pixels.
pub fn blur_plane(src: &[f32], width: i32, height: i32, radius: i32) -> Vec<f32> {
    let radius = radius.max(1);
    let sigma = (radius as f32) * 0.5;
    let mut kernel = Vec::new();
    let mut sum = 0.0f32;
    for i in -radius..=radius {
        let w = (-(i as f32) * (i as f32) / (2.0 * sigma * sigma)).exp();
        kernel.push(w);
        sum += w;
    }
    for w in &mut kernel {
        *w /= sum;
    }
    let w = width as usize;
    let h = height as usize;
    let mut tmp = vec![0.0f32; src.len()];
    let mut out = vec![0.0f32; src.len()];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, weight) in kernel.iter().enumerate() {
                let xx = (x as i32 + k as i32 - radius).clamp(0, width - 1) as usize;
                acc += src[y * w + xx] * weight;
            }
            tmp[y * w + x] = acc;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, weight) in kernel.iter().enumerate() {
                let yy = (y as i32 + k as i32 - radius).clamp(0, height - 1) as usize;
                acc += tmp[yy * w + x] * weight;
            }
            out[y * w + x] = acc;
        }
    }
    out
}

pub fn scale_nearest(frame: &Frame, width: i32, height: i32) -> Frame {
    let mut out = Frame::new(width.max(1), height.max(1), frame.rgba);
    for y in 0..out.height {
        let sy = ((y as i64 * frame.height as i64) / out.height as i64).min(frame.height as i64 - 1);
        for x in 0..out.width {
            let sx = ((x as i64 * frame.width as i64) / out.width as i64).min(frame.width as i64 - 1);
            let rgb = rgb_at(frame, sx as i32, sy as i32);
            put_rgb(&mut out, x, y, rgb);
        }
    }
    out
}

/// Area-average downscale used when a 2× tile is returned to source resolution.
pub fn downscale_box(frame: &Frame, width: i32, height: i32) -> Frame {
    if width >= frame.width && height >= frame.height {
        return scale_nearest(frame, width, height);
    }
    let mut out = Frame::new(width.max(1), height.max(1), frame.rgba);
    for y in 0..out.height {
        let y0 = (y as f32 / out.height as f32 * frame.height as f32) as i32;
        let y1 = (((y + 1) as f32 / out.height as f32) * frame.height as f32).ceil() as i32;
        let y1 = y1.clamp(y0 + 1, frame.height);
        for x in 0..out.width {
            let x0 = (x as f32 / out.width as f32 * frame.width as f32) as i32;
            let x1 = (((x + 1) as f32 / out.width as f32) * frame.width as f32).ceil() as i32;
            let x1 = x1.clamp(x0 + 1, frame.width);
            let mut acc = [0u32; 3];
            let mut n = 0u32;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    let rgb = rgb_at(frame, xx, yy);
                    acc[0] += rgb[0] as u32;
                    acc[1] += rgb[1] as u32;
                    acc[2] += rgb[2] as u32;
                    n += 1;
                }
            }
            let n = n.max(1);
            put_rgb(
                &mut out,
                x,
                y,
                [(acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8],
            );
        }
    }
    out
}
