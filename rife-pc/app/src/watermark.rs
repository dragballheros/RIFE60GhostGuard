//! Watermark mask raster and DeepCreamPy's input convention.
//! LaMa and DeepCream run through ONNX when those files are installed.
//! Otherwise a neighborhood fill covers the marked pixels and the log says so.
//! DeepCreamPy's checkpoint terms are not stated; the source project is
//! CC BY-NC-SA 4.0, so the UI treats it as non-commercial.

use anyhow::Result;
use rife_guards::Frame;

use crate::pixels::{put_rgb, rgb_at, scale_nearest};
use crate::settings::{MaskSpec, WatermarkModelKind};

pub fn rasterize(width: i32, height: i32, mask: &MaskSpec, padding: f32) -> Vec<u8> {
    let mut out = vec![0u8; width as usize * height as usize];
    let pad = padding.round().max(0.0) as i32;
    for rect in &mask.rects {
        stamp_rect(&mut out, width, height, rect[0], rect[1], rect[2], rect[3], pad);
    }
    for stroke in &mask.strokes {
        if stroke.len() < 2 {
            continue;
        }
        for window in stroke.windows(2) {
            stamp_segment(&mut out, width, height, window[0], window[1], 6 + pad);
        }
    }
    out
}

fn stamp_rect(mask: &mut [u8], width: i32, height: i32, x: f32, y: f32, w: f32, h: f32, pad: i32) {
    let x0 = (x * width as f32).floor() as i32 - pad;
    let y0 = (y * height as f32).floor() as i32 - pad;
    let x1 = ((x + w) * width as f32).ceil() as i32 + pad;
    let y1 = ((y + h) * height as f32).ceil() as i32 + pad;
    fill(mask, width, height, x0, y0, x1, y1);
}

fn stamp_segment(mask: &mut [u8], width: i32, height: i32, a: [f32; 2], b: [f32; 2], radius: i32) {
    let steps = 24;
    for step in 0..=steps {
        let t = step as f32 / steps as f32;
        let x = ((a[0] + (b[0] - a[0]) * t) * width as f32).round() as i32;
        let y = ((a[1] + (b[1] - a[1]) * t) * height as f32).round() as i32;
        fill(mask, width, height, x - radius, y - radius, x + radius, y + radius);
    }
}

fn fill(mask: &mut [u8], width: i32, height: i32, x0: i32, y0: i32, x1: i32, y1: i32) {
    for y in y0.max(0)..y1.min(height) {
        for x in x0.max(0)..x1.min(width) {
            mask[y as usize * width as usize + x as usize] = 1;
        }
    }
}

pub struct DeepCreamTile {
    pub side: i32,
    pub origin_x: i32,
    pub origin_y: i32,
    pub crop_w: i32,
    pub crop_h: i32,
    /// NHWC float in [-1, 1], batch 1. The ONNX generators are batch 1.
    /// The iOS Core ML packages were frozen at batch 8 and repeated one tile.
    pub masked_input: Vec<f32>,
    pub known_mask: Vec<f32>,
    pub source: Vec<f32>,
}

pub fn context_crop_side(target_w: f32, target_h: f32, frame_w: i32, frame_h: i32) -> i32 {
    let target_side = target_w.max(target_h);
    let preferred = 256.0f32.max((target_side * 1.5).ceil());
    let frame_limit = frame_w.min(frame_h) as f32;
    preferred.min(frame_limit).max(target_side).ceil() as i32
}

pub fn masked_input_pixel(source: f32, known: f32, model: WatermarkModelKind) -> f32 {
    if model == WatermarkModelKind::DeepCreamMosaic {
        return source;
    }
    if known == 0.0 {
        -1.0
    } else {
        source
    }
}

/// Build the three DeepCream tensors for one marked rectangle.
/// Mask convention matches DeepCreamPy: 1 keeps the pixel, 0 is inpainted.
pub fn deepcream_tile(
    frame: &Frame,
    mask: &[u8],
    model: WatermarkModelKind,
) -> Option<DeepCreamTile> {
    let bounds = mask_bounds(mask, frame.width, frame.height)?;
    let side = context_crop_side(
        (bounds.2 - bounds.0) as f32,
        (bounds.3 - bounds.1) as f32,
        frame.width,
        frame.height,
    )
    .max(1);
    let cx = (bounds.0 + bounds.2) / 2;
    let cy = (bounds.1 + bounds.3) / 2;
    let x0 = (cx - side / 2).clamp(0, (frame.width - side).max(0));
    let y0 = (cy - side / 2).clamp(0, (frame.height - side).max(0));
    let crop_w = side.min(frame.width - x0);
    let crop_h = side.min(frame.height - y0);
    let mut crop = Frame::new(crop_w, crop_h, frame.rgba);
    let mut crop_mask = vec![0u8; crop_w as usize * crop_h as usize];
    for y in 0..crop_h {
        for x in 0..crop_w {
            put_rgb(&mut crop, x, y, rgb_at(frame, x0 + x, y0 + y));
            crop_mask[y as usize * crop_w as usize + x as usize] =
                mask[(y0 + y) as usize * frame.width as usize + (x0 + x) as usize];
        }
    }
    let scaled = scale_nearest(&crop, 256, 256);
    let scaled_mask = scale_mask(&crop_mask, crop_w, crop_h, 256, 256);
    let spatial = 256 * 256;
    let mut source = vec![0.0f32; spatial * 3];
    let mut known = vec![0.0f32; spatial * 3];
    let mut masked = vec![0.0f32; spatial * 3];
    for y in 0..256 {
        for x in 0..256 {
            let rgb = rgb_at(&scaled, x, y);
            let i = y as usize * 256 + x as usize;
            let preserve = if scaled_mask[i] == 0 { 1.0 } else { 0.0 };
            for c in 0..3 {
                let value = rgb[c] as f32 / 255.0 * 2.0 - 1.0;
                source[i * 3 + c] = value;
                known[i * 3 + c] = preserve;
                masked[i * 3 + c] = masked_input_pixel(value, preserve, model);
            }
        }
    }
    Some(DeepCreamTile {
        side,
        origin_x: x0,
        origin_y: y0,
        crop_w,
        crop_h,
        masked_input: masked,
        known_mask: known,
        source,
    })
}

fn mask_bounds(mask: &[u8], width: i32, height: i32) -> Option<(i32, i32, i32, i32)> {
    let mut x0 = width;
    let mut y0 = height;
    let mut x1 = 0;
    let mut y1 = 0;
    let mut any = false;
    for y in 0..height {
        for x in 0..width {
            if mask[y as usize * width as usize + x as usize] != 0 {
                any = true;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    if any { Some((x0, y0, x1, y1)) } else { None }
}

fn scale_mask(mask: &[u8], sw: i32, sh: i32, dw: i32, dh: i32) -> Vec<u8> {
    let mut out = vec![0u8; dw as usize * dh as usize];
    for y in 0..dh {
        let sy = ((y as i64 * sh as i64) / dh as i64).min(sh as i64 - 1) as i32;
        for x in 0..dw {
            let sx = ((x as i64 * sw as i64) / dw as i64).min(sw as i64 - 1) as i32;
            out[y as usize * dw as usize + x as usize] = mask[sy as usize * sw as usize + sx as usize];
        }
    }
    out
}

pub struct Removal {
    pub frame: Frame,
    pub used_network: bool,
}

pub fn remove(frame: &Frame, mask: &MaskSpec, padding: f32, model: WatermarkModelKind) -> Result<Removal> {
    if mask.is_empty() {
        return Ok(Removal {
            frame: frame.clone(),
            used_network: false,
        });
    }
    let raster = rasterize(frame.width, frame.height, mask, padding);
    let name = format!("{}.onnx", model.file_stem());
    if let Ok(path) = crate::models::ensure_onnx(&name) {
        if let Ok(frame) = inpaint_network(frame, &raster, model, &path) {
            return Ok(Removal {
                frame,
                used_network: true,
            });
        }
    }
    Ok(Removal {
        frame: classical_fill(frame, &raster),
        used_network: false,
    })
}

fn inpaint_network(frame: &Frame, raster: &[u8], model: WatermarkModelKind, path: &std::path::Path) -> Result<Frame> {
    match model {
        WatermarkModelKind::Lama => lama(frame, raster, path),
        WatermarkModelKind::DeepCreamBar | WatermarkModelKind::DeepCreamMosaic => {
            deepcream(frame, raster, model, path)
        }
    }
}

fn lama(frame: &Frame, raster: &[u8], path: &std::path::Path) -> Result<Frame> {
    let session = crate::onnx_runtime::Session::open(path)?;
    let bounds = mask_bounds(raster, frame.width, frame.height).ok_or_else(|| anyhow_empty())?;
    let side = (bounds.2 - bounds.0).max(bounds.3 - bounds.1).max(256);
    let side = side.min(frame.width.min(frame.height));
    let cx = (bounds.0 + bounds.2) / 2;
    let cy = (bounds.1 + bounds.3) / 2;
    let x0 = (cx - side / 2).clamp(0, (frame.width - side).max(0));
    let y0 = (cy - side / 2).clamp(0, (frame.height - side).max(0));
    let mut crop = Frame::new(side, side, frame.rgba);
    let mut crop_mask = vec![0u8; side as usize * side as usize];
    for y in 0..side {
        for x in 0..side {
            let sx = (x0 + x).min(frame.width - 1);
            let sy = (y0 + y).min(frame.height - 1);
            put_rgb(&mut crop, x, y, rgb_at(frame, sx, sy));
            crop_mask[y as usize * side as usize + x as usize] =
                raster[sy as usize * frame.width as usize + sx as usize];
        }
    }
    let image = scale_nearest(&crop, 512, 512);
    let mask512 = scale_mask(&crop_mask, side, side, 512, 512);
    let mut image_nchw = vec![0.0f32; 3 * 512 * 512];
    let mut mask_nchw = vec![0.0f32; 512 * 512];
    for y in 0..512 {
        for x in 0..512 {
            let rgb = rgb_at(&image, x, y);
            let i = y as usize * 512 + x as usize;
            image_nchw[i] = rgb[0] as f32 / 255.0;
            image_nchw[512 * 512 + i] = rgb[1] as f32 / 255.0;
            image_nchw[2 * 512 * 512 + i] = rgb[2] as f32 / 255.0;
            mask_nchw[i] = if mask512[i] == 0 { 0.0 } else { 1.0 };
        }
    }
    let image_shape = [1i64, 3, 512, 512];
    let mask_shape = [1i64, 1, 512, 512];
    let mut output = vec![0.0f32; image_nchw.len()];
    session.run_named(
        &[("image", &image_nchw, &image_shape), ("mask", &mask_nchw, &mask_shape)],
        &[("output", &mut output)],
    )?;
    let patch = rgb_from_nchw(&output, 512, 512);
    let back = scale_nearest(&patch, side, side);
    composite(frame, &back, &crop_mask, x0, y0)
}

fn deepcream(frame: &Frame, raster: &[u8], model: WatermarkModelKind, path: &std::path::Path) -> Result<Frame> {
    let tile = deepcream_tile(frame, raster, model).ok_or_else(anyhow_empty)?;
    let session = crate::onnx_runtime::Session::open(path)?;
    let shape = [1i64, 256, 256, 3];
    let mut output = vec![0.0f32; tile.masked_input.len()];
    // iOS / DeepCreamPy order: masked crop, original crop, known-pixel mask.
    session.run_named(
        &[
            ("Placeholder:0", &tile.masked_input, &shape),
            ("Placeholder_1:0", &tile.source, &shape),
            ("Placeholder_2:0", &tile.known_mask, &shape),
        ],
        &[("add:0", &mut output)],
    )?;
    let mut patch = Frame::new(256, 256, false);
    for i in 0..(256 * 256) {
        let base = i * 3;
        let decode = |v: f32| ((v + 1.0) * 0.5 * 255.0).round().clamp(0.0, 255.0) as u8;
        let x = (i % 256) as i32;
        let y = (i / 256) as i32;
        put_rgb(
            &mut patch,
            x,
            y,
            [decode(output[base]), decode(output[base + 1]), decode(output[base + 2])],
        );
    }
    let back = scale_nearest(&patch, tile.crop_w.max(1), tile.crop_h.max(1));
    let mut local = vec![0u8; tile.crop_w as usize * tile.crop_h as usize];
    for y in 0..tile.crop_h {
        for x in 0..tile.crop_w {
            let sx = tile.origin_x + x;
            let sy = tile.origin_y + y;
            if sx < frame.width && sy < frame.height {
                local[y as usize * tile.crop_w as usize + x as usize] =
                    raster[sy as usize * frame.width as usize + sx as usize];
            }
        }
    }
    composite(frame, &back, &local, tile.origin_x, tile.origin_y)
}

fn rgb_from_nchw(data: &[f32], width: i32, height: i32) -> Frame {
    let mut frame = Frame::new(width, height, false);
    let plane = width as usize * height as usize;
    for y in 0..height {
        for x in 0..width {
            let i = y as usize * width as usize + x as usize;
            let sample = |offset: usize| {
                (data.get(offset + i).copied().unwrap_or(0.0) * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8
            };
            put_rgb(&mut frame, x, y, [sample(0), sample(plane), sample(2 * plane)]);
        }
    }
    frame
}

fn composite(frame: &Frame, patch: &Frame, mask: &[u8], x0: i32, y0: i32) -> Result<Frame> {
    let mut out = frame.clone();
    for y in 0..patch.height {
        for x in 0..patch.width {
            let sx = x0 + x;
            let sy = y0 + y;
            if sx < 0 || sy < 0 || sx >= frame.width || sy >= frame.height {
                continue;
            }
            if mask[y as usize * patch.width as usize + x as usize] == 0 {
                continue;
            }
            put_rgb(&mut out, sx, sy, rgb_at(patch, x, y));
        }
    }
    Ok(out)
}

fn classical_fill(frame: &Frame, raster: &[u8]) -> Frame {
    let mut out = frame.clone();
    for y in 0..frame.height {
        for x in 0..frame.width {
            let i = y as usize * frame.width as usize + x as usize;
            if raster[i] == 0 {
                continue;
            }
            let mut acc = [0u32; 3];
            let mut n = 0u32;
            for dy in -6..=6 {
                for dx in -6..=6 {
                    let sx = x + dx;
                    let sy = y + dy;
                    if sx < 0 || sy < 0 || sx >= frame.width || sy >= frame.height {
                        continue;
                    }
                    let j = sy as usize * frame.width as usize + sx as usize;
                    if raster[j] != 0 {
                        continue;
                    }
                    let rgb = rgb_at(frame, sx, sy);
                    acc[0] += rgb[0] as u32;
                    acc[1] += rgb[1] as u32;
                    acc[2] += rgb[2] as u32;
                    n += 1;
                }
            }
            if n > 0 {
                put_rgb(&mut out, x, y, [(acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8]);
            }
        }
    }
    out
}

fn anyhow_empty() -> anyhow::Error {
    anyhow::anyhow!("the watermark mask is empty")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bar_model_blanks_unknown_pixels_and_mosaic_keeps_them() {
        assert_eq!(masked_input_pixel(0.4, 0.0, WatermarkModelKind::DeepCreamBar), -1.0);
        assert_eq!(masked_input_pixel(0.4, 1.0, WatermarkModelKind::DeepCreamBar), 0.4);
        assert_eq!(masked_input_pixel(0.4, 0.0, WatermarkModelKind::DeepCreamMosaic), 0.4);
    }

    #[test]
    fn context_crop_is_at_least_256_and_stays_inside_the_frame() {
        assert_eq!(context_crop_side(100.0, 40.0, 1920, 1080), 256);
        assert_eq!(context_crop_side(400.0, 200.0, 1920, 1080), 600);
        assert_eq!(context_crop_side(400.0, 400.0, 320, 240), 400);
    }

    #[test]
    fn a_painted_rect_becomes_mask_pixels() {
        let spec = MaskSpec {
            strokes: Vec::new(),
            rects: vec![[0.25, 0.25, 0.5, 0.5]],
        };
        let mask = rasterize(20, 20, &spec, 0.0);
        assert_eq!(mask[0], 0);
        assert_eq!(mask[10 * 20 + 10], 1);
    }
}
