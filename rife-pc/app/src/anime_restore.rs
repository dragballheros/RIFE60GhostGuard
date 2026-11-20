//! Anime Restore. The iOS models are APISR and SPAN Core ML packages that
//! upscale a 256 tile to 512 and immediately scale it back, so the frame
//! stays the same size. This module does that tiling, overlap blend, and
//! strength mix. The network runs when the matching ONNX file is installed.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Result};
use rife_guards::Frame;

use crate::onnx_runtime::Session;
use crate::pixels::{downscale_box, put_rgb, rgb_at};
use crate::settings::AnimeQuality;

/// Auto picks APISR when a one-tile timing is under the iOS 1000 ms budget.
/// SPAN is the fallback. `duplicate_divisor` in the iOS chooser is 2, so the
/// measured time is halved before the comparison.
pub fn choose(dir: &Path, quality: AnimeQuality, frame: &Frame) -> Result<(PathBuf, &'static str)> {
    let apisr = dir.join("APISR2xRRDB.onnx");
    let span = dir.join("SPAN2xHFA2k.onnx");
    match quality {
        AnimeQuality::Apisr => {
            if apisr.is_file() {
                Ok((apisr, "APISR"))
            } else {
                Err(anyhow!("APISR ONNX is not installed ({})", apisr.display()))
            }
        }
        AnimeQuality::Span => {
            if span.is_file() {
                Ok((span, "SPAN"))
            } else {
                Err(anyhow!("SPAN ONNX is not installed ({})", span.display()))
            }
        }
        AnimeQuality::Auto => {
            if apisr.is_file() {
                let sample = crop256(frame);
                let started = Instant::now();
                let _ = restore(&sample, &apisr, 1.0);
                let ms = started.elapsed().as_secs_f64() * 1000.0 / 2.0;
                if ms <= 1000.0 {
                    return Ok((apisr, "APISR"));
                }
            }
            if span.is_file() {
                Ok((span, "SPAN"))
            } else if apisr.is_file() {
                Ok((apisr, "APISR"))
            } else {
                Err(anyhow!(
                    "Anime Restore ONNX models are not installed in {}",
                    dir.display()
                ))
            }
        }
    }
}

pub fn restore(frame: &Frame, model: &Path, strength: f64) -> Result<Frame> {
    let session = Session::open(model)?;
    restore_session(frame, &session, strength)
}

pub fn restore_session(frame: &Frame, session: &Session, strength: f64) -> Result<Frame> {
    let tile = 256i32;
    let overlap = 16i32;
    let stride = tile - overlap;
    let mut canvas = vec![[0.0f32; 3]; frame.width as usize * frame.height as usize];
    let mut weight = vec![0.0f32; canvas.len()];
    let mut origins_y = origins(frame.height, tile, stride);
    let origins_x = origins(frame.width, tile, stride);
    if origins_y.is_empty() {
        origins_y.push(0);
    }
    for y0 in &origins_y {
        for x0 in &origins_x {
            let patch = reflect_tile(frame, *x0, *y0, tile);
            let input = nchw(&patch);
            let shape = [1i64, 3, tile as i64, tile as i64];
            let out_shape = [1i64, 3, (tile * 2) as i64, (tile * 2) as i64];
            let output = session.run_one("input", &input, &shape, "output", &out_shape)?;
            let up_w = tile * 2;
            let up = rgb_frame_from_nchw(&output, up_w, up_w);
            let down = downscale_box(&up, tile, tile);
            for yy in 0..tile {
                for xx in 0..tile {
                    let dx = *x0 + xx;
                    let dy = *y0 + yy;
                    if dx < 0 || dy < 0 || dx >= frame.width || dy >= frame.height {
                        continue;
                    }
                    let wx = ramp(xx, tile, overlap, *x0 == 0, *x0 + tile >= frame.width);
                    let wy = ramp(yy, tile, overlap, *y0 == 0, *y0 + tile >= frame.height);
                    let wgt = wx * wy;
                    let rgb = rgb_at(&down, xx, yy);
                    let dest = dy as usize * frame.width as usize + dx as usize;
                    for c in 0..3 {
                        canvas[dest][c] += rgb[c] as f32 * wgt;
                    }
                    weight[dest] += wgt;
                }
            }
        }
    }
    let strength = strength.clamp(0.1, 1.0) as f32;
    let mut out = frame.clone();
    for y in 0..frame.height {
        for x in 0..frame.width {
            let i = y as usize * frame.width as usize + x as usize;
            let wgt = weight[i].max(1e-4);
            let restored = [
                (canvas[i][0] / wgt).round().clamp(0.0, 255.0) as u8,
                (canvas[i][1] / wgt).round().clamp(0.0, 255.0) as u8,
                (canvas[i][2] / wgt).round().clamp(0.0, 255.0) as u8,
            ];
            let src = rgb_at(frame, x, y);
            let mix = |a: u8, b: u8| {
                ((1.0 - strength) * a as f32 + strength * b as f32).round() as u8
            };
            put_rgb(&mut out, x, y, [mix(src[0], restored[0]), mix(src[1], restored[1]), mix(src[2], restored[2])]);
        }
    }
    Ok(out)
}

pub fn origins(length: i32, tile: i32, stride: i32) -> Vec<i32> {
    if length <= tile {
        return vec![0];
    }
    let mut out = Vec::new();
    let mut cursor = 0;
    while cursor + tile < length {
        out.push(cursor);
        cursor += stride;
    }
    let last = length - tile;
    if out.last().copied().unwrap_or(-1) != last {
        out.push(last);
    }
    out
}

fn ramp(pos: i32, tile: i32, overlap: i32, at_start: bool, at_end: bool) -> f32 {
    let mut w = 1.0f32;
    if !at_start {
        w = w.min((pos as f32 + 0.5) / overlap as f32);
    }
    if !at_end {
        w = w.min((tile - 1 - pos) as f32 / overlap as f32);
    }
    w.max(0.0)
}

fn crop256(frame: &Frame) -> Frame {
    let side = 256.min(frame.width).min(frame.height).max(1);
    let mut out = Frame::new(side, side, frame.rgba);
    for y in 0..side {
        for x in 0..side {
            put_rgb(&mut out, x, y, rgb_at(frame, x, y));
        }
    }
    out
}

fn reflect_tile(frame: &Frame, x0: i32, y0: i32, tile: i32) -> Frame {
    let mut patch = Frame::new(tile, tile, frame.rgba);
    for y in 0..tile {
        for x in 0..tile {
            let sx = reflect_index(x0 + x, frame.width);
            let sy = reflect_index(y0 + y, frame.height);
            put_rgb(&mut patch, x, y, rgb_at(frame, sx, sy));
        }
    }
    patch
}

fn reflect_index(mut v: i32, length: i32) -> i32 {
    if length <= 1 {
        return 0;
    }
    while v < 0 || v >= length {
        if v < 0 {
            v = -v;
        }
        if v >= length {
            v = 2 * length - 2 - v;
        }
    }
    v.clamp(0, length - 1)
}

fn nchw(frame: &Frame) -> Vec<f32> {
    let plane = frame.width as usize * frame.height as usize;
    let mut data = vec![0.0f32; plane * 3];
    for y in 0..frame.height {
        for x in 0..frame.width {
            let rgb = rgb_at(frame, x, y);
            let i = y as usize * frame.width as usize + x as usize;
            data[i] = rgb[0] as f32 / 255.0;
            data[plane + i] = rgb[1] as f32 / 255.0;
            data[2 * plane + i] = rgb[2] as f32 / 255.0;
        }
    }
    data
}

fn rgb_frame_from_nchw(data: &[f32], width: i32, height: i32) -> Frame {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_origins_cover_the_last_pixel() {
        let rows = origins(1000, 256, 240);
        assert_eq!(*rows.first().unwrap(), 0);
        assert_eq!(*rows.last().unwrap(), 1000 - 256);
    }
}
