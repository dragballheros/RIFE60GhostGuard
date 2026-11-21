//! Outline Rebuild is the classical gap closer from `OutlineRebuild.swift`.
//! Final Sharpie Outline runs the ONNX network when
//! `models/onnx/AnimeSharpieOutlineSubtle.onnx` is installed. Without that
//! file the same content-aware gates darken existing lines. That classical
//! path is the approximation, and the status line says so.

use std::path::Path;

use anyhow::Result;
use rife_guards::Frame;

use crate::pixels::{luma8, put_rgb, rgb_at};

pub fn rebuild(frame: &Frame, strength: f64) -> Frame {
    let amount = strength.clamp(0.0, 1.0);
    if amount <= 0.001 || frame.width <= 4 || frame.height <= 4 {
        return frame.clone();
    }
    let width = frame.width;
    let height = frame.height;
    let w = width as usize;
    let mut luma = vec![0u8; w * height as usize];
    for y in 0..height {
        for x in 0..width {
            luma[y as usize * w + x as usize] = luma8(rgb_at(frame, x, y));
        }
    }
    let gap = if amount > 0.66 { 2 } else { 1 };
    let line = line_mask(&luma, width, height, gap);
    let closed = closed_mask(&line, &luma, width, height, gap);
    paint(frame, &luma, &line, &closed, amount)
}

fn line_mask(luma: &[u8], width: i32, height: i32, gap: i32) -> Vec<u8> {
    let w = width as usize;
    let mut line = vec![0u8; luma.len()];
    for y in gap..height - gap {
        for x in gap..width - gap {
            let index = y as usize * w + x as usize;
            let center = luma[index] as i32;
            let mut local_max = center;
            for yy in (y - gap)..=(y + gap) {
                for xx in (x - gap)..=(x + gap) {
                    local_max = local_max.max(luma[yy as usize * w + xx as usize] as i32);
                }
            }
            if center < 150 && center + 16 < local_max {
                line[index] = 1;
            }
        }
    }
    line
}

fn closed_mask(line: &[u8], luma: &[u8], width: i32, height: i32, gap: i32) -> Vec<u8> {
    let mut closed = line.to_vec();
    let w = width as usize;
    for y in gap..height - gap {
        for x in gap..width - gap {
            let index = y as usize * w + x as usize;
            if line[index] == 1 {
                continue;
            }
            if luma[index] < 250 && opposing(line, x, y, width, height, gap) {
                closed[index] = 1;
            }
        }
    }
    closed
}

fn continuation(line: &[u8], x: i32, y: i32, dx: i32, dy: i32, direction: i32, width: i32, height: i32) -> bool {
    let w = width as usize;
    for distance in 1..=2 {
        let nx = x + dx * distance * direction;
        let ny = y + dy * distance * direction;
        if nx < 0 || ny < 0 || nx >= width || ny >= height {
            return false;
        }
        if line[ny as usize * w + nx as usize] != 0 {
            return true;
        }
    }
    false
}

fn opposing(line: &[u8], x: i32, y: i32, width: i32, height: i32, max_distance: i32) -> bool {
    let w = width as usize;
    let directions = [(1, 0), (0, 1), (1, 1), (1, -1)];
    for distance in 1..=max_distance {
        for (dx, dy) in directions {
            let ax = x - dx * distance;
            let ay = y - dy * distance;
            let bx = x + dx * distance;
            let by = y + dy * distance;
            if ax < 0 || ay < 0 || bx >= width || by < 0 || by >= height {
                continue;
            }
            if line[ay as usize * w + ax as usize] != 0
                && line[by as usize * w + bx as usize] != 0
                && continuation(line, ax, ay, dx, dy, -1, width, height)
                && continuation(line, bx, by, dx, dy, 1, width, height)
            {
                return true;
            }
        }
    }
    false
}

fn paint(frame: &Frame, luma: &[u8], line: &[u8], closed: &[u8], amount: f64) -> Frame {
    let mut out = frame.clone();
    let width = frame.width;
    let height = frame.height;
    let w = width as usize;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let index = y as usize * w + x as usize;
            if closed[index] == 0 && line[index] == 0 {
                continue;
            }
            if crate::blush::protects(frame, x, y) {
                continue;
            }
            let (neighbors, darkest) = neighbor_stats(closed, luma, x, y, width);
            let center = luma[index] as i32;
            if line[index] == 1 && protected_fine(frame, x, y) {
                continue;
            }
            if closed[index] == 1 && neighbors >= 2 {
                let lift = ((1.0 - amount) * 18.0) as i32;
                let target = center.min(darkest + lift);
                let keep = 1.0 - amount * 0.65;
                let mixed = (center as f64 * keep + target as f64 * (1.0 - keep)).round() as i32;
                if mixed < center {
                    scale_pixel(&mut out, x, y, mixed.max(1) as f64 / center.max(1) as f64);
                }
            } else if line[index] == 1 && neighbors <= 1 {
                if let Some(paper) = paper_luma(closed, luma, x, y, width) {
                    let keep = 1.0 - amount * 0.7;
                    let mixed = (center as f64 * keep + paper as f64 * (1.0 - keep)).round() as i32;
                    if mixed > center {
                        scale_pixel(&mut out, x, y, mixed as f64 / center.max(1) as f64);
                    }
                }
            }
        }
    }
    out
}

fn neighbor_stats(closed: &[u8], luma: &[u8], x: i32, y: i32, width: i32) -> (i32, i32) {
    let w = width as usize;
    let mut neighbors = 0;
    let mut darkest = 255;
    for yy in (y - 1)..=(y + 1) {
        for xx in (x - 1)..=(x + 1) {
            if xx == x && yy == y {
                continue;
            }
            let i = yy as usize * w + xx as usize;
            if closed[i] == 1 {
                neighbors += 1;
                darkest = darkest.min(luma[i] as i32);
            }
        }
    }
    (neighbors, darkest)
}

fn paper_luma(closed: &[u8], luma: &[u8], x: i32, y: i32, width: i32) -> Option<i32> {
    let w = width as usize;
    let mut paper = 0;
    let mut count = 0;
    for yy in (y - 1)..=(y + 1) {
        for xx in (x - 1)..=(x + 1) {
            let i = yy as usize * w + xx as usize;
            if closed[i] == 0 {
                paper += luma[i] as i32;
                count += 1;
            }
        }
    }
    if count == 0 {
        None
    } else {
        Some(paper / count)
    }
}

fn protected_fine(frame: &Frame, x: i32, y: i32) -> bool {
    let center = rgb_at(frame, x, y);
    let chroma = center[0].max(center[1]).max(center[2]) - center[0].min(center[1]).min(center[2]);
    if chroma >= 18 {
        return true;
    }
    let distance = |ax: i32, ay: i32, bx: i32, by: i32| {
        let a = rgb_at(frame, ax, ay);
        let b = rgb_at(frame, bx, by);
        (a[0] as i32 - b[0] as i32).abs()
            + (a[1] as i32 - b[1] as i32).abs()
            + (a[2] as i32 - b[2] as i32).abs()
    };
    let lx = (x - 2).max(0);
    let rx = (x + 2).min(frame.width - 1);
    let ty = (y - 2).max(0);
    let by = (y + 2).min(frame.height - 1);
    let horizontal = distance(x, y, lx, y).min(distance(x, y, rx, y));
    let horizontal_flanks = distance(lx, y, rx, y);
    let vertical = distance(x, y, x, ty).min(distance(x, y, x, by));
    let vertical_flanks = distance(x, ty, x, by);
    (horizontal >= 24 && horizontal_flanks <= 64) || (vertical >= 24 && vertical_flanks <= 64)
}

fn scale_pixel(frame: &mut Frame, x: i32, y: i32, factor: f64) {
    let rgb = rgb_at(frame, x, y);
    let scale = |c: u8| (c as f64 * factor).round().clamp(0.0, 255.0) as u8;
    put_rgb(frame, x, y, [scale(rgb[0]), scale(rgb[1]), scale(rgb[2])]);
}

pub struct SharpieResult {
    pub frame: Frame,
    pub used_network: bool,
}

/// Tile the Sharpie network when the ONNX file is present. Otherwise darken
/// pixels the line mask already marked, then blend 94% of that toward the
/// source the way the iOS post-network mix does, and pull blush strokes back.
pub fn sharpie(frame: &Frame, model: Option<&Path>) -> Result<SharpieResult> {
    if let Some(path) = model {
        if path.is_file() {
            if let Ok(enhanced) = sharpie_onnx(frame, path) {
                return Ok(SharpieResult {
                    frame: refine(frame, &enhanced),
                    used_network: true,
                });
            }
        }
    }
    Ok(SharpieResult {
        frame: refine(frame, &classical_emphasis(frame)),
        used_network: false,
    })
}

fn classical_emphasis(frame: &Frame) -> Frame {
    let rebuilt = rebuild(frame, 0.85);
    let mut out = frame.clone();
    let w = frame.width as usize;
    let luma = {
        let mut plane = vec![0u8; w * frame.height as usize];
        for y in 0..frame.height {
            for x in 0..frame.width {
                plane[y as usize * w + x as usize] = luma8(rgb_at(frame, x, y));
            }
        }
        plane
    };
    for y in 1..frame.height - 1 {
        for x in 1..frame.width - 1 {
            let i = y as usize * w + x as usize;
            let center = luma[i] as i32;
            let mut local = center;
            for yy in (y - 1)..=(y + 1) {
                for xx in (x - 1)..=(x + 1) {
                    local = local.max(luma[yy as usize * w + xx as usize] as i32);
                }
            }
            if center < 150 && center + 16 < local {
                let rgb = rgb_at(&rebuilt, x, y);
                let darker = [
                    (rgb[0] as f32 * 0.82).round() as u8,
                    (rgb[1] as f32 * 0.82).round() as u8,
                    (rgb[2] as f32 * 0.82).round() as u8,
                ];
                // iOS mixes the network 0.94 toward the model and 0.06 source.
                let src = rgb_at(frame, x, y);
                let mix = |a: u8, b: u8| (a as f32 * 0.06 + b as f32 * 0.94).round() as u8;
                put_rgb(&mut out, x, y, [mix(src[0], darker[0]), mix(src[1], darker[1]), mix(src[2], darker[2])]);
            }
        }
    }
    out
}

fn refine(source: &Frame, enhanced: &Frame) -> Frame {
    let mut out = enhanced.clone();
    for y in 0..source.height {
        for x in 0..source.width {
            if crate::blush::protects(source, x, y) {
                put_rgb(&mut out, x, y, rgb_at(source, x, y));
            }
        }
    }
    out
}

fn sharpie_onnx(frame: &Frame, model: &Path) -> Result<Frame> {
    let tile = 256;
    let shrink = 20;
    let core = tile - shrink * 2;
    let session = crate::onnx_runtime::Session::open(model)?;
    let mut acc = vec![[0.0f32; 3]; frame.width as usize * frame.height as usize];
    let mut weight = vec![0.0f32; acc.len()];
    let mut y0 = 0;
    while y0 < frame.height {
        let mut x0 = 0;
        while x0 < frame.width {
            let patch = sample_tile(frame, x0, y0, tile);
            let mut input = vec![0.0f32; 3 * tile as usize * tile as usize];
            for yy in 0..tile {
                for xx in 0..tile {
                    let rgb = rgb_at(&patch, xx, yy);
                    let base = yy as usize * tile as usize + xx as usize;
                    input[base] = rgb[0] as f32 / 255.0;
                    input[tile as usize * tile as usize + base] = rgb[1] as f32 / 255.0;
                    input[2 * tile as usize * tile as usize + base] = rgb[2] as f32 / 255.0;
                }
            }
            let shape = [1, 3, tile as i64, tile as i64];
            let output = session.run_one("x", &input, &shape, "var_244", &shape)?;
            let x_start = if x0 == 0 { 0 } else { shrink };
            let y_start = if y0 == 0 { 0 } else { shrink };
            let x_end = if x0 + tile >= frame.width { tile } else { tile - shrink };
            let y_end = if y0 + tile >= frame.height { tile } else { tile - shrink };
            for yy in y_start..y_end {
                for xx in x_start..x_end {
                    let dx = x0 + xx - if x0 == 0 { 0 } else { shrink };
                    let dy = y0 + yy - if y0 == 0 { 0 } else { shrink };
                    if dx < 0 || dy < 0 || dx >= frame.width || dy >= frame.height {
                        continue;
                    }
                    let base = yy as usize * tile as usize + xx as usize;
                    let plane = tile as usize * tile as usize;
                    let rgb = [
                        output.get(base).copied().unwrap_or(0.0),
                        output.get(plane + base).copied().unwrap_or(0.0),
                        output.get(2 * plane + base).copied().unwrap_or(0.0),
                    ];
                    let dest = dy as usize * frame.width as usize + dx as usize;
                    for c in 0..3 {
                        acc[dest][c] += rgb[c];
                    }
                    weight[dest] += 1.0;
                }
            }
            if x0 + core >= frame.width {
                break;
            }
            x0 += core;
        }
        if y0 + core >= frame.height {
            break;
        }
        y0 += core;
    }
    let mut out = frame.clone();
    for y in 0..frame.height {
        for x in 0..frame.width {
            let i = y as usize * frame.width as usize + x as usize;
            if weight[i] <= 0.0 {
                continue;
            }
            let src = rgb_at(frame, x, y);
            let model_rgb = [
                (acc[i][0] / weight[i] * 255.0).round().clamp(0.0, 255.0) as u8,
                (acc[i][1] / weight[i] * 255.0).round().clamp(0.0, 255.0) as u8,
                (acc[i][2] / weight[i] * 255.0).round().clamp(0.0, 255.0) as u8,
            ];
            let mix = |a: u8, b: u8| (a as f32 * 0.06 + b as f32 * 0.94).round() as u8;
            put_rgb(
                &mut out,
                x,
                y,
                [mix(src[0], model_rgb[0]), mix(src[1], model_rgb[1]), mix(src[2], model_rgb[2])],
            );
        }
    }
    Ok(out)
}

fn sample_tile(frame: &Frame, x0: i32, y0: i32, tile: i32) -> Frame {
    let mut patch = Frame::new(tile, tile, frame.rgba);
    for y in 0..tile {
        for x in 0..tile {
            let sx = (x0 + x - if x0 == 0 { 0 } else { 20 }).clamp(0, frame.width - 1);
            let sy = (y0 + y - if y0 == 0 { 0 } else { 20 }).clamp(0, frame.height - 1);
            put_rgb(&mut patch, x, y, rgb_at(frame, sx, sy));
        }
    }
    patch
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_one_pixel_gap_in_a_dark_line_is_closed() {
        let mut frame = Frame::new(32, 32, false);
        for y in 0..32 {
            for x in 0..32 {
                frame.set_rgb(x, y, 240);
            }
        }
        for x in 4..28 {
            if x != 16 {
                frame.set_rgb(x, 16, 10);
            }
        }
        let out = rebuild(&frame, 0.8);
        let rgb = rgb_at(&out, 16, 16);
        assert!(rgb[0] < 180, "gap pixel stayed paper-bright: {rgb:?}");
    }
}
