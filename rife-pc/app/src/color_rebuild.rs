//! Luma-guided chroma rebuild from `ColorRebuild.swift`.
//! Flat regions take a guided Cb/Cr. Edges, skin, and fine detail stay put.
//! A small temporal hold keeps chroma from flickering when luma barely moves.

use rife_guards::Frame;

use crate::pixels::{put_rgb, rgb_at, ycbcr, ycbcr_to_rgb};

#[derive(Clone, Debug, Default)]
pub struct ColorRebuildState {
    prev_y: Vec<u8>,
    prev_cb: Vec<i16>,
    prev_cr: Vec<i16>,
}

pub fn apply(frame: &Frame, strength: f64, state: &mut ColorRebuildState) -> Frame {
    let strength = strength.clamp(0.0, 1.0);
    if strength <= 0.001 || frame.width < 3 || frame.height < 3 {
        return frame.clone();
    }
    let w = frame.width as usize;
    let h = frame.height as usize;
    let mut y_plane = vec![0u8; w * h];
    let mut cb = vec![0i16; w * h];
    let mut cr = vec![0i16; w * h];
    for y in 0..h {
        for x in 0..w {
            let (yy, cbb, crr) = ycbcr(rgb_at(frame, x as i32, y as i32));
            y_plane[y * w + x] = yy;
            cb[y * w + x] = cbb;
            cr[y * w + x] = crr;
        }
    }
    let radius = if strength > 0.7 { 4 } else { 3 };
    let cb_g = guided(&cb, &y_plane, frame.width, frame.height, radius);
    let cr_g = guided(&cr, &y_plane, frame.width, frame.height, radius);
    let mut out = Frame::new(frame.width, frame.height, frame.rgba);
    let have_prev = state.prev_y.len() == y_plane.len();
    for y in 0..frame.height {
        for x in 0..frame.width {
            let i = y as usize * w + x as usize;
            let rgb = rgb_at(frame, x, y);
            let fine = fine_detail(&y_plane, &cb, &cr, frame.width, frame.height, x, y);
            if fine >= 0.82 || crate::blush::protects(frame, x, y) {
                put_rgb(&mut out, x, y, rgb);
                continue;
            }
            let edge = luma_edge(&y_plane, frame.width, frame.height, x, y);
            let flat = if edge < 12.0 {
                0.82
            } else if edge < 28.0 {
                0.38
            } else {
                0.12
            };
            let mut mix = strength as f32 * flat * (1.0 - fine);
            let yy = y_plane[i] as i32;
            let is_skin = (40..=235).contains(&yy)
                && (85..=135).contains(&cb[i])
                && (140..=180).contains(&cr[i]);
            if is_skin {
                mix *= 0.42;
            }
            let mut out_cb = cb[i] as f32 * (1.0 - mix) + cb_g[i] * mix;
            let mut out_cr = cr[i] as f32 * (1.0 - mix) + cr_g[i] * mix;
            if have_prev && (yy - state.prev_y[i] as i32).abs() < 8 {
                let hold = 0.34 * (1.0 - fine);
                out_cb = out_cb * (1.0 - hold) + state.prev_cb[i] as f32 * hold;
                out_cr = out_cr * (1.0 - hold) + state.prev_cr[i] as f32 * hold;
            }
            put_rgb(
                &mut out,
                x,
                y,
                ycbcr_to_rgb(yy, out_cb.round() as i32, out_cr.round() as i32),
            );
        }
    }
    state.prev_y = y_plane;
    state.prev_cb = cb;
    state.prev_cr = cr;
    out
}

fn guided(chroma: &[i16], guide: &[u8], width: i32, height: i32, radius: i32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let mut horiz = vec![0.0f32; chroma.len()];
    for y in 0..h {
        for x in 0..w {
            let center = guide[y * w + x] as i32;
            let mut acc = 0.0;
            let mut weight = 0.0;
            let x0 = (x as i32 - radius).max(0) as usize;
            let x1 = (x as i32 + radius).min(width - 1) as usize;
            for xx in x0..=x1 {
                let wgt = 64.0 / (1.0 + (center - guide[y * w + xx] as i32).abs() as f32 / 3.0);
                acc += chroma[y * w + xx] as f32 * wgt;
                weight += wgt;
            }
            horiz[y * w + x] = if weight > 0.0 { acc / weight } else { chroma[y * w + x] as f32 };
        }
    }
    let mut out = vec![0.0f32; chroma.len()];
    for y in 0..h {
        for x in 0..w {
            let center = guide[y * w + x] as i32;
            let mut acc = 0.0;
            let mut weight = 0.0;
            let y0 = (y as i32 - radius).max(0) as usize;
            let y1 = (y as i32 + radius).min(height - 1) as usize;
            for yy in y0..=y1 {
                let wgt = 64.0 / (1.0 + (center - guide[yy * w + x] as i32).abs() as f32 / 3.0);
                acc += horiz[yy * w + x] * wgt;
                weight += wgt;
            }
            out[y * w + x] = if weight > 0.0 { acc / weight } else { horiz[y * w + x] };
        }
    }
    out
}

fn luma_edge(luma: &[u8], width: i32, height: i32, x: i32, y: i32) -> f32 {
    let at = |x: i32, y: i32| luma[y as usize * width as usize + x as usize] as f32;
    let x0 = (x - 1).max(0);
    let x1 = (x + 1).min(width - 1);
    let y0 = (y - 1).max(0);
    let y1 = (y + 1).min(height - 1);
    let dx = (at(x1, y) - at(x0, y)).abs();
    let dy = (at(x, y1) - at(x, y0)).abs();
    dx.max(dy)
}

fn fine_detail(luma: &[u8], cb: &[i16], cr: &[i16], width: i32, height: i32, x: i32, y: i32) -> f32 {
    let sample = |dx: i32, dy: i32| -> (f32, f32) {
        let xx = (x + dx).clamp(0, width - 1);
        let yy = (y + dy).clamp(0, height - 1);
        let i = yy as usize * width as usize + xx as usize;
        let chroma = ((cb[i] - 128).abs() + (cr[i] - 128).abs()) as f32;
        (luma[i] as f32, chroma)
    };
    let (cy, cc) = sample(0, 0);
    let mut best = 0.0f32;
    for (dx, dy) in [(1, 0), (0, 1), (2, 0), (0, 2)] {
        let (ay, ac) = sample(-dx, -dy);
        let (by, bc) = sample(dx, dy);
        best = best.max(response(cy, ay, by, 10.0, 34.0));
        best = best.max(response(cc, ac, bc, 12.0, 38.0));
    }
    best
}

fn response(center: f32, a: f32, b: f32, lo: f32, hi: f32) -> f32 {
    let flank = ((a - b).abs()).min(hi);
    let contrast = (center - a).abs().max((center - b).abs());
    let shaped = contrast * (1.0 - flank / hi.max(1.0));
    crate::pixels::smoothstep(lo, hi, shaped)
}
