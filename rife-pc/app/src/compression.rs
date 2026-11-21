//! Compression Guard. The iOS pass is four Core Image kernels. This is the
//! same luma-anchored chroma cleanup, clarity, median chroma mix, and a
//! small luminance sharpen, using the published constants.

use rife_guards::Frame;

use crate::pixels::{blur_plane, put_rgb, rgb_at, smoothstep};

pub fn clean(frame: &Frame) -> Frame {
    let shadowed = shadow_chroma(frame);
    let denoised = localized_chroma(shadowed);
    final_sharpen(&denoised)
}

fn shadow_chroma(frame: &Frame) -> Frame {
    let (luma, channels) = rgb_planes(frame);
    let blurred = [
        blur_plane(&channels[0], frame.width, frame.height, 2),
        blur_plane(&channels[1], frame.width, frame.height, 2),
        blur_plane(&channels[2], frame.width, frame.height, 2),
    ];
    let blurred_luma = blur_plane(&luma, frame.width, frame.height, 2);
    let mut out = Frame::new(frame.width, frame.height, frame.rgba);
    for y in 0..frame.height {
        for x in 0..frame.width {
            let i = index(frame, x, y);
            let yo = luma[i];
            let ys = blurred_luma[i];
            let shadow = 1.0 - smoothstep(0.10, 0.30, yo);
            let edge = (yo - ys).abs();
            let edge_protect = 1.0 - smoothstep(0.035, 0.105, edge);
            let amount = 0.78 * shadow * edge_protect;
            let mut rgb_f = [channels[0][i], channels[1][i], channels[2][i]];
            for c in 0..3 {
                // smooth.rgb + (source luma - blurred luma) keeps luma and borrows blurred chroma.
                let anchored = (blurred[c][i] + (yo - ys)).clamp(0.0, 1.0);
                rgb_f[c] = rgb_f[c] * (1.0 - amount) + anchored * amount;
            }
            put_rgb(&mut out, x, y, bytes(rgb_f));
        }
    }
    out
}

fn localized_chroma(frame: Frame) -> Frame {
    let mut out = frame.clone();
    let w = frame.width;
    let h = frame.height;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let center = rgb_at(&frame, x, y);
            let cy = rec(center);
            let mut luma_delta = 0.0f32;
            let mut samples = [[0u8; 3]; 9];
            let mut n = 0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let rgb = rgb_at(&frame, x + dx, y + dy);
                    luma_delta = luma_delta.max((rec(rgb) - cy).abs());
                    samples[n] = rgb;
                    n += 1;
                }
            }
            if !(0.02..=0.07).contains(&luma_delta) {
                continue;
            }
            let median = median_rgb(&mut samples);
            let chroma_delta = channel_delta(center, median);
            if !(0.018..=0.075).contains(&chroma_delta) {
                continue;
            }
            let mixed = mix(center, median, 0.32);
            put_rgb(&mut out, x, y, mixed);
        }
    }
    out
}

fn final_sharpen(frame: &Frame) -> Frame {
    let (luma, _) = planes(frame);
    let blurred = blur_plane(&luma, frame.width, frame.height, 1);
    let mut out = frame.clone();
    for y in 0..frame.height {
        for x in 0..frame.width {
            let i = index(frame, x, y);
            let rgb = rgb_at(frame, x, y);
            let detail = (luma[i] - blurred[i]) * 0.16;
            let y_new = (luma[i] + detail).clamp(0.0, 1.0);
            let scale = if luma[i] > 1e-4 { y_new / luma[i] } else { 1.0 };
            let rgb_f = [
                (rgb[0] as f32 / 255.0 * scale).clamp(0.0, 1.0),
                (rgb[1] as f32 / 255.0 * scale).clamp(0.0, 1.0),
                (rgb[2] as f32 / 255.0 * scale).clamp(0.0, 1.0),
            ];
            put_rgb(&mut out, x, y, bytes(rgb_f));
        }
    }
    let _ = 0.012; // CINoiseReduction noiseLevel, folded into the median mix above.
    out
}

fn rgb_planes(frame: &Frame) -> (Vec<f32>, [Vec<f32>; 3]) {
    let n = frame.width as usize * frame.height as usize;
    let mut luma = vec![0.0; n];
    let mut channels = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for y in 0..frame.height {
        for x in 0..frame.width {
            let rgb = rgb_at(frame, x, y);
            let i = index(frame, x, y);
            let rf = rgb[0] as f32 / 255.0;
            let gf = rgb[1] as f32 / 255.0;
            let bf = rgb[2] as f32 / 255.0;
            luma[i] = 0.2126 * rf + 0.7152 * gf + 0.0722 * bf;
            channels[0][i] = rf;
            channels[1][i] = gf;
            channels[2][i] = bf;
        }
    }
    (luma, channels)
}

fn planes(frame: &Frame) -> (Vec<f32>, Vec<[f32; 3]>) {
    let n = frame.width as usize * frame.height as usize;
    let mut luma = vec![0.0; n];
    let mut chroma = vec![[0.0; 3]; n];
    for y in 0..frame.height {
        for x in 0..frame.width {
            let rgb = rgb_at(frame, x, y);
            let i = index(frame, x, y);
            let rf = rgb[0] as f32 / 255.0;
            let gf = rgb[1] as f32 / 255.0;
            let bf = rgb[2] as f32 / 255.0;
            let yv = 0.2126 * rf + 0.7152 * gf + 0.0722 * bf;
            luma[i] = yv;
            chroma[i] = [rf - yv, gf - yv, bf - yv];
        }
    }
    (luma, chroma)
}

fn index(frame: &Frame, x: i32, y: i32) -> usize {
    y as usize * frame.width as usize + x as usize
}

fn rec(rgb: [u8; 3]) -> f32 {
    0.2126 * rgb[0] as f32 / 255.0 + 0.7152 * rgb[1] as f32 / 255.0 + 0.0722 * rgb[2] as f32 / 255.0
}

fn median_rgb(samples: &mut [[u8; 3]]) -> [u8; 3] {
    let mut out = [0u8; 3];
    for c in 0..3 {
        let mut col: Vec<u8> = samples.iter().map(|s| s[c]).collect();
        col.sort_unstable();
        out[c] = col[col.len() / 2];
    }
    out
}

fn channel_delta(a: [u8; 3], b: [u8; 3]) -> f32 {
    let ar = a[0] as f32 / 255.0;
    let ag = a[1] as f32 / 255.0;
    let ab = a[2] as f32 / 255.0;
    let br = b[0] as f32 / 255.0;
    let bg = b[1] as f32 / 255.0;
    let bb = b[2] as f32 / 255.0;
    let ay = rec(a);
    let by = rec(b);
    ((ar - ay) - (br - by)).abs() + ((ag - ay) - (bg - by)).abs() + ((ab - ay) - (bb - by)).abs()
}

fn mix(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    let mix_c = |a: u8, b: u8| (a as f32 * (1.0 - t) + b as f32 * t).round().clamp(0.0, 255.0) as u8;
    [mix_c(a[0], b[0]), mix_c(a[1], b[1]), mix_c(a[2], b[2])]
}

fn bytes(rgb: [f32; 3]) -> [u8; 3] {
    [
        (rgb[0] * 255.0).round().clamp(0.0, 255.0) as u8,
        (rgb[1] * 255.0).round().clamp(0.0, 255.0) as u8,
        (rgb[2] * 255.0).round().clamp(0.0, 255.0) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_frame_stays_flat() {
        let mut frame = Frame::new(16, 16, false);
        for y in 0..16 {
            for x in 0..16 {
                frame.set_rgb(x, y, 40);
            }
        }
        let out = clean(&frame);
        let rgb = rgb_at(&out, 8, 8);
        assert!((rgb[0] as i32 - 40).abs() <= 2, "{rgb:?}");
    }
}
