//! Cheek-blush hatch guard, ported from `BlushStrokeGuard.swift`.
//! Strong dark outlines fail the contrast cap. A stroke counts only when both
//! flanks are the same warm pink, so hair and ordinary line art stay on the
//! normal cleanup path.

use rife_guards::Frame;

use crate::pixels::rgb_at;

struct Sample {
    r: i32,
    g: i32,
    b: i32,
    y: i32,
    cb: i32,
    cr: i32,
}

fn load(frame: &Frame, x: i32, y: i32) -> Sample {
    let rgb = rgb_at(frame, x, y);
    let r = rgb[0] as i32;
    let g = rgb[1] as i32;
    let b = rgb[2] as i32;
    let y8 = (77 * r + 150 * g + 29 * b) >> 8;
    let cb = ((-43 * r - 85 * g + 128 * b) >> 8) + 128;
    let cr = ((128 * r - 107 * g - 21 * b) >> 8) + 128;
    Sample {
        r,
        g,
        b,
        y: y8,
        cb,
        cr,
    }
}

fn warm_pink(sample: &Sample) -> bool {
    sample.y >= 105
        && sample.y <= 225
        && sample.cr >= 148
        && sample.cr <= 205
        && sample.cb >= 96
        && sample.cb <= 142
        && sample.r >= sample.b + 12
        && sample.r + 6 >= sample.g
}

fn is_stroke(frame: &Frame, x: i32, y: i32) -> bool {
    let w = frame.width;
    let h = frame.height;
    if x < 2 || y < 2 || x >= w - 2 || y >= h - 2 {
        return false;
    }
    let center = load(frame, x, y);
    if center.y < 80 || center.y > 205 || center.r < center.b + 6 || center.cr < 132 {
        return false;
    }
    let axes = [(1, 0), (0, 1), (1, 1), (1, -1)];
    for (dx, dy) in axes {
        for dist in [1, 2] {
            let ax = x - dx * dist;
            let ay = y - dy * dist;
            let bx = x + dx * dist;
            let by = y + dy * dist;
            if ax < 0 || ay < 0 || bx < 0 || by < 0 || ax >= w || ay >= h || bx >= w || by >= h {
                continue;
            }
            let flank_a = load(frame, ax, ay);
            let flank_b = load(frame, bx, by);
            if !warm_pink(&flank_a) || !warm_pink(&flank_b) {
                continue;
            }
            if (flank_a.y - flank_b.y).abs() > 14 {
                continue;
            }
            let darker = (flank_a.y - center.y).min(flank_b.y - center.y);
            if (8..=32).contains(&darker) {
                return true;
            }
        }
    }
    false
}

fn luma_range(frame: &Frame, x: i32, y: i32) -> i32 {
    let mut low = 255;
    let mut high = 0;
    for dy in -1..=1 {
        let sy = y + dy;
        if sy < 0 || sy >= frame.height {
            continue;
        }
        for dx in -1..=1 {
            let sx = x + dx;
            if sx < 0 || sx >= frame.width {
                continue;
            }
            let y8 = load(frame, sx, sy).y;
            low = low.min(y8);
            high = high.max(y8);
        }
    }
    high - low
}

pub fn protects(frame: &Frame, x: i32, y: i32) -> bool {
    if x < 0 || y < 0 || x >= frame.width || y >= frame.height {
        return false;
    }
    if is_stroke(frame, x, y) {
        return true;
    }
    let center = load(frame, x, y);
    if !warm_pink(&center) || luma_range(frame, x, y) < 10 {
        return false;
    }
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let sx = x + dx;
            let sy = y + dy;
            if sx < 0 || sy < 0 || sx >= frame.width || sy >= frame.height {
                continue;
            }
            if is_stroke(frame, sx, sy) {
                return true;
            }
        }
    }
    false
}
