//! Rejects an in-between whose outlines are not the sources' outlines.

use crate::frame::{dilate, sample_rgb, Frame};

pub fn rejects(previous: &Frame, generated: &Frame, next: &Frame, sensitivity: f64) -> bool {
    if previous.rgba != generated.rgba || previous.rgba != next.rgba {
        return false;
    }
    let width = previous.width.min(generated.width).min(next.width);
    let height = previous.height.min(generated.height).min(next.height);
    if width < 16 || height < 16 {
        return false;
    }
    let step = 2.max(width.max(height) / 480);
    let columns = (width - 1) / step + 1;
    let rows = (height - 1) / step + 1;
    if columns < 8 || rows < 8 {
        return false;
    }
    let rgba = previous.rgba;
    let luma_a = plane(previous, width, height, step, columns, rows, rgba);
    let luma_m = plane(generated, width, height, step, columns, rows, rgba);
    let luma_c = plane(next, width, height, step, columns, rows, rgba);
    let edges_a = edges(&luma_a, columns, rows);
    let edges_m = edges(&luma_m, columns, rows);
    let edges_c = edges(&luma_c, columns, rows);
    let count_a: i32 = edges_a.iter().map(|v| *v as i32).sum();
    let count_m: i32 = edges_m.iter().map(|v| *v as i32).sum();
    let count_c: i32 = edges_c.iter().map(|v| *v as i32).sum();
    if count_m < 24 {
        return false;
    }
    let tuned = crate::frame::clamp_sensitivity(sensitivity);
    // 1.50 at sensitivity 1.35, 1.75 at 0.65.
    let double_ratio = 1.50 + 0.25 * (1.35 - tuned) / 0.70;
    let edge_max = count_a.max(count_c);
    let edge_min = count_a.min(count_c);
    let sources_agree = edge_max >= 24 && edge_min * 3 >= edge_max * 2;
    if sources_agree && (count_m as f64) > (edge_max as f64) * double_ratio {
        return true;
    }
    let shift = max_edge_shift(&edges_a, &edges_c, columns, rows, 16);
    let radius = 8.min(2.max((shift + 3) / 2));
    let mut allowed = edges_a;
    for index in 0..allowed.len() {
        if edges_c[index] != 0 {
            allowed[index] = 1;
        }
    }
    dilate(&mut allowed, columns, rows, radius);
    let mut novel = 0i32;
    for index in 0..edges_m.len() {
        if edges_m[index] != 0 && allowed[index] == 0 {
            novel += 1;
        }
    }
    // 0.25 at sensitivity 1.35, 0.35 at 0.65.
    let novel_limit = 0.25 + 0.10 * (1.35 - tuned) / 0.70;
    (novel as f64) / (count_m as f64) > novel_limit
}

fn plane(
    frame: &Frame,
    width: i32,
    height: i32,
    step: i32,
    columns: i32,
    rows: i32,
    rgba: bool,
) -> Vec<f32> {
    let mut luma = vec![0.0f32; (columns * rows) as usize];
    for row in 0..rows {
        let y = (height - 1).min(row * step);
        let line = frame.row(y);
        for column in 0..columns {
            let x = (width - 1).min(column * step);
            let pixel = (x as usize) * 4;
            luma[(row * columns + column) as usize] = sample_rgb(&line[pixel..], rgba).y;
        }
    }
    luma
}

fn edges(luma: &[f32], columns: i32, rows: i32) -> Vec<u8> {
    let mut mask = vec![0u8; (columns * rows) as usize];
    if columns <= 2 || rows <= 2 {
        return mask;
    }
    for y in 1..(rows - 1) {
        for x in 1..(columns - 1) {
            let index = (y * columns + x) as usize;
            let magnitude = (luma[index + 1] - luma[index - 1]).abs()
                + (luma[index + columns as usize] - luma[index - columns as usize]).abs();
            if magnitude >= 0.12 {
                mask[index] = 1;
            }
        }
    }
    mask
}

fn max_edge_shift(source: &[u8], other: &[u8], columns: i32, rows: i32, search: i32) -> i32 {
    let mut farthest = 0i32;
    let cols = columns as usize;
    for y in 0..rows {
        for x in 0..columns {
            if source[(y as usize) * cols + x as usize] == 0 {
                continue;
            }
            let mut found = search;
            if other[(y as usize) * cols + x as usize] != 0 {
                found = 0;
            } else {
                'ring: for radius in 1..=search {
                    for dy in -radius..=radius {
                        let yy = y + dy;
                        if yy < 0 || yy >= rows {
                            continue;
                        }
                        for dx in -radius..=radius {
                            if dx.abs().max(dy.abs()) != radius {
                                continue;
                            }
                            let xx = x + dx;
                            if xx < 0 || xx >= columns {
                                continue;
                            }
                            if other[(yy as usize) * cols + xx as usize] != 0 {
                                found = radius;
                                break 'ring;
                            }
                        }
                    }
                }
            }
            if found > farthest {
                farthest = found;
            }
        }
    }
    farthest
}
