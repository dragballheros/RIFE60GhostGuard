//! Broken-tile stage diagnostics. Every 15th frame, score luminance steps
//! that land on a power-of-two grid. The JSON report matches the fields the
//! iOS exporter writes closely enough to compare runs.

use std::fs;
use std::path::{Path, PathBuf};

use rife_guards::Frame;
use serde::Serialize;

use crate::pixels::rgb_at;

#[derive(Clone, Debug, Serialize)]
pub struct SeamSample {
    pub stage: String,
    pub frame: u64,
    pub score: f32,
    pub grid: i32,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunReport {
    pub total_seconds: f64,
    pub stages: Vec<String>,
    pub notes: Vec<String>,
    pub metrics: std::collections::BTreeMap<String, f64>,
    pub samples: Vec<SeamSample>,
    pub rife_rejections: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Recorder {
    pub report: RunReport,
    pub dir: PathBuf,
    saved_images: usize,
}

impl Recorder {
    pub fn new(dir: PathBuf) -> Self {
        Self {
            report: RunReport {
                total_seconds: 0.0,
                stages: Vec::new(),
                notes: Vec::new(),
                metrics: std::collections::BTreeMap::new(),
                samples: Vec::new(),
                rife_rejections: Vec::new(),
            },
            dir,
            saved_images: 0,
        }
    }

    pub fn note(&mut self, text: impl Into<String>) {
        self.report.notes.push(text.into());
    }

    pub fn metric(&mut self, key: &str, value: f64) {
        self.report.metrics.insert(key.to_string(), value);
    }

    pub fn observe(&mut self, stage: &str, index: u64, frame: &Frame) {
        if index % 15 != 0 {
            return;
        }
        let (score, grid) = seam_score(frame);
        self.report.samples.push(SeamSample {
            stage: stage.to_string(),
            frame: index,
            score,
            grid,
        });
        if score > 1.5 && self.saved_images < 5 {
            let _ = fs::create_dir_all(&self.dir);
            let path = self.dir.join(format!("{stage}-f{index}-r{score:.3}.bmp"));
            if write_bmp(frame, &path).is_ok() {
                self.saved_images += 1;
            }
        }
    }

    pub fn finish(&self, seconds: f64) -> std::io::Result<PathBuf> {
        let mut report = self.report.clone();
        report.total_seconds = seconds;
        fs::create_dir_all(&self.dir)?;
        let path = self.dir.join("run.json");
        fs::write(&path, serde_json::to_vec_pretty(&report).unwrap_or_default())?;
        Ok(path)
    }
}

/// Score a grid-aligned luminance step against the pixels three samples away.
pub fn seam_score(frame: &Frame) -> (f32, i32) {
    let mut best = 0.0f32;
    let mut grid = 32;
    for g in [16, 32, 64, 128, 256] {
        if g >= frame.width || g >= frame.height {
            continue;
        }
        let mut seam = 0.0f32;
        let mut baseline = 0.0f32;
        let mut count = 0.0f32;
        let mut y = 0;
        while y < frame.height {
            let mut x = g;
            while x < frame.width {
                let a = luma(frame, x - 1, y);
                let b = luma(frame, x, y);
                let left = luma(frame, (x - 1 - 3).max(0), y);
                let right = luma(frame, (x + 3).min(frame.width - 1), y);
                seam += (a - b).abs();
                baseline += (left - a).abs().max((b - right).abs());
                count += 1.0;
                x += g;
            }
            y += 8.max(frame.height / 64);
        }
        if count < 1.0 {
            continue;
        }
        seam /= count;
        baseline = (baseline / count).max(1e-3);
        let score = (seam / baseline) * (1.0 + seam).ln();
        if score > best {
            best = score;
            grid = g;
        }
    }
    (best, grid)
}

fn luma(frame: &Frame, x: i32, y: i32) -> f32 {
    let rgb = rgb_at(frame, x, y);
    0.2126 * rgb[0] as f32 + 0.7152 * rgb[1] as f32 + 0.0722 * rgb[2] as f32
}

pub fn write_bmp(frame: &Frame, path: &Path) -> std::io::Result<()> {
    let width = frame.width.max(1);
    let height = frame.height.max(1);
    let row = (width as usize * 3 + 3) & !3;
    let pixels = row * height as usize;
    let mut data = vec![0u8; 54 + pixels];
    data[0] = b'B';
    data[1] = b'M';
    let size = data.len() as u32;
    data[2..6].copy_from_slice(&size.to_le_bytes());
    data[10..14].copy_from_slice(&54u32.to_le_bytes());
    data[14..18].copy_from_slice(&40u32.to_le_bytes());
    data[18..22].copy_from_slice(&width.to_le_bytes());
    data[22..26].copy_from_slice(&height.to_le_bytes());
    data[26..28].copy_from_slice(&1u16.to_le_bytes());
    data[28..30].copy_from_slice(&24u16.to_le_bytes());
    for y in 0..height {
        let src_y = height - 1 - y;
        for x in 0..width {
            let rgb = rgb_at(frame, x, src_y);
            let i = 54 + y as usize * row + x as usize * 3;
            data[i] = rgb[2];
            data[i + 1] = rgb[1];
            data[i + 2] = rgb[0];
        }
    }
    fs::write(path, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_grid_step_scores_higher_than_a_flat_frame() {
        let flat = Frame::new(64, 64, false);
        let mut seam = Frame::new(64, 64, false);
        for y in 0..64 {
            for x in 0..64 {
                let v = if x < 32 { 20 } else { 220 };
                seam.set_rgb(x, y, v);
            }
        }
        let (flat_score, _) = seam_score(&flat);
        let (seam_score, grid) = seam_score(&seam);
        assert!(seam_score > flat_score, "{seam_score} vs {flat_score}");
        assert_eq!(grid, 32);
    }
}
