//! Output timestamps. Source frame `i` is shown at `i / source_fps`.
//! The output grid runs until `source_count / source_fps`, so audio stays aligned.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rational {
    pub num: i64,
    pub den: i64,
}

impl Rational {
    pub fn new(num: i64, den: i64) -> Self {
        let den = if den == 0 { 1 } else { den };
        let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
        let g = gcd(num.abs(), den.abs());
        Self {
            num: num / g,
            den: den / g,
        }
    }

    pub fn scale(self, factor: i64) -> Self {
        Self::new(self.num * factor.max(1), self.den)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Slot {
    Source(usize),
    Interp { left: usize, timestep: f32 },
}

pub fn output_rate(source: Rational, multi: u32, fps: Option<f64>) -> Rational {
    if let Some(fps) = fps {
        if fps.is_finite() && fps > 1.0 {
            return rate_from_fps(fps);
        }
    }
    source.scale(multi.max(1) as i64)
}

pub fn output_count(source_frames: i64, source: Rational, out: Rational) -> i64 {
    if source_frames <= 0 {
        return 0;
    }
    div_round(source_frames * source.den * out.num, source.num * out.den).max(1)
}

/// iOS cadence. Output times step at `target_fps`, but each step is placed on
/// the real gap between source presentation timestamps. A long phone-video gap
/// gets more in-betweens than a short one. Constant 60 → 60 stays on the
/// source frames.
pub fn slots_from_pts(pts: &[f64], target_fps: f64) -> Vec<Slot> {
    if pts.is_empty() {
        return Vec::new();
    }
    if pts.len() == 1 || !pts.windows(2).all(|pair| pair[1] > pair[0] + 1e-5) {
        return (0..pts.len()).map(Slot::Source).collect();
    }
    let step = 1.0 / target_fps.max(1.0);
    let mut slots = vec![Slot::Source(0)];
    let mut next_t = pts[0] + step;
    let last = pts.len() - 1;
    for index in 0..last {
        let start = pts[index];
        let end = pts[index + 1];
        let span = (end - start).max(1.0 / 240.0);
        while next_t < end - 1e-5 {
            let rel = ((next_t - start) / span) as f32;
            slots.push(if rel <= 0.001 {
                Slot::Source(index)
            } else if rel >= 0.999 {
                Slot::Source(index + 1)
            } else {
                Slot::Interp {
                    left: index,
                    timestep: rel.clamp(0.001, 0.999),
                }
            });
            next_t += step;
        }
    }
    if !matches!(slots.last(), Some(Slot::Source(index)) if *index == last) {
        slots.push(Slot::Source(last));
    }
    slots
}

pub fn slot_at(j: i64, source_frames: i64, source: Rational, out: Rational) -> Slot {
    if source_frames <= 1 || j <= 0 {
        return Slot::Source(0);
    }
    let last = source_frames - 1;
    // Compare j / out_fps with i / source_fps via cross-multiplication.
    let tick = j * out.den * source.num;
    let unit = source.den * out.num;
    let last_tick = last * unit;
    if tick >= last_tick {
        return Slot::Source(last as usize);
    }
    let index = tick / unit;
    let rel_num = tick - index * unit;
    let rel = rel_num as f64 / unit as f64;
    if rel <= 0.001 {
        Slot::Source(index as usize)
    } else if rel >= 0.999 {
        Slot::Source((index as usize + 1).min(last as usize))
    } else {
        Slot::Interp {
            left: index as usize,
            timestep: rel as f32,
        }
    }
}

pub fn rate_from_fps(fps: f64) -> Rational {
    const NAMED: &[(f64, i64, i64)] = &[
        (23.976, 24000, 1001),
        (24.0, 24, 1),
        (25.0, 25, 1),
        (29.97, 30000, 1001),
        (30.0, 30, 1),
        (48.0, 48, 1),
        (50.0, 50, 1),
        (59.94, 60000, 1001),
        (60.0, 60, 1),
        (90.0, 90, 1),
        (120.0, 120, 1),
    ];
    for (value, num, den) in NAMED {
        if (fps - value).abs() < 0.02 {
            return Rational::new(*num, *den);
        }
    }
    Rational::new(fps.round().max(1.0) as i64, 1)
}

pub fn parse_rate(text: &str) -> Option<Rational> {
    let text = text.trim();
    if text.is_empty() || text == "0/0" || text == "N/A" {
        return None;
    }
    if let Some((num, den)) = text.split_once('/') {
        let num: i64 = num.parse().ok()?;
        let den: i64 = den.parse().ok()?;
        if num <= 0 || den <= 0 {
            return None;
        }
        return Some(Rational::new(num, den));
    }
    let value: f64 = text.parse().ok()?;
    if value > 0.0 {
        Some(rate_from_fps(value))
    } else {
        None
    }
}

fn div_round(num: i64, den: i64) -> i64 {
    if den == 0 {
        return 0;
    }
    if num >= 0 {
        (num + den / 2) / den
    } else {
        (num - den / 2) / den
    }
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_x_from_30_inserts_a_midpoint_and_holds_the_last_frame() {
        let src = Rational::new(30, 1);
        let out = output_rate(src, 2, None);
        assert_eq!(out, Rational::new(60, 1));
        let n = 4;
        let count = output_count(n, src, out);
        assert_eq!(count, 8);
        let slots: Vec<_> = (0..count).map(|j| slot_at(j, n, src, out)).collect();
        assert_eq!(
            slots,
            vec![
                Slot::Source(0),
                Slot::Interp {
                    left: 0,
                    timestep: 0.5
                },
                Slot::Source(1),
                Slot::Interp {
                    left: 1,
                    timestep: 0.5
                },
                Slot::Source(2),
                Slot::Interp {
                    left: 2,
                    timestep: 0.5
                },
                Slot::Source(3),
                Slot::Source(3),
            ]
        );
    }

    #[test]
    #[test]
    fn timestamps_keep_60_to_60_on_source_frames_and_a_long_gap_adds_frames() {
        let pts: Vec<f64> = (0..8).map(|i| i as f64 / 60.0).collect();
        let locked = slots_from_pts(&pts, 60.0);
        assert_eq!(locked.len(), pts.len());
        assert!(locked.iter().all(|slot| matches!(slot, Slot::Source(_))));
        let even = slots_from_pts(&[0.0, 0.1, 0.2, 0.3], 60.0);
        let vfr = slots_from_pts(&[0.0, 0.02, 0.04, 0.5], 60.0);
        let interps = |slots: &[Slot]| {
            slots
                .iter()
                .filter(|slot| matches!(slot, Slot::Interp { .. }))
                .count()
        };
        assert!(interps(&vfr) > interps(&even));
    }

    fn fps_flag_overrides_the_multiplier() {
        let src = Rational::new(24000, 1001);
        let out = output_rate(src, 2, Some(60.0));
        assert_eq!(out, Rational::new(60, 1));
        let doubled = output_rate(src, 2, None);
        assert_eq!(doubled, Rational::new(48000, 1001));
    }
}
