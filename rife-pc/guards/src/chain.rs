//! Applies the guards in the order the iOS pipeline uses them, then the
//! stroke-width rule. Scene boundaries repeat a real frame. Quality findings
//! keep the interpolation.

use crate::frame::Frame;
use crate::ghost::{endpoint_disposition, Disposition, GhostGuard};
use crate::outlier::SourceFrameOutlierGuard;
use crate::small_motion::SmallMotionLineStabilizer;
use crate::static_lock::StaticRegionLock;
use crate::stroke::{stroke_width_decision, StrokeDecision};
use crate::thin_line::ThinLinePreservationPass;

#[derive(Clone, Debug)]
pub struct PolishStats {
    pub scene_boundary: bool,
    pub thin_line: bool,
    pub static_lock: bool,
    pub small_motion: bool,
    pub stroke_repeat: bool,
}

/// Which guards run. The iOS toggles map onto `ghost` and `scene_cuts`.
/// The other flags are the rest of the Windows quality chain.
#[derive(Clone, Debug)]
pub struct GuardFlags {
    pub ghost: bool,
    pub scene_cuts: bool,
    pub thin_line: bool,
    pub static_lock: bool,
    pub small_motion: bool,
    pub stroke: bool,
}

impl Default for GuardFlags {
    fn default() -> Self {
        Self {
            ghost: true,
            scene_cuts: true,
            thin_line: true,
            static_lock: true,
            small_motion: true,
            stroke: true,
        }
    }
}

pub fn polish_inbetween(
    previous: &Frame,
    interpolated: &Frame,
    next: &Frame,
    progress: f64,
    sensitivity: f64,
    previous_context: Option<&Frame>,
) -> (Frame, PolishStats) {
    polish_inbetween_with(
        previous,
        interpolated,
        next,
        progress,
        sensitivity,
        previous_context,
        &GuardFlags::default(),
    )
}

pub fn polish_inbetween_with(
    previous: &Frame,
    interpolated: &Frame,
    next: &Frame,
    progress: f64,
    sensitivity: f64,
    previous_context: Option<&Frame>,
    flags: &GuardFlags,
) -> (Frame, PolishStats) {
    let mut stats = PolishStats {
        scene_boundary: false,
        thin_line: false,
        static_lock: false,
        small_motion: false,
        stroke_repeat: false,
    };
    if flags.ghost {
        let ghost = GhostGuard::new(sensitivity, flags.scene_cuts);
        let check = ghost.inspect(previous, interpolated, next, previous_context);
        if check.reject {
            match endpoint_disposition(&check.reason, progress) {
                Disposition::UsePreviousSource => {
                    stats.scene_boundary = true;
                    return (previous.clone(), stats);
                }
                Disposition::UseCurrentSource => {
                    stats.scene_boundary = true;
                    return (next.clone(), stats);
                }
                Disposition::RetainInterpolation => {}
            }
        }
    }
    let mut frame = interpolated.clone();
    if flags.thin_line {
        let thin = ThinLinePreservationPass::new(sensitivity);
        if let Some(restored) = thin.preserve(previous, &frame, next, progress) {
            frame = restored;
            stats.thin_line = true;
        }
    }
    if flags.static_lock {
        let lock = StaticRegionLock::new(sensitivity);
        if let Some(locked) = lock.lock(previous, &frame, next, progress) {
            frame = locked;
            stats.static_lock = true;
        }
    }
    if flags.small_motion {
        let stabilizer = SmallMotionLineStabilizer::new(sensitivity);
        if let Some(stable) = stabilizer.stabilize(previous, &frame, next, progress) {
            frame = stable;
            stats.small_motion = true;
        }
    }
    if flags.stroke {
        if let StrokeDecision::RepeatNearer { nearer_is_next } =
            stroke_width_decision(previous, &frame, next, progress, sensitivity)
        {
            frame = if nearer_is_next {
                next.clone()
            } else {
                previous.clone()
            };
            stats.stroke_repeat = true;
        }
    }
    (frame, stats)
}

/// Replace a stray source frame before RIFE sees it.
/// A cross-fade cut repeats the nearer real neighbor. A flash or detour is
/// replaced with a bridge of the two neighbors.
pub fn clean_source_outlier(
    previous: &Frame,
    candidate: &Frame,
    next: &Frame,
    progress: f64,
    sensitivity: f64,
) -> Frame {
    let mut guard = SourceFrameOutlierGuard::default();
    guard.sensitivity = sensitivity;
    let check = guard.inspect(previous, candidate, next);
    if !check.is_outlier {
        return candidate.clone();
    }
    if check.reason == "stray frame at scene cut" {
        if guard.nearer_is_next(previous, candidate, next) {
            next.clone()
        } else {
            previous.clone()
        }
    } else {
        guard
            .bridge(previous, next, progress)
            .unwrap_or_else(|| candidate.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_guards_keep_the_interpolated_frame() {
        let previous = Frame::new(8, 8, false);
        let mut mid = Frame::new(8, 8, false);
        mid.data.fill(40);
        let flags = GuardFlags {
            ghost: false,
            scene_cuts: false,
            thin_line: false,
            static_lock: false,
            small_motion: false,
            stroke: false,
        };
        let (frame, stats) =
            polish_inbetween_with(&previous, &mid, &previous, 0.5, 1.35, None, &flags);
        assert_eq!(frame.data, mid.data);
        assert!(!stats.scene_boundary);
        assert!(!stats.stroke_repeat);
    }
}
