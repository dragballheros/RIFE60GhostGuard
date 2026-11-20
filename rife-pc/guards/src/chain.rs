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

pub fn polish_inbetween(
    previous: &Frame,
    interpolated: &Frame,
    next: &Frame,
    progress: f64,
    sensitivity: f64,
    previous_context: Option<&Frame>,
) -> (Frame, PolishStats) {
    let mut stats = PolishStats {
        scene_boundary: false,
        thin_line: false,
        static_lock: false,
        small_motion: false,
        stroke_repeat: false,
    };
    let ghost = GhostGuard::new(sensitivity, true);
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
    let mut frame = interpolated.clone();
    let thin = ThinLinePreservationPass::new(sensitivity);
    if let Some(restored) = thin.preserve(previous, &frame, next, progress) {
        frame = restored;
        stats.thin_line = true;
    }
    let lock = StaticRegionLock::new(sensitivity);
    if let Some(locked) = lock.lock(previous, &frame, next, progress) {
        frame = locked;
        stats.static_lock = true;
    }
    let stabilizer = SmallMotionLineStabilizer::new(sensitivity);
    if let Some(stable) = stabilizer.stabilize(previous, &frame, next, progress) {
        frame = stable;
        stats.small_motion = true;
    }
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
