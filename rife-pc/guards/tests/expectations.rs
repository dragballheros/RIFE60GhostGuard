//! Ports `Scripts/check_thin_line_preservation.swift`.

use rife_guards::*;

fn require(condition: bool, message: &str) {
    if !condition {
        panic!("THIN_LINE_CHECK_FAILED: {message}");
    }
}

fn make_frame(
    width: i32,
    height: i32,
    fill: u8,
    column: Option<i32>,
    column_value: u8,
    block: Option<(i32, i32, i32, i32)>,
    block_value: u8,
) -> Frame {
    let mut frame = Frame::new(width, height, false);
    for y in 0..height {
        for x in 0..width {
            let mut value = fill;
            if let Some(column) = column {
                if x == column && y >= 8 && y < height - 8 {
                    value = column_value;
                }
            }
            if let Some((bx, by, bw, bh)) = block {
                if x >= bx && x < bx + bw && y >= by && y < by + bh {
                    value = block_value;
                }
            }
            frame.set_rgb(x, y, value);
        }
    }
    frame
}

fn paint_pixel(frame: &mut Frame, x: i32, y: i32, value: u8) {
    frame.set_rgb(x, y, value);
}

fn shape_frame(width: i32, height: i32, rect: (i32, i32, i32, i32)) -> Frame {
    let mut frame = Frame::new(width, height, false);
    let (rx, ry, rw, rh) = rect;
    for y in 0..height {
        for x in 0..width {
            let inside = x >= rx && x < rx + rw && y >= ry && y < ry + rh;
            let i = (y as usize) * frame.stride() + (x as usize) * 4;
            if inside {
                frame.data[i] = 140;
                frame.data[i + 1] = 150;
                frame.data[i + 2] = 180;
            } else {
                frame.data[i] = 42;
                frame.data[i + 1] = 36;
                frame.data[i + 2] = 40;
            }
            frame.data[i + 3] = 255;
        }
    }
    frame
}

fn blend_frames(a: &Frame, b: &Frame) -> Frame {
    let mut out = Frame::new(a.width, a.height, false);
    for i in 0..out.data.len() {
        out.data[i] = ((a.data[i] as u16 + b.data[i] as u16) / 2) as u8;
    }
    out
}

fn line_frame(width: i32, height: i32, column: i32, extra: Option<i32>) -> Frame {
    let mut frame = Frame::new(width, height, false);
    for y in 0..height {
        for x in 0..width {
            let mut value = 200u8;
            if y >= 8 && y < height - 8 && (x - column).abs() <= 1 {
                value = 16;
            }
            if let Some(extra) = extra {
                if y >= 8 && y < height - 8 && (x - extra).abs() <= 1 {
                    value = 16;
                }
            }
            frame.set_rgb(x, y, value);
        }
    }
    frame
}

fn paint_square(frame: &mut Frame, rect: (i32, i32, i32, i32), value: u8) {
    let (rx, ry, rw, rh) = rect;
    for y in 0..frame.height {
        if (ry..ry + rh).contains(&y) {
            for x in 0..frame.width {
                if (rx..rx + rw).contains(&x) {
                    frame.set_rgb(x, y, value);
                }
            }
        }
    }
}

fn pan_frame(width: i32, height: i32, shift: i32, soft: bool) -> Frame {
    let mut frame = Frame::new(width, height, false);
    let columns = [12, 28, 44, 60];
    for y in 0..height {
        for x in 0..width {
            let mut value = 210u8;
            if y >= 4 && y < height - 4 && columns.iter().any(|c| (x - (c + shift)).abs() == 0) {
                value = if soft { 120 } else { 18 };
            }
            frame.set_rgb(x, y, value);
        }
    }
    frame
}

#[test]
fn thin_line_preservation_matches_swift_script() {
    let pass = ThinLinePreservationPass::new(1.35);
    let width = 96;
    let height = 80;
    let previous = make_frame(width, height, 210, Some(40), 20, None, 15);
    let next = make_frame(width, height, 210, Some(40), 20, None, 15);
    let faded = make_frame(width, height, 210, Some(40), 170, None, 15);
    let repaired = pass.preserve(&previous, &faded, &next, 0.5);
    require(
        repaired.is_some(),
        "a static hair line missing from the in-between should be restored",
    );
    let repaired = repaired.unwrap();
    require(
        repaired.gray(40, 40) < 40,
        "the restored stroke should be dark again",
    );
    require(
        repaired.gray(40, 40) <= previous.gray(40, 40).saturating_add(2),
        "the stroke should come back from a real source",
    );
    require(
        repaired.gray(41, 40) >= 200,
        "the restore must not darken the neighbor into a wider stroke",
    );
    require(
        repaired.gray(48, 40) == 210,
        "pixels outside the 2px feather must stay untouched",
    );
    require(
        repaired.gray(10, 10) == faded.gray(10, 10),
        "the rest of the frame must not be replaced",
    );

    let slightly_different_next = make_frame(width, height, 210, Some(40), 60, None, 15);
    let faded_static = make_frame(width, height, 210, Some(40), 170, None, 15);
    let before = pass.preserve(&previous, &faded_static, &slightly_different_next, 0.49);
    let after = pass.preserve(&previous, &faded_static, &slightly_different_next, 0.51);
    require(
        before.is_some() && after.is_some(),
        "a shared static line should remain restored on both sides of the midpoint",
    );
    let before = before.unwrap();
    let after = after.unwrap();
    require(
        (before.gray(40, 40) as i32 - after.gray(40, 40) as i32).abs() <= 2,
        "line restoration must blend source colors continuously through t=0.5",
    );

    let soft = make_frame(width, height, 210, Some(40), 30, None, 15);
    require(
        pass.preserve(&previous, &soft, &next, 0.5).is_none(),
        "a stroke that is only slightly softer than the sources must be left alone",
    );

    let shifted_next = make_frame(width, height, 210, Some(41), 20, None, 15);
    require(
        pass.preserve(&previous, &faded, &shifted_next, 0.25)
            .is_none(),
        "a one-pixel moving contour must not be pinned to either source endpoint",
    );
    let moved_next = make_frame(width, height, 210, Some(46), 20, None, 15);
    require(
        pass.preserve(&previous, &faded, &moved_next, 0.5).is_none(),
        "a line that moved several pixels must stay under RIFE, not be pinned in place",
    );

    let block = Some((24, 16, 40, 40));
    let dark_previous = make_frame(width, height, 210, None, 20, block, 12);
    let dark_next = make_frame(width, height, 210, None, 20, block, 12);
    let lifted = make_frame(width, height, 210, None, 20, block, 160);
    let block_result = pass.preserve(&dark_previous, &lifted, &dark_next, 0.5);
    let block_center = block_result
        .as_ref()
        .map(|f| f.gray(44, 36))
        .unwrap_or_else(|| lifted.gray(44, 36));
    require(
        block_center == 160,
        "the inside of a large dark region is not a thin stroke and must not be repainted",
    );

    let cut_previous = make_frame(width, height, 220, Some(40), 20, None, 15);
    let cut_next = make_frame(width, height, 25, Some(40), 5, None, 15);
    require(
        pass.preserve(&cut_previous, &faded, &cut_next, 0.5)
            .is_none(),
        "a scene cut must skip the pass",
    );

    let mut contour_a = make_frame(width, height, 210, Some(40), 20, None, 15);
    let mut contour_c = make_frame(width, height, 210, Some(40), 20, None, 15);
    let contour_mid = make_frame(width, height, 210, Some(40), 170, None, 15);
    for (x, y) in [(70, 25), (76, 49)] {
        paint_pixel(&mut contour_a, x, y, 20);
        paint_pixel(&mut contour_c, x, y, 20);
    }
    let contour_repair = pass.preserve(&contour_a, &contour_mid, &contour_c, 0.5);
    require(
        contour_repair.is_some(),
        "a valid connected contour should still be repaired when unrelated specks exist",
    );
    let contour_repair = contour_repair.unwrap();
    require(
        contour_repair.gray(40, 40) < 40,
        "the connected one-pixel-wide contour must remain intact",
    );
    require(
        contour_repair.gray(70, 25) == 210 && contour_repair.gray(76, 49) == 210,
        "disconnected one-pixel source specks must not be restored as moving outline dots",
    );

    let mut tiny_previous = make_frame(width, height, 210, None, 20, None, 15);
    let mut tiny_next = make_frame(width, height, 210, None, 20, None, 15);
    let mut tiny_mid = make_frame(width, height, 210, None, 20, None, 15);
    for y in 30..36 {
        paint_pixel(&mut tiny_previous, 40, y, 20);
        paint_pixel(&mut tiny_next, 40, y, 20);
        paint_pixel(&mut tiny_mid, 40, y, 180);
    }
    require(
        pass.preserve(&tiny_previous, &tiny_mid, &tiny_next, 0.5)
            .is_none(),
        "a handful of speck pixels is below the agreement floor",
    );

    let borderline = make_frame(width, height, 210, Some(40), 48, None, 15);
    let strict = ThinLinePreservationPass::new(1.35);
    let loose = ThinLinePreservationPass::new(0.65);
    require(
        strict
            .preserve(&previous, &borderline, &next, 0.5)
            .is_some(),
        "default sensitivity should restore a clearly faded stroke",
    );
    require(
        loose.preserve(&previous, &borderline, &next, 0.5).is_none(),
        "lower sensitivity should leave a milder fade alone",
    );

    let left = shape_frame(160, 90, (16, 20, 32, 32));
    let right = shape_frame(160, 90, (108, 24, 36, 28));
    let shifted = shape_frame(160, 90, (21, 20, 32, 32));
    let different = assess_structure(&left, &right, 1.35, None).unwrap();
    require(
        different.structural_cut,
        "different shapes in the same palette should be a structural cut",
    );
    require(
        !different.classic_cut,
        "that cut should stay under the luma/RGB MAD gate",
    );
    require(
        different.appearance > 0.05 && different.appearance < 0.14,
        &format!(
            "appearance change should match the measured similar-palette range ({})",
            different.appearance
        ),
    );
    require(
        different.overlap.unwrap_or(1.0) < 0.30,
        &format!(
            "1px dilation should put a different layout under the 0.30 Dice cut ({:?})",
            different.overlap
        ),
    );
    let one = shape_frame(160, 90, (17, 20, 32, 32));
    let motion = assess_structure(&left, &one, 1.35, None).unwrap();
    require(
        motion.overlap.unwrap_or(0.0) > 0.90 && !motion.is_cut(),
        &format!(
            "a 1px shift should keep Dice near 1, like ordinary motion ({:?})",
            motion.overlap
        ),
    );
    let shifted_cut = assess_structure(&left, &shifted, 1.35, None).unwrap();
    require(
        !shifted_cut.is_cut(),
        "the same shape shifted a few pixels is motion, not a cut",
    );

    let blended = blend_frames(&left, &right);
    let ghost = GhostGuard::new(1.35, true);
    let ghost_cut = ghost.inspect(&left, &blended, &right, None);
    require(
        ghost_cut.reject && ghost_cut.reason == "scene cut / transition discontinuity",
        &format!(
            "GhostGuard should reject the blended midframe of a same-palette cut ({})",
            ghost_cut.reason
        ),
    );
    let ghost_motion = ghost.inspect(&left, &left, &shifted, None);
    require(
        !ghost_motion.reject,
        &format!(
            "a few pixels of motion should not be rejected as a cut ({})",
            ghost_motion.reason
        ),
    );

    let source_guard = SourceFrameOutlierGuard::default();
    let stray = source_guard.inspect(&left, &blended, &right);
    require(
        stray.is_outlier && stray.reason == "stray frame at scene cut",
        &format!(
            "a cross-fade parked on a structural cut should not be bridged ({})",
            stray.reason
        ),
    );
    let owned = source_guard.inspect(&left, &left, &right);
    require(
        !owned.is_outlier,
        "a frame that belongs to the outgoing shot must stay",
    );
    require(
        pass.preserve(&left, &blended, &right, 0.5).is_none(),
        "thin-line repair must not run across the structural cut",
    );

    let stable = line_frame(160, 90, 40, None);
    require(
        !ghost.inspect(&stable, &stable, &stable, None).reject,
        "an outline that is the same in all three frames is not warped",
    );
    let novel = line_frame(160, 90, 110, None);
    let novel_guard = ghost.inspect(&stable, &novel, &stable, None);
    require(
        novel_guard.reject && novel_guard.reason == "warped/doubled outline",
        &format!(
            "an outline present in neither source should be rejected ({})",
            novel_guard.reason
        ),
    );
    let start = line_frame(160, 90, 24, None);
    let end = line_frame(160, 90, 40, None);
    let between = line_frame(160, 90, 32, None);
    let moving = ghost.inspect(&start, &between, &end, None);
    require(
        !moving.reject,
        &format!(
            "a line halfway between the two source positions is legitimate motion ({})",
            moving.reason
        ),
    );
    let doubled = line_frame(160, 90, 40, Some(58));
    let doubled_guard = ghost.inspect(&stable, &doubled, &stable, None);
    require(
        doubled_guard.reject && doubled_guard.reason == "warped/doubled outline",
        &format!(
            "a second outline next to the real one should be rejected ({})",
            doubled_guard.reason
        ),
    );

    let mut static_a = line_frame(160, 90, 80, None);
    let mut static_c = line_frame(160, 90, 80, None);
    paint_square(&mut static_a, (8, 12, 16, 16), 30);
    paint_square(&mut static_c, (128, 12, 16, 16), 30);
    let mut dragged = line_frame(160, 90, 96, None);
    paint_square(&mut dragged, (60, 12, 16, 16), 30);
    let lock = StaticRegionLock::new(1.35);
    let locked = lock.lock(&static_a, &dragged, &static_c, 0.25);
    require(
        locked.is_some(),
        "static floor lines should be locked back to the sources",
    );
    let locked = locked.unwrap();
    require(
        locked.gray(80, 40) < 40,
        "the real static line should come back from the nearer source",
    );
    require(
        locked.gray(96, 40) > 150,
        "the dragged copy of that line should be removed",
    );
    require(
        locked.gray(68, 20) == 30,
        "a solid moving body between the two poses must stay",
    );
    require(
        lock.lock(&left, &blended, &right, 0.5).is_none(),
        "static lock must not run across a detected cut",
    );

    let mut slightly_changed = line_frame(160, 90, 80, None);
    paint_square(&mut slightly_changed, (79, 8, 3, 74), 22);
    let static_mid = line_frame(160, 90, 80, None);
    let locked_before = lock.lock(&static_a, &static_mid, &slightly_changed, 0.49);
    let locked_after = lock.lock(&static_a, &static_mid, &slightly_changed, 0.51);
    require(
        locked_before.is_some() && locked_after.is_some(),
        "static region lock should preserve an agreeing background region",
    );
    require(
        (locked_before.unwrap().gray(80, 40) as i32 - locked_after.unwrap().gray(80, 40) as i32)
            .abs()
            <= 1,
        "static region repair must not switch source frames at the midpoint",
    );

    let pan_a = pan_frame(128, 72, 0, false);
    let pan_c = pan_frame(128, 72, 3, false);
    let pan_m = pan_frame(128, 72, 0, true);
    let stabilizer = SmallMotionLineStabilizer::new(1.35);
    let panned = stabilizer.stabilize(&pan_a, &pan_m, &pan_c, 1.0 / 3.0);
    require(
        panned.is_some(),
        "a 3px pan of thin lines should be stabilized",
    );
    let panned = panned.unwrap();
    require(
        panned.gray(13, 30) < 40,
        &format!(
            "the line should be crisp at the 1px in-between position ({})",
            panned.gray(13, 30)
        ),
    );
    require(
        panned.gray(12, 30) > 160,
        &format!(
            "the soft unshifted line should not remain ({})",
            panned.gray(12, 30)
        ),
    );
    let pan_before = stabilizer.stabilize(&pan_a, &pan_m, &pan_c, 0.49).unwrap();
    let pan_after = stabilizer.stabilize(&pan_a, &pan_m, &pan_c, 0.51).unwrap();
    require(
        (pan_before.gray(13, 30) as i32 - pan_after.gray(13, 30) as i32).abs() <= 16,
        "motion-compensated line placement must not snap when t crosses 0.5",
    );
    let far = pan_frame(128, 72, 24, false);
    require(
        stabilizer.stabilize(&pan_a, &pan_m, &far, 0.5).is_none(),
        "a large pan must be left to RIFE",
    );
    let thin_stroke = make_frame(96, 80, 210, Some(50), 18, None, 15);
    let mut thick_stroke = make_frame(96, 80, 210, Some(50), 18, None, 15);
    paint_square(&mut thick_stroke, (49, 8, 1, 64), 18);
    paint_square(&mut thick_stroke, (51, 8, 1, 64), 18);
    require(
        stabilizer.stabilize(&thin_stroke, &thick_stroke, &thin_stroke, 0.5).is_none(),
        "without a verified motion field, line-width repair must leave the RIFE frame alone rather than snap to an endpoint",
    );

    let reasons = [
        "warped/doubled outline",
        "edge collapse / smeared midframe",
        "temporal outlier",
        "double-edge/ghost energy",
        "midframe inconsistency",
        "localized color anomaly",
        "fast-motion artifact rejection",
        "RIFE midpoint recovery rejected",
        "unrecognized future quality reason",
    ];
    for reason in reasons {
        for progress in [0.49, 0.50, 0.51] {
            require(
                endpoint_disposition(reason, progress) == Disposition::RetainInterpolation,
                &format!("{reason} at t={progress} must retain an interpolated frame, not snap to a source endpoint"),
            );
        }
    }
    require(
        endpoint_disposition("scene cut / transition discontinuity", 0.49)
            == Disposition::UsePreviousSource,
        "a confirmed cut just before the midpoint may select the previous source",
    );
    require(
        endpoint_disposition("scene cut / transition discontinuity", 0.51)
            == Disposition::UseCurrentSource,
        "a confirmed cut just after the midpoint may select the current source",
    );
    require(
        endpoint_disposition("stray frame at scene cut", 0.49) == Disposition::UsePreviousSource,
        "a stray cut frame must use the temporally nearer real source",
    );
    require(
        endpoint_disposition("isolated scene-end frame", 0.49) == Disposition::UseCurrentSource
            && endpoint_disposition("isolated scene-end frame", 0.51)
                == Disposition::UseCurrentSource,
        "an isolated prior scene-end source must be replaced by the current contextual source",
    );
}

#[test]
fn sensitivity_thresholds_match_swift_comments() {
    let overlap_065 = overlap_limit(0.65);
    let overlap_135 = overlap_limit(1.35);
    assert!((overlap_065 - 0.22).abs() < 1e-9, "{overlap_065}");
    assert!((overlap_135 - 0.30).abs() < 1e-9, "{overlap_135}");
    let floor_065 = appearance_floor(0.65);
    let floor_135 = appearance_floor(1.35);
    assert!((floor_065 - 0.058).abs() < 1e-9, "{floor_065}");
    assert!((floor_135 - 0.050).abs() < 1e-9, "{floor_135}");
}

#[test]
fn stroke_width_repeats_nearer_frame_when_wider_than_20_percent() {
    let thin = make_frame(96, 80, 210, Some(50), 18, None, 15);
    let mut thick = make_frame(96, 80, 210, Some(50), 18, None, 15);
    paint_square(&mut thick, (49, 8, 1, 64), 18);
    paint_square(&mut thick, (51, 8, 1, 64), 18);
    let decision = stroke_width_decision(&thin, &thick, &thin, 0.5, 1.35);
    assert_eq!(
        decision,
        StrokeDecision::RepeatNearer {
            nearer_is_next: false
        },
        "a stroke more than 20% wider than the nearer source repeats that real frame"
    );
    let same = stroke_width_decision(&thin, &thin, &thin, 0.5, 1.35);
    assert_eq!(same, StrokeDecision::Keep);
}

#[test]
fn stroke_width_leaves_unaligned_lines_alone_without_motion() {
    let source = make_frame(96, 80, 210, Some(20), 18, None, 15);
    let elsewhere = make_frame(96, 80, 210, Some(70), 18, None, 15);
    let decision = stroke_width_decision(&source, &elsewhere, &source, 0.5, 1.35);
    assert_eq!(decision, StrokeDecision::Keep);
}

#[test]
fn stroke_width_rejects_a_darker_stroke() {
    let source = make_frame(96, 80, 210, Some(50), 40, None, 15);
    let darker = make_frame(96, 80, 210, Some(50), 4, None, 15);
    let decision = stroke_width_decision(&source, &darker, &source, 0.4, 1.35);
    assert_eq!(
        decision,
        StrokeDecision::RepeatNearer {
            nearer_is_next: false
        }
    );
    let after = stroke_width_decision(&source, &darker, &source, 0.6, 1.35);
    assert_eq!(
        after,
        StrokeDecision::RepeatNearer {
            nearer_is_next: true
        }
    );
}
