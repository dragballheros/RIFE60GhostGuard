//! Source and output frame filters, in the iOS order.
//! Anime Restore and watermark removal use ONNX when the file is present.
//! Sharpie does the same. The classical fallbacks record a note.

use std::path::PathBuf;

use rife_guards::Frame;

use crate::color_pop::WarmthState;
use crate::color_rebuild::ColorRebuildState;
use crate::onnx_runtime::Session;
use crate::settings::ProcessingSettings;

pub struct FilterState {
    pub warmth: WarmthState,
    pub color: ColorRebuildState,
    pub notes: Vec<String>,
    anime_skipped: bool,
    sharpie_classical: bool,
    watermark_classical: bool,
    anime_path: Option<PathBuf>,
    anime_session: Option<Session>,
    anime_decided: bool,
}

impl Default for FilterState {
    fn default() -> Self {
        Self {
            warmth: WarmthState::default(),
            color: ColorRebuildState::default(),
            notes: Vec::new(),
            anime_skipped: false,
            sharpie_classical: false,
            watermark_classical: false,
            anime_path: None,
            anime_session: None,
            anime_decided: false,
        }
    }
}

pub fn pre_frame(frame: &Frame, settings: &ProcessingSettings, state: &mut FilterState) -> Frame {
    let mut current = frame.clone();
    if settings.anime_restore_active(frame.width, frame.height) {
        let _ = crate::models::ensure_onnx("APISR2xRRDB.onnx");
        let _ = crate::models::ensure_onnx("SPAN2xHFA2k.onnx");
        let dir = crate::models::onnx_dir();
        if !state.anime_decided {
            state.anime_decided = true;
            match crate::anime_restore::choose(&dir, settings.anime_restore_quality, &current) {
                Ok((path, name)) => match Session::open(&path) {
                    Ok(session) => {
                        state.notes.push(format!("Anime Restore used {name}."));
                        state.anime_session = Some(session);
                        state.anime_path = Some(path);
                    }
                    Err(err) => {
                        state.notes.push(format!("Anime Restore did not run: {err}"));
                        state.anime_skipped = true;
                    }
                },
                Err(err) => {
                    if settings.anime_restore {
                        state.notes.push(format!("Anime Restore did not run: {err}"));
                    } else {
                        state.notes.push(
                            "Anime Restore auto-skipped: the short side is at most 720 and the ONNX model is not installed.".into(),
                        );
                    }
                    state.anime_skipped = true;
                }
            }
        }
        if let Some(session) = &state.anime_session {
            if let Ok(restored) =
                crate::anime_restore::restore_session(&current, session, settings.anime_restore_strength)
            {
                current = restored;
            }
        }
    }
    if settings.watermark && !settings.watermark_mask.is_empty() {
        match crate::watermark::remove(
            &current,
            &settings.watermark_mask,
            settings.watermark_padding,
            settings.watermark_model,
        ) {
            Ok(removal) => {
                if !removal.used_network && !state.watermark_classical {
                    state.notes.push(
                        "Watermark removal used a neighborhood fill. Install the LaMa or DeepCream ONNX file for the neural inpaint. DeepCreamPy is CC BY-NC-SA 4.0 (non-commercial)."
                            .into(),
                    );
                    state.watermark_classical = true;
                }
                current = removal.frame;
            }
            Err(err) => state.notes.push(format!("Watermark removal failed: {err}")),
        }
    }
    if settings.compression_guard {
        current = crate::compression::clean(&current);
    }
    current
}

pub fn post_frame(frame: &Frame, settings: &ProcessingSettings, state: &mut FilterState) -> Frame {
    let mut current = frame.clone();
    if settings.color_rebuild {
        current = crate::color_rebuild::apply(&current, settings.color_rebuild_strength, &mut state.color);
    }
    if settings.outline_rebuild {
        current = crate::outline::rebuild(&current, settings.outline_rebuild_strength);
    }
    if settings.sharpie {
        let model = crate::models::ensure_onnx("AnimeSharpieOutlineSubtle.onnx").ok();
        match crate::outline::sharpie(&current, model.as_deref()) {
            Ok(result) => {
                if !result.used_network && !state.sharpie_classical {
                    state.notes.push(
                        "Final Sharpie Outline used the classical line emphasis because AnimeSharpieOutlineSubtle.onnx was missing or failed to load."
                            .into(),
                    );
                    state.sharpie_classical = true;
                }
                current = result.frame;
            }
            Err(err) => state.notes.push(format!("Sharpie failed: {err}")),
        }
    }
    if settings.color_pop || settings.warmth != crate::settings::CorrectionMode::Off || settings.skin_tone != crate::settings::CorrectionMode::Off
    {
        let strength = if settings.color_pop { settings.color_pop_strength as f32 } else { 0.0 };
        current = crate::color_pop::apply(&current, strength, settings.warmth, settings.skin_tone, &mut state.warmth);
    }
    current
}
