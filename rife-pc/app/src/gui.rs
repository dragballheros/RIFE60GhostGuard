//! Windowed controls matching the iOS toggles, minus Wan VACE.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use eframe::egui;

use crate::pipeline::{self, Options, Progress, Summary};
use crate::settings::{
    AnimeQuality, CorrectionMode, PerformanceTier, ProcessingSettings, QueueItem, RifeModelKind,
    WatermarkModelKind,
};

pub fn run() -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1040.0, 780.0]),
        ..Default::default()
    };
    eframe::run_native(
        "RIFE60 GhostGuard",
        options,
        Box::new(|_cc| Ok(Box::new(App::load()))),
    )
    .map_err(|err| anyhow::anyhow!("GUI failed to start: {err}"))
}

struct App {
    settings: ProcessingSettings,
    input: String,
    output: String,
    multi: i32,
    use_fps: bool,
    fps: f64,
    sensitivity: f64,
    tile: i32,
    gpu: i32,
    tab: Tab,
    running: bool,
    cancel: Option<Arc<AtomicBool>>,
    events: Option<Receiver<UiEvent>>,
    status: String,
    fraction: f32,
    h3_prompt: String,
    h3_frames: u32,
    h3_width: i32,
    h3_height: i32,
    h3_reference: String,
    h3_audio: bool,
    h3_vulkan: bool,
    h3_license: bool,
    studio_prompt: String,
    studio_negative: String,
    studio_width: i32,
    studio_height: i32,
    studio_steps: u32,
    studio_cfg: f64,
    studio_seed: i64,
    studio_outline: bool,
    studio_outline_strength: f64,
    studio_vulkan: bool,
    studio_anima: bool,
    mask_stroke: Vec<[f32; 2]>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Interpolate,
    Picture,
    Scale,
    Watermark,
    Export,
    Queue,
    H3,
    Studio,
    Models,
}

enum UiEvent {
    Progress(Progress),
    Done(Result<Summary, String>),
    Note(String),
}

impl App {
    fn load() -> Self {
        let mut settings = ProcessingSettings::gui();
        let path = settings_path();
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(saved) = serde_json::from_str::<ProcessingSettings>(&text) {
                settings = saved;
            }
        }
        Self {
            settings,
            input: String::new(),
            output: String::new(),
            multi: 2,
            use_fps: true,
            fps: 60.0,
            sensitivity: 1.35,
            tile: 512,
            gpu: -1,
            tab: Tab::Interpolate,
            running: false,
            cancel: None,
            events: None,
            status: "Choose a video. RIFE 4.26 is the iOS model. 4.6 stays available.".into(),
            fraction: 0.0,
            h3_prompt: String::new(),
            h3_frames: 56,
            h3_width: 864,
            h3_height: 480,
            h3_reference: String::new(),
            h3_audio: true,
            h3_vulkan: false,
            h3_license: false,
            studio_prompt: String::new(),
            studio_negative: String::new(),
            studio_width: 832,
            studio_height: 1216,
            studio_steps: 28,
            studio_cfg: 5.0,
            studio_seed: 1,
            studio_outline: true,
            studio_outline_strength: 0.5,
            studio_vulkan: false,
            studio_anima: false,
            mask_stroke: Vec::new(),
        }
    }

    fn save_settings(&self) {
        if let Ok(text) = serde_json::to_string_pretty(&self.settings) {
            let path = settings_path();
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(path, text);
        }
    }

    fn start(&mut self) {
        let queued: Vec<QueueItem> = if self.settings.queue.is_empty() {
            vec![QueueItem::new(self.input.trim())]
        } else {
            self.settings.queue.clone()
        };
        if queued.iter().all(|item| item.path.trim().is_empty() || !Path::new(item.path.trim()).is_file()) {
            self.status = "Pick an input video that exists.".into();
            return;
        }
        self.save_settings();
        let settings = self.settings.clone();
        let multi = self.multi.max(2) as u32;
        let fps = if self.use_fps { Some(self.fps) } else { None };
        let tile = self.tile;
        let gpu = self.gpu;
        let sensitivity = self.sensitivity;
        let output_override = self.output.trim().to_string();
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        self.cancel = Some(Arc::clone(&cancel));
        self.events = Some(rx);
        self.running = true;
        self.fraction = 0.0;
        self.status = "Starting…".into();
        thread::spawn(move || {
            let mut last_summary = None;
            for (index, item) in queued.iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                let input = PathBuf::from(item.path.trim());
                if !input.is_file() {
                    let _ = tx.send(UiEvent::Note(format!("Skipped missing file {}", input.display())));
                    continue;
                }
                let output = if index == 0 && !output_override.is_empty() {
                    PathBuf::from(&output_override)
                } else if !settings.export_dir.is_empty() {
                    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
                    let ext = if settings.video_to_gif { "gif" } else { "mp4" };
                    PathBuf::from(&settings.export_dir).join(format!("{stem}-60.{ext}"))
                } else {
                    PathBuf::from(default_output(&input, settings.video_to_gif))
                };
                let mut item_settings = settings.clone();
                if !item.remove_watermark {
                    item_settings.watermark = false;
                }
                let model = crate::models::rife_dir(item_settings.rife_model);
                let progress_tx = tx.clone();
                let result = pipeline::run(
                    &Options {
                        audio_source: input.clone(),
                        input,
                        output,
                        multi,
                        fps,
                        model,
                        gpu,
                        tile,
                        sensitivity,
                        ffmpeg: None,
                        settings: item_settings,
                    },
                    Arc::clone(&cancel),
                    move |progress| {
                        let _ = progress_tx.send(UiEvent::Progress(progress));
                    },
                );
                match result {
                    Ok(summary) => {
                        for note in &summary.notes {
                            let _ = tx.send(UiEvent::Note(note.clone()));
                        }
                        last_summary = Some(Ok(summary));
                    }
                    Err(err) => {
                        last_summary = Some(Err(err.to_string()));
                        break;
                    }
                }
            }
            let _ = tx.send(UiEvent::Done(last_summary.unwrap_or_else(|| Err("nothing was processed".into()))));
        });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(rx) = &self.events {
            while let Ok(event) = rx.try_recv() {
                match event {
                    UiEvent::Progress(progress) => {
                        self.fraction = if progress.fraction > 0.0 {
                            progress.fraction
                        } else {
                            let total = progress.output_total.max(1) as f32;
                            (progress.output_frame as f32 / total).clamp(0.0, 1.0)
                        };
                        let eta = if progress.eta_seconds > 1.0 {
                            format!("   about {:.0} min left", progress.eta_seconds / 60.0)
                        } else {
                            String::new()
                        };
                        self.status = format!(
                            "{}  {} / {}   {:.1} fps   {} MB free{}",
                            progress.stage,
                            progress.output_frame,
                            progress.output_total,
                            progress.fps,
                            progress.ram_mb,
                            eta
                        );
                    }
                    UiEvent::Note(text) => {
                        self.status = text;
                    }
                    UiEvent::Done(Ok(summary)) => {
                        self.running = false;
                        self.fraction = 1.0;
                        self.status = format!(
                            "Finished {} frames in {:.1}s ({:.1} fps) on {}. Scene boundaries {}, stroke repeats {}, source outliers {}.",
                            summary.output_frames,
                            summary.seconds,
                            summary.fps,
                            summary.gpu_name,
                            summary.scene_boundaries,
                            summary.stroke_repeats,
                            summary.source_outliers
                        );
                        if let Some(path) = &summary.diagnostics_path {
                            self.status.push_str(&format!(" Diagnostics {}", path.display()));
                        }
                    }
                    UiEvent::Done(Err(err)) => {
                        self.running = false;
                        self.status = err;
                    }
                }
            }
        }
        if self.running {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("RIFE60 GhostGuard");
            ui.horizontal_wrapped(|ui| {
                for (tab, label) in [
                    (Tab::Interpolate, "Interpolate"),
                    (Tab::Picture, "Picture"),
                    (Tab::Scale, "Scale"),
                    (Tab::Watermark, "Watermark"),
                    (Tab::Export, "Export"),
                    (Tab::Queue, "Queue"),
                    (Tab::H3, "MiniMax H3"),
                    (Tab::Studio, "AI Image Studio"),
                    (Tab::Models, "Models"),
                ] {
                    ui.selectable_value(&mut self.tab, tab, label);
                }
            });
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| match self.tab {
                Tab::Interpolate => self.interpolate_tab(ui),
                Tab::Picture => self.picture_tab(ui),
                Tab::Scale => self.scale_tab(ui),
                Tab::Watermark => self.watermark_tab(ui),
                Tab::Export => self.export_tab(ui),
                Tab::Queue => self.queue_tab(ui),
                Tab::H3 => self.h3_tab(ui),
                Tab::Studio => self.studio_tab(ui),
                Tab::Models => self.models_tab(ui),
            });
            ui.separator();
            ui.horizontal(|ui| {
                if ui.add_enabled(!self.running, egui::Button::new("Start")).clicked() {
                    self.start();
                }
                if ui.add_enabled(self.running, egui::Button::new("Cancel")).clicked() {
                    if let Some(flag) = &self.cancel {
                        flag.store(true, Ordering::Relaxed);
                    }
                    self.status = "Cancelling…".into();
                }
            });
            ui.add(egui::ProgressBar::new(self.fraction).show_percentage());
            ui.label(&self.status);
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.save_settings();
    }
}

impl App {
    fn interpolate_tab(&mut self, ui: &mut egui::Ui) {
        path_row(ui, "Input", &mut self.input, false, !self.running);
        path_row(ui, "Output", &mut self.output, true, !self.running);
        ui.horizontal(|ui| {
            ui.label("RIFE model");
            ui.selectable_value(&mut self.settings.rife_model, RifeModelKind::V426, "4.26");
            ui.selectable_value(&mut self.settings.rife_model, RifeModelKind::V46, "4.6");
        });
        ui.horizontal(|ui| {
            ui.label("Multiplier");
            ui.add(egui::Slider::new(&mut self.multi, 2..=8).text("×"));
            ui.checkbox(&mut self.use_fps, "Target fps");
            ui.add_enabled(self.use_fps, egui::Slider::new(&mut self.fps, 24.0..=120.0));
        });
        ui.add(egui::Slider::new(&mut self.sensitivity, 0.65..=1.35).text("Ghost sensitivity"));
        ui.checkbox(&mut self.settings.ghost_protection, "Ghost protection");
        ui.checkbox(&mut self.settings.scene_cut_protection, "Scene-cut protection");
        ui.checkbox(&mut self.settings.thin_line, "Thin-line preservation");
        ui.checkbox(&mut self.settings.line_stabilizers, "Line stabilizers (static lock, small motion, stroke width)");
        ui.checkbox(&mut self.settings.source_outlier, "Source outlier guard");
        ui.checkbox(&mut self.settings.render_power, "Render power mode");
        ui.horizontal(|ui| {
            ui.label("Performance tier");
            for tier in [PerformanceTier::Auto, PerformanceTier::Performance, PerformanceTier::Balanced, PerformanceTier::Safe] {
                ui.selectable_value(&mut self.settings.performance_tier, tier, tier.label());
            }
        });
        ui.label("Render power mode keeps Windows awake during the job. On this 64 GB machine Auto picks Performance and keeps the 512 tile, because the Radeon Pro 5300M and 5500M have a 4 GB framebuffer. Safe caps the tile at 256.");
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.tile, 0..=1024).text("Tile (0 = whole frame)"));
            ui.add(egui::Slider::new(&mut self.gpu, -1..=8).text("GPU index"));
        });
    }

    fn picture_tab(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.settings.compression_guard, "Compression Guard");
        toggle_strength(ui, &mut self.settings.color_rebuild, &mut self.settings.color_rebuild_strength, "Color Rebuild");
        toggle_strength(ui, &mut self.settings.outline_rebuild, &mut self.settings.outline_rebuild_strength, "Outline Rebuild");
        ui.checkbox(&mut self.settings.sharpie, "Final Sharpie Outline");
        ui.label("Sharpie uses AnimeSharpieOutlineSubtle.onnx when you convert the iOS Core ML package. Otherwise it darkens existing lines and says so.");
        toggle_strength(ui, &mut self.settings.color_pop, &mut self.settings.color_pop_strength, "Color Pop");
        mode_row(ui, "Automatic warmth", &mut self.settings.warmth);
        mode_row(ui, "Color Pop skin tone", &mut self.settings.skin_tone);
    }

    fn scale_tab(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.settings.upscale, "Real-CUGAN Native 2×");
        ui.add_enabled(self.settings.upscale, egui::Checkbox::new(&mut self.settings.upscale_first, "Upscale before interpolation (faster)"));
        ui.label("Noise level 3, sync gap 3, models-se. A second 2× pass runs when the long edge is under 1080 and 4× stays within 40 megapixels. The iOS intensity of 1.30 is inside the Core ML graph and is not a separate ncnn flag.");
        ui.horizontal(|ui| {
            ui.label("Anime Restore quality");
            for quality in [AnimeQuality::Auto, AnimeQuality::Apisr, AnimeQuality::Span] {
                ui.selectable_value(&mut self.settings.anime_restore_quality, quality, quality.label());
            }
        });
        ui.checkbox(&mut self.settings.anime_restore, "Anime Restore");
        ui.add(egui::Slider::new(&mut self.settings.anime_restore_strength, 0.1..=1.0).text("Anime Restore strength"));
        ui.checkbox(&mut self.settings.anime_restore_suppress_auto, "Do not auto-enable Anime Restore at 720p and below");
        ui.label("Short videos auto-enable Anime Restore unless that box is checked. The APISR and SPAN networks run from ONNX through DirectML when the files are in the model folder. The installer does not contain those weights.");
    }

    fn watermark_tab(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.settings.watermark, "Remove marked watermarks");
        ui.horizontal(|ui| {
            ui.label("Inpainting model");
            for model in [WatermarkModelKind::Lama, WatermarkModelKind::DeepCreamBar, WatermarkModelKind::DeepCreamMosaic] {
                ui.selectable_value(&mut self.settings.watermark_model, model, model.label());
            }
        });
        ui.label("LaMa is MIT (fine-tune) and Apache-2.0 (base). DeepCreamPy source is CC BY-NC-SA 4.0: non-commercial. Without the ONNX file the marked pixels are filled from nearby unmasked pixels, and the status line says so.");
        ui.add(egui::Slider::new(&mut self.settings.watermark_padding, 0.0..=16.0).text("Mask padding"));
        if ui.button("Clear mask").clicked() {
            self.settings.watermark_mask.strokes.clear();
            self.settings.watermark_mask.rects.clear();
        }
        let (response, painter) = ui.allocate_painter(egui::vec2(ui.available_width(), 280.0), egui::Sense::click_and_drag());
        painter.rect_filled(response.rect, 4.0, egui::Color32::from_rgb(24, 24, 28));
        for stroke in &self.settings.watermark_mask.strokes {
            paint_stroke(&painter, response.rect, stroke);
        }
        paint_stroke(&painter, response.rect, &self.mask_stroke);
        if let Some(pointer) = response.interact_pointer_pos() {
            if response.dragged() {
                self.mask_stroke.push(norm(response.rect, pointer));
            }
        }
        if response.drag_stopped() && self.mask_stroke.len() >= 2 {
            self.settings.watermark_mask.strokes.push(std::mem::take(&mut self.mask_stroke));
        }
        ui.label("Drag on the canvas to paint the watermark. Coordinates are saved with the queue and reused for every item whose per-item toggle is on.");
    }

    fn export_tab(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.settings.preserve_audio, "Preserve original audio");
        ui.checkbox(&mut self.settings.reddit_mode, "Reddit Mode (≤20 MB)");
        ui.checkbox(&mut self.settings.video_to_gif, "Convert video to GIF (≤20 MB)");
        ui.checkbox(&mut self.settings.diagnostics, "Broken-Tile Stage Diagnostics");
        ui.horizontal(|ui| {
            ui.label("Export folder");
            ui.add(egui::TextEdit::singleline(&mut self.settings.export_dir).desired_width(420.0));
            if ui.button("Browse…").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    self.settings.export_dir = path.display().to_string();
                }
            }
        });
        ui.label("Audio is stream-copied onto the final HEVC file. GIF export drops audio. Reddit Mode re-encodes toward 19.8 MB. Diagnostics writes a JSON report and up to five seam bitmaps under %LOCALAPPDATA%\\RIFE60GhostGuard\\Diagnostics.");
        if ui.button("Export the latest diagnostics report").clicked() {
            let path = crate::models::data_dir().join("Diagnostics").join("run.json");
            self.status = if path.is_file() {
                format!("Diagnostics report: {}", path.display())
            } else {
                "No diagnostics report yet. Turn the toggle on and finish a render.".into()
            };
        }
    }

    fn queue_tab(&mut self, ui: &mut egui::Ui) {
        if ui.button("Add files…").clicked() && !self.running {
            if let Some(paths) = rfd::FileDialog::new()
                .add_filter("Media", &["mp4", "mov", "mkv", "webm", "m4v", "gif", "png", "jpg", "jpeg", "webp"])
                .pick_files()
            {
                for path in paths {
                    self.settings.queue.push(QueueItem::new(path.display().to_string()));
                }
            }
        }
        let mut remove = None;
        for (index, item) in self.settings.queue.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.checkbox(&mut item.remove_watermark, "Remove watermark on this item");
                ui.label(&item.path);
                if ui.button("Remove").clicked() {
                    remove = Some(index);
                }
            });
        }
        if let Some(index) = remove {
            self.settings.queue.remove(index);
        }
        if self.settings.queue.is_empty() {
            ui.label("The queue is empty. Start uses the Input field. With items here, Start runs them in order and the per-item watermark toggle overrides the global one.");
        }
    }

    fn h3_tab(&mut self, ui: &mut egui::Ui) {
        ui.colored_label(egui::Color32::from_rgb(180, 50, 50), crate::h3::TERRITORY_WARNING);
        ui.hyperlink_to("Review the official MiniMax H3 model license", crate::h3::LICENSE_URL);
        ui.checkbox(&mut self.h3_license, "I have verified that my use is authorized by the license, including separate authorization if my territory is excluded");
        ui.label("Q4_K_M is the quant leejet/MiniMax-H3-GGUF publishes for Ref2VA. Q8_0 is not in that repo. Qwen3-VL Q4_K_M is the text encoder. Weights download on first use and are not in the installer.");
        ui.add(egui::TextEdit::multiline(&mut self.h3_prompt).desired_rows(4).desired_width(ui.available_width()).hint_text("Positive prompt"));
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.h3_width, 256..=1280).text("Width"));
            ui.add(egui::Slider::new(&mut self.h3_height, 256..=768).text("Height"));
            ui.add(egui::Slider::new(&mut self.h3_frames, 8..=121).text("Frames"));
        });
        path_row(ui, "Reference", &mut self.h3_reference, false, !self.running);
        ui.checkbox(&mut self.h3_audio, "Keep generated audio (needs the audio VAE)");
        ui.checkbox(&mut self.h3_vulkan, "Low-VRAM Vulkan offload (otherwise CPU, weights in the 64 GB of RAM)");
        let request = self.h3_request();
        ui.label(request.estimate_label());
        if ui.add_enabled(self.h3_license && !self.running, egui::Button::new("Generate H3")).clicked() {
            self.spawn_h3(request);
        }
        if !self.h3_license {
            ui.label("Accept the license statement before this PC downloads or runs the local H3 model.");
        }
    }

    fn studio_tab(&mut self, ui: &mut egui::Ui) {
        ui.label("Animagine XL v3.1 runs through stable-diffusion.cpp on the CPU, with weights in system RAM. Anima (Cosmos 2.5) matches the iOS picker and is non-commercial; this build does not start it.");
        ui.checkbox(&mut self.studio_anima, "Show the Anima refusal instead of generating");
        ui.add(egui::TextEdit::multiline(&mut self.studio_prompt).desired_rows(3).desired_width(ui.available_width()).hint_text("Positive prompt"));
        ui.add(egui::TextEdit::multiline(&mut self.studio_negative).desired_rows(2).desired_width(ui.available_width()).hint_text("Negative prompt"));
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.studio_width, 512..=1024).text("Width"));
            ui.add(egui::Slider::new(&mut self.studio_height, 512..=1536).text("Height"));
        });
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.studio_steps, 4..=40).text("Steps"));
            ui.add(egui::Slider::new(&mut self.studio_cfg, 1.0..=10.0).text("CFG"));
            ui.add(egui::DragValue::new(&mut self.studio_seed));
            ui.label("Seed");
        });
        ui.checkbox(&mut self.studio_outline, "Outline Rebuild after generation");
        ui.add(egui::Slider::new(&mut self.studio_outline_strength, 0.1..=1.0).text("Outline strength"));
        ui.checkbox(&mut self.studio_vulkan, "Low-VRAM Vulkan offload");
        ui.label(self.studio_request().estimate_label());
        if ui.add_enabled(!self.running, egui::Button::new("Generate image")).clicked() {
            let request = self.studio_request();
            self.spawn_studio(request);
        }
    }

    fn models_tab(&mut self, ui: &mut egui::Ui) {
        ui.label("Large files download into %LOCALAPPDATA%\\RIFE60GhostGuard and are not part of RIFE60GhostGuard-Setup.exe. The installer does include RIFE 4.6, RIFE 4.26, Real-CUGAN noise 3, ffmpeg, and the ONNX Runtime DirectML wrapper.");
        ui.label("Convert the iOS Core ML packages with rife-pc/tools/convert_coreml_to_onnx.py and place the ONNX files in models/onnx. Expected names: APISR2xRRDB, SPAN2xHFA2k, AnimeSharpieOutlineSubtle, AnimeWatermarkLaMa512, DeepCreamBar, DeepCreamMosaic.");
        for item in crate::models::catalog() {
            ui.horizontal(|ui| {
                if ui.button(format!("Download {}", item.label)).clicked() && !self.running {
                    self.spawn_download(item.id);
                }
            });
            ui.label(item.note);
        }
        ui.horizontal(|ui| {
            if ui.button("Download stable-diffusion.cpp CPU build").clicked() && !self.running {
                self.spawn_download("sd-cpu");
            }
            if ui.button("Download Vulkan build").clicked() && !self.running {
                self.spawn_download("sd-vulkan");
            }
        });
        ui.label("The CPU zip is the AVX2-class Windows build (bin-win-cpu-x64). The Vulkan zip is the optional low-VRAM offload. Neither is in the installer.");
    }

    fn h3_request(&self) -> crate::h3::Request {
        crate::h3::Request {
            prompt: self.h3_prompt.clone(),
            width: self.h3_width,
            height: self.h3_height,
            frames: self.h3_frames,
            fps: 24.0,
            cfg: 1.0,
            reference: if self.h3_reference.trim().is_empty() { None } else { Some(PathBuf::from(self.h3_reference.trim())) },
            keep_audio: self.h3_audio,
            vulkan: self.h3_vulkan,
            license_accepted: self.h3_license,
            output: PathBuf::from(if self.output.trim().is_empty() { "h3.mp4" } else { self.output.trim() }),
        }
    }

    fn studio_request(&self) -> crate::studio::Request {
        crate::studio::Request {
            profile: if self.studio_anima { crate::studio::Profile::Anima } else { crate::studio::Profile::Animagine },
            prompt: self.studio_prompt.clone(),
            negative: self.studio_negative.clone(),
            width: self.studio_width / 64 * 64,
            height: self.studio_height / 64 * 64,
            steps: self.studio_steps,
            cfg: self.studio_cfg,
            clip_skip: 2,
            sampler: "euler_a".into(),
            seed: self.studio_seed,
            hires: false,
            hires_scale: 1.5,
            outline: self.studio_outline,
            outline_strength: self.studio_outline_strength,
            vulkan: self.studio_vulkan,
            output: PathBuf::from(if self.output.trim().is_empty() { "studio.png" } else { self.output.trim() }),
        }
    }

    fn spawn_h3(&mut self, request: crate::h3::Request) {
        self.running = true;
        self.status = request.estimate_label();
        let (tx, rx) = mpsc::channel();
        self.events = Some(rx);
        thread::spawn(move || {
            let result = crate::h3::run(&request).map(|_| Summary {
                output_frames: request.frames as u64,
                seconds: 0.0,
                fps: 0.0,
                scene_boundaries: 0,
                stroke_repeats: 0,
                source_outliers: 0,
                gpu_name: "CPU".into(),
                notes: vec!["MiniMax H3 finished.".into()],
                diagnostics_path: None,
            });
            let _ = tx.send(UiEvent::Done(result.map_err(|err| err.to_string())));
        });
    }

    fn spawn_studio(&mut self, request: crate::studio::Request) {
        self.running = true;
        self.status = request.estimate_label();
        let (tx, rx) = mpsc::channel();
        self.events = Some(rx);
        thread::spawn(move || {
            let result = crate::studio::run(&request).map(|path| Summary {
                output_frames: 1,
                seconds: 0.0,
                fps: 0.0,
                scene_boundaries: 0,
                stroke_repeats: 0,
                source_outliers: 0,
                gpu_name: path.display().to_string(),
                notes: vec![format!("Wrote {}", path.display())],
                diagnostics_path: None,
            });
            let _ = tx.send(UiEvent::Done(result.map_err(|err| err.to_string())));
        });
    }

    fn spawn_download(&mut self, id: &'static str) {
        self.running = true;
        self.status = format!("Downloading {id}…");
        let (tx, rx) = mpsc::channel();
        self.events = Some(rx);
        thread::spawn(move || {
            let result = crate::models::download_id(id).map(|path| Summary {
                output_frames: 0,
                seconds: 0.0,
                fps: 0.0,
                scene_boundaries: 0,
                stroke_repeats: 0,
                source_outliers: 0,
                gpu_name: String::new(),
                notes: vec![format!("Downloaded {}", path.display())],
                diagnostics_path: None,
            });
            let _ = tx.send(UiEvent::Done(result.map_err(|err| err.to_string())));
        });
    }
}

fn path_row(ui: &mut egui::Ui, label: &str, text: &mut String, save: bool, enabled: bool) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add_enabled(enabled, egui::TextEdit::singleline(text).desired_width(640.0));
        if ui.add_enabled(enabled, egui::Button::new("Browse…")).clicked() {
            let dialog = rfd::FileDialog::new();
            let picked = if save {
                dialog.add_filter("Video", &["mp4", "mov", "gif", "png"]).save_file()
            } else {
                dialog
                    .add_filter("Media", &["mp4", "mov", "mkv", "webm", "m4v", "gif", "png", "jpg", "jpeg", "webp"])
                    .pick_file()
            };
            if let Some(path) = picked {
                *text = path.display().to_string();
            }
        }
    });
}

fn toggle_strength(ui: &mut egui::Ui, enabled: &mut bool, strength: &mut f64, label: &str) {
    ui.horizontal(|ui| {
        ui.checkbox(enabled, label);
        ui.add_enabled(*enabled, egui::Slider::new(strength, 0.1..=1.0).text("strength"));
    });
}

fn mode_row(ui: &mut egui::Ui, label: &str, mode: &mut CorrectionMode) {
    ui.horizontal(|ui| {
        ui.label(label);
        for value in [CorrectionMode::Off, CorrectionMode::OriginalOnly, CorrectionMode::ProcessedOnly, CorrectionMode::Full] {
            ui.selectable_value(mode, value, value.label());
        }
    });
}

fn paint_stroke(painter: &egui::Painter, rect: egui::Rect, stroke: &[[f32; 2]]) {
    if stroke.len() < 2 {
        return;
    }
    let points: Vec<egui::Pos2> = stroke.iter().map(|p| denorm(rect, *p)).collect();
    painter.add(egui::Shape::line(points, egui::Stroke::new(3.0_f32, egui::Color32::from_rgb(220, 70, 90))));
}

fn norm(rect: egui::Rect, pos: egui::Pos2) -> [f32; 2] {
    [
        ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0),
        ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0),
    ]
}

fn denorm(rect: egui::Rect, p: [f32; 2]) -> egui::Pos2 {
    egui::pos2(rect.min.x + p[0] * rect.width(), rect.min.y + p[1] * rect.height())
}

fn default_output(input: &Path, gif: bool) -> String {
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
    let ext = if gif { "gif" } else { "mp4" };
    input.with_file_name(format!("{stem}-60.{ext}")).display().to_string()
}

fn settings_path() -> PathBuf {
    crate::models::data_dir().join("settings.json")
}

