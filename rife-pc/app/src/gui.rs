//! Small window: pick a file, a multiplier, and Start.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use eframe::egui;

use crate::pipeline::{self, Options, Progress, Summary};

pub fn run() -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([560.0, 360.0]),
        ..Default::default()
    };
    eframe::run_native(
        "RIFE60 GhostGuard",
        options,
        Box::new(|_cc| Box::new(App::default())),
    )
    .map_err(|err| anyhow::anyhow!(err))
}

struct App {
    input: String,
    output: String,
    multi: i32,
    use_fps: bool,
    fps: f64,
    running: bool,
    cancel: Option<Arc<AtomicBool>>,
    events: Option<Receiver<UiEvent>>,
    status: String,
    fraction: f32,
}

impl Default for App {
    fn default() -> Self {
        Self {
            input: String::new(),
            output: String::new(),
            multi: 2,
            use_fps: false,
            fps: 60.0,
            running: false,
            cancel: None,
            events: None,
            status: "Choose a video.".into(),
            fraction: 0.0,
        }
    }
}

enum UiEvent {
    Progress(Progress),
    Done(Result<Summary, String>),
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(rx) = &self.events {
            while let Ok(event) = rx.try_recv() {
                match event {
                    UiEvent::Progress(progress) => {
                        let total = progress.output_total.max(1) as f32;
                        self.fraction = (progress.output_frame as f32 / total).clamp(0.0, 1.0);
                        self.status = format!(
                            "{} / {} frames   {:.1} fps",
                            progress.output_frame, progress.output_total, progress.fps
                        );
                    }
                    UiEvent::Done(Ok(summary)) => {
                        self.running = false;
                        self.fraction = 1.0;
                        self.status = format!(
                            "Finished {} frames in {:.1}s ({:.1} fps). Scene boundaries {}, stroke repeats {}, source outliers {}.",
                            summary.output_frames, summary.seconds, summary.fps, summary.scene_boundaries, summary.stroke_repeats, summary.source_outliers
                        );
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
            ui.label("2× (or target fps) RIFE interpolation on the AMD GPU, then the iOS quality guards and 10-bit HEVC.");
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label("Input");
                ui.add(egui::TextEdit::singleline(&mut self.input).desired_width(360.0));
                if ui.button("Browse…").clicked() && !self.running {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Video", &["mp4", "mov", "mkv", "webm", "m4v"])
                        .pick_file()
                    {
                        if self.output.is_empty() {
                            self.output = default_output(&path);
                        }
                        self.input = path.display().to_string();
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("Output");
                ui.add(egui::TextEdit::singleline(&mut self.output).desired_width(360.0));
                if ui.button("Browse…").clicked() && !self.running {
                    if let Some(path) = rfd::FileDialog::new().add_filter("MP4", &["mp4"]).save_file() {
                        self.output = path.display().to_string();
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("Multiplier");
                ui.add(egui::Slider::new(&mut self.multi, 2..=8).text("×"));
                ui.checkbox(&mut self.use_fps, "Target fps");
                ui.add_enabled(self.use_fps, egui::Slider::new(&mut self.fps, 24.0..=120.0));
            });
            ui.add_space(8.0);
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
            ui.add_space(12.0);
            ui.add(egui::ProgressBar::new(self.fraction).show_percentage());
            ui.label(&self.status);
        });
    }
}

impl App {
    fn start(&mut self) {
        let input = PathBuf::from(self.input.trim());
        let output = PathBuf::from(self.output.trim());
        if !input.is_file() {
            self.status = "Pick an input video that exists.".into();
            return;
        }
        if output.as_os_str().is_empty() {
            self.status = "Pick an output path.".into();
            return;
        }
        let model = default_model();
        let options = Options {
            input,
            output,
            multi: self.multi.max(2) as u32,
            fps: if self.use_fps { Some(self.fps) } else { None },
            model,
            gpu: -1,
            tile: 512,
            sensitivity: 1.35,
            ffmpeg: None,
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        self.cancel = Some(Arc::clone(&cancel));
        self.events = Some(rx);
        self.running = true;
        self.fraction = 0.0;
        self.status = "Starting…".into();
        thread::spawn(move || {
            let progress_tx = tx.clone();
            let result = pipeline::run(&options, cancel, move |progress| {
                let _ = progress_tx.send(UiEvent::Progress(progress));
            });
            let _ = tx.send(UiEvent::Done(result.map_err(|err| err.to_string())));
        });
    }
}

fn default_output(input: &std::path::Path) -> String {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    input
        .with_file_name(format!("{stem}-60.mp4"))
        .display()
        .to_string()
}

pub fn default_model() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let bundled = dir.join("models").join("rife-v4.6");
            if bundled.join("flownet.param").is_file() {
                return bundled;
            }
        }
    }
    PathBuf::from("models").join("rife-v4.6")
}
