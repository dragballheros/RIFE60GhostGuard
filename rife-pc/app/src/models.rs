//! Where models live, and the downloader that fetches them on first use.
//! Multi-gigabyte weights are never copied into the installer.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

use anyhow::{anyhow, Result};

use crate::settings::RifeModelKind;

pub fn install_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|dir| dir.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn data_dir() -> PathBuf {
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        if !local.is_empty() {
            return PathBuf::from(local).join("RIFE60GhostGuard");
        }
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        if !home.is_empty() {
            return PathBuf::from(home).join("AppData").join("Local").join("RIFE60GhostGuard");
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".rife60ghostguard");
    }
    std::env::temp_dir().join("RIFE60GhostGuard")
}

pub fn find_tool(name: &str) -> PathBuf {
    let beside = install_dir().join(name);
    if beside.exists() {
        return beside;
    }
    let downloaded = data_dir().join(name);
    if downloaded.exists() {
        return downloaded;
    }
    PathBuf::from(name)
}

pub fn rife_dir(kind: RifeModelKind) -> PathBuf {
    let name = kind.dir_name();
    let bundled = install_dir().join("models").join(name);
    if bundled.join("flownet.param").is_file() {
        return bundled;
    }
    data_dir().join("models").join(name)
}

pub fn onnx_dir() -> PathBuf {
    let bundled = install_dir().join("models").join("onnx");
    if bundled.is_dir() {
        return bundled;
    }
    data_dir().join("models").join("onnx")
}

struct BundledOnnx {
    name: &'static str,
    sha256: &'static str,
    url: Option<&'static str>,
}

fn bundled_onnx() -> &'static [BundledOnnx] {
    &[
        BundledOnnx {
            name: "APISR2xRRDB.onnx",
            sha256: "c80ac47ab07085ca6220a28840236d0b3bb5001b17204e967afb3748f7ee0f94",
            url: None,
        },
        BundledOnnx {
            name: "SPAN2xHFA2k.onnx",
            sha256: "c2a18389b5dbf67fd83825db076d88c6d0b0a52974d974dd1c32e7fa50e1558b",
            url: None,
        },
        BundledOnnx {
            name: "AnimeSharpieOutlineSubtle.onnx",
            sha256: "47e9a6db04c8b2ba492c1fbd3681d576d15c6027753c349fe186b1c65bfcdd90",
            url: None,
        },
        BundledOnnx {
            name: "DeepCreamBar.onnx",
            sha256: "ba32b5abe0303cd452c466516c2f88f3cbf0337b34867470dbdf4ffa6ec55b2a",
            url: Some("https://raw.githubusercontent.com/Cec1c/Aletheia-Lens/main/models/deepcreampy/bar.onnx"),
        },
        BundledOnnx {
            name: "DeepCreamMosaic.onnx",
            sha256: "339503f625ad0aea902b5acd4adc491741e53235e47b6c4ddd77aa3cc1d351f2",
            url: Some("https://raw.githubusercontent.com/Cec1c/Aletheia-Lens/main/models/deepcreampy/mosaic.onnx"),
        },
        BundledOnnx {
            name: "AnimeWatermarkLaMa512.onnx",
            sha256: "4512adab295ee5a5e02ccd1bdf8d45dccbac88309d9cff1532ffd5de876f02a4",
            url: Some("https://huggingface.co/mayocream/lama-manga-onnx/resolve/main/lama-manga.onnx"),
        },
    ]
}

pub fn onnx_file(name: &str) -> PathBuf {
    for candidate in onnx_candidates(name) {
        if candidate.is_file() {
            return candidate;
        }
    }
    data_dir().join("models").join("onnx").join(name)
}

fn onnx_candidates(name: &str) -> Vec<PathBuf> {
    vec![
        install_dir().join("models").join("onnx").join(name),
        data_dir().join("models").join("onnx").join(name),
        PathBuf::from("models").join("onnx").join(name),
        PathBuf::from("rife-pc").join("models").join("onnx").join(name),
    ]
}

/// Use a bundled network, or download it when a checksummed URL exists.
/// A missing file with no URL is an error so the caller can use the classical fallback.
fn verified_names() -> &'static Mutex<Vec<String>> {
    static VERIFIED: Mutex<Vec<String>> = Mutex::new(Vec::new());
    &VERIFIED
}

pub fn ensure_onnx(name: &str) -> Result<PathBuf> {
    if let Ok(guard) = verified_names().lock() {
        if guard.iter().any(|item| item == name) {
            if let Some(found) = onnx_candidates(name).into_iter().find(|path| path.is_file()) {
                return Ok(found);
            }
        }
    }
    if let Some(found) = onnx_candidates(name).into_iter().find(|path| path.is_file()) {
        if let Some(spec) = bundled_onnx().iter().find(|item| item.name == name) {
            let hash = sha256_file(&found)?;
            if !hash.eq_ignore_ascii_case(spec.sha256) {
                return Err(anyhow!(
                    "{name} is present but its SHA-256 does not match the shipped model"
                ));
            }
        }
        if let Ok(mut guard) = verified_names().lock() {
            guard.push(name.to_string());
        }
        return Ok(found);
    }
    let spec = bundled_onnx()
        .iter()
        .find(|item| item.name == name)
        .ok_or_else(|| anyhow!("no ONNX model named {name}"))?;
    let url = spec.url.ok_or_else(|| anyhow!("{name} is not installed"))?;
    let dest = data_dir().join("models").join("onnx").join(name);
    download_to(url, &dest)?;
    let hash = sha256_file(&dest)?;
    if !hash.eq_ignore_ascii_case(spec.sha256) {
        let _ = std::fs::remove_file(&dest);
        return Err(anyhow!("downloaded {name} did not match its checksum"));
    }
    if let Ok(mut guard) = verified_names().lock() {
        guard.push(name.to_string());
    }
    Ok(dest)
}

fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Clone, Debug)]
pub struct Download {
    pub id: &'static str,
    pub label: &'static str,
    pub url: &'static str,
    pub relative: &'static str,
    pub note: &'static str,
}

pub fn catalog() -> Vec<Download> {
    vec![
        Download {
            id: "rife426",
            label: "RIFE 4.26 flownet (about 11 MB)",
            url: "https://raw.githubusercontent.com/k4yt3x/video2x/master/models/rife/rife-v4.26/flownet.bin",
            relative: "models/rife-v4.26/flownet.bin",
            note: "ncnn weights published with video2x. The param file is downloaded with it.",
        },
        Download {
            id: "lama-onnx",
            label: "Anime LaMa 512 (about 198 MB)",
            url: "https://huggingface.co/mayocream/lama-manga-onnx/resolve/main/lama-manga.onnx",
            relative: "models/onnx/AnimeWatermarkLaMa512.onnx",
            note: "Same anime/manga LaMa fine-tune as the iOS app. Apache-2.0 export, MIT fine-tune, Apache-2.0 base. The installer downloads it; this button fetches it if it is missing. Checksummed.",
        },
        Download {
            id: "h3-dit-q4",
            label: "MiniMax H3 Ref2VA Q4_K_M GGUF",
            url: "https://huggingface.co/leejet/MiniMax-H3-GGUF/resolve/main/minimax_h3_ref2va_pruned-Q4_K_M.gguf",
            relative: "models/h3/minimax_h3_ref2va_pruned-Q4_K_M.gguf",
            note: "Community License excludes the US, EU, UK, and South Korea unless MiniMax authorized you. Not bundled.",
        },
        Download {
            id: "h3-llm-q4",
            label: "Qwen3-VL MiniMax H3 text encoder Q4_K_M",
            url: "https://huggingface.co/leejet/MiniMax-H3-GGUF/resolve/main/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
            relative: "models/h3/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
            note: "Same MiniMax H3 license as the diffusion weights. Several gigabytes.",
        },
        Download {
            id: "h3-vae",
            label: "MiniMax H3 video VAE fp16",
            url: "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors",
            relative: "models/h3/minimax_h3_video_vae_fp16.safetensors",
            note: "Required by stable-diffusion.cpp. Same upstream license.",
        },
        Download {
            id: "h3-audio-vae",
            label: "MiniMax H3 audio VAE fp32",
            url: "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors",
            relative: "models/h3/minimax_h3_audio_vae_fp32.safetensors",
            note: "Optional. Without it the CLI still writes video and skips decoded audio.",
        },
        Download {
            id: "animagine",
            label: "Animagine XL v3.1 (SDXL safetensors)",
            url: "https://huggingface.co/Linaqruf/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors",
            relative: "models/studio/animagine-xl-3.1.safetensors",
            note: "Fair-AI / CreativeML-style SDXL checkpoint used by AI Image Studio. Several gigabytes, downloaded on first use.",
        },
    ]
}

pub fn download_to(url: &str, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part = dest.with_extension("part");
    let status = Command::new(curl_bin())
        .args(["-L", "--fail", "--retry", "3", "-A", "rife-pc", "-o"])
        .arg(&part)
        .arg(url)
        .status()
        .map_err(|err| anyhow!("could not start curl ({err}). Windows 11 includes curl.exe."))?;
    if !status.success() {
        let _ = std::fs::remove_file(&part);
        return Err(anyhow!("download failed: {url}"));
    }
    std::fs::rename(&part, dest)?;
    Ok(())
}

pub fn download_id(id: &str) -> Result<PathBuf> {
    if id == "rife426" {
        let dir = data_dir().join("models").join("rife-v4.26");
        std::fs::create_dir_all(&dir)?;
        download_to(
            "https://raw.githubusercontent.com/k4yt3x/video2x/master/models/rife/rife-v4.26/flownet.param",
            &dir.join("flownet.param"),
        )?;
        download_to(
            "https://raw.githubusercontent.com/k4yt3x/video2x/master/models/rife/rife-v4.26/flownet.bin",
            &dir.join("flownet.bin"),
        )?;
        return Ok(dir);
    }
    if id == "sd-cpu" || id == "sd-vulkan" {
        return download_sd(id == "sd-vulkan");
    }
    let item = catalog()
        .into_iter()
        .find(|item| item.id == id)
        .ok_or_else(|| anyhow!("unknown download {id}"))?;
    let dest = data_dir().join(item.relative);
    download_to(item.url, &dest)?;
    if let Some(name) = dest.file_name().and_then(|value| value.to_str()) {
        if let Some(spec) = bundled_onnx().iter().find(|spec| spec.name == name) {
            let hash = sha256_file(&dest)?;
            if !hash.eq_ignore_ascii_case(spec.sha256) {
                let _ = std::fs::remove_file(&dest);
                return Err(anyhow!("downloaded {name} did not match its checksum"));
            }
        }
    }
    Ok(dest)
}

pub fn download_sd(vulkan: bool) -> Result<PathBuf> {
    let body = curl_text("https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest")?;
    let json: serde_json::Value = serde_json::from_str(&body)?;
    let needle = if vulkan { "bin-win-vulkan-x64.zip" } else { "bin-win-cpu-x64.zip" };
    let asset = json["assets"]
        .as_array()
        .and_then(|assets| {
            assets.iter().find(|asset| {
                asset["name"].as_str().unwrap_or("").contains(needle)
            })
        })
        .ok_or_else(|| anyhow!("the latest stable-diffusion.cpp release has no {needle}"))?;
    let url = asset["browser_download_url"]
        .as_str()
        .ok_or_else(|| anyhow!("release asset has no URL"))?;
    let zip_path = data_dir().join("tools").join(needle);
    download_to(url, &zip_path)?;
    let dest = data_dir().join("tools").join(if vulkan { "sd-vulkan" } else { "sd-cpu" });
    std::fs::create_dir_all(&dest)?;
    let status = Command::new("tar")
        .args(["-xf"])
        .arg(&zip_path)
        .arg("-C")
        .arg(&dest)
        .status()?;
    if !status.success() {
        return Err(anyhow!("could not unzip {needle}. Windows 11 tar.exe can unpack zip files."));
    }
    Ok(dest)
}

pub fn sd_cli(vulkan: bool) -> Option<PathBuf> {
    let root = data_dir().join("tools").join(if vulkan { "sd-vulkan" } else { "sd-cpu" });
    find_named(&root, "sd-cli.exe").or_else(|| find_named(&root, "sd-cli"))
}

fn find_named(dir: &Path, name: &str) -> Option<PathBuf> {
    let direct = dir.join(name);
    if direct.is_file() {
        return Some(direct);
    }
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_named(&path, name) {
                return Some(found);
            }
        } else if path.file_name().and_then(|n| n.to_str()) == Some(name) {
            return Some(path);
        }
    }
    None
}

fn curl_bin() -> String {
    if cfg!(windows) { "curl.exe".into() } else { "curl".into() }
}

fn curl_text(url: &str) -> Result<String> {
    let output = Command::new(curl_bin())
        .args(["-fsSL", "-A", "rife-pc", url])
        .stdout(Stdio::piped())
        .output()
        .map_err(|err| anyhow!("curl failed: {err}"))?;
    if !output.status.success() {
        return Err(anyhow!("could not read {url}"));
    }
    let mut text = String::new();
    output.stdout.as_slice().read_to_string(&mut text)?;
    Ok(text)
}

pub fn h3_paths() -> H3Paths {
    let root = data_dir().join("models").join("h3");
    H3Paths {
        diffusion: root.join("minimax_h3_ref2va_pruned-Q4_K_M.gguf"),
        llm: root.join("qwen3vl_32b_minimax_h3-Q4_K_M.gguf"),
        vae: root.join("minimax_h3_video_vae_fp16.safetensors"),
        audio_vae: root.join("minimax_h3_audio_vae_fp32.safetensors"),
    }
}

pub struct H3Paths {
    pub diffusion: PathBuf,
    pub llm: PathBuf,
    pub vae: PathBuf,
    pub audio_vae: PathBuf,
}

impl H3Paths {
    pub fn missing(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if !self.diffusion.is_file() {
            missing.push("Ref2VA Q4_K_M");
        }
        if !self.llm.is_file() {
            missing.push("Qwen3-VL Q4_K_M");
        }
        if !self.vae.is_file() {
            missing.push("video VAE");
        }
        missing
    }
}

