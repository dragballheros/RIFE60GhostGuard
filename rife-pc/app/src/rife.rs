//! RIFE v4.x on ncnn Vulkan.
//!
//! The Windows build links `rife_capi`, which embeds nihui's rife-ncnn-vulkan
//! runtime (the flownet plus the Vulkan warp shader). That is the stack that
//! already runs on AMD Radeon Pro 5300M/5500M/5600M under Adrenalin. A wgpu
//! port would not load the published rife-v4.x ncnn models.

use std::path::Path;

use anyhow::{anyhow, Result};
use rife_guards::Frame;

pub struct Engine {
    #[cfg(feature = "ncnn")]
    raw: *mut ffi::RifeEngine,
    pub gpu_name: String,
}

unsafe impl Send for Engine {}

impl Engine {
    pub fn open(model_dir: &Path, gpu: i32) -> Result<Self> {
        #[cfg(not(feature = "ncnn"))]
        {
            let _ = (model_dir, gpu);
            Err(anyhow!(
                "this build has no Vulkan RIFE backend. On Windows run rife-pc/build-windows.ps1, which links ncnn Vulkan."
            ))
        }
        #[cfg(feature = "ncnn")]
        {
            let dir = model_dir.to_string_lossy();
            let c = std::ffi::CString::new(dir.as_ref())
                .map_err(|_| anyhow!("model path contains a nul"))?;
            let raw = unsafe { ffi::rife_create(c.as_ptr(), gpu) };
            if raw.is_null() {
                return Err(anyhow!(error_text(std::ptr::null_mut())));
            }
            let gpu_name = name_text(raw);
            Ok(Self { raw, gpu_name })
        }
    }

    pub fn interpolate(
        &self,
        previous: &Frame,
        next: &Frame,
        timestep: f32,
        tile: i32,
    ) -> Result<Frame> {
        #[cfg(not(feature = "ncnn"))]
        {
            let _ = (previous, next, timestep, tile);
            Err(anyhow!("Vulkan RIFE is not linked"))
        }
        #[cfg(feature = "ncnn")]
        {
            if previous.width != next.width
                || previous.height != next.height
                || previous.data.len() != next.data.len()
            {
                return Err(anyhow!("RIFE inputs differ in size"));
            }
            let mut out = vec![0u8; previous.data.len()];
            let rc = unsafe {
                ffi::rife_interpolate(
                    self.raw,
                    previous.data.as_ptr(),
                    next.data.as_ptr(),
                    out.as_mut_ptr(),
                    previous.width,
                    previous.height,
                    timestep,
                    tile,
                )
            };
            if rc != 0 {
                return Err(anyhow!(error_text(self.raw)));
            }
            Ok(Frame {
                width: previous.width,
                height: previous.height,
                rgba: false,
                data: out,
            })
        }
    }
}

#[cfg(feature = "ncnn")]
impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { ffi::rife_destroy(self.raw) };
    }
}

#[cfg(feature = "ncnn")]
fn error_text(engine: *mut ffi::RifeEngine) -> String {
    let mut buf = vec![0u8; 512];
    unsafe {
        ffi::rife_last_error(engine, buf.as_mut_ptr() as *mut libc_char, buf.len() as i32);
    }
    cstr(&buf)
}

#[cfg(feature = "ncnn")]
fn name_text(engine: *mut ffi::RifeEngine) -> String {
    let mut buf = vec![0u8; 256];
    unsafe {
        ffi::rife_gpu_name(engine, buf.as_mut_ptr() as *mut libc_char, buf.len() as i32);
    }
    cstr(&buf)
}

#[cfg(feature = "ncnn")]
fn cstr(buf: &[u8]) -> String {
    let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

#[cfg(feature = "ncnn")]
#[allow(non_camel_case_types)]
mod ffi {
    use super::libc_char;
    #[repr(C)]
    pub struct RifeEngine {
        _private: [u8; 0],
    }
    extern "C" {
        pub fn rife_create(model_utf8: *const libc_char, gpu: i32) -> *mut RifeEngine;
        pub fn rife_destroy(engine: *mut RifeEngine);
        pub fn rife_interpolate(
            engine: *mut RifeEngine,
            bgra0: *const u8,
            bgra1: *const u8,
            bgra_out: *mut u8,
            width: i32,
            height: i32,
            timestep: f32,
            tile: i32,
        ) -> i32;
        pub fn rife_last_error(engine: *mut RifeEngine, buf: *mut libc_char, len: i32) -> i32;
        pub fn rife_gpu_name(engine: *mut RifeEngine, buf: *mut libc_char, len: i32) -> i32;
    }
}

#[cfg(feature = "ncnn")]
#[allow(non_camel_case_types)]
pub(crate) type libc_char = std::os::raw::c_char;
