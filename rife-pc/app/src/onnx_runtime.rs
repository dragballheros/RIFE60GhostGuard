//! Loads `ort_capi.dll`, the small DirectML wrapper built by `build-windows.ps1`.
//! The ONNX Runtime and DirectML binaries are installed beside the app.
//! Model files themselves are downloaded later and are not in the installer.

use std::ffi::CString;
use std::path::Path;

use anyhow::{anyhow, Result};

pub struct Session {
    _lib: libloading::Library,
    raw: *mut std::ffi::c_void,
    close: unsafe extern "C" fn(*mut std::ffi::c_void),
    run: RunFn,
}

type RunFn = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    i32,
    *const *const i8,
    *const *const f32,
    *const *const i64,
    *const i32,
    i32,
    *const *const i8,
    *mut *mut f32,
    *const i64,
    *mut i8,
    i32,
) -> i32;

unsafe impl Send for Session {}

impl Session {
    pub fn open(model: &Path) -> Result<Self> {
        let dll = crate::models::find_tool("ort_capi.dll");
        if !dll.is_file() {
            return Err(anyhow!(
                "ONNX Runtime is not installed (missing ort_capi.dll). Re-run the Windows installer build."
            ));
        }
        let lib = unsafe { libloading::Library::new(&dll) }
            .map_err(|err| anyhow!("failed to load {}: {err}", dll.display()))?;
        let (open_fn, close_fn, run_fn) = {
            let open: libloading::Symbol<
                unsafe extern "C" fn(*const i8, i32, *mut i8, i32) -> *mut std::ffi::c_void,
            > = unsafe { lib.get(b"ort_open") }
                .map_err(|err| anyhow!("ort_capi.dll has no ort_open: {err}"))?;
            let close: libloading::Symbol<unsafe extern "C" fn(*mut std::ffi::c_void)> =
                unsafe { lib.get(b"ort_close") }
                    .map_err(|err| anyhow!("ort_capi.dll has no ort_close: {err}"))?;
            let run: libloading::Symbol<RunFn> = unsafe { lib.get(b"ort_run") }
                .map_err(|err| anyhow!("ort_capi.dll has no ort_run: {err}"))?;
            (*open, *close, *run)
        };
        let path = CString::new(model.to_string_lossy().as_bytes())
            .map_err(|_| anyhow!("model path contains a nul"))?;
        let mut err = vec![0i8; 512];
        let raw = unsafe { open_fn(path.as_ptr(), 1, err.as_mut_ptr(), err.len() as i32) };
        if raw.is_null() {
            return Err(anyhow!("failed to open {}: {}", model.display(), c_string(&err)));
        }
        Ok(Self {
            _lib: lib,
            raw,
            close: close_fn,
            run: run_fn,
        })
    }

    pub fn run_one(
        &self,
        input: &str,
        data: &[f32],
        shape: &[i64],
        output: &str,
        out_shape: &[i64],
    ) -> Result<Vec<f32>> {
        let mut count = 1i64;
        for dim in out_shape {
            count *= (*dim).max(1);
        }
        let mut out = vec![0.0f32; count as usize];
        self.run_named(&[(input, data, shape)], &[(output, &mut out)])?;
        Ok(out)
    }

    pub fn run_named(&self, inputs: &[(&str, &[f32], &[i64])], outputs: &[(&str, &mut [f32])]) -> Result<()> {
        let input_names: Result<Vec<CString>> = inputs
            .iter()
            .map(|(name, _, _)| CString::new(*name).map_err(|_| anyhow!("bad input name")))
            .collect();
        let input_names = input_names?;
        let output_names: Result<Vec<CString>> = outputs
            .iter()
            .map(|(name, _)| CString::new(*name).map_err(|_| anyhow!("bad output name")))
            .collect();
        let output_names = output_names?;
        let name_ptrs: Vec<*const i8> = input_names.iter().map(|n| n.as_ptr()).collect();
        let data_ptrs: Vec<*const f32> = inputs.iter().map(|(_, data, _)| data.as_ptr()).collect();
        let shape_ptrs: Vec<*const i64> = inputs.iter().map(|(_, _, shape)| shape.as_ptr()).collect();
        let ranks: Vec<i32> = inputs.iter().map(|(_, _, shape)| shape.len() as i32).collect();
        let out_name_ptrs: Vec<*const i8> = output_names.iter().map(|n| n.as_ptr()).collect();
        let mut out_ptrs: Vec<*mut f32> = outputs.iter().map(|(_, data)| data.as_ptr() as *mut f32).collect();
        let out_lens: Vec<i64> = outputs.iter().map(|(_, data)| data.len() as i64).collect();
        let mut err = vec![0i8; 512];
        let rc = unsafe {
            (self.run)(
                self.raw,
                inputs.len() as i32,
                name_ptrs.as_ptr(),
                data_ptrs.as_ptr(),
                shape_ptrs.as_ptr(),
                ranks.as_ptr(),
                outputs.len() as i32,
                out_name_ptrs.as_ptr(),
                out_ptrs.as_mut_ptr(),
                out_lens.as_ptr(),
                err.as_mut_ptr(),
                err.len() as i32,
            )
        };
        if rc != 0 {
            return Err(anyhow!("{}", c_string(&err)));
        }
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { (self.close)(self.raw) };
    }
}

fn c_string(buf: &[i8]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|b| **b != 0).map(|b| *b as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
