fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_NCNN").is_none() {
        return;
    }
    let dir = std::env::var("RIFE_CAPI_DIR").unwrap_or_else(|_| {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
        format!("{manifest}/../native/dist")
    });
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=rife_capi");
    println!("cargo:rerun-if-env-changed=RIFE_CAPI_DIR");
}
