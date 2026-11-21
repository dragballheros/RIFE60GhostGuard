// Small ONNX Runtime wrapper. build-windows.ps1 compiles this against the
// DirectML package and copies ort_capi.dll next to rife-pc.exe.
// Prefer the DirectML execution provider so the Radeon Pro can run the
// converted Core ML networks. If DirectML refuses the model, fall back to CPU.

#include <cstdio>
#include <cstring>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

#include <onnxruntime_cxx_api.h>

#if defined(_WIN32) && __has_include("dml_provider_factory.h")
#include "dml_provider_factory.h"
#define RIFE_HAS_DML 1
#endif

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#define ORT_CAPI __declspec(dllexport)
#else
#define ORT_CAPI
#endif

struct OrtPack {
    Ort::Env env;
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    std::string error;
    OrtPack() : env(ORT_LOGGING_LEVEL_WARNING, "rife-pc") {}
};

static void copy_err(const std::string& text, char* err, int err_len) {
    if (!err || err_len <= 0) return;
    std::snprintf(err, static_cast<size_t>(err_len), "%s", text.c_str());
}

extern "C" {

ORT_CAPI OrtPack* ort_open(const char* model_utf8, int prefer_dml, char* err, int err_len) {
    try {
        auto pack = std::make_unique<OrtPack>();
        pack->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#if defined(RIFE_HAS_DML)
        if (prefer_dml) {
            OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_DML(pack->options, 0);
            if (status) {
                const char* message = Ort::GetApi().GetErrorMessage(status);
                pack->error = message ? message : "DirectML provider failed";
                Ort::GetApi().ReleaseStatus(status);
                pack->options = Ort::SessionOptions();
                pack->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            }
        }
#else
        (void)prefer_dml;
#endif
#if defined(_WIN32)
        std::wstring wide;
        if (model_utf8) {
            int chars = MultiByteToWideChar(CP_UTF8, 0, model_utf8, -1, nullptr, 0);
            if (chars > 0) {
                wide.assign(static_cast<size_t>(chars), L'\0');
                MultiByteToWideChar(CP_UTF8, 0, model_utf8, -1, wide.data(), chars);
                if (!wide.empty() && wide.back() == L'\0') wide.pop_back();
            }
        }
        pack->session = std::make_unique<Ort::Session>(pack->env, wide.c_str(), pack->options);
#else
        pack->session = std::make_unique<Ort::Session>(pack->env, model_utf8 ? model_utf8 : "", pack->options);
#endif
        return pack.release();
    } catch (const std::exception& ex) {
        copy_err(ex.what(), err, err_len);
        return nullptr;
    }
}

ORT_CAPI void ort_close(OrtPack* pack) { delete pack; }

ORT_CAPI int ort_run(
    OrtPack* pack,
    int n_in,
    const char** names,
    const float** data,
    const int64_t** shapes,
    const int* ranks,
    int n_out,
    const char** out_names,
    float** out_data,
    const int64_t* out_len,
    char* err,
    int err_len) {
    if (!pack || !pack->session || n_in <= 0 || n_out <= 0) {
        copy_err("bad ONNX arguments", err, err_len);
        return -1;
    }
    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        inputs.reserve(static_cast<size_t>(n_in));
        for (int i = 0; i < n_in; ++i) {
            int64_t count = 1;
            for (int d = 0; d < ranks[i]; ++d) count *= shapes[i][d];
            inputs.push_back(Ort::Value::CreateTensor<float>(
                mem, const_cast<float*>(data[i]), static_cast<size_t>(count), shapes[i], static_cast<size_t>(ranks[i])));
        }
        auto outputs = pack->session->Run(
            Ort::RunOptions{nullptr}, names, inputs.data(), static_cast<size_t>(n_in), out_names, static_cast<size_t>(n_out));
        for (int i = 0; i < n_out; ++i) {
            float* src = outputs[static_cast<size_t>(i)].GetTensorMutableData<float>();
            std::memcpy(out_data[i], src, static_cast<size_t>(out_len[i]) * sizeof(float));
        }
        return 0;
    } catch (const std::exception& ex) {
        copy_err(ex.what(), err, err_len);
        return -1;
    }
}

}
