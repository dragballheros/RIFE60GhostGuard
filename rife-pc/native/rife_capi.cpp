// C API around nihui rife-ncnn-vulkan (RIFE v4 flownet on Vulkan).
// The .cpp files from that project supply the network and the warp shader.
// This file only loads the model, picks an AMD GPU, and tiles large frames.

#include "rife.h"

#include "gpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <io.h>
#define RIFE_API __declspec(dllexport)
#else
#define RIFE_API
#include <stdio.h>
#endif

thread_local std::string g_error;
std::once_flag g_gpu_instance_once;

void set_error(const std::string& message) { g_error = message; }

#if defined(_WIN32)
std::wstring utf8_to_wide(const char* text) {
    if (!text || !*text) return std::wstring();
    int chars = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    if (chars <= 0) return std::wstring();
    std::wstring out(static_cast<size_t>(chars), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text, -1, out.data(), chars);
    if (!out.empty() && out.back() == L'\0') out.pop_back();
    return out;
}
#endif

bool contains_amd(const char* name) {
    if (!name) return false;
    std::string text(name);
    for (char& ch : text) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return text.find("amd") != std::string::npos || text.find("radeon") != std::string::npos
        || text.find("advanced micro") != std::string::npos;
}

int pick_gpu(int requested) {
    const int count = ncnn::get_gpu_count();
    if (count <= 0) return -1;
    if (requested >= 0 && requested < count) return requested;
    int amd = -1;
    for (int i = 0; i < count; ++i) {
        if (contains_amd(ncnn::get_gpu_info(i).device_name())) amd = i;
    }
    return amd >= 0 ? amd : 0;
}

void bgra_to_bgr(const uint8_t* src, uint8_t* dst, size_t pixels) {
    for (size_t i = 0; i < pixels; ++i) {
        dst[i * 3] = src[i * 4];
        dst[i * 3 + 1] = src[i * 4 + 1];
        dst[i * 3 + 2] = src[i * 4 + 2];
    }
}

void bgr_to_bgra(const uint8_t* src, uint8_t* dst, size_t pixels) {
    for (size_t i = 0; i < pixels; ++i) {
        dst[i * 4] = src[i * 3];
        dst[i * 4 + 1] = src[i * 3 + 1];
        dst[i * 4 + 2] = src[i * 3 + 2];
        dst[i * 4 + 3] = 255;
    }
}

struct Engine {
    RIFE* net = nullptr;
    int gpu = 0;
    uint64_t gpu_memory_mb = 0;
    char name[256];
    char error[512];
    Engine() { name[0] = 0; error[0] = 0; }
};

// Every RIFE worker owns its scratch on its own thread. This both avoids
// cross-thread races and retains vector capacity between consecutive frames.
struct Scratch {
    std::vector<uint8_t> bgr0;
    std::vector<uint8_t> bgr1;
    std::vector<uint8_t> bgr_out;
    std::vector<uint8_t> tile0;
    std::vector<uint8_t> tile1;
    std::vector<uint8_t> tile_out;
    std::vector<float> accum;
    std::vector<float> weights;
};
thread_local Scratch g_scratch;

int run_full(Engine* engine, const uint8_t* bgr0, const uint8_t* bgr1, uint8_t* bgr_out, int w, int h, float timestep) {
    const size_t bytes = static_cast<size_t>(w) * static_cast<size_t>(h) * 3;
    if (timestep <= 0.f) {
        std::memcpy(bgr_out, bgr0, bytes);
        return 0;
    }
    if (timestep >= 1.f) {
        std::memcpy(bgr_out, bgr1, bytes);
        return 0;
    }

    // RIFE::process consumes these external pixel buffers synchronously.
    // Wrapping the caller's storage directly removes two input copies, one
    // output allocation, and the final output copy per inference.
    ncnn::Mat in0m(w, h, const_cast<uint8_t*>(bgr0), static_cast<size_t>(3), 1);
    ncnn::Mat in1m(w, h, const_cast<uint8_t*>(bgr1), static_cast<size_t>(3), 1);
    ncnn::Mat outm(w, h, bgr_out, static_cast<size_t>(3), 1);
    const int rc = engine->net->process(in0m, in1m, timestep, outm);
    if (rc != 0) {
        char message[128];
        std::snprintf(message, sizeof(message), "RIFE process failed (%d)", rc);
        set_error(message);
    }
    return rc;
}

extern "C" {

RIFE_API Engine* rife_create(const char* model_utf8, int gpu) {
    g_error.clear();
    std::call_once(g_gpu_instance_once, []() {
        ncnn::create_gpu_instance();
    });
    int chosen = pick_gpu(gpu);
    if (chosen < 0) {
        set_error("No Vulkan device. Install the AMD Software: Adrenalin driver and reboot.");
        return nullptr;
    }
    Engine* engine = new Engine();
    engine->gpu = chosen;
    const ncnn::GpuInfo& gpu_info = ncnn::get_gpu_info(chosen);
    const char* name = gpu_info.device_name();
    std::snprintf(engine->name, sizeof(engine->name), "%s", name ? name : "Vulkan GPU");
    const VkPhysicalDeviceMemoryProperties& memory = gpu_info.physicalDeviceMemoryProperties();
    uint64_t local_bytes = 0;
    for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
        if ((memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            local_bytes += memory.memoryHeaps[i].size;
        }
    }
    engine->gpu_memory_mb = local_bytes / (1024ull * 1024ull);
    engine->net = new RIFE(chosen, false, false, false, 4, false, true);
#if defined(_WIN32)
    std::wstring wide = utf8_to_wide(model_utf8);
    if (wide.size() > 200) {
        std::snprintf(engine->error, sizeof(engine->error), "Model path is too long.");
        set_error(engine->error);
        delete engine->net;
        delete engine;
        return nullptr;
    }
    std::wstring param = wide + L"/flownet.param";
    std::wstring bin = wide + L"/flownet.bin";
    if (_waccess(param.c_str(), 0) != 0 || _waccess(bin.c_str(), 0) != 0) {
        std::snprintf(engine->error, sizeof(engine->error), "flownet.param or flownet.bin is missing.");
        set_error(engine->error);
        delete engine->net;
        delete engine;
        return nullptr;
    }
    int rc = engine->net->load(wide);
#else
    int rc = engine->net->load(model_utf8 ? model_utf8 : "");
#endif
    if (rc != 0) {
        std::snprintf(engine->error, sizeof(engine->error), "Failed to load the RIFE model (%d).", rc);
        set_error(engine->error);
        delete engine->net;
        delete engine;
        return nullptr;
    }
    return engine;
}

RIFE_API void rife_destroy(Engine* engine) {
    if (!engine) return;
    delete engine->net;
    delete engine;
}

RIFE_API int rife_gpu_name(Engine* engine, char* buf, int len) {
    if (!buf || len <= 0) return -1;
    const char* name = engine ? engine->name : "";
    std::snprintf(buf, static_cast<size_t>(len), "%s", name);
    return 0;
}

RIFE_API uint64_t rife_gpu_memory_mb(Engine* engine) {
    return engine ? engine->gpu_memory_mb : 0;
}

RIFE_API int rife_last_error(Engine* engine, char* buf, int len) {
    if (!buf || len <= 0) return -1;
    const char* text = engine && engine->error[0] ? engine->error : g_error.c_str();
    std::snprintf(buf, static_cast<size_t>(len), "%s", text ? text : "");
    return 0;
}

RIFE_API int rife_interpolate(
    Engine* engine,
    const uint8_t* bgra0,
    const uint8_t* bgra1,
    uint8_t* bgra_out,
    int width,
    int height,
    float timestep,
    int tile) {
    g_error.clear();
    if (!engine || !engine->net || !bgra0 || !bgra1 || !bgra_out || width < 8 || height < 8) {
        set_error("Bad RIFE arguments.");
        return -1;
    }
    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t frame_bytes = pixels * 3;
    Scratch& scratch = g_scratch;
    scratch.bgr0.resize(frame_bytes);
    scratch.bgr1.resize(frame_bytes);
    scratch.bgr_out.resize(frame_bytes);
    bgra_to_bgr(bgra0, scratch.bgr0.data(), pixels);
    bgra_to_bgr(bgra1, scratch.bgr1.data(), pixels);

    const bool whole = tile <= 0 || (tile >= width && tile >= height);
    if (whole) {
        const int rc = run_full(engine, scratch.bgr0.data(), scratch.bgr1.data(),
                                scratch.bgr_out.data(), width, height, timestep);
        if (rc != 0) return rc;
        bgr_to_bgra(scratch.bgr_out.data(), bgra_out, pixels);
        return 0;
    }

    int overlap = 32;
    if (overlap * 2 >= tile) overlap = std::max(8, tile / 4);
    const int step = std::max(32, tile - overlap);
    scratch.accum.assign(pixels * 3, 0.f);
    scratch.weights.assign(pixels, 0.f);

    for (int y0 = 0; y0 < height; y0 += step) {
        int y1 = std::min(height, y0 + tile);
        if (height - y1 > 0 && height - y1 < 32) y1 = height;
        for (int x0 = 0; x0 < width; x0 += step) {
            int x1 = std::min(width, x0 + tile);
            if (width - x1 > 0 && width - x1 < 32) x1 = width;
            const int tw = x1 - x0;
            const int th = y1 - y0;
            const size_t tile_bytes = static_cast<size_t>(tw) * static_cast<size_t>(th) * 3;
            scratch.tile0.resize(tile_bytes);
            scratch.tile1.resize(tile_bytes);
            scratch.tile_out.resize(tile_bytes);
            for (int y = 0; y < th; ++y) {
                const uint8_t* row0 = scratch.bgr0.data() + (static_cast<size_t>(y0 + y) * width + x0) * 3;
                const uint8_t* row1 = scratch.bgr1.data() + (static_cast<size_t>(y0 + y) * width + x0) * 3;
                std::memcpy(scratch.tile0.data() + static_cast<size_t>(y) * tw * 3, row0, static_cast<size_t>(tw) * 3);
                std::memcpy(scratch.tile1.data() + static_cast<size_t>(y) * tw * 3, row1, static_cast<size_t>(tw) * 3);
            }
            int rc = run_full(engine, scratch.tile0.data(), scratch.tile1.data(),
                              scratch.tile_out.data(), tw, th, timestep);
            if (rc != 0) return rc;
            for (int y = 0; y < th; ++y) {
                for (int x = 0; x < tw; ++x) {
                    float wx = 1.f;
                    float wy = 1.f;
                    if (x0 > 0) wx = std::min(wx, (x + 0.5f) / static_cast<float>(overlap));
                    if (x1 < width) wx = std::min(wx, (tw - 1 - x + 0.5f) / static_cast<float>(overlap));
                    if (y0 > 0) wy = std::min(wy, (y + 0.5f) / static_cast<float>(overlap));
                    if (y1 < height) wy = std::min(wy, (th - 1 - y + 0.5f) / static_cast<float>(overlap));
                    float wgt = std::max(0.f, wx) * std::max(0.f, wy);
                    const int dst = (y0 + y) * width + (x0 + x);
                    const uint8_t* pix = scratch.tile_out.data() + (static_cast<size_t>(y) * tw + x) * 3;
                    scratch.accum[static_cast<size_t>(dst) * 3] += pix[0] * wgt;
                    scratch.accum[static_cast<size_t>(dst) * 3 + 1] += pix[1] * wgt;
                    scratch.accum[static_cast<size_t>(dst) * 3 + 2] += pix[2] * wgt;
                    scratch.weights[dst] += wgt;
                }
            }
            if (x1 >= width) break;
        }
        if (y1 >= height) break;
    }

    for (size_t i = 0; i < pixels; ++i) {
        if (scratch.weights[i] <= 0.f) {
            scratch.bgr_out[i * 3] = static_cast<uint8_t>((scratch.bgr0[i * 3] + scratch.bgr1[i * 3]) / 2);
            scratch.bgr_out[i * 3 + 1] = static_cast<uint8_t>((scratch.bgr0[i * 3 + 1] + scratch.bgr1[i * 3 + 1]) / 2);
            scratch.bgr_out[i * 3 + 2] = static_cast<uint8_t>((scratch.bgr0[i * 3 + 2] + scratch.bgr1[i * 3 + 2]) / 2);
        } else {
            for (size_t channel = 0; channel < 3; ++channel) {
                float value = scratch.accum[i * 3 + channel] / scratch.weights[i] + 0.5f;
                if (value < 0.f) value = 0.f;
                if (value > 255.f) value = 255.f;
                scratch.bgr_out[i * 3 + channel] = static_cast<uint8_t>(value);
            }
        }
    }
    bgr_to_bgra(scratch.bgr_out.data(), bgra_out, pixels);
    return 0;
}

}  // extern "C"
