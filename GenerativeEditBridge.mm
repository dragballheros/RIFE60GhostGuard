#include "GenerativeEditBridge.h"
#include "stable-diffusion.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <exception>

struct GEHandle { sd_ctx_t *ctx = nullptr; };
static thread_local std::string ge_error;

static bool ge_is_noise(const char *message) {
    if (!message) return true;
    return strstr(message, "deallocating") || strstr(message, "ggml_metal_free") || strstr(message, "ggml_metal_init");
}

static void ge_log(enum sd_log_level_t level, const char *message, void *) {
    if (!message || ge_is_noise(message)) return;
    if (level == SD_LOG_ERROR || level == SD_LOG_WARN) ge_error = message;
}

GEHandle *ge_create(const char *diffusion, const char *vae, const char *t5) {
    ge_error.clear();
    if (!diffusion || !vae || !t5) { ge_error = "Missing generative model paths"; return nullptr; }
    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    p.diffusion_model_path = diffusion;
    p.vae_path = vae;
    p.t5xxl_path = t5;
    p.enable_mmap = true;
    p.flash_attn = false;
    p.diffusion_flash_attn = false;
    p.diffusion_conv_direct = false;
    p.vae_conv_direct = false;
    p.eager_load = false;
    p.max_vram = "1.2";
    p.n_threads = 2;
    p.backend = "metal";
    p.params_backend = "cpu";
    sd_set_log_callback(ge_log, nullptr);
    auto *h = (GEHandle *)calloc(1, sizeof(GEHandle));
    try {
        h->ctx = new_sd_ctx(&p);
    } catch (const std::exception &ex) {
        ge_error = ex.what();
        free(h);
        return nullptr;
    } catch (...) {
        ge_error = "Wan model load aborted. Close other apps and try again.";
        free(h);
        return nullptr;
    }
    if (!h->ctx) { free(h); if (ge_error.empty()) ge_error = "Could not load generative model"; return nullptr; }
    return h;
}

void ge_destroy(GEHandle *h) {
    if (!h) return;
    if (h->ctx) free_sd_ctx(h->ctx);
    free(h);
}

void ge_cancel(GEHandle *h) {
    if (h && h->ctx) sd_cancel_generation(h->ctx, SD_CANCEL_ALL);
}

const char *ge_last_error(void) { return ge_error.c_str(); }

int ge_generate(GEHandle *h, const uint8_t *rgba, int frame_count, int width, int height, int fps,
                const char *prompt, const char *negative_prompt, int steps, float strength, int64_t seed,
                uint8_t **out_rgba, int *out_count, int *out_fps) {
    ge_error.clear();
    if (!h || !h->ctx || !rgba || frame_count < 5 || width < 32 || height < 32 || !out_rgba || !out_count) {
        ge_error = "Invalid generative edit request"; return 0;
    }
    std::vector<sd_image_t> controls(frame_count);
    for (int i = 0; i < frame_count; ++i) {
        controls[i] = { (uint32_t)width, (uint32_t)height, 4, const_cast<uint8_t *>(rgba + (size_t)i * width * height * 4) };
    }
    sd_vid_gen_params_t p;
    sd_vid_gen_params_init(&p);
    p.prompt = prompt ? prompt : "";
    p.negative_prompt = negative_prompt ? negative_prompt : "";
    p.width = width; p.height = height; p.fps = fps > 0 ? fps : 16;
    p.video_frames = frame_count; p.seed = seed; p.strength = strength; p.vace_strength = 1.f;
    p.sample_params.sample_steps = steps > 0 ? steps : 8;
    p.sample_params.guidance.txt_cfg = 6.f;
    p.control_frames = controls.data();
    p.control_frames_size = frame_count;

    sd_image_t *generated = nullptr;
    int count = 0, effectiveFPS = 0;
    if (!generate_video(h->ctx, &p, &generated, &count, nullptr, &effectiveFPS) || !generated || count <= 0) {
        if (ge_error.empty() || ge_is_noise(ge_error.c_str())) ge_error = "Generative video edit failed. Close other apps and try a shorter clip.";
        if (generated) free_sd_images(generated, count);
        return 0;
    }
    const size_t frameBytes = (size_t)width * height * 4;
    uint8_t *copy = (uint8_t *)malloc((size_t)count * frameBytes);
    if (!copy) { free_sd_images(generated, count); ge_error = "Not enough memory for generated frames"; return 0; }
    for (int i = 0; i < count; ++i) {
        if (generated[i].channel == 4) memcpy(copy + (size_t)i * frameBytes, generated[i].data, frameBytes);
        else {
            for (size_t j = 0; j < (size_t)width * height; ++j) {
                const uint8_t *src = generated[i].data + j * generated[i].channel;
                uint8_t *dst = copy + (size_t)i * frameBytes + j * 4;
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = 255;
            }
        }
    }
    free_sd_images(generated, count);
    *out_rgba = copy; *out_count = count; if (out_fps) *out_fps = effectiveFPS > 0 ? effectiveFPS : p.fps;
    return 1;
}

void ge_free_frames(uint8_t *frames) { free(frames); }
