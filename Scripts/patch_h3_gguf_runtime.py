#!/usr/bin/env python3
"""Connect the pinned H3 runtime to on-device GGUF streaming."""
from pathlib import Path
import sys

def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"{path.name} is missing the expected GGUF patch site")
    if text.count(old) != 1:
        raise SystemExit(f"{path.name} patch site is not unique")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

def main() -> None:
    root = Path(sys.argv[1])
    weights = root / "h3_weights.c"
    gpu = root / "h3_gpu.m"
    core = root / "h3.c"
    inventory = root / "h3_safetensors.c"
    replace_once(
        weights,
        '#include "h3_weights.h"\n',
        '#include "h3_weights.h"\n#include "h3_gguf.h"\n',
    )
    replace_once(
        weights,
        "        if (!safetensors_name(entry->d_name)) continue;\n",
        "        {\n"
        "            size_t name_length = strlen(entry->d_name);\n"
        "            int is_gguf = name_length > 5 &&\n"
        "                strcmp(entry->d_name + name_length - 5, \".gguf\") == 0;\n"
        "            if (!safetensors_name(entry->d_name) && !is_gguf) continue;\n"
        "        }\n",
    )
    replace_once(
        weights,
        "        if (!h3_st_read_header(paths[index], &store->headers[index], detail,\n"
        "                               sizeof(detail))) {\n",
        "        int gguf = h3_gguf_import_header(paths[index], &store->headers[index],\n"
        "                                     detail, sizeof(detail));\n"
        "        if (gguf < 0 || (gguf == 0 && !h3_st_read_header(\n"
        "                paths[index], &store->headers[index], detail, sizeof(detail)))) {\n",
    )
    replace_once(
        weights,
        "    if (tensor->dtype != dtype || tensor->ndim != ndim) {\n",
        "    if (!((dtype == H3_DTYPE_F32 && tensor->dtype == H3_DTYPE_F16)) &&\n"
        "        (tensor->dtype != dtype || tensor->ndim != ndim)) {\n",
    )
    replace_once(
        weights,
        "    if (elements > SIZE_MAX) {\n",
        "    if (dtype == H3_DTYPE_F32 && tensor->dtype == H3_DTYPE_F16) {\n"
        "        return h3_gguf_load_f16_as_f32(gpu, header->path, tensor->file_offset,\n"
        "                                       (size_t)elements, error, error_size);\n"
        "    }\n"
        "    if (elements > SIZE_MAX) {\n",
    )
    replace_once(
        gpu,
        '#include "h3_gpu.h"\n',
        '#include "h3_gpu.h"\n#include "h3_gguf.h"\n',
    )
    replace_once(
        gpu,
        "            snprintf(error, error_size, \"%s file read range overflows\", label);\n"
        "        return 0;\n"
        "    }\n"
        "    int descriptor = open(path, O_RDONLY | O_CLOEXEC);\n",
        "            snprintf(error, error_size, \"%s file read range overflows\", label);\n"
        "        return 0;\n"
        "    }\n"
        "    if (h3_gguf_try_decode(path, file_offset, TENSOR(opaque).buffer.contents,\n"
        "                           elements, dtype == H3_GPU_BF16, error, error_size)) {\n"
        "        return !(error && error_size && error[0]);\n"
        "    }\n"
        "    int descriptor = open(path, O_RDONLY | O_CLOEXEC);\n",
    )
    replace_once(
        inventory,
        '#include "h3_safetensors.h"\n',
        '#include "h3_safetensors.h"\n#include "h3_gguf.h"\n',
    )
    replace_once(
        inventory,
        '        if (entry->d_name[0] == \'.\' || !h3_has_suffix(entry->d_name, ".safetensors")) continue;\n',
        "        if (entry->d_name[0] == '.') continue;\n"
        "        int is_gguf = h3_has_suffix(entry->d_name, \".gguf\");\n"
        "        if (!is_gguf && !h3_has_suffix(entry->d_name, \".safetensors\")) continue;\n",
    )
    replace_once(
        inventory,
        "        int ok = h3_st_read_header(path, &header, error, error_size);\n",
        "        int ok = is_gguf ? h3_gguf_import_header(path, &header, error, error_size) == 1\n"
        "                         : h3_st_read_header(path, &header, error, error_size);\n",
    )
    replace_once(
        core,
        '#include "h3_tokenizer.h"\n',
        '#include "h3_tokenizer.h"\n#include "h3_gguf.h"\n',
    )
    replace_once(
        core,
        "h3_ctx *h3_load_dir(const char *model_dir) {\n",
        "h3_ctx *h3_load_dir(const char *model_dir) {\n"
        "    h3_gguf_stage_tier1(model_dir);\n",
    )
    replace_once(
        weights,
        '        fail(error, error_size, "required weight is absent: %s", name);\n',
        '        h3_weight_format_absent(store, name, error, error_size);\n',
    )
    absent_fn = r'''
void h3_weight_format_absent(const h3_weight_store *store, const char *name,
                             char *error, size_t error_size) {
    const char *closest[4] = {0};
    int score[4] = {-1, -1, -1, -1};
    if (store && name) {
        for (size_t shard = 0; shard < store->count; shard++) {
            const h3_st_header *header = &store->headers[shard];
            for (size_t index = 0; index < header->tensor_count; index++) {
                const char *candidate = header->tensors[index].name;
                if (!candidate) continue;
                int shared = 0;
                while (name[shared] && candidate[shared] &&
                       name[shared] == candidate[shared]) shared++;
                for (int slot = 0; slot < 4; slot++) {
                    if (shared <= score[slot]) continue;
                    for (int shift = 3; shift > slot; shift--) {
                        score[shift] = score[shift - 1];
                        closest[shift] = closest[shift - 1];
                    }
                    score[slot] = shared;
                    closest[slot] = candidate;
                    break;
                }
            }
        }
    }
    char list[384] = "";
    size_t used = 0;
    for (int slot = 0; slot < 4; slot++) {
        if (!closest[slot] || used >= sizeof(list)) continue;
        int wrote = snprintf(list + used, sizeof(list) - used, "%s%s",
                             used ? ", " : "", closest[slot]);
        if (wrote < 0 || (size_t)wrote >= sizeof(list) - used) break;
        used += (size_t)wrote;
    }
    fail(error, error_size,
         "required weight is absent: %s; closest keys: %s",
         name ? name : "", list[0] ? list : "(none)");
}

'''
    text = weights.read_text(encoding="utf-8")
    marker = "const h3_st_tensor *h3_weight_find"
    if "void h3_weight_format_absent" not in text:
        if text.count(marker) != 1:
            raise SystemExit("h3_weight_find anchor is not unique")
        weights.write_text(text.replace(marker, absent_fn + marker, 1), encoding="utf-8")
    header = root / "h3_weights.h"
    decl = "void h3_weight_format_absent(const h3_weight_store *store, const char *name,\n                             char *error, size_t error_size);\n"
    header_text = header.read_text(encoding="utf-8")
    if decl not in header_text:
        anchor = "const h3_st_tensor *h3_weight_find"
        if header_text.count(anchor) != 1:
            raise SystemExit("h3_weights.h find anchor is not unique")
        header.write_text(header_text.replace(anchor, decl + "\n" + anchor, 1), encoding="utf-8")
    audio = root / "h3_audio_vae.c"
    old_load = '''    snprintf(name, sizeof(name), "%s.weight_v", prefix);
    conv->vector = f3(audio, name, outer, inner_channels, kernel,
                      error, error_size);
    if (!conv->vector) return 0;
    snprintf(name, sizeof(name), "%s.weight_g", prefix);
    conv->magnitude = f3(audio, name, outer, 1, 1, error, error_size);
    if (!conv->magnitude) return 0;
'''
    new_load = '''    /* Published MiniMax H3 audio VAE checkpoints store a folded conv weight.
     * Older exports use weight_g/weight_v, and torch weight_norm uses
     * parametrizations.weight.original0 (g) and original1 (v). Fold g/v once
     * into conv->weight. A plain .weight is already folded. */
    snprintf(name, sizeof(name), "%s.weight_v", prefix);
    int have_vector = h3_weight_find(audio->weights, name, NULL) != NULL;
    char magnitude_name[192];
    snprintf(magnitude_name, sizeof(magnitude_name), "%s.weight_g", prefix);
    int have_magnitude = h3_weight_find(audio->weights, magnitude_name, NULL) != NULL;
    char original0[192], original1[192], folded[192];
    snprintf(original0, sizeof(original0), "%s.parametrizations.weight.original0", prefix);
    snprintf(original1, sizeof(original1), "%s.parametrizations.weight.original1", prefix);
    snprintf(folded, sizeof(folded), "%s.weight", prefix);
    if (!have_vector && h3_weight_find(audio->weights, original1, NULL) &&
        h3_weight_find(audio->weights, original0, NULL)) {
        snprintf(name, sizeof(name), "%s", original1);
        snprintf(magnitude_name, sizeof(magnitude_name), "%s", original0);
        have_vector = 1;
        have_magnitude = 1;
    }
    if (have_vector && have_magnitude) {
        conv->vector = f3(audio, name, outer, inner_channels, kernel,
                          error, error_size);
        if (!conv->vector) return 0;
        conv->magnitude = f3(audio, magnitude_name, outer, 1, 1, error, error_size);
        if (!conv->magnitude) return 0;
    } else if (h3_weight_find(audio->weights, folded, NULL)) {
        conv->weight = f3(audio, folded, outer, inner_channels, kernel,
                          error, error_size);
        return conv->weight != NULL;
    } else {
        h3_weight_format_absent(audio->weights, name, error, error_size);
        return 0;
    }
'''
    audio_text = audio.read_text(encoding="utf-8")
    if old_load not in audio_text:
        if "parametrizations.weight.original0" not in audio_text:
            raise SystemExit("h3_audio_vae.c is missing the weight_v load site")
    else:
        audio.write_text(audio_text.replace(old_load, new_load, 1), encoding="utf-8")
    audio_text = audio.read_text(encoding="utf-8")
    old_norm = '''static int normalize_conv(audio_context *audio, audio_conv *conv,
                          char *error, size_t error_size) {
    uint32_t outer = conv->transpose ? conv->input_channels :
'''
    new_norm = '''static int normalize_conv(audio_context *audio, audio_conv *conv,
                          char *error, size_t error_size) {
    if (!conv->vector || !conv->magnitude) {
        if (conv->weight) return 1;
        fail(error, error_size, "audio convolution weight was not loaded");
        return 0;
    }
    uint32_t outer = conv->transpose ? conv->input_channels :
'''
    if old_norm not in audio_text:
        if "audio convolution weight was not loaded" not in audio_text:
            raise SystemExit("h3_audio_vae.c is missing normalize_conv")
    else:
        audio.write_text(audio_text.replace(old_norm, new_norm, 1), encoding="utf-8")
    progress_old = '''static void h3_progress_emit(h3_generation_progress *state, const char *phase,
                             int completed, int total) {
    if (!state || state->cancelled || !state->params->on_progress) return;
'''
    progress_new = '''extern void rife60_h3_stage(const char *text);
extern int rife60_h3_cancelled(void);

static void h3_progress_emit(h3_generation_progress *state, const char *phase,
                             int completed, int total) {
    char stage[160];
    snprintf(stage, sizeof(stage), "H3 %s %d/%d", phase ? phase : "work",
             completed, total);
    rife60_h3_stage(stage);
    if (!state || state->cancelled || rife60_h3_cancelled() ||
        !state->params->on_progress) {
        if (state && rife60_h3_cancelled()) state->cancelled = 1;
        return;
    }
'''
    core_text = core.read_text(encoding="utf-8")
    before_video = '''            if (!h3_video_vae_encode(
'''
    before_video_new = '''            h3_progress_emit(&progress, "video VAE encoder", 0, 1);
            if (progress.cancelled) goto cleanup;
            if (!h3_video_vae_encode(
'''
    if before_video not in core_text:
        if 'h3_progress_emit(&progress, "video VAE encoder", 0, 1)' not in core_text:
            raise SystemExit("h3.c video encode call changed")
    else:
        core_text = core_text.replace(before_video, before_video_new, 1)
        core.write_text(core_text, encoding="utf-8")
    before_vision = '''                    int ok = h3_vision_encode_bf16(
                        text_path, "h3_shaders.metal", pair, 2,
'''
    before_vision_new = '''                    h3_progress_emit(&progress, "Qwen vision", 0, 1);
                    if (progress.cancelled) goto cleanup;
                    int ok = h3_vision_encode_bf16(
                        text_path, "h3_shaders.metal", pair, 2,
'''
    core_text = core.read_text(encoding="utf-8")
    if before_vision not in core_text:
        if 'h3_progress_emit(&progress, "Qwen vision", 0, 1)' not in core_text:
            raise SystemExit("h3.c vision encode call changed")
    else:
        core.write_text(core_text.replace(before_vision, before_vision_new, 1), encoding="utf-8")
    core_text = core.read_text(encoding="utf-8")
    if progress_old not in core_text:
        if "rife60_h3_stage" not in core_text:
            raise SystemExit("h3.c progress emit site changed")
    else:
        core.write_text(core_text.replace(progress_old, progress_new, 1), encoding="utf-8")
    video = root / "h3_video_encoder.c"
    video_old = '''    snprintf(name, sizeof(name), "%s.weight", prefix);
    conv->weight = f5(encoder, name, output_channels, input_channels, kernel,
'''
    video_new = '''    extern void rife60_h3_note(const char *text);
    extern int rife60_h3_cancelled(void);
    char step[224];
    snprintf(step, sizeof(step), "video VAE weight %s", prefix);
    rife60_h3_note(step);
    if (rife60_h3_cancelled()) {
        fail(error, error_size, "generation cancelled during %s", step);
        return 0;
    }
    snprintf(name, sizeof(name), "%s.weight", prefix);
    conv->weight = f5(encoder, name, output_channels, input_channels, kernel,
'''
    video_text = video.read_text(encoding="utf-8")
    if video_old not in video_text:
        if "video VAE weight" not in video_text:
            raise SystemExit("h3_video_encoder.c load_conv site changed")
    else:
        video.write_text(video_text.replace(video_old, video_new, 1), encoding="utf-8")
    vision = root / "h3_vision_encoder.c"
    vision_old = '''    for (int layer = 0; layer < VISION_LAYERS && ok; layer++) {
        vision_block_weights block;
        ok = load_block_weights(weights, gpu, layer, &block,
'''
    vision_new = '''    for (int layer = 0; layer < VISION_LAYERS && ok; layer++) {
        extern void rife60_h3_note(const char *text);
        extern int rife60_h3_cancelled(void);
        char step[64];
        snprintf(step, sizeof(step), "Qwen vision layer %d/%d", layer + 1, VISION_LAYERS);
        rife60_h3_note(step);
        if (rife60_h3_cancelled()) {
            fail(error, error_size, "generation cancelled during %s", step);
            ok = 0;
            break;
        }
        if (progress) progress(layer, VISION_LAYERS, progress_opaque);
        vision_block_weights block;
        ok = load_block_weights(weights, gpu, layer, &block,
'''
    vision_text = vision.read_text(encoding="utf-8")
    if vision_old not in vision_text:
        if "Qwen vision layer" not in vision_text:
            raise SystemExit("h3_vision_encoder.c layer loop changed")
    else:
        vision.write_text(vision_text.replace(vision_old, vision_new, 1), encoding="utf-8")
    gpu_old = '''        for (NSString *name in names) {
            id<MTLFunction> function = [gpu.library newFunctionWithName:name];
'''
    gpu_new = '''        NSUInteger built = 0;
        for (NSString *name in names) {
            extern void rife60_h3_note(const char *text);
            extern int rife60_h3_cancelled(void);
            char step[128];
            snprintf(step, sizeof(step), "Metal pipeline %s %lu/%lu",
                     name.UTF8String, (unsigned long)built, (unsigned long)names.count);
            rife60_h3_note(step);
            if (rife60_h3_cancelled()) {
                if (error && error_size) snprintf(error, error_size, "generation cancelled during %s", step);
                return NULL;
            }
            built++;
            id<MTLFunction> function = [gpu.library newFunctionWithName:name];
'''
    gpu_text = gpu.read_text(encoding="utf-8")
    if gpu_old not in gpu_text:
        if "Metal pipeline" not in gpu_text:
            raise SystemExit("h3_gpu.m pipeline loop changed")
    else:
        gpu.write_text(gpu_text.replace(gpu_old, gpu_new, 1), encoding="utf-8")
    submit_old = '''        [command commit];
        [gpu.inflightCommands addObject:command];
        for (id<MTLCommandBuffer> pending in gpu.inflightCommands)
            [pending waitUntilCompleted];
'''
    submit_new = '''        [command commit];
        [gpu.inflightCommands addObject:command];
        extern int rife60_h3_cancelled(void);
        extern void rife60_h3_note(const char *text);
        for (id<MTLCommandBuffer> pending in gpu.inflightCommands) {
            while (pending.status != MTLCommandBufferStatusCompleted &&
                   pending.status != MTLCommandBufferStatusError) {
                if (rife60_h3_cancelled()) {
                    h3_gpu_set_error(gpu, @"generation cancelled while waiting for Metal");
                    [gpu.inflightCommands removeAllObjects];
                    return 0;
                }
                char step[64];
                snprintf(step, sizeof(step), "Metal wait status %lu",
                         (unsigned long)pending.status);
                rife60_h3_note(step);
                usleep(200000);
            }
        }
'''
    gpu_text = gpu.read_text(encoding="utf-8")
    if submit_old not in gpu_text:
        if "Metal wait status" not in gpu_text:
            raise SystemExit("h3_gpu.m submit wait changed")
    else:
        gpu.write_text(gpu_text.replace(submit_old, submit_new, 1), encoding="utf-8")
    print(f"Patched pinned H3 runtime in {root} for GGUF streaming")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_h3_gguf_runtime.py VENDOR_DIR")
    main()
