#ifndef H3_GGUF_H
#define H3_GGUF_H

#include <stddef.h>
#include <stdint.h>

/* Maps a GGUF tensor name onto the key h3_load_dir / the text encoder request. */
void h3_gguf_runtime_name(const char *gguf_name, char *out, size_t out_size);

/* Dequantize one Q8_0 block (34 bytes) to 32 floats. */
void h3_gguf_dequant_q8_0(const unsigned char block[34], float out[32]);

/* Dequantize one Q4_K block (144 bytes) to 256 floats. */
void h3_gguf_dequant_q4_k(const unsigned char block[144], float out[256]);

/* Dequantize one Q6_K block (210 bytes) to 256 floats. Q4_K_M stores some rows this way. */
void h3_gguf_dequant_q6_k(const unsigned char block[210], float out[256]);

#ifndef H3_GGUF_TEST
#include "h3_safetensors.h"

/* 1 if this path was a GGUF and the header is filled, 0 if it is not GGUF, -1 on error. */
int h3_gguf_import_header(const char *path, h3_st_header *header,
                          char *error, size_t error_size);

/* 1 if this offset is a GGUF tensor and the destination was filled (or an error was set).
 * 0 if the offset is ordinary safetensors payload and the caller should read it raw.
 * want_bf16 selects the destination element type. */
int h3_gguf_try_decode(const char *path, uint64_t file_offset, void *destination,
                       size_t elements, int want_bf16,
                       char *error, size_t error_size);

/* Read an F16 safetensors tensor into a newly allocated F32 GPU tensor. */
struct h3_gpu;
typedef struct h3_gpu h3_gpu;
typedef struct h3_gpu_tensor h3_gpu_tensor;
h3_gpu_tensor *h3_gguf_load_f16_as_f32(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements,
                                       char *error, size_t error_size);

/* Point the native FL2VA/Ref2VA directories at the Tier 1 files when they are present. */
void h3_gguf_stage_tier1(const char *model_dir);
#endif

#endif
