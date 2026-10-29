#include "h3_gguf.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef H3_GGUF_TEST
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include "h3_gpu.h"
#endif

void h3_gguf_runtime_name(const char *gguf_name, char *out, size_t out_size) {
    const char *layers = "model.layers.";
    size_t layers_length = strlen(layers);
    if (!gguf_name || !out || out_size == 0) return;
    if (strncmp(gguf_name, layers, layers_length) == 0) {
        snprintf(out, out_size, "model.language_model.layers.%s",
                 gguf_name + layers_length);
        return;
    }
    if (strcmp(gguf_name, "model.embed_tokens.weight") == 0) {
        snprintf(out, out_size, "model.language_model.embed_tokens.weight");
        return;
    }
    if (strcmp(gguf_name, "model.norm.weight") == 0) {
        snprintf(out, out_size, "model.language_model.norm.weight");
        return;
    }
    snprintf(out, out_size, "%s", gguf_name);
}

static float h3_f16_to_f32(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t fraction = bits & 0x03ffu;
    uint32_t out;
    if (exponent == 0) {
        if (fraction == 0) {
            out = sign;
        } else {
            exponent = 127 - 14;
            while ((fraction & 0x0400u) == 0) {
                fraction <<= 1;
                exponent--;
            }
            fraction &= 0x03ffu;
            out = sign | (exponent << 23) | (fraction << 13);
        }
    } else if (exponent == 31) {
        out = sign | 0x7f800000u | (fraction << 13);
    } else {
        out = sign | ((exponent + (127 - 15)) << 23) | (fraction << 13);
    }
    float value;
    memcpy(&value, &out, sizeof(value));
    return value;
}

static uint16_t h3_f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t least = (bits >> 16) & 1u;
    uint32_t rounded = bits + 0x7fffu + least;
    return (uint16_t)(rounded >> 16);
}

void h3_gguf_dequant_q8_0(const unsigned char block[34], float out[32]) {
    uint16_t scale_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
    float scale = h3_f16_to_f32(scale_bits);
    for (int index = 0; index < 32; index++) {
        int8_t quant = (int8_t)block[2 + index];
        out[index] = scale * (float)quant;
    }
}

static void h3_q4_scale_min(int group, const unsigned char scales[12],
                            unsigned char *scale, unsigned char *minimum) {
    if (group < 4) {
        *scale = scales[group] & 63;
        *minimum = scales[group + 4] & 63;
        return;
    }
    *scale = (unsigned char)((scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4));
    *minimum = (unsigned char)((scales[group + 4] >> 4) | ((scales[group] >> 6) << 4));
}

void h3_gguf_dequant_q4_k(const unsigned char block[144], float out[256]) {
    uint16_t d_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
    uint16_t min_bits = (uint16_t)block[2] | ((uint16_t)block[3] << 8);
    float delta = h3_f16_to_f32(d_bits);
    float minimum = h3_f16_to_f32(min_bits);
    const unsigned char *scales = block + 4;
    const unsigned char *quants = block + 16;
    int group = 0;
    for (int chunk = 0; chunk < 256; chunk += 64) {
        unsigned char scale0, min0, scale1, min1;
        h3_q4_scale_min(group, scales, &scale0, &min0);
        h3_q4_scale_min(group + 1, scales, &scale1, &min1);
        float d0 = delta * (float)scale0;
        float m0 = minimum * (float)min0;
        float d1 = delta * (float)scale1;
        float m1 = minimum * (float)min1;
        for (int lane = 0; lane < 32; lane++) {
            out[chunk + lane] = d0 * (float)(quants[lane] & 0x0f) - m0;
            out[chunk + 32 + lane] = d1 * (float)(quants[lane] >> 4) - m1;
        }
        quants += 32;
        group += 2;
    }
}

void h3_gguf_dequant_q6_k(const unsigned char block[210], float out[256]) {
    const unsigned char *ql = block;
    const unsigned char *qh = block + 128;
    const int8_t *scales = (const int8_t *)(block + 192);
    uint16_t d_bits = (uint16_t)block[208] | ((uint16_t)block[209] << 8);
    float delta = h3_f16_to_f32(d_bits);
    for (int half = 0; half < 2; half++) {
        for (int lane = 0; lane < 32; lane++) {
            int scale_index = lane / 16;
            int q1 = (int)(ql[lane] & 0x0f) | (((int)(qh[lane] >> 0) & 3) << 4);
            int q2 = (int)(ql[lane + 32] & 0x0f) | (((int)(qh[lane] >> 2) & 3) << 4);
            int q3 = (int)(ql[lane] >> 4) | (((int)(qh[lane] >> 4) & 3) << 4);
            int q4 = (int)(ql[lane + 32] >> 4) | (((int)(qh[lane] >> 6) & 3) << 4);
            float *row = out + half * 128;
            row[lane] = delta * (float)scales[scale_index] * (float)(q1 - 32);
            row[lane + 32] = delta * (float)scales[scale_index + 2] * (float)(q2 - 32);
            row[lane + 64] = delta * (float)scales[scale_index + 4] * (float)(q3 - 32);
            row[lane + 96] = delta * (float)scales[scale_index + 6] * (float)(q4 - 32);
        }
        ql += 64;
        qh += 32;
        scales += 8;
    }
}

#ifdef H3_GGUF_TEST
static int expect_close(float actual, float expected, const char *label) {
    float gap = fabsf(actual - expected);
    if (gap > 1e-4f) {
        fprintf(stderr, "%s got %g expected %g\n", label, actual, expected);
        return 1;
    }
    return 0;
}

static int test_dequant(void) {
    unsigned char q8[34] = {0x00, 0x3c};
    for (int index = 0; index < 32; index++) q8[2 + index] = (unsigned char)(index - 16);
    float q8_out[32];
    h3_gguf_dequant_q8_0(q8, q8_out);
    if (expect_close(q8_out[0], -16.f, "q8[0]")) return 1;
    if (expect_close(q8_out[16], 0.f, "q8[16]")) return 1;
    if (expect_close(q8_out[31], 15.f, "q8[31]")) return 1;

    unsigned char q4[144];
    memset(q4, 0, sizeof(q4));
    q4[0] = 0x00; q4[1] = 0x3c; /* fp16 1.0 */
    q4[2] = 0x00; q4[3] = 0x38; /* fp16 0.5 */
    q4[4] = 2; q4[8] = 1;       /* scale group 0 = 2, min group 0 = 1 */
    for (int index = 0; index < 32; index++) q4[16 + index] = 0x13;
    float q4_out[256];
    h3_gguf_dequant_q4_k(q4, q4_out);
    if (expect_close(q4_out[0], 1.f * 2.f * 3.f - 0.5f * 1.f, "q4[0]")) return 1;
    if (expect_close(q4_out[32], 0.f, "q4[32]")) return 1;

    unsigned char q6[210];
    memset(q6, 0, sizeof(q6));
    q6[0] = 0x21;              /* low nibbles 1 and 2 */
    q6[128] = 0x01;            /* high bits of q1 = 1, so q1 = 0x11 */
    ((int8_t *)q6)[192] = 3;   /* scale 0 */
    q6[208] = 0x00; q6[209] = 0x3c;
    float q6_out[256];
    h3_gguf_dequant_q6_k(q6, q6_out);
    if (expect_close(q6_out[0], 1.f * 3.f * (float)(0x11 - 32), "q6[0]")) return 1;
    return 0;
}

static int test_names(void) {
    FILE *index = fopen("Scripts/fixtures/h3_ref2va_index_keys.txt", "r");
    FILE *gguf = fopen("Scripts/fixtures/h3_ref2va_q8_names.txt", "r");
    if (!index || !gguf) {
        fprintf(stderr, "missing GGUF layout fixtures\n");
        if (index) fclose(index);
        if (gguf) fclose(gguf);
        return 1;
    }
    char (*names)[160] = calloc(600, 160);
    size_t name_count = 0;
    while (name_count < 600 && fgets(names[name_count], 160, gguf)) {
        size_t length = strlen(names[name_count]);
        if (length && names[name_count][length - 1] == '\n') names[name_count][length - 1] = 0;
        name_count++;
    }
    fclose(gguf);
    int missing = 0;
    int keys = 0;
    char line[160];
    char mapped[192];
    while (fgets(line, sizeof(line), index)) {
        size_t length = strlen(line);
        if (length && line[length - 1] == '\n') line[length - 1] = 0;
        if (!line[0]) continue;
        keys++;
        h3_gguf_runtime_name(line, mapped, sizeof(mapped));
        int found = 0;
        for (size_t index_name = 0; index_name < name_count; index_name++) {
            h3_gguf_runtime_name(names[index_name], mapped, sizeof(mapped));
            if (strcmp(mapped, line) == 0) { found = 1; break; }
        }
        if (!found && strncmp(line, "time_embedder.", 14) != 0) {
            fprintf(stderr, "unmapped Ref2VA key %s\n", line);
            missing++;
        }
    }
    fclose(index);
    free(names);
    /* The published Ref2VA safetensors index is 535 keys, not 635.
     * Four time-embedder keys are absent; GGUF stores adaln_t_table instead. */
    if (keys != 535 || name_count != 532 || missing) {
        fprintf(stderr, "layout keys=%d gguf=%zu missing=%d\n", keys, name_count, missing);
        return 1;
    }
    char renamed[192];
    h3_gguf_runtime_name("model.layers.3.self_attn.q_proj.weight", renamed, sizeof(renamed));
    if (strcmp(renamed, "model.language_model.layers.3.self_attn.q_proj.weight") != 0) {
        fprintf(stderr, "qwen map failed: %s\n", renamed);
        return 1;
    }
    return 0;
}

int main(void) {
    if (test_dequant() || test_names()) return 1;
    printf("h3 gguf dequant and 535-key Ref2VA map passed\n");
    return 0;
}
#else

typedef struct {
    char *path;
    uint64_t offset;
    uint32_t ggml_type;
    uint64_t elements;
} h3_gguf_recipe;

static h3_gguf_recipe *recipes;
static size_t recipe_count;

static int h3_gguf_fail(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
    return -1;
}

static int h3_read_full(int fd, void *destination, size_t bytes) {
    unsigned char *out = destination;
    size_t done = 0;
    while (done < bytes) {
        ssize_t count = read(fd, out + done, bytes - done);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return 0;
        done += (size_t)count;
    }
    return 1;
}

static int h3_read_u32(int fd, uint32_t *value) {
    unsigned char bytes[4];
    if (!h3_read_full(fd, bytes, 4)) return 0;
    *value = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
             ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    return 1;
}

static int h3_read_u64(int fd, uint64_t *value) {
    unsigned char bytes[8];
    if (!h3_read_full(fd, bytes, 8)) return 0;
    *value = 0;
    for (int index = 0; index < 8; index++) *value |= (uint64_t)bytes[index] << (8 * index);
    return 1;
}

static int h3_skip_value(int fd, uint32_t type) {
    uint32_t child;
    uint64_t count;
    size_t scalar = 0;
    switch (type) {
    case 0: case 1: case 7: scalar = 1; break;
    case 2: case 3: scalar = 2; break;
    case 4: case 5: case 6: scalar = 4; break;
    case 10: case 11: case 12: scalar = 8; break;
    case 8:
        if (!h3_read_u64(fd, &count)) return 0;
        return lseek(fd, (off_t)count, SEEK_CUR) >= 0;
    case 9:
        if (!h3_read_u32(fd, &child) || !h3_read_u64(fd, &count)) return 0;
        for (uint64_t index = 0; index < count; index++) {
            if (!h3_skip_value(fd, child)) return 0;
        }
        return 1;
    default:
        return 0;
    }
    return lseek(fd, (off_t)scalar, SEEK_CUR) >= 0;
}

static void h3_store_value(void *destination, size_t elements, int want_bf16, float value, size_t index) {
    if (want_bf16) ((uint16_t *)destination)[index] = h3_f32_to_bf16(value);
    else ((float *)destination)[index] = value;
}

static int h3_decode_range(int fd, const h3_gguf_recipe *recipe, void *destination, int want_bf16) {
    if (lseek(fd, (off_t)recipe->offset, SEEK_SET) < 0) return 0;
    size_t done = 0;
    size_t total = (size_t)recipe->elements;
    if (recipe->ggml_type == 0) {
        while (done < total) {
            float value;
            if (!h3_read_full(fd, &value, sizeof(value))) return 0;
            h3_store_value(destination, total, want_bf16, value, done);
            done++;
        }
        return 1;
    }
    if (recipe->ggml_type == 1 || recipe->ggml_type == 30) {
        while (done < total) {
            uint16_t bits;
            unsigned char raw[2];
            if (!h3_read_full(fd, raw, 2)) return 0;
            bits = (uint16_t)raw[0] | ((uint16_t)raw[1] << 8);
            float value = recipe->ggml_type == 1 ? h3_f16_to_f32(bits) : h3_f16_to_f32(0);
            if (recipe->ggml_type == 30) {
                uint32_t wide = (uint32_t)bits << 16;
                memcpy(&value, &wide, sizeof(value));
            }
            h3_store_value(destination, total, want_bf16, value, done);
            done++;
        }
        return 1;
    }
    if (recipe->ggml_type == 8) {
        while (done < total) {
            unsigned char block[34];
            float values[32];
            if (!h3_read_full(fd, block, sizeof(block))) return 0;
            h3_gguf_dequant_q8_0(block, values);
            for (int lane = 0; lane < 32 && done < total; lane++, done++) {
                h3_store_value(destination, total, want_bf16, values[lane], done);
            }
        }
        return 1;
    }
    if (recipe->ggml_type == 12) {
        while (done < total) {
            unsigned char block[144];
            float values[256];
            if (!h3_read_full(fd, block, sizeof(block))) return 0;
            h3_gguf_dequant_q4_k(block, values);
            for (int lane = 0; lane < 256 && done < total; lane++, done++) {
                h3_store_value(destination, total, want_bf16, values[lane], done);
            }
        }
        return 1;
    }
    if (recipe->ggml_type == 14) {
        while (done < total) {
            unsigned char block[210];
            float values[256];
            if (!h3_read_full(fd, block, sizeof(block))) return 0;
            h3_gguf_dequant_q6_k(block, values);
            for (int lane = 0; lane < 256 && done < total; lane++, done++) {
                h3_store_value(destination, total, want_bf16, values[lane], done);
            }
        }
        return 1;
    }
    return 0;
}

int h3_gguf_import_header(const char *path, h3_st_header *header,
                          char *error, size_t error_size) {
    size_t length = path ? strlen(path) : 0;
    if (!path || length < 5 || strcmp(path + length - 5, ".gguf") != 0) return 0;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return h3_gguf_fail(error, error_size, "cannot open GGUF");
    unsigned char magic[4];
    uint32_t version;
    uint64_t tensor_count, metadata_count;
    if (!h3_read_full(fd, magic, 4) || memcmp(magic, "GGUF", 4) != 0 ||
        !h3_read_u32(fd, &version) || version < 2 ||
        !h3_read_u64(fd, &tensor_count) || !h3_read_u64(fd, &metadata_count) ||
        tensor_count > 10000000ull || metadata_count > 10000000ull) {
        close(fd);
        return h3_gguf_fail(error, error_size, "not a readable GGUF header");
    }
    uint64_t alignment = 32;
    for (uint64_t index = 0; index < metadata_count; index++) {
        uint64_t key_length;
        uint32_t value_type;
        if (!h3_read_u64(fd, &key_length) || key_length > 16 * 1024 * 1024ull) {
            close(fd);
            return h3_gguf_fail(error, error_size, "GGUF metadata key is invalid");
        }
        char *key = malloc((size_t)key_length + 1);
        if (!key || !h3_read_full(fd, key, (size_t)key_length) || !h3_read_u32(fd, &value_type)) {
            free(key);
            close(fd);
            return h3_gguf_fail(error, error_size, "truncated GGUF metadata");
        }
        key[key_length] = 0;
        int is_alignment = strcmp(key, "general.alignment") == 0 && value_type == 4;
        free(key);
        if (is_alignment) {
            uint32_t value;
            if (!h3_read_u32(fd, &value) || value == 0) {
                close(fd);
                return h3_gguf_fail(error, error_size, "invalid GGUF alignment");
            }
            alignment = value;
        } else if (!h3_skip_value(fd, value_type)) {
            close(fd);
            return h3_gguf_fail(error, error_size, "cannot skip GGUF metadata");
        }
    }
    memset(header, 0, sizeof(*header));
    header->path = strdup(path);
    header->tensors = calloc((size_t)tensor_count, sizeof(*header->tensors));
    if (!header->path || (tensor_count && !header->tensors)) {
        close(fd);
        h3_st_free_header(header);
        return h3_gguf_fail(error, error_size, "out of memory reading GGUF");
    }
    header->tensor_count = (size_t)tensor_count;
    size_t recipe_begin = recipe_count;
    for (uint64_t index = 0; index < tensor_count; index++) {
        uint64_t name_length;
        uint32_t rank, ggml_type;
        uint64_t relative;
        if (!h3_read_u64(fd, &name_length) || name_length > 1024 * 1024ull) {
            close(fd);
            h3_st_free_header(header);
            return h3_gguf_fail(error, error_size, "invalid GGUF tensor name");
        }
        char *raw_name = malloc((size_t)name_length + 1);
        char mapped[512];
        if (!raw_name || !h3_read_full(fd, raw_name, (size_t)name_length) || !h3_read_u32(fd, &rank) || rank > 8) {
            free(raw_name);
            close(fd);
            h3_st_free_header(header);
            return h3_gguf_fail(error, error_size, "truncated GGUF tensor");
        }
        raw_name[name_length] = 0;
        h3_gguf_runtime_name(raw_name, mapped, sizeof(mapped));
        free(raw_name);
        h3_st_tensor *tensor = &header->tensors[index];
        tensor->name = strdup(mapped);
        tensor->ndim = (int)rank;
        tensor->dtype = H3_DTYPE_BF16;
        uint64_t elements = 1;
        uint64_t dims[8];
        for (uint32_t axis = 0; axis < rank; axis++) {
            if (!h3_read_u64(fd, &dims[axis])) {
                close(fd);
                h3_st_free_header(header);
                return h3_gguf_fail(error, error_size, "truncated GGUF shape");
            }
            elements *= dims[axis];
        }
        for (uint32_t axis = 0; axis < rank; axis++) {
            tensor->shape[axis] = dims[rank - 1 - axis];
        }
        if (!tensor->name || !h3_read_u32(fd, &ggml_type) || !h3_read_u64(fd, &relative)) {
            close(fd);
            h3_st_free_header(header);
            return h3_gguf_fail(error, error_size, "truncated GGUF tensor info");
        }
        if (ggml_type == 0) tensor->dtype = H3_DTYPE_F32;
        tensor->file_offset = relative;
        tensor->data_begin = 0;
        tensor->data_end = elements * (tensor->dtype == H3_DTYPE_F32 ? 4 : 2);
        h3_gguf_recipe *grown = realloc(recipes, (recipe_count + 1) * sizeof(*recipes));
        if (!grown) {
            close(fd);
            h3_st_free_header(header);
            return h3_gguf_fail(error, error_size, "out of memory registering GGUF tensor");
        }
        recipes = grown;
        recipes[recipe_count].path = strdup(path);
        recipes[recipe_count].offset = relative;
        recipes[recipe_count].ggml_type = ggml_type;
        recipes[recipe_count].elements = elements;
        recipe_count++;
    }
    off_t table_end = lseek(fd, 0, SEEK_CUR);
    close(fd);
    struct stat info;
    if (stat(path, &info) != 0) {
        h3_st_free_header(header);
        return h3_gguf_fail(error, error_size, "cannot stat GGUF");
    }
    header->file_size = (uint64_t)info.st_size;
    uint64_t data_offset = ((uint64_t)table_end + alignment - 1) & ~(alignment - 1);
    for (size_t index = 0; index < header->tensor_count; index++) {
        header->tensors[index].file_offset += data_offset;
    }
    for (size_t index = recipe_begin; index < recipe_count; index++) {
        recipes[index].offset += data_offset;
    }
    return 1;
}

int h3_gguf_try_decode(const char *path, uint64_t file_offset, void *destination,
                       size_t elements, int want_bf16,
                       char *error, size_t error_size) {
    if (!path || !destination) return 0;
    const h3_gguf_recipe *recipe = NULL;
    for (size_t index = 0; index < recipe_count; index++) {
        if (recipes[index].offset == file_offset && recipes[index].path &&
            strcmp(recipes[index].path, path) == 0) {
            recipe = &recipes[index];
            break;
        }
    }
    if (!recipe) return 0;
    if (recipe->elements != elements) {
        if (error && error_size) snprintf(error, error_size, "GGUF tensor size mismatch");
        return 1;
    }
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0 || !h3_decode_range(fd, recipe, destination, want_bf16)) {
        if (fd >= 0) close(fd);
        if (error && error_size) snprintf(error, error_size, "cannot stream GGUF tensor");
        return 1;
    }
    close(fd);
    if (error && error_size) error[0] = 0;
    return 1;
}

h3_gpu_tensor *h3_gguf_load_f16_as_f32(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements,
                                       char *error, size_t error_size) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    float *values = malloc(elements * sizeof(*values));
    if (fd < 0 || !values) {
        if (fd >= 0) close(fd);
        free(values);
        if (error && error_size) snprintf(error, error_size, "cannot convert F16 VAE tensor");
        return NULL;
    }
    if (lseek(fd, (off_t)file_offset, SEEK_SET) < 0) {
        close(fd);
        free(values);
        if (error && error_size) snprintf(error, error_size, "cannot seek F16 VAE tensor");
        return NULL;
    }
    for (size_t index = 0; index < elements; index++) {
        unsigned char raw[2];
        if (!h3_read_full(fd, raw, 2)) {
            close(fd);
            free(values);
            if (error && error_size) snprintf(error, error_size, "truncated F16 VAE tensor");
            return NULL;
        }
        uint16_t bits = (uint16_t)raw[0] | ((uint16_t)raw[1] << 8);
        values[index] = h3_f16_to_f32(bits);
    }
    close(fd);
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_f32(gpu, values, elements);
    free(values);
    if (!tensor && error && error_size) snprintf(error, error_size, "cannot upload F16 VAE tensor");
    return tensor;
}

static int h3_link_file(const char *directory, const char *name, const char *source) {
    if (mkdir(directory, 0755) != 0 && errno != EEXIST) return 0;
    char destination[1024];
    snprintf(destination, sizeof(destination), "%s/%s", directory, name);
    if (access(destination, F_OK) == 0) return 1;
    return symlink(source, destination) == 0;
}

static int h3_ensure_dir(const char *path) {
    if (mkdir(path, 0755) == 0 || errno == EEXIST) return 1;
    return 0;
}

static void h3_write_config(const char *path) {
    if (access(path, F_OK) == 0) return;
    FILE *file = fopen(path, "w");
    if (!file) return;
    fputs("{\"architectures\":[\"MiniMaxH3\"]}\n", file);
    fclose(file);
}

void h3_gguf_stage_tier1(const char *model_dir) {
    if (!model_dir || !*model_dir) return;
    const char *roots[] = {model_dir, NULL, NULL};
    char nested[1024];
    snprintf(nested, sizeof(nested), "%s/tier1-weights", model_dir);
    roots[1] = nested;
    const char *found = NULL;
    for (int index = 0; roots[index]; index++) {
        char probe[1024];
        snprintf(probe, sizeof(probe), "%s/minimax_h3_ref2va_pruned-Q8_0.gguf", roots[index]);
        if (access(probe, R_OK) == 0) { found = roots[index]; break; }
    }
    if (!found) return;
    char ref_gguf[1024], qwen[1024], video[1024], audio[1024];
    snprintf(ref_gguf, sizeof(ref_gguf), "%s/minimax_h3_ref2va_pruned-Q8_0.gguf", found);
    snprintf(qwen, sizeof(qwen), "%s/qwen3vl_32b_minimax_h3-Q4_K_M.gguf", found);
    snprintf(video, sizeof(video), "%s/vae_minimax_h3_video_vae_fp16.safetensors", found);
    snprintf(audio, sizeof(audio), "%s/vae_minimax_h3_audio_vae_fp32.safetensors", found);
    char video_nested[1024], audio_nested[1024];
    snprintf(video_nested, sizeof(video_nested), "%s/vae/minimax_h3_video_vae_fp16.safetensors", found);
    snprintf(audio_nested, sizeof(audio_nested), "%s/vae/minimax_h3_audio_vae_fp32.safetensors", found);
    const char *video_source = access(video, R_OK) == 0 ? video : video_nested;
    const char *audio_source = access(audio, R_OK) == 0 ? audio : audio_nested;
    char fl_text[1024], fl_dit[1024], fl_video[1024], fl_audio[1024], fl_tok[1024];
    char ref_text[1024], ref_dit[1024], ref_video[1024], ref_audio[1024];
    snprintf(fl_text, sizeof(fl_text), "%s/FL2VA/text_encoder", model_dir);
    snprintf(fl_dit, sizeof(fl_dit), "%s/FL2VA/transformer", model_dir);
    snprintf(fl_video, sizeof(fl_video), "%s/FL2VA/video_vae/source", model_dir);
    snprintf(fl_audio, sizeof(fl_audio), "%s/FL2VA/audio_vae", model_dir);
    snprintf(fl_tok, sizeof(fl_tok), "%s/FL2VA/tokenizer", model_dir);
    snprintf(ref_text, sizeof(ref_text), "%s/Ref2VA/text_encoder", model_dir);
    snprintf(ref_dit, sizeof(ref_dit), "%s/Ref2VA/transformer", model_dir);
    snprintf(ref_video, sizeof(ref_video), "%s/Ref2VA/video_vae/source", model_dir);
    snprintf(ref_audio, sizeof(ref_audio), "%s/Ref2VA/audio_vae", model_dir);
    h3_ensure_dir(model_dir);
    char parent[1024];
    snprintf(parent, sizeof(parent), "%s/FL2VA", model_dir);
    h3_ensure_dir(parent);
    snprintf(parent, sizeof(parent), "%s/FL2VA/video_vae", model_dir);
    h3_ensure_dir(parent);
    snprintf(parent, sizeof(parent), "%s/Ref2VA", model_dir);
    h3_ensure_dir(parent);
    snprintf(parent, sizeof(parent), "%s/Ref2VA/video_vae", model_dir);
    h3_ensure_dir(parent);
    h3_link_file(fl_dit, "minimax_h3_ref2va_pruned-Q8_0.gguf", ref_gguf);
    h3_link_file(ref_dit, "minimax_h3_ref2va_pruned-Q8_0.gguf", ref_gguf);
    if (access(qwen, R_OK) == 0) {
        h3_link_file(fl_text, "qwen3vl_32b_minimax_h3-Q4_K_M.gguf", qwen);
        h3_link_file(ref_text, "qwen3vl_32b_minimax_h3-Q4_K_M.gguf", qwen);
    }
    if (access(video_source, R_OK) == 0) {
        h3_link_file(fl_video, "minimax_h3_video_vae_fp16.safetensors", video_source);
        h3_link_file(ref_video, "minimax_h3_video_vae_fp16.safetensors", video_source);
    }
    if (access(audio_source, R_OK) == 0) {
        h3_link_file(fl_audio, "minimax_h3_audio_vae_fp32.safetensors", audio_source);
        h3_link_file(ref_audio, "minimax_h3_audio_vae_fp32.safetensors", audio_source);
    }
    char config_path[1024];
    snprintf(config_path, sizeof(config_path), "%s/FL2VA/transformer/config.json", model_dir);
    h3_ensure_dir(fl_dit);
    h3_write_config(config_path);
    (void)fl_tok;
}

#endif
