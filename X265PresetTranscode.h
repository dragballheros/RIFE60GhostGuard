#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* File-to-file preset transcode. Returns 0 on success, 1 if cancelled, and
 * -1 on failure with a message in error_out. The encoder is torn down before
 * return, so the caller does not keep x265 frames beside the RIFE pipeline.
 */
int rife60_x265_preset_transcode(
    const char *input_path,
    const char *output_path,
    volatile int32_t *cancel_flag,
    int (*progress)(int64_t frame, int64_t total, void *userdata),
    void *userdata,
    char *error_out,
    size_t error_out_size);

const char *rife60_x265_preset_params(void);
int rife60_x265_build_number(void);

#ifdef __cplusplus
}
#endif
