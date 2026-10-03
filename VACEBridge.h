#ifndef VACE_BRIDGE_H
#define VACE_BRIDGE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct VACEHandle VACEHandle;
VACEHandle *vace_create(const char *diffusion, const char *vae, const char *t5);
void vace_destroy(VACEHandle *h);
int vace_generate(VACEHandle *h, const uint8_t *rgba, int frame_count, int width, int height, int fps, const char *prompt, const char *negative_prompt, int steps, float strength, int64_t seed, uint8_t **out_rgba, int *out_count, int *out_fps);
void vace_free_frames(uint8_t *frames);
void vace_cancel(VACEHandle *h);
const char *vace_last_error(void);
#ifdef __cplusplus
}
#endif
#endif
