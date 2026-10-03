#ifndef GENERATIVE_EDIT_BRIDGE_H
#define GENERATIVE_EDIT_BRIDGE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct GEHandle GEHandle;
GEHandle *ge_create(const char *diffusion, const char *vae, const char *t5);
void ge_destroy(GEHandle *h);
int ge_generate(GEHandle *h, const uint8_t *rgba, int frame_count, int width, int height, int fps, const char *prompt, const char *negative_prompt, int steps, float strength, int64_t seed, uint8_t **out_rgba, int *out_count, int *out_fps);
void ge_free_frames(uint8_t *frames);
void ge_cancel(GEHandle *h);
const char *ge_last_error(void);
#ifdef __cplusplus
}
#endif
#endif
