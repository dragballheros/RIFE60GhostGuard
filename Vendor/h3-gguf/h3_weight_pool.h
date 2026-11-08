#ifndef H3_WEIGHT_POOL_H
#define H3_WEIGHT_POOL_H

#include <stddef.h>
#include <stdint.h>

/* Fixed shared-storage slots for one layer. Two slots prefetch; memory
 * pressure keeps a single slot so the resident set never grows past the
 * two-slot budget the DiT stream already uses. Scratch is owned by the
 * slot and freed when that slot is recycled. */

typedef struct h3_weight_pool h3_weight_pool;

h3_weight_pool *h3_weight_pool_create(size_t slot_bytes, int want_slots);
void h3_weight_pool_free(h3_weight_pool *pool);
int h3_weight_pool_slot_count(const h3_weight_pool *pool);
void h3_weight_pool_set_pressure(h3_weight_pool *pool, int pressure);
/* Slot written while `compute_slot` is in use. -1 when prefetch must wait. */
int h3_weight_pool_prefetch_slot(const h3_weight_pool *pool, int compute_slot);
void *h3_weight_pool_contents(h3_weight_pool *pool, int slot);
size_t h3_weight_pool_live_bytes(const h3_weight_pool *pool);
size_t h3_weight_pool_reclaimable_bytes(void);
void h3_weight_pool_trim_idle(void);
int h3_weight_pool_bind_scratch(h3_weight_pool *pool, int slot, size_t bytes);
void h3_weight_pool_recycle_slot(h3_weight_pool *pool, int slot);
size_t h3_weight_pool_scratch_bytes(const h3_weight_pool *pool);

/* Second slot only when it fits beside the current one-slot working set. */
int h3_weight_pool_allow_second_slot(uint64_t available_bytes, uint64_t slot_bytes);
uint64_t h3_weight_pool_available_bytes(void);

void h3_weight_pool_layer_boundary(void);

#endif
