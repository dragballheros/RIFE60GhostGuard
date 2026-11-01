#include "h3_weight_pool.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
extern void *objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void *);
static void *h3_weight_layer_pool;
#endif
#if defined(__APPLE__) && TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR
#include <os/proc.h>
#define H3_HAS_PROC_AVAILABLE 1
#endif

struct h3_weight_pool {
    int slots;
    int pressure;
    size_t slot_bytes;
    void *storage[2];
    void *scratch[2];
    size_t scratch_bytes[2];
};

h3_weight_pool *h3_weight_pool_create(size_t slot_bytes, int want_slots) {
    if (!slot_bytes || want_slots < 1) return NULL;
    h3_weight_pool *pool = calloc(1, sizeof(*pool));
    if (!pool) return NULL;
    pool->slots = want_slots > 2 ? 2 : want_slots;
    pool->slot_bytes = slot_bytes;
    for (int index = 0; index < pool->slots; index++) {
        pool->storage[index] = calloc(1, slot_bytes);
        if (!pool->storage[index]) {
            h3_weight_pool_free(pool);
            return NULL;
        }
    }
    return pool;
}

void h3_weight_pool_free(h3_weight_pool *pool) {
    if (!pool) return;
    for (int index = 0; index < 2; index++) {
        free(pool->scratch[index]);
        free(pool->storage[index]);
    }
    free(pool);
}

int h3_weight_pool_slot_count(const h3_weight_pool *pool) {
    if (!pool) return 0;
    if (pool->pressure) return 1;
    return pool->slots;
}

void h3_weight_pool_set_pressure(h3_weight_pool *pool, int pressure) {
    if (pool) pool->pressure = pressure ? 1 : 0;
}

int h3_weight_pool_prefetch_slot(const h3_weight_pool *pool, int compute_slot) {
    if (!pool || compute_slot < 0 || compute_slot > 1) return -1;
    if (h3_weight_pool_slot_count(pool) < 2) return -1;
    return compute_slot ^ 1;
}

void *h3_weight_pool_contents(h3_weight_pool *pool, int slot) {
    if (!pool || slot < 0 || slot >= h3_weight_pool_slot_count(pool)) return NULL;
    return pool->storage[slot];
}

size_t h3_weight_pool_live_bytes(const h3_weight_pool *pool) {
    if (!pool) return 0;
    size_t total = (size_t)h3_weight_pool_slot_count(pool) * pool->slot_bytes;
    int limit = h3_weight_pool_slot_count(pool);
    for (int index = 0; index < limit; index++) total += pool->scratch_bytes[index];
    return total;
}

int h3_weight_pool_bind_scratch(h3_weight_pool *pool, int slot, size_t bytes) {
    if (!pool || slot < 0 || slot >= h3_weight_pool_slot_count(pool) || !bytes) return 0;
    free(pool->scratch[slot]);
    pool->scratch[slot] = malloc(bytes);
    if (!pool->scratch[slot]) {
        pool->scratch_bytes[slot] = 0;
        return 0;
    }
    pool->scratch_bytes[slot] = bytes;
    return 1;
}

void h3_weight_pool_recycle_slot(h3_weight_pool *pool, int slot) {
    if (!pool || slot < 0 || slot > 1) return;
    free(pool->scratch[slot]);
    pool->scratch[slot] = NULL;
    pool->scratch_bytes[slot] = 0;
}

size_t h3_weight_pool_scratch_bytes(const h3_weight_pool *pool) {
    if (!pool) return 0;
    return pool->scratch_bytes[0] + pool->scratch_bytes[1];
}

uint64_t h3_weight_pool_available_bytes(void) {
#if defined(H3_HAS_PROC_AVAILABLE)
    return os_proc_available_memory();
#else
    return UINT64_MAX;
#endif
}

int h3_weight_pool_allow_second_slot(uint64_t available_bytes, uint64_t slot_bytes) {
    if (!slot_bytes) return 0;
    /* Keep the second copy only when it fits in memory that is already
     * spare. Three slot-sizes leaves the compute buffer and the rest of
     * the graph room, and never adds a third copy. */
    return available_bytes > slot_bytes * 3;
}

void h3_weight_pool_layer_boundary(void) {
#if defined(__APPLE__)
    if (h3_weight_layer_pool) objc_autoreleasePoolPop(h3_weight_layer_pool);
    h3_weight_layer_pool = objc_autoreleasePoolPush();
#else
#endif
}

#if defined(H3_WEIGHT_POOL_TEST)
#include <stdio.h>

static int fail(const char *message) {
    fprintf(stderr, "weight pool: %s\n", message);
    return 1;
}

int main(void) {
    h3_weight_pool *pool = h3_weight_pool_create(128, 2);
    if (!pool) return fail("create");
    if (h3_weight_pool_slot_count(pool) != 2) return fail("slot count");
    void *first = h3_weight_pool_contents(pool, 0);
    void *second = h3_weight_pool_contents(pool, 1);
    if (!first || !second || first == second) return fail("distinct slots");
    memset(first, 0x5a, 128);
    if (h3_weight_pool_prefetch_slot(pool, 0) != 1) return fail("prefetch order");
    if (h3_weight_pool_prefetch_slot(pool, 1) != 0) return fail("prefetch ping pong");
    if (!h3_weight_pool_bind_scratch(pool, 0, 64)) return fail("scratch");
    if (h3_weight_pool_scratch_bytes(pool) != 64) return fail("scratch bytes");
    h3_weight_pool_recycle_slot(pool, 0);
    if (h3_weight_pool_scratch_bytes(pool) != 0) return fail("scratch leak");
    if (h3_weight_pool_contents(pool, 0) != first) return fail("slot reused");
    if (((unsigned char *)first)[0] != 0x5a) return fail("reused storage cleared");
    h3_weight_pool_set_pressure(pool, 1);
    if (h3_weight_pool_slot_count(pool) != 1) return fail("pressure slots");
    if (h3_weight_pool_prefetch_slot(pool, 0) != -1) return fail("pressure prefetch");
    if (h3_weight_pool_contents(pool, 1) != NULL) return fail("hidden second slot");
    size_t before = h3_weight_pool_live_bytes(pool);
    h3_weight_pool_set_pressure(pool, 0);
    if (h3_weight_pool_live_bytes(pool) < before) return fail("pressure grew the pool");
    if (h3_weight_pool_allow_second_slot(1000, 400) != 0) return fail("tight budget");
    if (h3_weight_pool_allow_second_slot(2000, 400) != 1) return fail("spare budget");
    h3_weight_pool_free(pool);

    /* Reuse must beat allocating a fresh buffer for every layer. */
    enum { layers = 200, bytes = 1 << 20 };
    h3_weight_pool *timed = h3_weight_pool_create(bytes, 2);
    if (!timed) return fail("timed create");
    clock_t reuse_started = clock();
    for (int layer = 0; layer < layers; layer++) {
        int slot = layer & 1;
        unsigned char *dest = h3_weight_pool_contents(timed, slot);
        dest[layer & 255] = (unsigned char)layer;
        h3_weight_pool_recycle_slot(timed, slot);
    }
    double reuse_ms = (double)(clock() - reuse_started) * 1000.0 / CLOCKS_PER_SEC;
    clock_t fresh_started = clock();
    for (int layer = 0; layer < layers; layer++) {
        unsigned char *dest = malloc(bytes);
        if (!dest) return fail("fresh alloc");
        dest[layer & 255] = (unsigned char)layer;
        free(dest);
    }
    double fresh_ms = (double)(clock() - fresh_started) * 1000.0 / CLOCKS_PER_SEC;
    printf("weight pool reuse %.2f ms vs fresh alloc %.2f ms over %d x %d-byte layers\n",
           reuse_ms, fresh_ms, layers, bytes);
    h3_weight_pool_free(timed);
    printf("h3 weight pool reuse, prefetch, and pressure passed\n");
    return 0;
}
#endif
