#ifndef RRVM_H
#define RRVM_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef int (*rr_fetch_fn)(void *ctx, uint64_t off, void *buf, size_t len);       /* 0 ok */
typedef int (*rr_wb_fn)(void *ctx, uint64_t off, const void *buf, size_t len);    /* 0 ok */
typedef struct rrvm rrvm;
typedef struct rr_region rr_region;
enum { RR_WEIGHTS = 0, RR_STATE = 1 };
typedef struct {
  uint64_t resident_bytes, peak_resident_bytes, budget_bytes;
  uint64_t faults, fills, zero_fills, evictions, dirty_writebacks, stalls, wp_faults;
  uint64_t over_budget_fills, fetch_bytes, wb_bytes, fetch_retries;
} rr_stats;
/* page_bytes: power of 2 multiple of 4096. budget_bytes: hard cap on resident model pages (strict). nworkers>=2. */
rrvm *rrvm_create(size_t page_bytes, size_t budget_bytes, int nworkers);
/* Map a region. RR_WEIGHTS: read-only, always fetched. RR_STATE: RW, write-protect dirty tracking, never-persisted
 * pages are zero-filled with NO fetch, dirty pages are written back on eviction via wb. Returns NULL on error. */
rr_region *rrvm_map(rrvm *vm, int kind, size_t len, rr_fetch_fn fetch, rr_wb_fn wb, void *ctx, void **base_out);
void rrvm_mark_persisted(rr_region *r, uint64_t off, size_t len);   /* state pages that exist in backing store */
void rrvm_unmap(rrvm *vm, rr_region *r);
void rrvm_drain(rrvm *vm);              /* evict everything (writes back dirty). call with no concurrent faulters */
void rrvm_stats(rrvm *vm, rr_stats *out);
void rrvm_destroy(rrvm *vm);
const char *rrvm_last_error(void);
#ifdef __cplusplus
}
#endif
#endif
