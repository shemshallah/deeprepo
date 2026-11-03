#define _GNU_SOURCE
#include "rrvm.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/resource.h>

#define PAGE (256u << 10)
#define REGION (192u << 20)      /* 192 MiB virtual */
#define BUDGET (16u << 20)       /* 16 MiB resident cap */
static uint8_t *store;           /* backing for state region */
static _Atomic uint64_t fetch_calls_w, fetch_calls_s;
static inline uint8_t pat(uint64_t o) { return (uint8_t)((o * 2654435761u) >> 13); }

static int wfetch(void *c, uint64_t off, void *buf, size_t n) { (void)c; fetch_calls_w++; for (size_t i = 0; i < n; i++) ((uint8_t *)buf)[i] = pat(off + i); return 0; }
static int sfetch(void *c, uint64_t off, void *buf, size_t n) { (void)c; fetch_calls_s++; memcpy(buf, store + off, n); return 0; }
static int swb(void *c, uint64_t off, const void *buf, size_t n) { (void)c; memcpy(store + off, buf, n); return 0; }

static rrvm *vm; static uint8_t *W, *S; static rr_region *RW, *RS;
static size_t rss_kb(void) { FILE *f = fopen("/proc/self/statm", "r"); long a, b; fscanf(f, "%ld %ld", &a, &b); fclose(f); return b * 4; }

static void *reader(void *arg) {
  unsigned seed = (unsigned)(uintptr_t)arg; for (int i = 0; i < 4000; i++) {
    uint64_t o = ((uint64_t)rand_r(&seed) * 4099) % REGION; if (W[o] != pat(o)) { fprintf(stderr, "WEIGHT MISMATCH @%lu\n", o); exit(2); } }
  return NULL;
}
/* each thread owns a disjoint set of 4K blocks in state region: write unique value, later verify */
static void *writer(void *arg) {
  long id = (long)arg; unsigned seed = id * 77 + 1;
  for (int i = 0; i < 3000; i++) {
    uint64_t blk = ((uint64_t)rand_r(&seed) % (REGION / 4096 / 4)) * 4 + id;   /* blk%4==id */
    uint64_t o = blk * 4096; uint8_t v = (uint8_t)(blk * 31 + 7);
    S[o] = v; S[o + 4095] = v ^ 0xff;
  }
  return NULL;
}
static void *verifier(void *arg) {
  long id = (long)arg; unsigned seed = id * 77 + 1; int bad = 0;
  for (int i = 0; i < 3000; i++) {
    uint64_t blk = ((uint64_t)rand_r(&seed) % (REGION / 4096 / 4)) * 4 + id; uint64_t o = blk * 4096; uint8_t v = (uint8_t)(blk * 31 + 7);
    if (S[o] != v || S[o + 4095] != (uint8_t)(v ^ 0xff)) bad++;
  }
  return (void *)(long)bad;
}

int main(void) {
  store = calloc(1, REGION); vm = rrvm_create(PAGE, BUDGET, 4);
  if (!vm) { printf("FAIL create: %s\n", rrvm_last_error()); return 1; }
  RW = rrvm_map(vm, RR_WEIGHTS, REGION, wfetch, NULL, NULL, (void **)&W);
  RS = rrvm_map(vm, RR_STATE, REGION, sfetch, swb, NULL, (void **)&S);
  assert(RW && RS);
  pthread_t t[8]; size_t base_rss = rss_kb();
  /* 1. weights: 4 threads random reads, content-verified */
  for (long i = 0; i < 4; i++) pthread_create(&t[i], 0, reader, (void *)(i + 1));
  for (int i = 0; i < 4; i++) pthread_join(t[i], 0);
  rr_stats s; rrvm_stats(vm, &s);
  printf("weights: faults=%lu fills=%lu evictions=%lu resident=%luMiB peak=%luMiB over_budget=%lu\n", s.faults, s.fills, s.evictions, s.resident_bytes >> 20, s.peak_resident_bytes >> 20, s.over_budget_fills);
  /* 2. state: writes from 4 threads (zero-fill, WP dirty tracking, writeback under eviction pressure) */
  uint64_t fs0 = fetch_calls_s;
  for (long i = 0; i < 4; i++) pthread_create(&t[i], 0, writer, (void *)i);
  for (int i = 0; i < 4; i++) pthread_join(t[i], 0);
  rrvm_stats(vm, &s);
  printf("state write: zero_fills=%lu wp_faults=%lu dirty_wb=%lu fetch_during_first_touch=%lu resident=%luMiB\n", s.zero_fills, s.wp_faults, s.dirty_writebacks, (unsigned long)(fetch_calls_s - fs0), s.resident_bytes >> 20);
  /* 3. verify state survives eviction round trips (reads refetch persisted pages from store) */
  long bad = 0; for (long i = 0; i < 4; i++) pthread_create(&t[i], 0, verifier, (void *)i);
  for (int i = 0; i < 4; i++) { void *r; pthread_join(t[i], &r); bad += (long)r; }
  rrvm_stats(vm, &s);
  size_t rss = rss_kb();
  printf("state verify: bad=%ld fetches=%lu dirty_wb=%lu peak=%luMiB over_budget=%lu stalls=%lu rss=%zuMiB (base %zuMiB)\n", bad, (unsigned long)fetch_calls_s, s.dirty_writebacks, s.peak_resident_bytes >> 20, s.over_budget_fills, s.stalls, rss / 1024, base_rss / 1024);
  /* 4. drain flushes everything; store must equal logical content */
  rrvm_drain(vm); rrvm_stats(vm, &s);
  int ok = bad == 0 && s.over_budget_fills == 0 && s.peak_resident_bytes <= BUDGET && s.resident_bytes == 0 && s.zero_fills > 0;
  printf("drain: resident=%lu  => %s\n", s.resident_bytes, ok ? "PASS" : "FAIL");
  rrvm_unmap(vm, RW); rrvm_unmap(vm, RS); rrvm_destroy(vm);
  return ok ? 0 : 1;
}
