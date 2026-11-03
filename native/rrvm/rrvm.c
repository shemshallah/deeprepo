/* rrvm: strict-budget demand-paged virtual memory over userfaultfd, with write-protect dirty tracking.
 * Page states: ABSENT -> FILLING -> CLEAN -> (write fault) DIRTY -> EVICTING -> ABSENT.
 * FIFO eviction queue. Budget is strict: a fill only proceeds once a slot is reserved; otherwise the faulting
 * thread stalls (nothing over budget, ever).  See rrvm.h.  */
#define _GNU_SOURCE
#include "rrvm.h"
#include <errno.h>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

enum { ABSENT = 0, FILLING, CLEAN, DIRTY, EVICTING };
#define MAXREG 64
static __thread char errbuf[256];
const char *rrvm_last_error(void) { return errbuf; }
#define FATAL(...) do { fprintf(stderr, "rrvm FATAL: " __VA_ARGS__); fputc('\n', stderr); abort(); } while (0)

struct rr_region {
  int kind, live; char *base; size_t len, npages;
  rr_fetch_fn fetch; rr_wb_fn wb; void *ctx;
  uint8_t *st; uint8_t *persisted;
};
typedef struct { rr_region *r; uint32_t p; } qent;
struct rrvm {
  int uffd, nworkers, stop; size_t page, budget_pages;
  pthread_t *th; pthread_mutex_t mu; pthread_cond_t cv;
  rr_region reg[MAXREG];
  qent *q; size_t qcap, qh, qn;
  size_t used; rr_stats s;
};

static int uffd_open(void) {
  int fd = syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK);        /* kernel-mode faults handled (needs priv) */
  if (fd < 0) fd = syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
  return fd;
}

static void wake(rrvm *vm, char *a, size_t len) {
  struct uffdio_range rg = {.start = (uintptr_t)a, .len = len};
  ioctl(vm->uffd, UFFDIO_WAKE, &rg);
}
static int wp_set(rrvm *vm, char *a, size_t len, int on) {
  struct uffdio_writeprotect w = {.range = {.start = (uintptr_t)a, .len = len}, .mode = on ? UFFDIO_WRITEPROTECT_MODE_WP : 0};
  return ioctl(vm->uffd, UFFDIO_WRITEPROTECT, &w);
}
static rr_region *find(rrvm *vm, uintptr_t a) {
  for (int i = 0; i < MAXREG; i++) { rr_region *r = &vm->reg[i];
    if (r->live && a >= (uintptr_t)r->base && a < (uintptr_t)r->base + r->len) return r; }
  return NULL;
}

/* evict one page that has already been popped from the queue and marked EVICTING (lock NOT held) */
static void evict_io(rrvm *vm, rr_region *r, uint32_t p, int dirty, void *tmp) {
  char *a = r->base + (size_t)p * vm->page;
  if (dirty) {
    if (wp_set(vm, a, vm->page, 1) != 0) FATAL("wp_set failed %s", strerror(errno));   /* freeze writers */
    memcpy(tmp, a, vm->page);
    int ok = -1;
    for (int t = 0; t < 5 && ok; t++) { ok = r->wb(r->ctx, (uint64_t)p * vm->page, tmp, vm->page); if (ok) usleep(100000 << t); }
    if (ok) FATAL("writeback failed, refusing to drop dirty page");
    pthread_mutex_lock(&vm->mu); vm->s.dirty_writebacks++; vm->s.wb_bytes += vm->page; r->persisted[p] = 1; pthread_mutex_unlock(&vm->mu);
  }
  if (madvise(a, vm->page, MADV_DONTNEED) != 0) FATAL("madvise: %s", strerror(errno));
  pthread_mutex_lock(&vm->mu);
  r->st[p] = ABSENT; vm->used--; vm->s.evictions++;
  pthread_cond_broadcast(&vm->cv); pthread_mutex_unlock(&vm->mu);
}

/* lock held on entry/exit; pops head and evicts (drops lock during I/O). returns 1 if evicted */
static int evict_head_locked(rrvm *vm, void *tmp) {
  if (!vm->qn) return 0;
  qent e = vm->q[vm->qh]; vm->qh = (vm->qh + 1) % vm->qcap; vm->qn--;
  int dirty = e.r->st[e.p] == DIRTY; e.r->st[e.p] = EVICTING;
  pthread_mutex_unlock(&vm->mu);
  evict_io(vm, e.r, e.p, dirty, tmp);
  pthread_mutex_lock(&vm->mu);
  return 1;
}

static void handle_missing(rrvm *vm, rr_region *r, uint32_t p, void *buf) {
  char *a = r->base + (size_t)p * vm->page;
  pthread_mutex_lock(&vm->mu); vm->s.faults++;
  for (;;) {
    uint8_t s = r->st[p];
    if (s == FILLING || s == EVICTING) { pthread_cond_wait(&vm->cv, &vm->mu); continue; }
    if (s != ABSENT) { pthread_mutex_unlock(&vm->mu); wake(vm, a, vm->page); return; }   /* already resident */
    break;
  }
  r->st[p] = FILLING;
  while (vm->used >= vm->budget_pages) {
    if (!evict_head_locked(vm, buf)) { vm->s.stalls++; pthread_cond_wait(&vm->cv, &vm->mu); }
  }
  vm->used++;
  pthread_mutex_unlock(&vm->mu);

  uint64_t off = (uint64_t)p * vm->page; int zero = 0;
  if (r->kind == RR_STATE && !r->persisted[p]) { memset(buf, 0, vm->page); zero = 1; }
  else {
    int ok = -1, t;
    for (t = 0; t < 8 && ok; t++) { ok = r->fetch(r->ctx, off, buf, vm->page); if (ok) { usleep(50000 << (t < 6 ? t : 6)); pthread_mutex_lock(&vm->mu); vm->s.fetch_retries++; pthread_mutex_unlock(&vm->mu); } }
    if (ok) FATAL("fetch failed region off=%llu", (unsigned long long)off);
  }
  struct uffdio_copy c = {.dst = (uintptr_t)a, .src = (uintptr_t)buf, .len = vm->page, .mode = r->kind == RR_STATE ? UFFDIO_COPY_MODE_WP : 0};
  if (ioctl(vm->uffd, UFFDIO_COPY, &c) != 0 && errno != EEXIST) FATAL("UFFDIO_COPY %s", strerror(errno));
  pthread_mutex_lock(&vm->mu);
  r->st[p] = CLEAN; vm->q[(vm->qh + vm->qn) % vm->qcap] = (qent){r, p}; vm->qn++;
  vm->s.fills++; if (zero) vm->s.zero_fills++; else vm->s.fetch_bytes += vm->page;
  if (vm->used > vm->budget_pages) vm->s.over_budget_fills++;
  if (vm->used * vm->page > vm->s.peak_resident_bytes) vm->s.peak_resident_bytes = vm->used * vm->page;
  pthread_cond_broadcast(&vm->cv); pthread_mutex_unlock(&vm->mu);
}

static void handle_wp(rrvm *vm, rr_region *r, uint32_t p) {
  char *a = r->base + (size_t)p * vm->page;
  pthread_mutex_lock(&vm->mu); vm->s.wp_faults++;
  while (r->st[p] == EVICTING || r->st[p] == FILLING) pthread_cond_wait(&vm->cv, &vm->mu);
  if (r->st[p] == CLEAN) {
    r->st[p] = DIRTY;
    if (wp_set(vm, a, vm->page, 0) != 0) FATAL("wp clear %s", strerror(errno));    /* also wakes writer */
    pthread_mutex_unlock(&vm->mu); return;
  }
  pthread_mutex_unlock(&vm->mu);
  wake(vm, a, vm->page);          /* absent (evicted meanwhile) or already dirty: writer retries and re-faults if needed */
}

static void *worker(void *arg) {
  rrvm *vm = arg; void *buf = NULL;
  if (posix_memalign(&buf, 4096, vm->page)) FATAL("oom");
  while (!vm->stop) {
    struct pollfd pf = {.fd = vm->uffd, .events = POLLIN};
    if (poll(&pf, 1, 100) <= 0) continue;
    struct uffd_msg m;
    ssize_t n = read(vm->uffd, &m, sizeof m);
    if (n != sizeof m) continue;                       /* EAGAIN: other worker took it */
    if (m.event != UFFD_EVENT_PAGEFAULT) continue;
    uintptr_t a = m.arg.pagefault.address; rr_region *r = find(vm, a);
    if (!r) FATAL("fault outside regions %lx", (unsigned long)a);
    uint32_t p = (a - (uintptr_t)r->base) / vm->page;
    if (m.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WP) handle_wp(vm, r, p); else handle_missing(vm, r, p, buf);
  }
  free(buf); return NULL;
}

rrvm *rrvm_create(size_t page, size_t budget, int nw) {
  if (nw < 2) nw = 2;
  rrvm *vm = calloc(1, sizeof *vm); vm->page = page; vm->budget_pages = budget / page; vm->nworkers = nw;
  if (vm->budget_pages < (size_t)nw * 2) { snprintf(errbuf, sizeof errbuf, "budget too small for workers"); free(vm); return NULL; }
  vm->uffd = uffd_open();
  if (vm->uffd < 0) { snprintf(errbuf, sizeof errbuf, "userfaultfd: %s (need sudo or vm.unprivileged_userfaultfd=1)", strerror(errno)); free(vm); return NULL; }
  struct uffdio_api api = {.api = UFFD_API, .features = UFFD_FEATURE_PAGEFAULT_FLAG_WP};
  if (ioctl(vm->uffd, UFFDIO_API, &api) != 0) { snprintf(errbuf, sizeof errbuf, "UFFDIO_API: %s", strerror(errno)); free(vm); return NULL; }
  pthread_mutex_init(&vm->mu, NULL); pthread_cond_init(&vm->cv, NULL);
  vm->qcap = vm->budget_pages + 8; vm->q = calloc(vm->qcap, sizeof(qent));
  vm->s.budget_bytes = vm->budget_pages * page;
  vm->th = calloc(nw, sizeof(pthread_t));
  for (int i = 0; i < nw; i++) pthread_create(&vm->th[i], NULL, worker, vm);
  return vm;
}

rr_region *rrvm_map(rrvm *vm, int kind, size_t len, rr_fetch_fn fetch, rr_wb_fn wb, void *ctx, void **base_out) {
  len = (len + vm->page - 1) & ~(vm->page - 1);
  pthread_mutex_lock(&vm->mu);
  rr_region *r = NULL; for (int i = 0; i < MAXREG; i++) if (!vm->reg[i].live) { r = &vm->reg[i]; break; }
  if (!r) { pthread_mutex_unlock(&vm->mu); return NULL; }
  memset(r, 0, sizeof *r); r->live = 1;
  pthread_mutex_unlock(&vm->mu);
  /* page-aligned to vm->page so 256 KiB handler pages are naturally aligned */
  size_t al = vm->page;
  char *raw = mmap(NULL, len + al, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (raw == MAP_FAILED) { r->live = 0; return NULL; }
  char *base = (char *)(((uintptr_t)raw + al - 1) & ~(al - 1));
  if (base > raw) munmap(raw, base - raw);
  if (base + len < raw + len + al) munmap(base + len, raw + len + al - (base + len));
  if (mprotect(base, len, kind == RR_STATE ? PROT_READ | PROT_WRITE : PROT_READ) != 0) { r->live = 0; return NULL; }
  r->kind = kind; r->base = base; r->len = len; r->npages = len / vm->page; r->fetch = fetch; r->wb = wb; r->ctx = ctx;
  r->st = calloc(r->npages, 1); r->persisted = calloc(r->npages, 1);
  struct uffdio_register rg = {.range = {.start = (uintptr_t)base, .len = len},
    .mode = UFFDIO_REGISTER_MODE_MISSING | (kind == RR_STATE ? UFFDIO_REGISTER_MODE_WP : 0)};
  if (ioctl(vm->uffd, UFFDIO_REGISTER, &rg) != 0) { snprintf(errbuf, sizeof errbuf, "REGISTER: %s", strerror(errno)); munmap(base, len); r->live = 0; return NULL; }
  *base_out = base; return r;
}

void rrvm_mark_persisted(rr_region *r, uint64_t off, size_t len) {
  size_t ps = r->len / r->npages;
  for (uint64_t p = off / ps; p < (off + len + ps - 1) / ps && p < r->npages; p++) r->persisted[p] = 1;
}

void rrvm_drain(rrvm *vm) {
  void *tmp; posix_memalign(&tmp, 4096, vm->page);
  pthread_mutex_lock(&vm->mu); while (evict_head_locked(vm, tmp)) ; pthread_mutex_unlock(&vm->mu); free(tmp);
}

void rrvm_unmap(rrvm *vm, rr_region *r) {
  rrvm_drain(vm);
  pthread_mutex_lock(&vm->mu);
  for (size_t p = 0; p < r->npages; p++) while (r->st[p] == EVICTING || r->st[p] == FILLING) pthread_cond_wait(&vm->cv, &vm->mu);
  /* pages of r still queued (if any raced in) are removed lazily: rebuild queue without r */
  size_t n = 0; for (size_t i = 0; i < vm->qn; i++) { qent e = vm->q[(vm->qh + i) % vm->qcap];
    if (e.r != r) vm->q[(vm->qh + n++) % vm->qcap] = e; else vm->used--; }
  vm->qn = n;
  pthread_mutex_unlock(&vm->mu);
  struct uffdio_range rg = {.start = (uintptr_t)r->base, .len = r->len};
  ioctl(vm->uffd, UFFDIO_UNREGISTER, &rg); munmap(r->base, r->len);
  free(r->st); free(r->persisted); r->live = 0;
}

void rrvm_stats(rrvm *vm, rr_stats *o) {
  pthread_mutex_lock(&vm->mu); *o = vm->s; o->resident_bytes = vm->used * vm->page; pthread_mutex_unlock(&vm->mu);
}

void rrvm_destroy(rrvm *vm) {
  vm->stop = 1; for (int i = 0; i < vm->nworkers; i++) pthread_join(vm->th[i], NULL);
  close(vm->uffd); free(vm->th); free(vm->q); free(vm);
}
