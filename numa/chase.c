#define _GNU_SOURCE
#include <numa.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

static uint64_t s = 88172645463325252ULL;
static uint64_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static double measure(int mem_node, size_t bytes, long steps) {
    size_t n = bytes / sizeof(void *);
    void **buf = numa_alloc_onnode(bytes, mem_node);
    size_t *idx = malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) idx[i] = i;
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = rnd() % (i + 1);
        size_t t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    for (size_t i = 0; i < n; i++) buf[idx[i]] = &buf[idx[(i + 1) % n]];

    void **p = &buf[idx[0]];
    for (long i = 0; i < steps / 10; i++) p = (void **)*p;  /* warm up */
    double t0 = now();
    for (long i = 0; i < steps; i++) p = (void **)*p;
    double t1 = now();
    if (p == NULL) printf("x");
    free(idx);
    numa_free(buf, bytes);
    return (t1 - t0) / steps * 1e9;
}

int main(int argc, char **argv) {
    if (numa_available() < 0) { printf("no NUMA support\n"); return 1; }
    int mem_node = argc > 1 ? atoi(argv[1]) : 0;
    numa_run_on_node(0);
    size_t bytes = 512UL << 20;
    long steps = 300000000;
    printf("CPU node0 -> mem node%d: %.1f ns\n", mem_node, measure(mem_node, bytes, steps));
    return 0;
}
