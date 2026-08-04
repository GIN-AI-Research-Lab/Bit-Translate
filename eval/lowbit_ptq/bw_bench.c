/* Do bang thong DRAM THAT bang STREAM-style triad da luong (khong tin lai so cu 42GB/s
 * neu chua do bang benchmark chuyen dung) — AVX2, moi luong 1 vung nho rieng du lon de
 * khong vua cache L2/L3 (may B: chua ro dung luong cache chinh xac, dung 512MB/luong cho chac). */
#include <immintrin.h>
#include <process.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

#define SZ_PER_THREAD (256ull * 1024 * 1024)  /* 256MB/mang/luong, 3 mang (a,b,c) */

typedef struct { float *a, *b, *c; size_t n; } Job;
static unsigned __stdcall triad_worker(void *arg) {
    Job *j = (Job *)arg;
    __m256 scalar = _mm256_set1_ps(3.0f);
    for (size_t i = 0; i < j->n; i += 8) {
        __m256 bb = _mm256_load_ps(j->b + i);
        __m256 cc = _mm256_load_ps(j->c + i);
        __m256 aa = _mm256_fmadd_ps(bb, scalar, cc);
        _mm256_store_ps(j->a + i, aa);
    }
    return 0;
}

static double now_ms(void) {
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    int max_threads = argc > 1 ? atoi(argv[1]) : 14;
    size_t n = SZ_PER_THREAD / sizeof(float);

    for (int nt = 1; nt <= max_threads; nt++) {
        Job *jobs = malloc(sizeof(Job) * nt);
        for (int t = 0; t < nt; t++) {
            jobs[t].n = n;
            jobs[t].a = _aligned_malloc(n * sizeof(float), 64);
            jobs[t].b = _aligned_malloc(n * sizeof(float), 64);
            jobs[t].c = _aligned_malloc(n * sizeof(float), 64);
            for (size_t i = 0; i < n; i++) { jobs[t].b[i] = 1.0f; jobs[t].c[i] = 2.0f; }
        }
        int iters = 5;
        double t0 = now_ms();
        for (int it = 0; it < iters; it++) {
            HANDLE h[64];
            for (int t = 0; t < nt; t++)
                h[t] = (HANDLE)_beginthreadex(NULL, 0, triad_worker, &jobs[t], 0, NULL);
            WaitForMultipleObjects(nt, h, TRUE, INFINITE);
            for (int t = 0; t < nt; t++) CloseHandle(h[t]);
        }
        double dt = (now_ms() - t0) / iters;
        double bytes = (double)nt * n * sizeof(float) * 3.0;  /* doc b,c + ghi a */
        double gbs = bytes / dt / 1e6;
        printf("threads=%2d : %7.2f ms -> %7.2f GB/s hieu dung (triad a=b*3+c, %d MB/luong)\n",
               nt, dt, gbs, (int)(SZ_PER_THREAD / 1024 / 1024));
        for (int t = 0; t < nt; t++) {
            _aligned_free(jobs[t].a); _aligned_free(jobs[t].b); _aligned_free(jobs[t].c);
        }
        free(jobs);
    }
    return 0;
}
