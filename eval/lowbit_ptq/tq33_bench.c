/* TQ33 microbench — format tự chế 1.5 bpw cho 2:4-ternary.
 *
 * Block64 = 12 byte: [11 byte = 8 codeword 11-bit LSB-first][1 byte scale-idx].
 * Codeword 11-bit = cặp nhóm-4 (id0*33+id1), nhóm-4 có ≤2 nonzero ∈ {-1,0,+1}
 * => 33 pattern hợp lệ. Giải mã: LUT 1089×8 int8 (17KB — nằm gọn L1/L2).
 * Scale: codebook ≤256 giá trị fp32/tensor (scale bake vốn trên lưới f8).
 *
 * Đo trên tensor THẬT xuất từ ckpt gen4-0.6B:
 *   1) đúng/sai: GEMV fp32-exact vs y_ref (phải khớp ~1e-5)
 *   2) GEMV AVX2 int8-activation (đường inference thật) — sai số lượng tử x
 *   3) tốc độ: ms/GEMV, GB/s hiệu dụng trên buffer nén, so memcpy + fp32 GEMV
 *
 * Build:  python -m ziglang cc -O3 -mavx2 -mfma -o tq33_bench.exe tq33_bench.c
 * Run:    tq33_bench.exe D:\Bit-Translate-data\tq33_bench
 */
#include <immintrin.h>
#include <math.h>
#include <process.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

#define NCOPY 256

static int8_t PATS[33][4];
static int8_t LUT[1089][8]; /* codeword -> 8 trọng số ternary */
static int64_t LUT64[1089]; /* alias int64 cho decode gather */

static void build_tables(void) {
    int n = 0;
    for (int a = -1; a <= 1; a++)
        for (int b = -1; b <= 1; b++)
            for (int c = -1; c <= 1; c++)
                for (int d = -1; d <= 1; d++) {
                    int nz = (a != 0) + (b != 0) + (c != 0) + (d != 0);
                    if (nz <= 2) {
                        PATS[n][0] = (int8_t)a; PATS[n][1] = (int8_t)b;
                        PATS[n][2] = (int8_t)c; PATS[n][3] = (int8_t)d;
                        n++;
                    }
                }
    if (n != 33) { fprintf(stderr, "PATS != 33\n"); exit(1); }
    for (int i0 = 0; i0 < 33; i0++)
        for (int i1 = 0; i1 < 33; i1++) {
            int8_t *o = LUT[i0 * 33 + i1];
            memcpy(o, PATS[i0], 4);
            memcpy(o + 4, PATS[i1], 4);
        }
    memcpy(LUT64, LUT, sizeof LUT64);
}

/* giải 8 codeword 11-bit từ 11 byte -> 64 int8 */
static inline void decode_block(const uint8_t *p, int8_t *t64) {
    uint64_t u0, u1;
    memcpy(&u0, p, 8);
    memcpy(&u1, p + 3, 8);
    for (int k = 0; k < 8; k++) {
        uint32_t code = (k < 5) ? (uint32_t)((u0 >> (11 * k)) & 0x7FF)
                                : (uint32_t)((u1 >> (11 * k - 24)) & 0x7FF);
        memcpy(t64 + 8 * k, LUT[code], 8);
    }
}

/* dot 1 hàng (nb block) với xq int8 — kernel v3: decode LUT->buffer (v1 thắng
 * gather trên core này) + tích lũy float vector, hsum 1 lần/hàng */
static inline float row_dot_avx2(const uint8_t *row, int nb, const int8_t *xq,
                                 const float *xs, const float *table) {
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 facc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 12;
        _mm_prefetch((const char *)(p + 96), _MM_HINT_T0);
        int8_t t64[64] __attribute__((aligned(32)));
        decode_block(p, t64);
        const __m256i t0v = _mm256_load_si256((const __m256i *)t64);
        const __m256i t1v = _mm256_load_si256((const __m256i *)(t64 + 32));
        const __m256i x0 = _mm256_loadu_si256((const __m256i *)(xq + b * 64));
        const __m256i x1 = _mm256_loadu_si256((const __m256i *)(xq + b * 64 + 32));
        /* Σ t*xq:  maddubs(|t|, xq*sign(t)) — t∈{-1,0,1} nên |t|∈{0,1} */
        __m256i p0 = _mm256_maddubs_epi16(_mm256_abs_epi8(t0v), _mm256_sign_epi8(x0, t0v));
        __m256i p1 = _mm256_maddubs_epi16(_mm256_abs_epi8(t1v), _mm256_sign_epi8(x1, t1v));
        __m256i sum = _mm256_add_epi32(_mm256_madd_epi16(p0, ones16),
                                       _mm256_madd_epi16(p1, ones16));
        facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum),
                               _mm256_set1_ps(table[p[11]] * xs[b]), facc);
    }
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(facc), _mm256_extractf128_ps(facc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    return _mm_cvtss_f32(h);
}

/* bench đa luồng: thread t xử lý các copy cp%nt==t (mỗi copy = 1 GEMV đầy đủ) */
static int g_R, g_nb, g_ncopy, g_nt;
static uint8_t *g_big;
static int8_t *g_x;
static float *g_xs;
static float *g_tab;
static volatile float g_sink;

static unsigned __stdcall worker(void *arg) {
    int tid = (int)(intptr_t)arg;
    float local = 0.f;
    size_t stride = (size_t)g_R * g_nb * 12;
    for (int cp = tid; cp < g_ncopy; cp += g_nt) {
        const uint8_t *base = g_big + (size_t)cp * stride;
        for (int r = 0; r < g_R; r++)
            local += row_dot_avx2(base + (size_t)r * g_nb * 12, g_nb, g_x, g_xs, g_tab);
    }
    g_sink += local;
    return 0;
}

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void *xmalloc(size_t n) {
    void *p = _aligned_malloc(n, 64);
    if (!p) { fprintf(stderr, "OOM %zu\n", n); exit(1); }
    return p;
}

static float *read_f32(const char *dir, const char *name, size_t n) {
    char path[512];
    snprintf(path, sizeof path, "%s\\%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "thiếu %s\n", path); exit(1); }
    float *p = (float *)xmalloc(n * 4);
    if (fread(p, 4, n, f) != n) { fprintf(stderr, "đọc hụt %s\n", path); exit(1); }
    fclose(f);
    return p;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "D:\\Bit-Translate-data\\tq33_bench";
    build_tables();

    char path[512];
    snprintf(path, sizeof path, "%s\\meta.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "thiếu meta.txt\n"); return 1; }
    int R, C, G;
    if (fscanf(f, "%d %d %d", &R, &C, &G) != 3) return 1;
    fclose(f);
    int nb = C / 64;
    size_t nblk = (size_t)R * nb;
    printf("tensor %dx%d, %zu block64\n", R, C, nblk);

    /* codes: R*nb*11 byte */
    snprintf(path, sizeof path, "%s\\codes.bin", dir);
    f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "thiếu codes.bin\n"); return 1; }
    uint8_t *codes = (uint8_t *)xmalloc(nblk * 11);
    if (fread(codes, 1, nblk * 11, f) != nblk * 11) return 1;
    fclose(f);
    float *scales = read_f32(dir, "scales.bin", nblk);
    float *x = read_f32(dir, "x.bin", (size_t)C);
    float *y_ref = read_f32(dir, "y_ref.bin", (size_t)R);

    /* codebook scale -> byte (scale bake nằm trên lưới f8 -> ≤256 giá trị) */
    float table[256];
    int ntab = 0;
    uint8_t *sidx = (uint8_t *)xmalloc(nblk);
    for (size_t i = 0; i < nblk; i++) {
        float s = scales[i];
        int j = 0;
        for (; j < ntab; j++) if (table[j] == s) break;
        if (j == ntab) {
            if (ntab == 256) { fprintf(stderr, "scale >256 giá trị — cần f8 grid!\n"); return 1; }
            table[ntab++] = s;
        }
        sidx[i] = (uint8_t)j;
    }
    printf("codebook scale: %d giá trị duy nhất (≤256 ✓)\n", ntab);

    /* buffer nén xen kẽ 12B/block — đúng layout format thật */
    uint8_t *packed = (uint8_t *)xmalloc(nblk * 12);
    for (size_t i = 0; i < nblk; i++) {
        memcpy(packed + i * 12, codes + i * 11, 11);
        packed[i * 12 + 11] = sidx[i];
    }
    double mb = nblk * 12 / 1e6;
    printf("packed: %.2f MB (%.3f bpw)  | fp32 gốc: %.1f MB\n\n",
           mb, nblk * 12 * 8.0 / ((double)R * C), (double)R * C * 4 / 1e6);

    /* ---------- 1) GEMV fp32-exact (kiểm đúng/sai) ---------- */
    float *y = (float *)xmalloc((size_t)R * 4);
    for (int r = 0; r < R; r++) {
        const uint8_t *row = packed + (size_t)r * nb * 12;
        float acc = 0.f;
        for (int b = 0; b < nb; b++) {
            int8_t t64[64];
            decode_block(row + b * 12, t64);
            float s = table[row[b * 12 + 11]];
            float d = 0.f;
            const float *xb = x + b * 64;
            for (int i = 0; i < 64; i++) d += (float)t64[i] * xb[i];
            acc += s * d;
        }
        y[r] = acc;
    }
    double err = 0, ref = 0;
    for (int r = 0; r < R; r++) {
        err += ((double)y[r] - y_ref[r]) * ((double)y[r] - y_ref[r]);
        ref += (double)y_ref[r] * y_ref[r];
    }
    double rel_exact = sqrt(err / (ref + 1e-30));
    printf("[1] GEMV fp32-exact  : rel err = %.2e  %s\n", rel_exact,
           rel_exact < 1e-5 ? "✓ KHỚP y_ref" : "✗ LỆCH!");

    /* ---------- 2) lượng tử hoá x -> int8 per-64 (đường inference thật) ---------- */
    int8_t *xq = (int8_t *)xmalloc((size_t)C);
    float xs[4096];
    for (int b = 0; b < nb; b++) {
        float m = 0.f;
        for (int i = 0; i < 64; i++) { float a = fabsf(x[b * 64 + i]); if (a > m) m = a; }
        float s = m > 0 ? m / 127.f : 1.f;
        xs[b] = s;
        for (int i = 0; i < 64; i++) xq[b * 64 + i] = (int8_t)lrintf(x[b * 64 + i] / s);
    }

    /* ---------- 3) GEMV AVX2 v2 (gather-decode + sign_epi8 dot) ---------- */
    float *y2 = (float *)xmalloc((size_t)R * 4);
    int iters = 200;
    double t0 = now_ms();
    for (int it = 0; it < iters; it++)
        for (int r = 0; r < R; r++)
            y2[r] = row_dot_avx2(packed + (size_t)r * nb * 12, nb, xq, xs, table);
    double dt = (now_ms() - t0) / iters;
    err = 0;
    for (int r = 0; r < R; r++)
        err += ((double)y2[r] - y_ref[r]) * ((double)y2[r] - y_ref[r]);
    double rel_q = sqrt(err / (ref + 1e-30));
    double gbs = nblk * 12 / dt / 1e6;
    printf("[2] GEMV AVX2 int8-x : rel err = %.2e (lượng tử x, kỳ vọng ~1e-2..1e-3)\n", rel_q);
    printf("[3] tốc độ AVX2      : %.3f ms/GEMV  -> %.1f GB/s hiệu dụng (1 luồng)\n", dt, gbs);

    /* ---------- 4) baseline: memcpy + fp32 GEMV ---------- */
    uint8_t *dst = (uint8_t *)xmalloc(nblk * 12);
    t0 = now_ms();
    for (int it = 0; it < iters; it++) memcpy(dst, packed, nblk * 12);
    double dtc = (now_ms() - t0) / iters;
    printf("[4] memcpy buffer nén: %.3f ms -> %.1f GB/s (trần băng thông 1 luồng)\n",
           dtc, nblk * 12 / dtc / 1e6);

    float *Wf = (float *)xmalloc((size_t)R * C * 4);
    for (int r = 0; r < R; r++) {
        const uint8_t *row = packed + (size_t)r * nb * 12;
        for (int b = 0; b < nb; b++) {
            int8_t t64[64];
            decode_block(row + b * 12, t64);
            float s = table[row[b * 12 + 11]];
            for (int i = 0; i < 64; i++) Wf[(size_t)r * C + b * 64 + i] = s * (float)t64[i];
        }
    }
    int it_f = 50;
    t0 = now_ms();
    for (int it = 0; it < it_f; it++)
        for (int r = 0; r < R; r++) {
            const float *w = Wf + (size_t)r * C;
            __m256 acc0 = _mm256_setzero_ps();
            for (int i = 0; i < C; i += 8)
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i), acc0);
            float tmp[8];
            _mm256_storeu_ps(tmp, acc0);
            y[r] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        }
    double dtf = (now_ms() - t0) / it_f;
    printf("[5] fp32 GEMV AVX2   : %.3f ms (%.1f GB/s trên %d MB fp32)\n",
           dtf, (double)R * C * 4 / dtf / 1e6, (int)((double)R * C * 4 / 1e6));
    printf("\n=> TQ33 vs fp32: đọc ít hơn %.1fx, thời gian %.2fx  (memory-bound lý tưởng = %.1fx)\n",
           (double)R * C * 4 / (nblk * 12.0), dtf / dt, (double)R * C * 4 / (nblk * 12.0));

    /* ---------- 6) DRAM-stream + đa luồng: nhân bản tensor NCOPY lần (~151MB) ---------- */
    printf("\n[6] DRAM-stream (%d copy = %.0f MB, đè cache) + đa luồng:\n",
           NCOPY, NCOPY * mb);
    uint8_t *big = (uint8_t *)xmalloc(nblk * 12 * (size_t)NCOPY);
    for (int cp = 0; cp < NCOPY; cp++) memcpy(big + (size_t)cp * nblk * 12, packed, nblk * 12);
    for (int nt = 1; nt <= 6; nt++) {
        g_R = R; g_nb = nb; g_big = big; g_x = xq; g_xs = xs; g_tab = table;
        g_ncopy = NCOPY; g_nt = nt;
        double tb0 = now_ms();
        HANDLE th[8];
        for (int t = 0; t < nt; t++)
            th[t] = (HANDLE)_beginthreadex(NULL, 0, worker, (void *)(intptr_t)t, 0, NULL);
        WaitForMultipleObjects(nt, th, TRUE, INFINITE);
        for (int t = 0; t < nt; t++) CloseHandle(th[t]);
        double dtb = now_ms() - tb0;
        double bytes = (double)nblk * 12 * NCOPY;
        double gbs_t = bytes / dtb / 1e6;
        printf("    %d luồng: %7.1f ms/%d GEMV lớn -> %5.1f GB/s -> 30B-A3B ≈ %4.1f tok/s\n",
               nt, dtb, NCOPY, gbs_t, gbs_t / 0.62);
    }
    return 0;
}
