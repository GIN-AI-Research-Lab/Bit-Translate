/* TQ33 Runner FAST — muc tieu 100-200 tok/s tren Qwen3-0.6B (proof of concept toc do).
 * BAN MOI, KHONG dung den qwen3_runner_tq33.c goc (giu nguyen de so sanh/rollback).
 *
 * Khac biet so voi qwen3_runner_tq33.c (RESEARCH_TQ33_RUNNER.md muc 4.3-4.4 chi ro 2 nut that):
 *  [1] embed_tokens/lm_head int8 per-row (155.58MB/token thay vi 622MB f32) — nut that #1
 *      (63-87%% thoi gian) — kernel int8 x int8-activation moi (row_dot_i8_*).
 *      Input lookup CUNG dung bang int8 (dequant 1 hang/token, ~1KB) -> file model that su
 *      chi can 82.58MB TQ33 + 155.58MB int8 head + norm/bias nho.
 *  [2] Thread-pool SPIN-BARRIER (khong Event/kernel round-trip) + master cung lam viec —
 *      nut that #2 la 196 lan dong bo Event/token an het loi da luong (chi dat 25-30%%
 *      microbench o 6+ luong). Spin ~0.5-2us/lan thay vi ~5-20us.
 *  [3] GOP GEMV: QKV chung 1 dispatch (chung input, quantize 1 lan), gate+up+silu chung 1
 *      dispatch (silu tinh ngay trong worker khi xong hang) -> 5 dispatch/layer + 1 head
 *      = 141/token thay vi 197.
 *  [4] AVX-VNNI-INT8 (VPDPBSSD, do cpuid) cho ca kernel TQ33 lan int8-head — thay chuoi
 *      abs/sign/maddubs/madd (4 lenh) bang 1 lenh dot s8xs8. KET QUA SO HOC GIONG HET
 *      (tich int8 chinh xac, cung nhom 4 phan tu -> cung tong i32) — chi nhanh hon.
 *      AVX-512: cpuid tren may nay (Core Ultra 5 225H, Arrow Lake-H) tra ve avx512f=0 —
 *      KHONG co phan cung, khong phai van de toolchain (da probe truc tiep bang cpuid).
 *  [5] Attention song song theo head + dot/accum AVX2; RoPE dung bang cos/sin tinh truoc
 *      (cung cong thuc tuan tu -> gia tri giong het); quantize activation AVX2 (div + cvt
 *      round-nearest-even, GIONG HET lrintf(x/sc) scalar).
 *
 * Build:  python -m ziglang cc -O3 -mavx2 -mfma -o qwen3_runner_tq33_fast.exe qwen3_runner_tq33_fast.c -lm
 * Run:    qwen3_runner_tq33_fast.exe <weights_f32_dir> <tq33_dir> <embed_int8_dir> <oracle_dir>
 *                                    <out_bin> [n_gen=32] [repeats=3] [threads_csv=1,2,4,6,8,12,14] [flags]
 *         flags: novnni (ep dung duong maddubs AVX2), nocompare (bo Phan A), nobench (bo Phan B)
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

#define N_LAYER 28
#define HIDDEN 1024
#define N_HEAD 16
#define N_KV_HEAD 8
#define HEAD_DIM 128
#define Q_DIM (N_HEAD * HEAD_DIM)     /* 2048 */
#define KV_DIM (N_KV_HEAD * HEAD_DIM) /* 1024 */
#define FFN 3072
#define VOCAB 151936
#define RMS_EPS 1e-6f
#define ROPE_THETA 1000000.0f
#define MAX_TENSORS 600
#define MAX_POS 128
#define MAX_THREADS 16
#define MAX_INDIM 4096

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ================= cpuid probe ================= */
#include <cpuid.h>
static int g_has_avxvnniint8 = 0, g_has_avx512f = 0, g_use_vnni = 0;
static void probe_cpu(void) {
    unsigned int a, b, c, d;
    __cpuid_count(7, 0, a, b, c, d);
    g_has_avx512f = (b >> 16) & 1;
    __cpuid_count(7, 1, a, b, c, d);
    g_has_avxvnniint8 = (d >> 4) & 1; /* AVX-VNNI-INT8: VPDPBSSD s8 x s8 */
    printf("[cpu] avx512f=%d avxvnniint8=%d\n", g_has_avx512f, g_has_avxvnniint8);
}

/* ================= F32 weight store (norm/bias — giong runner goc) ================= */

typedef struct { char name[160]; int ndim; int64_t dim[4]; int64_t offset; int64_t numel; } TensorMeta;
static TensorMeta g_meta[MAX_TENSORS];
static int g_n_meta;
static float *g_w;

static void load_meta(const char *dir) {
    char path[512];
    snprintf(path, sizeof path, "%s\\meta_index.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "khong mo duoc %s\n", path); exit(1); }
    if (fscanf(f, "%d", &g_n_meta) != 1) { fprintf(stderr, "meta_index rong\n"); exit(1); }
    if (g_n_meta > MAX_TENSORS) { fprintf(stderr, "MAX_TENSORS qua nho\n"); exit(1); }
    for (int i = 0; i < g_n_meta; i++) {
        TensorMeta *m = &g_meta[i];
        if (fscanf(f, "%159s %d", m->name, &m->ndim) != 2) exit(1);
        for (int d = 0; d < m->ndim; d++)
            if (fscanf(f, "%lld", (long long *)&m->dim[d]) != 1) exit(1);
        if (fscanf(f, "%lld %lld", (long long *)&m->offset, (long long *)&m->numel) != 2) exit(1);
    }
    fclose(f);
    printf("[f32] da doc meta: %d tensor\n", g_n_meta);
}

static void load_weights(const char *dir) {
    size_t total_elems = 0;
    for (int i = 0; i < g_n_meta; i++) {
        size_t end = (size_t)g_meta[i].offset + (size_t)g_meta[i].numel;
        if (end > total_elems) total_elems = end;
    }
    char path[512];
    snprintf(path, sizeof path, "%s\\weights.bin", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "khong mo duoc %s\n", path); exit(1); }
    printf("[f32] dang doc %s (%.3f GB — chi can norm/bias, embed f32 khong dung nua nhung van "
           "nam trong file dump)...\n", path, total_elems * 4 / 1e9);
    g_w = (float *)malloc(total_elems * sizeof(float));
    if (!g_w) exit(1);
    double t0 = now_ms();
    if (fread(g_w, sizeof(float), total_elems, f) != total_elems) exit(1);
    fclose(f);
    printf("[f32] doc xong (%.1fs)\n", (now_ms() - t0) / 1000.0);
}

static const float *find_t(const char *name) {
    for (int i = 0; i < g_n_meta; i++)
        if (strcmp(g_meta[i].name, name) == 0) return g_w + g_meta[i].offset;
    fprintf(stderr, "KHONG TIM THAY tensor f32: %s\n", name);
    exit(1);
}

static const float *find_layer(int l, const char *suffix) {
    char name[160];
    snprintf(name, sizeof name, "model.layers.%d.%s", l, suffix);
    return find_t(name);
}

/* ================= TQ33 packed store (giong runner goc) ================= */

typedef struct { char name[160]; int64_t R, C, nb, byte_offset, cb_offset; } TQ33Meta;
static TQ33Meta g_tq[MAX_TENSORS];
static int g_n_tq;
static uint8_t *g_packed;
static float *g_codebooks;
static double g_tq33_total_bytes = 0; /* de tinh GB/s hieu dung */

static void load_tq33_meta(const char *dir) {
    char path[512];
    snprintf(path, sizeof path, "%s\\tq33_meta_index.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "khong mo duoc %s\n", path); exit(1); }
    if (fscanf(f, "%d", &g_n_tq) != 1) exit(1);
    if (g_n_tq > MAX_TENSORS) exit(1);
    for (int i = 0; i < g_n_tq; i++) {
        TQ33Meta *m = &g_tq[i];
        if (fscanf(f, "%159s %lld %lld %lld %lld %lld", m->name,
                   (long long *)&m->R, (long long *)&m->C, (long long *)&m->nb,
                   (long long *)&m->byte_offset, (long long *)&m->cb_offset) != 6) exit(1);
        g_tq33_total_bytes += (double)m->R * m->nb * 12;
    }
    fclose(f);
    printf("[tq33] da doc meta: %d tensor (%.2f MB packed doc/token)\n", g_n_tq,
           g_tq33_total_bytes / 1e6);
}

static void load_tq33_data(const char *dir) {
    size_t total_bytes = 0, total_cb = 0;
    for (int i = 0; i < g_n_tq; i++) {
        size_t end = (size_t)g_tq[i].byte_offset + (size_t)g_tq[i].R * g_tq[i].nb * 12;
        if (end > total_bytes) total_bytes = end;
        size_t endc = (size_t)g_tq[i].cb_offset + 256;
        if (endc > total_cb) total_cb = endc;
    }
    char path[512];
    snprintf(path, sizeof path, "%s\\tq33_packed.bin", dir);
    FILE *f = fopen(path, "rb");
    if (!f) exit(1);
    g_packed = (uint8_t *)malloc(total_bytes);
    if (fread(g_packed, 1, total_bytes, f) != total_bytes) exit(1);
    fclose(f);
    snprintf(path, sizeof path, "%s\\tq33_codebooks.bin", dir);
    f = fopen(path, "rb");
    if (!f) exit(1);
    g_codebooks = (float *)malloc(total_cb * sizeof(float));
    if (fread(g_codebooks, sizeof(float), total_cb, f) != total_cb) exit(1);
    fclose(f);
    printf("[tq33] doc xong: %.2f MB packed + %.3f MB codebook\n",
           total_bytes / 1e6, total_cb * 4 / 1e6);
}

static const TQ33Meta *find_tq33_layer(int l, const char *suffix) {
    char name[160];
    snprintf(name, sizeof name, "model.layers.%d.%s.weight", l, suffix);
    for (int i = 0; i < g_n_tq; i++)
        if (strcmp(g_tq[i].name, name) == 0) return &g_tq[i];
    fprintf(stderr, "KHONG TIM THAY tensor tq33: %s\n", name);
    exit(1);
}

/* ================= embed int8 store (MOI — thay embed/lm_head f32) ================= */

static int8_t *g_emb_q;      /* [VOCAB, HIDDEN] row-major */
static float *g_emb_s;       /* [VOCAB] per-row scale */

static void load_embed_int8(const char *dir) {
    char path[512];
    snprintf(path, sizeof path, "%s\\embed_int8_meta.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "khong mo duoc %s\n", path); exit(1); }
    long long rows, cols;
    if (fscanf(f, "%lld %lld", &rows, &cols) != 2) exit(1);
    fclose(f);
    if (rows != VOCAB || cols != HIDDEN) { fprintf(stderr, "embed_int8 shape la\n"); exit(1); }
    g_emb_q = (int8_t *)malloc((size_t)VOCAB * HIDDEN);
    g_emb_s = (float *)malloc((size_t)VOCAB * 4);
    snprintf(path, sizeof path, "%s\\embed_int8.bin", dir);
    f = fopen(path, "rb");
    if (!f) exit(1);
    if (fread(g_emb_q, 1, (size_t)VOCAB * HIDDEN, f) != (size_t)VOCAB * HIDDEN) exit(1);
    fclose(f);
    snprintf(path, sizeof path, "%s\\embed_scales.bin", dir);
    f = fopen(path, "rb");
    if (!f) exit(1);
    if (fread(g_emb_s, 4, VOCAB, f) != VOCAB) exit(1);
    fclose(f);
    printf("[emb-i8] doc xong: %.2f MB int8 + %.2f MB scale\n",
           (double)VOCAB * HIDDEN / 1e6, (double)VOCAB * 4 / 1e6);
}

/* lookup input: dequant 1 hang int8 -> f32 (giong het numpy Q*scale da validate) */
static void embed_lookup_int8(float *out, int32_t token_id) {
    const int8_t *q = g_emb_q + (size_t)token_id * HIDDEN;
    float s = g_emb_s[token_id];
    for (int i = 0; i < HIDDEN; i++) out[i] = (float)q[i] * s;
}

/* ================= kernel TQ33 (decode giong het runner goc) ================= */

static int8_t PATS[33][4];
static int8_t LUT[1089][8];

static void build_tq33_tables(void) {
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
}

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

/* decode 12-byte block -> 2 vector 32B TRUC TIEP TRONG THANH GHI (khong qua buffer t64):
 * duong cu ghi 8x8B store roi load lai 2x32B -> store-to-load-forwarding FAIL (~12+ chu ky
 * stall moi load) trong vong lap nong nhat. set_epi64x tu 8 muc LUT (load 8B + insert) ne
 * hoan toan round-trip bo nho. Cung byte, cung thu tu -> ket qua GIONG HET decode_block
 * (kiem chung runtime bang row_dot_tq33_ref luc khoi dong). */
#define DECODE_REG(p, t0v, t1v)                                                        \
    do {                                                                               \
        uint64_t u0_, u1_;                                                             \
        memcpy(&u0_, (p), 8);                                                          \
        memcpy(&u1_, (p) + 3, 8);                                                      \
        uint64_t v0_, v1_, v2_, v3_, v4_, v5_, v6_, v7_;                               \
        memcpy(&v0_, LUT[u0_ & 0x7FF], 8);                                             \
        memcpy(&v1_, LUT[(u0_ >> 11) & 0x7FF], 8);                                     \
        memcpy(&v2_, LUT[(u0_ >> 22) & 0x7FF], 8);                                     \
        memcpy(&v3_, LUT[(u0_ >> 33) & 0x7FF], 8);                                     \
        memcpy(&v4_, LUT[(u0_ >> 44) & 0x7FF], 8);                                     \
        memcpy(&v5_, LUT[(u1_ >> 31) & 0x7FF], 8);                                     \
        memcpy(&v6_, LUT[(u1_ >> 42) & 0x7FF], 8);                                     \
        memcpy(&v7_, LUT[(u1_ >> 53) & 0x7FF], 8);                                     \
        t0v = _mm256_set_epi64x((int64_t)v3_, (int64_t)v2_, (int64_t)v1_, (int64_t)v0_); \
        t1v = _mm256_set_epi64x((int64_t)v7_, (int64_t)v6_, (int64_t)v5_, (int64_t)v4_); \
    } while (0)

/* duong maddubs AVX2 — cung phep toan int nhu runner goc (da validate) */
static inline float row_dot_tq33_avx2(const uint8_t *row, int nb, const int8_t *xq,
                                      const float *xs, const float *table) {
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 facc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 12;
        _mm_prefetch((const char *)(p + 96), _MM_HINT_T0);
        __m256i t0v, t1v;
        DECODE_REG(p, t0v, t1v);
        const __m256i x0 = _mm256_loadu_si256((const __m256i *)(xq + b * 64));
        const __m256i x1 = _mm256_loadu_si256((const __m256i *)(xq + b * 64 + 32));
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

/* duong VPDPBSSD (AVX-VNNI-INT8): tich s8 x s8 chinh xac, cung nhom-4 -> cung tong i32
 * -> KET QUA float GIONG HET duong maddubs (kiem chung runtime luc khoi dong). */
__attribute__((target("avx2,fma,avxvnniint8")))
static float row_dot_tq33_vnni(const uint8_t *row, int nb, const int8_t *xq,
                               const float *xs, const float *table) {
    __m256 facc = _mm256_setzero_ps();
    const __m256i zero = _mm256_setzero_si256();
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 12;
        _mm_prefetch((const char *)(p + 96), _MM_HINT_T0);
        __m256i t0v, t1v;
        DECODE_REG(p, t0v, t1v);
        const __m256i x0 = _mm256_loadu_si256((const __m256i *)(xq + b * 64));
        const __m256i x1 = _mm256_loadu_si256((const __m256i *)(xq + b * 64 + 32));
        __m256i sum = _mm256_dpbssd_epi32(_mm256_dpbssd_epi32(zero, t0v, x0), t1v, x1);
        facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum),
                               _mm256_set1_ps(table[p[11]] * xs[b]), facc);
    }
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(facc), _mm256_extractf128_ps(facc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    return _mm_cvtss_f32(h);
}

/* ban THAM CHIEU dung decode_block cu (y het qwen3_runner_tq33.c) — chi de kiem chung
 * runtime rang DECODE_REG cho ket qua giong het, khong dung trong duong nong */
static float row_dot_tq33_ref(const uint8_t *row, int nb, const int8_t *xq,
                              const float *xs, const float *table) {
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 facc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = row + b * 12;
        int8_t t64[64] __attribute__((aligned(32)));
        decode_block(p, t64);
        const __m256i t0v = _mm256_load_si256((const __m256i *)t64);
        const __m256i t1v = _mm256_load_si256((const __m256i *)(t64 + 32));
        const __m256i x0 = _mm256_loadu_si256((const __m256i *)(xq + b * 64));
        const __m256i x1 = _mm256_loadu_si256((const __m256i *)(xq + b * 64 + 32));
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

static float (*row_dot_tq33)(const uint8_t *, int, const int8_t *, const float *, const float *)
    = row_dot_tq33_avx2;

/* ================= kernel int8-head (MOI): w int8 per-row x act int8 per-64-group ========= */

static inline float row_dot_i8_avx2(const int8_t *w, int nb, const int8_t *xq, const float *xs) {
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 facc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const int8_t *p = w + b * 64;
        _mm_prefetch((const char *)(p + 256), _MM_HINT_T0);
        const __m256i w0 = _mm256_loadu_si256((const __m256i *)p);
        const __m256i w1 = _mm256_loadu_si256((const __m256i *)(p + 32));
        const __m256i x0 = _mm256_loadu_si256((const __m256i *)(xq + b * 64));
        const __m256i x1 = _mm256_loadu_si256((const __m256i *)(xq + b * 64 + 32));
        /* |w|<=127 (clip luc quantize) nen maddubs khong saturate: 127*127*2 < 32767 */
        __m256i p0 = _mm256_maddubs_epi16(_mm256_abs_epi8(w0), _mm256_sign_epi8(x0, w0));
        __m256i p1 = _mm256_maddubs_epi16(_mm256_abs_epi8(w1), _mm256_sign_epi8(x1, w1));
        __m256i sum = _mm256_add_epi32(_mm256_madd_epi16(p0, ones16),
                                       _mm256_madd_epi16(p1, ones16));
        facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum), _mm256_set1_ps(xs[b]), facc);
    }
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(facc), _mm256_extractf128_ps(facc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    return _mm_cvtss_f32(h);
}

__attribute__((target("avx2,fma,avxvnniint8")))
static float row_dot_i8_vnni(const int8_t *w, int nb, const int8_t *xq, const float *xs) {
    __m256 facc = _mm256_setzero_ps();
    const __m256i zero = _mm256_setzero_si256();
    for (int b = 0; b < nb; b++) {
        const int8_t *p = w + b * 64;
        _mm_prefetch((const char *)(p + 256), _MM_HINT_T0);
        const __m256i w0 = _mm256_loadu_si256((const __m256i *)p);
        const __m256i w1 = _mm256_loadu_si256((const __m256i *)(p + 32));
        const __m256i x0 = _mm256_loadu_si256((const __m256i *)(xq + b * 64));
        const __m256i x1 = _mm256_loadu_si256((const __m256i *)(xq + b * 64 + 32));
        __m256i sum = _mm256_dpbssd_epi32(_mm256_dpbssd_epi32(zero, w0, x0), w1, x1);
        facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum), _mm256_set1_ps(xs[b]), facc);
    }
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(facc), _mm256_extractf128_ps(facc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    return _mm_cvtss_f32(h);
}

static float (*row_dot_i8)(const int8_t *, int, const int8_t *, const float *) = row_dot_i8_avx2;

/* ================= quantize activation int8 per-64-group (AVX2, GIONG lrintf(x/sc)) ========= */

static void quantize_x_int8(int8_t *xq, float *xs, const float *x, int in_dim) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    int nb = in_dim / 64;
    for (int b = 0; b < nb; b++) {
        const float *xb = x + b * 64;
        __m256 mx = _mm256_setzero_ps();
        for (int i = 0; i < 64; i += 8)
            mx = _mm256_max_ps(mx, _mm256_and_ps(_mm256_loadu_ps(xb + i), absmask));
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
        float m = _mm_cvtss_f32(h);
        float sc = m > 0 ? m / 127.0f : 1.0f;
        xs[b] = sc;
        /* x/sc (CHIA that, khong nhan nghich dao — giu giong het lrintf(x[i]/sc) scalar);
         * cvtps_epi32 lam tron nearest-even = lrintf (che do mac dinh MXCSR) */
        __m256 vsc = _mm256_set1_ps(sc);
        for (int i = 0; i < 64; i += 32) {
            __m256i a = _mm256_cvtps_epi32(_mm256_div_ps(_mm256_loadu_ps(xb + i), vsc));
            __m256i b2 = _mm256_cvtps_epi32(_mm256_div_ps(_mm256_loadu_ps(xb + i + 8), vsc));
            __m256i c = _mm256_cvtps_epi32(_mm256_div_ps(_mm256_loadu_ps(xb + i + 16), vsc));
            __m256i d = _mm256_cvtps_epi32(_mm256_div_ps(_mm256_loadu_ps(xb + i + 24), vsc));
            __m256i ab = _mm256_packs_epi32(a, b2);
            __m256i cd = _mm256_packs_epi32(c, d);
            __m256i abcd = _mm256_packs_epi16(ab, cd);
            abcd = _mm256_permutevar8x32_epi32(abcd, perm);
            _mm256_storeu_si256((__m256i *)(xq + b * 64 + i), abcd);
        }
    }
}

/* ================= dot f32 AVX2 (giu cho attention) ================= */

static inline float dot_f32_avx2(const float *a, const float *b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    float s = _mm_cvtss_f32(h);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

/* ================= thread pool SPIN-BARRIER (thay Event cua runner goc) ================= */

typedef void (*RangeFn)(void *ctx, int r0, int r1, int slot);
typedef struct { RangeFn fn; void *ctx; int r0, r1; } Job;

/* GHI CHU thuc nghiem: da THU work-stealing dong (atomic fetch-add grab tung khuc 32 hang)
 * — KEM HON han chia khuc tinh (8T: 26-53 tok/s vs 82-90; 12T sap 0.66 tok/s). Nguyen
 * nhan: (a) cac bien dong bo nong chung cache-line -> fetch-add lam ping-pong line ma moi
 * worker doc trong vong lap grab; (b) khuc 32-hang roi rac pha stream tuan tu/prefetch.
 * => quay lai chia khuc TINH lien tuc (moi luong 1 dai hang lien mach), chi them padding
 * 64B cho g_gen/g_done de spin cua master khong bi worker lam ban line. */
static Job g_jobs[MAX_THREADS];
static volatile long g_gen __attribute__((aligned(64)));
static volatile long g_done __attribute__((aligned(64)));
static volatile long g_quit;
static HANDLE g_th[MAX_THREADS];
static int g_nworkers = 0;  /* so worker (KHONG tinh master) */
static int g_nthreads = 1;  /* tong luong tinh ca master */
static int g_affin = 0;     /* pin luong vao core 0..nt-1 (ne LP-E core 12/13 khi nt<=12) */
static int g_head_maxparts = 0; /* 0 = khong gioi han; >0 = so luong toi da cho head-i8 */

static void steal_run(int slot); /* dinh nghia ben duoi (work-stealing v2) */
static long g_smode_fwd(void);

static unsigned __stdcall pool_worker(void *arg) {
    int tid = (int)(intptr_t)arg;
    long seen = 0;
    for (;;) {
        while (__atomic_load_n(&g_gen, __ATOMIC_ACQUIRE) == seen) _mm_pause();
        seen = __atomic_load_n(&g_gen, __ATOMIC_ACQUIRE);
        if (__atomic_load_n(&g_quit, __ATOMIC_ACQUIRE)) break;
        if (g_smode_fwd()) {
            steal_run(tid);
        } else {
            Job j = g_jobs[tid];
            if (j.r1 > j.r0) j.fn(j.ctx, j.r0, j.r1, tid);
        }
        __atomic_add_fetch(&g_done, 1, __ATOMIC_RELEASE);
    }
    return 0;
}

static void pool_start(int nthreads) {
    g_nthreads = nthreads;
    int nw = nthreads - 1;
    if (nw < 0) nw = 0;
    if (nw > MAX_THREADS) nw = MAX_THREADS;
    g_quit = 0; g_gen = 0; g_done = 0;
    for (int t = 0; t < nw; t++) {
        g_th[t] = (HANDLE)_beginthreadex(NULL, 0, pool_worker, (void *)(intptr_t)t, 0, NULL);
        if (g_affin) SetThreadAffinityMask(g_th[t], 1ull << (t + 1));
    }
    if (g_affin) SetThreadAffinityMask(GetCurrentThread(), 1ull << 0);
    g_nworkers = nw;
}

static void pool_stop(void) {
    if (g_affin) {
        DWORD_PTR pmask, smask;
        if (GetProcessAffinityMask(GetCurrentProcess(), &pmask, &smask))
            SetThreadAffinityMask(GetCurrentThread(), pmask);
    }
    if (g_nworkers == 0) { g_nthreads = 1; return; }
    __atomic_store_n(&g_quit, 1, __ATOMIC_RELEASE);
    __atomic_add_fetch(&g_gen, 1, __ATOMIC_RELEASE);
    WaitForMultipleObjects(g_nworkers, g_th, TRUE, INFINITE);
    for (int t = 0; t < g_nworkers; t++) CloseHandle(g_th[t]);
    g_nworkers = 0; g_nthreads = 1; g_quit = 0;
}

/* chia [0,n) thanh toi da min(g_nthreads, max_parts) khuc lien tuc >= min_chunk;
 * master lam khuc DAU (slot=g_nworkers), worker tid lam khuc tid+1 (slot=tid).
 * Worker khong co viec van tang g_done (de dem du). max_parts=0 nghia la khong gioi han
 * (dung cho head-i8: bang thong DRAM may nay DINH o 4 luong — bw_bench 41GB/s@4T, GIAM
 * dan sau do — nen phan viec thuan stream khong nen dung het luong). */
static void dispatch_mp(RangeFn fn, void *ctx, int n, int min_chunk, int max_parts) {
    int parts = g_nthreads;
    if (max_parts > 0 && parts > max_parts) parts = max_parts;
    if (g_nworkers == 0 || parts <= 1 || n < 2 * min_chunk) { fn(ctx, 0, n, g_nworkers); return; }
    if (parts > n / min_chunk) parts = n / min_chunk;
    if (parts < 1) parts = 1;
    int chunk = (n + parts - 1) / parts;
    for (int t = 0; t < g_nworkers; t++) {
        int p = t + 1;
        int r0 = p * chunk, r1 = r0 + chunk;
        if (r0 > n) r0 = n;
        if (r1 > n) r1 = n;
        g_jobs[t].fn = fn; g_jobs[t].ctx = ctx; g_jobs[t].r0 = r0; g_jobs[t].r1 = r1;
    }
    g_done = 0;
    __atomic_add_fetch(&g_gen, 1, __ATOMIC_RELEASE);
    int m1 = chunk < n ? chunk : n;
    fn(ctx, 0, m1, g_nworkers); /* master tu lam khuc 0 */
    while (__atomic_load_n(&g_done, __ATOMIC_ACQUIRE) != g_nworkers) _mm_pause();
}

/* WORK-STEALING v2 — lan 1 that bai vi false sharing (g_gen/g_done/next/job-desc chung
 * cache line -> fetch-add ping-pong pha line ma moi worker doc moi vong grab) va grain qua
 * nho (32 hang) pha stream lien tuc. v2: moi bien nong 1 line rieng, grain lon hon
 * (64-2048 hang lien tuc), job-desc line rieng chi-doc trong luc chay. Muc dich: tren may
 * hybrid P/E, core P nhanh tu grab nhieu khuc hon — barrier khong con bi gate boi E-core
 * (chia tinh: moi luong khuc bang nhau, ca token doi core cham nhat). */
static struct { RangeFn fn; void *ctx; int n; int grain; } g_sj __attribute__((aligned(64)));
static volatile long g_snext __attribute__((aligned(64)));
static volatile long g_smode __attribute__((aligned(64))); /* 1 = wake nay la steal-job */
static int g_use_steal = 0;
static int g_absteal = 0; /* benchmark xen ke static/steal theo tung lan lap */

static long g_smode_fwd(void) { return __atomic_load_n(&g_smode, __ATOMIC_ACQUIRE); }

static void steal_run(int slot) {
    RangeFn fn = g_sj.fn; void *ctx = g_sj.ctx;
    int n = g_sj.n, grain = g_sj.grain;
    for (;;) {
        long i = __atomic_fetch_add(&g_snext, (long)grain, __ATOMIC_RELAXED);
        if (i >= n) break;
        int r1 = (int)i + grain;
        if (r1 > n) r1 = n;
        fn(ctx, (int)i, r1, slot);
    }
}

static void dispatch_steal(RangeFn fn, void *ctx, int n, int grain) {
    if (g_nworkers == 0 || n <= 2 * grain) { fn(ctx, 0, n, g_nworkers); return; }
    g_sj.fn = fn; g_sj.ctx = ctx; g_sj.n = n; g_sj.grain = grain;
    g_snext = 0;
    g_done = 0;
    __atomic_store_n(&g_smode, 1, __ATOMIC_RELEASE);
    __atomic_add_fetch(&g_gen, 1, __ATOMIC_RELEASE);
    steal_run(g_nworkers);
    while (__atomic_load_n(&g_done, __ATOMIC_ACQUIRE) != g_nworkers) _mm_pause();
    __atomic_store_n(&g_smode, 0, __ATOMIC_RELAXED);
}

static void dispatch(RangeFn fn, void *ctx, int n, int min_chunk) {
    if (g_use_steal) dispatch_steal(fn, ctx, n, min_chunk * 2);
    else dispatch_mp(fn, ctx, n, min_chunk, 0);
}

/* ================= cac range-func ================= */

/* 1 tensor TQ33: out[o] = row_dot + bias[o] */
typedef struct {
    float *out; const int8_t *xq; const float *xs; const float *table;
    const uint8_t *base; size_t stride; int nb; const float *bias;
} TqCtx;

static void tq33_range(void *ctx_, int r0, int r1, int slot) {
    (void)slot;
    TqCtx *c = (TqCtx *)ctx_;
    for (int o = r0; o < r1; o++) {
        float v = row_dot_tq33(c->base + (size_t)o * c->stride, c->nb, c->xq, c->xs, c->table);
        c->out[o] = v + (c->bias ? c->bias[o] : 0.0f);
    }
}

/* QKV gop: hang ao 0..Q_DIM-1 -> Q, tiep KV_DIM -> K, tiep KV_DIM -> V (chung xq/xs) */
typedef struct {
    const int8_t *xq; const float *xs;
    const TQ33Meta *Wq, *Wk, *Wv; const float *bq, *bk, *bv;
    float *q, *k, *v;
} QkvCtx;

static void qkv_range(void *ctx_, int r0, int r1, int slot) {
    (void)slot;
    QkvCtx *c = (QkvCtx *)ctx_;
    for (int o = r0; o < r1; o++) {
        const TQ33Meta *tm; float *out; const float *bias; int local;
        if (o < Q_DIM) { tm = c->Wq; out = c->q; bias = c->bq; local = o; }
        else if (o < Q_DIM + KV_DIM) { tm = c->Wk; out = c->k; bias = c->bk; local = o - Q_DIM; }
        else { tm = c->Wv; out = c->v; bias = c->bv; local = o - Q_DIM - KV_DIM; }
        const uint8_t *base = g_packed + tm->byte_offset;
        const float *table = g_codebooks + tm->cb_offset;
        float v = row_dot_tq33(base + (size_t)local * tm->nb * 12, (int)tm->nb, c->xq, c->xs, table);
        out[local] = v + bias[local];
    }
}

/* gate+up+silu gop: hang ao i in [0,FFN): tinh gate[i], up[i] roi h_ffn[i]=silu(g)*u ngay */
typedef struct {
    const int8_t *xq; const float *xs;
    const TQ33Meta *Wg, *Wu; const float *bg, *bu;
    float *h_ffn;
} GateUpCtx;

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

static void gateup_range(void *ctx_, int r0, int r1, int slot) {
    (void)slot;
    GateUpCtx *c = (GateUpCtx *)ctx_;
    const uint8_t *gb = g_packed + c->Wg->byte_offset;
    const uint8_t *ub = g_packed + c->Wu->byte_offset;
    const float *gt = g_codebooks + c->Wg->cb_offset;
    const float *ut = g_codebooks + c->Wu->cb_offset;
    size_t stride = (size_t)c->Wg->nb * 12;
    int nb = (int)c->Wg->nb;
    for (int i = r0; i < r1; i++) {
        float g = row_dot_tq33(gb + (size_t)i * stride, nb, c->xq, c->xs, gt) + c->bg[i];
        float u = row_dot_tq33(ub + (size_t)i * stride, nb, c->xq, c->xs, ut) + c->bu[i];
        c->h_ffn[i] = silu(g) * u;
    }
}

/* head int8: logits[o] = row_dot_i8 * s[o]; kem argmax cuc bo per-slot */
typedef struct {
    float *logits; const int8_t *xq; const float *xs;
    float best[MAX_THREADS + 1]; int bidx[MAX_THREADS + 1];
} HeadCtx;

static void head_i8_range(void *ctx_, int r0, int r1, int slot) {
    HeadCtx *c = (HeadCtx *)ctx_;
    float best = -1e30f; int bidx = -1;
    int nb = HIDDEN / 64;
    for (int o = r0; o < r1; o++) {
        float v = row_dot_i8(g_emb_q + (size_t)o * HIDDEN, nb, c->xq, c->xs) * g_emb_s[o];
        c->logits[o] = v;
        if (v > best) { best = v; bidx = o; }
    }
    /* 1 slot co the nhan NHIEU khuc (work-stealing) -> hop dan; uu tien index nho khi hoa
       de argmax deterministic giong argmax tuan tu */
    if (bidx >= 0 && (best > c->best[slot] || (best == c->best[slot] && bidx < c->bidx[slot])))
        { c->best[slot] = best; c->bidx[slot] = bidx; }
}

/* attention theo head (KV-cache), dot/accum AVX2 */
typedef struct {
    const float *Qb; const float *K_cache; const float *V_cache;
    float *attn_concat; int l, pos;
} AttnCtx;

static void attn_range(void *ctx_, int h0, int h1, int slot) {
    (void)slot;
    AttnCtx *c = (AttnCtx *)ctx_;
    float scores[MAX_POS];
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (int h = h0; h < h1; h++) {
        int kv_h = h / (N_HEAD / N_KV_HEAD);
        const float *qi = c->Qb + (size_t)h * HEAD_DIM;
        float maxs = -1e30f;
        for (int j = 0; j <= c->pos; j++) {
            const float *kj = c->K_cache + ((size_t)c->l * MAX_POS + j) * KV_DIM + (size_t)kv_h * HEAD_DIM;
            float dot = dot_f32_avx2(qi, kj, HEAD_DIM) * scale;
            scores[j] = dot;
            if (dot > maxs) maxs = dot;
        }
        float sum = 0.0f;
        for (int j = 0; j <= c->pos; j++) { scores[j] = expf(scores[j] - maxs); sum += scores[j]; }
        float inv = 1.0f / sum;
        float *outh = c->attn_concat + (size_t)h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d += 8) _mm256_storeu_ps(outh + d, _mm256_setzero_ps());
        for (int j = 0; j <= c->pos; j++) {
            float wgt = scores[j] * inv;
            const float *vj = c->V_cache + ((size_t)c->l * MAX_POS + j) * KV_DIM + (size_t)kv_h * HEAD_DIM;
            __m256 w8 = _mm256_set1_ps(wgt);
            for (int d = 0; d < HEAD_DIM; d += 8)
                _mm256_storeu_ps(outh + d,
                                 _mm256_fmadd_ps(w8, _mm256_loadu_ps(vj + d), _mm256_loadu_ps(outh + d)));
        }
    }
}

/* ================= phep toan co ban ================= */

static void rmsnorm(float *out, const float *x, const float *w, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float inv = 1.0f / sqrtf((float)(ss / n) + RMS_EPS);
    for (int i = 0; i < n; i++) out[i] = w[i] * (x[i] * inv);
}

/* bang RoPE tinh truoc — CUNG cong thuc tuan tu (theta *= scale) nhu rope_neox goc
 * -> tung gia tri cos/sin GIONG HET tung bit so voi runner goc */
static float g_rope_cos[MAX_POS][HEAD_DIM / 2];
static float g_rope_sin[MAX_POS][HEAD_DIM / 2];

static void build_rope_table(void) {
    float theta_scale = powf(ROPE_THETA, -2.0f / HEAD_DIM);
    for (int pos = 0; pos < MAX_POS; pos++) {
        float theta = (float)pos;
        for (int j = 0; j < HEAD_DIM / 2; j++) {
            g_rope_cos[pos][j] = cosf(theta);
            g_rope_sin[pos][j] = sinf(theta);
            theta *= theta_scale;
        }
    }
}

static void rope_neox_tbl(float *vec, int pos) {
    const float *ct = g_rope_cos[pos], *st = g_rope_sin[pos];
    int half = HEAD_DIM / 2;
    for (int j = 0; j < half; j++) {
        float c = ct[j], s = st[j];
        float x0 = vec[j], x1 = vec[j + half];
        vec[j] = x0 * c - x1 * s;
        vec[j + half] = x0 * s + x1 * c;
    }
}

/* ================= per-layer weight cache ================= */

typedef struct {
    const float *attn_norm_w, *ffn_norm_w, *qn, *kn;
    const float *bq, *bk, *bv, *bo, *bgate, *bup, *bdown;
    const TQ33Meta *Wq, *Wk, *Wv, *Wo, *Wgate, *Wup, *Wdown;
} LayerW;

static void load_layer_weights(LayerW *lw, int l) {
    lw->attn_norm_w = find_layer(l, "input_layernorm.weight");
    lw->ffn_norm_w = find_layer(l, "post_attention_layernorm.weight");
    lw->qn = find_layer(l, "self_attn.q_norm.weight");
    lw->kn = find_layer(l, "self_attn.k_norm.weight");
    lw->bq = find_layer(l, "self_attn.q_proj.bias");
    lw->bk = find_layer(l, "self_attn.k_proj.bias");
    lw->bv = find_layer(l, "self_attn.v_proj.bias");
    lw->bo = find_layer(l, "self_attn.o_proj.bias");
    lw->bgate = find_layer(l, "mlp.gate_proj.bias");
    lw->bup = find_layer(l, "mlp.up_proj.bias");
    lw->bdown = find_layer(l, "mlp.down_proj.bias");
    lw->Wq = find_tq33_layer(l, "self_attn.q_proj");
    lw->Wk = find_tq33_layer(l, "self_attn.k_proj");
    lw->Wv = find_tq33_layer(l, "self_attn.v_proj");
    lw->Wo = find_tq33_layer(l, "self_attn.o_proj");
    lw->Wgate = find_tq33_layer(l, "mlp.gate_proj");
    lw->Wup = find_tq33_layer(l, "mlp.up_proj");
    lw->Wdown = find_tq33_layer(l, "mlp.down_proj");
}

/* ================= cac helper per-token dung chung (compare + bench) ================= */

static int8_t g_xq[MAX_INDIM] __attribute__((aligned(32)));
static float g_xs[MAX_INDIM / 64];

static void fused_qkv(const LayerW *lw, const float *cur, float *Qb, float *Kb, float *Vb) {
    quantize_x_int8(g_xq, g_xs, cur, HIDDEN);
    QkvCtx ctx = {g_xq, g_xs, lw->Wq, lw->Wk, lw->Wv, lw->bq, lw->bk, lw->bv, Qb, Kb, Vb};
    dispatch(qkv_range, &ctx, Q_DIM + KV_DIM + KV_DIM, 32);
}

static void run_o_proj(const LayerW *lw, const float *attn_concat, float *o_out) {
    quantize_x_int8(g_xq, g_xs, attn_concat, Q_DIM);
    TqCtx ctx = {o_out, g_xq, g_xs, g_codebooks + lw->Wo->cb_offset,
                 g_packed + lw->Wo->byte_offset, (size_t)lw->Wo->nb * 12, (int)lw->Wo->nb, lw->bo};
    dispatch(tq33_range, &ctx, HIDDEN, 32);
}

static void fused_gateup_silu(const LayerW *lw, const float *cur, float *h_ffn) {
    quantize_x_int8(g_xq, g_xs, cur, HIDDEN);
    GateUpCtx ctx = {g_xq, g_xs, lw->Wgate, lw->Wup, lw->bgate, lw->bup, h_ffn};
    dispatch(gateup_range, &ctx, FFN, 32);
}

static void run_down_proj(const LayerW *lw, const float *h_ffn, float *down_out) {
    quantize_x_int8(g_xq, g_xs, h_ffn, FFN);
    TqCtx ctx = {down_out, g_xq, g_xs, g_codebooks + lw->Wdown->cb_offset,
                 g_packed + lw->Wdown->byte_offset, (size_t)lw->Wdown->nb * 12,
                 (int)lw->Wdown->nb, lw->bdown};
    dispatch(tq33_range, &ctx, HIDDEN, 32);
}

/* head int8: tra ve argmax; logits ghi day du vao logits_out */
static int head_int8(float *logits_out, const float *final_h) {
    quantize_x_int8(g_xq, g_xs, final_h, HIDDEN);
    HeadCtx ctx;
    ctx.logits = logits_out; ctx.xq = g_xq; ctx.xs = g_xs;
    for (int i = 0; i <= MAX_THREADS; i++) { ctx.best[i] = -1e30f; ctx.bidx[i] = -1; }
    if (g_use_steal) dispatch_steal(head_i8_range, &ctx, VOCAB, 2048);
    else dispatch_mp(head_i8_range, &ctx, VOCAB, 512, g_head_maxparts);
    float best = -1e30f; int bidx = -1;
    for (int i = 0; i <= MAX_THREADS; i++)
        if (ctx.bidx[i] >= 0 && (ctx.best[i] > best || (ctx.best[i] == best && ctx.bidx[i] < bidx))) {
            best = ctx.best[i]; bidx = ctx.bidx[i];
        }
    return bidx;
}

/* ================= Phan A: prefill so khop oracle (khong cache — nhu runner goc) ============ */

static void run_compare(LayerW *LW, const int32_t *tokens, int seq_len,
                        const float *norm_w, const char *out_path) {
    float *hidden = malloc((size_t)seq_len * HIDDEN * 4);
    float *cur = malloc((size_t)seq_len * HIDDEN * 4);
    float *Q = malloc((size_t)seq_len * Q_DIM * 4);
    float *K = malloc((size_t)seq_len * KV_DIM * 4);
    float *V = malloc((size_t)seq_len * KV_DIM * 4);
    float *attn_concat = malloc((size_t)seq_len * Q_DIM * 4);
    float *o_out = malloc((size_t)seq_len * HIDDEN * 4);
    float *ffn_inp = malloc((size_t)seq_len * HIDDEN * 4);
    float *h_ffn = malloc((size_t)seq_len * FFN * 4);
    float *down_out = malloc((size_t)seq_len * HIDDEN * 4);
    float *scores = malloc((size_t)seq_len * 4);

    FILE *of = fopen(out_path, "wb");
    if (!of) exit(1);
    int32_t hdr[4] = {seq_len, HIDDEN, VOCAB, N_LAYER};
    fwrite(hdr, 4, 4, of);

    for (int t = 0; t < seq_len; t++)
        embed_lookup_int8(hidden + (size_t)t * HIDDEN, tokens[t]);

    double t_start = now_ms();
    for (int l = 0; l < N_LAYER; l++) {
        LayerW *lw = &LW[l];
        for (int t = 0; t < seq_len; t++)
            rmsnorm(cur + (size_t)t * HIDDEN, hidden + (size_t)t * HIDDEN, lw->attn_norm_w, HIDDEN);

        for (int t = 0; t < seq_len; t++) {
            fused_qkv(lw, cur + (size_t)t * HIDDEN, Q + (size_t)t * Q_DIM,
                      K + (size_t)t * KV_DIM, V + (size_t)t * KV_DIM);
            for (int h = 0; h < N_HEAD; h++) {
                float *qh = Q + (size_t)t * Q_DIM + (size_t)h * HEAD_DIM;
                rmsnorm(qh, qh, lw->qn, HEAD_DIM);
                rope_neox_tbl(qh, t);
            }
            for (int h = 0; h < N_KV_HEAD; h++) {
                float *kh = K + (size_t)t * KV_DIM + (size_t)h * HEAD_DIM;
                rmsnorm(kh, kh, lw->kn, HEAD_DIM);
                rope_neox_tbl(kh, t);
            }
        }

        /* attention brute-force toan seq (scalar, GIONG HET runner goc — chi chay 1 lan de
         * validate nen khong can toi uu; duong KV-cache ben duoi moi la duong do toc do) */
        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        for (int h = 0; h < N_HEAD; h++) {
            int kv_h = h / (N_HEAD / N_KV_HEAD);
            for (int i = 0; i < seq_len; i++) {
                const float *qi = Q + (size_t)i * Q_DIM + (size_t)h * HEAD_DIM;
                float maxs = -1e30f;
                for (int j = 0; j <= i; j++) {
                    const float *kj = K + (size_t)j * KV_DIM + (size_t)kv_h * HEAD_DIM;
                    float dot = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++) dot += qi[d] * kj[d];
                    dot *= scale;
                    scores[j] = dot;
                    if (dot > maxs) maxs = dot;
                }
                float sum = 0.0f;
                for (int j = 0; j <= i; j++) { scores[j] = expf(scores[j] - maxs); sum += scores[j]; }
                float inv = 1.0f / sum;
                float *outh = attn_concat + (size_t)i * Q_DIM + (size_t)h * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; d++) outh[d] = 0.0f;
                for (int j = 0; j <= i; j++) {
                    float wgt = scores[j] * inv;
                    const float *vj = V + (size_t)j * KV_DIM + (size_t)kv_h * HEAD_DIM;
                    for (int d = 0; d < HEAD_DIM; d++) outh[d] += wgt * vj[d];
                }
            }
        }

        for (int t = 0; t < seq_len; t++)
            run_o_proj(lw, attn_concat + (size_t)t * Q_DIM, o_out + (size_t)t * HIDDEN);

        for (int t = 0; t < seq_len; t++)
            for (int i = 0; i < HIDDEN; i++)
                ffn_inp[(size_t)t * HIDDEN + i] = hidden[(size_t)t * HIDDEN + i] + o_out[(size_t)t * HIDDEN + i];

        for (int t = 0; t < seq_len; t++)
            rmsnorm(cur + (size_t)t * HIDDEN, ffn_inp + (size_t)t * HIDDEN, lw->ffn_norm_w, HIDDEN);

        for (int t = 0; t < seq_len; t++) {
            fused_gateup_silu(lw, cur + (size_t)t * HIDDEN, h_ffn + (size_t)t * FFN);
            run_down_proj(lw, h_ffn + (size_t)t * FFN, down_out + (size_t)t * HIDDEN);
        }

        for (int t = 0; t < seq_len; t++)
            for (int i = 0; i < HIDDEN; i++)
                hidden[(size_t)t * HIDDEN + i] = down_out[(size_t)t * HIDDEN + i] + ffn_inp[(size_t)t * HIDDEN + i];

        fwrite(hidden, 4, (size_t)seq_len * HIDDEN, of);
    }
    printf("[compare] 28 layer xong (%.1fs)\n", (now_ms() - t_start) / 1000.0);

    float *final_h = malloc((size_t)seq_len * HIDDEN * 4);
    for (int t = 0; t < seq_len; t++)
        rmsnorm(final_h + (size_t)t * HIDDEN, hidden + (size_t)t * HIDDEN, norm_w, HIDDEN);
    fwrite(final_h, 4, (size_t)seq_len * HIDDEN, of);

    float *logits = malloc((size_t)VOCAB * 4);
    int last_top1 = -1;
    for (int t = 0; t < seq_len; t++) {
        int am = head_int8(logits, final_h + (size_t)t * HIDDEN);
        fwrite(logits, 4, VOCAB, of);
        if (t == seq_len - 1) {
            last_top1 = am;
            int top_idx[5] = {-1, -1, -1, -1, -1};
            float top_val[5] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
            for (int v = 0; v < VOCAB; v++) {
                float val = logits[v];
                for (int r = 0; r < 5; r++) {
                    if (val > top_val[r]) {
                        for (int s = 4; s > r; s--) { top_val[s] = top_val[s - 1]; top_idx[s] = top_idx[s - 1]; }
                        top_val[r] = val; top_idx[r] = v;
                        break;
                    }
                }
            }
            printf("[compare] top-5 next-token (TQ33+int8-head) tai vi tri cuoi:\n");
            for (int r = 0; r < 5; r++) printf("    id=%6d  logit=%.4f\n", top_idx[r], top_val[r]);
        }
    }
    fclose(of);
    printf("[compare] xong -> %s (top-1 cuoi = %d)\n\n", out_path, last_top1);

    free(hidden); free(cur); free(Q); free(K); free(V); free(attn_concat); free(o_out);
    free(ffn_inp); free(h_ffn); free(down_out); free(scores); free(final_h); free(logits);
}

/* ================= Phan B: benchmark KV-cache autoregressive ================= */

static double g_t_linear = 0, g_t_attn = 0, g_t_norm_rope = 0, g_t_head = 0;

static int forward_one_token(LayerW *LW, int32_t token_id, int pos,
                             const float *norm_w, float *K_cache, float *V_cache,
                             float *logits_out, int want_logits) {
    static float hidden[HIDDEN], cur[HIDDEN], Qb[Q_DIM], Kb[KV_DIM], Vb[KV_DIM];
    static float attn_concat[Q_DIM], o_out[HIDDEN], ffn_inp[HIDDEN];
    static float h_ffn[FFN], down_out[HIDDEN];
    double t0;

    embed_lookup_int8(hidden, token_id);

    for (int l = 0; l < N_LAYER; l++) {
        LayerW *lw = &LW[l];
        t0 = now_ms();
        rmsnorm(cur, hidden, lw->attn_norm_w, HIDDEN);
        g_t_norm_rope += now_ms() - t0;

        t0 = now_ms();
        fused_qkv(lw, cur, Qb, Kb, Vb);
        g_t_linear += now_ms() - t0;

        t0 = now_ms();
        for (int h = 0; h < N_HEAD; h++) {
            float *qh = Qb + (size_t)h * HEAD_DIM;
            rmsnorm(qh, qh, lw->qn, HEAD_DIM);
            rope_neox_tbl(qh, pos);
        }
        for (int h = 0; h < N_KV_HEAD; h++) {
            float *kh = Kb + (size_t)h * HEAD_DIM;
            rmsnorm(kh, kh, lw->kn, HEAD_DIM);
            rope_neox_tbl(kh, pos);
        }
        g_t_norm_rope += now_ms() - t0;

        memcpy(K_cache + ((size_t)l * MAX_POS + pos) * KV_DIM, Kb, KV_DIM * 4);
        memcpy(V_cache + ((size_t)l * MAX_POS + pos) * KV_DIM, Vb, KV_DIM * 4);

        t0 = now_ms();
        AttnCtx actx = {Qb, K_cache, V_cache, attn_concat, l, pos};
        dispatch(attn_range, &actx, N_HEAD, 1);
        g_t_attn += now_ms() - t0;

        t0 = now_ms();
        run_o_proj(lw, attn_concat, o_out);
        g_t_linear += now_ms() - t0;

        for (int i = 0; i < HIDDEN; i++) ffn_inp[i] = hidden[i] + o_out[i];

        t0 = now_ms();
        rmsnorm(cur, ffn_inp, lw->ffn_norm_w, HIDDEN);
        g_t_norm_rope += now_ms() - t0;

        t0 = now_ms();
        fused_gateup_silu(lw, cur, h_ffn);
        run_down_proj(lw, h_ffn, down_out);
        g_t_linear += now_ms() - t0;

        for (int i = 0; i < HIDDEN; i++) hidden[i] = down_out[i] + ffn_inp[i];
    }

    t0 = now_ms();
    rmsnorm(cur, hidden, norm_w, HIDDEN);
    int next_id = -1;
    if (want_logits) next_id = head_int8(logits_out, cur);
    g_t_head += now_ms() - t0;
    return next_id;
}

static void run_benchmark(LayerW *LW, const int32_t *prompt_tokens, int prompt_len,
                          const float *norm_w, int n_gen, const int *thread_list,
                          int n_thread_list, int repeats) {
    float *K_cache = malloc((size_t)N_LAYER * MAX_POS * KV_DIM * 4);
    float *V_cache = malloc((size_t)N_LAYER * MAX_POS * KV_DIM * 4);
    float *logits = malloc((size_t)VOCAB * 4);

    /* byte doc/moi token (uoc luong de tinh GB/s hieu dung):
       tq33 + int8-head + scale-head + norm/bias/qn/kn + kv-cache doc trung binh */
    double biasnorm_bytes = 0;
    biasnorm_bytes += (double)N_LAYER * (HIDDEN + HIDDEN + HEAD_DIM + HEAD_DIM) * 4; /* norms */
    biasnorm_bytes += (double)N_LAYER * (Q_DIM + KV_DIM + KV_DIM + HIDDEN + FFN + FFN + HIDDEN) * 4; /* bias */
    biasnorm_bytes += HIDDEN * 4; /* final norm */
    double head_bytes = (double)VOCAB * HIDDEN + (double)VOCAB * 4;
    double avg_pos = prompt_len + (n_gen - 1) / 2.0;
    double kv_bytes = (double)N_LAYER * (avg_pos + 1) * KV_DIM * 4 * 2 /* doc K+V */
                    + (double)N_LAYER * KV_DIM * 4 * 2;                /* ghi K+V */
    double bytes_per_tok = g_tq33_total_bytes + head_bytes + biasnorm_bytes + kv_bytes + HIDDEN + 4;

    printf("\n================ PHAN B: BENCHMARK (KV-cache, %d token sinh, %d lan lap) ================\n",
           n_gen, repeats);
    printf("byte doc/token (uoc luong): tq33=%.1fMB head-i8=%.1fMB norm/bias=%.1fMB kv~%.1fMB "
           "=> tong ~%.1fMB\n", g_tq33_total_bytes / 1e6, head_bytes / 1e6, biasnorm_bytes / 1e6,
           kv_bytes / 1e6, bytes_per_tok / 1e6);

    for (int ti = 0; ti < n_thread_list; ti++) {
        for (int rep = 0; rep < repeats; rep++) {
            if (g_absteal) g_use_steal = rep % 2; /* xen ke static/steal de trung hoa troi nhiet */
            pool_start(thread_list[ti]);
            g_t_linear = g_t_attn = g_t_norm_rope = g_t_head = 0;

            int pos = 0;
            for (; pos < prompt_len; pos++)
                forward_one_token(LW, prompt_tokens[pos], pos, norm_w, K_cache, V_cache, logits, 0);
            int next = forward_one_token(LW, prompt_tokens[prompt_len - 1], prompt_len - 1,
                                         norm_w, K_cache, V_cache, logits, 1);
            if (ti == 0 && rep == 0)
                printf("[kiem chung KV-cache] token dau tien sinh ra = %d (phai KHOP top-1 cua "
                       "duong compare phia tren)\n", next);
            int cur_tok = next;

            double t0 = now_ms();
            g_t_linear = g_t_attn = g_t_norm_rope = g_t_head = 0;
            int n_done = 0;
            for (int g = 0; g < n_gen; g++) {
                int p = prompt_len + g;
                if (p >= MAX_POS - 1) break;
                cur_tok = forward_one_token(LW, cur_tok, p, norm_w, K_cache, V_cache, logits, 1);
                n_done++;
            }
            double dt = now_ms() - t0;
            double toks = n_done / (dt / 1000.0);
            double total_t = g_t_linear + g_t_attn + g_t_norm_rope + g_t_head;
            double gbs = toks * bytes_per_tok / 1e9;
            printf("threads=%2d run%d %s: %7.1f ms / %d tok -> %6.2f tok/s  (~%5.1f GB/s hieu dung, "
                   "%4.1f%% cua 41GB/s) | tq33=%.0f%% attn=%.0f%% norm=%.0f%% head-i8=%.0f%%\n",
                   thread_list[ti], rep + 1, g_use_steal ? "steal " : "static", dt, n_done, toks,
                   gbs, 100.0 * gbs / 41.0,
                   100.0 * g_t_linear / total_t, 100.0 * g_t_attn / total_t,
                   100.0 * g_t_norm_rope / total_t, 100.0 * g_t_head / total_t);
            pool_stop();
        }
    }
    free(K_cache); free(V_cache); free(logits);
}

/* ================= kiem chung 2 duong kernel (maddubs vs VNNI) cho GIONG HET nhau ========= */

static void verify_kernels_equiv(void) {
    /* int8-head: maddubs vs vnni */
    int8_t w[128], x[128];
    float xs[2] = {0.011f, 0.037f};
    for (int i = 0; i < 128; i++) { w[i] = (int8_t)((i * 37 % 255) - 127); x[i] = (int8_t)((i * 91 % 255) - 127); }
    float a = row_dot_i8_avx2(w, 2, x, xs);
    float b = g_has_avxvnniint8 ? row_dot_i8_vnni(w, 2, x, xs) : a;
    /* tq33: ref (decode_block cu) vs DECODE_REG avx2 vs DECODE_REG vnni, tren NHIEU hang
       THAT cua nhieu tensor (100 hang dau cua 3 tensor) */
    int n_bad = 0;
    for (int ti = 0; ti < 3; ti++) {
        const TQ33Meta *tm = &g_tq[ti * 60]; /* rai deu cac layer */
        float xs2[64];
        static int8_t x2[MAX_INDIM];
        for (int i = 0; i < tm->C; i++) x2[i] = (int8_t)((i * 53 % 255) - 127);
        for (int i = 0; i < tm->C / 64; i++) xs2[i] = 0.01f + 0.001f * i;
        for (int r = 0; r < 100 && r < tm->R; r++) {
            const uint8_t *row = g_packed + tm->byte_offset + (size_t)r * tm->nb * 12;
            float vref = row_dot_tq33_ref(row, (int)tm->nb, x2, xs2, g_codebooks + tm->cb_offset);
            float v1 = row_dot_tq33_avx2(row, (int)tm->nb, x2, xs2, g_codebooks + tm->cb_offset);
            float v2 = g_has_avxvnniint8
                     ? row_dot_tq33_vnni(row, (int)tm->nb, x2, xs2, g_codebooks + tm->cb_offset)
                     : v1;
            if (v1 != vref || v2 != vref) n_bad++;
        }
    }
    printf("[kernel-check] i8 maddubs=%.6f vnni=%.6f %s | tq33 decode-reg vs decode_block "
           "(300 hang that): %s\n", a, b, a == b ? "KHOP" : "LECH!",
           n_bad == 0 ? "KHOP tuyet doi" : "LECH!");
    if (a != b || n_bad) { fprintf(stderr, "kernel LECH — dung lai\n"); exit(1); }
}

/* ================= kernel A/B microbench (1 luong, XEN KE de trung hoa troi nhiet) ====== */

static void kernel_bench(LayerW *LW) {
    static int8_t xq[MAX_INDIM] __attribute__((aligned(32)));
    static float xs[MAX_INDIM / 64];
    static float xf[MAX_INDIM];
    for (int i = 0; i < MAX_INDIM; i++) xf[i] = sinf(i * 0.01f);
    typedef float (*K)(const uint8_t *, int, const int8_t *, const float *, const float *);
    K ks[3] = {row_dot_tq33_ref, row_dot_tq33_avx2, row_dot_tq33_vnni};
    const char *kn[3] = {"decode_block+maddubs (kernel goc)", "decode-reg+maddubs",
                         "decode-reg+VNNI"};
    double acc_ms[3] = {0, 0, 0};
    double bytes_pass = 0;
    volatile float sink = 0;
    int n_rep = 8;
    for (int rep = 0; rep < n_rep; rep++) {
        for (int v = 0; v < 3; v++) {
            if (v == 2 && !g_has_avxvnniint8) continue;
            double t0 = now_ms();
            for (int l = 10; l < 18; l++) {
                LayerW *lw = &LW[l];
                const TQ33Meta *ts[7] = {lw->Wq, lw->Wk, lw->Wv, lw->Wo, lw->Wgate, lw->Wup, lw->Wdown};
                for (int t = 0; t < 7; t++) {
                    const TQ33Meta *tm = ts[t];
                    quantize_x_int8(xq, xs, xf, (int)tm->C);
                    const uint8_t *base = g_packed + tm->byte_offset;
                    const float *tab = g_codebooks + tm->cb_offset;
                    for (int r = 0; r < tm->R; r++)
                        sink += ks[v](base + (size_t)r * tm->nb * 12, (int)tm->nb, xq, xs, tab);
                    if (rep == 0 && v == 0) bytes_pass += (double)tm->R * tm->nb * 12;
                }
            }
            acc_ms[v] += now_ms() - t0;
        }
    }
    printf("\n[kbench] 1 luong, 8 layer x 7 tensor (%.1f MB/luot), %d luot XEN KE:\n",
           bytes_pass / 1e6, n_rep);
    for (int v = 0; v < 3; v++) {
        if (v == 2 && !g_has_avxvnniint8) continue;
        printf("  %-36s: %7.1f ms tong -> %5.2f GB/s\n", kn[v], acc_ms[v],
               bytes_pass * n_rep / (acc_ms[v] / 1000.0) / 1e9);
    }
    printf("  (sink=%.3f — chi de chan dead-code-elimination)\n", (float)sink);
}

/* ================= main ================= */

int main(int argc, char **argv) {
    const char *wdir = argc > 1 ? argv[1] : "D:\\Bit-Translate-data\\tq33_runner\\weights_f32";
    const char *tqdir = argc > 2 ? argv[2] : "D:\\Bit-Translate-data\\tq33_runner\\tq33_packed";
    const char *edir = argc > 3 ? argv[3] : "D:\\Bit-Translate-data\\tq33_runner\\embed_int8";
    const char *odir = argc > 4 ? argv[4] : "D:\\Bit-Translate-data\\tq33_runner\\oracle";
    const char *out_path = argc > 5 ? argv[5] : "D:\\Bit-Translate-data\\tq33_runner\\runner_output_tq33_fast.bin";
    int n_gen = argc > 6 ? atoi(argv[6]) : 32;
    int repeats = argc > 7 ? atoi(argv[7]) : 3;
    const char *tcsv = argc > 8 ? argv[8] : "1,2,4,6,8,12,14";
    const char *flags = argc > 9 ? argv[9] : "";

    probe_cpu();
    g_use_vnni = g_has_avxvnniint8 && !strstr(flags, "novnni");
    g_affin = strstr(flags, "affin") != NULL;
    const char *hc = strstr(flags, "headcap");
    if (hc) g_head_maxparts = atoi(hc + 7);
    if (strstr(flags, "absteal")) g_absteal = 1;
    else if (strstr(flags, "steal")) g_use_steal = 1;
    printf("[cfg] affin=%d head_maxparts=%d steal=%d absteal=%d\n",
           g_affin, g_head_maxparts, g_use_steal, g_absteal);

    build_tq33_tables();
    build_rope_table();
    load_meta(wdir);
    load_weights(wdir);
    load_tq33_meta(tqdir);
    load_tq33_data(tqdir);
    load_embed_int8(edir);

    verify_kernels_equiv();
    if (g_use_vnni) {
        row_dot_tq33 = row_dot_tq33_vnni;
        row_dot_i8 = row_dot_i8_vnni;
        printf("[kernel] dung AVX-VNNI-INT8 (VPDPBSSD) — ket qua so hoc da kiem chung GIONG HET maddubs\n");
    } else {
        printf("[kernel] dung AVX2 maddubs (%s)\n",
               g_has_avxvnniint8 ? "bi ep boi flag novnni" : "CPU khong co avxvnniint8");
    }

    char tpath[512];
    snprintf(tpath, sizeof tpath, "%s\\tokens.bin", odir);
    FILE *tf = fopen(tpath, "rb");
    if (!tf) { fprintf(stderr, "khong mo duoc %s\n", tpath); return 1; }
    int32_t seq_len;
    if (fread(&seq_len, 4, 1, tf) != 1) return 1;
    int32_t *tokens = (int32_t *)malloc(sizeof(int32_t) * seq_len);
    if (fread(tokens, 4, seq_len, tf) != (size_t)seq_len) return 1;
    fclose(tf);
    printf("seq_len=%d  tokens: ", seq_len);
    for (int i = 0; i < seq_len; i++) printf("%d ", tokens[i]);
    printf("\n\n");

    const float *norm_w = find_t("model.norm.weight");
    LayerW LW[N_LAYER];
    for (int l = 0; l < N_LAYER; l++) load_layer_weights(&LW[l], l);

    if (strstr(flags, "kbench")) {
        kernel_bench(LW);
        return 0;
    }

    if (!strstr(flags, "nocompare")) {
        pool_start(4); /* compare mode: da luong an toan (moi phan tu output do dung 1 luong tinh) */
        printf("================ PHAN A: SO KHOP ORACLE (prefill %d token, khong cache) ================\n", seq_len);
        run_compare(LW, tokens, seq_len, norm_w, out_path);
        pool_stop();
    }

    if (!strstr(flags, "nobench")) {
        int thread_list[32], n_list = 0;
        char buf[256];
        strncpy(buf, tcsv, sizeof buf - 1);
        buf[sizeof buf - 1] = 0;
        for (char *tk = strtok(buf, ","); tk && n_list < 32; tk = strtok(NULL, ","))
            thread_list[n_list++] = atoi(tk);
        run_benchmark(LW, tokens, seq_len, norm_w, n_gen, thread_list, n_list, repeats);
    }
    return 0;
}
