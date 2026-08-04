/* TQ33 Runner — Qwen3-30B-A3B (MoE: 48 layer, 128 expert/layer, 8 active/token).
 * Mo rong TRUC TIEP tu qwen3_runner_tq33.c (0.6B, DA VALIDATE: Giai doan 1 F32 khop oracle
 * rel err 2.9e-6, Giai doan 2 TQ33 top-1/top-5 khop tuyet doi). RMSNorm/RoPE NEOX/GQA
 * attention brute-force/QK-norm per-head TAI SU DUNG Y HET cong thuc 0.6B (chi doi hang so
 * kich thuoc n_embd/n_head/n_head_kv/head_dim). Kernel TQ33 decode_block/row_dot_avx2 TAI
 * SU DUNG NGUYEN VAN. Phan THEM MOI: FFN MoE (routing dung moe_common.h, da validate rieng
 * o Giai doan A boi validate_moe_routing.c/.exe — xem RESEARCH_TQ33_RUNNER_30B.md muc Giai
 * doan A), doc du lieu qua manifest da tien xu ly (prepare_30b_runner_data.py), kernel AVX2
 * rieng cho bf16 (embed/lm_head/router KHONG luong tu hoa — giu bf16 de KHONG tang gap doi
 * bang thong doc so voi thiet ke goc), va da luong theo TUNG EXPERT (khong theo hang cua
 * tung GEMV nho — bai hoc tu 0.6B muc 4.4: 196 GEMV/token dong bo hoa qua nhieu lam mat loi
 * the da luong; 30B co 1392 GEMV/token neu lam sai y het se te hon nua).
 *
 * KHONG bias (ckpt train --no-bias, da xac nhan qua bulk_encode_tq33_30b.py + kiem manifest).
 * lm_head.weight VA model.embed_tokens.weight la 2 tensor RIENG (tie_word_embeddings=false)
 * — KHONG co su mo ho nhu 0.6B, dung dung ten tuong ung.
 *
 * Build:  python -m ziglang cc -O3 -mavx2 -mfma -o qwen3moe_runner_tq33.exe qwen3moe_runner_tq33.c -lm
 * Run:    qwen3moe_runner_tq33.exe <runner_dir> <extras_bin> <oracle_dir> <out_dir> [n_gen]
 *   runner_dir  = D:\Bit-Translate-data\tq33_30b\runner   (linear_index.txt, packed_30b.bin,
 *                 codebooks_30b.bin, extras_index.txt — xem prepare_30b_runner_data.py)
 *   extras_bin  = D:\Bit-Translate-data\tq33_30b\extras.bin
 *   oracle_dir  = D:\Bit-Translate-data\tq33_30b\runner\oracle   (tokens.bin)
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

#include "moe_common.h"

/* ================= kien truc Qwen3-30B-A3B (xac nhan qua HF config + llama.cpp source that) */
#define N_LAYER 48
#define HIDDEN 2048
#define N_HEAD 32
#define N_KV_HEAD 4
#define HEAD_DIM 128
#define Q_DIM (N_HEAD * HEAD_DIM)      /* 4096 */
#define KV_DIM (N_KV_HEAD * HEAD_DIM)  /* 512 */
#define N_EXPERT 128
#define N_ACTIVE 8
#define MOE_FFN 768
#define VOCAB 151936
#define RMS_EPS 1e-6f
#define ROPE_THETA 1000000.0f

#define MAX_POS 48          /* prompt (7) + n_gen benchmark, du dung */
#define MAX_THREADS 16
#define MAX_INDIM 4096      /* max(HIDDEN=2048, Q_DIM=4096, MOE_FFN=768) */

#define N_ATTN_PER_LAYER 4
#define N_PER_LAYER (N_ATTN_PER_LAYER + N_EXPERT * 3)   /* 4+384=388 */
#define N_TOTAL_LINEAR (N_LAYER * N_PER_LAYER)          /* 18624 */
#define N_TOTAL_EXTRAS (1 + N_LAYER * 5 + 2)            /* 243 */

#define ATTN_Q 0
#define ATTN_K 1
#define ATTN_V 2
#define ATTN_O 3
#define EXP_GATE 0
#define EXP_UP 1
#define EXP_DOWN 2

#define N_DUMP_LAYER N_LAYER  /* dump hidden+routing sau MOI layer (48) de so oracle full-model,
                                 giong 0.6B (RESEARCH_TQ33_RUNNER.md muc 2.2/3.3) — du lieu nho
                                 (48*7*2048*4 byte =~2.75MB) nen dump het khong ton kem */

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ================= TQ33 packed store (1 file lon + index co dinh, xem
 * prepare_30b_runner_data.py — KHAC 0.6B: 18624 tensor qua nhieu de tra ten, dung O(1)
 * arithmetic indexing thay vi linear search theo ten) ================= */
typedef struct { int64_t R, C, nb, byte_offset, cb_offset; } TQ33Meta;
static TQ33Meta g_tq[N_TOTAL_LINEAR];
static uint8_t *g_packed;
static float *g_codebooks;

static void load_linear_index(const char *dir) {
    char path[512];
    snprintf(path, sizeof path, "%s\\linear_index.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "khong mo duoc %s\n", path); exit(1); }
    int n;
    if (fscanf(f, "%d", &n) != 1 || n != N_TOTAL_LINEAR) {
        fprintf(stderr, "linear_index.txt: n=%d, ky vong %d\n", n, N_TOTAL_LINEAR); exit(1);
    }
    char name[192];
    for (int i = 0; i < n; i++) {
        long long R, C, nb, bo, co;
        if (fscanf(f, "%191s %lld %lld %lld %lld %lld", name, &R, &C, &nb, &bo, &co) != 6) {
            fprintf(stderr, "loi doc dong %d cua linear_index.txt\n", i); exit(1);
        }
        g_tq[i].R = R; g_tq[i].C = C; g_tq[i].nb = nb; g_tq[i].byte_offset = bo; g_tq[i].cb_offset = co;
    }
    fclose(f);
    printf("[linear] da doc index: %d tensor\n", n);
}

/* Kich thuoc buffer lay TRUC TIEP TU FILE SIZE (fseek/_ftelli64), KHONG suy tu tong hop
 * cb_offset[i]+hang_so nhu ban dau (BUG DA BAT: codebook la biến-do-dai/tensor, KHONG dem
 * deu 256 phan tu/tensor — "+256" chi la gia dinh sai, gay fread thieu du lieu "doc hut
 * codebook". packed_30b.bin/codebooks_30b.bin duoc prepare_30b_runner_data.py ghi NOI TIEP
 * KHONG dem, nen file size chinh la tong that, don gian va chac chan hon nhieu so voi cong
 * don qua 18624 phan tu). */
static void load_packed_data(const char *dir) {
    char path[512];
    snprintf(path, sizeof path, "%s\\packed_30b.bin", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "khong mo duoc %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long long packed_size = _ftelli64(f);
    fseek(f, 0, SEEK_SET);
    printf("[linear] dang doc %s (%.3f GB)...\n", path, packed_size / 1e9);
    g_packed = (uint8_t *)malloc((size_t)packed_size);
    if (!g_packed) { fprintf(stderr, "OOM packed %lld byte\n", packed_size); exit(1); }
    double t0 = now_ms();
    if (fread(g_packed, 1, (size_t)packed_size, f) != (size_t)packed_size) {
        fprintf(stderr, "doc hut packed\n"); exit(1);
    }
    fclose(f);
    printf("[linear] doc packed xong (%.1fs)\n", (now_ms() - t0) / 1000.0);

    snprintf(path, sizeof path, "%s\\codebooks_30b.bin", dir);
    f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "khong mo duoc %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long long cb_size = _ftelli64(f);
    fseek(f, 0, SEEK_SET);
    size_t total_cb = (size_t)cb_size / sizeof(float);
    if (total_cb * sizeof(float) != (size_t)cb_size) {
        fprintf(stderr, "codebooks_30b.bin: kich thuoc %lld khong chia het 4\n", cb_size); exit(1);
    }
    g_codebooks = (float *)malloc((size_t)cb_size);
    if (fread(g_codebooks, 1, (size_t)cb_size, f) != (size_t)cb_size) {
        fprintf(stderr, "doc hut codebook\n"); exit(1);
    }
    fclose(f);
    printf("[linear] doc codebook xong: %.2f MB packed + %.3f MB codebook (%zu gia tri)\n",
           packed_size / 1e6, cb_size / 1e6, total_cb);
}

static inline int idx_attn(int l, int which) { return l * N_PER_LAYER + which; }
static inline int idx_expert(int l, int e, int which) {
    return l * N_PER_LAYER + N_ATTN_PER_LAYER + e * 3 + which;
}
static inline const TQ33Meta *tq_attn(int l, int which) { return &g_tq[idx_attn(l, which)]; }
static inline const TQ33Meta *tq_expert(int l, int e, int which) { return &g_tq[idx_expert(l, e, which)]; }

/* kiem tra 1 lan luc khoi dong: moi tensor dung vi tri phai dung shape ky vong — bat som
 * loi indexing (thay vi de lan ra sai so nho kho phat hien trong forward pass) */
static void sanity_check_shapes(void) {
    for (int l = 0; l < N_LAYER; l++) {
        struct { int which, R, C; } attn[4] = {
            {ATTN_Q, Q_DIM, HIDDEN}, {ATTN_K, KV_DIM, HIDDEN},
            {ATTN_V, KV_DIM, HIDDEN}, {ATTN_O, HIDDEN, Q_DIM},
        };
        for (int a = 0; a < 4; a++) {
            const TQ33Meta *m = tq_attn(l, attn[a].which);
            if (m->R != attn[a].R || m->C != attn[a].C) {
                fprintf(stderr, "sanity FAIL: layer %d attn which=%d shape=(%lld,%lld) ky vong (%d,%d)\n",
                        l, attn[a].which, (long long)m->R, (long long)m->C, attn[a].R, attn[a].C);
                exit(1);
            }
        }
        for (int e = 0; e < N_EXPERT; e++) {
            const TQ33Meta *g = tq_expert(l, e, EXP_GATE);
            const TQ33Meta *u = tq_expert(l, e, EXP_UP);
            const TQ33Meta *d = tq_expert(l, e, EXP_DOWN);
            if (g->R != MOE_FFN || g->C != HIDDEN || u->R != MOE_FFN || u->C != HIDDEN ||
                d->R != HIDDEN || d->C != MOE_FFN) {
                fprintf(stderr, "sanity FAIL: layer %d expert %d shape sai\n", l, e);
                exit(1);
            }
        }
    }
    printf("[sanity] 18624 tensor: shape khop ky vong (%d layer x (4 attn + 128 expert x 3))\n", N_LAYER);
}

/* ================= extras (bf16, giu nguyen — embed/lm_head/router giu bf16 de KHONG tang
 * gap doi bang thong; norm/q_norm/k_norm nho (843KB tong) nen convert F32 1 lan luc khoi
 * dong cho don gian, KHONG anh huong bang thong dang ke) ================= */
typedef struct { int64_t offset, nbytes; } ExtrasMeta;
static ExtrasMeta g_ex[N_TOTAL_EXTRAS];
static uint8_t *g_extras;

static void load_extras_index(const char *dir) {
    char path[512];
    snprintf(path, sizeof path, "%s\\extras_index.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "khong mo duoc %s\n", path); exit(1); }
    int n;
    if (fscanf(f, "%d", &n) != 1 || n != N_TOTAL_EXTRAS) {
        fprintf(stderr, "extras_index.txt: n=%d ky vong %d\n", n, N_TOTAL_EXTRAS); exit(1);
    }
    char name[192];
    for (int i = 0; i < n; i++) {
        int ndim;
        if (fscanf(f, "%191s %d", name, &ndim) != 2) exit(1);
        long long dim[4];
        for (int d = 0; d < ndim; d++) if (fscanf(f, "%lld", &dim[d]) != 1) exit(1);
        long long off, nb;
        if (fscanf(f, "%lld %lld", &off, &nb) != 2) exit(1);
        g_ex[i].offset = off; g_ex[i].nbytes = nb;
    }
    fclose(f);
    printf("[extras] da doc index: %d tensor\n", n);
}

static void load_extras_data(const char *extras_bin_path) {
    FILE *f = fopen(extras_bin_path, "rb");
    if (!f) { fprintf(stderr, "khong mo duoc %s\n", extras_bin_path); exit(1); }
    fseek(f, 0, SEEK_END);
    long long fsize = _ftelli64(f);
    fseek(f, 0, SEEK_SET);
    printf("[extras] dang doc %s (%.3f GB)...\n", extras_bin_path, fsize / 1e9);
    g_extras = (uint8_t *)malloc((size_t)fsize);
    if (!g_extras) { fprintf(stderr, "OOM extras\n"); exit(1); }
    double t0 = now_ms();
    if (fread(g_extras, 1, (size_t)fsize, f) != (size_t)fsize) { fprintf(stderr, "doc hut extras\n"); exit(1); }
    fclose(f);
    printf("[extras] doc xong (%.1fs)\n", (now_ms() - t0) / 1000.0);
}

#define EX_EMBED 0
static inline int ex_layer_base(int l) { return 1 + l * 5; }
#define EX_LN_IN_OFF 0
#define EX_LN_POST_OFF 1
#define EX_QNORM_OFF 2
#define EX_KNORM_OFF 3
#define EX_ROUTER_OFF 4
static inline int ex_layer(int l, int off) { return ex_layer_base(l) + off; }
#define EX_NORM_FINAL (1 + N_LAYER * 5)
#define EX_LM_HEAD (1 + N_LAYER * 5 + 1)

static inline const uint16_t *ex_ptr(int idx) { return (const uint16_t *)(g_extras + g_ex[idx].offset); }

/* ================= bf16 -> f32 (dung cho embed lookup 1 hang + convert norm 1 lan) ====== */
static inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = ((uint32_t)h) << 16;
    float f; memcpy(&f, &bits, 4);
    return f;
}
static void bf16_row_to_f32(float *out, const uint16_t *row, int n) {
    for (int i = 0; i < n; i++) out[i] = bf16_to_f32(row[i]);
}

/* norm weight (input_layernorm/post_attention_layernorm/q_norm/k_norm/final norm) convert
 * F32 MOT LAN luc khoi dong — cac tensor nay nho (2048 hoac 128 phan tu), doc lai bf16 moi
 * lan goi rmsnorm se ton cong vo ich, KHONG lien quan gi den van de bang thong lm_head/embed
 * (622MB) hay TQ33-active (~486MB/token) — 843KB tong cho toan bo 48 layer la khong dang ke. */
static float g_ln_in[N_LAYER][HIDDEN];
static float g_ln_post[N_LAYER][HIDDEN];
static float g_qn[N_LAYER][HEAD_DIM];
static float g_kn[N_LAYER][HEAD_DIM];
static float g_norm_final[HIDDEN];

static void convert_norms(void) {
    for (int l = 0; l < N_LAYER; l++) {
        bf16_row_to_f32(g_ln_in[l], ex_ptr(ex_layer(l, EX_LN_IN_OFF)), HIDDEN);
        bf16_row_to_f32(g_ln_post[l], ex_ptr(ex_layer(l, EX_LN_POST_OFF)), HIDDEN);
        bf16_row_to_f32(g_qn[l], ex_ptr(ex_layer(l, EX_QNORM_OFF)), HEAD_DIM);
        bf16_row_to_f32(g_kn[l], ex_ptr(ex_layer(l, EX_KNORM_OFF)), HEAD_DIM);
    }
    bf16_row_to_f32(g_norm_final, ex_ptr(EX_NORM_FINAL), HIDDEN);
    printf("[extras] da convert norm/q_norm/k_norm sang F32 (843KB, 1 lan luc khoi dong)\n");
}

/* ================= kernel TQ33 (TAI SU DUNG Y HET tq33_bench.c / qwen3_runner_tq33.c) ===== */
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

static void quantize_x_int8(int8_t *xq, float *xs, const float *x, int in_dim) {
    int nb = in_dim / 64;
    for (int b = 0; b < nb; b++) {
        float m = 0.0f;
        for (int i = 0; i < 64; i++) { float a = fabsf(x[b * 64 + i]); if (a > m) m = a; }
        float sc = m > 0 ? m / 127.0f : 1.0f;
        xs[b] = sc;
        for (int i = 0; i < 64; i++) xq[b * 64 + i] = (int8_t)lrintf(x[b * 64 + i] / sc);
    }
}

/* dot bf16(w) . f32(x) bang AVX2 — BAT BUOC viet AVX2 tuong minh ngay tu dau (bai hoc 0.6B
 * muc 4.1: scalar loop KHONG duoc clang tu dong vector hoa voi strict FP semantics). bf16->f32
 * bang cach shift trai 16 bit thanh bit-pattern f32 (dung chuan, khong mat gi vi bf16 la 8 bit
 * exponent + 7 bit mantissa cao cua f32, phan thap 16 bit mantissa =0). */
static inline float dot_bf16_f32_avx2(const uint16_t *w, const float *x, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i wraw = _mm_loadu_si128((const __m128i *)(w + i));
        __m256i wi32 = _mm256_cvtepu16_epi32(wraw);
        __m256i wshl = _mm256_slli_epi32(wi32, 16);
        __m256 wf = _mm256_castsi256_ps(wshl);
        acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + i), acc);
    }
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    float s = _mm_cvtss_f32(h);
    for (; i < n; i++) s += bf16_to_f32(w[i]) * x[i];
    return s;
}

/* ================= song song hoa (thread-pool thuong truc — TAI SU DUNG 0.6B, THEM parallel_jobs
 * cho MoE theo TUNG EXPERT thay vi theo hang GEMV — bai hoc 0.6B muc 4.4) ================= */
static int g_nthreads = 1;

typedef struct { void (*rowfunc)(void *, int); void *ctx; int r0, r1; } RowJob;

static HANDLE g_start_evt[MAX_THREADS];
static HANDLE g_done_evt[MAX_THREADS];
static RowJob g_pool_job[MAX_THREADS];
static int g_pool_started = 0;

static unsigned __stdcall pool_worker(void *arg) {
    int tid = (int)(intptr_t)arg;
    for (;;) {
        WaitForSingleObject(g_start_evt[tid], INFINITE);
        RowJob *j = &g_pool_job[tid];
        for (int r = j->r0; r < j->r1; r++) j->rowfunc(j->ctx, r);
        SetEvent(g_done_evt[tid]);
    }
    return 0;
}

static void ensure_pool(void) {
    if (g_pool_started) return;
    for (int t = 0; t < MAX_THREADS; t++) {
        g_start_evt[t] = CreateEvent(NULL, FALSE, FALSE, NULL);
        g_done_evt[t] = CreateEvent(NULL, FALSE, FALSE, NULL);
        _beginthreadex(NULL, 0, pool_worker, (void *)(intptr_t)t, 0, NULL);
    }
    g_pool_started = 1;
}

/* row-split: dung cho GEMV LON (attn q/k/v/o, router, lm_head) — chia theo HANG cua 1 GEMV.
 * Fallback serial neu out_dim qua nho so voi so luong (< nt*4 hang/luong). */
static void parallel_rows(int out_dim, void (*rowfunc)(void *, int), void *ctx) {
    int nt = g_nthreads;
    if (nt <= 1 || out_dim < nt * 4) {
        for (int o = 0; o < out_dim; o++) rowfunc(ctx, o);
        return;
    }
    if (nt > MAX_THREADS) nt = MAX_THREADS;
    ensure_pool();
    int chunk = (out_dim + nt - 1) / nt;
    HANDLE waits[MAX_THREADS];
    int n_active = 0;
    for (int t = 0; t < nt; t++) {
        int r0 = t * chunk, r1 = r0 + chunk;
        if (r1 > out_dim) r1 = out_dim;
        if (r0 >= r1) continue;
        g_pool_job[t].rowfunc = rowfunc; g_pool_job[t].ctx = ctx;
        g_pool_job[t].r0 = r0; g_pool_job[t].r1 = r1;
        SetEvent(g_start_evt[t]);
        waits[n_active++] = g_done_evt[t];
    }
    if (n_active) WaitForMultipleObjects(n_active, waits, TRUE, INFINITE);
}

/* job-split: dung cho MoE — chia theo TUNG EXPERT (moi "job" la 1 expert TRON VEN: gate+up+
 * down+SwiGLU, khong phai 1 hang GEMV). Khac parallel_rows o nguong: ngay ca 1 job/luong cung
 * dang song song hoa (moi job ~0.84MB + tinh toan that, khong nhu 1 hang GEMV vai chuc byte),
 * nen KHONG fallback serial chi vi it job/luong — day chinh la thay doi chinh de tranh lap lai
 * loi 0.6B (196 GEMV/token dong bo qua nhieu -> chi dat 25-30% GB/s ky vong o 6+ luong). */
static void parallel_jobs(int n_jobs, void (*jobfunc)(void *, int), void *ctx) {
    int nt = g_nthreads;
    if (nt <= 1 || n_jobs < 2) {
        for (int j = 0; j < n_jobs; j++) jobfunc(ctx, j);
        return;
    }
    if (nt > n_jobs) nt = n_jobs;
    if (nt > MAX_THREADS) nt = MAX_THREADS;
    ensure_pool();
    int chunk = (n_jobs + nt - 1) / nt;
    HANDLE waits[MAX_THREADS];
    int n_active = 0;
    for (int t = 0; t < nt; t++) {
        int j0 = t * chunk, j1 = j0 + chunk;
        if (j1 > n_jobs) j1 = n_jobs;
        if (j0 >= j1) continue;
        g_pool_job[t].rowfunc = jobfunc; g_pool_job[t].ctx = ctx;
        g_pool_job[t].r0 = j0; g_pool_job[t].r1 = j1;
        SetEvent(g_start_evt[t]);
        waits[n_active++] = g_done_evt[t];
    }
    if (n_active) WaitForMultipleObjects(n_active, waits, TRUE, INFINITE);
}

/* linear TQ33 row-parallel (dung TRONG MAIN THREAD — attn q/k/v/o). static xq/xs AN TOAN vi
 * CHI goi tu main thread, KHONG BAO GIO long trong 1 job cua parallel_jobs. */
typedef struct {
    float *out; const int8_t *xq; const float *xs; const float *table;
    const uint8_t *packed_base; size_t row_stride; int nb;
} Tq33RowCtx;

static void tq33_row_func(void *ctx_, int o) {
    Tq33RowCtx *c = (Tq33RowCtx *)ctx_;
    c->out[o] = row_dot_avx2(c->packed_base + (size_t)o * c->row_stride, c->nb, c->xq, c->xs, c->table);
}

static void linear_tq33(float *out, const float *x, const TQ33Meta *tm, int in_dim, int out_dim) {
    if (tm->C != in_dim || tm->R != out_dim) {
        fprintf(stderr, "linear_tq33: shape mismatch (%lldx%lld) vs in=%d out=%d\n",
                (long long)tm->R, (long long)tm->C, in_dim, out_dim);
        exit(1);
    }
    static int8_t xq[MAX_INDIM];
    static float xs[MAX_INDIM / 64];
    quantize_x_int8(xq, xs, x, in_dim);
    Tq33RowCtx ctx;
    ctx.out = out; ctx.xq = xq; ctx.xs = xs;
    ctx.table = g_codebooks + tm->cb_offset;
    ctx.packed_base = g_packed + tm->byte_offset;
    ctx.row_stride = (size_t)tm->nb * 12;
    ctx.nb = (int)tm->nb;
    parallel_rows(out_dim, tq33_row_func, &ctx);
}

/* linear TQ33 SERIAL (dung BEN TRONG 1 job cua parallel_jobs — KHONG duoc goi parallel_rows o
 * day vi se long 2 tang thread-pool va dam vao chinh no / dua du lieu g_pool_job dung chung).
 * xq/xs phai da quantize SAN va truyen vao (khong dung static buffer noi bo — nhieu job chay
 * DONG THOI tren nhieu thread, static se bi race). */
static void linear_tq33_rows_from_q(float *out, const int8_t *xq, const float *xs,
                                     const TQ33Meta *tm, int out_dim) {
    const float *table = g_codebooks + tm->cb_offset;
    const uint8_t *packed_base = g_packed + tm->byte_offset;
    size_t row_stride = (size_t)tm->nb * 12;
    for (int o = 0; o < out_dim; o++)
        out[o] = row_dot_avx2(packed_base + (size_t)o * row_stride, (int)tm->nb, xq, xs, table);
}

/* linear bf16 row-parallel (router + lm_head — dung dot_bf16_f32_avx2, KHONG upconvert ca
 * ma tran sang F32 de giu nguyen loi ich bang thong bf16, xem chu thich dau file). */
typedef struct { float *out; const float *x; const uint16_t *W; int in_dim; } Bf16RowCtx;

static void bf16_row_func(void *ctx_, int o) {
    Bf16RowCtx *c = (Bf16RowCtx *)ctx_;
    const uint16_t *row = c->W + (size_t)o * c->in_dim;
    c->out[o] = dot_bf16_f32_avx2(row, c->x, c->in_dim);
}

static void linear_bf16(float *out, const float *x, const uint16_t *W, int in_dim, int out_dim) {
    Bf16RowCtx ctx = {out, x, W, in_dim};
    parallel_rows(out_dim, bf16_row_func, &ctx);
}

/* ================= cac phep toan co ban (giong het 0.6B — cong thuc KHONG doi, chi doi hang so) */
static void rmsnorm(float *out, const float *x, const float *w, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float inv = 1.0f / sqrtf((float)(ss / n) + RMS_EPS);
    for (int i = 0; i < n; i++) out[i] = w[i] * (x[i] * inv);
}

static void rope_neox(float *vec, int dim, int pos) {
    float theta_scale = powf(ROPE_THETA, -2.0f / dim);
    float theta = (float)pos;
    int half = dim / 2;
    for (int j = 0; j < half; j++) {
        float c = cosf(theta), s = sinf(theta);
        float x0 = vec[j], x1 = vec[j + half];
        vec[j] = x0 * c - x1 * s;
        vec[j + half] = x0 * s + x1 * c;
        theta *= theta_scale;
    }
}

/* ================= MoE FFN (moi expert la 1 "job" song song — xem parallel_jobs o tren) ==== */
static float g_moe_gate[N_ACTIVE][MOE_FFN];
static float g_moe_up[N_ACTIVE][MOE_FFN];
static float g_moe_h[N_ACTIVE][MOE_FFN];
static int8_t g_moe_hq[N_ACTIVE][MOE_FFN];
static float g_moe_hs[N_ACTIVE][MOE_FFN / 64];

typedef struct {
    int layer;
    const int8_t *xq; const float *xs;   /* x (post ffn_norm) DA quantize SAN, dung CHUNG cho
                                             gate/up cua CA 8 expert — chi tinh 1 lan/layer */
    const int *sel_idx;                   /* expert id duoc chon, [N_ACTIVE] */
    float *expert_out;                    /* [N_ACTIVE][HIDDEN], MOI JOB ghi vao slot j RIENG */
} MoeJobCtx;

static void moe_expert_job(void *ctx_, int j) {
    MoeJobCtx *c = (MoeJobCtx *)ctx_;
    int e = c->sel_idx[j];
    const TQ33Meta *wg = tq_expert(c->layer, e, EXP_GATE);
    const TQ33Meta *wu = tq_expert(c->layer, e, EXP_UP);
    const TQ33Meta *wd = tq_expert(c->layer, e, EXP_DOWN);
    float *gate = g_moe_gate[j], *up = g_moe_up[j], *h = g_moe_h[j];

    linear_tq33_rows_from_q(gate, c->xq, c->xs, wg, MOE_FFN);
    linear_tq33_rows_from_q(up, c->xq, c->xs, wu, MOE_FFN);
    for (int i = 0; i < MOE_FFN; i++) h[i] = moe_silu(gate[i]) * up[i];

    /* down: input h RIENG cho tung expert -> quantize RIENG (buffer slot j, khong static
     * chung -> an toan khi nhieu thread chay dong thoi) */
    quantize_x_int8(g_moe_hq[j], g_moe_hs[j], h, MOE_FFN);
    float *eo = c->expert_out + (size_t)j * HIDDEN;
    linear_tq33_rows_from_q(eo, g_moe_hq[j], g_moe_hs[j], wd, HIDDEN);
}

/* ================= debug dump (Giai doan B.4 — so oracle 1-2 layer dau) ================== */
static int g_dump_enabled = 0;
static float g_dump_hidden[N_DUMP_LAYER][MAX_POS][HIDDEN];
static int g_dump_idx[N_DUMP_LAYER][MAX_POS][N_ACTIVE];
static float g_dump_weight[N_DUMP_LAYER][MAX_POS][N_ACTIVE];

/* timers tich luy (chi do o 1-luong de phan tich ty le breakdown, giong 0.6B muc 4.3) */
static double g_t_attn_linear = 0, g_t_attn_math = 0, g_t_norm_rope = 0;
static double g_t_router = 0, g_t_moe_expert = 0, g_t_embed_logits = 0;

static int forward_one_token(int32_t token_id, int pos, float *K_cache, float *V_cache,
                              float *logits_out, int want_logits) {
    static float hidden[HIDDEN], cur[HIDDEN];
    static float Qb[Q_DIM], Kb[KV_DIM], Vb[KV_DIM];
    static float attn_concat[Q_DIM], o_out[HIDDEN], ffn_inp[HIDDEN];
    static float router_logits[N_EXPERT];
    static int8_t shared_xq[HIDDEN];
    static float shared_xs[HIDDEN / 64];
    static float expert_out[N_ACTIVE][HIDDEN];
    static float moe_combined[HIDDEN];
    static int sel_idx[N_ACTIVE];
    static float sel_weight[N_ACTIVE];
    static float scores[MAX_POS];
    double t0;

    bf16_row_to_f32(hidden, ex_ptr(EX_EMBED) + (size_t)token_id * HIDDEN, HIDDEN);

    for (int l = 0; l < N_LAYER; l++) {
        t0 = now_ms();
        rmsnorm(cur, hidden, g_ln_in[l], HIDDEN);
        g_t_norm_rope += now_ms() - t0;

        t0 = now_ms();
        linear_tq33(Qb, cur, tq_attn(l, ATTN_Q), HIDDEN, Q_DIM);
        linear_tq33(Kb, cur, tq_attn(l, ATTN_K), HIDDEN, KV_DIM);
        linear_tq33(Vb, cur, tq_attn(l, ATTN_V), HIDDEN, KV_DIM);
        g_t_attn_linear += now_ms() - t0;

        t0 = now_ms();
        for (int h = 0; h < N_HEAD; h++) {
            float *qh = Qb + (size_t)h * HEAD_DIM;
            rmsnorm(qh, qh, g_qn[l], HEAD_DIM);
            rope_neox(qh, HEAD_DIM, pos);
        }
        for (int h = 0; h < N_KV_HEAD; h++) {
            float *kh = Kb + (size_t)h * HEAD_DIM;
            rmsnorm(kh, kh, g_kn[l], HEAD_DIM);
            rope_neox(kh, HEAD_DIM, pos);
        }
        g_t_norm_rope += now_ms() - t0;

        memcpy(K_cache + ((size_t)l * MAX_POS + pos) * KV_DIM, Kb, KV_DIM * 4);
        memcpy(V_cache + ((size_t)l * MAX_POS + pos) * KV_DIM, Vb, KV_DIM * 4);

        t0 = now_ms();
        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        for (int h = 0; h < N_HEAD; h++) {
            int kv_h = h / (N_HEAD / N_KV_HEAD);
            const float *qi = Qb + (size_t)h * HEAD_DIM;
            float maxs = -1e30f;
            for (int j = 0; j <= pos; j++) {
                const float *kj = K_cache + ((size_t)l * MAX_POS + j) * KV_DIM + (size_t)kv_h * HEAD_DIM;
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) dot += qi[d] * kj[d];
                dot *= scale;
                scores[j] = dot;
                if (dot > maxs) maxs = dot;
            }
            float sum = 0.0f;
            for (int j = 0; j <= pos; j++) { scores[j] = expf(scores[j] - maxs); sum += scores[j]; }
            float inv = 1.0f / sum;
            float *outh = attn_concat + (size_t)h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) outh[d] = 0.0f;
            for (int j = 0; j <= pos; j++) {
                float wgt = scores[j] * inv;
                const float *vj = V_cache + ((size_t)l * MAX_POS + j) * KV_DIM + (size_t)kv_h * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; d++) outh[d] += wgt * vj[d];
            }
        }
        g_t_attn_math += now_ms() - t0;

        t0 = now_ms();
        linear_tq33(o_out, attn_concat, tq_attn(l, ATTN_O), Q_DIM, HIDDEN);
        g_t_attn_linear += now_ms() - t0;

        for (int i = 0; i < HIDDEN; i++) ffn_inp[i] = hidden[i] + o_out[i];

        t0 = now_ms();
        rmsnorm(cur, ffn_inp, g_ln_post[l], HIDDEN);
        g_t_norm_rope += now_ms() - t0;

        /* ---- router: softmax TOAN BO 128 expert -> chon top-8 -> renormalize ---- */
        t0 = now_ms();
        linear_bf16(router_logits, cur, ex_ptr(ex_layer(l, EX_ROUTER_OFF)), HIDDEN, N_EXPERT);
        moe_softmax_inplace(router_logits, N_EXPERT);
        moe_topk_renorm(router_logits, N_EXPERT, N_ACTIVE, sel_idx, sel_weight);
        g_t_router += now_ms() - t0;

        if (g_dump_enabled && l < N_DUMP_LAYER && pos < MAX_POS) {
            memcpy(g_dump_idx[l][pos], sel_idx, sizeof(int) * N_ACTIVE);
            memcpy(g_dump_weight[l][pos], sel_weight, sizeof(float) * N_ACTIVE);
        }

        /* ---- 8 expert song song theo TUNG EXPERT (khong theo hang GEMV) ---- */
        t0 = now_ms();
        quantize_x_int8(shared_xq, shared_xs, cur, HIDDEN);
        MoeJobCtx jctx;
        jctx.layer = l; jctx.xq = shared_xq; jctx.xs = shared_xs;
        jctx.sel_idx = sel_idx; jctx.expert_out = &expert_out[0][0];
        parallel_jobs(N_ACTIVE, moe_expert_job, &jctx);
        moe_combine(moe_combined, &expert_out[0][0], sel_weight, N_ACTIVE, HIDDEN);
        g_t_moe_expert += now_ms() - t0;

        for (int i = 0; i < HIDDEN; i++) hidden[i] = moe_combined[i] + ffn_inp[i];

        if (g_dump_enabled && l < N_DUMP_LAYER && pos < MAX_POS)
            memcpy(g_dump_hidden[l][pos], hidden, HIDDEN * 4);
    }

    t0 = now_ms();
    rmsnorm(cur, hidden, g_norm_final, HIDDEN);
    int next_id = -1;
    if (want_logits) {
        linear_bf16(logits_out, cur, ex_ptr(EX_LM_HEAD), HIDDEN, VOCAB);
        float best = -1e30f;
        for (int v = 0; v < VOCAB; v++) if (logits_out[v] > best) { best = logits_out[v]; next_id = v; }
    }
    g_t_embed_logits += now_ms() - t0;
    return next_id;
}

/* ================= benchmark + dump oracle ================= */
static void write_dump(const char *path, int prompt_len) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "khong ghi duoc %s\n", path); return; }
    int32_t hdr[4] = {prompt_len, N_DUMP_LAYER, HIDDEN, N_ACTIVE};
    fwrite(hdr, 4, 4, f);
    for (int l = 0; l < N_DUMP_LAYER; l++) {
        for (int p = 0; p < prompt_len; p++) fwrite(g_dump_hidden[l][p], 4, HIDDEN, f);
        for (int p = 0; p < prompt_len; p++) fwrite(g_dump_idx[l][p], 4, N_ACTIVE, f);
        for (int p = 0; p < prompt_len; p++) fwrite(g_dump_weight[l][p], 4, N_ACTIVE, f);
    }
    fclose(f);
    printf("[dump] da ghi hidden+routing layer 0..%d (pos 0..%d) -> %s\n", N_DUMP_LAYER - 1,
           prompt_len - 1, path);
}

static void run_benchmark(const int32_t *prompt_tokens, int prompt_len, int n_gen,
                          const int *thread_list, int n_thread_list, const char *dump_path) {
    float *K_cache = malloc((size_t)N_LAYER * MAX_POS * KV_DIM * 4);
    float *V_cache = malloc((size_t)N_LAYER * MAX_POS * KV_DIM * 4);
    float *logits = malloc((size_t)VOCAB * 4);

    printf("\n================ BENCHMARK TOC DO (KV-cache, %d token sinh) ================\n", n_gen);
    for (int ti = 0; ti < n_thread_list; ti++) {
        g_nthreads = thread_list[ti];
        g_t_attn_linear = g_t_attn_math = g_t_norm_rope = 0;
        g_t_router = g_t_moe_expert = g_t_embed_logits = 0;
        g_dump_enabled = (ti == 0);

        int32_t cur_tok = 0;
        int pos = 0;
        for (; pos < prompt_len; pos++)
            forward_one_token(prompt_tokens[pos], pos, K_cache, V_cache, logits, 0);
        int next = forward_one_token(prompt_tokens[prompt_len - 1], prompt_len - 1, K_cache, V_cache,
                                      logits, 1);
        cur_tok = next;

        if (ti == 0) {
            printf("[kiem chung KV-cache] token dau tien sinh ra (vi tri %d) = %d\n",
                   prompt_len - 1, next);
            write_dump(dump_path, prompt_len);
        }
        g_dump_enabled = 0;

        double t0 = now_ms();
        g_t_attn_linear = g_t_attn_math = g_t_norm_rope = 0;
        g_t_router = g_t_moe_expert = g_t_embed_logits = 0;
        int gen_tokens[256];
        for (int g = 0; g < n_gen; g++) {
            int p = prompt_len + g;
            if (p >= MAX_POS - 1) break;
            int nx = forward_one_token(cur_tok, p, K_cache, V_cache, logits, 1);
            if (g < 256) gen_tokens[g] = nx;
            cur_tok = nx;
        }
        if (ti == 0) {
            printf("[kiem chung] %d token sinh tiep theo (id): ", n_gen);
            for (int g = 0; g < n_gen && g < 256; g++) printf("%d ", gen_tokens[g]);
            printf("\n");
        }
        double dt = now_ms() - t0;
        double toks = n_gen / (dt / 1000.0);
        double total_t = g_t_attn_linear + g_t_attn_math + g_t_norm_rope + g_t_router +
                          g_t_moe_expert + g_t_embed_logits;
        printf("threads=%2d : %8.1f ms / %d token -> %6.2f tok/s | breakdown: "
               "attn-lin=%.0f%% attn-math=%.0f%% norm+rope=%.0f%% router=%.0f%% "
               "moe-expert=%.0f%% lm_head(bf16)=%.0f%%\n",
               g_nthreads, dt, n_gen, toks,
               100.0 * g_t_attn_linear / total_t, 100.0 * g_t_attn_math / total_t,
               100.0 * g_t_norm_rope / total_t, 100.0 * g_t_router / total_t,
               100.0 * g_t_moe_expert / total_t, 100.0 * g_t_embed_logits / total_t);
    }
    free(K_cache); free(V_cache); free(logits);
}

/* ================= main ================= */
int main(int argc, char **argv) {
    const char *runner_dir = argc > 1 ? argv[1] : "D:\\Bit-Translate-data\\tq33_30b\\runner";
    const char *extras_bin = argc > 2 ? argv[2] : "D:\\Bit-Translate-data\\tq33_30b\\extras.bin";
    const char *oracle_dir = argc > 3 ? argv[3] : "D:\\Bit-Translate-data\\tq33_30b\\runner\\oracle";
    const char *out_dir = argc > 4 ? argv[4] : "D:\\Bit-Translate-data\\tq33_30b\\runner\\oracle";
    int n_gen = argc > 5 ? atoi(argv[5]) : 20;

    build_tq33_tables();
    load_linear_index(runner_dir);
    load_packed_data(runner_dir);
    load_extras_index(runner_dir);
    load_extras_data(extras_bin);
    sanity_check_shapes();
    convert_norms();

    char tpath[512], dpath[512];
    snprintf(tpath, sizeof tpath, "%s\\tokens.bin", oracle_dir);
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

    snprintf(dpath, sizeof dpath, "%s\\runner_dump_layers.bin", out_dir);

    int threads_try[] = {1, 2, 4, 6, 8, 12, 14};
    int n_try = (int)(sizeof(threads_try) / sizeof(threads_try[0]));
    run_benchmark(tokens, seq_len, n_gen, threads_try, n_try, dpath);

    return 0;
}
