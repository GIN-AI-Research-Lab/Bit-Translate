/* TQ33 Runner — Giai đoạn 2: forward pass Qwen3-0.6B (28 layer) DÙNG KERNEL TQ33 cho 7
 * loại linear (q/k/v/o/gate/up/down), phần còn lại (RMSNorm/RoPE/attention/SwiGLU
 * activation) vẫn float32. Kernel TQ33 (decode_block/row_dot_avx2) TÁI SỬ DỤNG Y HỆT
 * logic đã đo tốc độ trong tq33_bench.c (kernel v3 — LUT->buffer + FMA-accumulate).
 *
 * 2 việc trong 1 lần chạy:
 *   Phần A — so khớp oracle: prefill 7 token prompt giống Phase 1 (không cache), dump
 *            per-layer hidden + logits ra file để compare_phase2.py so rel err.
 *   Phần B — benchmark tốc độ: sau prompt, generate N_GEN token tiếp bằng KV-cache đơn
 *            giản (append-only, không tối ưu), đo tok/s ở nhiều mức số luồng (row-parallel
 *            trên GEMV bằng Windows thread — zig cc không có omp.h nên không dùng OpenMP
 *            được, đã thử xác nhận thiếu header).
 *
 * Build:  python -m ziglang cc -O3 -mavx2 -mfma -o qwen3_runner_tq33.exe qwen3_runner_tq33.c -lm
 * Run:    qwen3_runner_tq33.exe <weights_f32_dir> <tq33_packed_dir> <oracle_dir> <out_bin> [n_gen]
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

/* ================= F32 weight store (embed/norm/bias — KHÔNG đổi Phase 1) ================= */

typedef struct { char name[160]; int ndim; int64_t dim[4]; int64_t offset; int64_t numel; } TensorMeta;
static TensorMeta g_meta[MAX_TENSORS];
static int g_n_meta;
static float *g_w;

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

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
    printf("[f32] dang doc %s (%.3f GB)...\n", path, total_elems * 4 / 1e9);
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

/* ================= TQ33 packed store ================= */

typedef struct { char name[160]; int64_t R, C, nb, byte_offset, cb_offset; } TQ33Meta;
static TQ33Meta g_tq[MAX_TENSORS];
static int g_n_tq;
static uint8_t *g_packed;
static float *g_codebooks;

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
    }
    fclose(f);
    printf("[tq33] da doc meta: %d tensor\n", g_n_tq);
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

static const TQ33Meta *find_tq33(const char *name) {
    for (int i = 0; i < g_n_tq; i++)
        if (strcmp(g_tq[i].name, name) == 0) return &g_tq[i];
    fprintf(stderr, "KHONG TIM THAY tensor tq33: %s\n", name);
    exit(1);
}

static const TQ33Meta *find_tq33_layer(int l, const char *suffix) {
    char name[160];
    snprintf(name, sizeof name, "model.layers.%d.%s.weight", l, suffix);
    return find_tq33(name);
}

/* ================= kernel TQ33 (TÁI SỬ DỤNG Y HỆT tq33_bench.c kernel v3) ================= */

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

/* lượng tử hoá activation x -> int8 per-64-group (đường inference thật, y hệt tq33_bench.c) */
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

/* ================= song song hoá theo hàng (Windows thread, không có omp.h qua zig cc) ============ */

static int g_nthreads = 1;

/* THREAD POOL THUONG TRUC — ban dau dung _beginthreadex/CloseHandle MOI LAN goi GEMV
 * (196 lan/token: 7 linear x 28 layer), tao/huy thread that su moi lan -> overhead lon,
 * lam so lieu >4 luong xau di GIA (nham lan voi kernel cham). Sua: spawn N thread MOT
 * LAN duy nhat luc khoi dong, dong bo qua Event (auto-reset) — moi lan goi parallel_rows
 * chi SetEvent (~1-2us) thay vi tao thread moi (~50-200us). */
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

typedef struct {
    float *out; const int8_t *xq; const float *xs; const float *table;
    const uint8_t *packed_base; size_t row_stride; int nb; const float *bias;
} Tq33RowCtx;

static void tq33_row_func(void *ctx_, int o) {
    Tq33RowCtx *c = (Tq33RowCtx *)ctx_;
    float v = row_dot_avx2(c->packed_base + (size_t)o * c->row_stride, c->nb, c->xq, c->xs, c->table);
    c->out[o] = v + (c->bias ? c->bias[o] : 0.0f);
}

/* y = TQ33(W) @ x + b .  tm->C phải == in_dim, tm->R phải == out_dim. */
static void linear_tq33(float *out, const float *x, const TQ33Meta *tm, const float *bias,
                         int in_dim, int out_dim) {
    if (tm->C != in_dim || tm->R != out_dim) {
        fprintf(stderr, "linear_tq33: shape mismatch %s (%lldx%lld) vs in=%d out=%d\n",
                tm->name, (long long)tm->R, (long long)tm->C, in_dim, out_dim);
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
    ctx.bias = bias;
    parallel_rows(out_dim, tq33_row_func, &ctx);
}

/* dot product AVX2+FMA tuong minh — plain scalar loop KHONG tu dong vector hoa duoc voi
 * strict FP semantics (clang khong sap xep lai thu tu cong don float neu khong -ffast-math),
 * viet tay giong style row_dot_avx2/tq33_bench.c "[5] fp32 GEMV AVX2" de so sanh cong bang
 * (khong de overhead embed/logits bi thoi phong chi vi vong lap F32 khong toi uu). */
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

typedef struct { float *out; const float *x; const float *W; const float *b; int in_dim; } F32RowCtx;
static void f32_row_func(void *ctx_, int o) {
    F32RowCtx *c = (F32RowCtx *)ctx_;
    const float *row = c->W + (size_t)o * c->in_dim;
    float acc = (c->b ? c->b[o] : 0.0f) + dot_f32_avx2(row, c->x, c->in_dim);
    c->out[o] = acc;
}

static void linear_f32(float *out, const float *x, const float *W, const float *b,
                        int in_dim, int out_dim) {
    F32RowCtx ctx = {out, x, W, b, in_dim};
    parallel_rows(out_dim, f32_row_func, &ctx);
}

/* ================= các phép toán cơ bản (giống hệt Phase 1) ================= */

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

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

/* ================= per-layer weight cache (tránh find() lap lai trong vong lap timed) ========== */

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

/* ================= Phần A: prefill so khớp oracle (giống Phase 1, không cache) ================= */

static void run_compare(LayerW *LW, const int32_t *tokens, int seq_len, const float *embed,
                        const float *norm_w, const char *out_path) {
    float *hidden = malloc((size_t)seq_len * HIDDEN * 4);
    float *cur = malloc((size_t)seq_len * HIDDEN * 4);
    float *Q = malloc((size_t)seq_len * Q_DIM * 4);
    float *K = malloc((size_t)seq_len * KV_DIM * 4);
    float *V = malloc((size_t)seq_len * KV_DIM * 4);
    float *attn_concat = malloc((size_t)seq_len * Q_DIM * 4);
    float *o_out = malloc((size_t)seq_len * HIDDEN * 4);
    float *ffn_inp = malloc((size_t)seq_len * HIDDEN * 4);
    float *gate = malloc((size_t)seq_len * FFN * 4);
    float *up = malloc((size_t)seq_len * FFN * 4);
    float *h_ffn = malloc((size_t)seq_len * FFN * 4);
    float *down_out = malloc((size_t)seq_len * HIDDEN * 4);
    float *scores = malloc((size_t)seq_len * 4);

    FILE *of = fopen(out_path, "wb");
    if (!of) exit(1);
    int32_t hdr[4] = {seq_len, HIDDEN, VOCAB, N_LAYER};
    fwrite(hdr, 4, 4, of);

    for (int t = 0; t < seq_len; t++)
        memcpy(hidden + (size_t)t * HIDDEN, embed + (size_t)tokens[t] * HIDDEN, HIDDEN * 4);

    double t_start = now_ms();
    for (int l = 0; l < N_LAYER; l++) {
        LayerW *lw = &LW[l];
        for (int t = 0; t < seq_len; t++)
            rmsnorm(cur + (size_t)t * HIDDEN, hidden + (size_t)t * HIDDEN, lw->attn_norm_w, HIDDEN);

        for (int t = 0; t < seq_len; t++) {
            linear_tq33(Q + (size_t)t * Q_DIM, cur + (size_t)t * HIDDEN, lw->Wq, lw->bq, HIDDEN, Q_DIM);
            linear_tq33(K + (size_t)t * KV_DIM, cur + (size_t)t * HIDDEN, lw->Wk, lw->bk, HIDDEN, KV_DIM);
            linear_tq33(V + (size_t)t * KV_DIM, cur + (size_t)t * HIDDEN, lw->Wv, lw->bv, HIDDEN, KV_DIM);
            for (int h = 0; h < N_HEAD; h++) {
                float *qh = Q + (size_t)t * Q_DIM + (size_t)h * HEAD_DIM;
                rmsnorm(qh, qh, lw->qn, HEAD_DIM);
                rope_neox(qh, HEAD_DIM, t);
            }
            for (int h = 0; h < N_KV_HEAD; h++) {
                float *kh = K + (size_t)t * KV_DIM + (size_t)h * HEAD_DIM;
                rmsnorm(kh, kh, lw->kn, HEAD_DIM);
                rope_neox(kh, HEAD_DIM, t);
            }
        }

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
            linear_tq33(o_out + (size_t)t * HIDDEN, attn_concat + (size_t)t * Q_DIM, lw->Wo, lw->bo, Q_DIM, HIDDEN);

        for (int t = 0; t < seq_len; t++)
            for (int i = 0; i < HIDDEN; i++)
                ffn_inp[(size_t)t * HIDDEN + i] = hidden[(size_t)t * HIDDEN + i] + o_out[(size_t)t * HIDDEN + i];

        for (int t = 0; t < seq_len; t++)
            rmsnorm(cur + (size_t)t * HIDDEN, ffn_inp + (size_t)t * HIDDEN, lw->ffn_norm_w, HIDDEN);

        for (int t = 0; t < seq_len; t++) {
            linear_tq33(gate + (size_t)t * FFN, cur + (size_t)t * HIDDEN, lw->Wgate, lw->bgate, HIDDEN, FFN);
            linear_tq33(up + (size_t)t * FFN, cur + (size_t)t * HIDDEN, lw->Wup, lw->bup, HIDDEN, FFN);
            for (int i = 0; i < FFN; i++) {
                size_t idx = (size_t)t * FFN + i;
                h_ffn[idx] = silu(gate[idx]) * up[idx];
            }
            linear_tq33(down_out + (size_t)t * HIDDEN, h_ffn + (size_t)t * FFN, lw->Wdown, lw->bdown, FFN, HIDDEN);
        }

        for (int t = 0; t < seq_len; t++)
            for (int i = 0; i < HIDDEN; i++)
                hidden[(size_t)t * HIDDEN + i] = down_out[(size_t)t * HIDDEN + i] + ffn_inp[(size_t)t * HIDDEN + i];

        fwrite(hidden, 4, (size_t)seq_len * HIDDEN, of);
        printf("[compare] layer %2d xong (%.1fs)\n", l, (now_ms() - t_start) / 1000.0);
    }

    float *final_h = malloc((size_t)seq_len * HIDDEN * 4);
    for (int t = 0; t < seq_len; t++)
        rmsnorm(final_h + (size_t)t * HIDDEN, hidden + (size_t)t * HIDDEN, norm_w, HIDDEN);
    fwrite(final_h, 4, (size_t)seq_len * HIDDEN, of);

    float *logits = malloc((size_t)VOCAB * 4);
    for (int t = 0; t < seq_len; t++) {
        linear_f32(logits, final_h + (size_t)t * HIDDEN, embed, NULL, HIDDEN, VOCAB);
        fwrite(logits, 4, VOCAB, of);
        if (t == seq_len - 1) {
            int top_idx[5] = {-1,-1,-1,-1,-1};
            float top_val[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
            for (int v = 0; v < VOCAB; v++) {
                float val = logits[v];
                for (int r = 0; r < 5; r++) {
                    if (val > top_val[r]) {
                        for (int s = 4; s > r; s--) { top_val[s]=top_val[s-1]; top_idx[s]=top_idx[s-1]; }
                        top_val[r] = val; top_idx[r] = v;
                        break;
                    }
                }
            }
            printf("[compare] top-5 next-token (TQ33) tai vi tri cuoi:\n");
            for (int r = 0; r < 5; r++) printf("    id=%6d  logit=%.4f\n", top_idx[r], top_val[r]);
        }
    }
    fclose(of);
    printf("[compare] xong (%.1fs) -> %s\n\n", (now_ms() - t_start) / 1000.0, out_path);

    free(hidden); free(cur); free(Q); free(K); free(V); free(attn_concat); free(o_out);
    free(ffn_inp); free(gate); free(up); free(h_ffn); free(down_out); free(scores);
    free(final_h); free(logits);
}

/* ================= Phần B: benchmark tốc độ — KV-cache autoregressive ================= */

/* timers tich luy (chi do o 1-luong de phan tich ty le, khong anh huong benchmark chinh) */
static double g_t_linear = 0, g_t_attn = 0, g_t_norm_rope = 0, g_t_embed_logits = 0;

static int forward_one_token(LayerW *LW, int32_t token_id, int pos, const float *embed,
                              const float *norm_w, float *K_cache, float *V_cache,
                              float *logits_out, int want_logits) {
    static float hidden[HIDDEN], cur[HIDDEN], Qb[Q_DIM], Kb[KV_DIM], Vb[KV_DIM];
    static float attn_concat[Q_DIM], o_out[HIDDEN], ffn_inp[HIDDEN];
    static float gate[FFN], up[FFN], h_ffn[FFN], down_out[HIDDEN];
    static float scores[MAX_POS];
    double t0;

    memcpy(hidden, embed + (size_t)token_id * HIDDEN, HIDDEN * 4);

    for (int l = 0; l < N_LAYER; l++) {
        LayerW *lw = &LW[l];
        t0 = now_ms();
        rmsnorm(cur, hidden, lw->attn_norm_w, HIDDEN);
        g_t_norm_rope += now_ms() - t0;

        t0 = now_ms();
        linear_tq33(Qb, cur, lw->Wq, lw->bq, HIDDEN, Q_DIM);
        linear_tq33(Kb, cur, lw->Wk, lw->bk, HIDDEN, KV_DIM);
        linear_tq33(Vb, cur, lw->Wv, lw->bv, HIDDEN, KV_DIM);
        g_t_linear += now_ms() - t0;

        t0 = now_ms();
        for (int h = 0; h < N_HEAD; h++) {
            float *qh = Qb + (size_t)h * HEAD_DIM;
            rmsnorm(qh, qh, lw->qn, HEAD_DIM);
            rope_neox(qh, HEAD_DIM, pos);
        }
        for (int h = 0; h < N_KV_HEAD; h++) {
            float *kh = Kb + (size_t)h * HEAD_DIM;
            rmsnorm(kh, kh, lw->kn, HEAD_DIM);
            rope_neox(kh, HEAD_DIM, pos);
        }
        g_t_norm_rope += now_ms() - t0;

        /* ghi vao KV-cache tai vi tri pos */
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
        g_t_attn += now_ms() - t0;

        t0 = now_ms();
        linear_tq33(o_out, attn_concat, lw->Wo, lw->bo, Q_DIM, HIDDEN);
        g_t_linear += now_ms() - t0;

        for (int i = 0; i < HIDDEN; i++) ffn_inp[i] = hidden[i] + o_out[i];

        t0 = now_ms();
        rmsnorm(cur, ffn_inp, lw->ffn_norm_w, HIDDEN);
        g_t_norm_rope += now_ms() - t0;

        t0 = now_ms();
        linear_tq33(gate, cur, lw->Wgate, lw->bgate, HIDDEN, FFN);
        linear_tq33(up, cur, lw->Wup, lw->bup, HIDDEN, FFN);
        for (int i = 0; i < FFN; i++) h_ffn[i] = silu(gate[i]) * up[i];
        linear_tq33(down_out, h_ffn, lw->Wdown, lw->bdown, FFN, HIDDEN);
        g_t_linear += now_ms() - t0;

        for (int i = 0; i < HIDDEN; i++) hidden[i] = down_out[i] + ffn_inp[i];
    }

    t0 = now_ms();
    rmsnorm(cur, hidden, norm_w, HIDDEN);
    int next_id = -1;
    if (want_logits) {
        linear_f32(logits_out, cur, embed, NULL, HIDDEN, VOCAB);
        float best = -1e30f;
        for (int v = 0; v < VOCAB; v++) if (logits_out[v] > best) { best = logits_out[v]; next_id = v; }
    }
    g_t_embed_logits += now_ms() - t0;
    return next_id;
}

static void run_benchmark(LayerW *LW, const int32_t *prompt_tokens, int prompt_len,
                          const float *embed, const float *norm_w, int n_gen, const int *thread_list, int n_thread_list) {
    float *K_cache = malloc((size_t)N_LAYER * MAX_POS * KV_DIM * 4);
    float *V_cache = malloc((size_t)N_LAYER * MAX_POS * KV_DIM * 4);
    float *logits = malloc((size_t)VOCAB * 4);

    printf("\n================ PHAN B: BENCHMARK TOC DO (KV-cache, %d token sinh) ================\n", n_gen);
    for (int ti = 0; ti < n_thread_list; ti++) {
        g_nthreads = thread_list[ti];
        g_t_linear = g_t_attn = g_t_norm_rope = g_t_embed_logits = 0;

        /* prefill prompt tuan tu (khong tinh gio) */
        int32_t cur_tok = 0;
        int pos = 0;
        for (; pos < prompt_len; pos++)
            forward_one_token(LW, prompt_tokens[pos], pos, embed, norm_w, K_cache, V_cache, logits, 0);
        /* logits cho vi tri cuoi cua prompt (de lay token dau tien sinh ra) */
        int next = forward_one_token(LW, prompt_tokens[prompt_len - 1], prompt_len - 1, embed, norm_w,
                                      K_cache, V_cache, logits, 1);
        cur_tok = next;

        if (ti == 0) {
            printf("[kiem chung KV-cache] token dau tien sinh ra (tai vi tri %d, dung cache "
                   "vi tri 0..%d) = %d  (PHAI KHOP top-1=11 da xac nhan boi compare_phase2.py "
                   "o duong batch/prefill khong-cache)\n", prompt_len - 1, prompt_len - 1, next);
        }

        double t0 = now_ms();
        g_t_linear = g_t_attn = g_t_norm_rope = g_t_embed_logits = 0; /* reset sau warmup prefill */
        int gen_tokens[256];
        for (int g = 0; g < n_gen; g++) {
            int p = prompt_len + g;
            if (p >= MAX_POS - 1) break;
            int nx = forward_one_token(LW, cur_tok, p, embed, norm_w, K_cache, V_cache, logits, 1);
            if (g < 256) gen_tokens[g] = nx;
            cur_tok = nx;
        }
        if (ti == 0) {
            printf("[kiem chung KV-cache] %d token sinh ra tiep theo (id thoi): ", n_gen);
            for (int g = 0; g < n_gen && g < 256; g++) printf("%d ", gen_tokens[g]);
            printf("\n");
        }
        double dt = now_ms() - t0;
        double toks = n_gen / (dt / 1000.0);
        double total_t = g_t_linear + g_t_attn + g_t_norm_rope + g_t_embed_logits;
        printf("threads=%2d : %7.1f ms / %d token -> %6.2f tok/s  "
               "| breakdown: linear-tq33=%.0f%% attn=%.0f%% norm+rope=%.0f%% embed+logits(f32)=%.0f%%\n",
               g_nthreads, dt, n_gen, toks,
               100.0 * g_t_linear / total_t, 100.0 * g_t_attn / total_t,
               100.0 * g_t_norm_rope / total_t, 100.0 * g_t_embed_logits / total_t);
    }
    free(K_cache); free(V_cache); free(logits);
}

/* ================= main ================= */

int main(int argc, char **argv) {
    const char *wdir = argc > 1 ? argv[1] : "D:\\Bit-Translate-data\\tq33_runner\\weights_f32";
    const char *tqdir = argc > 2 ? argv[2] : "D:\\Bit-Translate-data\\tq33_runner\\tq33_packed";
    const char *odir = argc > 3 ? argv[3] : "D:\\Bit-Translate-data\\tq33_runner\\oracle";
    const char *out_path = argc > 4 ? argv[4] : "D:\\Bit-Translate-data\\tq33_runner\\runner_output_tq33.bin";
    int n_gen = argc > 5 ? atoi(argv[5]) : 30;

    build_tq33_tables();
    load_meta(wdir);
    load_weights(wdir);
    load_tq33_meta(tqdir);
    load_tq33_data(tqdir);

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

    const float *embed = find_t("model.embed_tokens.weight");
    const float *norm_w = find_t("model.norm.weight");

    LayerW LW[N_LAYER];
    for (int l = 0; l < N_LAYER; l++) load_layer_weights(&LW[l], l);

    g_nthreads = 1;
    printf("================ PHAN A: SO KHOP ORACLE (prefill %d token, khong cache) ================\n", seq_len);
    run_compare(LW, tokens, seq_len, embed, norm_w, out_path);

    int threads_try[] = {1, 2, 4, 6, 8, 12, 14};
    int n_try = (int)(sizeof(threads_try) / sizeof(threads_try[0]));
    run_benchmark(LW, tokens, seq_len, embed, norm_w, n_gen, threads_try, n_try);

    return 0;
}
