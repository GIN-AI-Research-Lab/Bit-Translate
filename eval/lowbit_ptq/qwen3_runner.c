/* TQ33 Runner — Giai đoạn 1: forward pass Qwen3-0.6B (28 layer) THUẦN float32, C độc lập
 * (không ggml, không llama.cpp). Đọc weights_f32/weights.bin (dump từ
 * export_full_model_f32.py) + oracle/tokens.bin (token id thật, từ ref_forward_pytorch.py),
 * chạy đúng thứ tự phép toán Qwen3 (xem CLAUDE.md/context nhiệm vụ mục 7), in hidden state
 * SAU MỖI LAYER + logits cuối ra file để so khớp với oracle PyTorch (compare_phase1.py).
 *
 * Kiến trúc xác nhận từ transformers/models/qwen3/modeling_qwen3.py THẬT (đã đọc trực tiếp
 * lúc viết code này, không đoán):
 *   - RoPE: NEOX (xoay cặp x[j], x[j+dim/2]), theta_scale = theta^(-2/head_dim)
 *   - q_norm/k_norm: RMSNorm per-head (dim=head_dim=128), áp dụng SAU reshape, TRƯỚC RoPE
 *   - GQA: q head h dùng kv head h/(n_head/n_kv_head) = h/2 (nhóm liền kề, khớp
 *     repeat_kv/torch.repeat_interleave convention)
 *   - bias CÓ THẬT trên cả 7 loại linear (q/k/v/o/gate/up/down) dù kiến trúc Qwen3 chuẩn
 *     không có (attention_bias=false, MLP luôn bias=False) — ckpt bake tự thêm qua
 *     LearnQLinear, đã verify absmax > 0 trên toàn bộ 196 bias tensor.
 *   - output head: dùng model.embed_tokens.weight (không dùng lm_head.weight dù tồn tại
 *     riêng trong ckpt — xem export_full_model_f32.py để biết lý do, đã verify bằng PPL).
 *
 * Build:  python -m ziglang cc -O3 -mavx2 -mfma -o qwen3_runner.exe qwen3_runner.c -lm
 * Run:    qwen3_runner.exe <weights_dir> <oracle_dir> <out_bin>
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_LAYER 28
#define HIDDEN 1024
#define N_HEAD 16
#define N_KV_HEAD 8
#define HEAD_DIM 128
#define Q_DIM (N_HEAD * HEAD_DIM)   /* 2048 */
#define KV_DIM (N_KV_HEAD * HEAD_DIM) /* 1024 */
#define FFN 3072
#define VOCAB 151936
#define RMS_EPS 1e-6f
#define ROPE_THETA 1000000.0f
#define MAX_TENSORS 600

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
    if (g_n_meta > MAX_TENSORS) { fprintf(stderr, "MAX_TENSORS qua nho (%d)\n", g_n_meta); exit(1); }
    for (int i = 0; i < g_n_meta; i++) {
        TensorMeta *m = &g_meta[i];
        if (fscanf(f, "%159s %d", m->name, &m->ndim) != 2) { fprintf(stderr, "loi doc dong %d\n", i); exit(1); }
        for (int d = 0; d < m->ndim; d++) {
            if (fscanf(f, "%lld", (long long *)&m->dim[d]) != 1) { fprintf(stderr, "loi doc dim\n"); exit(1); }
        }
        if (fscanf(f, "%lld %lld", (long long *)&m->offset, (long long *)&m->numel) != 2) {
            fprintf(stderr, "loi doc offset/numel\n"); exit(1);
        }
    }
    fclose(f);
    printf("da doc meta: %d tensor\n", g_n_meta);
}

/* CHÚ Ý: KHÔNG dùng ftell/fseek để lấy size — trên Windows `long` là 32-bit (LLP64)
 * dù chương trình 64-bit, ftell() sẽ TRÀN SỐ với file >2GB (weights.bin ~3GB). Tính
 * tổng số phần tử trực tiếp từ meta (đã đọc offset+numel của tensor cuối) rồi fread
 * đúng số đó — fread/fwrite nhận size_t (64-bit trên Win64) nên không bị giới hạn này. */
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
    printf("dang doc %s (%.3f GB, %zu phan tu float32)...\n", path, total_elems * 4 / 1e9, total_elems);
    g_w = (float *)malloc(total_elems * sizeof(float));
    if (!g_w) { fprintf(stderr, "OOM %zu phan tu\n", total_elems); exit(1); }
    double t0 = now_ms();
    size_t got = fread(g_w, sizeof(float), total_elems, f);
    if (got != total_elems) { fprintf(stderr, "doc hut: %zu/%zu phan tu\n", got, total_elems); exit(1); }
    fclose(f);
    printf("doc xong (%.1fs)\n", (now_ms() - t0) / 1000.0);
}

static const float *find_t(const char *name) {
    for (int i = 0; i < g_n_meta; i++)
        if (strcmp(g_meta[i].name, name) == 0) return g_w + g_meta[i].offset;
    fprintf(stderr, "KHONG TIM THAY tensor: %s\n", name);
    exit(1);
}

static const float *find_layer(int l, const char *suffix) {
    char name[160];
    snprintf(name, sizeof name, "model.layers.%d.%s", l, suffix);
    return find_t(name);
}

/* ---------- các phép toán cơ bản ---------- */

/* RMSNorm: out[i] = w[i] * x[i] / sqrt(mean(x^2)+eps) — CHÍNH XÁC công thức
 * Qwen3RMSNorm (transformers/models/qwen3/modeling_qwen3.py): tính variance = mean(x^2)
 * (không trừ mean trước, khác LayerNorm thường). */
static void rmsnorm(float *out, const float *x, const float *w, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float inv = 1.0f / sqrtf((float)(ss / n) + RMS_EPS);
    for (int i = 0; i < n; i++) out[i] = w[i] * (x[i] * inv);
}

/* y = W @ x + b ,  W shape [out_dim, in_dim] row-major (convention nn.Linear: weight[o,i]) */
static void linear(float *out, const float *x, const float *W, const float *b,
                    int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        const float *row = W + (size_t)o * in_dim;
        float acc = b ? b[o] : 0.0f;
        for (int i = 0; i < in_dim; i++) acc += row[i] * x[i];
        out[o] = acc;
    }
}

/* RoPE kiểu NEOX: xoay cặp (x[j], x[j+dim/2]) cho j=0..dim/2-1, theta = pos*theta_scale^j.
 * Khớp CHÍNH XÁC rotate_half()+apply_rotary_pos_emb() của HF (đã đối chiếu 2 nguồn:
 * ggml_compute_forward_rope_flt (GGML_ROPE_TYPE_NEOX) và modeling_qwen3.py — cùng công thức). */
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

/* ---------- forward pass ---------- */

int main(int argc, char **argv) {
    const char *wdir = argc > 1 ? argv[1] : "D:\\Bit-Translate-data\\tq33_runner\\weights_f32";
    const char *odir = argc > 2 ? argv[2] : "D:\\Bit-Translate-data\\tq33_runner\\oracle";
    const char *out_path = argc > 3 ? argv[3] : "D:\\Bit-Translate-data\\tq33_runner\\runner_output.bin";

    load_meta(wdir);
    load_weights(wdir);

    /* đọc token ids */
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
    printf("\n");

    const float *embed = find_t("model.embed_tokens.weight"); /* [VOCAB, HIDDEN] */

    /* buffers */
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

    /* output buffer: header + 28*seq*HIDDEN (raw layer out) + seq*HIDDEN (final) + seq*VOCAB (logits) */
    FILE *of = fopen(out_path, "wb");
    if (!of) { fprintf(stderr, "khong ghi duoc %s\n", out_path); return 1; }
    int32_t hdr[4] = {seq_len, HIDDEN, VOCAB, N_LAYER};
    fwrite(hdr, 4, 4, of);

    /* embed lookup */
    for (int t = 0; t < seq_len; t++)
        memcpy(hidden + (size_t)t * HIDDEN, embed + (size_t)tokens[t] * HIDDEN, HIDDEN * 4);

    double t_start = now_ms();
    for (int l = 0; l < N_LAYER; l++) {
        const float *attn_norm_w = find_layer(l, "input_layernorm.weight");
        const float *Wq = find_layer(l, "self_attn.q_proj.weight"), *bq = find_layer(l, "self_attn.q_proj.bias");
        const float *Wk = find_layer(l, "self_attn.k_proj.weight"), *bk = find_layer(l, "self_attn.k_proj.bias");
        const float *Wv = find_layer(l, "self_attn.v_proj.weight"), *bv = find_layer(l, "self_attn.v_proj.bias");
        const float *Wo = find_layer(l, "self_attn.o_proj.weight"), *bo = find_layer(l, "self_attn.o_proj.bias");
        const float *qn = find_layer(l, "self_attn.q_norm.weight");
        const float *kn = find_layer(l, "self_attn.k_norm.weight");
        const float *ffn_norm_w = find_layer(l, "post_attention_layernorm.weight");
        const float *Wgate = find_layer(l, "mlp.gate_proj.weight"), *bgate = find_layer(l, "mlp.gate_proj.bias");
        const float *Wup = find_layer(l, "mlp.up_proj.weight"), *bup = find_layer(l, "mlp.up_proj.bias");
        const float *Wdown = find_layer(l, "mlp.down_proj.weight"), *bdown = find_layer(l, "mlp.down_proj.bias");

        /* a. attn_norm */
        for (int t = 0; t < seq_len; t++)
            rmsnorm(cur + (size_t)t * HIDDEN, hidden + (size_t)t * HIDDEN, attn_norm_w, HIDDEN);

        /* b+c+d+e. Q/K/V proj + per-head norm + RoPE */
        for (int t = 0; t < seq_len; t++) {
            linear(Q + (size_t)t * Q_DIM, cur + (size_t)t * HIDDEN, Wq, bq, HIDDEN, Q_DIM);
            linear(K + (size_t)t * KV_DIM, cur + (size_t)t * HIDDEN, Wk, bk, HIDDEN, KV_DIM);
            linear(V + (size_t)t * KV_DIM, cur + (size_t)t * HIDDEN, Wv, bv, HIDDEN, KV_DIM);
            for (int h = 0; h < N_HEAD; h++) {
                float *qh = Q + (size_t)t * Q_DIM + (size_t)h * HEAD_DIM;
                rmsnorm(qh, qh, qn, HEAD_DIM);
                rope_neox(qh, HEAD_DIM, t);
            }
            for (int h = 0; h < N_KV_HEAD; h++) {
                float *kh = K + (size_t)t * KV_DIM + (size_t)h * HEAD_DIM;
                rmsnorm(kh, kh, kn, HEAD_DIM);
                rope_neox(kh, HEAD_DIM, t);
            }
        }

        /* g(part1). attention brute-force O(seq^2), causal, GQA (q head h -> kv head h/2) */
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

        /* g(part2). o_proj */
        for (int t = 0; t < seq_len; t++)
            linear(o_out + (size_t)t * HIDDEN, attn_concat + (size_t)t * Q_DIM, Wo, bo, Q_DIM, HIDDEN);

        /* h. residual */
        for (int t = 0; t < seq_len; t++)
            for (int i = 0; i < HIDDEN; i++)
                ffn_inp[(size_t)t * HIDDEN + i] = hidden[(size_t)t * HIDDEN + i] + o_out[(size_t)t * HIDDEN + i];

        /* i. ffn_norm */
        for (int t = 0; t < seq_len; t++)
            rmsnorm(cur + (size_t)t * HIDDEN, ffn_inp + (size_t)t * HIDDEN, ffn_norm_w, HIDDEN);

        /* j. SwiGLU */
        for (int t = 0; t < seq_len; t++) {
            linear(gate + (size_t)t * FFN, cur + (size_t)t * HIDDEN, Wgate, bgate, HIDDEN, FFN);
            linear(up + (size_t)t * FFN, cur + (size_t)t * HIDDEN, Wup, bup, HIDDEN, FFN);
            for (int i = 0; i < FFN; i++) {
                size_t idx = (size_t)t * FFN + i;
                h_ffn[idx] = silu(gate[idx]) * up[idx];
            }
            linear(down_out + (size_t)t * HIDDEN, h_ffn + (size_t)t * FFN, Wdown, bdown, FFN, HIDDEN);
        }

        /* k. residual -> input cho layer sau */
        for (int t = 0; t < seq_len; t++)
            for (int i = 0; i < HIDDEN; i++)
                hidden[(size_t)t * HIDDEN + i] = down_out[(size_t)t * HIDDEN + i] + ffn_inp[(size_t)t * HIDDEN + i];

        fwrite(hidden, 4, (size_t)seq_len * HIDDEN, of);
        printf("layer %2d xong (%.1fs)\n", l, (now_ms() - t_start) / 1000.0);
    }

    /* final norm */
    const float *norm_w = find_t("model.norm.weight");
    float *final_h = malloc((size_t)seq_len * HIDDEN * 4);
    for (int t = 0; t < seq_len; t++)
        rmsnorm(final_h + (size_t)t * HIDDEN, hidden + (size_t)t * HIDDEN, norm_w, HIDDEN);
    fwrite(final_h, 4, (size_t)seq_len * HIDDEN, of);

    /* logits = final @ embed_tokens.weight^T  (embed dùng chung làm output head, xem ghi chú đầu file) */
    float *logits = malloc((size_t)VOCAB * 4);
    for (int t = 0; t < seq_len; t++) {
        linear(logits, final_h + (size_t)t * HIDDEN, embed, NULL, HIDDEN, VOCAB);
        fwrite(logits, 4, VOCAB, of);
        if (t == seq_len - 1) {
            /* top-5 tại vị trí cuối, in ra cho người kiểm tra nhanh */
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
            printf("top-5 next-token (C runner) tai vi tri cuoi:\n");
            for (int r = 0; r < 5; r++) printf("    id=%6d  logit=%.4f\n", top_idx[r], top_val[r]);
        }
    }

    fclose(of);
    printf("\nxong toan bo forward pass (%.1fs). Da ghi -> %s\n",
           (now_ms() - t_start) / 1000.0, out_path);
    return 0;
}
