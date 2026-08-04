/* Giai doan A — validate thuat toan dieu phoi MoE CO LAP (khong TQ33, khong load 30B that).
 * Doc du lieu ngau nhien + oracle PyTorch tu validate_moe_routing_gen.py, chay CUNG thuat
 * toan bang C (dung moe_common.h — se duoc TAI SU DUNG NGUYEN VAN trong runner day du), so
 * rel err voi oracle. Muc dich: bat loi thuat toan (off-by-one renorm, sai thu tu
 * softmax/topk...) TRUOC khi tich hop vao runner 30B day du.
 *
 * Build:  python -m ziglang cc -O2 -o validate_moe_routing.exe validate_moe_routing.c -lm
 * Run:    validate_moe_routing.exe D:\Bit-Translate-data\tq33_30b\moe_validate
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "moe_common.h"

static float *read_f32(const char *dir, const char *name, size_t n) {
    char path[512];
    snprintf(path, sizeof path, "%s\\%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "khong mo duoc %s\n", path); exit(1); }
    float *p = (float *)malloc(n * sizeof(float));
    if (fread(p, sizeof(float), n, f) != n) { fprintf(stderr, "doc hut %s\n", path); exit(1); }
    fclose(f);
    return p;
}

/* dot product don gian (khong AVX2 — Giai doan A chi test thuat toan dieu phoi, kich thuoc
 * nho nen toc do khong quan trong; kernel GEMV that (AVX2 TQ33) da validate rieng o 0.6B). */
static float dot_f32(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "D:\\Bit-Translate-data\\tq33_30b\\moe_validate";

    char path[512];
    snprintf(path, sizeof path, "%s\\meta.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "thieu meta.txt\n"); return 1; }
    int n_expert, n_active, hidden, moe_ffn, n_tokens;
    if (fscanf(f, "%d %d %d %d %d", &n_expert, &n_active, &hidden, &moe_ffn, &n_tokens) != 5) return 1;
    fclose(f);
    printf("n_expert=%d n_active=%d hidden=%d moe_ffn=%d n_tokens=%d\n",
           n_expert, n_active, hidden, moe_ffn, n_tokens);

    float *router_w = read_f32(dir, "router_w.bin", (size_t)n_expert * hidden);
    float *gate_w = read_f32(dir, "gate_w.bin", (size_t)n_expert * moe_ffn * hidden);
    float *up_w = read_f32(dir, "up_w.bin", (size_t)n_expert * moe_ffn * hidden);
    float *down_w = read_f32(dir, "down_w.bin", (size_t)n_expert * hidden * moe_ffn);
    float *x = read_f32(dir, "x.bin", (size_t)n_tokens * hidden);
    float *oracle_out = read_f32(dir, "oracle_out.bin", (size_t)n_tokens * hidden);
    float *oracle_weight = read_f32(dir, "oracle_weight.bin", (size_t)n_tokens * n_active);

    snprintf(path, sizeof path, "%s\\oracle_idx.bin", dir);
    f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "thieu oracle_idx.bin\n"); return 1; }
    int32_t *oracle_idx = (int32_t *)malloc((size_t)n_tokens * n_active * sizeof(int32_t));
    if (fread(oracle_idx, sizeof(int32_t), (size_t)n_tokens * n_active, f) != (size_t)n_tokens * n_active) return 1;
    fclose(f);

    float *probs = (float *)malloc(n_expert * sizeof(float));
    float *gate = (float *)malloc(moe_ffn * sizeof(float));
    float *up = (float *)malloc(moe_ffn * sizeof(float));
    float *h_ffn = (float *)malloc(moe_ffn * sizeof(float));
    float *expert_out_flat = (float *)malloc((size_t)n_active * hidden * sizeof(float));
    float *my_out = (float *)malloc((size_t)n_tokens * hidden * sizeof(float));
    int *idx = (int *)malloc(n_active * sizeof(int));
    float *weight = (float *)malloc(n_active * sizeof(float));

    int idx_mismatch = 0;
    double werr = 0, wref = 0;

    for (int t = 0; t < n_tokens; t++) {
        const float *xt = x + (size_t)t * hidden;

        /* 1) router logits + softmax TREN TOAN BO n_expert */
        for (int e = 0; e < n_expert; e++)
            probs[e] = dot_f32(router_w + (size_t)e * hidden, xt, hidden);
        moe_softmax_inplace(probs, n_expert);

        /* 2) topk + renormalize (dung ham CHIA SE se tai su dung trong runner that) */
        moe_topk_renorm(probs, n_expert, n_active, idx, weight);

        /* kiem idx/weight khop oracle (thu tu GIAM DAN nhu topk PyTorch) */
        for (int j = 0; j < n_active; j++) {
            if (idx[j] != oracle_idx[t * n_active + j]) {
                idx_mismatch++;
                printf("  [LECH idx] token %d vi tri %d: C=%d oracle=%d\n", t, j, idx[j],
                       oracle_idx[t * n_active + j]);
            }
            double d = (double)weight[j] - oracle_weight[t * n_active + j];
            werr += d * d;
            wref += (double)oracle_weight[t * n_active + j] * oracle_weight[t * n_active + j];
        }

        /* 3) moi expert duoc chon: gate/up/down + SwiGLU (F32 GEMV thuan) */
        for (int j = 0; j < n_active; j++) {
            int e = idx[j];
            for (int i = 0; i < moe_ffn; i++) {
                gate[i] = dot_f32(gate_w + ((size_t)e * moe_ffn + i) * hidden, xt, hidden);
                up[i] = dot_f32(up_w + ((size_t)e * moe_ffn + i) * hidden, xt, hidden);
                h_ffn[i] = moe_silu(gate[i]) * up[i];
            }
            float *eo = expert_out_flat + (size_t)j * hidden;
            for (int o = 0; o < hidden; o++)
                eo[o] = dot_f32(down_w + (size_t)e * hidden * moe_ffn + (size_t)o * moe_ffn, h_ffn, moe_ffn);
        }

        /* 4) weighted combine (ham chia se) */
        moe_combine(my_out + (size_t)t * hidden, expert_out_flat, weight, n_active, hidden);
    }

    double err = 0, ref = 0;
    for (size_t i = 0; i < (size_t)n_tokens * hidden; i++) {
        double d = (double)my_out[i] - oracle_out[i];
        err += d * d;
        ref += (double)oracle_out[i] * oracle_out[i];
    }
    double rel_out = sqrt(err / (ref + 1e-30));
    double rel_w = sqrt(werr / (wref + 1e-30));

    printf("\n=== KET QUA GIAI DOAN A ===\n");
    printf("idx mismatch: %d / %d (0 la DAT)\n", idx_mismatch, n_tokens * n_active);
    printf("rel err weight (renorm)   : %.3e  %s\n", rel_w, rel_w < 1e-5 ? "OK" : "LECH!");
    printf("rel err moe_output (final): %.3e  %s\n", rel_out, rel_out < 1e-4 ? "OK" : "LECH!");
    if (idx_mismatch == 0 && rel_w < 1e-5 && rel_out < 1e-4)
        printf("=> GIAI DOAN A: DAT — thuat toan dieu phoi (softmax/topk/renorm/combine) DUNG.\n");
    else
        printf("=> GIAI DOAN A: THAT BAI — co bug thuat toan, DUNG LAI truoc khi sang Giai doan B.\n");

    return (idx_mismatch == 0 && rel_w < 1e-5 && rel_out < 1e-4) ? 0 : 1;
}
