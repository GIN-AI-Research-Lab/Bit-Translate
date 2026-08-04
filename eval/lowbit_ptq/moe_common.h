/* moe_common.h — logic MoE dung CHUNG giua validate_moe_routing.c (oracle test co lap, GEMV
 * F32 thuan) va qwen3moe_runner_tq33.c (runner that, GEMV bang kernel TQ33). Phan chia se la
 * dung phan DE SAI nhat (softmax toan bo / chon top-k / renormalize / weighted-combine) —
 * phan GEMV gate/up/down KHAC NHAU giua 2 noi (F32 vs TQ33) nen khong chia se duoc bang code,
 * nhung khong sao vi kernel GEMV TQ33 da validate doc lap roi (0.6B Giai doan 2, tai su dung
 * nguyen ven — xem RESEARCH_TQ33_RUNNER.md). Muc dich file nay: bat loi THUAT TOAN dieu phoi
 * (off-by-one trong renorm, sai thu tu softmax/topk...) truoc khi tich hop vao runner day du.
 *
 * Thuat toan da xac nhan DOC TRUC TIEP source llama.cpp that (khong chi tin mo ta nhiem vu):
 *   - llm_graph_context::build_moe_ffn (llama-graph.cpp dong 1914-2140):
 *     probs = softmax(logits) tren TOAN BO n_expert (dong 1962-1964)
 *     selected = argsort_top_k(probs, n_expert_used)  (dong 2030 — top-k LON NHAT)
 *     weights = get_rows(probs, selected)             (dong 2044 — LAY GIA TRI SOFTMAX DA CO,
 *               khong tinh lai/khong softmax rieng tren k gia tri da chon)
 *     if (norm_w) weights = weights / sum(weights)     (dong 2055-2069, co clamp chong chia 0)
 *     w_scale: qwen3moe.cpp goi voi hparams.expert_weights_scale — default field =0.0f
 *              (llama-hparams.h dong 100), KHONG co assignment nao rieng cho QWEN3MOE trong
 *              llama-model.cpp (da grep xac nhan) => giu 0.0f => dieu kien "w_scale!=0.0f"
 *              (dong 2070) SAI => buoc scale-them bi BO QUA hoan toan (0.0f la sentinel
 *              "tat", khong phai "nhan voi 0") — khop mo ta nhiem vu "khong co weight scale
 *              dac biet khac 1.0".
 *   - qwen3moe.cpp dong 136-151: goi build_moe_ffn voi type_op=LLM_FFN_SILU, norm_w=true,
 *     gating_op=SOFTMAX, KHONG co bias nao (gate_inp_b/up_exps_b/gate_exps_b/down_exps_b/
 *     exp_probs_b deu nullptr qua overload rut gon dong 1890-1911).
 *   - SwiGLU: build_moe_ffn dong 2169-2171 `cur = ggml_swiglu_split(cur=gate, up)` khi
 *     has_gate=true, KHONG clamp (swiglu_clamp_exp mac dinh 0.0f cho moi arch tru
 *     dflash/deepseek4/step35 — da grep xac nhan QWEN3MOE khong nam trong danh sach nay).
 *     ggml_swiglu_split(a,b) = silu(a)*b (xac nhan qua backward-pass hint trong ggml.c dong
 *     7070-7078: d/da dung silu_back(grad*b, a) => forward = silu(a)*b). Voi a=gate,b=up
 *     => h = silu(gate)*up, KHOP mo ta nhiem vu.
 */
#ifndef MOE_COMMON_H
#define MOE_COMMON_H

#include <math.h>
#include <stddef.h>

#define MOE_MAX_K 16

static inline float moe_silu(float x) { return x / (1.0f + expf(-x)); }

/* softmax tai cho, n phan tu (dung double cho tong de on dinh, ep lai float — n<=128 nen
 * khong quan trong lam, nhung lam dung tu dau cho chac). */
static inline void moe_softmax_inplace(float *x, int n) {
    float m = -1e30f;
    for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += (double)x[i]; }
    float inv = (float)(1.0 / s);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* Chon k gia tri LON NHAT tu probs[n_expert] (probs DA la softmax rieng, ham nay KHONG tu
 * softmax) — tra ve out_idx[k] (giam dan theo gia tri) va out_weight[k] = renormalize sao
 * cho sum(out_weight)=1. Insertion-sort O(k*n_expert) — k<=16, n_expert<=256 nen re hon
 * argsort day du nhieu, dung y het huong build_moe_ffn dong 2030+2044+2055-2069 (norm_w=true,
 * w_scale=0=tat). KHONG re-softmax tren k gia tri da chon — chi renormalize gia tri softmax
 * GOC cua chung (get_rows roi div-by-sum, khong phai softmax-lai). */
static inline void moe_topk_renorm(const float *probs, int n_expert, int k,
                                    int *out_idx, float *out_weight) {
    float top_val[MOE_MAX_K];
    int top_idx[MOE_MAX_K];
    for (int r = 0; r < k; r++) { top_val[r] = -1e30f; top_idx[r] = -1; }
    for (int e = 0; e < n_expert; e++) {
        float v = probs[e];
        if (v > top_val[k - 1]) {
            int r = k - 1;
            while (r > 0 && v > top_val[r - 1]) {
                top_val[r] = top_val[r - 1];
                top_idx[r] = top_idx[r - 1];
                r--;
            }
            top_val[r] = v;
            top_idx[r] = e;
        }
    }
    double sum = 0.0;
    for (int r = 0; r < k; r++) sum += (double)top_val[r];
    /* clamp chong chia 0 giong ggml_clamp(weights_sum, 6.103515625e-5, INFINITY) dong 2062 */
    if (sum < 6.103515625e-5) sum = 6.103515625e-5;
    float inv = (float)(1.0 / sum);
    for (int r = 0; r < k; r++) { out_idx[r] = top_idx[r]; out_weight[r] = top_val[r] * inv; }
}

/* out[hidden] = sum_{i=0}^{k-1} weight[i] * expert_out_flat[i*hidden .. i*hidden+hidden-1] */
static inline void moe_combine(float *out, const float *expert_out_flat, const float *weight,
                                int k, int hidden) {
    for (int h = 0; h < hidden; h++) out[h] = 0.0f;
    for (int i = 0; i < k; i++) {
        const float *eo = expert_out_flat + (size_t)i * hidden;
        float w = weight[i];
        for (int h = 0; h < hidden; h++) out[h] += w * eo[h];
    }
}

#endif /* MOE_COMMON_H */
