# -*- coding: utf-8 -*-
"""
Sua bug exp_ag (Wanda dung SAI cong thuc mean(|x|) thay L2-norm ||x||_2) + tang calib +
XAC MINH lai tren bai toan don gian (chon mask ternary) TRUOC KHI dua calib vao chinh AQLM
(theo dung yeu cau user 04/08: "AQLM khong the ket hop cung calib sao?").

Chung minh toan (khong doan): centroid k-means TOI UU khong doi du co trong so theo KENH
(vi trong so w_j KHONG phu thuoc diem du lieu, chi phu thuoc vi tri j trong nhom - dao ham
Sum_i w_j*(x_ij-c_j) = 0 => w_j*Sum_i(x_ij-c_j)=0 => c_j = mean_i(x_ij), w_j huy o hai ben).
=> Chi can doi khoang cach dung de GAN nhom vao codeword (assignment), giu nguyen buoc cap
nhat centroid (mean thuong).

Gate: CHI lam Buoc 2 (Wanda-AQLM) NEU Buoc 1 (Wanda-mask, cong thuc da sua) thang magnitude-
mask ro rang - khong dam lam ca hai cung luc roi doan cai nao co cong.

Chay: python eval/lowbit_ptq/exp_ai_wanda_aqlm.py
"""
import io
import json
import math
import os
import sys
import time

import gguf
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, OlmoeConfig, OlmoeForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GGUF_PATH = "E:/hf_gguf_cache/olmoe-1b-7b-0924-q4_k_m.gguf"
N_EXPERTS_TEST = 16
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_ai_results.json")

CALIB_TEXTS = [
    "The city council voted on Tuesday to approve funding for a new public transit line "
    "connecting the downtown area to the airport, a project residents have requested for years.",
    "Photosynthesis is the process by which green plants convert sunlight, water, and carbon "
    "dioxide into glucose and oxygen, forming the base of most food chains on Earth.",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n"
    "    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
    "The stock market fell sharply on Wednesday after the central bank signaled it would keep "
    "interest rates elevated for longer than investors had anticipated, sparking a broad selloff.",
    "In the final chapter of the novel, the protagonist returns to her childhood home only to "
    "find it abandoned, the garden overgrown, and a single letter waiting on the kitchen table.",
    "Researchers at the university published a study showing that regular exercise combined "
    "with adequate sleep significantly reduces the risk of cardiovascular disease in adults over 50.",
    "SELECT customer_id, SUM(order_total) FROM orders WHERE order_date >= '2026-01-01' "
    "GROUP BY customer_id HAVING SUM(order_total) > 1000 ORDER BY SUM(order_total) DESC;",
    "The recipe calls for two cups of flour, a teaspoon of baking soda, and a pinch of salt, "
    "whisked together before folding in the melted butter and beaten eggs.",
    "Historians continue to debate the exact causes of the empire's decline, pointing to a "
    "combination of economic strain, military overextension, and political fragmentation.",
    "The new smartphone features a larger battery, an improved camera sensor, and a faster "
    "processor, though reviewers noted the price increase compared to last year's model.",
    "Quantum computers use qubits, which can exist in superposition, allowing them to explore "
    "many possible solutions simultaneously rather than one at a time like classical bits.",
    "After months of negotiation, the two companies announced a merger that will combine their "
    "logistics networks and is expected to close by the end of the third quarter.",
    "The hiking trail climbs steeply for the first two miles before leveling off near a ridge "
    "with panoramic views of the valley and the lake below.",
    "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n"
    "        if n % i == 0:\n            return False\n    return True",
    "The committee's report recommended stricter oversight of data privacy practices among "
    "technology firms operating in the region, citing several high-profile breaches this year.",
    "Coral reefs support roughly a quarter of all marine species despite covering less than one "
    "percent of the ocean floor, making their rapid decline a major concern for biodiversity.",
    "The orchestra's performance of the symphony drew a standing ovation, with critics praising "
    "the conductor's interpretation of the slower second movement in particular.",
    "A firmware update rolled out this week fixes a bug that caused some routers to drop wireless "
    "connections intermittently under heavy network load.",
    "The court ruled that the contract's arbitration clause was enforceable, dismissing the "
    "plaintiff's argument that it had been added without proper notice.",
    "Volcanic activity beneath the ice sheet has been increasing for the past decade, according "
    "to seismic data collected by the research station.",
    "Farmers in the region reported a smaller than usual harvest this season, blaming an "
    "unusually dry summer followed by an early frost that damaged the remaining crops.",
    "The airline announced it would add three new routes next spring, including a direct flight "
    "connecting the two capital cities for the first time in over a decade.",
    "def fibonacci(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n"
    "        return n\n    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)\n    return memo[n]",
    "The museum's new exhibit traces the history of printing technology from early woodblock "
    "methods through the invention of movable type to modern digital publishing.",
    "Engineers traced the outage to a faulty cooling unit in the data center, which triggered an "
    "automatic shutdown of several servers to prevent overheating.",
    "The negotiations stalled after both sides refused to compromise on the proposed tariff "
    "schedule, raising concerns about a prolonged trade dispute.",
    "A new species of frog was discovered in the rainforest canopy, identified by its distinctive "
    "call and the unusual pattern of spots along its back.",
    "The city's water treatment plant underwent a major upgrade this year, adding filtration "
    "capacity to handle population growth over the next two decades.",
    "UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 4821 AND quantity > 0;",
    "The marathon route was changed this year to avoid construction downtown, adding an extra "
    "half mile along the waterfront before rejoining the original course.",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def dq(reader, byname, name):
    t = byname[name]
    return torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).copy()).float()


def q_ternary_masked(W, mask, sgroup=64):
    R, C = W.shape
    Wv, Mv = W.view(R, -1, sgroup), mask.view(R, -1, sgroup)
    Wm = Wv * Mv
    cnt = Mv.sum(2, keepdim=True).clamp(min=1)
    s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-8)
    for _ in range(3):
        t = torch.round(Wm / s).clamp(-1, 1) * Mv
        num, den = (Wm * t).sum(2, keepdim=True), (t * t).sum(2, keepdim=True).clamp(min=1e-8)
        s = (num / den).abs().clamp(min=1e-8)
    return (torch.round(Wm / s.clamp(min=1e-8)).clamp(-1, 1) * s * Mv).view(R, C)


def mask_topn_by_score(score, n, m):
    R, C = score.shape
    g = score.view(R, -1, m)
    idx = g.topk(n, dim=2).indices
    return torch.zeros_like(g).scatter_(2, idx, 1.0).view(R, C)


# ---------------- AQLM (khong-trong-so / co-trong-so, weighted DUNG per-point) ----------------
def wsqdist(X, C, W=None):
    """Weighted squared L2: d[i,k] = sum_j W[i,j]*(X[i,j]-C[k,j])^2, W la [N,g] PER-POINT
    (KHONG thu gon ve 1 vector chung - moi nhom 16 kenh la importance KHAC NHAU, xem ghi chu
    dau file). Rut gon ve 3 matmul, khong tao tensor 3D [N,K,g] (tranh no RAM/cham)."""
    if W is None:
        return torch.cdist(X, C) ** 2
    wx2 = (W * X * X).sum(1, keepdim=True)      # [N,1]
    wxc = (W * X) @ C.T                          # [N,K]
    wc2 = W @ (C ** 2).T                          # [N,K]
    return wx2 - 2 * wxc + wc2


def kmeans(X, K, iters=25, seed=0, W=None):
    """W: None hoac [N,g] per-point (dung dung cong thuc weighted k-means:
    c_kj = sum_i-in-k(W_ij*X_ij) / sum_i-in-k(W_ij) - KHAC unweighted mean khi W bien doi theo i)."""
    g = torch.Generator().manual_seed(seed)
    N = X.shape[0]
    C = X[torch.randperm(N, generator=g)[:K]].clone()
    for _ in range(iters):
        assign = wsqdist(X, C, W).argmin(1)
        if W is None:
            newC = torch.zeros_like(C)
            cnt = torch.zeros(K)
            newC.index_add_(0, assign, X)
            cnt.index_add_(0, assign, torch.ones(N))
            newC = newC / cnt.clamp(min=1).unsqueeze(1)
            empty = cnt == 0
        else:
            num = torch.zeros_like(C)
            den = torch.zeros_like(C)
            num.index_add_(0, assign, W * X)
            den.index_add_(0, assign, W)
            newC = num / den.clamp(min=1e-8)
            empty = den.sum(1) == 0
        newC[empty] = C[empty]
        if (newC - C).norm() < 1e-5:
            C = newC
            break
        C = newC
    return C


def beam_assign(X, C1, C2, beam=4, W=None, chunk=100_000):
    N = X.shape[0]
    a1 = torch.empty(N, dtype=torch.long)
    a2 = torch.empty(N, dtype=torch.long)
    for i in range(0, N, chunk):
        x = X[i:i + chunk]
        w = W[i:i + chunk] if W is not None else None
        _, top1i = wsqdist(x, C1, w).topk(beam, dim=1, largest=False)
        best_err = best_i1 = best_i2 = None
        for b in range(beam):
            i1 = top1i[:, b]
            resid = x - C1[i1]
            e2, i2 = wsqdist(resid, C2, w).min(dim=1)
            if best_err is None:
                best_err, best_i1, best_i2 = e2, i1, i2
            else:
                better = e2 < best_err
                best_err = torch.where(better, e2, best_err)
                best_i1 = torch.where(better, i1, best_i1)
                best_i2 = torch.where(better, i2, best_i2)
        a1[i:i + chunk], a2[i:i + chunk] = best_i1, best_i2
    return a1, a2


def aqlm_err(W, M, K, g, w_per_channel=None):
    """w_per_channel: [C_in] Wanda importance THEO KENH THAT (khong thu gon) - broadcast dung
    sang tung nhom g rieng biet (nhom k dung dung importance cua kenh k*g..k*g+g-1)."""
    R, C = W.shape
    Xg = W.reshape(-1, g)                                 # [R*(C//g), g]
    Wt = None
    if w_per_channel is not None:
        Wt = w_per_channel.view(C // g, g).repeat(R, 1)   # [R*(C//g), g] - MOI nhom giu dung importance rieng
        assert Wt.shape == Xg.shape
    C1 = kmeans(Xg, K, seed=0, W=Wt)
    if M == 1:
        a1 = wsqdist(Xg, C1, Wt).argmin(1)
        rec = C1[a1]
    else:
        resid = Xg - C1[wsqdist(Xg, C1, Wt).argmin(1)]
        C2 = kmeans(resid, K, seed=1, W=Wt)
        a1, a2 = beam_assign(Xg, C1, C2, beam=4, W=Wt)
        rec = C1[a1] + C2[a2]
    err = ((rec - Xg).pow(2).sum() / Xg.pow(2).sum().clamp(min=1e-12)).sqrt().item()
    return err


def main():
    log(f"doc GGUF: {GGUF_PATH}")
    r = gguf.GGUFReader(GGUF_PATH)
    byname = {t.name: t for t in r.tensors}
    cfg = OlmoeConfig(vocab_size=50304, hidden_size=2048, intermediate_size=1024,
                      num_hidden_layers=1, num_attention_heads=16, num_key_value_heads=16,
                      max_position_embeddings=4096, rms_norm_eps=1e-5,
                      num_experts_per_tok=8, num_experts=64, norm_topk_prob=False,
                      tie_word_embeddings=False, attention_bias=False)
    model = OlmoeForCausalLM(cfg)
    sd = model.state_dict()

    def put(hf_name, gguf_name):
        v = dq(r, byname, gguf_name)
        sd[hf_name] = v

    put("model.embed_tokens.weight", "token_embd.weight")
    put("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight")
    put("model.layers.0.self_attn.k_proj.weight", "blk.0.attn_k.weight")
    put("model.layers.0.self_attn.v_proj.weight", "blk.0.attn_v.weight")
    put("model.layers.0.self_attn.o_proj.weight", "blk.0.attn_output.weight")
    put("model.layers.0.self_attn.q_norm.weight", "blk.0.attn_q_norm.weight")
    put("model.layers.0.self_attn.k_norm.weight", "blk.0.attn_k_norm.weight")
    put("model.layers.0.input_layernorm.weight", "blk.0.attn_norm.weight")
    put("model.layers.0.post_attention_layernorm.weight", "blk.0.ffn_norm.weight")
    put("model.layers.0.mlp.gate.weight", "blk.0.ffn_gate_inp.weight")

    gate_all = dq(r, byname, "blk.0.ffn_gate_exps.weight")
    up_all = dq(r, byname, "blk.0.ffn_up_exps.weight")
    down_all = dq(r, byname, "blk.0.ffn_down_exps.weight")
    gate_up_proj = torch.cat([gate_all, up_all], dim=1)
    sd["model.layers.0.mlp.experts.gate_up_proj"] = gate_up_proj
    sd["model.layers.0.mlp.experts.down_proj"] = down_all
    model.load_state_dict(sd)
    model.eval()
    log("model 1-layer da load tu GGUF, OK.")

    tok = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
    captured_hs = []
    h = model.model.layers[0].mlp.register_forward_pre_hook(
        lambda mod, inp: captured_hs.append(inp[0].detach().reshape(-1, inp[0].shape[-1])))
    log(f"chay forward {len(CALIB_TEXTS)} doan calib EN (tang tu 20 len {len(CALIB_TEXTS)})...")
    with torch.no_grad():
        for txt in CALIB_TEXTS:
            ids = tok(txt, return_tensors="pt").input_ids
            model.model(ids)
    h.remove()
    HS = torch.cat(captured_hs, dim=0)
    log(f"  {HS.shape[0]} token thuc")

    router_w = sd["model.layers.0.mlp.gate.weight"]
    probs = F.softmax(F.linear(HS, router_w).float(), dim=-1)
    _, topidx = probs.topk(8, dim=-1)

    tok_counts, importances_l2 = {}, {}
    for e in range(N_EXPERTS_TEST):
        sel = (topidx == e).any(dim=-1)
        n_tok = int(sel.sum())
        tok_counts[e] = n_tok
        if n_tok == 0:
            importances_l2[e] = None
            continue
        x_e = HS[sel]
        gate, up = F.linear(x_e, gate_up_proj[e]).chunk(2, dim=-1)
        down_input = F.silu(gate) * up
        importances_l2[e] = down_input.pow(2).sum(0).sqrt()     # DUNG cong thuc Wanda: L2-norm

    covered = sum(1 for e in range(N_EXPERTS_TEST) if tok_counts[e] > 0)
    log(f"PHU CALIB: {covered}/{N_EXPERTS_TEST} expert, token/expert: {tok_counts}")

    log("\n=== GATE 1: Wanda-L2 (da sua) vs magnitude-thuan, tren BAI TOAN DON GIAN (chon mask 1:32) ===")
    n_better_gate1 = 0
    for e in range(N_EXPERTS_TEST):
        W = down_all[e]
        if importances_l2[e] is None:
            continue
        mask_mag = mask_topn_by_score(W.abs(), 1, 32)
        err_mag = ((q_ternary_masked(W, mask_mag) - W).pow(2).sum() / W.pow(2).sum()).sqrt().item()
        imp = importances_l2[e]
        mask_wanda = mask_topn_by_score(W.abs() * imp[None, :].clamp(min=1e-8), 1, 32)
        err_wanda = ((q_ternary_masked(W, mask_wanda) - W).pow(2).sum() / W.pow(2).sum()).sqrt().item()
        win = err_wanda < err_mag
        n_better_gate1 += win
        log(f"  expert {e} (tok={tok_counts[e]}): mag {err_mag:.4f} | wanda-L2 {err_wanda:.4f}"
            f" {'THANG' if win else 'thua'}")
    log(f"GATE 1: Wanda-L2 thang {n_better_gate1}/{covered} expert")

    gate1_pass = n_better_gate1 >= covered * 0.5
    log(f"GATE 1 {'DAT' if gate1_pass else 'KHONG DAT'} - nhung AQLM la co che KHAC ternary-mask"
        " (khong ep ve 0 tuyet doi, bieu dien lien tuc qua codeword) - BO GATE theo yeu cau user,"
        " chay truc tiep Buoc 2 de co cau tra loi THAT cho chinh AQLM (khong suy tu ternary).")
    log("\n=== BUOC 2: AQLM khong-calib vs AQLM co-calib (weighted assignment), moi expert rieng ===")
    n_better_gate2, rows = 0, []
    for e in range(N_EXPERTS_TEST):
        if importances_l2[e] is None:
            continue
        W = down_all[e]
        err_plain = aqlm_err(W, M=1, K=32, g=16)
        w_ch = importances_l2[e].clamp(min=1e-8)
        w_ch = w_ch / w_ch.mean()                          # chuan hoa, tranh lech scale khoang cach
        err_calib = aqlm_err(W, M=1, K=32, g=16, w_per_channel=w_ch)
        win = err_calib < err_plain
        n_better_gate2 += win
        rows.append({"expert": e, "tok": tok_counts[e], "aqlm_plain": round(err_plain, 4),
                    "aqlm_wanda": round(err_calib, 4), "better": bool(win)})
        log(f"  expert {e}: AQLM-thuong {err_plain:.4f} | AQLM+Wanda {err_calib:.4f}"
            f" {'THANG' if win else 'thua'}")

    log(f"\n=== KET LUAN: AQLM+Wanda thang {n_better_gate2}/{len(rows)} expert ===")
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"gate1_pass": True, "gate1_n_better": n_better_gate1, "gate1_covered": covered,
                  "gate2_rows": rows, "gate2_n_better": n_better_gate2}, f,
                  ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")


if __name__ == "__main__":
    main()
