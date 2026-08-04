# -*- coding: utf-8 -*-
"""
Wanda-style calibration cho moc 0.3bpw (yeu cau user: "cach cuu voi calib de dat 0.3bit").
Khac AQLM Buoc 0/1/2 (thuan trong so): dung dequant Q4_K_M GGUF + transformers.OlmoeForCausalLM
(model 1-layer, chi layer 0) de chay forward pass THAT qua embedding+attention, hook INPUT
cua ca block MoE (khong hook trong vong lap tung expert - OlmoeExperts.forward moi ban
transformers nay gop gate+up thanh 1 Parameter 3D, khong con submodule rieng tung expert -
xem source modeling_olmoe.py doc truc tiep 04/08), roi TU TINH LAI dung cong thuc goc
(gate,up = chunk(x @ gate_up_proj[e].T, 2); down_input = silu(gate)*up) cho 16 expert dang
test - dam bao dung 100% cong thuc thuc, khong doan.

Calib text: van xuoi tieng Anh da chu de (OLMoE la model EN). Kiem tra PHU expert truoc khi
tin bat ky so Wanda nao.

Chay: python eval/lowbit_ptq/exp_ag_wanda_calib.py
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
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_ag_results.json")

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
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def dq(reader, byname, name):
    t = byname[name]
    arr = gguf.quants.dequantize(t.data, t.tensor_type)
    return torch.from_numpy(arr.copy()).float()


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
        assert sd[hf_name].shape == v.shape, f"{hf_name}: {sd[hf_name].shape} != {v.shape}"
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

    log("dequant expert tensors (gate/up/down, ALL 64 expert)...")
    gate_all = dq(r, byname, "blk.0.ffn_gate_exps.weight")     # (64, 1024, 2048) = (e,out=ffn,in=hid)
    up_all = dq(r, byname, "blk.0.ffn_up_exps.weight")         # (64, 1024, 2048)
    down_all = dq(r, byname, "blk.0.ffn_down_exps.weight")     # (64, 2048, 1024) = (e,out=hid,in=ffn)
    log(f"  gate {tuple(gate_all.shape)} up {tuple(up_all.shape)} down {tuple(down_all.shape)}")

    # dung DUNG cong thuc doc tu modeling_olmoe.py (khong doan):
    # gate,up = F.linear(x, gate_up_proj[e]).chunk(2,-1) -> gate_up_proj[e] = cat([gate,up], dim=0)
    gate_up_proj = torch.cat([gate_all, up_all], dim=1)         # (64, 2048, 2048)
    down_proj = down_all                                       # (64, 2048, 1024) - khop thang
    assert sd["model.layers.0.mlp.experts.gate_up_proj"].shape == gate_up_proj.shape
    assert sd["model.layers.0.mlp.experts.down_proj"].shape == down_proj.shape
    sd["model.layers.0.mlp.experts.gate_up_proj"] = gate_up_proj
    sd["model.layers.0.mlp.experts.down_proj"] = down_proj
    model.load_state_dict(sd)
    model.eval()
    log("load_state_dict OK - moi shape khop dung cong thuc source, khong doan.")

    log("tai tokenizer OLMoE...")
    tok = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

    # hook INPUT cua ca block MoE (= output cua post_attention_layernorm, DUNG THU input
    # router+experts thay nhan theo - xem OlmoeDecoderLayer.forward)
    captured_hs = []
    h = model.model.layers[0].mlp.register_forward_pre_hook(
        lambda mod, inp: captured_hs.append(inp[0].detach().reshape(-1, inp[0].shape[-1])))

    log(f"chay forward THAT {len(CALIB_TEXTS)} doan calib EN qua embedding+attention layer 0...")
    with torch.no_grad():
        for txt in CALIB_TEXTS:
            ids = tok(txt, return_tensors="pt").input_ids
            model.model(ids)
    h.remove()
    HS = torch.cat(captured_hs, dim=0)          # [n_token_total, 2048] - dung 100% hidden state thuc
    log(f"  thu duoc {HS.shape[0]} token hidden-state thuc")

    # tu tinh lai router + expert dispatch (dung cong thuc doc tu source, KHONG hook trong loop)
    router_w = sd["model.layers.0.mlp.gate.weight"]         # (64,2048)
    logits = F.linear(HS, router_w)
    probs = F.softmax(logits.float(), dim=-1)
    topval, topidx = probs.topk(8, dim=-1)                  # (n_token, 8), norm_topk_prob=False

    tok_counts, importances = {}, {}
    for e in range(N_EXPERTS_TEST):
        sel = (topidx == e).any(dim=-1)                     # token nao co expert e trong top-8
        n_tok = int(sel.sum())
        tok_counts[e] = n_tok
        if n_tok == 0:
            importances[e] = None
            continue
        x_e = HS[sel]                                        # [n_tok, 2048] input THAT cua expert e
        gate, up = F.linear(x_e, gate_up_proj[e]).chunk(2, dim=-1)
        down_input = F.silu(gate) * up                       # [n_tok, 1024] - DUNG input down_proj
        importances[e] = down_input.abs().mean(0)             # Wanda-style ||x_j|| xap xi bang mean|.|

    covered = sum(1 for e in range(N_EXPERTS_TEST) if tok_counts[e] > 0)
    starved = [e for e in range(N_EXPERTS_TEST) if tok_counts[e] == 0]
    log(f"PHU CALIB: {covered}/{N_EXPERTS_TEST} expert duoc cham >=1 token thuc. "
        f"So token/expert: {tok_counts}")
    if starved:
        log(f"  DOI-EXPERT: {starved} khong duoc cham - Wanda cho expert nay se fallback |W| thuan "
            f"(dung bai hoc Bai 16: mask khong co tin hieu that = nhieu hon co ich)")

    # ---- so sanh nhanh: Wanda-weighted vs |W|-thuan tren down_proj cua expert co du calib ----
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

    results = {}
    for e in range(N_EXPERTS_TEST):
        W = down_proj[e]                        # [2048,1024], import[e] neu co la [1024] theo IN-dim
        R, C = W.shape
        n, m = 1, 32                             # moc thap nhat, dung cau hoi user
        Wg = W.view(R, -1, m)
        mag_idx = Wg.abs().topk(n, dim=2).indices
        mask_mag = torch.zeros_like(Wg).scatter_(2, mag_idx, 1.0).view(R, C)
        Wt_mag = q_ternary_masked(W, mask_mag)
        err_mag = ((Wt_mag - W).pow(2).sum() / W.pow(2).sum()).sqrt().item()

        row = {"tok_count": tok_counts[e], "err_magnitude_only": round(err_mag, 4)}
        if importances[e] is not None:
            imp = importances[e]                 # [1024] theo IN-dim cua down_proj
            Wimp = W.abs() * imp[None, :].clamp(min=1e-8)     # Wanda: |W|*||x_in||
            Wimpg = Wimp.view(R, -1, m)
            wanda_idx = Wimpg.topk(n, dim=2).indices
            mask_wanda = torch.zeros_like(Wimpg).scatter_(2, wanda_idx, 1.0).view(R, C)
            Wt_wanda = q_ternary_masked(W, mask_wanda)
            err_wanda = ((Wt_wanda - W).pow(2).sum() / W.pow(2).sum()).sqrt().item()
            row["err_wanda_calib"] = round(err_wanda, 4)
            row["wanda_better"] = bool(err_wanda < err_mag)
        results[e] = row
        log(f"  expert {e} (tok={tok_counts[e]}): |W|-thuan {err_mag:.4f}"
            + (f" | Wanda-calib {row.get('err_wanda_calib', '-')}"
               f" ({'TOT HON' if row.get('wanda_better') else 'KHONG tot hon'})"
              if importances[e] is not None else " | (doi-expert, khong co Wanda)"))

    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"tok_counts": tok_counts, "starved": starved, "per_expert": results}, f,
                  ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")

    n_better = sum(1 for v in results.values() if v.get("wanda_better"))
    n_cmp = sum(1 for v in results.values() if "wanda_better" in v)
    log(f"\n=== TOM TAT: Wanda-calib tot hon |W|-thuan o {n_better}/{n_cmp} expert co du calib ===")


if __name__ == "__main__":
    main()
