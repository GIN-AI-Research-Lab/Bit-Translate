# -*- coding: utf-8 -*-
"""
Refinement cua exp_aq: oracle top-k nhung xep hang theo IMPORTANCE DUNG cho output =
|activation_j| * ||down_proj[:,j]|| (dong gop that vao y = down @ intermediate), thay vi chi
|activation_j| nhu exp_aq/exp_ap. Neuron activation to nhung cot down be -> dong gop nho, dang
le bo duoc; nguoc lai. Neu refinement nay cuu duoc keep-23%/30% -> phan quyet "yeu" cua exp_aq
la do SAI proxy, khong phai huong chet. Neu van ~ exp_aq -> phan quyet giu nguyen (chan tren
that su o ~keep-50%).

So SONG SONG voi exp_aq (cung KEEP_FRACS, cung PPL_TEXTS) de doi chieu truc tiep.

Chay: python eval/lowbit_ptq/exp_ar_oracle_downnorm.py
"""
import io
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = ("F:/Project Ai/Quantazation/hf_cache/hub/models--Qwen--Qwen3-0.6B/"
             "snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_ay_results.json")
KEEP_FRACS = [1.0, 0.50, 0.30, 0.23, 0.15]

PPL_TEXTS = [
    "The committee released its final report after months of deliberation and public hearings.",
    "A new bridge will connect the two districts, cutting the commute time nearly in half.",
    "Buổi hòa nhạc tối qua thu hút hàng nghìn khán giả đến từ khắp các tỉnh thành lân cận.",
    "Chính phủ vừa công bố kế hoạch đầu tư mới cho hệ thống giao thông công cộng đô thị.",
    "先週の会議では、来年度の予算配分について長time間にわたる議論が行われました。",
    "この新しい technology は、医療分野での応用が大きく期待されています。",
    "The recipe requires fresh herbs, a splash of olive oil, and a pinch of sea salt.",
    "Scientists observed unusual patterns in the migration of several bird species this year.",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class DownNormMasker:
    """importance_j = |x_j| * colnorm_j (colnorm co dinh theo layer, precompute 1 lan)."""
    def __init__(self, colnorm):
        self.colnorm = colnorm      # [I]
        self.frac = 1.0

    def __call__(self, mod, inp):
        if self.frac >= 1.0:
            return None
        x = inp[0]
        I = x.shape[-1]
        k = max(1, int(round(I * self.frac)))
        flat = x.reshape(-1, I)
        imp = flat.abs() * self.colnorm.unsqueeze(0)      # dong gop that vao output
        thr = imp.kthvalue(I - k + 1, dim=1, keepdim=True).values
        mask = imp >= thr
        return (flat.mul(mask).view_as(x),) + inp[1:]


@torch.no_grad()
def compute_ppl(model, tok, texts):
    losses = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids
        logits = model(ids).logits[0, :-1].float()
        loss = F.cross_entropy(logits, ids[0, 1:])
        losses.append(loss.item())
    return math.exp(sum(losses) / len(losses)), losses


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("tai Qwen3-0.6B dense...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    NL = model.config.num_hidden_layers

    maskers = []
    for L in range(NL):
        W = model.model.layers[L].mlp.down_proj.weight     # [hidden, I]
        colnorm = W.norm(dim=0)                              # [I] = ||cot j||
        mk = DownNormMasker(colnorm)
        model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk)
        maskers.append(mk)

    results, base = {}, None
    for frac in KEEP_FRACS:
        for mk in maskers:
            mk.frac = frac
        ppl, _ = compute_ppl(model, tok, PPL_TEXTS)
        if base is None:
            base = ppl
        tag = "baseline" if frac >= 1.0 else f"keep{int(frac*100)}%"
        results[tag] = {"keep_frac": frac, "ppl": round(ppl, 3), "x_baseline": round(ppl / base, 3)}
        log(f"  {tag:10s} (giu {frac:.0%}): PPL {ppl:.3f}  (x{ppl/base:.2f})")

    # doi chieu voi exp_aq (|x| thuan) neu co
    aq = {}
    aq_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_ax_results.json")
    if os.path.exists(aq_path):
        try:
            aq = json.load(io.open(aq_path, encoding="utf-8")).get("results", {})
        except Exception:
            aq = {}

    log("\n=== DOI CHIEU: |x|*||down|| (exp_ar) vs |x| thuan (exp_aq) ===")
    for tag in ["keep50%", "keep30%", "keep23%", "keep15%"]:
        ar_x = results.get(tag, {}).get("x_baseline")
        aq_x = aq.get(tag, {}).get("x_baseline")
        if ar_x and aq_x:
            better = "TOT HON" if ar_x < aq_x else ("~ngang" if abs(ar_x - aq_x) < 0.05 else "te hon")
            log(f"  {tag}: ar x{ar_x} vs aq x{aq_x} -> importance dung {better}")

    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"model": "Qwen3-0.6B", "importance": "|x|*||down_col||",
                  "results": results, "compare_exp_aq_absx": aq}, f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")


if __name__ == "__main__":
    main()
