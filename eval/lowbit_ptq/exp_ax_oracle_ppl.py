# -*- coding: utf-8 -*-
"""
Oracle-PPL cho gia thuyet 2B (contextual sparsity) tren Qwen3-0.6B dense. exp_ap da do TIEN
DE (nang luong tap trung 23% neuron/95%, contextual manh). O day dong vong TRUNG THUC: gia
lap "bo du doan HOAN HAO" - moi token, moi FFN layer, chi giu top-k% neuron trung gian (theo
|activation|, dung tin hieu exp_ap do), ZERO phan con lai, chay FULL 28 layer, do PPL that
so baseline. Neu oracle hoan hao MA PPL van sup -> ca huong chet (khoi xay predictor). Neu
giu duoc PPL -> tran dang gia, moi dau tu predictor.

⚠️ Day la CHAN TREN LAC QUAN: predictor thuc te khong bao gio hoan hao. Va oracle nay van
"gian lan" o cho no dung |activation| THAT (da tinh gate/up roi) de chon - tiet kiem byte
thuc chi co neu predictor doan duoc tap nay MA khong can doc gate/up. Do la viec ke, khong
phai do o day. O day chi tra loi: TRAN chat luong co ton tai khong.

Chay: python eval/lowbit_ptq/exp_aq_oracle_ppl.py
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
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_ax_results.json")
KEEP_FRACS = [1.0, 0.50, 0.30, 0.23, 0.15]   # 1.0 = baseline khong mask

# HELD-OUT (khac calib exp_ap) - en/vi/ja de dai dien dung cham dich du an
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


class TopKMasker:
    """forward_pre_hook tren down_proj: giu top-frac neuron |activation| moi token, zero con lai."""
    def __init__(self):
        self.frac = 1.0

    def __call__(self, mod, inp):
        if self.frac >= 1.0:
            return None
        x = inp[0]                                  # [..., I]
        I = x.shape[-1]
        k = max(1, int(round(I * self.frac)))
        flat = x.reshape(-1, I)
        thr = flat.abs().kthvalue(I - k + 1, dim=1, keepdim=True).values   # nguong top-k
        mask = flat.abs() >= thr
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

    masker = TopKMasker()
    for L in range(NL):
        model.model.layers[L].mlp.down_proj.register_forward_pre_hook(masker)

    results = {}
    base = None
    for frac in KEEP_FRACS:
        masker.frac = frac
        ppl, losses = compute_ppl(model, tok, PPL_TEXTS)
        if base is None:
            base = ppl
        tag = "baseline" if frac >= 1.0 else f"keep{int(frac*100)}%"
        results[tag] = {"keep_frac": frac, "ppl": round(ppl, 3),
                        "x_baseline": round(ppl / base, 3)}
        log(f"  {tag:10s} (giu {frac:.0%} neuron): PPL {ppl:.3f}  (x{ppl/base:.2f} baseline)")

    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"model": "Qwen3-0.6B", "note": "oracle top-k activation mask, chan tren lac quan",
                  "results": results}, f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")

    log("\n=== DIEN GIAI ===")
    r23 = results.get("keep23%", {}).get("x_baseline", 999)
    r30 = results.get("keep30%", {}).get("x_baseline", 999)
    if r23 <= 1.15:
        log(f"  Giu 23% neuron: PPL chi x{r23} -> TRAN THAT SU giu chat luong o muc thua exp_ap do")
        log("  => contextual sparsity DANG dau tu predictor (tran chat luong da xac nhan bang PPL that)")
    elif r30 <= 1.2:
        log(f"  Giu 23% sup (x{r23}) nhung giu 30% x{r30} -> tran o ~30%, thap hon exp_ap goi y")
    else:
        log(f"  Giu 23% x{r23}, 30% x{r30} -> PPL sup du oracle hoan hao: nang luong-proxy DANH LUA "
            "(dung bai hoc AQLM) - cat neuron pha lan truyen qua 28 layer. Huong nay YEU hon exp_ap tuong.")


if __name__ == "__main__":
    main()
