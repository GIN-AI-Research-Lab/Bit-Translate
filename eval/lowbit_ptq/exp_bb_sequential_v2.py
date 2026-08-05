# -*- coding: utf-8 -*-
"""
exp_at v2: sequential reconstruction voi RECIPE DUNG theo lab (exp_ab/exp_r s1_sequential).
exp_at that bai vi bo mat 3 ingredient lab da chung minh bat buoc (Bai 4/10):
  1. Scale HOC DUOC (raw_s qua softplus, grad chay qua t*s) - khong phai absmean co dinh.
  2. 2 PASS qua block (pass 2 toi uu lai voi ngu canh da sua cua pass 1).
  3. Toi uu THEM norm weights (input/post-attn layernorm) cung block.
+ track-best (da co).

Muc tieu: xac nhan sequential CUU duoc collapse cua naive ternary trong khung do cua ta
(lab da dat 594 @1.94bpw tren Qwen3-0.6B voi recipe day du - o day tai lap).

Chay: python eval/lowbit_ptq/exp_au_sequential_v2.py
"""
import gc
import io
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = ("F:/Project Ai/Quantazation/hf_cache/hub/models--Qwen--Qwen3-0.6B/"
             "snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_bb_results.json")
LIN_NAMES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
PASSES = 2
STEPS = 35

CALIB = [
    "The city council approved funding for a new transit line connecting downtown to the airport.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen for plants.",
    "def merge_sort(a):\n    if len(a) <= 1: return a\n    m = len(a)//2\n    return merge(merge_sort(a[:m]), merge_sort(a[m:]))",
    "The central bank signaled it would keep interest rates elevated to combat persistent inflation.",
    "Hôm nay trời mưa to nên tôi phải mang theo ô khi đi làm và đường phố rất đông đúc.",
    "今日は天気がとてもいいので、家族と一緒に公園へ散歩に行くことにしました。",
    "Regular physical exercise combined with proper sleep reduces the risk of chronic disease.",
    "Quantum computers exploit superposition and entanglement to process many states in parallel.",
    "Việc học một ngôn ngữ mới đòi hỏi sự kiên nhẫn, luyện tập đều đặn và tiếp xúc thường xuyên.",
    "The ancient library preserved thousands of manuscripts documenting the history of the region.",
    "科学者たちは、この地域で新しい種類の昆虫を発見したと学術誌で報告しました。",
    "Rising global temperatures are accelerating the melting of polar ice and glaciers worldwide.",
    "A skilled chef balances salt, acid, fat, and heat to bring out the flavor in every dish.",
    "Máy tính lượng tử hứa hẹn giải quyết một số bài toán mà máy tính thông thường bó tay.",
    "The committee reviewed the proposal carefully before recommending it for final approval.",
    "Migratory birds travel thousands of kilometers each year following seasonal food sources.",
]
EVAL = [
    "The two nations signed a trade agreement that will phase out tariffs over five years.",
    "def binary_search(a, t):\n    lo, hi = 0, len(a)-1\n    while lo <= hi:\n        m=(lo+hi)//2\n        if a[m]==t: return m\n        elif a[m]<t: lo=m+1\n        else: hi=m-1\n    return -1",
    "Chính phủ công bố kế hoạch đầu tư lớn cho hệ thống giao thông công cộng trong thập kỷ tới.",
    "この新しい技術は医療分野での幅広い応用が期待されており、多くの研究が進行中です。",
    "Scientists observed unusual migration patterns among several bird species this year.",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def inv_softplus(y):
    return torch.log(torch.expm1(y.clamp(min=1e-6)))


def naive_ternary_24(W, sgroup=64):
    R, C = W.shape
    Wg4 = W.view(R, -1, 4)
    idx = Wg4.abs().topk(2, dim=2).indices
    mask = torch.zeros_like(Wg4).scatter_(2, idx, 1.0).view(R, C)
    Wv, Mv = W.view(R, -1, sgroup), mask.view(R, -1, sgroup)
    Wm = Wv * Mv
    cnt = Mv.sum(2, keepdim=True).clamp(min=1)
    s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-8)
    for _ in range(3):
        t = torch.round(Wm / s).clamp(-1, 1) * Mv
        num, den = (Wm * t).sum(2, keepdim=True), (t * t).sum(2, keepdim=True).clamp(min=1e-8)
        s = (num / den).abs().clamp(min=1e-8)
    return (torch.round(Wm / s.clamp(min=1e-8)).clamp(-1, 1) * s * Mv).view(R, C)


class TernQLinear(nn.Module):
    """Mask 2:4 co dinh, Wfp master, raw_s HOC DUOC (softplus), STE. Theo NestedQLinear cua lab."""
    def __init__(self, lin, G=64):
        super().__init__()
        W0 = lin.weight.data.float()
        R, C = W0.shape
        self.R, self.C, self.G = R, C, G
        Wg4 = W0.view(R, -1, 4)
        idx = Wg4.abs().topk(2, dim=2).indices
        mask = torch.zeros_like(Wg4).scatter_(2, idx, 1.0).view(R, C)
        self.register_buffer("mask", mask)
        self.Wfp = nn.Parameter(W0.clone())
        Wv, Mv = (W0 * mask).view(R, -1, G), mask.view(R, -1, G)
        cnt = Mv.sum(2, keepdim=True).clamp(min=1)
        s = (Wv.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-8)
        for _ in range(3):
            t = torch.round(Wv / s).clamp(-1, 1) * Mv
            num, den = (Wv * t).sum(2, keepdim=True), (t * t).sum(2, keepdim=True).clamp(min=1e-8)
            s = (num / den).abs().clamp(min=1e-8)
        self.raw_s = nn.Parameter(inv_softplus(s))       # [R, C//G, 1] hoc duoc
        self.bias = None

    def quant(self):
        s = F.softplus(self.raw_s).clamp(min=1e-6)         # grad chay qua t*s
        Wv = (self.Wfp * self.mask).view(self.R, -1, self.G)
        Mv = self.mask.view(self.R, -1, self.G)
        t = (torch.round(Wv / s.detach()).clamp(-1, 1) * Mv).detach()
        return (t * s).view(self.R, self.C)

    def forward(self, x):
        q = self.quant()
        Wq = q + (self.Wfp - self.Wfp.detach()) * self.mask     # masked-STE cho W
        return F.linear(x, Wq.to(x.dtype), None)

    @torch.no_grad()
    def bake(self):
        return self.quant().detach()


@torch.no_grad()
def ppl(model, tok, texts):
    losses = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids
        lg = model(ids).logits[0, :-1].float()
        losses.append(F.cross_entropy(lg, ids[0, 1:]).item())
    return math.exp(sum(losses) / len(losses))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    base = ppl(model, tok, EVAL)
    log(f"baseline FP: {base:.3f}")

    layers = model.model.layers
    NB = len(layers)
    wrapped = {}
    for L in range(NB):
        for nm in LIN_NAMES:
            parent = layers[L]
            *path, last = nm.split(".")
            for p in path:
                parent = getattr(parent, p)
            w = TernQLinear(getattr(parent, last))
            setattr(parent, last, w)
            wrapped[(L, nm)] = w

    # precompute quy dao FP (dung Wfp thuan, khong ternary)
    log("precompute quy dao FP...")
    calib_ids = [tok(t, return_tensors="pt").input_ids for t in CALIB]
    H_in0, H_out = [], [[] for _ in range(NB)]
    ropes = []
    orig_fwd = TernQLinear.forward
    with torch.no_grad():
        def fp_forward(self, x):
            return F.linear(x, self.Wfp.to(x.dtype), None)
        TernQLinear.forward = fp_forward
        for ids in calib_ids:
            h = model.model.embed_tokens(ids)
            H_in0.append(h.clone())
            pos = torch.arange(ids.shape[1])[None]
            cos, sin = model.model.rotary_emb(h, pos)
            ropes.append((cos, sin))
            for b, blk in enumerate(layers):
                o = blk(h, position_embeddings=(cos, sin))
                h = o[0] if isinstance(o, tuple) else o
                H_out[b].append(h.clone())
        TernQLinear.forward = orig_fwd

    NC = len(calib_ids)
    log(f"sequential {PASSES} pass x {STEPS} step/block, scale HOC DUOC + norm co-tune...")
    for p_idx in range(PASSES):
        H_q = [H_in0[s].detach().clone() for s in range(NC)]
        for b, blk in enumerate(layers):
            mods = [wrapped[(b, nm)] for nm in LIN_NAMES]
            norms = [layers[b].input_layernorm.weight, layers[b].post_attention_layernorm.weight]
            for nw in norms:
                nw.requires_grad_(True)
            opt = torch.optim.Adam([
                {"params": [m.Wfp for m in mods], "lr": 1e-3},
                {"params": [m.raw_s for m in mods], "lr": 5e-3},
                {"params": norms, "lr": 5e-4}])

            def blk_out(s):
                o = blk(H_q[s], position_embeddings=ropes[s])
                return o[0] if isinstance(o, tuple) else o

            def full_eval():
                with torch.no_grad():
                    return sum(F.mse_loss(blk_out(s), H_out[b][s]).item() for s in range(NC))

            best = full_eval()
            best_state = ([m.Wfp.detach().clone() for m in mods],
                          [m.raw_s.detach().clone() for m in mods],
                          [nw.detach().clone() for nw in norms])
            for step in range(STEPS):
                idx = torch.randperm(NC)[:6].tolist()
                opt.zero_grad()
                loss = sum(F.mse_loss(blk_out(s), H_out[b][s]) for s in idx) / len(idx)
                loss.backward()
                opt.step()
                if step % 10 == 9 or step == STEPS - 1:
                    v = full_eval()
                    if v < best:
                        best = v
                        best_state = ([m.Wfp.detach().clone() for m in mods],
                                      [m.raw_s.detach().clone() for m in mods],
                                      [nw.detach().clone() for nw in norms])
            with torch.no_grad():
                for i, m in enumerate(mods):
                    m.Wfp.data = best_state[0][i]
                    m.raw_s.data = best_state[1][i]
                for nw, nb in zip(norms, best_state[2]):
                    nw.data = nb
            for nw in norms:
                nw.requires_grad_(False)
            with torch.no_grad():
                for s in range(NC):
                    H_q[s] = blk_out(s).detach()
            if b % 7 == 0:
                log(f"  pass{p_idx + 1} block {b}/{NB}: mse {best:.3f}")

    # bake
    log("bake + eval...")
    with torch.no_grad():
        for L in range(NB):
            for nm in LIN_NAMES:
                parent = layers[L]
                *path, last = nm.split(".")
                for p in path:
                    parent = getattr(parent, p)
                w = getattr(parent, last)
                nl = nn.Linear(w.C, w.R, bias=False)
                nl.weight.data = w.bake().float()
                setattr(parent, last, nl)
    seq = ppl(model, tok, EVAL)
    log(f"SEQUENTIAL v2 PPL: {seq:.3f} (x{seq/base:.2f})")

    naive = 2178033.809     # tu exp_at (cung model, cung EVAL, cung ternary 2:4)
    out = {"baseline": round(base, 3), "naive_all": naive, "sequential_v2_all": round(seq, 3),
           "x_baseline": {"naive": round(naive / base, 1), "seq_v2": round(seq / base, 2)}}
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")

    log("\n=== KET LUAN ===")
    log(f"  baseline FP:        {base:.1f}")
    log(f"  naive ternary all:  {naive:.0f}  (x{naive/base:.0f})")
    log(f"  sequential v2 all:  {seq:.1f}  (x{seq/base:.1f})")
    if seq < naive / 100:
        log(f"  => SEQUENTIAL v2 CUU duoc: {naive/seq:.0f}x tot hon naive. Co che + recipe xac nhan.")
    elif seq < naive:
        log(f"  => giup {naive/seq:.0f}x nhung chua ve vung dung duoc - can them pass/calib/gauge.")
    else:
        log("  => VAN khong cuu - bug sau hon, nen tai dung exp_r truc tiep.")


if __name__ == "__main__":
    main()
