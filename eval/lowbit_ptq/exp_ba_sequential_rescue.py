# -*- coding: utf-8 -*-
"""
Kiem chung gia thuyet: SEQUENTIAL reconstruction (BRECQ-lite, moi block bu sai so block truoc)
CUU duoc collapse cua NAIVE ternary khi nen day du (gate/up/down/attn). exp_as vua cho thay
nen naive gate_up -> sup x1700. Lab tung dat PPL 594 @1.94bpw voi sequential tren Qwen3-0.6B
(exp_k) - o day tai lap SACH trong khung do cua ta, A/B truc tiep:
  - NAIVE: quantize tat ca linear ternary 2:4, khong recon -> ky vong sup
  - SEQUENTIAL: cung ternary 2:4 nhung toi uu master-weight tung block khop quy dao FP
    (input da luong tu cua prefix) -> ky vong cuu

Chay tren Qwen3-0.6B DENSE (nho, an toan bo nho - OLMoE 7B qua rui ro cho gradient loop).
Neu cuu duoc o day -> co che + code dung -> moi port sang OLMoE MoE (viec nang hon).

Chay: python eval/lowbit_ptq/exp_at_sequential_rescue.py
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
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_ba_results.json")
LIN_NAMES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

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


def ternary_24(W, sgroup=64):
    """Ternary 2:4 (mask top-2/4 theo |W|) + scale Lloyd per-group-64. Tra ban da luong tu (fp)."""
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
    return (torch.round(Wm / s.clamp(min=1e-8)).clamp(-1, 1) * s * Mv).view(R, C), mask


class TernQLinear(nn.Module):
    """Master fp32 Wfp + mask 2:4 CO DINH (tu |W| ban dau). Forward STE. Toi uu Wfp."""
    def __init__(self, lin):
        super().__init__()
        W0 = lin.weight.data.float()
        self.R, self.C = W0.shape
        _, mask = ternary_24(W0)
        self.register_buffer("mask", mask)
        self.Wfp = nn.Parameter(W0.clone())
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data.float().clone())

    def _quant(self):
        R, C, sg = self.R, self.C, 64
        Wm = (self.Wfp * self.mask).view(R, -1, sg)
        Mv = self.mask.view(R, -1, sg)
        cnt = Mv.sum(2, keepdim=True).clamp(min=1)
        s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-8).detach()
        t = (torch.round(Wm / s).clamp(-1, 1) * Mv)
        return (t * s).view(R, C)

    def forward(self, x):
        q = self._quant()
        Wq = q.detach() + (self.Wfp - self.Wfp.detach())      # STE
        Wq = Wq * self.mask                                    # grad chi qua o giu (masked-STE)
        return F.linear(x, Wq.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

    @torch.no_grad()
    def bake(self):
        return self._quant().detach()


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
    log("=== baseline (FP) ===")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    base_ppl = ppl(model, tok, EVAL)
    log(f"  baseline PPL {base_ppl:.3f}")
    results = {"baseline": round(base_ppl, 3)}

    # ---------- NAIVE: quantize tat ca linear, khong recon ----------
    log("=== NAIVE: ternary 2:4 tat ca linear (khong recon) ===")
    with torch.no_grad():
        for L in range(model.config.num_hidden_layers):
            layer = model.model.layers[L]
            for nm in LIN_NAMES:
                obj = layer
                for p in nm.split("."):
                    obj = getattr(obj, p)
                q, _ = ternary_24(obj.weight.data.float())
                obj.weight.data.copy_(q.to(obj.weight.dtype))
    naive_ppl = ppl(model, tok, EVAL)
    log(f"  NAIVE PPL {naive_ppl:.3f}  (x{naive_ppl/base_ppl:.1f} baseline)")
    results["naive_all"] = round(naive_ppl, 3)
    del model
    gc.collect()

    # ---------- SEQUENTIAL: wrap + block-wise recon ----------
    log("=== SEQUENTIAL: ternary 2:4 + block-wise reconstruction ===")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    layers = model.model.layers
    NB = len(layers)

    # wrap tat ca linear
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

    # precompute quy dao FP (dung MASTER Wfp = FP goc, chua ep ternary) qua manual layer loop
    log("  precompute quy dao FP...")
    calib_ids = [tok(t, return_tensors="pt").input_ids for t in CALIB]
    H_in = [[] for _ in range(NB)]          # input moi block (quy dao FP)
    H_out = [[] for _ in range(NB)]         # output FP moi block
    ropes = []
    with torch.no_grad():
        # de lay quy dao FP that: tam thoi cho cac wrapper tra ve FP (dung Wfp truc tiep)
        for w in wrapped.values():
            w._ste_off = True
        # monkeypatch forward tam: dung Wfp thuan (khong ternary) de co quy dao FP dung
        def fp_forward(self, x):
            return F.linear(x, self.Wfp.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        orig_fwd = TernQLinear.forward
        TernQLinear.forward = fp_forward
        for ids in calib_ids:
            h = model.model.embed_tokens(ids)
            pos = torch.arange(ids.shape[1])[None]
            cos, sin = model.model.rotary_emb(h, pos)
            ropes.append((cos, sin))
            for b, blk in enumerate(layers):
                H_in[b].append(h)
                out = blk(h, position_embeddings=(cos, sin))
                h = out[0] if isinstance(out, tuple) else out
                H_out[b].append(h)
        TernQLinear.forward = orig_fwd       # tra lai STE

    # block-wise sequential: H_q bat dau = input FP layer 0; moi block toi uu khop H_out FP
    log("  toi uu tung block (input da luong tu, khop quy dao FP)...")
    NC = len(calib_ids)
    H_q = [H_in[0][s].detach().clone() for s in range(NC)]
    STEPS = 60
    for b, blk in enumerate(layers):
        mods = [wrapped[(b, nm)] for nm in LIN_NAMES]
        params = [m.Wfp for m in mods] + [m.bias for m in mods if m.bias is not None]
        opt = torch.optim.Adam(params, lr=1e-3)

        def blk_out(s):
            return blk(H_q[s], position_embeddings=ropes[s])

        best = float("inf")
        best_state = None
        for step in range(STEPS):
            idx = torch.randperm(NC)[:6].tolist()
            opt.zero_grad()
            loss = 0.0
            for s in idx:
                out = blk_out(s)
                out = out[0] if isinstance(out, tuple) else out
                loss = loss + F.mse_loss(out, H_out[b][s])
            loss = loss / len(idx)
            loss.backward()
            opt.step()
            if step % 15 == 14 or step == STEPS - 1:
                with torch.no_grad():
                    v = sum(F.mse_loss((lambda o: o[0] if isinstance(o, tuple) else o)(blk_out(s)),
                                       H_out[b][s]).item() for s in range(NC))
                if v < best:
                    best = v
                    best_state = ([m.Wfp.detach().clone() for m in mods],
                                  [m.bias.detach().clone() for m in mods if m.bias is not None])
        # nap best
        with torch.no_grad():
            bi = 0
            for m in mods:
                m.Wfp.data = best_state[0][mods.index(m)]
                if m.bias is not None:
                    m.bias.data = best_state[1][bi]
                    bi += 1
            # cap nhat H_q = output da luong tu cua block nay
            for s in range(NC):
                o = blk(H_q[s], position_embeddings=ropes[s])
                H_q[s] = (o[0] if isinstance(o, tuple) else o).detach()
        if b % 6 == 0:
            log(f"    block {b}/{NB} xong (mse {best:.4f})")

    # bake: thay wrapper bang Linear ternary tinh
    log("  bake + eval...")
    with torch.no_grad():
        for L in range(NB):
            for nm in LIN_NAMES:
                parent = layers[L]
                *path, last = nm.split(".")
                for p in path:
                    parent = getattr(parent, p)
                w = getattr(parent, last)
                nl = nn.Linear(w.C, w.R, bias=w.bias is not None)
                nl.weight.data = w.bake().float()
                if w.bias is not None:
                    nl.bias.data = w.bias.detach().float()
                setattr(parent, last, nl)
    seq_ppl = ppl(model, tok, EVAL)
    log(f"  SEQUENTIAL PPL {seq_ppl:.3f}  (x{seq_ppl/base_ppl:.1f} baseline)")
    results["sequential_all"] = round(seq_ppl, 3)

    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"model": "Qwen3-0.6B", "note": "naive vs sequential, ternary 2:4 TAT CA linear",
                  "ppl": results, "x_baseline": {k: round(v / base_ppl, 2) for k, v in results.items()}},
                  f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")

    log("\n=== KET LUAN ===")
    log(f"  baseline FP:        {base_ppl:.1f}")
    log(f"  naive ternary all:  {naive_ppl:.1f}  (x{naive_ppl/base_ppl:.0f})")
    log(f"  sequential all:     {seq_ppl:.1f}  (x{seq_ppl/base_ppl:.1f})")
    if seq_ppl < naive_ppl / 10:
        log(f"  => SEQUENTIAL CUU duoc: giam sup {naive_ppl/seq_ppl:.0f}x so naive. Co che xac nhan.")
    elif seq_ppl < naive_ppl:
        log(f"  => sequential co giup ({naive_ppl/seq_ppl:.1f}x) nhung chua du manh - can tinh chinh.")
    else:
        log("  => sequential KHONG cuu - nghi code sai (lab da chung minh no PHAI cuu tren 0.6B).")


if __name__ == "__main__":
    main()
