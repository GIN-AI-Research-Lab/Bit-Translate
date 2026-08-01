# -*- coding: utf-8 -*-
"""
Exp N — SỬA 3 BIẾN TỰ TRÓI TAY (không phải công thức paper nào — sửa pipeline của chính ta):

  FIX-1 SCALE HỌC ĐƯỢC: exp_l chứng minh "thuế scale" thống trị (per-tensor ×15, f8/g64 ×8).
        Nguyên nhân ta tự gây: scale là HÀM Lloyd của trọng số, optimizer không có quyền đặt nó.
        Sửa: scale = tham số tự do (softplus, STE qua f16/f8) — CÙNG chi phí lưu trữ, thêm tự do.
  FIX-2 NORM ĐỒNG-TỐI-ƯU: block 2/27 nổ mse; các RMSNorm/q_norm/k_norm FP (0 bit thêm) nằm ngay
        cạnh nhưng bị đóng băng. Sửa: cho norm của block học cùng trong lúc tối ưu block.
  FIX-3 HAI PASS: sequential 1 lượt đóng đinh block sớm trước khi thấy đuôi đã lượng tử.
        Sửa: giữ master sống cho CẢ 28 block, đi pass 2 (coordinate descent) rồi mới bake.

3 arm:
  N1 t2:4  f16/g32  (1.94 bpw) + đủ 3 fix  -> mục tiêu: THẮNG mốc 594 của exp_k
  N3 t2:4  f8/g64   (1.56 bpw) + đủ 3 fix  -> nếu FIX-1 xóa được thuế scale: chất lượng ~N1
                                              ở DƯỚI ngân sách BitNet 1.58 — kết quả đắt nhất
  N2 dense scale/TENSOR (1.58 bpw) + đủ 3 fix -> đối chứng: scale học được cứu nổi per-tensor không
Đối chiếu: exp_k t2:4+SEQ(Lloyd,1pass) = vi 594/ja 8069; exp_m đang cho Lloyd-SEQ của N2/N3-họ hàng.
"""
import glob
import io
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
random.seed(0)
torch.set_num_threads(5)

MODEL_DIR = glob.glob(r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*")[0]
DEV_VI = r"D:\Bit-Translate-data\clean_v7g\dev.vi"
DEV_JA = r"D:\Bit-Translate-data\clean_v7g\dev.ja"
OUT_JSON = r"e:\Bit-Translate\eval\lowbit_ptq\exp_n_results.json"
PASS_STEPS = (70, 30)      # FIX-3: pass1, pass2
SEQ_LR = 1e-3
SCALE_LR = 5e-3            # scale học nhanh hơn weight một chút
SEQ_BATCH = 6
N_CALIB = 60
N_EVAL = 16
MAX_TOK = 96
LIN_PATHS = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
             ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]
NORM_PATHS = [("", "input_layernorm"), ("", "post_attention_layernorm"),
              ("self_attn", "q_norm"), ("self_attn", "k_norm")]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def read_lines(path, n):
    out = []
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(ln)
            if len(out) >= n:
                break
    return out


def to_f8(s):
    try:
        return s.to(torch.float8_e4m3fn).to(torch.float32)
    except Exception:
        sign = torch.sign(s)
        a = s.abs().clamp(min=2 ** -9)
        e = torch.floor(torch.log2(a))
        return sign * (2 ** e) * (torch.round(a / (2 ** e) * 8) / 8)


def to_f16(s):
    return s.to(torch.float16).to(torch.float32)


SCALE_Q = {"f16": to_f16, "f8": to_f8}


def wanda_nm_mask(W, xnorm, N, M):
    R, C = W.shape
    imp = W.abs() * xnorm[None, :].clamp(min=1e-8)
    pad = (M - C % M) % M
    A = F.pad(imp, (0, pad)) if pad else imp
    g = A.view(R, -1, M)
    kth = g.kthvalue(M - N + 1, dim=2, keepdim=True).values
    return (g >= kth).view(R, -1)[:, :C]


def inv_softplus(y):
    return y + torch.log(-torch.expm1(-y))


class LearnQLinear(nn.Module):
    """FIX-1: ternary (mask cố định) với SCALE là tham số học được (STE qua f16/f8)."""

    def __init__(self, lin, mask, sgroup, sdtype):
        super().__init__()
        W0 = lin.weight.data
        self.Wfp = nn.Parameter(W0.clone())
        self.sgroup, self.sdtype = sgroup, sdtype
        R, C = W0.shape
        self.R, self.C = R, C
        m = torch.ones_like(W0) if mask is None else mask.float()
        if sgroup == "tensor":
            self.G = R * C
        else:
            self.G = int(sgroup)
            pad = (self.G - C % self.G) % self.G
            self.pad = pad
        self.register_buffer("maskf", m)
        # init scale bằng Lloyd 3 vòng trên survivor (điểm khởi đầu tốt, sau đó HỌC)
        with torch.no_grad():
            Wv, Mv = self._views(W0)
            Wm = Wv * Mv
            cnt = Mv.sum(2, keepdim=True).clamp(min=1)
            s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-6)
            for _ in range(3):
                t = torch.round(Wm / s).clamp(-1, 1) * Mv
                num = (Wm * t).sum(2, keepdim=True)
                den = (t * t).sum(2, keepdim=True).clamp(min=1e-8)
                s = (num / den).abs().clamp(min=1e-6)
        self.raw_s = nn.Parameter(inv_softplus(s))

    def _views(self, W):
        m = self.maskf
        if self.sgroup == "tensor":
            return W.reshape(1, 1, -1), m.reshape(1, 1, -1)
        pad = self.pad
        Wp = F.pad(W, (0, pad)) if pad else W
        Mp = F.pad(m, (0, pad)) if pad else m
        return Wp.view(self.R, -1, self.G), Mp.view(self.R, -1, self.G)

    def quant(self):
        # sàn 4e-3 > mức subnormal nhỏ nhất của f8 (2^-9≈0.002) — chống scale trôi về 0
        # (bug NaN 2026-08-01: scale học tự do xuống dưới lưới f8 -> cast=0 -> 0/0 ở phần tử mask)
        s = F.softplus(self.raw_s).clamp(min=4e-3)
        s_q = s + (SCALE_Q[self.sdtype](s) - s).detach()      # STE qua dtype scale
        s_q = s_q.clamp(min=1e-6)                              # chống chia 0
        Wv, Mv = self._views(self.Wfp)
        t = (torch.round((Wv * Mv) / s_q.detach()).clamp(-1, 1) * Mv).detach()  # gán mức: hằng số
        q = (t * s_q).reshape(self.R, -1)[:, :self.C]          # grad chảy vào s_q
        return q

    def forward(self, x):
        q = self.quant()
        Wq = q + self.Wfp - self.Wfp.detach()                  # STE cho weight
        return F.linear(x, Wq)

    @torch.no_grad()
    def bake(self):
        return self.quant().detach().clone()


@torch.no_grad()
def eval_ppl(model, tok, lines):
    nll, ntok = 0.0, 0
    for s in lines:
        ids = tok(s, return_tensors="pt", truncation=True, max_length=MAX_TOK).input_ids
        if ids.shape[1] < 2:
            continue
        nll += model(ids, labels=ids).loss.item() * (ids.shape[1] - 1)
        ntok += ids.shape[1] - 1
    return float(np.exp(nll / max(ntok, 1)))


CONFIGS = [
    ("N1_t24_g32f16_3fix_1.94bpw", (2, 4), "32", "f16", 1.94),
    ("N3_t24_f8g64_3fix_1.56bpw", (2, 4), "64", "f8", 1.56),
    ("N2_dense_tensor_3fix_1.58bpw", None, "tensor", "f16", 1.58),
    # vòng O — nhánh sub-1.3 bpw nhận đủ 3 fix (bpw = payload + mask + scale f8/g64)
    ("O1_t14_f8g64_3fix_1.02bpw", (1, 4), "64", "f8", 1.02),
    ("O2_t18_f8g64_3fix_0.70bpw", (1, 8), "64", "f8", 0.70),
    ("O3_t110_f8g64_3fix_0.62bpw", (1, 10), "64", "f8", 0.62),
]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("Nạp Qwen3-0.6B FP32")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and "layers." in n]

    vi = read_lines(DEV_VI, N_CALIB // 2)
    ja = read_lines(DEV_JA, N_CALIB // 2)
    calib = [x for pr in zip(vi, ja) for x in pr]
    eval_vi = read_lines(DEV_VI, 400)[-N_EVAL:]
    eval_ja = read_lines(DEV_JA, 400)[-N_EVAL:]
    ppl_fp = (eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja))
    log(f"FP32: vi {ppl_fp[0]:.1f} / ja {ppl_fp[1]:.1f}")
    results = {"fp32": {"ppl_vi": ppl_fp[0], "ppl_ja": ppl_fp[1]}}
    # gộp kết quả cũ (bỏ NaN) để chạy lại từng arm không mất arm đã xong
    if os.path.exists(OUT_JSON):
        try:
            with io.open(OUT_JSON, "r", encoding="utf-8") as f:
                old = json.load(f)
            for k, v in old.items():
                if k != "fp32" and v.get("ppl_vi") == v.get("ppl_vi"):
                    results[k] = v
        except Exception:
            pass
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None

    # xnorm cho Wanda
    xn_acc = {n: None for n, _ in linears}
    handles = []
    def mk(nm):
        def h(mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            v = x.pow(2).sum(0)
            xn_acc[nm] = v if xn_acc[nm] is None else xn_acc[nm] + v
        return h
    for n, m in linears:
        handles.append(m.register_forward_pre_hook(mk(n)))
    calib_ids = []
    with torch.no_grad():
        for s in calib:
            ids = tok(s, return_tensors="pt", truncation=True, max_length=MAX_TOK).input_ids
            if ids.shape[1] >= 4:
                calib_ids.append(ids)
                model(ids)
    for h in handles:
        h.remove()
    xnorm = {n: xn_acc[n].sqrt() for n in xn_acc}
    orig_lin = {n: m.weight.data.clone() for n, m in linears}
    orig_norm = {n: m.weight.data.clone() for n, m in model.named_modules()
                 if isinstance(m, nn.Module) and hasattr(m, "weight") and "norm" in n}

    layers = model.model.layers
    NB = len(layers)
    log("Tính quỹ đạo FP...")
    ropes, H_fp = [], [[] for _ in range(NB + 1)]
    with torch.no_grad():
        for ids in calib_ids:
            h = model.model.embed_tokens(ids)
            pos = torch.arange(ids.shape[1])[None]
            cos, sin = model.model.rotary_emb(h, pos)
            ropes.append((cos, sin))
            for b, blk in enumerate(layers):
                H_fp[b].append(h.clone())
                h = blk(h, position_embeddings=(cos, sin))
            H_fp[NB].append(h.clone())
    NC = len(calib_ids)
    eval_sub = list(range(0, NC, max(1, NC // 12)))[:12]

    for cname, nm, sgroup, sdtype, bpw_val in CONFIGS:
        if only and cname not in only:
            continue
        log(f"=== {cname} ===")
        # restore FP
        with torch.no_grad():
            for n, m in linears:
                m.weight.data = orig_lin[n].clone()
            for n, w in orig_norm.items():
                dict(model.named_modules())[n].weight.data = w.clone()
        # wrap TOÀN BỘ block (FIX-3 cần master sống cả 28 block)
        wrapped = []
        for b, blk in enumerate(layers):
            for sub, name in LIN_PATHS:
                parent = getattr(blk, sub) if sub else blk
                lin = getattr(parent, name)
                key = f"model.layers.{b}.{'{}.'.format(sub) if sub else ''}{name}"
                mask = wanda_nm_mask(orig_lin[key], xnorm[key], nm[0], nm[1]) if nm else None
                w = LearnQLinear(lin, mask, sgroup, sdtype)
                setattr(parent, name, w)
                wrapped.append((b, parent, name, lin, w))
        t0 = time.time()
        for p_idx, steps in enumerate(PASS_STEPS):
            # quỹ đạo lượng tử hiện hành (đầu pass)
            H_q = [H_fp[0][s].clone() for s in range(NC)]
            for b, blk in enumerate(layers):
                mods = [(par, nam, l, w) for (bb, par, nam, l, w) in wrapped if bb == b]
                params = [w.Wfp for _, _, _, w in mods] + [w.raw_s for _, _, _, w in mods]
                # FIX-2: norm của block học cùng
                norm_ws = []
                for sub, name in NORM_PATHS:
                    parent = getattr(blk, sub) if sub else blk
                    norm_ws.append(getattr(parent, name).weight)
                for nw in norm_ws:
                    nw.requires_grad_(True)
                opt = torch.optim.Adam([
                    {"params": [w.Wfp for _, _, _, w in mods], "lr": SEQ_LR},
                    {"params": [w.raw_s for _, _, _, w in mods], "lr": SCALE_LR},
                    {"params": norm_ws, "lr": 5e-4},
                ])
                best, best_state = float("inf"), None

                def eval_block():
                    with torch.no_grad():
                        v = 0.0
                        for s in eval_sub:
                            v += F.mse_loss(blk(H_q[s], position_embeddings=ropes[s]), H_fp[b + 1][s]).item()
                    return v

                for step in range(steps):
                    idx = random.sample(range(NC), min(SEQ_BATCH, NC))
                    opt.zero_grad()
                    loss = 0.0
                    for s in idx:
                        loss = loss + F.mse_loss(blk(H_q[s], position_embeddings=ropes[s]), H_fp[b + 1][s])
                    (loss / len(idx)).backward()
                    opt.step()
                    if step % 10 == 9 or step == steps - 1:
                        v = eval_block()
                        if v < best:
                            best = v
                            best_state = ([w.Wfp.detach().clone() for _, _, _, w in mods],
                                          [w.raw_s.detach().clone() for _, _, _, w in mods],
                                          [nw.detach().clone() for nw in norm_ws])
                if best_state is not None:
                    with torch.no_grad():
                        for (_, _, _, w), Wb, Sb in zip(mods, best_state[0], best_state[1]):
                            w.Wfp.data = Wb
                            w.raw_s.data = Sb
                        for nw, nb in zip(norm_ws, best_state[2]):
                            nw.data = nb
                for nw in norm_ws:
                    nw.requires_grad_(False)
                with torch.no_grad():
                    for s in range(NC):
                        H_q[s] = blk(H_q[s], position_embeddings=ropes[s]).detach()
            log(f"    pass {p_idx+1} xong ({time.time()-t0:.0f}s)")
        # bake
        with torch.no_grad():
            for b, parent, name, lin, w in wrapped:
                lin.weight.data = w.bake()
                setattr(parent, name, lin)
        pv, pj = eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja)
        log(f"    => {cname}: PPL vi {pv:.1f} / ja {pj:.1f}")
        results[cname] = {"bpw": bpw_val, "ppl_vi": pv, "ppl_ja": pj}
        with io.open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("EXP N — SCALE HỌC ĐƯỢC + NORM ĐỒNG-TỐI-ƯU + 2 PASS")
    print("=" * 78)
    for k, v in results.items():
        if k == "fp32":
            continue
        print(f"{k:40s} bpw {v['bpw']:.2f}  vi {v['ppl_vi']:10.1f}  ja {v['ppl_ja']:10.1f}")
    print(f"{'fp32':40s} bpw 16.0  vi {ppl_fp[0]:10.1f}  ja {ppl_fp[1]:10.1f}")
    print("Mốc: exp_k t2:4+SEQ(Lloyd,1pass) vi 594/ja 8069; exp_m cho M1/M2/M3 Lloyd-SEQ.")
    print("=" * 78)


if __name__ == "__main__":
    main()
