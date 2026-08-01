# -*- coding: utf-8 -*-
"""
Exp L — SUB-1-BIT THẬT (kế toán đủ: payload + mask + scale). Ba đòn bẩy giá-bit:
  1. Scale rẻ: f16/g32 (0.500 bpw) -> f8/g64 (0.125) -> f16/row (~0.016) -> f16/tensor (~0).
  2. Mask cấu trúc: 1:4 (0.50 bpw) / 1:8 (0.375) / 2:8 (0.601) — rẻ hơn hẳn mask tự do (H(p)).
  3. Payload: ternary (1.58/kept) vs binary (1.0/kept).

6 arm — mỗi arm cô lập một câu hỏi (tất cả: mask Wanda, scale Lloyd-survivor, TF recon 100 step
+ track-best — fix-pack như exp_k để so được với arm1/1b của nó):

  A  ternary dense, scale/TENSOR   ~1.58 bpw  "scale granularity có quan trọng không?" (so 2.08 exp_k)
  B  ternary 1:4,  f8/g64          ~1.02 bpw  ternary rẻ nhất trên ngưỡng 1
  F  binary  2:8,  f8/g64          ~0.98 bpw  cùng mật độ 25% như D nhưng mask TỰ DO hơn (+0.1 bpw)
  C  ternary 1:4,  f16/ROW         ~0.91 bpw  ✓ sub-1: scale per-row gần miễn phí
  D  binary  1:4,  f8/g64          ~0.88 bpw  ✓ sub-1: binary + mask 1:4
  E  ternary 1:8,  f8/g64          ~0.70 bpw  ✓ sub-1 sâu: 12.5%% trọng số sống

bpw tính CHÍNH XÁC per-matrix rồi lấy trung bình có trọng số — in ra để khỏi tự lừa.
"""
import glob
import io
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
torch.set_num_threads(5)

MODEL_DIR = glob.glob(r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*")[0]
DEV_VI = r"D:\Bit-Translate-data\clean_v7g\dev.vi"
DEV_JA = r"D:\Bit-Translate-data\clean_v7g\dev.ja"
OUT_JSON = r"e:\Bit-Translate\eval\lowbit_ptq\exp_l_results.json"
TF_STEPS = 100
TF_LR = 1.5e-3
N_CALIB = 60
N_EVAL = 16
MAX_TOK = 96


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
    """mô phỏng float8 e4m3 (round-trip qua dtype thật nếu có)."""
    try:
        return s.to(torch.float8_e4m3fn).to(torch.float32)
    except Exception:
        sign = torch.sign(s)
        a = s.abs().clamp(min=2 ** -9)
        e = torch.floor(torch.log2(a))
        m = a / (2 ** e)
        return sign * (2 ** e) * (torch.round(m * 8) / 8)


def to_f16(s):
    return s.to(torch.float16).to(torch.float32)


SCALE_Q = {"f16": to_f16, "f8": to_f8}


def group_views(W, mask, sgroup):
    """trả (Wv, Mv, unview) — Wv/Mv dạng [G_rows, n_groups, group_len]."""
    R, C = W.shape
    M = torch.ones_like(W) if mask is None else mask.float()
    if sgroup == "tensor":
        return W.reshape(1, 1, -1), M.reshape(1, 1, -1), (R, C, 0)
    if sgroup == "row":
        return W.view(R, 1, C), M.view(R, 1, C), (R, C, 0)
    G = int(sgroup)
    pad = (G - C % G) % G
    Wp = F.pad(W, (0, pad)) if pad else W
    Mp = F.pad(M, (0, pad)) if pad else M
    return Wp.view(R, -1, G), Mp.view(R, -1, G), (R, C, pad)


def q_ternary(W, mask, sgroup, sdtype):
    Wv, Mv, (R, C, pad) = group_views(W, mask, sgroup)
    Wm = Wv * Mv
    cnt = Mv.sum(2, keepdim=True).clamp(min=1)
    s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-8)
    for _ in range(3):
        t = torch.round(Wm / s).clamp(-1, 1) * Mv
        num = (Wm * t).sum(2, keepdim=True)
        den = (t * t).sum(2, keepdim=True).clamp(min=1e-8)
        s = (num / den).abs().clamp(min=1e-8)
    sq = SCALE_Q[sdtype](s)
    t = torch.round(Wm / sq.clamp(min=1e-8)).clamp(-1, 1) * Mv
    out = (t * sq).reshape(R, -1)[:, :C]
    return out


def q_binary(W, mask, sgroup, sdtype):
    Wv, Mv, (R, C, pad) = group_views(W, mask, sgroup)
    Wm = Wv * Mv
    cnt = Mv.sum(2, keepdim=True).clamp(min=1)
    s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-8)  # LS-optimal cho binary
    sq = SCALE_Q[sdtype](s)
    out = (torch.sign(Wm) * sq * Mv).reshape(R, -1)[:, :C]
    return out


def wanda_nm_mask(W, xnorm, N, M):
    R, C = W.shape
    imp = W.abs() * xnorm[None, :].clamp(min=1e-8)
    pad = (M - C % M) % M
    A = F.pad(imp, (0, pad)) if pad else imp
    g = A.view(R, -1, M)
    kth = g.kthvalue(M - N + 1, dim=2, keepdim=True).values
    return (g >= kth).view(R, -1)[:, :C]


def bpw_exact(shape, N, M, sgroup, scale_bits, payload):
    R, C = shape
    n = R * C
    pay = (N / M if N else 1.0) * payload
    mask = (math.log2(math.comb(M, N)) / M) if N else 0.0
    if sgroup == "tensor":
        sc = scale_bits / n
    elif sgroup == "row":
        sc = scale_bits / C
    else:
        sc = scale_bits / int(sgroup)
    return pay + mask + sc


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


def tf_reconstruct(W0, X, qfn):
    target = (X @ W0.t()).detach()
    Wfp = W0.clone().requires_grad_(True)
    opt = torch.optim.Adam([Wfp], lr=TF_LR)
    best, bestW = float("inf"), None
    for _ in range(TF_STEPS):
        opt.zero_grad()
        with torch.no_grad():
            q = qfn(Wfp)
        Wq = Wfp + (q - Wfp).detach()
        loss = (X @ Wq.t() - target).pow(2).mean()
        if loss.item() < best:
            best = loss.item()
            bestW = q.detach().clone()
        loss.backward()
        opt.step()
    return bestW


ARMS = [
    # (tên, base, N, M, sgroup, sdtype, scale_bits)
    ("A_ternary_dense_scaleTENSOR", "t", None, None, "tensor", "f16", 16),
    ("B_ternary_1:4_f8_g64",        "t", 1, 4, "64", "f8", 8),
    ("F_binary_2:8_f8_g64",         "b", 2, 8, "64", "f8", 8),
    ("C_ternary_1:4_f16_row",       "t", 1, 4, "row", "f16", 16),
    ("D_binary_1:4_f8_g64",         "b", 1, 4, "64", "f8", 8),
    ("E_ternary_1:8_f8_g64",        "t", 1, 8, "64", "f8", 8),
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

    log("Thu activation calibration...")
    acts = {n: [] for n, _ in linears}
    handles = []
    def mk(nm):
        def h(mod, inp):
            acts[nm].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float16))
        return h
    for n, m in linears:
        handles.append(m.register_forward_pre_hook(mk(n)))
    with torch.no_grad():
        for s in calib:
            ids = tok(s, return_tensors="pt", truncation=True, max_length=MAX_TOK).input_ids
            if ids.shape[1] >= 4:
                model(ids)
    for h in handles:
        h.remove()
    X_all, xnorm = {}, {}
    for n in acts:
        X = torch.cat(acts[n], 0)
        X_all[n] = X
        xnorm[n] = X.float().pow(2).mean(0).sqrt()
        acts[n] = None
    orig = {n: m.weight.data.clone() for n, m in linears}
    total_p = sum(v.numel() for v in orig.values())

    for name, base, N, M, sgroup, sdtype, sbits in ARMS:
        payload = 1.58 if base == "t" else 1.0
        bpw_avg = sum(bpw_exact(orig[n].shape, N, M, sgroup, sbits, payload) * orig[n].numel()
                      for n, _ in linears) / total_p
        log(f"=== ARM {name}  (bpw thật = {bpw_avg:.3f}) ===")
        t0 = time.time()
        for i, (n, m) in enumerate(linears):
            W0 = orig[n]
            mask = wanda_nm_mask(W0, xnorm[n], N, M) if N else None
            qfn = (lambda W, _m=mask: q_ternary(W, _m, sgroup, sdtype)) if base == "t" \
                else (lambda W, _m=mask: q_binary(W, _m, sgroup, sdtype))
            X = X_all[n].to(torch.float32)
            m.weight.data = tf_reconstruct(W0, X, qfn)
            if (i + 1) % 98 == 0:
                log(f"    {i+1}/196 ({time.time()-t0:.0f}s)")
        pv, pj = eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja)
        log(f"    => PPL vi {pv:.1f} / ja {pj:.1f}")
        results[name] = {"bpw": round(bpw_avg, 3), "ppl_vi": pv, "ppl_ja": pj}
        with io.open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with torch.no_grad():
            for n, m in linears:
                m.weight.data = orig[n].clone()

    print("\n" + "=" * 78)
    print("EXP L — SUB-1-BIT THẬT (kế toán đủ payload+mask+scale, Wanda+Lloyd+recon100)")
    print("=" * 78)
    print(f"{'arm':34s}{'bpw':>7s}{'PPL vi':>12s}{'PPL ja':>12s}")
    print("-" * 78)
    for k, v in results.items():
        if k == "fp32":
            continue
        print(f"{k:34s}{v['bpw']:7.3f}{v['ppl_vi']:12.1f}{v['ppl_ja']:12.1f}")
    print(f"{'fp32':34s}{16.0:7.1f}{ppl_fp[0]:12.1f}{ppl_fp[1]:12.1f}")
    print("=" * 78)
    print("Đối chiếu exp_k (cùng fix-pack): arm1 t2:4 1.94bpw & arm1b dense 2.08bpw & arm2 SEQ.")


if __name__ == "__main__":
    main()
