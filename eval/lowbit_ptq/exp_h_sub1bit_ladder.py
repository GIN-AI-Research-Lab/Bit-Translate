# -*- coding: utf-8 -*-
"""
Exp H — DẢI SUB-1.58 / SUB-1-BIT trên Qwen3-0.6B (reconstruction, không train toàn model).

Trả lời trực tiếp: convert Qwen3-0.6B xuống DƯỚI 1.58 bit và DƯỚI 1 bit thì PPL sụp cỡ nào?
Dùng reconstruction (STE + track-best, đã sửa ở exp_f) — cách "không train" fair nhất.

Format (đều reconstruct 196 ma trận rồi đo PPL dev vi/ja):
  - ternary-g32       ~1.58 bit   (mốc, exp_f cho ~5185)
  - binary-g32        1.00 bit    {-1,+1}*absmean (bỏ mức 0)
  - binary + 5% salient int8      ~1.35 bit  (kiểu PB-LLM: giữ 5% trọng số lớn ở int8)
  - ternary 50% sparse ~1.0 bit   (giữ 50% |W| lớn nhất làm ternary, còn lại = 0)
  - ternary 75% sparse ~0.6 bit   (sub-1-bit)
Bit tính theo: keep_frac*payload + H(keep_frac) (mask) — chi phí thật của sparsity.

Baseline SOTA để so (từ paper, model 7B): STBLLM 0.55bit LLaMA-7B PPL 31.7; BiLLM 0.55bit 688.
7B có DƯ THỪA lớn; 0.6B thì không => đây là ca KHÓ NHẤT cho sub-1-bit.
"""
import glob
import io
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
GROUP = 32
RECON_STEPS = 80
LR = 1.5e-3
N_CALIB = 40
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


def Hbin(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def _grp(W):
    R, C = W.shape
    pad = (GROUP - C % GROUP) % GROUP
    Wp = F.pad(W, (0, pad)) if pad else W
    return Wp.view(R, -1, GROUP), C


def q_ternary(W):
    Wg, C = _grp(W)
    s = Wg.abs().mean(dim=2, keepdim=True).clamp(min=1e-8)
    return (torch.round(Wg / s).clamp(-1, 1) * s).view(W.shape[0], -1)[:, :C]


def q_binary(W):
    Wg, C = _grp(W)
    s = Wg.abs().mean(dim=2, keepdim=True).clamp(min=1e-8)
    return (torch.sign(Wg) * s).view(W.shape[0], -1)[:, :C]


def salient_mask(W, frac):
    k = max(1, int(W.numel() * frac))
    thr = torch.kthvalue(W.abs().flatten(), W.numel() - k).values
    return W.abs() >= thr


def make_q(mode):
    """trả về hàm quant theo mode; kèm bit/w danh nghĩa."""
    if mode == "ternary":
        return (lambda W: q_ternary(W)), 1.58 + 0.5
    if mode == "binary":
        return (lambda W: q_binary(W)), 1.0 + 0.5
    if mode == "binary+5%int8":
        def f(W):
            m = salient_mask(W, 0.05)
            base = q_binary(W)
            # 5% salient giữ int8 (absmax/127 toàn ma trận)
            s = (W.abs().max() / 127).clamp(min=1e-8)
            hi = torch.round(W / s).clamp(-127, 127) * s
            return torch.where(m, hi, base)
        return f, 0.95 * 1.0 + 0.05 * 8 + 0.5 + Hbin(0.05)
    if mode.startswith("ternary_sparse"):
        keep = float(mode.split("@")[1])
        def f(W):
            m = salient_mask(W, keep)
            return q_ternary(W) * m
        return f, keep * 1.58 + Hbin(keep) + 0.5 * keep
    raise ValueError(mode)


def ste(W, qfn):
    return W + (qfn(W) - W).detach()


def reconstruct(W0, X, qfn):
    target = (X @ W0.t()).detach()
    tnorm = target.norm() + 1e-12
    Wfp = W0.clone().requires_grad_(True)
    opt = torch.optim.Adam([Wfp], lr=LR)
    best, bestW = float("inf"), qfn(W0).detach().clone()
    for _ in range(RECON_STEPS):
        opt.zero_grad()
        loss = (X @ ste(Wfp, qfn).t() - target).pow(2).mean()
        if loss.item() < best:
            best = loss.item(); bestW = qfn(Wfp).detach().clone()
        loss.backward(); opt.step()
    return bestW


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


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("Nạp Qwen3-0.6B FP32")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and "layers." in n]

    vi = read_lines(DEV_VI, N_CALIB // 2); ja = read_lines(DEV_JA, N_CALIB // 2)
    calib = [x for pr in zip(vi, ja) for x in pr]
    eval_vi = read_lines(DEV_VI, 400)[-N_EVAL:]; eval_ja = read_lines(DEV_JA, 400)[-N_EVAL:]
    ppl_fp = (eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja))
    log(f"FP32: vi {ppl_fp[0]:.1f} / ja {ppl_fp[1]:.1f}")

    acts, handles = {n: [] for n, _ in linears}, []
    def mk(nm):
        def h(mod, inp):
            acts[nm].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float16))
        return h
    for n, m in linears:
        handles.append(m.register_forward_pre_hook(mk(n)))
    with torch.no_grad():
        for s in calib:
            model(tok(s, return_tensors="pt", truncation=True, max_length=MAX_TOK).input_ids)
    for h in handles:
        h.remove()
    for n in acts:
        acts[n] = torch.cat(acts[n], 0)
    orig = {n: m.weight.data.clone() for n, m in linears}

    FORMATS = ["ternary", "binary", "binary+5%int8", "ternary_sparse@0.5", "ternary_sparse@0.25"]
    rows = []
    for mode in FORMATS:
        qfn, bpw = make_q(mode)
        log(f"--- {mode} (~{bpw:.2f} bit/w) ---")
        t0 = time.time()
        for i, (n, m) in enumerate(linears):
            X = acts[n].to(torch.float32)
            m.weight.data = reconstruct(orig[n], X, qfn)
            if (i + 1) % 49 == 0:
                log(f"    {i+1}/{len(linears)} ({time.time()-t0:.0f}s)")
        pv, pj = eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja)
        rows.append((mode, bpw, pv, pj))
        log(f"    => PPL vi {pv:.1f} / ja {pj:.1f}")
        # khôi phục gốc cho format sau
        with torch.no_grad():
            for n, m in linears:
                m.weight.data = orig[n]

    print("\n" + "=" * 74)
    print("DẢI SUB-1.58 / SUB-1-BIT trên Qwen3-0.6B (reconstruction, không train toàn model)")
    print("=" * 74)
    print(f"{'format':22s}{'bit/w':>8s}{'PPL vi':>11s}{'PPL ja':>11s}")
    print("-" * 74)
    for mode, bpw, pv, pj in rows:
        print(f"{mode:22s}{bpw:8.2f}{pv:11.1f}{pj:11.1f}")
    print(f"{'FP32':22s}{16.0:8.2f}{ppl_fp[0]:11.1f}{ppl_fp[1]:11.1f}")
    print("=" * 74)
    print("So SOTA (paper, LLaMA-7B): STBLLM 0.55bit PPL 31.7 | BTC-LLM 0.7bit PPL 11.0")
    print("=> 7B dư thừa lớn nên sub-1-bit sống; 0.6B ít dư thừa => sụp mạnh hơn nhiều.")


if __name__ == "__main__":
    main()
