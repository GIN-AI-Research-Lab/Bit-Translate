# -*- coding: utf-8 -*-
"""
(c) STBLLM-style: structured binary + N:M sparsity trên Qwen3-0.6B (reconstruction, không train).

Dựa phát hiện exp_h: MỨC 0 (khả năng tắt) là thứ quý nhất. STBLLM phá rào 1-bit bằng cách
kết hợp binary (cho phần non-zero) + structured N:M sparsity (đưa mức 0 vào có cấu trúc, rẻ + nhanh).
So xem cách này có cứu binary (exp_h: 1.9 TRIỆU) về gần ternary-sparse (33k) hay tốt hơn không.

N:M = mỗi nhóm M cột giữ N phần tử |W| lớn nhất, còn lại = 0. Bit = (N/M)*payload + mask_structured.
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


def mask_bits(N, M):
    return math.log2(math.comb(M, N)) / M


def nm_mask(W, N, M):
    R, C = W.shape
    pad = (M - C % M) % M
    A = F.pad(W.abs(), (0, pad)) if pad else W.abs()
    g = A.view(R, -1, M)
    kth = g.kthvalue(M - N + 1, dim=2, keepdim=True).values
    return (g >= kth).view(R, -1)[:, :C]


def _grp(W):
    R, C = W.shape
    pad = (GROUP - C % GROUP) % GROUP
    Wp = F.pad(W, (0, pad)) if pad else W
    return Wp.view(R, -1, GROUP), C


def q_binary(W):
    Wg, C = _grp(W)
    s = Wg.abs().mean(dim=2, keepdim=True).clamp(min=1e-8)
    return (torch.sign(Wg) * s).view(W.shape[0], -1)[:, :C]


def q_ternary(W):
    Wg, C = _grp(W)
    s = Wg.abs().mean(dim=2, keepdim=True).clamp(min=1e-8)
    return (torch.round(Wg / s).clamp(-1, 1) * s).view(W.shape[0], -1)[:, :C]


def make_fmt(mode, N, M, base):
    payload = 1.0 if base == "binary" else 1.58
    bpw = (N / M) * payload + mask_bits(N, M)
    qbase = q_binary if base == "binary" else q_ternary
    def qfn(W, _mask=[None]):
        if _mask[0] is None or _mask[0].shape != W.shape:
            _mask[0] = nm_mask(W, N, M)
        return qbase(W) * _mask[0]
    return qfn, bpw


def ste(W, qfn):
    return W + (qfn(W) - W).detach()


def reconstruct(W0, X, qfn):
    # mask cố định từ W0
    mask = None
    target = (X @ W0.t()).detach()
    Wfp = W0.clone().requires_grad_(True)
    opt = torch.optim.Adam([Wfp], lr=LR)
    best, bestW = float("inf"), None
    for _ in range(RECON_STEPS):
        opt.zero_grad()
        loss = (X @ ste(Wfp, qfn).t() - target).pow(2).mean()
        if loss.item() < best:
            best = loss.item(); bestW = qfn(Wfp).detach().clone()
        loss.backward(); opt.step()
    return bestW if bestW is not None else qfn(W0)


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

    # (base, N, M): binary/ternary với N:M sparsity
    FORMATS = [
        ("binary 2:4", "binary", 2, 4),
        ("binary 1:4 (sub-1)", "binary", 1, 4),
        ("ternary 2:4", "ternary", 2, 4),
        ("ternary 1:4 (sub-1)", "ternary", 1, 4),
    ]
    rows = []
    for name, base, N, M in FORMATS:
        qfn, bpw = make_fmt(name, N, M, base)
        log(f"--- {name} (~{bpw:.2f} bit/w) ---")
        t0 = time.time()
        for i, (n, m) in enumerate(linears):
            X = acts[n].to(torch.float32)
            # mask riêng mỗi ma trận -> tạo qfn cục bộ để không chia sẻ state
            local_qfn, _ = make_fmt(name, N, M, base)
            m.weight.data = reconstruct(orig[n], X, local_qfn)
            if (i + 1) % 98 == 0:
                log(f"    {i+1}/{len(linears)} ({time.time()-t0:.0f}s)")
        pv, pj = eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja)
        rows.append((name, bpw, pv, pj))
        log(f"    => PPL vi {pv:.1f} / ja {pj:.1f}")
        with torch.no_grad():
            for n, m in linears:
                m.weight.data = orig[n]

    print("\n" + "=" * 76)
    print("STBLLM-STYLE: structured binary/ternary + N:M sparsity (Qwen3-0.6B, reconstruction)")
    print("=" * 76)
    print(f"{'format':24s}{'bit/w':>8s}{'PPL vi':>12s}{'PPL ja':>12s}")
    print("-" * 76)
    for name, bpw, pv, pj in rows:
        print(f"{name:24s}{bpw:8.2f}{pv:12.1f}{pj:12.1f}")
    print(f"{'--- mốc exp_h ---':24s}")
    print(f"{'binary dense':24s}{1.50:8.2f}{1861463.7:12.1f}{6897504.4:12.1f}")
    print(f"{'ternary 25% giữ':24s}{1.33:8.2f}{33686.7:12.1f}{47152.0:12.1f}")
    print(f"{'FP32':24s}{16.0:8.2f}{ppl_fp[0]:12.1f}{ppl_fp[1]:12.1f}")
    print("=" * 76)
    print("Câu hỏi: structured N:M có kéo binary từ 'triệu' về gần ternary-sparse không?")
    print("(SOTA STBLLM đạt 0.55bit PPL 31.7 trên LLaMA-7B; đây là 0.6B nên khó hơn nhiều.)")


if __name__ == "__main__":
    main()
