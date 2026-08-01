# -*- coding: utf-8 -*-
"""
Exp G — KẾT HỢP reconstruction + "lục lọi layer" (mixed precision). Đường cong PPL-vs-bit.

Trả lời: nếu cho phép nâng các lớp NHẠY NHẤT lên int4 (còn lại ternary), thì cần "ngân sách
bit trung bình" bao nhiêu để PPL về vùng dùng được? Đây là 2 hướng người dùng đề xuất gộp lại:
  - reconstruction (exp_f) — bố trí thông minh, không train toàn model.
  - mixed precision theo độ nhạy — "lục lọi weight/layer", giữ lớp nhạy ở bit cao.

Mỗi linear được reconstruct ở CẢ ternary-g32 LẪN int4-g32 (STE + track-best, LR thấp — đã sửa
ở exp_f). Xếp hạng độ nhạy bằng out-err ternary sau recon (proxy: lớp nào ternary còn sai nhiều
nhất thì nâng int4 trước). Quét tỉ lệ int4 -> vẽ đường cong bit trung bình vs PPL dev vi/ja.

LƯU Ý: vẫn là reconstruction teacher-forcing (chưa sequential) => chặn trên lạc quan cho phần
error-propagation. Đường cong cho biết mixed precision kéo được tới đâu TRƯỚC khi cần sequential.
"""
import glob
import io
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
RECON_STEPS = 100
LR = 1.5e-3
N_CALIB = 40
N_EVAL = 16
MAX_TOK = 96
BIT_TERNARY = 1.58 + 16.0 / GROUP   # payload log2(3) + scale f16 mỗi 32 cột = 2.08 bpw
BIT_INT4 = 4.0 + 16.0 / GROUP       # 4.50 bpw
SWEEP = [0.0, 0.15, 0.30, 0.50, 1.0]  # tỉ lệ lớp (theo độ nhạy) được nâng int4


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def read_lines(path, n):
    out = []
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line)
            if len(out) >= n:
                break
    return out


def _grouped(W):
    R, C = W.shape
    pad = (GROUP - C % GROUP) % GROUP
    Wp = F.pad(W, (0, pad)) if pad else W
    return Wp.view(R, -1, GROUP), C


def ternary_g32(W):
    Wg, C = _grouped(W)
    s = Wg.abs().mean(dim=2, keepdim=True).clamp(min=1e-8)
    return (torch.round(Wg / s).clamp(-1, 1) * s).view(W.shape[0], -1)[:, :C]


def int4_g32(W):
    Wg, C = _grouped(W)
    s = (Wg.abs().amax(dim=2, keepdim=True) / 7.0).clamp(min=1e-8)
    return (torch.round(Wg / s).clamp(-7, 7) * s).view(W.shape[0], -1)[:, :C]


def ste(W, qfn):
    return W + (qfn(W) - W).detach()


def reconstruct(W0, X, qfn):
    """Tối ưu master weight (STE) để (X @ Wq.t()) khớp (X @ W0.t()). Track-best. Trả (Wq, out-err)."""
    target = (X @ W0.t()).detach()
    tnorm = target.norm() + 1e-12
    Wfp = W0.clone().requires_grad_(True)
    opt = torch.optim.Adam([Wfp], lr=LR)
    best_loss, best_W = float("inf"), qfn(W0).detach().clone()
    for _ in range(RECON_STEPS):
        opt.zero_grad()
        loss = (X @ ste(Wfp, qfn).t() - target).pow(2).mean()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_W = qfn(Wfp).detach().clone()
        loss.backward()
        opt.step()
    with torch.no_grad():
        err = float(((X @ best_W.t()) - target).norm() / tnorm)
    return best_W, err


@torch.no_grad()
def eval_ppl(model, tok, lines):
    nll, ntok = 0.0, 0
    for s in lines:
        ids = tok(s, return_tensors="pt", truncation=True, max_length=MAX_TOK).input_ids
        if ids.shape[1] < 2:
            continue
        out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        nll += out.loss.item() * n
        ntok += n
    return float(np.exp(nll / max(ntok, 1)))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"Nạp Qwen3-0.6B FP32")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()

    linears = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and "layers." in n]
    log(f"{len(linears)} ma trận linear (giữ FP: embedding tied, lm_head, norm)")

    vi = read_lines(DEV_VI, N_CALIB // 2)
    ja = read_lines(DEV_JA, N_CALIB // 2)
    calib = [x for pair in zip(vi, ja) for x in pair]
    eval_vi = read_lines(DEV_VI, 400)[-N_EVAL:]
    eval_ja = read_lines(DEV_JA, 400)[-N_EVAL:]

    ppl_fp = (eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja))
    log(f"FP32 baseline: PPL vi {ppl_fp[0]:.1f} / ja {ppl_fp[1]:.1f}")

    # thu activation (teacher forcing)
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
        acts[n] = torch.cat(acts[n], dim=0)

    orig = {n: m.weight.data.clone() for n, m in linears}
    params = {n: m.weight.numel() for n, m in linears}
    total_p = sum(params.values())

    # reconstruct ternary + int4 cho mọi lớp
    log(f"Reconstruct ternary + int4 ({RECON_STEPS} step/ma trận x2) ...")
    W_t, W_i4, err_t = {}, {}, {}
    t0 = time.time()
    for i, (n, m) in enumerate(linears):
        X = acts[n].to(torch.float32)
        W_t[n], err_t[n] = reconstruct(orig[n], X, ternary_g32)
        W_i4[n], _ = reconstruct(orig[n], X, int4_g32)
        acts[n] = None
        if (i + 1) % 28 == 0 or i == len(linears) - 1:
            log(f"  [{i+1:3d}/{len(linears)}] ({time.time()-t0:.0f}s)")

    ranked = sorted(linears, key=lambda nm: err_t[nm[0]], reverse=True)  # nhạy nhất trước

    print("\n" + "=" * 72)
    print("ĐƯỜNG CONG PPL vs NGÂN SÁCH BIT (reconstruction + mixed precision)")
    print("=" * 72)
    print(f"{'%int4':>7s}{'bit TB':>9s}{'~MB linear':>12s}{'PPL vi':>10s}{'PPL ja':>10s}")
    print("-" * 72)
    rows = []
    for p in SWEEP:
        k = int(round(p * len(linears)))
        int4_set = set(n for n, _ in ranked[:k])
        bit_sum = 0.0
        with torch.no_grad():
            for n, m in linears:
                if n in int4_set:
                    m.weight.data = W_i4[n]; bit_sum += params[n] * BIT_INT4
                else:
                    m.weight.data = W_t[n]; bit_sum += params[n] * BIT_TERNARY
        bit_avg = bit_sum / total_p
        mb = bit_sum / 8 / 1024 / 1024
        pv, pj = eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja)
        rows.append((p, bit_avg, mb, pv, pj))
        print(f"{p*100:6.0f}%{bit_avg:9.2f}{mb:12.0f}{pv:10.1f}{pj:10.1f}")
    print("-" * 72)
    print(f"{'FP32':>7s}{16.0:9.2f}{'':>12s}{ppl_fp[0]:10.1f}{ppl_fp[1]:10.1f}")
    print("=" * 72)
    print("Đọc: %int4=0 là toàn ternary (như exp_f). Tăng %int4 = nâng lớp nhạy nhất lên int4.")
    print("Tìm điểm bit TB nhỏ nhất mà PPL về gần FP => đó là 'ngân sách bit' thực tế của convert.")
    print("Vẫn teacher-forcing (chưa sequential) => còn error-propagation chưa xử lý.")


if __name__ == "__main__":
    main()
