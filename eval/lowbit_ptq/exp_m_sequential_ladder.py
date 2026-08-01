# -*- coding: utf-8 -*-
"""
Exp M — SEQUENTIAL cho các format chưa được hưởng nó (trả nợ thang exp_l).

3 cấu hình, mỗi cái đóng một ô trống:
  M1 dense ternary g32-f16 (2.08 bpw) + SEQ  -> ô trống exp_k: dense+seq có THẮNG t2:4+seq (594)?
  M2 dense ternary scale/TENSOR (1.58 bpw) + SEQ -> đúng bit-budget BitNet/i2_s: 1.58 bpw + full combo
                                                    tới đâu? (TF = 66.999)
  M3 ternary 1:4 f8/g64 (1.02 bpw) + SEQ      -> hy vọng ~1 bit cuối cùng (TF = 245.163)

Sequential y hệt exp_k arm2 (BRECQ-lite): đi tuần tự 28 block, input = hidden ĐÃ lượng tử của
prefix, target = quỹ đạo FP, tối ưu 7 ma trận/block cùng lúc (STE + track-best), bake rồi đi tiếp.
Kỳ vọng ghi trước: nếu 10× của exp_k chuyển giao -> M1 ~440 (thắng 594?), M2 ~6.7k, M3 ~24k.
Per-layer error cao hơn có thể làm sequential ăn ít hơn — đo mới biết.
"""
import glob
import io
import json
import math
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
OUT_JSON = r"e:\Bit-Translate\eval\lowbit_ptq\exp_m_results.json"
SEQ_STEPS = 70
SEQ_LR = 1e-3
SEQ_BATCH = 6
N_CALIB = 60
N_EVAL = 16
MAX_TOK = 96
LIN_PATHS = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
             ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]


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


def group_views(W, mask, sgroup):
    R, C = W.shape
    M = torch.ones_like(W) if mask is None else mask.float()
    if sgroup == "tensor":
        return W.reshape(1, 1, -1), M.reshape(1, 1, -1), (R, C)
    G = int(sgroup)
    pad = (G - C % G) % G
    Wp = F.pad(W, (0, pad)) if pad else W
    Mp = F.pad(M, (0, pad)) if pad else M
    return Wp.view(R, -1, G), Mp.view(R, -1, G), (R, C)


def q_ternary(W, mask, sgroup, sdtype):
    Wv, Mv, (R, C) = group_views(W, mask, sgroup)
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
    return (t * sq).reshape(R, -1)[:, :C]


def wanda_nm_mask(W, xnorm, N, M):
    R, C = W.shape
    imp = W.abs() * xnorm[None, :].clamp(min=1e-8)
    pad = (M - C % M) % M
    A = F.pad(imp, (0, pad)) if pad else imp
    g = A.view(R, -1, M)
    kth = g.kthvalue(M - N + 1, dim=2, keepdim=True).values
    return (g >= kth).view(R, -1)[:, :C]


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


class STELinear(nn.Module):
    def __init__(self, lin, qfn):
        super().__init__()
        self.Wfp = nn.Parameter(lin.weight.data.clone())
        self.qfn = qfn

    def q(self, W=None):
        with torch.no_grad():
            return self.qfn(self.Wfp if W is None else W)

    def forward(self, x):
        q = self.q()
        Wq = self.Wfp + (q - self.Wfp).detach()
        return F.linear(x, Wq)


CONFIGS = [
    # (tên, dùng mask 1:4?, sgroup, sdtype, bpw)
    ("M1_dense_g32f16_SEQ_2.08bpw", None, "32", "f16", 2.08),
    ("M2_dense_tensor_SEQ_1.58bpw", None, "tensor", "f16", 1.58),
    ("M3_t14_f8g64_SEQ_1.02bpw", (1, 4), "64", "f8", 1.02),
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

    # xnorm cho mask Wanda (M3)
    log("Thu xnorm calibration...")
    xn_acc = {n: None for n, _ in linears}
    handles = []
    def mk(nm):
        def h(mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            s = x.pow(2).sum(0)
            xn_acc[nm] = s if xn_acc[nm] is None else xn_acc[nm] + s
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
    orig = {n: m.weight.data.clone() for n, m in linears}

    # quỹ đạo FP
    log("Tính quỹ đạo FP (H_fp) cho 28 block...")
    layers = model.model.layers
    NB = len(layers)
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
        log(f"=== {cname} ===")
        with torch.no_grad():
            for n, m in linears:
                m.weight.data = orig[n].clone()
        H_q = [H_fp[0][s].clone() for s in range(NC)]
        t0 = time.time()
        for b, blk in enumerate(layers):
            stes = []
            for sub, name in LIN_PATHS:
                parent = getattr(blk, sub)
                lin = getattr(parent, name)
                key = f"model.layers.{b}.{sub}.{name}"
                mask = wanda_nm_mask(orig[key], xnorm[key], nm[0], nm[1]) if nm else None
                qfn = (lambda W, _m=mask: q_ternary(W, _m, sgroup, sdtype))
                sm = STELinear(lin, qfn)
                setattr(parent, name, sm)
                stes.append((parent, name, lin, sm))
            params = [sm.Wfp for _, _, _, sm in stes]
            opt = torch.optim.Adam(params, lr=SEQ_LR)
            best, best_state = float("inf"), None

            def eval_block():
                with torch.no_grad():
                    v = 0.0
                    for s in eval_sub:
                        v += F.mse_loss(blk(H_q[s], position_embeddings=ropes[s]), H_fp[b + 1][s]).item()
                return v

            for step in range(SEQ_STEPS):
                idx = random.sample(range(NC), min(SEQ_BATCH, NC))
                opt.zero_grad()
                loss = 0.0
                for s in idx:
                    loss = loss + F.mse_loss(blk(H_q[s], position_embeddings=ropes[s]), H_fp[b + 1][s])
                (loss / len(idx)).backward()
                opt.step()
                if step % 10 == 9 or step == SEQ_STEPS - 1:
                    v = eval_block()
                    if v < best:
                        best = v
                        best_state = [sm.Wfp.detach().clone() for _, _, _, sm in stes]
            for (parent, name, lin, sm), Wb in zip(stes, best_state if best_state else [sm.Wfp.detach() for _, _, _, sm in stes]):
                lin.weight.data = sm.q(Wb)
                setattr(parent, name, lin)
            with torch.no_grad():
                for s in range(NC):
                    H_q[s] = blk(H_q[s], position_embeddings=ropes[s]).detach()
            if b % 7 == 6 or b == NB - 1:
                log(f"    block {b+1}/28 ({time.time()-t0:.0f}s)")
        pv, pj = eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja)
        log(f"    => {cname}: PPL vi {pv:.1f} / ja {pj:.1f}")
        results[cname] = {"bpw": bpw_val, "ppl_vi": pv, "ppl_ja": pj}
        with io.open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 76)
    print("EXP M — SEQUENTIAL LADDER (đối chiếu: t2:4+SEQ 1.94bpw = vi 594 / ja 8069)")
    print("=" * 76)
    for k, v in results.items():
        if k == "fp32":
            continue
        print(f"{k:38s} bpw {v['bpw']:.2f}  vi {v['ppl_vi']:10.1f}  ja {v['ppl_ja']:10.1f}")
    print(f"{'fp32':38s} bpw 16.00  vi {ppl_fp[0]:10.1f}  ja {ppl_fp[1]:10.1f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
