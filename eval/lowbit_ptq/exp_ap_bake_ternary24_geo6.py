# -*- coding: utf-8 -*-
"""
Exp AP — BAKE THẬT ternary N:M 2:4 (~1,5bpw, kỹ thuật thắng cuộc của toàn bộ nhánh PTQ
tối nay: Wanda 2:4 mask + Lloyd-scale-trên-survivor + SEQUENTIAL/BRECQ-lite) trên TOÀN BỘ
196 ma trận Qwen3-0.6B, rồi LƯU checkpoint đầy đủ + đo geo6 (6 miền: vi/ja/en/code/zh/math,
không chỉ vi/ja — đúng bài học "eval hẹp = thiên vị" của lab).

Khác exp_k_fixpack_sequential.py (chỉ đo PPL rồi revert, không lưu gì):
  1. GROUP=64 (không phải 32 như exp_k) — khớp CHÍNH XÁC granularity scale mà format TQ33
     đã thiết kế sẵn (`export_tq33_all_linears.py`: absmax per group-64) — để bước encode
     TQ33 sau đó KHÔNG bị lượng tử hóa lại lần 2 (group32-Lloyd -> group64-absmax sẽ làm
     tròn sai nếu 2 sub-group-32 trong cùng block64 có scale khác nhau). Bake trực tiếp ở
     đúng khung đích (đúng "luật F0" đã biết: train/bake TRONG khung đích rẻ hơn convert
     hậu kỳ ×4000 lần).
  2. Bỏ ARM1b/ARM1 (TF-only, đã biết thua SEQUENTIAL) — chạy thẳng SEQUENTIAL để tiết kiệm
     thời gian, đây là arm THẮNG duy nhất cần dùng thật.
  3. LƯU state_dict cuối cùng ra .pt — để bước sau (export_tq33_from_baked.py) đóng gói
     thành TQ33 thật + đo tok/s thật qua runner đã tối ưu (qwen3_runner_tq33_fast.exe).
  4. Đo geo6 (import EN_EVAL/CODE_EVAL/ZH_EVAL/MATH_EVAL từ exp_r_qat_lite.py — bộ 6 miền
     chuẩn đã dùng xuyên suốt lab) thay vì chỉ vi/ja.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
random.seed(0)
torch.set_num_threads(5)

from exp_r_qat_lite import (EN_EVAL, CODE_EVAL, ZH_EVAL, MATH_EVAL,  # noqa: E402
                             CAL_EN, CAL_CODE, CAL_ZH, CAL_MATH, CAL_CHAT, to_f8)

MODEL_DIR = glob.glob(r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*")[0]
DEV_VI = r"D:\Bit-Translate-data\clean_v7g\dev.vi"
DEV_JA = r"D:\Bit-Translate-data\clean_v7g\dev.ja"
OUT_JSON = r"e:\Bit-Translate\eval\lowbit_ptq\exp_ap_results.json"
OUT_CKPT = r"D:\Bit-Translate-data\qat_ckpts\ternary24_seq_g64_baked.pt"
GROUP = 64          # KHỚP granularity scale của format TQ33 (export_tq33_all_linears.py)
N_NM, M_NM = 2, 4    # 2:4 sparsity
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


def grp_pad(T, group=GROUP):
    R, C = T.shape
    pad = (group - C % group) % group
    Tp = F.pad(T, (0, pad)) if pad else T
    return Tp.view(R, -1, group), C


def lloyd_masked_ternary(W, mask, group=GROUP):
    """Ternary {-1,0,+1}*s per group-64, mask cố định (True=giữ), scale Lloyd trên survivor
    (đúng fix-pack đã verify thắng absmean/absmax thuần — exp_k)."""
    Wg, C = grp_pad(W, group)
    if mask is None:
        Mg = torch.ones_like(Wg)
    else:
        Mg, _ = grp_pad(mask.float(), group)
    Wm = Wg * Mg
    cnt = Mg.sum(2, keepdim=True).clamp(min=1)
    s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-8)
    for _ in range(3):
        t = torch.round(Wm / s).clamp(-1, 1) * Mg
        num = (Wm * t).sum(2, keepdim=True)
        den = (t * t).sum(2, keepdim=True).clamp(min=1e-8)
        s = (num / den).abs().clamp(min=1e-8)
    # ÉP scale về lưới f8 (<=256 giá trị/tensor) — BẮT BUỘC để TQ33 pack được (codebook 1-byte
    # scale-idx). Lloyd-scale liên tục (float) cho >30k giá trị khác nhau/tensor -> export crash.
    s = to_f8(s)
    t = torch.round(Wm / s).clamp(-1, 1) * Mg
    return (t * s).view(W.shape[0], -1)[:, :C]


def wanda_nm_mask(W, xnorm, N=N_NM, M=M_NM):
    R, C = W.shape
    imp = W.abs() * xnorm[None, :].clamp(min=1e-8)
    pad = (M - C % M) % M
    A = F.pad(imp, (0, pad)) if pad else imp
    g = A.view(R, -1, M)
    kth = g.kthvalue(M - N + 1, dim=2, keepdim=True).values
    return (g >= kth).view(R, -1)[:, :C]


def bpw_accounting(N=N_NM, M=M_NM, group=GROUP):
    payload = (N / M) * math.log2(3)
    mask_cost = math.log2(math.comb(M, N)) / M
    scale_cost = 16.0 / group
    return payload + mask_cost + scale_cost


@torch.no_grad()
def eval_ppl(model, tok, lines, max_tok=MAX_TOK):
    nll, ntok = 0.0, 0
    for s in lines:
        ids = tok(s, return_tensors="pt", truncation=True, max_length=max_tok).input_ids
        if ids.shape[1] < 2:
            continue
        nll += model(ids, labels=ids).loss.item() * (ids.shape[1] - 1)
        ntok += ids.shape[1] - 1
    return float(np.exp(nll / max(ntok, 1)))


class STELinear(nn.Module):
    def __init__(self, lin, mask):
        super().__init__()
        self.Wfp = nn.Parameter(lin.weight.data.clone())
        self.mask = mask

    def q(self, W=None):
        with torch.no_grad():
            return lloyd_masked_ternary(self.Wfp if W is None else W, self.mask)

    def forward(self, x):
        q = self.q()
        Wq = self.Wfp + (q - self.Wfp).detach()
        return F.linear(x, Wq)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"Nạp Qwen3-0.6B FP32 từ {MODEL_DIR}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and "layers." in n]
    log(f"{len(linears)} ma trận linear. bpw ternary2:4 (group={GROUP}): {bpw_accounting():.3f}")

    vi = read_lines(DEV_VI, N_CALIB // 2)
    ja = read_lines(DEV_JA, N_CALIB // 2)
    # TRỘN 6 miền cho calib (không chỉ vi/ja) — bài học "mixw" đã biết: calib hẹp làm miền
    # vắng mặt sụp (đo được lần chạy trước: en x16539, code x44978 khi calib chỉ vi/ja).
    calib = [x for pr in zip(vi, ja) for x in pr] + CAL_EN + CAL_CODE + CAL_ZH + CAL_MATH + CAL_CHAT
    random.shuffle(calib)
    eval_vi = read_lines(DEV_VI, 400)[-N_EVAL:]
    eval_ja = read_lines(DEV_JA, 400)[-N_EVAL:]

    def geo6(model):
        v = eval_ppl(model, tok, eval_vi)
        j = eval_ppl(model, tok, eval_ja)
        e = eval_ppl(model, tok, EN_EVAL)
        c = eval_ppl(model, tok, CODE_EVAL, max_tok=160)
        z = eval_ppl(model, tok, ZH_EVAL)
        m = eval_ppl(model, tok, MATH_EVAL)
        g = (v * j * e * c * z * m) ** (1 / 6)
        return {"vi": v, "ja": j, "en": e, "code": c, "zh": z, "math": m, "geo6": g}

    fp_geo6 = geo6(model)
    log(f"FP32 geo6: {json.dumps({k: round(x,1) for k,x in fp_geo6.items()})}")
    results = {"fp32": fp_geo6, "config": {"group": GROUP, "n_nm": N_NM, "m_nm": M_NM,
                                            "seq_steps": SEQ_STEPS, "n_calib": N_CALIB,
                                            "bpw_nominal": bpw_accounting()}}

    # ---- thu activation + xnorm cho mask Wanda ----
    log("Thu activation calibration...")
    acts = {n: [] for n, _ in linears}
    handles = []

    def mk(nm):
        def h(mod, inp):
            acts[nm].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float16))
        return h

    for n, m in linears:
        handles.append(m.register_forward_pre_hook(mk(n)))
    calib_ids = []
    with torch.no_grad():
        for s in calib:
            ids = tok(s, return_tensors="pt", truncation=True, max_length=MAX_TOK).input_ids
            if ids.shape[1] < 4:
                continue
            calib_ids.append(ids)
            model(ids)
    for h in handles:
        h.remove()
    xnorm = {}
    for n in acts:
        X = torch.cat(acts[n], 0)
        xnorm[n] = X.float().pow(2).mean(0).sqrt()
        acts[n] = None
    orig = {n: m.weight.data.clone() for n, m in linears}
    masks = {n: wanda_nm_mask(orig[n], xnorm[n]) for n, _ in linears}
    log(f"Đã dựng mask Wanda 2:4 cho {len(masks)} ma trận.")

    # ---- SEQUENTIAL block-wise (BRECQ-lite), đúng arm thắng cuộc exp_k ----
    log("=== SEQUENTIAL ternary 2:4 (group=64) — bake toàn bộ 28 block ===")
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
    H_q = [H_fp[0][s].clone() for s in range(NC)]
    eval_sub = list(range(0, NC, max(1, NC // 12)))[:12]
    t0 = time.time()
    for b, blk in enumerate(layers):
        stes = []
        for sub, name in LIN_PATHS:
            parent = getattr(blk, sub)
            lin = getattr(parent, name)
            key = f"model.layers.{b}.{sub}.{name}"
            sm = STELinear(lin, masks[key])
            setattr(parent, name, sm)
            stes.append((parent, name, lin, sm))
        params = [sm.Wfp for _, _, _, sm in stes]
        opt = torch.optim.Adam(params, lr=SEQ_LR)
        best, best_state = float("inf"), None

        def eval_block():
            with torch.no_grad():
                v = 0.0
                for s in eval_sub:
                    out = blk(H_q[s], position_embeddings=ropes[s])
                    v += F.mse_loss(out, H_fp[b + 1][s]).item()
            return v

        e0 = eval_block()
        for step in range(SEQ_STEPS):
            idx = random.sample(range(NC), min(SEQ_BATCH, NC))
            opt.zero_grad()
            loss = 0.0
            for s in idx:
                out = blk(H_q[s], position_embeddings=ropes[s])
                loss = loss + F.mse_loss(out, H_fp[b + 1][s])
            (loss / len(idx)).backward()
            opt.step()
            if step % 10 == 9 or step == SEQ_STEPS - 1:
                v = eval_block()
                if v < best:
                    best = v
                    best_state = [sm.Wfp.detach().clone() for _, _, _, sm in stes]
        for (parent, name, lin, sm), Wb in zip(
                stes, best_state if best_state else [sm.Wfp.detach() for _, _, _, sm in stes]):
            lin.weight.data = sm.q(Wb)
            setattr(parent, name, lin)
        with torch.no_grad():
            for s in range(NC):
                H_q[s] = blk(H_q[s], position_embeddings=ropes[s]).detach()
        log(f"    block {b:2d}/{NB}: mse {e0:.4f} -> {best:.4f}  ({time.time()-t0:.0f}s)")

    baked_geo6 = geo6(model)
    log(f"BAKED geo6: {json.dumps({k: round(x,1) for k,x in baked_geo6.items()})}")
    results["baked_ternary24_seq_g64"] = baked_geo6

    # ---- verify 2:4 THẬT trên checkpoint đã bake (đếm vi phạm nếu có) ----
    n_groups_total, n_violated_total = 0, 0
    for n, m in linears:
        W = m.weight.data
        R, C = W.shape
        pad = (4 - C % 4) % 4
        Wp = F.pad(W, (0, pad)) if pad else W
        g = Wp.view(R, -1, 4)
        nz = (g != 0).sum(-1)
        n_groups_total += nz.numel()
        n_violated_total += int((nz > 2).sum().item())
    log(f"Verify 2:4: {n_violated_total}/{n_groups_total} nhóm vi phạm "
        f"({100*n_violated_total/max(n_groups_total,1):.4f}%)")
    results["nm_violation"] = {"n_groups": n_groups_total, "n_violated": n_violated_total}

    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- LƯU checkpoint đầy đủ (để bước sau đóng gói TQ33 + đo tok/s thật) ----
    os.makedirs(os.path.dirname(OUT_CKPT), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "meta": {"recipe": "ternary_2:4_wanda_lloyd_sequential_g64",
                         "bpw_nominal": bpw_accounting(), "geo6": baked_geo6}}, OUT_CKPT)
    log(f"Đã lưu checkpoint -> {OUT_CKPT}")

    print("\n" + "=" * 92)
    print("EXP AP — BAKE THẬT ternary N:M 2:4 (group=64) + geo6 (6 miền) trên Qwen3-0.6B")
    print("=" * 92)
    print(f"{'miền':10s}{'FP32':>14s}{'baked 2:4':>14s}{'x FP':>10s}")
    for k in ("vi", "ja", "en", "code", "zh", "math", "geo6"):
        ratio = baked_geo6[k] / fp_geo6[k]
        print(f"{k:10s}{fp_geo6[k]:14.2f}{baked_geo6[k]:14.2f}{ratio:10.2f}")
    print(f"\nbpw danh nghĩa: {bpw_accounting():.3f}  |  vi phạm 2:4: {n_violated_total}/{n_groups_total}")
    print(f"checkpoint: {OUT_CKPT}")
    print("=" * 92)


if __name__ == "__main__":
    main()
