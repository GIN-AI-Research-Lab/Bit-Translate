# -*- coding: utf-8 -*-
"""
Exp AT — MIXED-PRECISION + SEQUENTIAL THẬT: kết hợp phát hiện của exp_as (xếp hạng độ nhạy
từng ma trận, top-20% nhạy nhất là down_proj tầng sâu) VỚI đòn bẩy mạnh nhất lab (SEQUENTIAL/
BRECQ-lite của exp_ap) — câu hỏi CHƯA ai trả lời: mixed-precision có còn lợi khi đã có
SEQUENTIAL, hay SEQUENTIAL tự nó đã "cứu" đủ nên mixed không cần thiết nữa?

Thiết kế: 20% ma trận NHẠY NHẤT (theo sai số ternary TF đo ở exp_as, chủ yếu down_proj tầng
sâu 16-27) dùng int4-g32 (4,5bpw); 80% còn lại dùng ternary 2:4 (1,689bpw) — CẢ HAI loại đều
được tinh chỉnh qua CÙNG vòng lặp SEQUENTIAL 28-block (mỗi block tối ưu khớp output FP, đúng
kỹ thuật exp_ap/exp_k). bpw trung bình ~2,31 (so với 1,689 ternary thuần).

So trực tiếp: exp_ap ternary-thuần+SEQUENTIAL (geo6 634,9 @1,689bpw) vs kết quả ở đây
(mixed 20%+SEQUENTIAL, ~2,31bpw) — nếu KHÔNG cải thiện đáng kể dù tốn thêm ~37% bit, mixed-
precision không đáng đầu tư thêm cho model 0.6B (SEQUENTIAL đã là đòn chính, mixed chỉ có
ích khi CHƯA có SEQUENTIAL).
"""
import gc
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
OUT_JSON = r"e:\Bit-Translate\eval\lowbit_ptq\exp_at_results.json"
OUT_CKPT = r"D:\Bit-Translate-data\qat_ckpts\mixed20_seq_baked.pt"
GROUP = 64
N_NM, M_NM = 2, 4
UPGRADE_FRAC = 0.20
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
    Wg, C = grp_pad(W, group)
    Mg, _ = grp_pad(mask.float(), group)
    Wm = Wg * Mg
    cnt = Mg.sum(2, keepdim=True).clamp(min=1)
    s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-8)
    for _ in range(3):
        t = torch.round(Wm / s).clamp(-1, 1) * Mg
        num = (Wm * t).sum(2, keepdim=True)
        den = (t * t).sum(2, keepdim=True).clamp(min=1e-8)
        s = (num / den).abs().clamp(min=1e-8)
    s = to_f8(s)
    t = torch.round(Wm / s).clamp(-1, 1) * Mg
    return (t * s).view(W.shape[0], -1)[:, :C]


def int4_g32(W, group=32):
    Wg, C = grp_pad(W, group)
    s = (Wg.abs().amax(2, keepdim=True) / 7.0).clamp(min=1e-8)
    s = to_f8(s)
    t = torch.round(Wg / s).clamp(-8, 7)
    return (t * s).view(W.shape[0], -1)[:, :C]


def wanda_nm_mask(W, xnorm, N=N_NM, M=M_NM):
    R, C = W.shape
    imp = W.abs() * xnorm[None, :].clamp(min=1e-8)
    pad = (M - C % M) % M
    A = F.pad(imp, (0, pad)) if pad else imp
    g = A.view(R, -1, M)
    kth = g.kthvalue(M - N + 1, dim=2, keepdim=True).values
    return (g >= kth).view(R, -1)[:, :C]


def bpw_ternary24(group=GROUP):
    payload = (N_NM / M_NM) * math.log2(3)
    mask_cost = math.log2(math.comb(M_NM, N_NM)) / M_NM
    return payload + mask_cost + 16.0 / group


def bpw_int4(group=32):
    return 4.0 + 16.0 / group


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


class TernarySTELinear(nn.Module):
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


class Int4STELinear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.Wfp = nn.Parameter(lin.weight.data.clone())

    def q(self, W=None):
        with torch.no_grad():
            return int4_g32(self.Wfp if W is None else W)

    def forward(self, x):
        q = self.q()
        Wq = self.Wfp + (q - self.Wfp).detach()
        return F.linear(x, Wq)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"Nạp Qwen3-0.6B FP32 từ {MODEL_DIR}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    linears = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and "layers." in n]
    log(f"{len(linears)} ma trận. bpw ternary2:4={bpw_ternary24():.3f}  int4-g32={bpw_int4():.3f}")

    vi = read_lines(DEV_VI, N_CALIB // 2)
    ja = read_lines(DEV_JA, N_CALIB // 2)
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
        return {"vi": v, "ja": j, "en": e, "code": c, "zh": z, "math": m,
                "geo6": (v * j * e * c * z * m) ** (1 / 6)}

    fp_geo6 = geo6(model)
    log(f"FP32 geo6: {json.dumps({k: round(x,1) for k,x in fp_geo6.items()})}")
    results = {"fp32": fp_geo6, "config": {"upgrade_frac": UPGRADE_FRAC, "group": GROUP,
                                            "seq_steps": SEQ_STEPS}}

    log("Thu activation calibration (mix 6 miền)...")
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

    # ---- xếp hạng độ nhạy (TF ternary error), chọn top-20% nâng int4 (đúng exp_as) ----
    log("Xếp hạng độ nhạy từng ma trận...")
    sens = []
    for n, m in linears:
        Wt = lloyd_masked_ternary(orig[n], masks[n])
        relerr = float((Wt - orig[n]).pow(2).sum() / orig[n].pow(2).sum().clamp(min=1e-12))
        sens.append((relerr, n))
    sens.sort(reverse=True)
    n_upgrade = int(round(UPGRADE_FRAC * len(sens)))
    int4_names = set(n for _, n in sens[:n_upgrade])
    log(f"Nâng {n_upgrade}/{len(sens)} ma trận lên int4: " +
        ", ".join(sorted(int4_names))[:300] + " ...")

    bpw_total = sum((bpw_int4() if n in int4_names else bpw_ternary24()) * orig[n].numel()
                    for n, _ in linears) / sum(orig[n].numel() for n, _ in linears)
    log(f"bpw trung bình thực tế: {bpw_total:.3f}")
    results["config"]["bpw_avg"] = bpw_total
    results["config"]["n_int4"] = n_upgrade

    # ---- SEQUENTIAL block-wise, MIXED precision per-tensor ----
    log("=== SEQUENTIAL mixed-precision — bake toàn bộ 28 block ===")
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
            if key in int4_names:
                sm = Int4STELinear(lin)
            else:
                sm = TernarySTELinear(lin, masks[key])
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
        gc.collect()

    baked_geo6 = geo6(model)
    log(f"BAKED geo6: {json.dumps({k: round(x,1) for k,x in baked_geo6.items()})}")
    results["baked_mixed20_seq"] = baked_geo6
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(OUT_CKPT), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "meta": {"recipe": "mixed20_int4_ternary24_sequential", "bpw_avg": bpw_total,
                         "int4_layers": sorted(int4_names), "geo6": baked_geo6}}, OUT_CKPT)
    log(f"Đã lưu checkpoint -> {OUT_CKPT}")

    print("\n" + "=" * 92)
    print(f"EXP AT — MIXED 20% int4 + 80% ternary2:4, CẢ HAI qua SEQUENTIAL (bpw={bpw_total:.3f})")
    print("=" * 92)
    print(f"{'miền':10s}{'FP32':>14s}{'mixed+SEQ':>14s}{'x FP':>10s}")
    for k in ("vi", "ja", "en", "code", "zh", "math", "geo6"):
        print(f"{k:10s}{fp_geo6[k]:14.2f}{baked_geo6[k]:14.2f}{baked_geo6[k]/fp_geo6[k]:10.2f}")
    print(f"\nSo mốc exp_ap (ternary thuần + SEQUENTIAL, bpw=1.689): geo6=634.9")
    print(f"Mixed ở đây (bpw={bpw_total:.3f}, +{(bpw_total/1.689-1)*100:.0f}% bit): geo6={baked_geo6['geo6']:.1f}")
    print("=" * 92)


if __name__ == "__main__":
    main()
