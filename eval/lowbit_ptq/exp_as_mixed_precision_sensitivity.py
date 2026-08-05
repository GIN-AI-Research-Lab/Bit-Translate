# -*- coding: utf-8 -*-
"""
Exp AS — MIXED-PRECISION theo ĐỘ NHẠY từng ma trận: thay vì ép ĐỀU 196 ma trận về ternary
2:4 (như exp_ap), đo ma trận nào "chịu đau" nhiều nhất khi ternary hóa, nâng riêng chúng lên
int4-g32 (đắt hơn ~3x bit nhưng chính xác hơn nhiều), giữ phần còn lại ở ternary rẻ.

Khác exp_g_mixed_precision_curve.py cũ (chỉ TF reconstruction, không geo6, không đúng recipe
SEQUENTIAL/Wanda-2:4 đã proven): dùng ĐÚNG mask Wanda 2:4 + Lloyd-scale-g64 (giống exp_ap) làm
baseline ternary, đo geo6 đầy đủ 6 miền, và sweep tỉ lệ % ma trận nâng lên int4 để vẽ đường cong
bpw-vs-geo6 thật trên CHÍNH model đã đóng gói.

BƯỚC 0 (thăm dò rẻ, TF-only KHÔNG sequential, ~2-3 phút) — nếu đường cong cho tín hiệu tốt
(geo6 cải thiện nhiều mà bpw tăng ít) mới đáng đầu tư chạy lại SEQUENTIAL đầy đủ (~30 phút)
cho cấu hình mixed tốt nhất.
"""
import glob
import io
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
torch.set_num_threads(5)

from exp_r_qat_lite import (EN_EVAL, CODE_EVAL, ZH_EVAL, MATH_EVAL,  # noqa: E402
                             CAL_EN, CAL_CODE, CAL_ZH, CAL_MATH, CAL_CHAT, to_f8)

MODEL_DIR = glob.glob(r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*")[0]
DEV_VI = r"D:\Bit-Translate-data\clean_v7g\dev.vi"
DEV_JA = r"D:\Bit-Translate-data\clean_v7g\dev.ja"
OUT_JSON = r"e:\Bit-Translate\eval\lowbit_ptq\exp_as_results.json"
GROUP = 64
N_NM, M_NM = 2, 4
N_CALIB = 60
N_EVAL = 16
MAX_TOK = 96
UPGRADE_FRACS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]  # % ma trận (theo sai số cao nhất) nâng lên int4


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


def wanda_nm_mask(W, xnorm, N=N_NM, M=M_NM):
    R, C = W.shape
    imp = W.abs() * xnorm[None, :].clamp(min=1e-8)
    pad = (M - C % M) % M
    A = F.pad(imp, (0, pad)) if pad else imp
    g = A.view(R, -1, M)
    kth = g.kthvalue(M - N + 1, dim=2, keepdim=True).values
    return (g >= kth).view(R, -1)[:, :C]


def int4_g32(W, group=32):
    """int4 per-group-32 absmax, scale f8-grid (đóng gói được, giống lưới ternary)."""
    Wg, C = grp_pad(W, group)
    s = (Wg.abs().amax(2, keepdim=True) / 7.0).clamp(min=1e-8)
    s = to_f8(s)
    t = torch.round(Wg / s).clamp(-8, 7)
    return (t * s).view(W.shape[0], -1)[:, :C]


def bpw_ternary24(group=GROUP):
    payload = (N_NM / M_NM) * math.log2(3)
    mask_cost = math.log2(math.comb(M_NM, N_NM)) / M_NM
    scale_cost = 16.0 / group
    return payload + mask_cost + scale_cost


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


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"Nạp Qwen3-0.6B FP32 từ {MODEL_DIR}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    linears = [(n, m) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and "layers." in n]
    log(f"{len(linears)} ma trận. bpw ternary2:4={bpw_ternary24():.3f}  bpw int4-g32={bpw_int4():.3f}")

    vi = read_lines(DEV_VI, N_CALIB // 2)
    ja = read_lines(DEV_JA, N_CALIB // 2)
    calib = [x for pr in zip(vi, ja) for x in pr] + CAL_EN + CAL_CODE + CAL_ZH + CAL_MATH + CAL_CHAT
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
    results = {"fp32": fp_geo6}

    log("Thu activation calibration (mix 6 miền)...")
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
            if ids.shape[1] < 4:
                continue
            model(ids)
    for h in handles:
        h.remove()
    xnorm = {}
    for n in acts:
        X = torch.cat(acts[n], 0)
        xnorm[n] = X.float().pow(2).mean(0).sqrt()
        acts[n] = None
    orig = {n: m.weight.data.clone() for n, m in linears}

    # ---- Bước 1: tính SAI SỐ TERNARY từng ma trận riêng (để xếp hạng độ nhạy) ----
    log("Đo sai số ternary 2:4 từng ma trận (xếp hạng độ nhạy)...")
    sens = []  # (relerr, name)
    masks, Wtern, Wint4 = {}, {}, {}
    for n, m in linears:
        mask = wanda_nm_mask(orig[n], xnorm[n])
        masks[n] = mask
        Wt = lloyd_masked_ternary(orig[n], mask)
        Wtern[n] = Wt
        Wint4[n] = int4_g32(orig[n])
        relerr = float((Wt - orig[n]).pow(2).sum() / orig[n].pow(2).sum().clamp(min=1e-12))
        sens.append((relerr, n))
    sens.sort(reverse=True)  # sai số cao nhất trước -> ưu tiên nâng lên int4 trước
    log("Top-5 ma trận NHẠY NHẤT (sai số ternary cao nhất): " +
        ", ".join(f"{n}({e*100:.1f}%)" for e, n in sens[:5]))
    log("Top-5 ma trận CHỊU TERNARY TỐT NHẤT: " +
        ", ".join(f"{n}({e*100:.1f}%)" for e, n in sens[-5:]))

    total_w = sum(orig[n].numel() for n, _ in linears)

    # ---- Bước 2: sweep tỉ lệ % nâng lên int4 (TF-only, không sequential — thăm dò rẻ) ----
    for frac in UPGRADE_FRACS:
        n_upgrade = int(round(frac * len(sens)))
        upgrade_names = set(n for _, n in sens[:n_upgrade])
        bits_used = 0.0
        for n, m in linears:
            if n in upgrade_names:
                m.weight.data = Wint4[n]
                bits_used += orig[n].numel() * bpw_int4()
            else:
                m.weight.data = Wtern[n]
                bits_used += orig[n].numel() * bpw_ternary24()
        bpw_avg = bits_used / total_w
        g = geo6(model)
        tag = f"upgrade_{int(frac*100)}pct_int4"
        log(f"  {tag}: bpw={bpw_avg:.3f}  geo6={g['geo6']:.1f}  "
            f"(vi {g['vi']:.1f} ja {g['ja']:.1f} en {g['en']:.1f} code {g['code']:.1f} "
            f"zh {g['zh']:.1f} math {g['math']:.1f})")
        results[tag] = {**g, "bpw": bpw_avg, "n_upgraded": n_upgrade, "n_total": len(sens)}
        with io.open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with torch.no_grad():
            for n, m in linears:
                m.weight.data = orig[n].clone()

    print("\n" + "=" * 88)
    print("EXP AS — MIXED-PRECISION theo độ nhạy (ternary 2:4 + int4-g32 cho lớp nhạy nhất)")
    print("=" * 88)
    print(f"{'config':26s}{'bpw':>8s}{'geo6':>12s}{'vi':>10s}{'code':>10s}{'zh':>10s}")
    print("-" * 88)
    for k, v in results.items():
        if k == "fp32":
            continue
        print(f"{k:26s}{v['bpw']:8.3f}{v['geo6']:12.1f}{v['vi']:10.1f}{v['code']:10.1f}{v['zh']:10.1f}")
    print(f"{'fp32':26s}{16.0:8.2f}{fp_geo6['geo6']:12.1f}{fp_geo6['vi']:10.1f}"
          f"{fp_geo6['code']:10.1f}{fp_geo6['zh']:10.1f}")
    print("=" * 88)
    print("Mốc so sánh: exp_ap ternary2:4 THUẦN + SEQUENTIAL (không mixed) = bpw 1.689 geo6 634.9")
    print("=" * 88)


if __name__ == "__main__":
    main()
