# -*- coding: utf-8 -*-
"""
Exp AC — DIVERGENCE PROBE: đo trực tiếp "exposure bias" của model nén trên Qwen3-0.6B THẬT.

Câu hỏi lượng hóa được: KL(teacher_FP || student) trên ngữ cảnh TỰ SINH (student sampling)
có LỚN HƠN và TĂNG THEO VỊ TRÍ so với trên ngữ cảnh VÀNG (teacher-forced) không?
  - Nếu CÓ (self-KL >> TF-KL, dốc lên): mất mạch lạc có thành phần compounding-drift lớn
    → rollout-KD (exp_ab) đánh đúng chỗ. Đây là điều kiện CẦN trước khi tốn tiền Modal.
  - Nếu KHÔNG (self ≈ TF, phẳng): lỗi là per-step capacity → rollout-KD sẽ vô ích,
    hủy các arm rollout (tiết kiệm ~$17), pivot sang curriculum/capacity.

Arms local ($0, máy B):  fp (sanity KL≈0) | int4 (control tốt ~Q4) | ptq24 (2:4 ternary
PTQ thô — vùng chết đã biết, chỉ xem HÌNH DÁNG curve).
Arm Modal (P0, ~$1):     --ckpt qat_gen4_n4.pt (student 1.56bpw ĐÃ KD — số quyết định).

Chạy local:  set OMP_NUM_THREADS=5 && python eval/lowbit_ptq/exp_ac_divergence_probe.py \
                 --arms fp,int4,ptq24
Smoke:       ... --smoke
"""
import argparse
import glob
import io
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)

OUT_DIR = os.environ.get("EXPR_OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root")
from exp_r_qat_lite import (EN_EVAL, LIN_PATHS, read_lines)  # noqa: E402

DEF_MODEL = r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*"
DEF_DVI = r"E:\Bit-Translate\cloud\qat_data\dev.vi"
BUCKETS = ((0, 8), (8, 16), (16, 32), (32, 48))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------- students PTQ nhanh (không train — chỉ để soi HÌNH DÁNG curve) ----------
def _to_f8(s):
    try:
        return s.to(torch.float8_e4m3fn).to(s.dtype)
    except Exception:
        return s


@torch.no_grad()
def fit_scale(Wm, act, G=64, iters=3):
    R, C = Wm.shape
    pad = (G - C % G) % G
    Wv = F.pad(Wm * act, (0, pad)).view(R, -1, G)
    Av = F.pad(act, (0, pad)).view(R, -1, G)
    cnt = Av.sum(2, keepdim=True).clamp(min=1)
    s = (Wv.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-6)
    for _ in range(iters):
        t = torch.round(Wv / s).clamp(-1, 1) * Av
        num = (Wv * t).sum(2, keepdim=True)
        den = (t * t).sum(2, keepdim=True).clamp(min=1e-8)
        s = (num / den).abs().clamp(min=1e-6)
    return s, Wv, Av, pad


@torch.no_grad()
def quantize_student(model, mode, xnorm=None):
    """int4: g32 absmax round. ptq24: Wanda 2:4 + ternary + scale Lloyd f8-g64 (PTQ thô)."""
    n = 0
    for name, m in model.named_modules():
        if not (isinstance(m, nn.Linear) and "layers." in name):
            continue
        W = m.weight.data
        if mode == "int4":
            R, C = W.shape
            G = 32
            pad = (G - C % G) % G
            Wv = F.pad(W, (0, pad)).view(R, -1, G)
            s = Wv.abs().amax(2, keepdim=True).clamp(min=1e-8) / 7.0
            Wq = (torch.round(Wv / s).clamp(-7, 7) * s).view(R, -1)[:, :C]
        else:  # ptq24
            xn = xnorm[name].clamp(min=1e-8)
            imp = W.abs() * xn[None, :]
            R, C = W.shape
            g4 = F.pad(imp, (0, (4 - C % 4) % 4)).view(R, -1, 4)
            kth = g4.kthvalue(3, dim=2, keepdim=True).values
            mask = (g4 >= kth).view(R, -1)[:, :C].float()
            s, Wv, Av, pad = fit_scale(W, mask)
            s = _to_f8(s)
            t = torch.round(Wv / s).clamp(-1, 1) * Av
            Wq = (t * s).view(R, -1)[:, :C]
        m.weight.data = Wq
        n += 1
    log(f"  student {mode}: lượng tử {n} linear")


@torch.no_grad()
def calib_xnorm(model, tok, lines, dev, seq=96):
    linears = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and "layers." in n]
    acc, hs = {n: None for n, _ in linears}, []

    def mk(nm):
        def h(mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            v = x.pow(2).sum(0)
            acc[nm] = v if acc[nm] is None else acc[nm] + v
        return h
    for n, m in linears:
        hs.append(m.register_forward_pre_hook(mk(n)))
    for s in lines:
        ids = tok(s, return_tensors="pt", truncation=True, max_length=seq).input_ids.to(dev)
        if ids.shape[1] >= 4:
            model(ids)
    for h in hs:
        h.remove()
    return {n: acc[n].sqrt() for n in acc}


# ---------------- lõi probe ----------------
def build_prompt_texts(dev_vi, n_vi=12, n_en=4, skip=100):
    """Ghép 3 dòng dev liên tiếp → đoạn đủ dài cho prompt 24 tok + gold 48 tok."""
    lines = read_lines(dev_vi, skip + 3 * n_vi)[skip:]
    texts = [" ".join(lines[i:i + 3]) for i in range(0, 3 * n_vi, 3)]
    texts += [" ".join(EN_EVAL[i:i + 3]) for i in range(0, 3 * n_en, 3)]
    return texts


@torch.no_grad()
def kl_bucketed(teacher, student, ids, am, plen, dev, chunk=4):
    """KL(T||S) full-vocab per-position trên vùng sau prompt → mean theo bucket."""
    sums = np.zeros(len(BUCKETS))
    cnts = np.zeros(len(BUCKETS))
    for i in range(0, ids.shape[0], chunk):
        idc, amc = ids[i:i + chunk].to(dev), am[i:i + chunk].to(dev)
        tl = teacher(idc, attention_mask=amc).logits.float()
        sl = student(idc, attention_mask=amc).logits.float()
        # logits tại vị trí p dự đoán token p+1 → KL tại p chấm "bước sinh token p+1"
        lt, ls = F.log_softmax(tl, -1), F.log_softmax(sl, -1)
        kl = (lt.exp() * (lt - ls)).sum(-1)                 # [b, T]
        for b, (lo, hi) in enumerate(BUCKETS):
            p0, p1 = plen - 1 + lo, plen - 1 + hi
            m = amc[:, p0:p1].float() * (amc[:, p0 + 1:p1 + 1].float()
                                         if p1 + 1 <= amc.shape[1] else 1.0)
            seg = kl[:, p0:p1]
            sums[b] += (seg * m).sum().item()
            cnts[b] += m.sum().item()
    return [round(s / max(c, 1), 4) for s, c in zip(sums, cnts)]


def loop_stats(tok, ids, plen):
    """distinct-2 + max lặp 4-gram liên tiếp trên vùng tự sinh."""
    d2s, reps = [], []
    for row in ids:
        seq = [t for t in row[plen:].tolist() if t is not None]
        if len(seq) < 8:
            continue
        bg = list(zip(seq, seq[1:]))
        d2s.append(len(set(bg)) / max(len(bg), 1))
        best = 1
        for k in range(len(seq) - 8):
            g = tuple(seq[k:k + 4])
            r = 1
            while k + 4 * (r + 1) <= len(seq) and tuple(seq[k + 4 * r:k + 4 * r + 4]) == g:
                r += 1
            best = max(best, r)
        reps.append(best)
    return round(float(np.mean(d2s)), 3), round(float(np.mean(reps)), 2)


def run_arm(arm, teacher, tok, texts, dev, args, xnorm=None, sd_path=""):
    from transformers import AutoModelForCausalLM
    mdir = glob.glob(args.model_glob)[0]
    log(f"=== ARM {arm} ===")
    if arm == "fp":
        student = teacher
    else:
        student = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32).to(dev).eval()
        if arm in ("int4", "ptq24"):
            quantize_student(student, arm, xnorm)
        elif arm == "ckpt":
            for blk in student.model.layers:      # ckpt bake exp_r có bias=True
                for sub, name in LIN_PATHS:
                    parent = getattr(blk, sub)
                    lin = getattr(parent, name)
                    nl = nn.Linear(lin.in_features, lin.out_features, bias=True)
                    nl.weight.data = lin.weight.data.clone()
                    nl.bias.data.zero_()
                    setattr(parent, name, nl.to(dev))
            sd = torch.load(sd_path, map_location="cpu", weights_only=False)
            state = {k: v.float() for k, v in sd["state_dict"].items()}
            missing, unexpected = student.load_state_dict(state, strict=False)
            assert not unexpected, f"unexpected: {unexpected[:5]}"
            log(f"  nạp ckpt {sd_path} | meta {sd.get('meta', {})}")

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True,
              max_length=args.plen)
    pids = enc.input_ids.to(dev)
    # (a) TỰ SINH: student sampling từ prompt
    torch.manual_seed(args.seed)
    gen = student.generate(pids, attention_mask=enc.attention_mask.to(dev),
                           max_new_tokens=args.gen_tokens, do_sample=True,
                           temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id)
    am_gen = torch.ones_like(gen)
    am_gen[:, :args.plen] = enc.attention_mask
    kl_self = kl_bucketed(teacher, student, gen, am_gen, args.plen, dev)
    d2, rep = loop_stats(tok, gen.cpu(), args.plen)
    # (b) TEACHER-FORCED: cùng prompt + tiếp diễn VÀNG (chính đoạn text nguồn)
    encf = tok(texts, return_tensors="pt", padding="max_length", truncation=True,
               max_length=args.plen + args.gen_tokens)
    kl_tf = kl_bucketed(teacher, student, encf.input_ids.to(dev),
                        encf.attention_mask.to(dev), args.plen, dev)
    ratio = [round(s / max(t, 1e-6), 2) for s, t in zip(kl_self, kl_tf)]
    slope = round(kl_self[-1] / max(kl_self[0], 1e-6), 2)
    out = {"kl_self_buckets": kl_self, "kl_tf_buckets": kl_tf, "self_over_tf": ratio,
           "self_slope_last_over_first": slope, "distinct2": d2, "max_4gram_rep": rep,
           "sample": tok.decode(gen[0][args.plen:args.plen + 40].cpu(),
                                skip_special_tokens=True)}
    log(f"  KL self  {kl_self}")
    log(f"  KL tf    {kl_tf}")
    log(f"  self/tf  {ratio} | slope(self) {slope} | distinct2 {d2} | rep4 {rep}")
    log(f"  mẫu: {out['sample']!r}")
    if arm != "fp":
        del student
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-glob", default=DEF_MODEL)
    ap.add_argument("--dev-vi", default=DEF_DVI)
    ap.add_argument("--arms", default="fp,int4,ptq24",
                    help="fp|int4|ptq24|ckpt (ckpt cần --ckpt)")
    ap.add_argument("--ckpt", default="", help="ckpt bake exp_r (vd /vol/out/qat_gen4_n4.pt)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--plen", type=int, default=24)
    ap.add_argument("--gen-tokens", type=int, default=48)
    ap.add_argument("--n-vi", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_vi, args.gen_tokens = 2, 12
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(5)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    mdir = glob.glob(args.model_glob)[0]
    tok = AutoTokenizer.from_pretrained(mdir)
    log(f"nạp teacher FP32 ({dev})...")
    teacher = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32).to(dev).eval()
    texts = build_prompt_texts(args.dev_vi, n_vi=args.n_vi, n_en=(1 if args.smoke else 4))
    log(f"probe: {len(texts)} prompt × sinh {args.gen_tokens} tok, bucket {BUCKETS}")

    arms = args.arms.split(",")
    xnorm = None
    if "ptq24" in arms:
        log("calib xnorm (24 câu dev) cho Wanda...")
        xnorm = calib_xnorm(teacher, tok, read_lines(args.dev_vi, 24), dev)

    res = {}
    for arm in arms:
        res[arm] = run_arm(arm, teacher, tok, texts, dev, args, xnorm=xnorm,
                           sd_path=args.ckpt)
        if arm == "fp":
            assert max(res["fp"]["kl_self_buckets"]) < 1e-3, "sanity FAIL: KL(T||T) != 0"
    res["_meta"] = {"gen_tokens": args.gen_tokens, "plen": args.plen,
                    "n_prompts": len(texts), "seed": args.seed,
                    "doc": "self_over_tf tăng theo bucket = chữ ký exposure-bias (H-A); "
                           "phẳng = lỗi per-step capacity, rollout-KD sẽ vô ích"}
    p = os.path.join(OUT_DIR, "exp_ac_results.json")
    old = {}
    if os.path.exists(p):
        try:
            with io.open(p, "r", encoding="utf-8") as f:
                old = json.load(f)
        except Exception:
            old = {}
    key = args.tag or f"probe[{args.arms}{',ckpt' if args.ckpt else ''}]"
    old[key] = res
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=2)
    log(f"đã ghi {p} (key={key})")


if __name__ == "__main__":
    main()
