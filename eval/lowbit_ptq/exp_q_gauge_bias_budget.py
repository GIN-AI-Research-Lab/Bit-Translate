# -*- coding: utf-8 -*-
"""
Exp Q — GAUGE PRECONDITIONING + BIAS HỌC ĐƯỢC + NGÂN SÁCH THEO BLOCK
(trên nền công thức exp_n SẠCH đã thắng ablation; các thứ bị bác — refresh/relMSE — không có ở đây)

ĐÒN-1 GAUGE (0 bit, tương đương CHÍNH XÁC về toán — có sanity check tự động):
  - up↔down: scale hàng i của up bởi d_i, chia cột i của down bởi d_i. FFN output không đổi.
    Chọn d_i = absmean(W_down[:,i]) (chuẩn hóa geomean=1) -> các cột down đều biên độ
    -> group-64 của down hết chênh lệch nội nhóm -> ternary fit tốt hơn.
  - v↔o: tương tự cho cặp value/output (GQA: 1 hàng v phục vụ 2 block cột của o -> m = geomean 2 cột).
  - KHÔNG gauge q↔k (QK-norm của Qwen3 phá tương đương) và gate (SiLU phi tuyến).
  - Sau gauge: đo lại PPL FP — phải BẰNG baseline (|Δ|<1%%), sai thì abort.
ĐÒN-2 BIAS: mỗi linear thêm bias FP học được (Qwen vốn không bias; ~0.006 bpw) — hấp thụ
  trung bình lệch của lỗi quantize per-output-channel, mịn hơn adapter per-block đã bị bác.
ĐÒN-3 NGÂN SÁCH: block khó (2,26,27 theo trace exp_k) x2 bước; block giữa trơ x0.75 — tổng ~không đổi.

Chạy: python exp_q...py [config_name]  (mặc định N3 1.56bpw để so mốc 536.3)
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
OUT_JSON = r"e:\Bit-Translate\eval\lowbit_ptq\exp_q_results.json"
PASS_STEPS = (70, 30)
SEQ_LR = 1e-3
SCALE_LR = 5e-3
BIAS_LR = 5e-4
SEQ_BATCH = 6
N_CALIB = 60
N_EVAL = 16
MAX_TOK = 96
HARD_BLOCKS = {2: 2.0, 26: 2.0, 27: 2.0, 0: 1.5, 1: 1.5, 3: 1.25}
EASY_RANGE = set(range(8, 21))  # x0.75
LIN_PATHS = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
             ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]
NORM_PATHS = [("", "input_layernorm"), ("", "post_attention_layernorm"),
              ("self_attn", "q_norm"), ("self_attn", "k_norm")]

KNOWN = {
    "N3_t24_f8g64_3fix_1.56bpw": ((2, 4), "64", "f8", 1.56, 536.3),
    "O1_t14_f8g64_3fix_1.02bpw": ((1, 4), "64", "f8", 1.02, 766.9),
    "O2_t18_f8g64_3fix_0.70bpw": ((1, 8), "64", "f8", 0.70, 1475.9),
}


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
    """công thức exp_n + ĐÒN-2 bias học được."""

    def __init__(self, lin, mask, sgroup, sdtype):
        super().__init__()
        W0 = lin.weight.data
        self.Wfp = nn.Parameter(W0.clone())
        self.bias = nn.Parameter(torch.zeros(W0.shape[0]))
        self.sgroup, self.sdtype = sgroup, sdtype
        R, C = W0.shape
        self.R, self.C = R, C
        self.G = int(sgroup)
        self.pad = (self.G - C % self.G) % self.G
        self.register_buffer("maskf", (torch.ones_like(W0) if mask is None else mask.float()))
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
        pad = self.pad
        Wp = F.pad(W, (0, pad)) if pad else W
        Mp = F.pad(m, (0, pad)) if pad else m
        return Wp.view(self.R, -1, self.G), Mp.view(self.R, -1, self.G)

    def quant(self):
        s = F.softplus(self.raw_s).clamp(min=4e-3)
        s_q = s + (SCALE_Q[self.sdtype](s) - s).detach()
        s_q = s_q.clamp(min=1e-6)
        Wv, Mv = self._views(self.Wfp)
        t = (torch.round((Wv * Mv) / s_q.detach()).clamp(-1, 1) * Mv).detach()
        return (t * s_q).reshape(self.R, -1)[:, :self.C]

    def forward(self, x):
        q = self.quant()
        Wq = q + self.Wfp - self.Wfp.detach()
        return F.linear(x, Wq, self.bias)


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


@torch.no_grad()
def apply_gauge(model):
    """ĐÒN-1: up↔down và v↔o. Trả về thống kê để log."""
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    hd = cfg.head_dim
    rep = n_q // n_kv
    stats = []
    for b, blk in enumerate(model.model.layers):
        # --- up <-> down ---
        Wd = blk.mlp.down_proj.weight.data          # [hidden, inter]
        Wu = blk.mlp.up_proj.weight.data            # [inter, hidden]
        c = Wd.abs().mean(dim=0).clamp(min=1e-8)    # absmean từng cột down [inter]
        d = c / torch.exp(torch.log(c).mean())      # geomean = 1
        Wu.mul_(d[:, None])
        Wd.div_(d[None, :])
        # --- v <-> o ---
        Wv = blk.self_attn.v_proj.weight.data       # [n_kv*hd, hidden]
        Wo = blk.self_attn.o_proj.weight.data       # [hidden, n_q*hd]
        co = Wo.abs().mean(dim=0).clamp(min=1e-8)   # [n_q*hd]
        co_h = co.view(n_q, hd)
        m = torch.ones(n_kv * hd)
        for kv in range(n_kv):
            qs = [kv * rep + r for r in range(rep)]
            mk = torch.exp(sum(torch.log(co_h[q]) for q in qs) / rep)   # geomean qua rep block
            m[kv * hd:(kv + 1) * hd] = mk
        m = m / torch.exp(torch.log(m).mean())
        Wv.mul_(m[:, None])
        for kv in range(n_kv):
            for r in range(rep):
                q = kv * rep + r
                Wo[:, q * hd:(q + 1) * hd] /= m[kv * hd:(kv + 1) * hd][None, :]
        stats.append((float(d.max() / d.min()), float(m.max() / m.min())))
    return stats


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cname = sys.argv[1] if len(sys.argv) > 1 else "N3_t24_f8g64_3fix_1.56bpw"
    nm, sgroup, sdtype, bpw_val, base_ppl = KNOWN[cname]
    log(f"Config: {cname} (mốc exp_n: vi {base_ppl}); gauge + bias(+0.006bpw) + budget")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()

    vi = read_lines(DEV_VI, N_CALIB // 2)
    ja = read_lines(DEV_JA, N_CALIB // 2)
    calib = [x for pr in zip(vi, ja) for x in pr]
    eval_vi = read_lines(DEV_VI, 400)[-N_EVAL:]
    eval_ja = read_lines(DEV_JA, 400)[-N_EVAL:]
    ppl_fp = (eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja))
    log(f"FP32 trước gauge: vi {ppl_fp[0]:.2f} / ja {ppl_fp[1]:.2f}")

    # ===== ĐÒN-1: gauge + sanity check =====
    stats = apply_gauge(model)
    dmax = max(s[0] for s in stats)
    log(f"Gauge xong (max spread d={dmax:.1f}x). Sanity check FP...")
    ppl_g = (eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja))
    log(f"FP32 SAU gauge:   vi {ppl_g[0]:.2f} / ja {ppl_g[1]:.2f}")
    if abs(ppl_g[0] - ppl_fp[0]) / ppl_fp[0] > 0.01:
        log("!!! GAUGE PHÁ MODEL — ABORT")
        sys.exit(1)
    log("Sanity OK — gauge là tương đương chính xác.")

    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and "layers." in n]
    # xnorm thu SAU gauge (activation của down/o đã đổi scale)
    xn_acc = {n: None for n, _ in linears}
    handles = []
    def mk(nm_):
        def h(mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            v = x.pow(2).sum(0)
            xn_acc[nm_] = v if xn_acc[nm_] is None else xn_acc[nm_] + v
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

    layers = model.model.layers
    NB = len(layers)
    log("Quỹ đạo FP (sau gauge)...")
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

    wrapped = []
    for b, blk in enumerate(layers):
        for sub, name in LIN_PATHS:
            parent = getattr(blk, sub)
            lin = getattr(parent, name)
            key = f"model.layers.{b}.{sub}.{name}"
            mask = wanda_nm_mask(orig_lin[key], xnorm[key], nm[0], nm[1])
            w = LearnQLinear(lin, mask, sgroup, sdtype)
            setattr(parent, name, w)
            wrapped.append((b, parent, name, lin, w))

    t0 = time.time()
    for p_idx, base_steps in enumerate(PASS_STEPS):
        H_q = [H_fp[0][s].clone() for s in range(NC)]
        for b, blk in enumerate(layers):
            # ĐÒN-3: ngân sách theo block
            f = HARD_BLOCKS.get(b, 0.75 if b in EASY_RANGE else 1.0)
            steps = max(10, int(round(base_steps * f)))
            mods = [w for (bb, _, _, _, w) in wrapped if bb == b]
            norm_ws = []
            for sub, name in NORM_PATHS:
                parent = getattr(blk, sub) if sub else blk
                norm_ws.append(getattr(parent, name).weight)
            for nw_ in norm_ws:
                nw_.requires_grad_(True)
            opt = torch.optim.Adam([
                {"params": [w.Wfp for w in mods], "lr": SEQ_LR},
                {"params": [w.raw_s for w in mods], "lr": SCALE_LR},
                {"params": [w.bias for w in mods], "lr": BIAS_LR},
                {"params": norm_ws, "lr": 5e-4},
            ])

            def eval_block():
                with torch.no_grad():
                    v = 0.0
                    for s in eval_sub:
                        v += F.mse_loss(blk(H_q[s], position_embeddings=ropes[s]), H_fp[b + 1][s]).item()
                return v

            best = eval_block()  # guard step-0 (vá-1, đã kiểm chứng an toàn)
            best_state = ([w.Wfp.detach().clone() for w in mods],
                          [w.raw_s.detach().clone() for w in mods],
                          [w.bias.detach().clone() for w in mods],
                          [nw_.detach().clone() for nw_ in norm_ws])
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
                        best_state = ([w.Wfp.detach().clone() for w in mods],
                                      [w.raw_s.detach().clone() for w in mods],
                                      [w.bias.detach().clone() for w in mods],
                                      [nw_.detach().clone() for nw_ in norm_ws])
            with torch.no_grad():
                for w, Wb, Sb, Bb in zip(mods, best_state[0], best_state[1], best_state[2]):
                    w.Wfp.data = Wb
                    w.raw_s.data = Sb
                    w.bias.data = Bb
                for nw_, nb_ in zip(norm_ws, best_state[3]):
                    nw_.data = nb_
            for nw_ in norm_ws:
                nw_.requires_grad_(False)
            with torch.no_grad():
                for s in range(NC):
                    H_q[s] = blk(H_q[s], position_embeddings=ropes[s]).detach()
        log(f"    pass {p_idx+1} xong ({time.time()-t0:.0f}s)")

    # bake: Linear mới có bias
    with torch.no_grad():
        for b, parent, name, lin, w in wrapped:
            new_lin = nn.Linear(w.C, w.R, bias=True)
            new_lin.weight.data = w.quant().detach()
            new_lin.bias.data = w.bias.detach()
            setattr(parent, name, new_lin)
    pv, pj = eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja)
    log(f"=> EXP Q [{cname}]: PPL vi {pv:.1f} / ja {pj:.1f}  (mốc exp_n: {base_ppl})")
    out = {}
    if os.path.exists(OUT_JSON):
        try:
            with io.open(OUT_JSON, "r", encoding="utf-8") as f:
                out = json.load(f)
        except Exception:
            out = {}
    out["fp32"] = {"ppl_vi": ppl_fp[0], "ppl_ja": ppl_fp[1]}
    out[f"exp_q[{cname}]"] = {"bpw": bpw_val + 0.006, "ppl_vi": pv, "ppl_ja": pj, "base_ppl_vi": base_ppl}
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("=" * 72)
    print(f"EXP Q — gauge + bias + budget | {cname}")
    print(f"  mốc exp_n : vi {base_ppl:10.1f}")
    print(f"  exp_q     : vi {pv:10.1f}  ja {pj:10.1f}  (fp32 {ppl_fp[0]:.1f}/{ppl_fp[1]:.1f})")
    print("=" * 72)


if __name__ == "__main__":
    main()
