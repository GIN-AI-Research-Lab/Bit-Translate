# -*- coding: utf-8 -*-
"""
Exp P — VÁ 4 SƠ HỞ CODE + 2 BƯỚC CONVERT MỚI (trên config tốt nhất <=1.6bpw tự chọn từ JSON):

  VÁ-1 regression guard: eval step-0 làm mốc best (pass 2 không bao giờ bake tệ hơn pass 1).
  VÁ-2 relative-MSE: loss từng câu chia năng lượng target -> vi/ja cân gradient.
  VÁ-3 mask refresh đầu pass 2: chọn lại N:M theo |Wfp hiện tại| x xnorm.
  MỚI-1 adapter affine per-channel (scale+bias FP) sau MỖI block: +0.03 bpw, hấp thụ trôi dạt
        hệ thống giữa tầng (bias là năng lực RMSNorm không có).
  MỚI-2 tail KL-polish: sau 2 pass, mở khóa 4 block cuối + model.norm + adapter đuôi, tối ưu
        KL(logits_FP || logits_q) — đánh thẳng cái PPL đo, thay vì proxy MSE hidden.

So sánh trực tiếp với kết quả cùng-config trong exp_n_results.json (không fix mới nào).
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
IN_JSON = r"e:\Bit-Translate\eval\lowbit_ptq\exp_n_results.json"
OUT_JSON = r"e:\Bit-Translate\eval\lowbit_ptq\exp_p_results.json"
PASS_STEPS = (70, 30)
SEQ_LR = 1e-3
SCALE_LR = 5e-3
ADPT_LR = 5e-4
SEQ_BATCH = 6
POLISH_STEPS = 40
POLISH_TAIL = 4          # số block cuối mở khóa cho KL-polish
POLISH_BATCH = 3
N_CALIB = 60
N_EVAL = 16
MAX_TOK = 96
RMS_EPS = 1e-6
LIN_PATHS = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
             ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]
NORM_PATHS = [("", "input_layernorm"), ("", "post_attention_layernorm"),
              ("self_attn", "q_norm"), ("self_attn", "k_norm")]

# config ứng viên (khớp tên trong exp_n_results.json)
KNOWN = {
    "N3_t24_f8g64_3fix_1.56bpw": ((2, 4), "64", "f8", 1.56),
    "O1_t14_f8g64_3fix_1.02bpw": ((1, 4), "64", "f8", 1.02),
    "O2_t18_f8g64_3fix_0.70bpw": ((1, 8), "64", "f8", 0.70),
    "O3_t110_f8g64_3fix_0.62bpw": ((1, 10), "64", "f8", 0.62),
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
    def __init__(self, lin, mask, sgroup, sdtype):
        super().__init__()
        W0 = lin.weight.data
        self.Wfp = nn.Parameter(W0.clone())
        self.sgroup, self.sdtype = sgroup, sdtype
        R, C = W0.shape
        self.R, self.C = R, C
        self.G = int(sgroup)
        self.pad = (self.G - C % self.G) % self.G
        self.register_buffer("maskf", (torch.ones_like(W0) if mask is None else mask.float()))
        with torch.no_grad():
            s = self._lloyd_init(W0)
        self.raw_s = nn.Parameter(inv_softplus(s))

    def _views(self, W):
        m = self.maskf
        pad = self.pad
        Wp = F.pad(W, (0, pad)) if pad else W
        Mp = F.pad(m, (0, pad)) if pad else m
        return Wp.view(self.R, -1, self.G), Mp.view(self.R, -1, self.G)

    def _lloyd_init(self, W0):
        Wv, Mv = self._views(W0)
        Wm = Wv * Mv
        cnt = Mv.sum(2, keepdim=True).clamp(min=1)
        s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-6)
        for _ in range(3):
            t = torch.round(Wm / s).clamp(-1, 1) * Mv
            num = (Wm * t).sum(2, keepdim=True)
            den = (t * t).sum(2, keepdim=True).clamp(min=1e-8)
            s = (num / den).abs().clamp(min=1e-6)
        return s

    def refresh_mask(self, xnorm, N, M):
        with torch.no_grad():
            self.maskf.copy_(wanda_nm_mask(self.Wfp.data, xnorm, N, M).float())

    def quant(self):
        s = F.softplus(self.raw_s).clamp(min=4e-3)
        s_q = s + (SCALE_Q[self.sdtype](s) - s).detach()
        s_q = s_q.clamp(min=1e-6)
        Wv, Mv = self._views(self.Wfp)
        t = (torch.round((Wv * Mv) / s_q.detach()).clamp(-1, 1) * Mv).detach()
        q = (t * s_q).reshape(self.R, -1)[:, :self.C]
        return q

    def forward(self, x):
        q = self.quant()
        Wq = q + self.Wfp - self.Wfp.detach()
        return F.linear(x, Wq)

    @torch.no_grad()
    def bake(self):
        return self.quant().detach().clone()


class BlockAdapter(nn.Module):
    """MỚI-1: affine per-channel sau block (scale init 1, bias init 0)."""

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, h):
        return h * self.g + self.b


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
    # chọn config: argv[1] (nếu có) hoặc tốt nhất <=1.6bpw từ kết quả hiện có
    prev = {}
    if os.path.exists(IN_JSON):
        with io.open(IN_JSON, "r", encoding="utf-8") as f:
            prev = json.load(f)
    pick_name, pick_ppl = "N3_t24_f8g64_3fix_1.56bpw", float("inf")
    force = sys.argv[1] if len(sys.argv) > 1 else None
    if force and force in KNOWN:
        pick_name = force
        pick_ppl = prev.get(force, {}).get("ppl_vi", float("inf"))
    else:
        for k, v in prev.items():
            if k in KNOWN and v.get("ppl_vi") == v.get("ppl_vi") and v.get("bpw", 9) <= 1.6:
                if v["ppl_vi"] < pick_ppl:
                    pick_name, pick_ppl = k, v["ppl_vi"]
    nm, sgroup, sdtype, bpw_val = KNOWN[pick_name]
    # cờ ablation: NOADPT (bỏ adapter), NOREFRESH (bỏ mask refresh), NOKL (bỏ tail polish),
    # NORELMSE (dùng MSE tuyệt đối như exp_n)
    flags = set(a.upper() for a in sys.argv[2:])
    use_adpt = "NOADPT" not in flags
    use_refresh = "NOREFRESH" not in flags
    use_kl = "NOKL" not in flags
    use_relmse = "NORELMSE" not in flags
    log(f"Config chọn: {pick_name} (mốc: vi {pick_ppl:.1f}); flags={sorted(flags) or 'FULL'}")

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

    # xnorm
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
    log("Quỹ đạo FP (trước khi gắn adapter)...")
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

    # wrap + adapter (hook giữ adapter trong forward của model)
    wrapped, adapters, hook_hs = [], [], []
    for b, blk in enumerate(layers):
        for sub, name in LIN_PATHS:
            parent = getattr(blk, sub)
            lin = getattr(parent, name)
            key = f"model.layers.{b}.{sub}.{name}"
            mask = wanda_nm_mask(orig_lin[key], xnorm[key], nm[0], nm[1])
            w = LearnQLinear(lin, mask, sgroup, sdtype)
            setattr(parent, name, w)
            wrapped.append((b, parent, name, lin, w))
        ad = BlockAdapter(model.config.hidden_size)
        adapters.append(ad)
        if use_adpt:
            def mk_ah(ad_):
                def ahook(mod, args, output):
                    return ad_(output)
                return ahook
            hook_hs.append(blk.register_forward_hook(mk_ah(ad)))

    norm_frozen = model.model.norm.weight.data.clone()
    lm_w = model.lm_head.weight

    t0 = time.time()
    for p_idx, steps in enumerate(PASS_STEPS):
        if p_idx == 1 and use_refresh:  # VÁ-3: refresh mask đầu pass 2
            for b, parent, name, lin, w in wrapped:
                sub = "self_attn" if name in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                w.refresh_mask(xnorm[f"model.layers.{b}.{sub}.{name}"], nm[0], nm[1])
            log("    đã refresh mask theo |Wfp| x xnorm")
        H_q = [H_fp[0][s].clone() for s in range(NC)]
        for b, blk in enumerate(layers):
            mods = [w for (bb, _, _, _, w) in wrapped if bb == b]
            norm_ws = []
            for sub, name in NORM_PATHS:
                parent = getattr(blk, sub) if sub else blk
                norm_ws.append(getattr(parent, name).weight)
            for nw in norm_ws:
                nw.requires_grad_(True)
            ad = adapters[b]
            groups = [
                {"params": [w.Wfp for w in mods], "lr": SEQ_LR},
                {"params": [w.raw_s for w in mods], "lr": SCALE_LR},
                {"params": norm_ws, "lr": 5e-4},
            ]
            if use_adpt:
                groups.append({"params": ad.parameters(), "lr": ADPT_LR})
            opt = torch.optim.Adam(groups)

            def eval_block():
                with torch.no_grad():
                    v = 0.0
                    for s in eval_sub:
                        out = blk(H_q[s], position_embeddings=ropes[s])
                        tgt = H_fp[b + 1][s]
                        den = tgt.pow(2).mean().clamp(min=1e-8) if use_relmse else 1.0
                        v += (F.mse_loss(out, tgt) / den).item()
                return v

            best = eval_block()                                   # VÁ-1: mốc step-0
            best_state = ([w.Wfp.detach().clone() for w in mods],
                          [w.raw_s.detach().clone() for w in mods],
                          [nw.detach().clone() for nw in norm_ws],
                          [p.detach().clone() for p in ad.parameters()])
            for step in range(steps):
                idx = random.sample(range(NC), min(SEQ_BATCH, NC))
                opt.zero_grad()
                loss = 0.0
                for s in idx:
                    out = blk(H_q[s], position_embeddings=ropes[s])
                    tgt = H_fp[b + 1][s]
                    den = tgt.pow(2).mean().clamp(min=1e-8) if use_relmse else 1.0
                    loss = loss + F.mse_loss(out, tgt) / den  # VÁ-2 (tắt bằng NORELMSE)
                (loss / len(idx)).backward()
                opt.step()
                if step % 10 == 9 or step == steps - 1:
                    v = eval_block()
                    if v < best:
                        best = v
                        best_state = ([w.Wfp.detach().clone() for w in mods],
                                      [w.raw_s.detach().clone() for w in mods],
                                      [nw.detach().clone() for nw in norm_ws],
                                      [p.detach().clone() for p in ad.parameters()])
            with torch.no_grad():
                for w, Wb, Sb in zip(mods, best_state[0], best_state[1]):
                    w.Wfp.data = Wb
                    w.raw_s.data = Sb
                for nw, nb in zip(norm_ws, best_state[2]):
                    nw.data = nb
                for p, pb in zip(ad.parameters(), best_state[3]):
                    p.data = pb
            for nw in norm_ws:
                nw.requires_grad_(False)
            with torch.no_grad():
                for s in range(NC):
                    H_q[s] = blk(H_q[s], position_embeddings=ropes[s]).detach()
        log(f"    pass {p_idx+1} xong ({time.time()-t0:.0f}s)")

    # ===== MỚI-2: TAIL KL-POLISH =====
    best_kl = -1.0
    if not use_kl:
        log("Bỏ tail KL-polish (NOKL)")
    if use_kl:
        log(f"Tail KL-polish: {POLISH_TAIL} block cuối + model.norm + adapter đuôi ({POLISH_STEPS} step)")
        tail_start = NB - POLISH_TAIL
        # cache input vào block tail_start (theo trạng thái hiện hành)
        H_tail = []
        with torch.no_grad():
            for s in range(NC):
                h = H_fp[0][s].clone()
                for b in range(tail_start):
                    h = layers[b](h, position_embeddings=ropes[s])
                H_tail.append(h)

        def fwd_tail(s):
            h = H_tail[s]
            for b in range(tail_start, NB):
                h = layers[b](h, position_embeddings=ropes[s])
            hn = h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + RMS_EPS) * model.model.norm.weight
            return F.linear(hn, lm_w)

        @torch.no_grad()
        def ref_logits(s):
            h = H_fp[NB][s]
            hn = h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + RMS_EPS) * norm_frozen
            return F.linear(hn, lm_w)

        tail_params = []
        for (bb, _, _, _, w) in wrapped:
            if bb >= tail_start:
                tail_params += [w.Wfp, w.raw_s]
        model.model.norm.weight.requires_grad_(True)
        tail_ad = []
        for b in range(tail_start, NB):
            tail_ad += list(adapters[b].parameters())
        groups_kl = [
            {"params": tail_params, "lr": 5e-4},
            {"params": [model.model.norm.weight], "lr": 5e-4},
        ]
        if use_adpt:
            groups_kl.append({"params": tail_ad, "lr": ADPT_LR})
        opt = torch.optim.Adam(groups_kl)

        def eval_kl():
            with torch.no_grad():
                v = 0.0
                for s in eval_sub:
                    lq = F.log_softmax(fwd_tail(s), dim=-1)
                    lf = F.log_softmax(ref_logits(s), dim=-1)
                    v += F.kl_div(lq, lf, log_target=True, reduction="batchmean").item()
            return v

        best_kl = eval_kl()
        best_tail = ([p.detach().clone() for p in tail_params],
                     model.model.norm.weight.detach().clone(),
                     [p.detach().clone() for p in tail_ad])
        log(f"    KL step-0: {best_kl:.4f}")
        for step in range(POLISH_STEPS):
            idx = random.sample(range(NC), POLISH_BATCH)
            opt.zero_grad()
            loss = 0.0
            for s in idx:
                lq = F.log_softmax(fwd_tail(s), dim=-1)
                lf = F.log_softmax(ref_logits(s), dim=-1)
                loss = loss + F.kl_div(lq, lf, log_target=True, reduction="batchmean")
            (loss / len(idx)).backward()
            opt.step()
            if step % 8 == 7 or step == POLISH_STEPS - 1:
                v = eval_kl()
                if v < best_kl:
                    best_kl = v
                    best_tail = ([p.detach().clone() for p in tail_params],
                                 model.model.norm.weight.detach().clone(),
                                 [p.detach().clone() for p in tail_ad])
        with torch.no_grad():
            for p, pb in zip(tail_params, best_tail[0]):
                p.data = pb
            model.model.norm.weight.data = best_tail[1]
            for p, pb in zip(tail_ad, best_tail[2]):
                p.data = pb
        model.model.norm.weight.requires_grad_(False)
        log(f"    KL best: {best_kl:.4f}")

    # bake + eval
    with torch.no_grad():
        for b, parent, name, lin, w in wrapped:
            lin.weight.data = w.bake()
            setattr(parent, name, lin)
    pv, pj = eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja)
    tag = ",".join(sorted(flags)) if flags else "FULL"
    log(f"=> EXP P [{pick_name} | {tag}]: PPL vi {pv:.1f} / ja {pj:.1f}")
    out = {}
    if os.path.exists(OUT_JSON):
        try:
            with io.open(OUT_JSON, "r", encoding="utf-8") as f:
                out = json.load(f)
        except Exception:
            out = {}
    out["fp32"] = {"ppl_vi": ppl_fp[0], "ppl_ja": ppl_fp[1]}
    out[f"exp_p[{pick_name}][{tag}]"] = {"bpw": bpw_val + (0.03 if use_adpt else 0.0),
                                         "ppl_vi": pv, "ppl_ja": pj,
                                         "kl_best": best_kl, "base_ppl_vi": pick_ppl}
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 72)
    print(f"EXP P — {pick_name} + 3 vá + adapter(+0.03bpw) + tail-KL-polish")
    print(f"  mốc cũ  : vi {pick_ppl:10.1f}")
    print(f"  exp_p   : vi {pv:10.1f}  ja {pj:10.1f}   (fp32: {ppl_fp[0]:.1f}/{ppl_fp[1]:.1f})")
    print("=" * 72)


if __name__ == "__main__":
    main()
