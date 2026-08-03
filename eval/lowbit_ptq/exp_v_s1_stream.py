# -*- coding: utf-8 -*-
"""
Exp V — S1-ONLY STREAMING cho model LỚN (Qwen3-30B-A3B MoE) trên 1 GPU nhỏ.

Thiết kế:
  - Model bf16 nằm TRỌN trên CPU RAM (container ~140GB); GPU chỉ cầm TỪNG BLOCK (~1.3GB).
  - Sweep 1 (GPU, stream block): thu xnorm mọi linear (kể cả 128 expert × 48 block).
  - Gauge (up↔down per-EXPERT, v↔o per-kv-head) + PERM kênh exact (hidden toàn cục gồm cả
    cột ROUTER + lm_head untied; intermediate per-expert; head-dim v↔o) — trên CPU.
  - Sweep 2: cổng FP-bất-biến (PPL vi phải khớp) + cache H_fp (49 biên × calib, CPU ~2.4GB).
  - S1 single-pass bake-as-you-go: mỗi block lên GPU → wrap LearnQLinear (trừ router — giữ FP)
    → opt scales/W/bias/norm theo MSE khớp H_fp → BAKE ngay → truyền H_q sang block sau.
  - Đo PPL 6 miền + val-100 (stream), lưu state bf16 đã bake.

Khác exp_r: KHÔNG có S2/KD (30B e2e cần multi-GPU), 1 pass (không 2), không orig toàn cục.
Ckpt bake có bias=True trong Linear + router/embed/norm FP — nạp lại cần phẫu thuật như exp_s.

Smoke: --model-id Qwen/Qwen3-0.6B --smoke (nhánh dense dùng chung code path).
"""
import argparse
import glob
import io
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
random.seed(0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root")
from exp_r_qat_lite import (CAL_CHAT, CAL_CODE, CAL_EN, CAL_MATH, CAL_ZH,  # noqa: E402
                            CODE_EVAL, EN_EVAL, MATH_EVAL, ZH_EVAL,
                            LearnQLinear, balanced_idx, nm_mask_from_imp, read_lines)

OUT_DIR = os.environ.get("EXPR_OUT_DIR", os.path.dirname(os.path.abspath(__file__)))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def block_linears(blk, moe):
    """[(key_suffix, module)] các linear LƯỢNG TỬ trong 1 block. Router KHÔNG nằm đây."""
    out = [("self_attn.q_proj", blk.self_attn.q_proj),
           ("self_attn.k_proj", blk.self_attn.k_proj),
           ("self_attn.v_proj", blk.self_attn.v_proj),
           ("self_attn.o_proj", blk.self_attn.o_proj)]
    if moe:
        for i, ex in enumerate(blk.mlp.experts):
            out += [(f"mlp.experts.{i}.gate_proj", ex.gate_proj),
                    (f"mlp.experts.{i}.up_proj", ex.up_proj),
                    (f"mlp.experts.{i}.down_proj", ex.down_proj)]
    else:
        out += [("mlp.gate_proj", blk.mlp.gate_proj),
                ("mlp.up_proj", blk.mlp.up_proj),
                ("mlp.down_proj", blk.mlp.down_proj)]
    return out


def blk_forward(blk, h, pe):
    o = blk(h, position_embeddings=pe)
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def stream_sweep(model, seqs, dev, collect_xnorm=False, collect_h=False, nll_domains=None):
    """Chạy calib qua model bằng cách stream TỪNG BLOCK lên GPU.
    seqs: list[Tensor ids CPU]. Trả (xnorm, H[49], nll_per_domain).
    nll_domains: dict name->list idx (tính PPL các miền trong CÙNG sweep)."""
    layers = model.model.layers
    moe = hasattr(layers[0].mlp, "experts")
    emb = model.model.embed_tokens
    rot = model.model.rotary_emb.to(dev)
    hs, pes = [], []
    for ids in seqs:
        h = emb(ids).to(dev, torch.bfloat16)   # embed trên CPU, đẩy GPU
        pos = torch.arange(ids.shape[1])[None].to(dev)
        pes.append(rot(h, pos))
        hs.append(h)
    xnorm, H = {}, ([ [x.cpu() for x in hs] ] if collect_h else None)
    for b, blk in enumerate(layers):
        blk.to(dev)
        handles = []
        if collect_xnorm:
            for suf, lin in block_linears(blk, moe):
                key = f"model.layers.{b}.{suf}"
                acc = torch.zeros(lin.in_features, device=dev, dtype=torch.float32)
                xnorm[key] = acc
                def mk(a):
                    def hk(mod, inp):
                        x = inp[0]
                        a.add_(x.detach().reshape(-1, x.shape[-1]).float().pow(2).sum(0))
                    return hk
                handles.append(lin.register_forward_pre_hook(mk(acc)))
        for i in range(len(hs)):
            hs[i] = blk_forward(blk, hs[i], pes[i])
        for hd in handles:
            hd.remove()
        if collect_h:
            H.append([x.cpu() for x in hs])
        blk.to("cpu")
    norm = model.model.norm.to(dev)
    head = model.lm_head.to(dev)
    nll = {}
    if nll_domains:
        for name, idxs in nll_domains.items():
            s_nll, s_tok = 0.0, 0
            for i in idxs:
                lg = head(norm(hs[i])).float()
                ids = seqs[i].to(dev)
                if ids.shape[1] < 2:
                    continue
                s_nll += F.cross_entropy(lg[0, :-1], ids[0, 1:], reduction="sum").item()
                s_tok += ids.shape[1] - 1
            nll[name] = math.exp(s_nll / max(s_tok, 1))
    norm.to("cpu")
    head.to("cpu")
    if collect_xnorm:
        xnorm = {k: v.sqrt().cpu() for k, v in xnorm.items()}
    return xnorm, H, nll


@torch.no_grad()
def apply_gauge_moe(model):
    """up↔down chéo per-expert (silu chỉ trên gate -> scale up là exact) + v↔o per-kv-head."""
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // n_q)
    rep = n_q // n_kv
    for blk in model.model.layers:
        moe = hasattr(blk.mlp, "experts")
        exps = blk.mlp.experts if moe else [blk.mlp]
        for ex in exps:
            Wd, Wu = ex.down_proj.weight.data, ex.up_proj.weight.data
            c = Wd.abs().float().mean(dim=0).clamp(min=1e-8)
            d = (c / torch.exp(torch.log(c).mean())).to(Wu.dtype)
            Wu.mul_(d[:, None])
            Wd.div_(d[None, :])
        Wv = blk.self_attn.v_proj.weight.data
        Wo = blk.self_attn.o_proj.weight.data
        co = Wo.abs().float().mean(dim=0).clamp(min=1e-8).view(n_q, hd)
        for kv in range(n_kv):
            qs = [kv * rep + r for r in range(rep)]
            m = torch.exp(sum(torch.log(co[q]) for q in qs) / rep)
            m = (m / torch.exp(torch.log(m).mean())).to(Wv.dtype)
            Wv[kv * hd:(kv + 1) * hd, :].mul_(m[:, None])
            for q in qs:
                Wo[:, q * hd:(q + 1) * hd].div_(m[None, :])


@torch.no_grad()
def apply_perm_moe(model, xnorm, M):
    """Hoán vị kênh exact cho MoE: (a) hidden toàn cục — cột q/k/v + ROUTER + expert gate/up,
    hàng o/expert-down, embed + lm_head (untied) + norms; (b) intermediate per-EXPERT;
    (c) head-dim v↔o per-kv-head. Cập nhật xnorm tại chỗ."""
    cfg = model.config
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // n_q)
    rep = n_q // n_kv
    layers = model.model.layers
    moe = hasattr(layers[0].mlp, "experts")
    D = cfg.hidden_size
    imp = torch.zeros(D)
    for b in range(len(layers)):
        for suf in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
            xn = xnorm[f"model.layers.{b}.{suf}"]
            imp += xn / xn.mean().clamp(min=1e-8)
        pref = f"model.layers.{b}."
        for k, xn in xnorm.items():
            if k.startswith(pref) and ("gate_proj" in k or "up_proj" in k):
                imp += xn / xn.mean().clamp(min=1e-8)
    pi = balanced_idx(imp, M)
    model.model.embed_tokens.weight.data = model.model.embed_tokens.weight.data[:, pi].contiguous()
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight.data = model.lm_head.weight.data[:, pi].contiguous()
    model.model.norm.weight.data = model.model.norm.weight.data[pi]
    for b, blk in enumerate(layers):
        blk.input_layernorm.weight.data = blk.input_layernorm.weight.data[pi]
        blk.post_attention_layernorm.weight.data = blk.post_attention_layernorm.weight.data[pi]
        for name in ("q_proj", "k_proj", "v_proj"):
            lin = getattr(blk.self_attn, name)
            lin.weight.data = lin.weight.data[:, pi].contiguous()
            key = f"model.layers.{b}.self_attn.{name}"
            xnorm[key] = xnorm[key][pi]
        blk.self_attn.o_proj.weight.data = blk.self_attn.o_proj.weight.data[pi, :].contiguous()
        if moe:
            blk.mlp.gate.weight.data = blk.mlp.gate.weight.data[:, pi].contiguous()  # ROUTER
            exps = blk.mlp.experts
        else:
            exps = [blk.mlp]
        for i, ex in enumerate(exps):
            for nm in ("gate_proj", "up_proj"):
                getattr(ex, nm).weight.data = getattr(ex, nm).weight.data[:, pi].contiguous()
                key = (f"model.layers.{b}.mlp.experts.{i}.{nm}" if moe
                       else f"model.layers.{b}.mlp.{nm}")
                xnorm[key] = xnorm[key][pi]
            ex.down_proj.weight.data = ex.down_proj.weight.data[pi, :].contiguous()
    for b, blk in enumerate(layers):
        moe_b = hasattr(blk.mlp, "experts")
        exps = blk.mlp.experts if moe_b else [blk.mlp]
        for i, ex in enumerate(exps):
            kd = (f"model.layers.{b}.mlp.experts.{i}.down_proj" if moe_b
                  else f"model.layers.{b}.mlp.down_proj")
            pj = balanced_idx(xnorm[kd], M)
            ex.gate_proj.weight.data = ex.gate_proj.weight.data[pj, :].contiguous()
            ex.up_proj.weight.data = ex.up_proj.weight.data[pj, :].contiguous()
            ex.down_proj.weight.data = ex.down_proj.weight.data[:, pj].contiguous()
            xnorm[kd] = xnorm[kd][pj]
        ko = f"model.layers.{b}.self_attn.o_proj"
        xo = xnorm[ko]
        Wv = blk.self_attn.v_proj.weight.data
        Wo = blk.self_attn.o_proj.weight.data
        xo_new = xo.clone()
        for kv in range(n_kv):
            agg = sum(xo[(kv * rep + r) * hd:(kv * rep + r + 1) * hd] for r in range(rep))
            ph = balanced_idx(agg, M)
            Wv[kv * hd:(kv + 1) * hd, :] = Wv[kv * hd:(kv + 1) * hd, :][ph, :]
            for r in range(rep):
                q = kv * rep + r
                Wo[:, q * hd:(q + 1) * hd] = Wo[:, q * hd:(q + 1) * hd][:, ph]
                xo_new[q * hd:(q + 1) * hd] = xo[q * hd:(q + 1) * hd][ph]
        xnorm[ko] = xo_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--model-glob", default="")
    ap.add_argument("--dev-vi", default="/root/qat_data/dev.vi")
    ap.add_argument("--dev-ja", default="/root/qat_data/dev.ja")
    ap.add_argument("--nm-n", type=int, default=2)
    ap.add_argument("--nm-m", type=int, default=4)
    ap.add_argument("--sgroup", type=int, default=64)
    ap.add_argument("--steps-block", type=int, default=60)
    ap.add_argument("--cal-scale", type=int, default=1,
                    help="x4 = calib 4x (vi/ja từ dev, en/code/zh/math từ file kd nếu có)")
    ap.add_argument("--ja-share", type=int, default=32,
                    help="số câu ja mỗi đơn vị cal-scale (v3: 44 — vá seesaw ja)")
    ap.add_argument("--kd-en", default="")
    ap.add_argument("--kd-code", default="")
    ap.add_argument("--kd-zh", default="")
    ap.add_argument("--kd-math", default="")
    ap.add_argument("--calib-seq", type=int, default=96)
    ap.add_argument("--val100", type=int, default=1)
    ap.add_argument("--save", type=int, default=1)
    ap.add_argument("--tag", default="exp_v")
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(4, os.cpu_count() - 2))
    log(f"exp_v: model={args.model_glob or args.model_id} nm {args.nm_n}:{args.nm_m}"
        f" steps/block={args.steps_block} dev={dev}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    src = glob.glob(args.model_glob)[0] if args.model_glob else args.model_id
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    moe = hasattr(layers[0].mlp, "experts")
    log(f"nạp xong: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params, moe={moe},"
        f" {len(layers)} block")

    # ---- calib mixwj (×cal-scale cho MoE — chống đói-expert) + bộ đo 6 miền ----
    k = 4 if args.smoke else None
    cs = 1 if args.smoke else max(1, args.cal_scale)
    js = args.ja_share
    vi_c = read_lines(args.dev_vi, 20 * cs + 20)[: (k or 20 * cs)]
    ja_c = read_lines(args.dev_ja, js * cs + 20)[: (k or js * cs)]

    def dom_cal(path, base, n):
        if path and os.path.exists(path):
            got = read_lines(path, n)
            if len(got) >= n // 2:
                # unescape newline (kd_code_ml giữ cấu trúc code); vô hại với văn xuôi
                return [s.replace("\\n", "\n").replace("\\\\", "\\") for s in got]
        return (base * ((n + len(base) - 1) // len(base)))[:n]
    en_c = CAL_EN[:k] if k else dom_cal(args.kd_en, CAL_EN, 24 * cs)
    co_c = CAL_CODE[:k] if k else dom_cal(args.kd_code, CAL_CODE, 16 * cs)
    zh_c = CAL_ZH[:k] if k else dom_cal(args.kd_zh, CAL_ZH, 20 * cs)
    ma_c = CAL_MATH[:k] if k else dom_cal(args.kd_math, CAL_MATH, 14 * cs)
    calib_txt = vi_c + ja_c + en_c + co_c + zh_c + ma_c + (CAL_CHAT * cs if not k else CAL_CHAT[:k])
    random.Random(0).shuffle(calib_txt)
    dev_vi = read_lines(args.dev_vi, 400)[-16:]
    dev_ja = read_lines(args.dev_ja, 400)[-16:]
    probes = {"vi": dev_vi, "ja": dev_ja, "en": EN_EVAL, "code": CODE_EVAL,
              "zh": ZH_EVAL, "math": MATH_EVAL}
    seqs, dom_idx, cal_idx = [], {}, []
    for s in calib_txt:
        ids = tok(s, return_tensors="pt", truncation=True, max_length=args.calib_seq).input_ids
        if ids.shape[1] >= 4:
            cal_idx.append(len(seqs))
            seqs.append(ids)
    for name, lines in probes.items():
        dom_idx[name] = []
        for s in lines:
            ids = tok(s, return_tensors="pt", truncation=True,
                      max_length=(160 if name == "code" else 96)).input_ids
            if ids.shape[1] >= 2:
                dom_idx[name].append(len(seqs))
                seqs.append(ids)
    log(f"calib {len(cal_idx)} câu + probe {sum(len(v) for v in dom_idx.values())} câu")

    t0 = time.time()
    xnorm, _, nll_fp = stream_sweep(model, seqs, dev, collect_xnorm=True, nll_domains=dom_idx)
    fp_line = " / ".join(f"{k_} {v:.1f}" for k_, v in nll_fp.items())
    log(f"FP bf16: {fp_line}  (sweep {time.time()-t0:.0f}s)")

    log("gauge per-expert + perm kênh exact...")
    apply_gauge_moe(model)
    apply_perm_moe(model, xnorm, args.nm_m)
    t0 = time.time()
    _, H, nll_g = stream_sweep(model, seqs, dev, collect_h=True, nll_domains={"vi": dom_idx["vi"]})
    log(f"FP sau gauge+perm: vi {nll_g['vi']:.1f} (sweep+cache H {time.time()-t0:.0f}s)")
    if abs(nll_g["vi"] - nll_fp["vi"]) / nll_fp["vi"] > 0.015:
        log("!!! gauge/perm sai — abort")
        sys.exit(1)

    # ---- S1 single-pass, bake-as-you-go ----
    rot = model.model.rotary_emb.to(dev)
    pes = []
    for ids in seqs:
        pos = torch.arange(ids.shape[1])[None].to(dev)
        h0 = torch.zeros(1, ids.shape[1], model.config.hidden_size, device=dev,
                         dtype=torch.bfloat16)
        pes.append(rot(h0, pos))
    NC = len(cal_idx)
    Hq = [H[0][i].clone() for i in cal_idx]           # trạng thái lượng-tử-hóa, CPU
    eval_sub = list(range(0, NC, max(1, NC // 10)))[:10]
    t0 = time.time()
    for b, blk in enumerate(layers):
        tb = time.time()
        blk.to(dev)
        lins = block_linears(blk, moe)
        wrapped = []
        for suf, lin in lins:
            key = f"model.layers.{b}.{suf}"
            imp = lin.weight.data.abs().float() * xnorm[key].to(dev)[None, :].clamp(min=1e-8)
            m_ = nm_mask_from_imp(imp, args.nm_n, args.nm_m)
            w = LearnQLinear(lin, m_, G=args.sgroup, use_bias=True, sdtype="f8")
            parent = blk
            parts = suf.split(".")
            for q_ in parts[:-1]:
                parent = getattr(parent, q_) if not q_.isdigit() else parent[int(q_)]
            setattr(parent, parts[-1], w)
            wrapped.append((parent, parts[-1], lin, w))
        norm_ws = [blk.input_layernorm.weight, blk.post_attention_layernorm.weight]
        for nw in norm_ws:
            nw.requires_grad_(True)
        steps = args.steps_block
        if b <= 1 or b >= len(layers) - 2:
            steps = int(steps * 1.5)
        if args.smoke:
            steps = 4
        opt = torch.optim.Adam([
            {"params": [w.Wfp for *_, w in wrapped], "lr": 1e-3},
            {"params": [w.raw_s for *_, w in wrapped], "lr": 5e-3},
            {"params": [w.bias for *_, w in wrapped], "lr": 5e-4},
            {"params": norm_ws, "lr": 5e-4},
        ])
        tgt = [H[b + 1][i] for i in cal_idx]

        def eval_block():
            with torch.no_grad():
                e = 0.0
                for s_ in eval_sub:
                    o = blk_forward(blk, Hq[s_].to(dev), pes[cal_idx[s_]])
                    e += F.mse_loss(o.float(), tgt[s_].to(dev).float()).item()
                return e
        best = eval_block()
        best_state = ([w.Wfp.detach().clone() for *_, w in wrapped],
                      [w.raw_s.detach().clone() for *_, w in wrapped],
                      [w.bias.detach().clone() for *_, w in wrapped],
                      [nw.detach().clone() for nw in norm_ws])
        for st in range(steps):
            idx = random.sample(range(NC), min(6, NC))
            opt.zero_grad()
            loss = sum(F.mse_loss(
                blk_forward(blk, Hq[s_].to(dev), pes[cal_idx[s_]]).float(),
                tgt[s_].to(dev).float()) for s_ in idx) / len(idx)
            loss.backward()
            opt.step()
            if st % 10 == 9 or st == steps - 1:
                v = eval_block()
                if v < best:
                    best = v
                    best_state = ([w.Wfp.detach().clone() for *_, w in wrapped],
                                  [w.raw_s.detach().clone() for *_, w in wrapped],
                                  [w.bias.detach().clone() for *_, w in wrapped],
                                  [nw.detach().clone() for nw in norm_ws])
        with torch.no_grad():
            for (parent, name, lin, w), Wb, Sb, Bb in zip(wrapped, *best_state[:3]):
                w.Wfp.data, w.raw_s.data, w.bias.data = Wb, Sb, Bb
            for nw, nb in zip(norm_ws, best_state[3]):
                nw.data = nb
            # BAKE: đổ ternary về Linear bias=True rồi giải phóng wrapper
            for parent, name, lin, w in wrapped:
                nl = nn.Linear(w.C, w.R, bias=True)
                nl = nl.to(dev, torch.bfloat16)
                nl.weight.data = w.quant().detach().to(torch.bfloat16)
                nl.bias.data = w.bias.detach().to(torch.bfloat16)
                setattr(parent, name, nl)
            for nw in norm_ws:
                nw.requires_grad_(False)
            for i in range(NC):
                Hq[i] = blk_forward(blk, Hq[i].to(dev), pes[cal_idx[i]]).cpu()
        del wrapped, opt, tgt
        torch.cuda.empty_cache() if dev == "cuda" else None
        blk.to("cpu")
        H[b] = None   # giải phóng dần cache FP
        if b % 4 == 0 or b == len(layers) - 1:
            log(f"  block {b+1}/{len(layers)} xong ({time.time()-tb:.0f}s/blk,"
                f" tổng {(time.time()-t0)/60:.1f}ph)")
    del H, Hq

    # ---- đo cuối (stream) ----
    _, _, nll_q = stream_sweep(model, seqs, dev, nll_domains=dom_idx)
    q_line = " / ".join(f"{k_} {v:.1f}" for k_, v in nll_q.items())
    geo6 = math.exp(sum(math.log(max(v, 1e-9)) for v in nll_q.values()) / len(nll_q))
    log(f"=> EXP V S1 nm{args.nm_n}:{args.nm_m}: {q_line} | geo6 {geo6:.1f}")
    big_vi = big_ja = None
    if args.val100 and not args.smoke:
        v_seqs, vd = [], {"vi100": [], "ja100": []}
        for name, path in (("vi100", args.dev_vi), ("ja100", args.dev_ja)):
            for s in read_lines(path, 200)[-100:]:
                ids = tok(s, return_tensors="pt", truncation=True, max_length=96).input_ids
                if ids.shape[1] >= 2:
                    vd[name].append(len(v_seqs))
                    v_seqs.append(ids)
        _, _, nv = stream_sweep(model, v_seqs, dev, nll_domains=vd)
        big_vi, big_ja = nv["vi100"], nv["ja100"]
        log(f"   val-100 vi: {big_vi:.1f} | val-100 ja: {big_ja:.1f}")

    rj = os.path.join(OUT_DIR, "exp_v_results.json")
    outj = {}
    if os.path.exists(rj):
        try:
            with io.open(rj, "r", encoding="utf-8") as f:
                outj = json.load(f)
        except Exception:
            outj = {}
    outj[args.tag] = {"model": args.model_id or args.model_glob,
                      "nm": f"{args.nm_n}:{args.nm_m}", "steps_block": args.steps_block,
                      "cal_scale": args.cal_scale, "ja_share": args.ja_share,
                      "fp": nll_fp, "s1": nll_q, "geo6": geo6,
                      "val100_vi": big_vi, "val100_ja": big_ja}
    with io.open(rj, "w", encoding="utf-8") as f:
        json.dump(outj, f, ensure_ascii=False, indent=2)
    if args.save and args.out and not args.smoke:
        log(f"lưu state bake bf16 -> {args.out} (model lớn: nhiều phút)")
        torch.save({"state_dict": model.state_dict(),
                    "meta": {"tag": args.tag, "nm": f"{args.nm_n}:{args.nm_m}",
                             "s1": nll_q, "geo6": geo6, "val100_vi": big_vi}}, args.out)
    log("EXP V XONG")


if __name__ == "__main__":
    main()
