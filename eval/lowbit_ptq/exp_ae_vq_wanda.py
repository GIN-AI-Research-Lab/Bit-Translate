# -*- coding: utf-8 -*-
"""
Exp AE — VQ/codebook PTQ CÓ TRỌNG SỐ WANDA (|W|·‖X‖ kiểu Wanda) trên Qwen3-0.6B.

TIẾP NỐI exp_ad (KẾT QUẢ ÂM TÍNH): VQ thuần (k-means KHÔNG trọng số) thua scalar ternary
~10 lần ở cùng bpw sau sequential-refine. Nghi vấn chưa loại trừ: recipe VQ đó dùng k-means
trên trọng số THÔ (mọi cột như nhau), trong khi scalar ternary trong lab này CHỈ đạt hiệu năng
tốt SAU KHI thêm trọng số Wanda |W|·‖X‖ để chọn ngưỡng/mask (xác nhận "bắt buộc" trong nhiều
exp trước — mask |W| thuần luôn thua |W|·‖X‖). Đây là so sánh KHÔNG công bằng: "VQ trần trụi"
vs "scalar đã có Wanda". Exp này vá đúng 1 biến đó — thêm Wanda-scaling vào k-means/assignment
của VQ (PTQ THUẦN, không train qua data thật — chỉ đổi metric k-means bằng activation calib):

  1. Tính importance mỗi CỘT (chiều input) imp[c] = RMS(X[:, c]) trên activation calib thật
     (giống Wanda/AWQ/SmoothQuant-style per-channel scaling).
  2. Nhân W theo cột với imp TRƯỚC khi nhóm vector + k-means (làm k-means "quan tâm" đúng
     cột ảnh hưởng nhiều tới output, giống hệt lý do Wanda hơn |W| thuần ở scalar).
  3. Reconstruct ở miền scaled rồi CHIA lại cho imp để về miền trọng số thật trước khi tính
     loss/PPL — không đổi ý nghĩa vật lý của trọng số, chỉ đổi METRIC lúc chọn codebook.
  4. Local reconstruction + SEQUENTIAL giống hệt exp_ad (PTQ hợp lệ, calibration-based).

Chỉ test 2 điểm quan trọng nhất (đỡ tốn thời gian, đã biết đây là 2 điểm đáng chú ý từ exp_ad):
  - VQ_d16_M2K1024 (~1.48bpw, TF) — ở exp_ad đã "ngang" scalar TF, xem Wanda có đẩy vượt không.
  - VQ_d16_M3K256_SEQ (~1.59bpw, + sequential) — ở exp_ad thua scalar SEQUENTIAL ~10x, đây là
    phép thử quan trọng nhất: Wanda-weighting có thu hẹp khoảng cách này không.

Mốc so sánh (khác script, cùng dev set):
  exp_ad VQ_d16_M2K1024 KHÔNG trọng số: 1.483bpw, PPL vi 34194.8 / ja 70749.9
  exp_ad VQ_d16_M3K256_SEQ KHÔNG trọng số: 1.587bpw, PPL vi 6044.3 / ja 10095.7
  exp_h ternary_sparse@0.25 (scalar, Wanda, TF): 1.33bpw, PPL vi 33686.7 / ja 47152.0
  exp_k t2:4 fixpack SEQUENTIAL (scalar, Wanda): 1.94bpw, PPL vi 594.3 / ja 8068.6
"""
import gc
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
OUT_JSON = r"e:\Bit-Translate\eval\lowbit_ptq\exp_ae_results.json"

N_CALIB = 48
N_EVAL = 16
MAX_TOK = 96

KMEANS_ITERS = 6
KMEANS_SAMPLE_CAP = 60000
REFINE_STEPS = 40
REFINE_LR = 2e-3

SEQ_STEPS = 50
SEQ_LR = 1e-3
SEQ_BATCH = 6

LIN_PATHS = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
             ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]

TF_CFG = ("VQ_d16_M2K1024_wanda", 16, 2, 1024)
SEQ_CFG = ("VQ_d16_M3K256_wanda_SEQ", 16, 3, 256)


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


def col_importance(X, C):
    """RMS mỗi cột (chiều input) trên activation calib thật -- kiểu Wanda/AWQ.
    CLAMP dynamic range quanh 1.0: bản đầu KHÔNG clamp bị nổ số (werr 8137%, MSE sequential
    tăng thay vì giảm) vì reconstruct chia lại cho imp -- cột có RMS activation gần 0 (kênh
    "chết"/outlier ngược, đã biết tồn tại trong lab này -- xem RESEARCH_TQ33_OUTLIER_FIX.md,
    massive-activation channel làm các cột khác co lại tương đối) khiến chia ra giá trị khổng
    lồ. Clamp [0.2, 5.0] quanh trung vị giữ ĐÚNG thứ tự quan trọng tương đối (cột RMS cao vẫn
    được k-means ưu tiên hơn) mà không cho phép khuếch đại vô hạn khi unscale."""
    imp = torch.sqrt((X.float() ** 2).mean(0) + 1e-8)
    if imp.shape[0] < C:
        imp = F.pad(imp, (0, C - imp.shape[0]), value=1.0)
    med = imp.median().clamp_min(1e-6)
    imp = (imp / med).clamp(0.2, 5.0)
    return imp


def build_vectors(Wscaled, d):
    R, C = Wscaled.shape
    pad = (d - C % d) % d
    Wp = F.pad(Wscaled, (0, pad)) if pad else Wscaled
    return Wp.reshape(-1, d), (R, C, Wp.shape[1], pad)


def assign_nearest(vecs, centroids, chunk=131072):
    N = vecs.shape[0]
    c_sq = (centroids * centroids).sum(1)
    out = torch.empty(N, dtype=torch.long)
    for i in range(0, N, chunk):
        v = vecs[i:i + chunk]
        v_sq = (v * v).sum(1, keepdim=True)
        dot = v @ centroids.t()
        dist = v_sq - 2.0 * dot + c_sq[None, :]
        out[i:i + chunk] = dist.argmin(1)
    return out


def kmeans_fit(vecs, K, iters, sample_cap):
    N = vecs.shape[0]
    if N > sample_cap:
        idx = torch.randperm(N)[:sample_cap]
        pool = vecs[idx].clone()
    else:
        pool = vecs.clone()
    Np = pool.shape[0]
    K_eff = min(K, Np)
    init_idx = torch.randperm(Np)[:K_eff]
    centroids = pool[init_idx].clone()
    for _ in range(iters):
        assign = assign_nearest(pool, centroids)
        new_c = torch.zeros_like(centroids)
        counts = torch.zeros(K_eff)
        new_c.index_add_(0, assign, pool)
        counts.index_add_(0, assign, torch.ones(Np))
        nonempty = counts > 0
        new_c[nonempty] = new_c[nonempty] / counts[nonempty].unsqueeze(1)
        n_empty = int((~nonempty).sum().item())
        if n_empty > 0:
            reidx = torch.randperm(Np)[:n_empty]
            new_c[~nonempty] = pool[reidx]
        centroids = new_c
    if K_eff < K:
        pad_c = centroids[0:1].repeat(K - K_eff, 1)
        centroids = torch.cat([centroids, pad_c], 0)
    return centroids


def residual_vq_init_weighted(W, imp, d, M, K, iters=KMEANS_ITERS, sample_cap=KMEANS_SAMPLE_CAP):
    """k-means TRÊN W*imp (miền scaled) -- assignment "quan tâm" đúng cột ảnh hưởng output nhiều."""
    R, C = W.shape
    pad = (d - C % d) % d
    imp_p = F.pad(imp, (0, pad), value=1.0) if pad else imp
    Wscaled = W * imp[None, :]
    vecs, shp = build_vectors(Wscaled, d)
    residual = vecs.clone()
    codebooks, assigns = [], []
    for _m in range(M):
        centroids = kmeans_fit(residual, K, iters, sample_cap)
        assign = assign_nearest(residual, centroids)
        codebooks.append(centroids)
        assigns.append(assign)
        residual = residual - centroids[assign]
    return codebooks, assigns, shp, imp_p


def reconstruct_vq_weighted(codebooks, assigns, shp, imp_p):
    """Reconstruct ở miền scaled rồi CHIA lại imp -> miền trọng số thật."""
    R, C, Cp, pad = shp
    acc = None
    for cb, asg in zip(codebooks, assigns):
        term = cb[asg]
        acc = term if acc is None else acc + term
    Wp_scaled = acc.reshape(R, Cp)
    Wp = Wp_scaled / imp_p[None, :].clamp_min(1e-8)
    return Wp[:, :C] if pad else Wp


def codebook_bits(M, K, d, num_vectors):
    total_bits = M * K * d * 16.0
    total_weights = num_vectors * d
    return total_bits / total_weights


def local_refine_weighted(codebooks, assigns, shp, imp_p, W0, X, steps=REFINE_STEPS, lr=REFINE_LR):
    params = [cb.clone().requires_grad_(True) for cb in codebooks]
    opt = torch.optim.Adam(params, lr=lr)
    target = (X @ W0.t()).detach()
    best_loss = float("inf")
    best_W = reconstruct_vq_weighted(codebooks, assigns, shp, imp_p).detach().clone()
    for _ in range(steps):
        opt.zero_grad()
        Wc = reconstruct_vq_weighted(params, assigns, shp, imp_p)
        loss = (X @ Wc.t() - target).pow(2).mean()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_W = Wc.detach().clone()
        loss.backward()
        opt.step()
    return best_W, best_loss


class WeightedVQSTELinear(nn.Module):
    def __init__(self, codebooks, assigns, shp, imp_p):
        super().__init__()
        self.codebooks = nn.ParameterList([nn.Parameter(cb.clone()) for cb in codebooks])
        self.assigns = assigns
        self.shp = shp
        self.register_buffer("imp_p", imp_p)

    def current_W(self):
        return reconstruct_vq_weighted(list(self.codebooks), self.assigns, self.shp, self.imp_p)

    def forward(self, x):
        return F.linear(x, self.current_W())


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"Nạp Qwen3-0.6B FP32 từ {MODEL_DIR}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and "layers." in n]
    log(f"{len(linears)} ma trận linear")

    vi = read_lines(DEV_VI, N_CALIB // 2)
    ja = read_lines(DEV_JA, N_CALIB // 2)
    calib = [x for pr in zip(vi, ja) for x in pr]
    eval_vi = read_lines(DEV_VI, 400)[-N_EVAL:]
    eval_ja = read_lines(DEV_JA, 400)[-N_EVAL:]

    try:
        with io.open(OUT_JSON, "r", encoding="utf-8") as f:
            results = json.load(f)
        log(f"RESUME: đã có sẵn {[k for k in results if k not in ('fp32', 'config')]}")
    except Exception:
        results = {}
    if "fp32" in results:
        ppl_fp = (results["fp32"]["ppl_vi"], results["fp32"]["ppl_ja"])
        log(f"RESUME: dùng lại FP32 vi {ppl_fp[0]:.2f} / ja {ppl_fp[1]:.2f}")
    else:
        ppl_fp = (eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja))
        results["fp32"] = {"ppl_vi": ppl_fp[0], "ppl_ja": ppl_fp[1]}
        log(f"FP32: vi {ppl_fp[0]:.2f} / ja {ppl_fp[1]:.2f}")
    results["config"] = {"n_calib": N_CALIB, "kmeans_iters": KMEANS_ITERS,
                          "refine_steps": REFINE_STEPS, "weighting": "wanda_col_rms"}

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
    for n in acts:
        acts[n] = torch.cat(acts[n], 0)
    orig = {n: m.weight.data.clone() for n, m in linears}
    imps = {n: col_importance(acts[n].float(), orig[n].shape[1]) for n, _ in linears}

    # ---- TF config (không sequential) ----
    name, d, M, K = TF_CFG
    if name not in results:
        payload_bpw = M * math.log2(K) / d
        log(f"--- {name}  d={d} M={M} K={K}  (payload ~{payload_bpw:.3f} bpw) ---")
        t0 = time.time()
        sse_total, ref_total = 0.0, 0.0
        cb_bits_total, weight_total = 0.0, 0.0
        for i, (n, m) in enumerate(linears):
            W0 = orig[n]
            X = acts[n].to(torch.float32)
            imp = imps[n]
            codebooks, assigns, shp, imp_p = residual_vq_init_weighted(W0, imp, d, M, K)
            num_vectors = assigns[0].shape[0]
            Wbest, _ = local_refine_weighted(codebooks, assigns, shp, imp_p, W0, X)
            m.weight.data = Wbest
            sse_total += float((Wbest - W0).pow(2).sum())
            ref_total += float(W0.pow(2).sum())
            cb_bits_total += M * K * d * 16.0
            weight_total += num_vectors * d
            if (i + 1) % 49 == 0:
                log(f"    {i+1}/{len(linears)} ({time.time()-t0:.0f}s)")
        werr = math.sqrt(sse_total / max(ref_total, 1e-12))
        overhead_bpw = cb_bits_total / max(weight_total, 1.0)
        bpw_total = payload_bpw + overhead_bpw
        pv, pj = eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja)
        log(f"    => werr {werr*100:.1f}%  bpw={bpw_total:.3f}  PPL vi {pv:.1f} / ja {pj:.1f}")
        results[name] = {"d": d, "M": M, "K": K, "payload_bpw": payload_bpw,
                          "codebook_overhead_bpw": overhead_bpw, "bpw_total": bpw_total,
                          "werr": werr, "ppl_vi": pv, "ppl_ja": pj}
        with io.open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with torch.no_grad():
            for n, m in linears:
                m.weight.data = orig[n].clone()
        gc.collect()
    else:
        log(f"--- {name}: RESUME bỏ qua ---")

    # ---- SEQUENTIAL config ----
    name, d, M, K = SEQ_CFG
    if name in results:
        log(f"=== {name}: RESUME bỏ qua ===")
        with io.open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print_summary(results, ppl_fp)
        return
    payload_bpw = M * math.log2(K) / d
    log(f"=== SEQUENTIAL {name}  d={d} M={M} K={K}  (payload ~{payload_bpw:.3f} bpw) ===")
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

    cb_bits_total, weight_total = 0.0, 0.0
    t0 = time.time()
    for b, blk in enumerate(layers):
        stes = []
        for sub, lname in LIN_PATHS:
            parent = getattr(blk, sub)
            lin = getattr(parent, lname)
            key = f"model.layers.{b}.{sub}.{lname}"
            imp = imps[key]
            codebooks0, assigns0, shp0, imp_p0 = residual_vq_init_weighted(orig[key], imp, d, M, K)
            num_vectors = assigns0[0].shape[0]
            cb_bits_total += M * K * d * 16.0
            weight_total += num_vectors * d
            sm = WeightedVQSTELinear(codebooks0, assigns0, shp0, imp_p0)
            setattr(parent, lname, sm)
            stes.append((parent, lname, lin, sm))
        params = [p for _, _, _, sm in stes for p in sm.parameters()]
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
                    best_state = [[p.detach().clone() for p in sm.parameters()] for _, _, _, sm in stes]
        for (parent, lname, lin, sm), pstate in zip(
                stes, best_state if best_state else [[p.detach() for p in sm.parameters()] for _, _, _, sm in stes]):
            with torch.no_grad():
                for p, val in zip(sm.parameters(), pstate):
                    p.copy_(val)
                lin.weight.data = sm.current_W().detach()
            setattr(parent, lname, lin)
        with torch.no_grad():
            for s in range(NC):
                H_q[s] = blk(H_q[s], position_embeddings=ropes[s]).detach()
        log(f"    block {b:2d}/{NB}: mse {e0:.4f} -> {best:.4f}  ({time.time()-t0:.0f}s)")
        gc.collect()

    overhead_bpw = cb_bits_total / max(weight_total, 1.0)
    bpw_total = payload_bpw + overhead_bpw
    pv, pj = eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja)
    log(f"    => SEQ PPL vi {pv:.1f} / ja {pj:.1f}  bpw_total {bpw_total:.3f}")
    results[name] = {"d": d, "M": M, "K": K, "payload_bpw": payload_bpw,
                      "codebook_overhead_bpw": overhead_bpw, "bpw_total": bpw_total,
                      "ppl_vi": pv, "ppl_ja": pj}
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print_summary(results, ppl_fp)


def print_summary(results, ppl_fp):
    print("\n" + "=" * 88)
    print("EXP AE — VQ/codebook PTQ + Wanda column-importance weighting trên Qwen3-0.6B")
    print("=" * 88)
    print(f"{'config':26s}{'bpw(total)':>12s}{'werr':>9s}{'PPL vi':>12s}{'PPL ja':>12s}")
    print("-" * 88)
    for k, v in results.items():
        if k in ("fp32", "config"):
            continue
        werr_s = f"{v['werr']*100:.1f}%" if "werr" in v else "  -  "
        print(f"{k:26s}{v['bpw_total']:12.3f}{werr_s:>9s}{v['ppl_vi']:12.1f}{v['ppl_ja']:12.1f}")
    print(f"{'fp32':26s}{16.0:12.2f}{'0.0%':>9s}{ppl_fp[0]:12.1f}{ppl_fp[1]:12.1f}")
    print("=" * 88)
    print("So sánh exp_ad (VQ KHÔNG trọng số) + mốc scalar:")
    print("  exp_ad VQ_d16_M2K1024 (no-weight)  1.483bpw  PPL vi 34194.8 / ja 70749.9")
    print("  exp_ad VQ_d16_M3K256_SEQ (no-w)     1.587bpw  PPL vi  6044.3 / ja 10095.7")
    print("  exp_h  ternary_sparse@0.25 (scalar) 1.33bpw   PPL vi 33686.7 / ja 47152.0")
    print("  exp_k  t2:4 fixpack SEQUENTIAL       1.94bpw   PPL vi   594.3 / ja  8068.6")
    print("=" * 88)


if __name__ == "__main__":
    main()
