# -*- coding: utf-8 -*-
"""
Exp AO — đóng câu hỏi treo giữa 2 phiên song song (Máy A OLMoE-AQLM thắng vs Máy B Qwen3-0.6B
VQ-greedy thua): AQLM với BEAM-SEARCH THẬT (2 codebook, thuật toán của máy A trong
exp_am_aqlm_full.py, KHÔNG phải residual-greedy đơn giản của exp_ad/ae/af) + SEQUENTIAL
(BRECQ-lite, đòn mạnh nhất lab từng đo) trên CHÍNH Qwen3-0.6B, so trực tiếp 3 số CÙNG bảng:

  1. scalar ternary 2:4 fixpack + SEQUENTIAL (exp_k, đã có): 1,94bpw -> PPL vi 594,3
  2. VQ residual-GREEDY + SEQUENTIAL (exp_ad, đã có, recipe của máy B):
     1,587bpw -> PPL vi 6.044,3 (thua scalar ~10x)
  3. AQLM beam-search + SEQUENTIAL (VIỆC MỚI ở đây, recipe của máy A) -- câu hỏi mở duy nhất
     còn lại: beam-search có đổi kết luận #2 không?

Theo đúng khuyến nghị trong HANDOFF_MAYB_AQLM_SEQUENTIAL.md: dùng g=8, M=2, K=256 (~2,06bpw,
ĐÚNG cấu hình máy A đã test trên OLMoE — err 0,3195 best, không có sequential) để so được cả
với số của máy A lẫn số của máy B. beam_assign() PORT NGUYÊN THUẬT TOÁN từ
exp_am_aqlm_full.py (không sửa logic, chỉ đổi cách tính khoảng cách sang dạng chunk giống
assign_nearest() của exp_ad để nhất quán style code máy B).

KHÔNG dùng gradient-refine-trên-weight-MSE (máy A đã đo: LÀM TỆ HƠN beam-search đơn thuần,
xem exp_am ablation (c) tệ hơn (b)) -- dùng "k-means + beam-search, bỏ refine" làm INIT, sau đó
refine bằng chính SEQUENTIAL block-wise (khớp OUTPUT block downstream, khác hẳn refine-trên-
weight-MSE của máy A -- đây là đòn CHƯA ai thử với beam-search assignment).
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
OUT_JSON = r"e:\Bit-Translate\eval\lowbit_ptq\exp_ao_aqlm_beam_seq_results.json"

N_CALIB = 48
N_EVAL = 16
MAX_TOK = 96

KMEANS_ITERS = 6
KMEANS_SAMPLE_CAP = 60000
BEAM = 4
REFINE_STEPS = 40
REFINE_LR = 2e-3

SEQ_STEPS = 50
SEQ_LR = 1e-3
SEQ_BATCH = 6

LIN_PATHS = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
             ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]

# g=8, M=2, K=256 -- DUNG cau hinh may A da test tren OLMoE (khong sequential): err 0.3195 @2.0625bpw
TF_CFG = ("AQLM_beam_g8_M2K256", 8, 2, 256)
SEQ_CFG = ("AQLM_beam_g8_M2K256_SEQ", 8, 2, 256)


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


def build_vectors(W, d):
    R, C = W.shape
    pad = (d - C % d) % d
    Wp = F.pad(W, (0, pad)) if pad else W
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


def beam_assign(vecs, C1, C2, beam=BEAM, chunk=131072):
    """PORT NGUYÊN THUẬT TOÁN từ exp_am_aqlm_full.py (máy A): giữ top-`beam` ứng viên C1, với
    MỖI ứng viên tìm C2 tốt nhất cho residual, chọn cặp tổng lỗi bé nhất. Khác greedy-residual
    (exp_ad): greedy chỉ chọn C1 gần nhất rồi MỚI tìm C2 -- có thể bỏ lỡ cặp (C1,C2) tổng tốt
    hơn mà C1 riêng lẻ không phải gần nhất."""
    N = vecs.shape[0]
    a1 = torch.empty(N, dtype=torch.long)
    a2 = torch.empty(N, dtype=torch.long)
    c1_sq = (C1 * C1).sum(1)
    c2_sq = (C2 * C2).sum(1)
    beam_eff = min(beam, C1.shape[0])
    for i in range(0, N, chunk):
        x = vecs[i:i + chunk]
        x_sq = (x * x).sum(1, keepdim=True)
        d1 = x_sq - 2.0 * (x @ C1.t()) + c1_sq[None, :]
        _, topi = d1.topk(beam_eff, dim=1, largest=False)
        best_err = best_i1 = best_i2 = None
        for b in range(beam_eff):
            i1 = topi[:, b]
            resid = x - C1[i1]
            r_sq = (resid * resid).sum(1, keepdim=True)
            d2 = r_sq - 2.0 * (resid @ C2.t()) + c2_sq[None, :]
            e2, i2 = d2.min(dim=1)
            if best_err is None:
                best_err, best_i1, best_i2 = e2, i1, i2
            else:
                better = e2 < best_err
                best_err = torch.where(better, e2, best_err)
                best_i1 = torch.where(better, i1, best_i1)
                best_i2 = torch.where(better, i2, best_i2)
        a1[i:i + chunk], a2[i:i + chunk] = best_i1, best_i2
    return a1, a2


def residual_vq_init_beam(W, d, M, K, iters=KMEANS_ITERS, sample_cap=KMEANS_SAMPLE_CAP):
    """M PHẢI = 2 (beam_assign chỉ hỗ trợ 2 codebook, đúng recipe máy A đã verify tốt nhất
    trong ablation -- (b) beam-search thắng cả (c) gradient-refine và (d) alternating)."""
    assert M == 2, "beam_assign hiện chỉ hỗ trợ M=2 (đúng recipe máy A đã chọn là tốt nhất)"
    vecs, shp = build_vectors(W, d)
    C1 = kmeans_fit(vecs, K, iters, sample_cap)
    a1_greedy = assign_nearest(vecs, C1)
    resid = vecs - C1[a1_greedy]
    C2 = kmeans_fit(resid, K, iters, sample_cap)
    a1, a2 = beam_assign(vecs, C1, C2, beam=BEAM)
    return [C1, C2], [a1, a2], shp


def reconstruct_vq(codebooks, assigns, shp):
    R, C, Cp, pad = shp
    acc = None
    for cb, asg in zip(codebooks, assigns):
        term = cb[asg]
        acc = term if acc is None else acc + term
    Wp = acc.reshape(R, Cp)
    return Wp[:, :C] if pad else Wp


def codebook_bits(M, K, d, num_vectors):
    total_bits = M * K * d * 16.0
    total_weights = num_vectors * d
    return total_bits / total_weights


def local_refine(codebooks, assigns, shp, W0, X, steps=REFINE_STEPS, lr=REFINE_LR):
    """Giữ nguyên assignment (beam đã chọn), tinh chỉnh GIÁ TRỊ codebook khớp OUTPUT downstream
    (X@Ŵ.T vs X@W0.T) -- khác hẳn refine-trên-weight-MSE của máy A (đã đo TỆ HƠN beam đơn
    thuần) vì đây tối ưu đúng cái ảnh hưởng PPL, giống mọi refine khác trong toàn bộ lab."""
    params = [cb.clone().requires_grad_(True) for cb in codebooks]
    opt = torch.optim.Adam(params, lr=lr)
    target = (X @ W0.t()).detach()
    best_loss = float("inf")
    best_W = reconstruct_vq(codebooks, assigns, shp).detach().clone()
    for _ in range(steps):
        opt.zero_grad()
        Wc = reconstruct_vq(params, assigns, shp)
        loss = (X @ Wc.t() - target).pow(2).mean()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_W = Wc.detach().clone()
        loss.backward()
        opt.step()
    return best_W, best_loss


class VQBeamSTELinear(nn.Module):
    def __init__(self, codebooks, assigns, shp):
        super().__init__()
        self.codebooks = nn.ParameterList([nn.Parameter(cb.clone()) for cb in codebooks])
        self.assigns = assigns
        self.shp = shp

    def current_W(self):
        return reconstruct_vq(list(self.codebooks), self.assigns, self.shp)

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
    else:
        ppl_fp = (eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja))
        results["fp32"] = {"ppl_vi": ppl_fp[0], "ppl_ja": ppl_fp[1]}
    log(f"FP32: vi {ppl_fp[0]:.2f} / ja {ppl_fp[1]:.2f}")
    results["config"] = {"n_calib": N_CALIB, "kmeans_iters": KMEANS_ITERS, "beam": BEAM,
                          "refine_steps": REFINE_STEPS, "method": "aqlm_beam_search_port_tu_may_A"}

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

    # ---- TF config (kiem tra nhanh, khong sequential) ----
    name, d, M, K = TF_CFG
    if name not in results:
        payload_bpw = M * math.log2(K) / d
        log(f"--- {name}  d={d} M={M} K={K} beam={BEAM}  (payload ~{payload_bpw:.3f} bpw) ---")
        t0 = time.time()
        sse_total, ref_total = 0.0, 0.0
        cb_bits_total, weight_total = 0.0, 0.0
        for i, (n, m) in enumerate(linears):
            W0 = orig[n]
            X = acts[n].to(torch.float32)
            codebooks, assigns, shp = residual_vq_init_beam(W0, d, M, K)
            num_vectors = assigns[0].shape[0]
            Wbest, _ = local_refine(codebooks, assigns, shp, W0, X)
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

    # ---- SEQUENTIAL config (phep so sanh QUYET DINH) ----
    name, d, M, K = SEQ_CFG
    if name in results:
        log(f"=== {name}: RESUME bỏ qua ===")
        print_summary(results, ppl_fp)
        return
    payload_bpw = M * math.log2(K) / d
    log(f"=== SEQUENTIAL {name}  d={d} M={M} K={K} beam={BEAM}  (payload ~{payload_bpw:.3f} bpw) ===")
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
            codebooks0, assigns0, shp0 = residual_vq_init_beam(orig[key], d, M, K)
            num_vectors = assigns0[0].shape[0]
            cb_bits_total += M * K * d * 16.0
            weight_total += num_vectors * d
            sm = VQBeamSTELinear(codebooks0, assigns0, shp0)
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
    print("\n" + "=" * 92)
    print("EXP AO — AQLM beam-search (recipe máy A) + SEQUENTIAL (recipe máy B) trên Qwen3-0.6B")
    print("=" * 92)
    print(f"{'config':28s}{'bpw(total)':>12s}{'werr':>9s}{'PPL vi':>12s}{'PPL ja':>12s}")
    print("-" * 92)
    for k, v in results.items():
        if k in ("fp32", "config"):
            continue
        werr_s = f"{v['werr']*100:.1f}%" if "werr" in v else "  -  "
        print(f"{k:28s}{v['bpw_total']:12.3f}{werr_s:>9s}{v['ppl_vi']:12.1f}{v['ppl_ja']:12.1f}")
    print(f"{'fp32':28s}{16.0:12.2f}{'0.0%':>9s}{ppl_fp[0]:12.1f}{ppl_fp[1]:12.1f}")
    print("=" * 92)
    print("Mốc 3 chiều so sánh (khác script, cùng dev set vi/ja Qwen3-0.6B):")
    print("  scalar t2:4 fixpack SEQUENTIAL (exp_k)     1.94bpw   PPL vi   594.3 / ja  8068.6")
    print("  VQ residual-GREEDY + SEQUENTIAL (exp_ad)   1.587bpw  PPL vi  6044.3 / ja 10095.7")
    print("  AQLM beam-search + SEQUENTIAL (đây)        xem trên")
    print("=" * 92)


if __name__ == "__main__":
    main()
