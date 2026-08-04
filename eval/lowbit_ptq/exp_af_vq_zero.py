# -*- coding: utf-8 -*-
"""
Exp AF — VQ/codebook PTQ với ENTRY-0 TƯỜNG MINH, nhắm đúng 0.3bpw (nơi sụp nặng nhất).

TIẾP NỐI exp_ad (VQ thuần, âm tính) + exp_ae (Wanda-weighting, CÒN TỆ HƠN cả exp_ad -- xem
RESEARCH_VQ_CODEBOOK_PTQ.md phần cập nhật). Phát hiện MẠNH NHẤT xuyên suốt lab này (exp_h/i):
"mức 0 là vua" -- ternary hơn hẳn binary CHỈ vì có mức 0 (binary dense 1.5bit PPL 1.9 TRIỆU vì
mất mức 0; ternary-sparse cứu được). Mọi thử nghiệm VQ trước giờ (exp_ad/ae) codebook toàn vector
HỌC từ k-means -- KHÔNG có vector-0 tường minh nào được đảm bảo; nếu 1 vùng trọng số thật sự gần 0
(rất có thể ở model đã train, nhiều trọng số nhỏ), k-means vẫn ép nó về entry gần nhất (có thể
không phải 0), "tốn" cùng ngân sách bit như 1 pattern quan trọng.

PHƯƠNG PHÁP (PTQ THUẦN, calibration-based, không train qua data thật):
  Mỗi codebook M có K entry: (K-1) học bằng k-means Lloyd như exp_ad, CỘNG 1 entry CỐ ĐỊNH =
  vector-0 (không bao giờ update qua Lloyd iterations -- luôn là ứng viên trong bước gán gần
  nhất). Chi phí bit KHÔNG đổi (vẫn log2(K) bit/index) -- đây là "miễn phí" nếu k-means tận dụng
  đúng, giống hệt lý do ternary hơn binary ở CÙNG SỐ MỨC (2bit vs 1bit không phải điểm mấu chốt,
  MỨC 0 mới là điểm mấu chốt).

Test đúng tại 0.3bpw (mốc đã sụp nặng nhất ở exp_ad: M1K32, d16, 0.316bpw, PPL vi 65.184.339):
  - TF (không sequential): so trực tiếp AF vs AD tại CÙNG (d,M,K).
  - SEQUENTIAL: nếu TF có tín hiệu tốt, chạy tiếp qua BRECQ-lite để xem có "cứu" được về mức
    đọc-được không (dù kỳ vọng thấp -- exp_h cho thấy MỌI thứ <1.58bit sụp PPL>10k dù đủ chiêu).

Mốc so sánh:
  exp_ad VQ_d16_M1K32 (không entry-0): 0.316bpw, PPL vi 65.184.339 / ja 56.777.103
  exp_h  ternary-sparse tốt nhất @0.3x bit tương tự: xem README/exp_h_results.json
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
OUT_JSON = r"e:\Bit-Translate\eval\lowbit_ptq\exp_af_results.json"

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

TF_CFG = ("VQ_d16_M1K32_zero", 16, 1, 32)          # 0.3125bpw payload -- mốc sụp nặng nhất
SEQ_CFG = ("VQ_d16_M1K32_zero_SEQ", 16, 1, 32)


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


def kmeans_fit_with_zero(vecs, K, iters, sample_cap):
    """(K-1) centroid HỌC bằng Lloyd + 1 entry CỐ ĐỊNH = vector-0 (không bao giờ update),
    luôn là ứng viên trong bước gán gần nhất -- hiện thực hoá "mức 0 là vua" cho VQ."""
    N, d = vecs.shape
    if N > sample_cap:
        idx = torch.randperm(N)[:sample_cap]
        pool = vecs[idx].clone()
    else:
        pool = vecs.clone()
    Np = pool.shape[0]
    K_learn = max(K - 1, 1)
    K_eff = min(K_learn, Np)
    init_idx = torch.randperm(Np)[:K_eff]
    learned = pool[init_idx].clone()
    zero_c = torch.zeros(1, d)
    for _ in range(iters):
        centroids = torch.cat([learned, zero_c], 0)
        assign = assign_nearest(pool, centroids)
        mask_learn = assign < K_eff
        new_c = torch.zeros_like(learned)
        counts = torch.zeros(K_eff)
        if mask_learn.any():
            idxs = assign[mask_learn]
            new_c.index_add_(0, idxs, pool[mask_learn])
            counts.index_add_(0, idxs, torch.ones(int(mask_learn.sum().item())))
        nonempty = counts > 0
        new_c[nonempty] = new_c[nonempty] / counts[nonempty].unsqueeze(1)
        n_empty = int((~nonempty).sum().item())
        if n_empty > 0:
            reidx = torch.randperm(Np)[:n_empty]
            new_c[~nonempty] = pool[reidx]
        learned = new_c
    if K_eff < K_learn:
        pad_c = learned[0:1].repeat(K_learn - K_eff, 1)
        learned = torch.cat([learned, pad_c], 0)
    return torch.cat([learned, zero_c], 0)  # length K_learn + 1 == K


def residual_vq_init_zero(W, d, M, K, iters=KMEANS_ITERS, sample_cap=KMEANS_SAMPLE_CAP):
    vecs, shp = build_vectors(W, d)
    residual = vecs.clone()
    codebooks, assigns = [], []
    for _m in range(M):
        centroids = kmeans_fit_with_zero(residual, K, iters, sample_cap)
        assign = assign_nearest(residual, centroids)
        codebooks.append(centroids)
        assigns.append(assign)
        residual = residual - centroids[assign]
    return codebooks, assigns, shp


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


def local_refine(codebooks, assigns, shp, W0, X, steps=REFINE_STEPS, lr=REFINE_LR, zero_idx=None):
    """Giữ nguyên assignment; tinh chỉnh giá trị codebook -- entry-0 (zero_idx, chỉ số cuối
    mỗi codebook) KHÔNG được train (giữ đúng =0 tuyệt đối, đảm bảo tính chất "miễn phí")."""
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
        if zero_idx is not None:
            with torch.no_grad():
                for p, zi in zip(params, zero_idx):
                    p[zi].zero_()
    return best_W, best_loss


class VQZeroSTELinear(nn.Module):
    def __init__(self, codebooks, assigns, shp, zero_idx):
        super().__init__()
        self.codebooks = nn.ParameterList([nn.Parameter(cb.clone()) for cb in codebooks])
        self.assigns = assigns
        self.shp = shp
        self.zero_idx = zero_idx

    def clamp_zero(self):
        with torch.no_grad():
            for p, zi in zip(self.codebooks, self.zero_idx):
                p[zi].zero_()

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
    results["config"] = {"n_calib": N_CALIB, "kmeans_iters": KMEANS_ITERS,
                          "refine_steps": REFINE_STEPS, "method": "residual_vq_forced_zero_entry"}

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

    # ---- TF config ----
    name, d, M, K = TF_CFG
    if name not in results:
        payload_bpw = M * math.log2(K) / d
        log(f"--- {name}  d={d} M={M} K={K}  (payload ~{payload_bpw:.3f} bpw, entry-0 CO SAN) ---")
        t0 = time.time()
        sse_total, ref_total = 0.0, 0.0
        cb_bits_total, weight_total = 0.0, 0.0
        zero_frac_total, n_vec_total = 0.0, 0
        for i, (n, m) in enumerate(linears):
            W0 = orig[n]
            X = acts[n].to(torch.float32)
            codebooks, assigns, shp = residual_vq_init_zero(W0, d, M, K)
            zero_idx = [cb.shape[0] - 1 for cb in codebooks]
            num_vectors = assigns[0].shape[0]
            for a, zi in zip(assigns, zero_idx):
                zero_frac_total += float((a == zi).float().sum())
                n_vec_total += a.shape[0]
            Wbest, _ = local_refine(codebooks, assigns, shp, W0, X, zero_idx=zero_idx)
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
        zero_frac = zero_frac_total / max(n_vec_total, 1)
        pv, pj = eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja)
        log(f"    => werr {werr*100:.1f}%  bpw={bpw_total:.3f}  zero_frac={zero_frac*100:.1f}%  PPL vi {pv:.1f} / ja {pj:.1f}")
        results[name] = {"d": d, "M": M, "K": K, "payload_bpw": payload_bpw,
                          "codebook_overhead_bpw": overhead_bpw, "bpw_total": bpw_total,
                          "werr": werr, "zero_frac": zero_frac, "ppl_vi": pv, "ppl_ja": pj}
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
            codebooks0, assigns0, shp0 = residual_vq_init_zero(orig[key], d, M, K)
            zero_idx0 = [cb.shape[0] - 1 for cb in codebooks0]
            num_vectors = assigns0[0].shape[0]
            cb_bits_total += M * K * d * 16.0
            weight_total += num_vectors * d
            sm = VQZeroSTELinear(codebooks0, assigns0, shp0, zero_idx0)
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
            for _, _, _, sm in stes:
                sm.clamp_zero()
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
    print("EXP AF — VQ/codebook PTQ + entry-0 tường minh (0.3bpw) trên Qwen3-0.6B")
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
    print("Mốc so sánh (exp_ad, cùng d/M/K, KHÔNG entry-0):")
    print("  exp_ad VQ_d16_M1K32 (no zero)  0.316bpw  PPL vi 65184338.8 / ja 56777102.8")
    print("=" * 88)


if __name__ == "__main__":
    main()
