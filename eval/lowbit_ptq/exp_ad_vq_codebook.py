# -*- coding: utf-8 -*-
"""
Exp AD — Vector Quantization / codebook PTQ (AQLM/QuIP#-style) trên Qwen3-0.6B.

CÂU HỎI: mọi PTQ trước giờ trong lab (exp_b/c/f/g/h/i/k/m) lượng tử hoá TỪNG SỐ ĐỘC LẬP
(scalar: ternary/int-N, dù có Hadamard/gauge/hoán vị trước) -> trần cứng đo được ~4bit
(exp_g: int4 4.5bpw PPL~86 gần FP~69; mọi thứ <1.58bit sụp PPL>10k ở exp_h/i). H1/H2
(gauge-folding, intrinsic-dim probe) chỉ chứng minh KHÔNG có dư thừa dạng biến đổi
tuyến tính/hoán vị đơn giản -- CHƯA test VECTOR QUANTIZATION thật: nhóm d trọng số liền
kề thành 1 vector, lượng tử hoá CẢ VECTOR về entry gần nhất trong 1 codebook HỌC từ chính
trọng số thật (kiểu AQLM "additive/residual quantization"). VQ khai thác tương quan CHÉO
giữa các chiều trong vector mà scalar (dù permute) không làm được.

PHƯƠNG PHÁP (residual/additive VQ, giống AQLM), PTQ THUẦN -- không train toàn model:
  1. Nhóm mỗi d trọng số liền nhau theo chiều input (trong 1 hàng) thành 1 vector.
  2. Residual k-means TỪNG MA TRẬN LINEAR RIÊNG (calibration-free, trên chính trọng số
     thật của lớp đó): codebook 1 học trên vector gốc, codebook 2 học trên residual sau
     codebook 1, ... tới M codebook, mỗi codebook K entry (K-means Lloyd, vài vòng).
  3. LOCAL reconstruction (PTQ hợp lệ, giống exp_f/h/k/m): GIỮ NGUYÊN assignment (chỉ số
     entry) đã chọn ở bước 2, tinh chỉnh GIÁ TRỊ các entry codebook bằng vài chục bước
     Adam tối thiểu hoá MSE(X @ Ŵ.T, X @ W0.T) trên activation calibration THẬT -- đây
     CHÍNH LÀ kỹ thuật "fix codes, optimize codebook values" của AQLM, không phải train
     model qua data dịch.
  4. bit/weight = M*log2(K)/d (payload) + M*K*16/num_vectors (overhead lưu codebook, tính
     THẬT theo từng ma trận, không bỏ qua như free-lunch).

SO SÁNH trực tiếp với trần scalar đã biết tại CÙNG mức bpw (exp_h ternary-sparse 1.33bpw
PPL 33686/47152; exp_i/k t2:4 1.94bpw PPL 6832/21469 (TF) hay 594/8069 (SEQUENTIAL); exp_m
dense ternary 2.08bpw SEQ PPL 591/2564). FP32 baseline vi 68.97 / ja 125.07 (khớp exp_k/m).

Mục 4 của brief (VQ + sequential/BRECQ-lite): chạy 1 config đại diện (~1.5bpw, gần mức
ternary dense/t2:4) qua vòng lặp 28-block BRECQ-lite y hệt exp_k/exp_m, để xem tổ hợp
CHƯA ai thử trong lab (VQ + sequential) có ăn thêm không.
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
OUT_JSON = r"e:\Bit-Translate\eval\lowbit_ptq\exp_ad_results.json"

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

# ladder (tên, d, M, K) -- bpw danh nghĩa = M*log2(K)/d, cộng overhead codebook đo thật sau
LADDER = [
    ("VQ_d16_M1K32",   16, 1, 32),
    ("VQ_d16_M1K256",  16, 1, 256),
    ("VQ_d8_M1K16",     8, 1, 16),
    ("VQ_d16_M1K2048", 16, 1, 2048),
    ("VQ_d16_M2K256",  16, 2, 256),
    ("VQ_d8_M1K256",    8, 1, 256),
    ("VQ_d16_M2K1024", 16, 2, 1024),
    ("VQ_d16_M3K256",  16, 3, 256),
]
SEQ_CFG = ("VQ_d16_M3K256_SEQ", 16, 3, 256)  # ~1.5bpw payload, so trực tiếp với t2:4 SEQ 1.94bpw (exp_k) / dense SEQ 2.08bpw (exp_m)


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


# ---------------------------------------------------------------------------
# Residual / additive VQ (kiểu AQLM) -- k-means Lloyd vectorized, chunk theo N
# ---------------------------------------------------------------------------

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
    if K_eff < K:  # pad codebook (hiếm, chỉ khi ma trận quá nhỏ) bằng bản sao entry 0
        pad_c = centroids[0:1].repeat(K - K_eff, 1)
        centroids = torch.cat([centroids, pad_c], 0)
    return centroids


def residual_vq_init(W, d, M, K, iters=KMEANS_ITERS, sample_cap=KMEANS_SAMPLE_CAP):
    vecs, shp = build_vectors(W, d)
    residual = vecs.clone()
    codebooks, assigns = [], []
    for _m in range(M):
        centroids = kmeans_fit(residual, K, iters, sample_cap)
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
    """chi phí lưu M codebook (K entry, d chiều, f16) chia cho tổng trọng số của ma trận."""
    total_bits = M * K * d * 16.0
    total_weights = num_vectors * d
    return total_bits / total_weights


def local_refine(codebooks, assigns, shp, W0, X, steps=REFINE_STEPS, lr=REFINE_LR):
    """Fix assignment (giống AQLM 'fix codes'), tinh chỉnh GIÁ TRỊ codebook bằng Adam
    tối thiểu hoá MSE output trên activation calibration thật. Track-best: không bao giờ
    tệ hơn bản k-means init (bước đầu params == init nên best_W tự động = init)."""
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


class VQSTELinear(nn.Module):
    """Tham số HOÁ TRỰC TIẾP bằng giá trị codebook (không phải full W) -- không cần STE
    kiểu round-to-nearest vì assignment cố định; forward = tra bảng + cộng (rẻ)."""
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
    log(f"{len(linears)} ma trận linear (giữ FP: embedding tied, lm_head, mọi norm)")

    vi = read_lines(DEV_VI, N_CALIB // 2)
    ja = read_lines(DEV_JA, N_CALIB // 2)
    calib = [x for pr in zip(vi, ja) for x in pr]
    eval_vi = read_lines(DEV_VI, 400)[-N_EVAL:]
    eval_ja = read_lines(DEV_JA, 400)[-N_EVAL:]
    # ---- RESUME: nạp lại kết quả cũ nếu có (tiến trình trước chết giữa chừng vì RAM) ----
    try:
        with io.open(OUT_JSON, "r", encoding="utf-8") as f:
            results = json.load(f)
        log(f"RESUME: đã có sẵn {[k for k in results if k not in ('fp32','config')]}")
    except Exception:
        results = {}
    ppl_fp = (results.get("fp32", {}).get("ppl_vi"), results.get("fp32", {}).get("ppl_ja"))
    if ppl_fp[0] is None:
        ppl_fp = (eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja))
        results["fp32"] = {"ppl_vi": ppl_fp[0], "ppl_ja": ppl_fp[1]}
    else:
        log(f"RESUME: dùng lại FP32 đã đo vi {ppl_fp[0]:.2f} / ja {ppl_fp[1]:.2f}")
    results["config"] = {"n_calib": N_CALIB, "kmeans_iters": KMEANS_ITERS,
                          "kmeans_sample_cap": KMEANS_SAMPLE_CAP, "refine_steps": REFINE_STEPS}

    # ---- thu activation calibration (teacher-forcing, 1 lượt forward FP) + calib_ids cho SEQ ----
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

    # ---- LADDER: mỗi config quantize TOÀN BỘ 196 ma trận, đo werr + PPL ----
    for name, d, M, K in LADDER:
        if name in results:
            log(f"--- {name}: RESUME bỏ qua (đã có kết quả) ---")
            continue
        payload_bpw = M * math.log2(K) / d
        log(f"--- {name}  d={d} M={M} K={K}  (payload ~{payload_bpw:.3f} bpw) ---")
        t0 = time.time()
        sse_total, ref_total = 0.0, 0.0
        cb_bits_total, weight_total = 0.0, 0.0
        for i, (n, m) in enumerate(linears):
            W0 = orig[n]
            X = acts[n].to(torch.float32)
            codebooks, assigns, shp = residual_vq_init(W0, d, M, K)
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
        log(f"    => werr {werr*100:.1f}%  bpw(payload+codebook)={bpw_total:.3f}  PPL vi {pv:.1f} / ja {pj:.1f}")
        results[name] = {"d": d, "M": M, "K": K, "payload_bpw": payload_bpw,
                          "codebook_overhead_bpw": overhead_bpw, "bpw_total": bpw_total,
                          "werr": werr, "ppl_vi": pv, "ppl_ja": pj}
        with io.open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with torch.no_grad():
            for n, m in linears:
                m.weight.data = orig[n].clone()
        gc.collect()

    # ---- VQ + SEQUENTIAL (BRECQ-lite), 1 config đại diện ~1.5bpw ----
    name, d, M, K = SEQ_CFG
    if name in results:
        log(f"=== {name}: RESUME bỏ qua (đã có kết quả) ===")
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
            codebooks0, assigns0, shp0 = residual_vq_init(orig[key], d, M, K)
            num_vectors = assigns0[0].shape[0]
            cb_bits_total += M * K * d * 16.0
            weight_total += num_vectors * d
            sm = VQSTELinear(codebooks0, assigns0, shp0)
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

    overhead_bpw = cb_bits_total / max(weight_total, 1.0)
    bpw_total = payload_bpw + overhead_bpw
    pv, pj = eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja)
    log(f"    => SEQ PPL vi {pv:.1f} / ja {pj:.1f}  bpw_total {bpw_total:.3f}")
    results[f"{name}"] = {"d": d, "M": M, "K": K, "payload_bpw": payload_bpw,
                           "codebook_overhead_bpw": overhead_bpw, "bpw_total": bpw_total,
                           "ppl_vi": pv, "ppl_ja": pj}
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print_summary(results, ppl_fp)


def print_summary(results, ppl_fp):
    print("\n" + "=" * 88)
    print("EXP AD — VQ / ADDITIVE CODEBOOK PTQ (residual k-means + local reconstruction) trên Qwen3-0.6B")
    print("=" * 88)
    print(f"{'config':22s}{'bpw(total)':>12s}{'werr':>9s}{'PPL vi':>12s}{'PPL ja':>12s}")
    print("-" * 88)
    for k, v in results.items():
        if k in ("fp32", "config"):
            continue
        werr_s = f"{v['werr']*100:.1f}%" if "werr" in v else "  -  "
        print(f"{k:22s}{v['bpw_total']:12.3f}{werr_s:>9s}{v['ppl_vi']:12.1f}{v['ppl_ja']:12.1f}")
    print(f"{'fp32':22s}{16.0:12.2f}{'0.0%':>9s}{ppl_fp[0]:12.1f}{ppl_fp[1]:12.1f}")
    print("=" * 88)
    print("Mốc scalar đã biết (khác script, cùng dev set/config):")
    print("  exp_h ternary_sparse@0.25   1.33bpw  PPL vi 33686.7 / ja 47152.0  (TF, không sequential)")
    print("  exp_k t2:4 fixpack SEQUENTIAL 1.94bpw PPL vi   594.3 / ja  8068.6")
    print("  exp_m dense ternary g32 SEQUENTIAL 2.08bpw PPL vi 591.5 / ja 2564.2")
    print("  exp_g int4-g32 (TF recon)    4.50bpw  PPL vi   ~86   (~FP 68.97)")
    print("=" * 88)


if __name__ == "__main__":
    main()
