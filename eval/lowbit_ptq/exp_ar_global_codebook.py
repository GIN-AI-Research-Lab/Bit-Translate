# -*- coding: utf-8 -*-
"""
Exp AR — CODEBOOK TOÀN CỤC (global, dùng chung cho CẢ 196 ma trận) nhắm 0,03bpw — hướng
HOÀN TOÀN KHÁC mọi thử nghiệm trước (exp_ad/ae/af/ao/aq đều dùng codebook RIÊNG từng tensor,
d nhỏ 8-16, chỉ khai thác tương quan CỤC BỘ trong 1 ma trận).

GIẢ THUYẾT MỚI: có thể tồn tại một tập nhỏ "pattern phổ quát" (universal template, vector
chiều LỚN d≈256) lặp lại xuyên suốt nhiều ma trận/layer khác nhau — nếu đúng, 1 codebook DUY
NHẤT (chia sẻ overhead cho CẢ model thay vì nhân theo 196 tensor) có thể nén cực sâu vì chi phí
lưu codebook gần như = 0 (chia đều cho 440 triệu trọng số).

TOÁN: bpw = M*log2(K)/d (bỏ qua overhead codebook, KHÔNG đáng kể khi chia cho toàn model:
M=1,K=256,d=256 -> log2(256)/256 = 8/256 = 0,03125 bpw ≈ ĐÚNG mục tiêu 0,03bpw).

ĐÂY LÀ BƯỚC 0 (thăm dò rẻ, ~vài phút, CHỈ đo reconstruction error toàn cục — CHƯA SEQUENTIAL/
PPL) theo đúng kỷ luật "thăm dò rẻ trước khi đầu tư pipeline đầy đủ" của lab. Cảnh báo trước
(pre-registered): H1 (gauge-folding, 98,4% null) + H2 (intrinsic-dim, dùng 645/1024 chiều thật)
đã cho thấy KHÔNG có dư thừa dạng đơn giản ở tầng ma trận đơn lẻ — giả thuyết "pattern phổ
quát xuyên layer" là CƠ CHẾ KHÁC (chưa bị 2 test đó loại trừ trực tiếp) nên đáng thử, nhưng
xác suất thành công thấp theo đúng những gì đã biết.
"""
import glob
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
torch.set_num_threads(5)

MODEL_DIR = glob.glob(r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*")[0]
D, K = 256, 256          # log2(256)/256 = 0.03125 bpw
KMEANS_ITERS = 8
SAMPLE_CAP = 300_000     # pool that ~1.72M vector, fit tren sample nay cho nhanh


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_vectors(W, d):
    R, C = W.shape
    pad = (d - C % d) % d
    Wp = F.pad(W, (0, pad)) if pad else W
    return Wp.reshape(-1, d)


def assign_nearest(vecs, centroids, chunk=65536):
    N = vecs.shape[0]
    c_sq = (centroids * centroids).sum(1)
    out = torch.empty(N, dtype=torch.long)
    for i in range(0, N, chunk):
        v = vecs[i:i + chunk]
        v_sq = (v * v).sum(1, keepdim=True)
        dist = v_sq - 2.0 * (v @ centroids.t()) + c_sq[None, :]
        out[i:i + chunk] = dist.argmin(1)
    return out


def kmeans_fit(vecs, K, iters, sample_cap):
    N = vecs.shape[0]
    idx = torch.randperm(N)[:min(sample_cap, N)]
    pool = vecs[idx].clone()
    Np = pool.shape[0]
    init_idx = torch.randperm(Np)[:K]
    centroids = pool[init_idx].clone()
    for it in range(iters):
        assign = assign_nearest(pool, centroids)
        new_c = torch.zeros_like(centroids)
        counts = torch.zeros(K)
        new_c.index_add_(0, assign, pool)
        counts.index_add_(0, assign, torch.ones(Np))
        nonempty = counts > 0
        new_c[nonempty] = new_c[nonempty] / counts[nonempty].unsqueeze(1)
        n_empty = int((~nonempty).sum().item())
        if n_empty > 0:
            reidx = torch.randperm(Np)[:n_empty]
            new_c[~nonempty] = pool[reidx]
        centroids = new_c
        log(f"    kmeans iter {it+1}/{iters}, empty={n_empty}")
    return centroids


def main():
    from transformers import AutoModelForCausalLM
    log(f"Nạp Qwen3-0.6B FP32 từ {MODEL_DIR}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    linears = [(n, m) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and "layers." in n]
    log(f"{len(linears)} ma trận linear")

    # ---- gộp TOÀN BỘ 196 ma trận thành 1 pool vector chung (chuẩn hoá per-tensor trước khi
    # gộp, vì các loại ma trận có magnitude khác hẳn nhau (q_proj vs down_proj) -- nếu không
    # chuẩn hoá, codebook chung sẽ bị lấn át bởi ma trận magnitude lớn nhất) ----
    all_vecs, per_tensor = [], []
    t0 = time.time()
    for n, m in linears:
        W = m.weight.data.float()
        rms = W.pow(2).mean().sqrt().clamp(min=1e-8)
        vecs = build_vectors(W / rms, D)   # chuẩn hoá RMS=1 truoc khi gop
        all_vecs.append(vecs)
        per_tensor.append((n, W, rms, vecs.shape[0]))
    pool = torch.cat(all_vecs, dim=0)
    log(f"Tổng {pool.shape[0]:,} vector chiều {D} từ {len(linears)} ma trận "
        f"({pool.shape[0]*D:,} trọng số, {time.time()-t0:.0f}s)")

    log(f"Fit k-means TOÀN CỤC (K={K}, sample={SAMPLE_CAP:,})...")
    t0 = time.time()
    C = kmeans_fit(pool, K, KMEANS_ITERS, SAMPLE_CAP)
    log(f"  xong ({time.time()-t0:.0f}s)")

    log("Gán + đo sai số tái tạo TOÀN CỤC + từng loại ma trận...")
    t0 = time.time()
    by_type_sse, by_type_ref = {}, {}
    total_sse, total_ref = 0.0, 0.0
    for n, W, rms, nvec in per_tensor:
        vecs = build_vectors(W / rms, D)
        a = assign_nearest(vecs, C)
        Wq_norm = C[a].reshape(-1)[:W.numel()].reshape(W.shape)
        Wq = Wq_norm * rms
        sse = float((Wq - W).pow(2).sum())
        ref = float(W.pow(2).sum())
        total_sse += sse
        total_ref += ref
        typ = n.split(".")[-2] + "." + n.split(".")[-1] if False else \
            ".".join(n.split(".")[-2:])
        by_type_sse[typ] = by_type_sse.get(typ, 0.0) + sse
        by_type_ref[typ] = by_type_ref.get(typ, 0.0) + ref
    werr_global = (total_sse / max(total_ref, 1e-12)) ** 0.5
    log(f"  xong ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 70)
    print(f"EXP AR — CODEBOOK TOÀN CỤC (M=1,K={K},d={D}, bpw={np.log2(K)/D:.5f}) trên Qwen3-0.6B")
    print("=" * 70)
    print(f"{'loại ma trận':22s}{'werr':>10s}")
    for typ in sorted(by_type_sse):
        w = (by_type_sse[typ] / max(by_type_ref[typ], 1e-12)) ** 0.5
        print(f"{typ:22s}{w*100:9.1f}%")
    print("-" * 70)
    print(f"{'TOÀN CỤC (196 tensor)':22s}{werr_global*100:9.1f}%")
    print("=" * 70)
    print("Mốc so sánh (khác script, cùng model):")
    print("  VQ per-tensor greedy @~1.5bpw (exp_ad, M3K256):     werr 53.6%")
    print("  ternary N:M 2:4 + SEQUENTIAL @1.94bpw (exp_k):      werr thấp hơn scalar rõ rệt")
    print(f"  ĐÂY (0.03bpw, {np.log2(K)/D*50:.0f}x ít bit hơn ternary 1.5bpw):    werr {werr_global*100:.1f}%")
    if werr_global < 0.6:
        print(">>> TÍN HIỆU ĐÁNG CHÚ Ý (werr < 60% dù bpw cực thấp) — cân nhắc đo PPL thật.")
    else:
        print(">>> KHÔNG CÓ TÍN HIỆU HỮU ÍCH — werr quá cao để dùng được, đúng dự đoán từ H1/H2.")
    print("=" * 70)


if __name__ == "__main__":
    main()
