# -*- coding: utf-8 -*-
"""
Exp AQ — THĂM DÒ RẺ ($0, ~1 phút): Hadamard incoherence + Vector Quantization (beam-search),
kiểm tra hướng CUỐI CÙNG của QTIP chưa thử (exp_c chỉ thử Hadamard cho SCALAR ternary — hiệu
quả nhỏ 2-5%; AQLM/VQ đã thử KHÔNG Hadamard — thua scalar). QTIP/QuIP# claim SOTA 2-bit dựa
trên đúng tổ hợp NÀY: xoay Hadamard làm phân bố "incoherent" (đẳng hướng, ít outlier) TRƯỚC
khi lượng tử hóa vector — giả thuyết là VQ hưởng lợi từ Hadamard NHIỀU HƠN scalar (vì VQ vốn
nhạy với cấu trúc tương quan mà Hadamard phá vỡ).

CHỈ đo weight reconstruction error (||Ŵ-W||/||W||) trên 6 ma trận đại diện — CHƯA chạy PPL/
SEQUENTIAL/model đầy đủ. Đây là "Bước 0" đúng thứ tự "thăm dò rẻ trước khi code đầy đủ" —
nếu Hadamard không giúp VQ nhiều hơn đã giúp scalar (~2-5%), KHÔNG đầu tư thêm (không xoay
activation runtime, không sequential, không trellis-coded quantization — TCQ phức tạp hơn
AQLM nhiều, chỉ đáng làm nếu bước rẻ này cho tín hiệu tốt).

Tái dùng: random_orthogonal_for_dim (exp_c_incoherence.py, đã verify Q trực giao thật),
kmeans_fit/beam_assign/residual_vq_init_beam (exp_ao_aqlm_beam_sequential.py, đã verify beam
tốt hơn greedy thật trên tensor Qwen3-0.6B).
"""
import glob
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)

from exp_c_incoherence import random_orthogonal_for_dim  # noqa: E402
from exp_ao_aqlm_beam_sequential import (kmeans_fit, assign_nearest, beam_assign,  # noqa: E402
                                          build_vectors, reconstruct_vq, BEAM)

SNAP = glob.glob(r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*\model.safetensors")[0]

TARGETS = [
    ("model.layers.0.mlp.down_proj.weight", "L0 down_proj"),
    ("model.layers.13.self_attn.q_proj.weight", "L13 q_proj"),
    ("model.layers.13.self_attn.o_proj.weight", "L13 o_proj"),
    ("model.layers.13.mlp.gate_proj.weight", "L13 gate_proj"),
    ("model.layers.13.mlp.down_proj.weight", "L13 down_proj"),
    ("model.layers.27.mlp.down_proj.weight", "L27 down_proj"),
]

D, M, K = 8, 2, 256  # đúng cấu hình exp_ao (2.0bpw payload) để so trực tiếp được


def vq_beam_reconstruct(W, d=D, M=M, K=K):
    """residual VQ (M=2) + beam-search assign, KHÔNG refine (đo đúng tác dụng của
    representation/Hadamard, tách bạch khỏi đóng góp của gradient-refine — đúng văn hóa
    ablation lab)."""
    vecs, shp = build_vectors(W, d)
    C1 = kmeans_fit(vecs, K, iters=6, sample_cap=60000)
    a1_greedy = assign_nearest(vecs, C1)
    resid = vecs - C1[a1_greedy]
    C2 = kmeans_fit(resid, K, iters=6, sample_cap=60000)
    a1, a2 = beam_assign(vecs, C1, C2, beam=BEAM)
    Wq = reconstruct_vq([C1, C2], [a1, a2], shp)
    return Wq


def rel_err(a, b):
    return float((a - b).pow(2).sum().sqrt() / (a.pow(2).sum().sqrt() + 1e-12))


def main():
    print(f"Trọng số thật: {SNAP}\n")
    print(f"{'Ma trận':16s} {'VQ base':>10s} {'VQ+Hadamard':>13s} {'giảm':>8s}")
    print("-" * 55)
    base_list, had_list = [], []
    t0 = time.time()
    with safe_open(SNAP, framework="pt") as f:
        for key, label in TARGETS:
            W = f.get_tensor(key).float()
            R, in_dim = W.shape
            Q_np = random_orthogonal_for_dim(in_dim, seed=hash(key) & 0xFFFF)
            Q = torch.from_numpy(Q_np).float()
            Wr = W @ Q  # xoay incoherence (đúng chiều input, giống exp_c)

            Wq_base = vq_beam_reconstruct(W)
            e_base = rel_err(W, Wq_base)

            Wq_had = vq_beam_reconstruct(Wr)
            e_had = rel_err(Wr, Wq_had)  # Q trực giao -> đây CHÍNH LÀ sai số output thật

            drop = (e_base - e_had) / e_base * 100
            print(f"{label:16s} {e_base*100:9.2f}% {e_had*100:12.2f}% {drop:7.1f}%")
            base_list.append(e_base)
            had_list.append(e_had)

    b = np.mean(base_list) * 100
    h = np.mean(had_list) * 100
    print("=" * 55)
    print(f"TB VQ+beam (M=2,K=256,g=8, ~2.0bpw): base {b:.2f}%  ->  +Hadamard {h:.2f}%  "
          f"(giảm {(b-h)/b*100:.1f}% tương đối)  [{time.time()-t0:.0f}s]")
    print("So mốc exp_c (Hadamard cho SCALAR ternary): giảm 2-5% tương đối, không đủ cứu 1.58bit.")
    print("=" * 55)
    if (b - h) / b > 0.15:
        print(">>> TÍN HIỆU TỐT (giảm >15%, nhiều hơn hẳn scalar) — đáng đầu tư tiếp: xoay "
              "activation runtime + SEQUENTIAL + đo PPL đầy đủ trước khi cân nhắc TCQ.")
    else:
        print(">>> KHÔNG đủ tín hiệu (giảm <=15%, cùng bậc với scalar đã biết) — Hadamard KHÔNG "
              "giúp VQ nhiều hơn giúp scalar. Không đầu tư thêm (không TCQ, không pipeline đầy đủ).")


if __name__ == "__main__":
    main()
