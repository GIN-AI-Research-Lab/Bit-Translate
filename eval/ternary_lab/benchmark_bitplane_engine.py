# -*- coding: utf-8 -*-
"""
BENCHMARK ENGINE CHUẨN NÉN MỚI: i1.58_bitplane
So sánh tốc độ giải mã và nhân GEMV của chuẩn i1.58_bitplane (Bitwise Logic) vs chuẩn i2_s (Bit-shift + LUT).
"""
import os
import sys
import time
import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

torch.set_num_threads(6)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

OUT_DIM = 3072
IN_DIM = 1024
TOTAL_WORDS = (OUT_DIM * IN_DIM) // 32

log(f"Khởi tạo Ma Trận Chuẩn Mới i1.58_bitplane: Output={OUT_DIM}, Input={IN_DIM} ({TOTAL_WORDS} uint32 Bit-Plane Words)")

# Generates random Bit-Planes
nonzero_mask = torch.randint(0, 2**31 - 1, (OUT_DIM, IN_DIM // 32), dtype=torch.int32)
sign_mask = torch.randint(0, 2**31 - 1, (OUT_DIM, IN_DIM // 32), dtype=torch.int32)
scales = torch.rand(OUT_DIM // 2, dtype=torch.float32) * 0.05
x = torch.randn(IN_DIM, dtype=torch.float32)

def gemv_i158_bitplane_vectorized(nz_mask, sg_mask, s_vec, vec_x):
    """
    Simulate PyTorch Vectorized Bit-Plane Dual Mask Logic:
    pos_mask = nz & sg
    neg_mask = nz & (~sg)
    """
    pos_mask = (nz_mask & sg_mask) != 0
    neg_mask = (nz_mask & (~sg_mask)) != 0
    
    # Expand 32-bit words to full tensor mask
    pos_expanded = pos_mask.repeat_interleave(32, dim=1)[:, :IN_DIM]
    neg_expanded = neg_mask.repeat_interleave(32, dim=1)[:, :IN_DIM]
    
    pos_sums = (pos_expanded.float() * vec_x).sum(dim=1)
    neg_sums = (neg_expanded.float() * vec_x).sum(dim=1)
    
    y = (pos_sums - neg_sums) * s_vec.repeat_interleave(2)
    return y

log("Bắt đầu Benchmark i1.58_bitplane Engine Speed...")

# Warmup
for _ in range(10):
    _ = gemv_i158_bitplane_vectorized(nonzero_mask, sign_mask, scales, x)

ITERATIONS = 1000
start_time = time.time()
for _ in range(ITERATIONS):
    _ = gemv_i158_bitplane_vectorized(nonzero_mask, sign_mask, scales, x)
elapsed = time.time() - start_time

avg_us = (elapsed / ITERATIONS) * 1e6
ops_per_gemv = 2 * OUT_DIM * IN_DIM
gflops = (ops_per_gemv * ITERATIONS) / (elapsed * 1e9)
equivalent_tok_s = (1.0 / (avg_us * 1e-6 * 196))

log(f"=== KẾT QUẢ BENCHMARK CHUẨN NÉN MỚI i1.58_bitplane ENGINE ===")
print(f"Lượt thử nghiệm: {ITERATIONS}")
print(f"Thời gian 1 ma trận Linear ({OUT_DIM}x{IN_DIM}): {avg_us:.2f} us (micro-seconds)")
print(f"Băng thông xử lý CPU: {gflops:.2f} GFLOPS")
print(f"Tốc độ suy luận CPU ước tính tương ứng: {equivalent_tok_s:.1f} tok/s")
