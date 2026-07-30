# -*- coding: utf-8 -*-
"""
BENCHMARK PIONEER ENGINE: i1.58_bitplane VS STANDARD i2_s
Đo băng thông xử lý CPU (GFLOPS) và tốc độ suy luận ước tính (tok/s).
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

log(f"Khởi tạo Ma Trận Pioneer i1.58_bitplane: Output={OUT_DIM}, Input={IN_DIM}")

nz_words = torch.randint(0, 2**31 - 1, (OUT_DIM, IN_DIM // 32), dtype=torch.int32)
sg_words = torch.randint(0, 2**31 - 1, (OUT_DIM, IN_DIM // 32), dtype=torch.int32)
scales = torch.rand(OUT_DIM // 2, dtype=torch.float32) * 0.05
x = torch.randn(IN_DIM, dtype=torch.float32)

def gemv_simd_fast(nz, sg, s, vec_x):
    pos_mask = (nz & sg) != 0
    neg_mask = (nz & (~sg)) != 0
    
    pos_expanded = pos_mask.repeat_interleave(32, dim=1)[:, :IN_DIM]
    neg_expanded = neg_mask.repeat_interleave(32, dim=1)[:, :IN_DIM]
    
    pos_sums = (pos_expanded.float() * vec_x).sum(dim=1)
    neg_sums = (neg_expanded.float() * vec_x).sum(dim=1)
    
    return (pos_sums - neg_sums) * s.repeat_interleave(2)

log("Bắt đầu Benchmark Pioneer i1.58_bitplane Engine...")
for _ in range(10):
    _ = gemv_simd_fast(nz_words, sg_words, scales, x)

ITERATIONS = 1000
start_time = time.time()
for _ in range(ITERATIONS):
    _ = gemv_simd_fast(nz_words, sg_words, scales, x)
elapsed = time.time() - start_time

avg_us = (elapsed / ITERATIONS) * 1e6
gflops = (2 * OUT_DIM * IN_DIM * ITERATIONS) / (elapsed * 1e9)
equivalent_tok_s = (1.0 / (avg_us * 1e-6 * 196))

log(f"=== KẾT QUẢ BENCHMARK PIONEER ENGINE ===")
print(f"Lượt thử nghiệm: {ITERATIONS}")
print(f"Thời gian 1 ma trận Linear ({OUT_DIM}x{IN_DIM}): {avg_us:.2f} us")
print(f"Băng thông xử lý CPU: {gflops:.2f} GFLOPS")
print(f"Tốc độ suy luận CPU tương ứng: {equivalent_tok_s:.1f} tok/s")
