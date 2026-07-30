# -*- coding: utf-8 -*-
"""
OPTION 4: BENCHMARK UNPACKER & GEMV SPEED (VECTORIZED BITPACK ENGINE)
Đo tốc độ giải mã Bitpack i2_s và phép nhân ma trận GEMV (GFLOPS & tok/s CPU)
bằng giải pháp Vectorized Bit-manipulation trong PyTorch Engine.
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
BLOCKS_OUT = OUT_DIM // 2
BLOCKS_IN = IN_DIM // 2
TOTAL_BLOCKS = BLOCKS_OUT * BLOCKS_IN

log(f"Khởi tạo ma trận nén Bitpack 2-bit (i2_s): Output={OUT_DIM}, Input={IN_DIM} ({TOTAL_BLOCKS} khối 2x2)")

w1_packed = torch.randint(0, 256, (TOTAL_BLOCKS,), dtype=torch.uint8)
w2_packed = torch.randint(0, 256, (TOTAL_BLOCKS,), dtype=torch.uint8)

scale1 = torch.rand(TOTAL_BLOCKS, dtype=torch.float32) * 0.05
scale2 = torch.rand(TOTAL_BLOCKS, dtype=torch.float32) * 0.01

x = torch.randn(IN_DIM, dtype=torch.float32)

MAP_LUT = torch.tensor([-1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

def gemv_pytorch_vectorized_i2s(w1_p, s1, w2_p, s2, vec_x):
    """
    Giải mã Bitpack i2_s và nhân Vector X tốc độ cao bằng PyTorch Vectorized Operations.
    """
    idx1_0 = (w1_p & 0x03).long()
    idx1_1 = ((w1_p >> 2) & 0x03).long()
    idx1_2 = ((w1_p >> 4) & 0x03).long()
    idx1_3 = ((w1_p >> 6) & 0x03).long()

    t1_0 = MAP_LUT[idx1_0]
    t1_1 = MAP_LUT[idx1_1]
    t1_2 = MAP_LUT[idx1_2]
    t1_3 = MAP_LUT[idx1_3]

    idx2_0 = (w2_p & 0x03).long()
    idx2_1 = ((w2_p >> 2) & 0x03).long()
    idx2_2 = ((w2_p >> 4) & 0x03).long()
    idx2_3 = ((w2_p >> 6) & 0x03).long()

    t2_0 = MAP_LUT[idx2_0]
    t2_1 = MAP_LUT[idx2_1]
    t2_2 = MAP_LUT[idx2_2]
    t2_3 = MAP_LUT[idx2_3]

    w_00 = t1_0 * s1 + t2_0 * s2
    w_01 = t1_1 * s1 + t2_1 * s2
    w_10 = t1_2 * s1 + t2_2 * s2
    w_11 = t1_3 * s1 + t2_3 * s2

    w_blocks = torch.stack([w_00, w_01, w_10, w_11], dim=1).view(BLOCKS_OUT, BLOCKS_IN, 2, 2)
    W_full = w_blocks.permute(0, 2, 1, 3).reshape(OUT_DIM, IN_DIM)

    return torch.mv(W_full, vec_x)

log("Bắt đầu Benchmark Unpacker + GEMV Execution Speed...")
# Warmup
for _ in range(10):
    _ = gemv_pytorch_vectorized_i2s(w1_packed, scale1, w2_packed, scale2, x)

ITERATIONS = 1000
start_time = time.time()
for _ in range(ITERATIONS):
    _ = gemv_pytorch_vectorized_i2s(w1_packed, scale1, w2_packed, scale2, x)
elapsed = time.time() - start_time

avg_us = (elapsed / ITERATIONS) * 1e6
ops_per_gemv = 2 * OUT_DIM * IN_DIM
gflops = (ops_per_gemv * ITERATIONS) / (elapsed * 1e9)
equivalent_tok_s = (1.0 / (avg_us * 1e-6 * 196))

log(f"=== KẾT QUẢ BENCHMARK FAST UNPACKER + GEMV (PYTORCH ENGINE) ===")
print(f"Lượt thử nghiệm: {ITERATIONS}")
print(f"Thời gian 1 phép nhân Linear ({OUT_DIM}x{IN_DIM}): {avg_us:.2f} us (micro-seconds)")
print(f"Tốc độ xử lý: {gflops:.2f} GFLOPS")
print(f"Tốc độ suy luận CPU ước tính tương ứng: {equivalent_tok_s:.1f} tok/s")
