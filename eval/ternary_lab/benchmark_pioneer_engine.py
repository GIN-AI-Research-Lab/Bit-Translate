# -*- coding: utf-8 -*-
"""
GIAI ĐOẠN 3: BENCHMARK NATIVE SIMD BIT-PLANE EXECUTION ENGINE (`i1.58_bitplane`)
Đo băng thông xử lý GFLOPS và tốc độ suy luận tok/s thực tế của ma trận nén Laguna S 2.1 MoE.
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

def run_pioneer_engine_benchmark():
    log("=== KÍCH HOẠT GIAI ĐOẠN 3: NATIVE SIMD BIT-PLANE EXECUTION ENGINE ===")
    
    # Kích thước ma trận Linear điển hình của Laguna S 2.1 MoE (4096 x 4096 & 3072 x 1024)
    LAYERS = [
        ("Expert Linear Layer (3072 x 1024)", 3072, 1024),
        ("Attention / Router Layer (4096 x 4096)", 4096, 4096)
    ]
    
    for name, OUT_DIM, IN_DIM in LAYERS:
        log(f"--- Đo đạc Ma Trận {name} ---")
        
        # Khởi tạo ma trận nén nhị phận i1.58_bitplane (nonzero_mask & sign_mask)
        nz_words = torch.randint(0, 2**31 - 1, (OUT_DIM, IN_DIM // 32), dtype=torch.int32)
        sg_words = torch.randint(0, 2**31 - 1, (OUT_DIM, IN_DIM // 32), dtype=torch.int32)
        scales = torch.rand(OUT_DIM // 2, dtype=torch.float32) * 0.05
        x = torch.randn(IN_DIM, dtype=torch.float32)
        
        # Kernel SIMD Bitwise nhị phân kép
        def gemv_i158_bitplane_simd_kernel(nz, sg, s, vec_x):
            pos_mask = (nz & sg) != 0
            neg_mask = (nz & (~sg)) != 0
            
            pos_exp = pos_mask.repeat_interleave(32, dim=1)[:, :IN_DIM]
            neg_exp = neg_mask.repeat_interleave(32, dim=1)[:, :IN_DIM]
            
            pos_sums = (pos_exp.float() * vec_x).sum(dim=1)
            neg_sums = (neg_exp.float() * vec_x).sum(dim=1)
            
            return (pos_sums - neg_sums) * s.repeat_interleave(2)
        
        # Warmup CPU caches
        for _ in range(20):
            _ = gemv_i158_bitplane_simd_kernel(nz_words, sg_words, scales, x)
            
        ITERATIONS = 1000
        start_t = time.time()
        for _ in range(ITERATIONS):
            _ = gemv_i158_bitplane_simd_kernel(nz_words, sg_words, scales, x)
        elapsed = time.time() - start_t
        
        avg_us = (elapsed / ITERATIONS) * 1e6
        gflops = (2 * OUT_DIM * IN_DIM * ITERATIONS) / (elapsed * 1e9)
        # Giả định 196 layers trong mô hình
        equivalent_tok_s = (1.0 / (avg_us * 1e-6 * 196))
        
        print("=" * 65)
        print(f"📊 KẾT QUẢ BENCHMARK {name.upper()}")
        print("=" * 65)
        print(f"Số lượt thử nghiệm: {ITERATIONS:,} runs")
        print(f"Thời gian xử lý 1 ma trận Linear: {avg_us:.2f} microseconds (us)")
        print(f"Băng thông xử lý CPU: {gflops:.2f} GFLOPS")
        print(f"Tốc độ suy luận CPU ước tính: {equivalent_tok_s:.1f} tok/s")
        print("=" * 65)
        print()

if __name__ == "__main__":
    run_pioneer_engine_benchmark()
