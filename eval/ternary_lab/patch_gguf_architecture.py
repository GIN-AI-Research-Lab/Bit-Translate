# -*- coding: utf-8 -*-
"""
THUẬT TOÁN PATCH TOÀN BỘ PREFIX METADATA 'laguna.' SANG 'qwen2.' CHO LLAMA.CPP
Đảm bảo llama.cpp nạp toàn bộ Hyperparameters (context_length, block_count, expert_count).
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

GGUF_PATH = "E:/Laguna_S_2.1/Laguna-S-2.1-Q4_K_M-00001-of-00002.gguf"

def patch_all_laguna_prefixes():
    print("=" * 80)
    print("🚀 CHẠY THUẬT TOÁN FULL METADATA PREFIX PATCHING CHO LLAMA.CPP ENGINE")
    print("=" * 80)

    if not os.path.exists(GGUF_PATH):
        print(f"❌ Không tìm thấy file GGUF gốc tại: {GGUF_PATH}")
        return

    with open(GGUF_PATH, "r+b") as f:
        # Read header buffer (first 10MB)
        header = bytearray(f.read(10 * 1024 * 1024))
        
        count = 0
        pos = 0
        while True:
            idx = header.find(b"laguna.", pos)
            if idx == -1:
                break
            # Replace 'laguna.' (7 bytes) with 'qwen2. ' (7 bytes) or 'qwen2..'
            header[idx:idx+7] = b"qwen2. "
            count += 1
            pos = idx + 7
            
        print(f"✅ Đã patch {count} Metadata keys từ 'laguna.' -> 'qwen2. '")
        f.seek(0)
        f.write(header)

    print("=" * 80)

if __name__ == "__main__":
    patch_all_laguna_prefixes()
