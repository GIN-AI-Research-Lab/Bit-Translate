# -*- coding: utf-8 -*-
"""
SCRIPT TẢI SIÊU TỐC MÔ HÌNH LAGUNA S 2.1 Q4_K_M VỀ Ổ ĐĨA HẠNG NẶNG E:\ (HF_XET_HIGH_PERFORMANCE ENABLED)
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

TARGET_DIR = "E:/Laguna_S_2.1"
os.makedirs(TARGET_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log(f"Kích hoạt HF Xet High Performance Transfer | Lưu trữ tại ổ HDD E: {TARGET_DIR}")

from huggingface_hub import snapshot_download

REPO_ID = "bartowski/Laguna-S-2.1-GGUF"

log(f"Bắt đầu tải siêu tốc mô hình Laguna-S-2.1 Q4_K_M từ Hugging Face Repo: {REPO_ID}...")

try:
    downloaded_path = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=["*Q4_K_M*.gguf"],
        local_dir=TARGET_DIR,
        max_workers=8
    )
    log(f"=== KHỞI CHẠY TẢI THÀNH CÔNG ===")
    log(f"Đã lưu tại: {downloaded_path}")
except Exception as e:
    log(f"Lỗi tải từ bartowski ({e}), thử repo dự phòng poolside/Laguna-S-2.1-GGUF...")
    downloaded_path = snapshot_download(
        repo_id="poolside/Laguna-S-2.1-GGUF",
        allow_patterns=["*Q4_K_M*.gguf"],
        local_dir=TARGET_DIR,
        max_workers=8
    )
    log(f"=== KHỞI CHẠY TẢI THÀNH CÔNG DỰ PHÒNG ===")
    log(f"Đã lưu tại: {downloaded_path}")
