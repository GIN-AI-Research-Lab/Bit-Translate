# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH GỘP TOÀN BỘ CHUNKS THÀNH FILE SAFETENSORS DUY NHẤT
Mục tiêu: Đóng gói 814 file chunk rải rác trên đĩa E:\ thành 1 file Safetensors chuẩn siêu gọn
triệt tiêu 100% Header overhead của PyTorch Zip.
"""
import os
import sys
import time
import gc
import torch
from safetensors.torch import save_file

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

CHUNK_DIR = "E:/Laguna_S_2.1_exported_chunks"
OUTPUT_FILE = "E:/Laguna_S_2.1_i158_bitplane_model.safetensors"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def consolidate_chunks_to_safetensors():
    log(f"=== BẮT ĐẦU ĐÓNG GÓI 814 CHUNKS THÀNH FILE SAFETENSORS DUY NHẤT ===")
    start_t = time.time()
    
    files = [f for f in os.listdir(CHUNK_DIR) if f.endswith(".pt")]
    log(f"Tìm thấy {len(files)} file chunk `.pt` trên đĩa E:\\...")

    consolidated_tensors = {}
    
    for idx, fname in enumerate(files):
        fpath = os.path.join(CHUNK_DIR, fname)
        base_name = fname[:-3] # Bỏ đuôi .pt
        
        try:
            chunk = torch.load(fpath, map_location="cpu")
            if "nz" in chunk:
                consolidated_tensors[f"{base_name}.nz"] = chunk["nz"]
                consolidated_tensors[f"{base_name}.sg"] = chunk["sg"]
                consolidated_tensors[f"{base_name}.scale"] = chunk["scale"]
            elif "raw" in chunk:
                consolidated_tensors[base_name] = chunk["raw"]
        except Exception as e:
            log(f"Lỗi đọc file chunk {fname}: {e}")

        if (idx + 1) % 100 == 0:
            log(f"Đã nạp {idx + 1}/{len(files)} chunks vào bộ nhớ tạm...")

    log(f"Lưu toàn bộ ma trận nén thành file Safetensors chuẩn: {OUTPUT_FILE}...")
    save_file(consolidated_tensors, OUTPUT_FILE)
    
    elapsed = time.time() - start_t
    final_size_gb = os.path.getsize(OUTPUT_FILE) / (1024**3)

    log(f"=== HOÀN THÀNH ĐÓNG GÓI SAFETENSORS ===")
    print(f"Tổng số Tensor trong file: {len(consolidated_tensors)}")
    print(f"Thời gian đóng gói: {elapsed:.2f} giây")
    print(f"Dung lượng file Safetensors chuẩn cuối cùng trên ổ E:\\: {final_size_gb:.2f} GB")

if __name__ == "__main__":
    consolidate_chunks_to_safetensors()
