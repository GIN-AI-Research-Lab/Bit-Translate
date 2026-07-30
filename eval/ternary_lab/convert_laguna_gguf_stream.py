# -*- coding: utf-8 -*-
"""
TRÌNH NÉN GGUF MOE LAGUNA S 2.1 THỜI ĐẠI MỚI (ZERO-RAM ACCUMULATION + DRIVE E:\ SAFE EXPORTER)
Nén trực tiếp 814 Tensors từ GGUF 66.83 GB sang đĩa E:\ (970GB Free) tự động loại bỏ file hỏng 0-byte.
"""
import os
import sys
import time
import gc
import torch
import numpy as np
from gguf import GGUFReader

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

GGUF_FILE_1 = "E:/Laguna_S_2.1/Laguna-S-2.1-Q4_K_M-00001-of-00002.gguf"
GGUF_FILE_2 = "E:/Laguna_S_2.1/Laguna-S-2.1-Q4_K_M-00002-of-00002.gguf"

# Đường dẫn xuất file trên ổ đĩa E:\ khổng lồ (970GB Free)
EXPORT_DIR = "E:/Laguna_S_2.1_exported_chunks"
os.makedirs(EXPORT_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def quantize_block2x2_tensor(W_reshaped):
    scales = W_reshaped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)
    W_ternary = torch.round(W_reshaped / scales).clamp(-1, 1)
    nz = (W_ternary != 0).to(torch.uint8)
    sg = (W_ternary == 1).to(torch.uint8)
    return nz, sg, scales

def run_laguna_gguf_stream_compression():
    log(f"=== KÍCH HOẠT TRÌNH NÉN STREAMING MOE TRÊN Ổ HDD E:\\: LAGUNA S 2.1 (118B) ===")
    start_t = time.time()
    
    # Clean zero-byte corrupted chunk files if any
    for fn in os.listdir(EXPORT_DIR):
        fp = os.path.join(EXPORT_DIR, fn)
        if os.path.isfile(fp) and os.path.getsize(fp) < 10:
            try:
                os.remove(fp)
            except OSError:
                pass

    readers = []
    for gpath in [GGUF_FILE_1, GGUF_FILE_2]:
        if os.path.exists(gpath):
            log(f"Đọc Metadata GGUF: {os.path.basename(gpath)}...")
            readers.append(GGUFReader(gpath))
            
    total_tensors = sum(len(r.tensors) for r in readers)
    log(f"Tìm thấy tổng cộng {total_tensors} Tensors trong GGUF MoE.")

    converted_count = 0
    skipped_count = 0

    for r_idx, reader in enumerate(readers):
        log(f"--- Đang nén các Tensors từ Part {r_idx + 1}/{len(readers)} ---")
        for tensor in reader.tensors:
            t_name = tensor.name.replace("/", "_").replace(".", "_")
            out_chunk_path = os.path.join(EXPORT_DIR, f"{t_name}.pt")
            
            # TỰ ĐỘNG BỎ QUA CÁC CHUNKS ĐÃ NÉN THÀNH CÔNG (> 100 bytes)
            if os.path.exists(out_chunk_path) and os.path.getsize(out_chunk_path) > 100:
                converted_count += 1
                continue

            # Read raw tensor data
            arr_copy = np.copy(tensor.data)
            
            chunk_dict = {}
            if "weight" in tensor.name and len(tensor.shape) == 2 and any(k in tensor.name for k in ["attn", "ffn", "mlp", "proj", "expert"]):
                W = torch.from_numpy(arr_copy.astype(np.float32) if arr_copy.dtype != np.float32 else arr_copy)
                
                out_dim, in_dim = W.shape
                if out_dim % 2 == 0 and in_dim % 2 == 0:
                    W_reshaped = W.view(out_dim // 2, 2, in_dim // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
                    nz, sg, scale = quantize_block2x2_tensor(W_reshaped)
                    
                    chunk_dict["nz"] = nz.contiguous()
                    chunk_dict["sg"] = sg.contiguous()
                    chunk_dict["scale"] = scale.half().contiguous()
                    converted_count += 1
                else:
                    chunk_dict["raw"] = W.half().contiguous()
                    skipped_count += 1
            else:
                chunk_dict["raw"] = torch.from_numpy(arr_copy).half().contiguous()
                skipped_count += 1

            # Save single chunk file directly to drive E:\ and purge RAM
            tmp_target = out_chunk_path + ".tmp"
            torch.save(chunk_dict, tmp_target)
            if os.path.exists(out_chunk_path):
                try:
                    os.remove(out_chunk_path)
                except OSError:
                    pass
            os.rename(tmp_target, out_chunk_path)
            
            del arr_copy, chunk_dict
            if (converted_count + skipped_count) % 50 == 0:
                log(f"Đã nén & lưu trực tiếp ổ E:\\: {converted_count + skipped_count}/{total_tensors} Tensors...")
                gc.collect()

    elapsed = time.time() - start_t
    
    total_disk_bytes = sum(os.path.getsize(os.path.join(EXPORT_DIR, f)) for f in os.listdir(EXPORT_DIR))
    out_file_size_gb = total_disk_bytes / (1024**3)

    log(f"=== KẾT QUẢ NÉN MOE TRÊN Ổ E:\\ THÀNH CÔNG RỰC RỠ ===")
    print(f"Tổng số Tensor ma trận Linear đã nén $i2_s$ / Bitplane: {converted_count}")
    print(f"Thời gian nén theo luồng: {elapsed:.2f} giây ({elapsed/60:.2f} phút)")
    print(f"Dung lượng đĩa mô hình nén đầu ra trên ổ E:\\: {out_file_size_gb:.2f} GB")
    print(f"Tỷ lệ nén: Giảm từ ~66.83 GB gốc xuống còn {out_file_size_gb:.2f} GB (~{66.83/max(out_file_size_gb, 0.1):.1f}x nhỏ hơn!)")

if __name__ == "__main__":
    run_laguna_gguf_stream_compression()
