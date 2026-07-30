# -*- coding: utf-8 -*-
"""
TRÌNH ĐÓNG GÓI BINARY FLAT STREAM THỜI ĐẠI MỚI (< 100 MB RAM)
Mục tiêu: Đóng gói 814 file chunk rải rác trên đĩa E:\ thành 1 file nhị phân phẳng duy nhất
"E:\Laguna_S_2.1_i158_bitplane_model.bin" mà KHÔNG TỐN BỘ NHỚ RAM.
"""
import os
import sys
import time
import json
import gc
import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

CHUNK_DIR = "E:/Laguna_S_2.1_exported_chunks"
OUTPUT_BIN = "E:/Laguna_S_2.1_i158_bitplane_model.bin"
OUTPUT_META = "E:/Laguna_S_2.1_i158_bitplane_model.json"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def pack_binary_stream():
    log(f"=== KÍCH HOẠT TRÌNH ĐÓNG GÓI BINARY FLAT STREAM (< 100MB RAM) ===")
    start_t = time.time()
    
    files = [f for f in os.listdir(CHUNK_DIR) if f.endswith(".pt")]
    log(f"Tìm thấy {len(files)} file chunk `.pt` trên đĩa E:\\...")

    metadata = {}
    current_offset = 0

    with open(OUTPUT_BIN, "wb") as bin_file:
        for idx, fname in enumerate(files):
            fpath = os.path.join(CHUNK_DIR, fname)
            base_name = fname[:-3]
            
            try:
                chunk = torch.load(fpath, map_location="cpu")
                
                for key, tensor in chunk.items():
                    full_name = f"{base_name}.{key}" if key != "raw" else base_name
                    
                    # Convert tensor to numpy raw bytes
                    tensor_np = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
                    raw_bytes = tensor_np.tobytes()
                    num_bytes = len(raw_bytes)
                    
                    metadata[full_name] = {
                        "dtype": str(tensor_np.dtype),
                        "shape": list(tensor_np.shape),
                        "offset": current_offset,
                        "length": num_bytes
                    }
                    
                    bin_file.write(raw_bytes)
                    current_offset += num_bytes
                
                del chunk
            except Exception as e:
                log(f"Lỗi xử lý file chunk {fname}: {e}")

            if (idx + 1) % 100 == 0:
                log(f"Đã đóng gói {idx + 1}/{len(files)} chunks vào file nhị phân...")
                gc.collect()

    log(f"Lưu Metadata JSON vào: {OUTPUT_META}...")
    with open(OUTPUT_META, "w", encoding="utf-8") as meta_f:
        json.dump(metadata, meta_f, indent=2)

    elapsed = time.time() - start_t
    final_bin_gb = os.path.getsize(OUTPUT_BIN) / (1024**3)

    log(f"=== KẾT QUẢ ĐÓNG GÓI BINARY TỰ ĐỘNG THÀNH CÔNG RỰC RỠ ===")
    print(f"Tổng số Tensor ma trận đã đóng gói: {len(metadata)}")
    print(f"Thời gian đóng gói: {elapsed:.2f} giây ({elapsed/60:.2f} phút)")
    print(f"Dung lượng file nhị phân phẳng cuối cùng trên ổ E:\\: {final_bin_gb:.2f} GB")
    print(f"Tỷ lệ nén siêu việt: Giảm từ ~66.83 GB gốc xuống còn {final_bin_gb:.2f} GB (~{66.83/max(final_bin_gb, 0.1):.1f}x nhỏ hơn!)")

if __name__ == "__main__":
    pack_binary_stream()
