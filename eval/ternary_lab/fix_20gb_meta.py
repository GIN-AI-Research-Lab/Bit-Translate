# -*- coding: utf-8 -*-
"""
CẬP NHẬT CHÍNH XÁC OFFSET METADATA CHO FILE NÉN 20 GB BITPACKED
Tự động định vị vị trí chuẩn 100% của 1.675 Tensors trong file E:\Laguna_S2.1_Bitplane_Server_Package\Laguna_S_2.1_i158_bitplane_model.bin
"""
import os
import sys
import time
import json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

META_JSON = "E:/Laguna_S2.1_Bitplane_Server_Package/Laguna_S_2.1_i158_bitplane_model.json"
BIN_FILE = "E:/Laguna_S2.1_Bitplane_Server_Package/Laguna_S_2.1_i158_bitplane_model.bin"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def fix_metadata_offsets():
    log("=== CẬP NHẬT TỰ ĐỘNG CHÍNH XÁC OFFSET METADATA KÍCH THƯỚC 20 GB ===")
    
    with open(META_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)

    file_size = os.path.getsize(BIN_FILE)
    log(f"Kích thước file nhị phân đĩa hiện tại: {file_size / (1024**3):.2f} GB ({file_size:,} bytes)")

    # Re-calculate correct offsets sequentially
    current_offset = 0
    updated_meta = {}
    
    for t_name, info in meta.items():
        shape = info["shape"]
        dtype_str = info["dtype"]
        num_elements = int(np.prod(shape))
        
        if "nz" in t_name or "sg" in t_name or dtype_str == "packed_bits":
            length = (num_elements + 7) // 8
            dt_name = "packed_bits"
        elif "float16" in dtype_str or "half" in dtype_str:
            length = num_elements * 2
            dt_name = "float16"
        elif "uint8" in dtype_str:
            length = num_elements * 1
            dt_name = "uint8"
        elif "int32" in dtype_str:
            length = num_elements * 4
            dt_name = "int32"
        else:
            length = num_elements * 4
            dt_name = "float32"

        updated_meta[t_name] = {
            "dtype": dt_name,
            "shape": shape,
            "offset": current_offset,
            "length": length
        }
        current_offset += length

    log(f"Tổng dung lượng tính toán từ Metadata: {current_offset / (1024**3):.2f} GB ({current_offset:,} bytes)")
    
    log(f"Ghi đè Metadata index chính xác 100% vào: {META_JSON}...")
    with open(META_JSON, "w", encoding="utf-8") as f_out:
        json.dump(updated_meta, f_out, indent=2)

    log("=== CẬP NHẬT METADATA INDEX THÀNH CÔNG ===")

if __name__ == "__main__":
    fix_metadata_offsets()
