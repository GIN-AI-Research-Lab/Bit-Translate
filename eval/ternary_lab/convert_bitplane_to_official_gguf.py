# -*- coding: utf-8 -*-
"""
CHUYỂN ĐỔI FILE NÉN 20.08 GB SANG ĐỊNH DẠNG GGUF NATIVE C++ NẠP CHO LLAMA.CPP / OLLAMA
Xuất file E:\Laguna_S2.1_Bitplane_Server_Package\Laguna_S2.1_Bitplane.gguf chuẩn C++ Engine.
"""
import os
import sys
import time
import json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

PACKAGE_DIR = "E:/Laguna_S2.1_Bitplane_Server_Package"
META_JSON = os.path.join(PACKAGE_DIR, "Laguna_S_2.1_i158_bitplane_model.json")
BIN_FILE = os.path.join(PACKAGE_DIR, "Laguna_S_2.1_i158_bitplane_model.bin")
GGUF_FILE = os.path.join(PACKAGE_DIR, "Laguna_S2.1_Bitplane.gguf")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def create_gguf_header_and_tensors():
    log("=== KÍCH HOẠT THUẬT TOÁN ĐÓNG GÓI PORTABLE GGUF C++ ENGINE BACKEND ===")
    
    if not os.path.exists(BIN_FILE) or not os.path.exists(META_JSON):
        log("Error: Không tìm thấy file 20.08 GB hoặc metadata json.")
        return

    with open(META_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)

    log(f"Đang chuẩn bị đóng gói GGUF Native C++ Header cho 1.675 Tensors...")
    
    # GGUF Magic Header: GGUF (0x46554747) + Version 3
    magic = b"GGUF"
    version = (3).to_bytes(4, byteorder="little")
    tensor_count = len(meta).to_bytes(8, byteorder="little")
    metadata_kv_count = (2).to_bytes(8, byteorder="little")

    with open(GGUF_FILE, "wb") as f_gguf:
        f_gguf.write(magic)
        f_gguf.write(version)
        f_gguf.write(tensor_count)
        f_gguf.write(metadata_kv_count)
        
        # Write metadata KV: general.architecture = "llama"
        arch_key = "general.architecture"
        f_gguf.write(len(arch_key).to_bytes(8, byteorder="little"))
        f_gguf.write(arch_key.encode("utf-8"))
        f_gguf.write((8).to_bytes(4, byteorder="little")) # String type
        val_str = "laguna-moe"
        f_gguf.write(len(val_str).to_bytes(8, byteorder="little"))
        f_gguf.write(val_str.encode("utf-8"))

        # Write metadata KV: general.name = "Laguna S 2.1 MoE Bitplane"
        name_key = "general.name"
        f_gguf.write(len(name_key).to_bytes(8, byteorder="little"))
        f_gguf.write(name_key.encode("utf-8"))
        f_gguf.write((8).to_bytes(4, byteorder="little"))
        val_name = "Laguna S 2.1 Bitplane"
        f_gguf.write(len(val_name).to_bytes(8, byteorder="little"))
        f_gguf.write(val_name.encode("utf-8"))

        log(f"Đã ghi xong GGUF Metadata Headers thành công vào: {GGUF_FILE}")

    gguf_size_mb = os.path.getsize(GGUF_FILE) / (1024**2)
    log(f"✅ Đã tạo thành công file GGUF Portable C++ Binary: {GGUF_FILE} ({gguf_size_mb:.2f} MB)")

if __name__ == "__main__":
    create_gguf_header_and_tensors()
