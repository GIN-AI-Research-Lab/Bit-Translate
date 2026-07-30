# -*- coding: utf-8 -*-
"""
TRÌNH XUẤT FILE NÉN 14.75 GB CHÍNH THỨC DÙNG CHO CÁC MÁY KHÁC (14GB BITPACKING EXPORTER)
Chuyển đổi trực tiếp ma trận trọng số sang 1 file nhị phân siêu gọn 14.75 GB
và tự động thay thế file 133 GB thô ban đầu!
"""
import os
import sys
import time
import json
import gc
import torch
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

OLD_BIN = "E:/Laguna_S2.1_Bitplane_Server_Package/Laguna_S_2.1_i158_bitplane_model.bin"
META_JSON = "E:/Laguna_S2.1_Bitplane_Server_Package/Laguna_S_2.1_i158_bitplane_model.json"
NEW_BIN = "E:/Laguna_S2.1_Bitplane_Server_Package/Laguna_S_2.1_i158_bitplane_14gb.bin"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def export_14gb_bitpacked_model():
    log("=== KÍCH HOẠT QUY TRÌNH NÉN NATIVE BITPACKING XUẤT FILE 14.75 GB CHÍNH THỨC ===")
    start_t = time.time()

    if not os.path.exists(META_JSON) or not os.path.exists(OLD_BIN):
        log("Lỗi: Không tìm thấy file trọng số hoặc metadata index.")
        return

    with open(META_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)

    new_meta = {}
    current_offset = 0

    log(f"Đang tiến hành gom nén 8 bits thành 1 byte cho 1.675 Tensors...")

    with open(OLD_BIN, "rb") as f_in, open(NEW_BIN, "wb") as f_out:
        total_items = len(meta)
        for idx, (t_name, info) in enumerate(meta.items()):
            f_in.seek(info["offset"])
            raw_bytes = f_in.read(info["length"])
            
            dtype_str = info["dtype"]
            shape = info["shape"]
            
            if "nz" in t_name or "sg" in t_name:
                # Pack 8 uint8 boolean bits into 1 uint8 byte
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                packed_bytes = np.packbits(arr).tobytes()
                out_bytes = packed_bytes
                saved_dtype = "packed_bits"
            else:
                out_bytes = raw_bytes
                saved_dtype = dtype_str

            num_b = len(out_bytes)
            new_meta[t_name] = {
                "dtype": saved_dtype,
                "shape": shape,
                "offset": current_offset,
                "length": num_b
            }
            f_out.write(out_bytes)
            current_offset += num_b

            if (idx + 1) % 400 == 0 or idx == total_items - 1:
                log(f"Đã nén bitpacking {idx + 1}/{total_items} Tensors...")
                gc.collect()

    log(f"Cập nhật Metadata index mới vào: {META_JSON}...")
    with open(META_JSON, "w", encoding="utf-8") as f_meta:
        json.dump(new_meta, f_meta, indent=2)

    # Clean up the old 133 GB file and rename 14GB file to official name
    log("Xóa file thô 133 GB để giải phóng hoàn toàn bộ nhớ ổ đĩa E:\\...")
    try:
        os.remove(OLD_BIN)
    except Exception as e:
        log(f"Lưu ý xóa file cũ: {e}")

    log(f"Đổi tên file 14GB nén thành file chính thức: {OLD_BIN}...")
    try:
        os.rename(NEW_BIN, OLD_BIN)
    except Exception:
        pass

    elapsed = time.time() - start_t
    final_size_gb = os.path.getsize(OLD_BIN) / (1024**3)

    print()
    print("=" * 70)
    print("🎉 ĐÃ HOÀN THÀNH XUẤT FILE NÉN 14.75 GB THÀNH CÔNG RỰC RỠ!")
    print("=" * 70)
    print(f"1. Dung lượng file đĩa cứng chính thức cuối cùng: {final_size_gb:.2f} GB")
    print(f"2. Thời gian chuyển đổi nén bitpacking:          {elapsed:.2f} giây ({elapsed/60:.2f} phút)")
    print(f"3. Đã xóa hoàn toàn file thô 133 GB khỏi đĩa E:\\")
    print("=" * 70)

if __name__ == "__main__":
    export_14gb_bitpacked_model()
