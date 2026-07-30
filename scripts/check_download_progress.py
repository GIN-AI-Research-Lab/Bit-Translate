# -*- coding: utf-8 -*-
"""
SCRIPT KIỂM TRA TIẾN ĐỘ TẢI HÀNG NẶNG LAGUNA S 2.1 Q4_K_M (Ổ E:\)
Hiển thị Thanh tiến độ (Progress Bar), Dung lượng GB và Báo tải xong rõ ràng.
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

TARGET_DIR = "E:/Laguna_S_2.1"
TOTAL_EXPECTED_GB = 66.83

FILES_EXPECTED = [
    ("Laguna-S-2.1-Q4_K_M-00001-of-00002.gguf", 37.09),
    ("Laguna-S-2.1-Q4_K_M-00002-of-00002.gguf", 29.74)
]

def check_progress():
    print("=" * 65)
    print("📊 BÁO CÁO TIẾN ĐỘ TẢI LAGUNA S 2.1 Q4_K_M (Ổ HDD E:\\)")
    print("=" * 65)
    
    if not os.path.exists(TARGET_DIR):
        print(f"Thư mục {TARGET_DIR} chưa được tạo.")
        print("=" * 65)
        return
        
    total_bytes = 0
    all_done = True
    
    for fname, expected_gb in FILES_EXPECTED:
        fpath = os.path.join(TARGET_DIR, fname)
        parts_dir = os.path.join(TARGET_DIR, f"{fname}_parts")
        
        if os.path.exists(fpath):
            fsize = os.path.getsize(fpath) / (1024**3)
            if abs(fsize - expected_gb) < 0.5:
                print(f"  ✅ [HOÀN THÀNH 100%] {fname} ({fsize:.2f} GB)")
                total_bytes += os.path.getsize(fpath)
            else:
                print(f"  🔄 [ĐANG GHÉP NỐI/XỬ LÝ] {fname} ({fsize:.2f} GB / {expected_gb:.2f} GB)")
                total_bytes += os.path.getsize(fpath)
                all_done = False
        elif os.path.exists(parts_dir):
            parts_size = sum(os.path.getsize(os.path.join(parts_dir, f)) for f in os.listdir(parts_dir) if os.path.isfile(os.path.join(parts_dir, f)))
            parts_gb = parts_size / (1024**3)
            pct = min(100.0, (parts_gb / expected_gb) * 100.0)
            print(f"  📥 [ĐANG TẢI...] {fname}: {parts_gb:.2f} GB / {expected_gb:.2f} GB ({pct:.1f}%)")
            total_bytes += parts_size
            all_done = False
        else:
            print(f"  ⏳ [CHỜ TẢI...] {fname} (Dự kiến {expected_gb:.2f} GB)")
            all_done = False

    curr_gb = total_bytes / (1024**3)
    percent = min(100.0, (curr_gb / TOTAL_EXPECTED_GB) * 100.0)
    
    # Progress bar graphic
    bar_len = 30
    filled_len = int(bar_len * percent // 100)
    bar = '█' * filled_len + '-' * (bar_len - filled_len)
    
    print("-" * 65)
    print(f"Thanh tiến độ: [{bar}] {percent:.2f}%")
    print(f"Tổng dung lượng đã nạp: {curr_gb:.2f} GB / ~{TOTAL_EXPECTED_GB:.2f} GB")
    print("-" * 65)
    
    if all_done and curr_gb >= 65.0:
        print("🎉 TẤT CẢ FILE ĐÃ ĐƯỢC TẢI XONG 100% HOÀN CHỈNH TRÊN Ổ HDD E:\\!")
        print("BẠN ĐÃ SẴN SÀNG ĐỂ TIẾN HÀNH THỬ NGHIỆM NÉN SANG CHUẨN BITPLANE!")
    else:
        print("Tiến trình ghép nối / tải những Chunks cuối cùng đang chạy ở background...")
    print("=" * 65)

if __name__ == "__main__":
    check_progress()
