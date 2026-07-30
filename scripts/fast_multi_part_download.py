# -*- coding: utf-8 -*-
"""
TRÌNH TẢI ĐA LUỒNG BĂNG THÔNG SIÊU TỐC TỰ ĐỘNG PHỤC HỒI (EXACT SIZE CHECK + 16 THREADS)
Tự động nối tiếp tiến độ, thử lại 5 lần khi mất mạng, không bao giờ mất dữ liệu đã tải.
"""
import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_url

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

TARGET_DIR = "E:/Laguna_S_2.1"
os.makedirs(TARGET_DIR, exist_ok=True)

REPO_ID = "bartowski/Laguna-S-2.1-GGUF"
FILES = [
    "Laguna-S-2.1-Q4_K_M/Laguna-S-2.1-Q4_K_M-00001-of-00002.gguf",
    "Laguna-S-2.1-Q4_K_M/Laguna-S-2.1-Q4_K_M-00002-of-00002.gguf"
]

NUM_THREADS = 16
CHUNK_SIZE = 64 * 1024 * 1024 # 64 MB per chunk

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def download_chunk_part_robust(cdn_url, start, end, chunk_path, max_retries=5):
    expected_len = end - start + 1
    if os.path.exists(chunk_path) and os.path.getsize(chunk_path) == expected_len:
        return expected_len
        
    for attempt in range(max_retries):
        try:
            headers = {"Range": f"bytes={start}-{end}"}
            r = requests.get(cdn_url, headers=headers, stream=True, timeout=45)
            r.raise_for_status()
            
            with open(chunk_path + ".tmp", "wb") as f:
                for chunk in r.iter_content(chunk_size=512*1024):
                    if chunk:
                        f.write(chunk)
                        
            if os.path.getsize(chunk_path + ".tmp") == expected_len:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                os.rename(chunk_path + ".tmp", chunk_path)
                return expected_len
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)
            else:
                raise e

def download_lfs_file_instant(rel_filepath):
    fname = os.path.basename(rel_filepath)
    filepath = os.path.join(TARGET_DIR, fname)
    tmp_dir = os.path.join(TARGET_DIR, f"{fname}_parts")
    os.makedirs(tmp_dir, exist_ok=True)

    hub_url = hf_hub_url(repo_id=REPO_ID, filename=rel_filepath)
    
    log(f"Đang kết nối CDN HuggingFace cho file: {fname}...")
    res = requests.head(hub_url, allow_redirects=True)
    cdn_url = res.url
    file_size = int(res.headers.get("Content-Length", 0))
    
    if file_size < 1000:
        res_get = requests.get(hub_url, allow_redirects=True, stream=True)
        cdn_url = res_get.url
        file_size = int(res_get.headers.get("Content-Length", 0))

    file_size_gb = file_size / (1024**3)
    
    if os.path.exists(filepath):
        if os.path.getsize(filepath) == file_size:
            log(f"File {fname} ({file_size_gb:.2f} GB) đã hoàn tất 100% trên đĩa E:\\, bỏ qua.")
            return
        else:
            log(f"Xóa file dở dang cũ {fname} ({os.path.getsize(filepath)/(1024**3):.2f} GB) để ghép từ parts...")
            os.remove(filepath)

    log(f"Dung lượng thực tế: {fname} = {file_size_gb:.2f} GB ({file_size:,} bytes)")

    # Build chunk ranges
    chunks = []
    chunk_paths = []
    idx = 0
    for start in range(0, file_size, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE - 1, file_size - 1)
        cpath = os.path.join(tmp_dir, f"part_{idx:05d}.tmp")
        chunks.append((start, end, cpath))
        chunk_paths.append(cpath)
        idx += 1

    existing_bytes = sum(os.path.getsize(c) for c in chunk_paths if os.path.exists(c))
    log(f"TẢI NỐI TIẾP (16 LUỒNG SONG SONG): Đã có sẵn {existing_bytes / (1024**3):.2f} GB / {file_size_gb:.2f} GB. Tải tiếp phần còn lại...")
    
    start_t = time.time()
    downloaded_bytes = existing_bytes

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(download_chunk_part_robust, cdn_url, c[0], c[1], c[2]): c
            for c in chunks
        }
        
        for future in as_completed(futures):
            b = future.result()
            
            elapsed = time.time() - start_t
            curr_existing = sum(os.path.getsize(c) for c in chunk_paths if os.path.exists(c))
            speed_mb = ((curr_existing - existing_bytes) / (1024**2)) / max(elapsed, 0.1)
            curr_gb = curr_existing / (1024**3)
            pct = (curr_existing / file_size) * 100.0
            
            if curr_existing % (256 * 1024 * 1024) < CHUNK_SIZE or curr_existing == file_size:
                log(f"PROGRESS [{fname}]: {curr_gb:.2f}/{file_size_gb:.2f} GB ({pct:.1f}%) | TỐC ĐỘ: {speed_mb:.2f} MB/s")

    log(f"Đang ghép nối {len(chunk_paths)} chunks thành file hoàn chỉnh {fname}...")
    with open(filepath, "wb") as outfile:
        for cpath in chunk_paths:
            with open(cpath, "rb") as infile:
                outfile.write(infile.read())
            os.remove(cpath)
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass
    log(f"=== TẢI VÀ GHÉP NỐI HOÀN TẤT FILE {fname} ===")

def main():
    log("=== KÍCH HOẠT TRÌNH TẢI ĐA LUỒNG TỰ ĐỘNG PHỤC HỒI (EXACT SIZE CHECK) ===")
    for rel_path in FILES:
        download_lfs_file_instant(rel_path)
    log("=== ĐÃ TẢI XONG TOÀN BỘ 2 FILE LAGUNA S 2.1 Q4_K_M (~65 GB) ===")

if __name__ == "__main__":
    main()
