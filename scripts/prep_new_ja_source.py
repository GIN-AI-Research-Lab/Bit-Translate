#!/usr/bin/env python3
"""Lọc nguồn câu Nhật MỚI để KD dịch (vòng 4) — loại trùng corpus cũ + rác phụ đề.

Nguồn: OPUS OpenSubtitles ja mono (`D:/Bit-Translate-data/raw/os_ja.txt.gz`, 3,14M câu).
Đo trước: **48,6% đã trùng** với 11,88M câu cũ (corpus cũ vốn lấy từ OpenSubtitles)
⇒ chỉ ~1,6M câu thật sự mới. Script này lọc ra phần đó.

Rác đặc thù phụ đề cần loại: dòng chỉ có ♪ / dấu gạch, tên nhân vật viết hoa, dòng
chỉ số/dấu câu, câu quá ngắn (không đủ ngữ cảnh để dịch) hoặc quá dài (>200, thường
là nhiều câu dính nhau).

  python scripts/prep_new_ja_source.py \
      --src D:/Bit-Translate-data/raw/os_ja.txt.gz \
      --out D:/Bit-Translate-data/raw/os_ja_new.txt
"""
import argparse
import gzip
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent.parent
JA = re.compile(r"[぀-ヿ一-鿿]")
# Rác phụ đề: nốt nhạc, dòng chỉ dấu câu/số, thẻ HTML, tên nhân vật kiểu "TANAKA:"
JUNK = re.compile(r"^[\s♪♬~〜ー\-–—…・.,!?！？。、0-9０-９:：]*$")
TAG = re.compile(r"<[^>]{1,20}>")
LATIN_NAME = re.compile(r"^[A-Z][A-Z\s]{2,}[:：]")


def ok(s):
    n = len(s)
    if not (8 <= n <= 200):
        return False
    if JUNK.match(s) or LATIN_NAME.match(s):
        return False
    if len(JA.findall(s)) / n < 0.35:      # phải đủ đậm đặc chữ Nhật
        return False
    if s.count("♪") or s.count("…") > 3:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/Bit-Translate-data/raw/os_ja.txt.gz")
    ap.add_argument("--out", default="D:/Bit-Translate-data/raw/os_ja_new.txt")
    ap.add_argument("--old", default=str(ROOT / "data" / "full_11.88m_ja_clean.txt"))
    ap.add_argument("--limit", type=int, default=0, help="0 = không giới hạn")
    a = ap.parse_args()

    print("nạp corpus cũ để loại trùng...", flush=True)
    old = set()
    with open(a.old, encoding="utf-8") as f:
        for line in f:
            old.add(line.strip())
    print(f"  {len(old):,} câu cũ", flush=True)

    opener = gzip.open if a.src.endswith(".gz") else open
    seen = set()
    n = kept = dup_old = dup_self = bad = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with opener(a.src, "rt", encoding="utf-8", errors="ignore") as fin, \
            open(a.out, "w", encoding="utf-8") as fo:
        for line in fin:
            n += 1
            s = TAG.sub("", unicodedata.normalize("NFKC", line).strip())
            if not ok(s):
                bad += 1
                continue
            if s in old:
                dup_old += 1
                continue
            if s in seen:
                dup_self += 1
                continue
            seen.add(s)
            fo.write(s + "\n")
            kept += 1
            if a.limit and kept >= a.limit:
                break
            if n % 500_000 == 0:
                print(f"  ...{n:,} đọc | giữ {kept:,}", flush=True)

    print(f"\n=== LỌC XONG ===")
    print(f"Đọc          : {n:,}")
    print(f"GIỮ (câu mới): {kept:,} ({100*kept/max(1,n):.1f}%)")
    print(f"Trùng cũ     : {dup_old:,} | trùng nội bộ: {dup_self:,} | rác/loại: {bad:,}")
    print(f"Output       : {a.out}")
    print(f"\nDịch bằng:  KD_INPUT={a.out} KD_OUTPUT=D:/Bit-Translate-data/raw/kd_os_new.jsonl \\")
    print(f"            WORKERS_PER_KEY=6 BATCH_SIZE=1 python scripts/run_kd_batch.py")


if __name__ == "__main__":
    main()
