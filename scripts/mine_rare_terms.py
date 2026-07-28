#!/usr/bin/env python3
"""Đào kho CC-100 gốc (417M câu) — CHỌN câu chứa thuật ngữ đang dưới ngưỡng.

Một lượt quét làm ba việc cùng lúc:
  1. KIỂM tính đồng đều: đếm riêng theo từng lô 40M câu -> nếu phân bố thuật ngữ giữa
     các lô lệch nhau thì ngoại suy ×21 ở bản thăm dò (lấy mẫu ĐẦU file) là sai.
  2. CHỌN câu: mỗi thuật ngữ thiếu lấy tối đa (thiếu bao nhiêu × 1,3) câu, để dư biên
     cho khâu QC loại bỏ (~15% theo các vòng trước).
  3. ĐẾM đầy đủ trên toàn kho -> thay số ngoại suy bằng số thật.

Vì sao chọn theo THUẬT NGỮ chứ không theo chủ đề: đo được ở DOMAIN_MAP.md §2 — thêm
câu "nói về nông nghiệp" không thêm từ vựng nông nghiệp. Kho rộng nhưng nông. Chọn theo
từ đang thiếu thì chỉ +9% corpus mà vá 50-80% lỗ hổng.

HAI CHỖ TỪNG LÀM CHẬM, ĐÃ ĐO VÀ SỬA (giữ lại để không lặp):
  - module `lzma` của Python ~4 MB/s -> đọc stdin từ `xz -dc` (133 MB/s), nhanh 33×.
  - `pyahocorasick.iter()` 7,1 MB/s (chi phí generator Python + trie thưa Unicode)
    -> `ahocorasick_rs` (backend Rust, DFA) **170 MB/s, nhanh 24×**, số match y hệt.
    Bench: 122MB/400k dòng — pyahocorasick 17,2s vs Rust 0,72s.
    Toàn kho 69GB: 2,8 GIỜ -> ~9 PHÚT (lúc này xz mới là nút thắt).
  GPU/CUDA KHÔNG giúp ở đây: việc này là đi bộ trên máy trạng thái (truy cập bộ nhớ
  bất quy tắc, rẽ nhánh theo dữ liệu) chứ không thiếu FLOP — đúng thứ GPU làm tệ nhất.

  xz -dc D:/Bit-Translate-data/raw/ja.txt.xz | python scripts/mine_rare_terms.py
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import ahocorasick_rs as ar

D = Path("D:/Bit-Translate-data")


def dig(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", default=str(D / "domain_terms.json"))
    ap.add_argument("--cov", default="eval/term_coverage.json")
    ap.add_argument("--out", default=str(D / "raw" / "mined_rare.txt"))
    ap.add_argument("--stats", default="eval/mine_stats.json")
    ap.add_argument("--min-occ", type=int, default=60)
    ap.add_argument("--margin", type=float, default=1.3, help="lấy dư để bù QC loại")
    ap.add_argument("--cap-per-term", type=int, default=150)
    ap.add_argument("--max-out", type=int, default=700000)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--chunk", type=int, default=40_000_000, help="lô để kiểm đồng đều")
    a = ap.parse_args()

    cov = json.loads(Path(a.cov).read_text(encoding="utf-8"))
    cur, pool = cov["counts_cur"], cov["counts_pool"]
    doms = json.loads(Path(a.terms).read_text(encoding="utf-8"))
    term2dom = {}
    for d, items in doms.items():
        for o in items:
            term2dom.setdefault(o["t"], d)

    # chỉ nhắm từ đang dưới ngưỡng sau khi đã tính corpus + kho đã tải
    want = {}
    for t in cur:
        have = cur[t] + pool[t]
        if have < a.min_occ:
            want[t] = min(a.cap_per_term, int((a.min_occ - have) * a.margin))
    print(f"nhắm {len(want):,} thuật ngữ thiếu | tổng hạn mức "
          f"{sum(want.values()):,} câu", file=sys.stderr, flush=True)

    wl = list(want)
    A = ar.AhoCorasick(wl, implementation=ar.Implementation.DFA)

    # loại câu đã có trong corpus (cc100_pick vốn cắt ra từ chính kho này)
    print("dựng tập câu đã dùng...", file=sys.stderr, flush=True)
    used = set()
    with open(D / "kd_v5_merged.jsonl", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                used.add(dig(json.loads(line)["ja"]))
            except Exception:
                pass
    print(f"  {len(used):,}", file=sys.stderr, flush=True)

    got = Counter()          # đã chọn bao nhiêu câu cho mỗi từ
    total_occ = Counter()    # đếm đầy đủ trên toàn kho
    chunk_occ = []           # đếm theo lô -> kiểm đồng đều
    cur_chunk = Counter()
    seen_out = set()
    n = kept = 0

    fo = open(a.out, "w", encoding="utf-8")
    for line in sys.stdin:
        s = line.strip()
        n += 1
        if n % a.chunk == 0:
            chunk_occ.append(sum(cur_chunk.values()))
            cur_chunk = Counter()
            print(f"  {n:,} câu | chọn {kept:,}", file=sys.stderr, flush=True)

        L = len(s)
        if L < 10:
            continue
        hits = {wl[i] for i, _, _ in A.find_matches_as_indexes(s, overlapping=True)}
        if not hits:
            continue
        for t in hits:
            total_occ[t] += 1
            cur_chunk[t] += 1
        if kept >= a.max_out or not (a.min_len <= L <= a.max_len):
            continue
        # chỉ giữ nếu còn giúp được ít nhất một từ chưa đủ hạn mức
        need = [t for t in hits if got[t] < want[t]]
        if not need:
            continue
        h = dig(s)
        if h in used or h in seen_out:
            continue
        seen_out.add(h)
        for t in need:
            got[t] += 1
        fo.write(s + "\n")
        kept += 1
    fo.close()

    if sum(cur_chunk.values()):
        chunk_occ.append(sum(cur_chunk.values()))

    filled = sum(1 for t in want if cur[t] + pool[t] + total_occ[t] >= a.min_occ)
    zero = sum(1 for t in want if total_occ[t] == 0)
    print(f"\n=== ĐÀO XONG ===", file=sys.stderr)
    print(f"quét      : {n:,} câu", file=sys.stderr)
    print(f"chọn      : {kept:,} câu -> {a.out}", file=sys.stderr)
    print(f"lấp được  : {filled:,}/{len(want):,} từ thiếu ({100*filled/len(want):.0f}%)",
          file=sys.stderr)
    print(f"vẫn 0 lần : {zero:,} từ -> bắt buộc SINH", file=sys.stderr)
    if len(chunk_occ) > 2:
        m = sum(chunk_occ) / len(chunk_occ)
        sd = (sum((x - m) ** 2 for x in chunk_occ) / len(chunk_occ)) ** 0.5
        print(f"đồng đều  : {len(chunk_occ)} lô, lệch chuẩn/TB = {sd/m:.1%} "
              f"({'ĐỀU, ngoại suy đầu file tin được' if sd/m < 0.25 else 'LỆCH, mẫu đầu file KHÔNG đại diện'})",
              file=sys.stderr)

    # độ phủ mới theo miền
    per = {}
    for name, items in doms.items():
        ts = [o["t"] for o in items]
        before = sum(1 for t in ts if cur[t] + pool[t] >= a.min_occ)
        after = sum(1 for t in ts if cur[t] + pool[t] + total_occ[t] >= a.min_occ)
        per[name] = {"before": round(100 * before / len(ts)),
                     "after": round(100 * after / len(ts)),
                     "still": sum(1 for t in ts if cur[t] + pool[t] + total_occ[t] < a.min_occ)}
    Path(a.stats).write_text(json.dumps(
        {"scanned": n, "kept": kept, "chunk_occ": chunk_occ, "per_domain": per,
         "counts_archive_full": {t: total_occ[t] for t in want}},
        ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'miền':<26}{'+kho tải':>10}{'+ĐÀO HẾT':>10}{'còn thiếu':>11}", file=sys.stderr)
    for k, v in sorted(per.items(), key=lambda x: x[1]["after"]):
        print(f"{k:<26}{v['before']:>9}%{v['after']:>9}%{v['still']:>11}", file=sys.stderr)
    print(f"\n-> {a.stats}", file=sys.stderr)


if __name__ == "__main__":
    main()
