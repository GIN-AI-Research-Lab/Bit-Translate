#!/usr/bin/env python3
"""Đo ĐỘ PHỦ thuật ngữ trong corpus train — biết chính xác đang thiếu gì.

Vì sao cần: ba vòng qua đều sinh data rồi mới biết phủ được gì. Kết quả: vòng 3
sinh 241k cặp -> hardbench +18 điểm nhưng bench câu THẬT 0 điểm (v3 66% = v2 66%).
Sinh ngẫu nhiên chỉ củng cố cái model đã biết.

Cách đúng: LẬP DANH SÁCH -> ĐO THIẾU -> SINH ĐÚNG CHỖ THIẾU -> ĐO LẠI.
Bằng chứng nguyên lý này đúng: vòng 2 bơm 30.778 cặp / 1.241 quán ngữ -> +0,8;
vòng 3 bơm 95.782 cặp / 4.691 quán ngữ -> +1,7. Cùng loại data, khác độ phủ.

Thuật toán: sliding-window + hash set (không dùng str.count 18k lần — quá chậm).

  python scripts/measure_term_coverage.py --sample 2000000
  # -> D:/Bit-Translate-data/term_gap.json  (danh sách thuật ngữ thiếu)
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = Path("D:/Bit-Translate-data/term_gap.json")


def load_terms():
    terms = {}
    p = ROOT / "data" / "glossary" / "glossary_merged.csv"
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ja = (r.get("ja") or "").strip()
            vi = (r.get("vi") or "").strip()
            if 2 <= len(ja) <= 12 and vi and not ja.isascii():
                terms[ja] = vi
    return terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="D:/Bit-Translate-data/clean_v3/train.ja")
    ap.add_argument("--sample", type=int, default=2_000_000)
    ap.add_argument("--gap-threshold", type=int, default=5,
                    help="thuật ngữ xuất hiện <= ngưỡng này coi là THIẾU")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    terms = load_terms()
    by_len = {}
    for t in terms:
        by_len.setdefault(len(t), set()).add(t)
    lens = sorted(by_len)
    print(f"glossary: {len(terms):,} thuật ngữ | độ dài {lens[0]}-{lens[-1]}", flush=True)

    cnt = Counter()
    n = 0
    t0 = time.time()
    with open(a.corpus, encoding="utf-8") as f:
        for line in f:
            n += 1
            if n > a.sample:
                break
            s = line.rstrip("\n")
            L = len(s)
            for k in lens:                     # trượt cửa sổ theo từng độ dài
                if k > L:
                    break
                bag = by_len[k]
                for i in range(L - k + 1):
                    w = s[i:i + k]
                    if w in bag:
                        cnt[w] += 1
            if n % 200_000 == 0:
                print(f"  ...{n:,} câu | {time.time()-t0:.0f}s", flush=True)

    print(f"\nquét {min(n, a.sample):,} câu trong {time.time()-t0:.0f}s")
    print(f"{'ngưỡng':>10} {'số thuật ngữ':>14} {'%':>7}")
    for th in [0, 1, 5, 20, 100, 500]:
        c = sum(1 for t in terms if cnt.get(t, 0) <= th)
        print(f"{'<= '+str(th):>10} {c:14,} {100*c/len(terms):6.1f}%")

    gap = {t: {"vi": terms[t], "freq": cnt.get(t, 0)}
           for t in terms if cnt.get(t, 0) <= a.gap_threshold}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(gap, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {len(gap):,} thuật ngữ THIẾU (<= {a.gap_threshold} lần) lưu ở {a.out}")
    print("--- mẫu thuật ngữ thiếu ---")
    for t in list(gap)[:15]:
        print(f"   {t:12} = {gap[t]['vi'][:42]:44} ({gap[t]['freq']} lần)")


if __name__ == "__main__":
    main()
