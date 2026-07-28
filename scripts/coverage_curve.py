#!/usr/bin/env python3
"""Đường cong phủ: cần BAO NHIÊU câu trong một miền thì thuật ngữ của miền đó đủ dày?

Lý do phải đo: từ vòng 4 đã biết ngưỡng "đủ dày" — thuật ngữ xuất hiện ~78 lần/corpus
thì được vá, <=5 lần thì model bịa. Nhưng chưa ai biết một miền có bao nhiêu thuật ngữ
riêng, nên không quy ra được số CÂU cần thu thập. Script này nối hai đầu đó lại:

  đếm thuật ngữ riêng của miền  ->  lấy N câu ngẫu nhiên  ->  bao nhiêu % thuật ngữ
  đạt >= K lần?  ->  vẽ đường cong N vs độ phủ  ->  đọc ra N cần thiết.

"Thuật ngữ" = cụm kanji 2-4 ký tự (đơn vị mang nghĩa chính của tiếng Nhật viết).
Loại cụm chỉ số đếm (八条一号...) vì vòng 4 đã bị nó làm sai thống kê một lần.

  python scripts/coverage_curve.py --src D:/Bit-Translate-data/raw/kokkai_ja.txt --label kokkai
"""
import argparse
import random
import re
from collections import Counter
from pathlib import Path

TERM = re.compile(r"[一-龥]{2,4}")
# số đếm viết bằng kanji -> không phải thuật ngữ
NUMISH = re.compile(r"^[〇一二三四五六七八九十百千万億兆第条号項年月日時分回個人名]+$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--label", default="src")
    ap.add_argument("--pool", type=int, default=1500000, help="số câu nạp làm tổng thể")
    ap.add_argument("--min-occ", type=int, default=60, help="ngưỡng 'đủ dày' mỗi thuật ngữ")
    ap.add_argument("--head", type=int, default=8000, help="xét N thuật ngữ phổ biến nhất")
    a = ap.parse_args()

    rnd = random.Random(11)
    sents = []
    with open(a.src, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if len(s) >= 15:
                sents.append(s)
                if len(sents) >= a.pool:
                    break
    rnd.shuffle(sents)
    print(f"[{a.label}] nạp {len(sents):,} câu")

    # thuật ngữ của miền = N cụm phổ biến nhất trong toàn bộ pool
    gl = Counter()
    for s in sents:
        for t in TERM.findall(s):
            if not NUMISH.match(t):
                gl[t] += 1
    vocab = [t for t, _ in gl.most_common(a.head)]
    vset = set(vocab)
    print(f"[{a.label}] {len(gl):,} cụm riêng biệt; xét top {len(vocab):,}")
    print(f"[{a.label}] tần suất trong pool: "
          f"top1={gl[vocab[0]]:,}  top{len(vocab)}={gl[vocab[-1]]:,}")

    print(f"\nĐỘ PHỦ = % trong top {a.head} thuật ngữ đạt >= {a.min_occ} lần\n")
    print(f"{'số câu':>10}  {'độ phủ':>7}  {'ngưỡng 20 lần':>14}  {'trung vị lần':>13}")
    for n in [25000, 50000, 100000, 200000, 400000, 800000, 1200000, 1500000]:
        if n > len(sents):
            break
        c = Counter()
        for s in sents[:n]:
            for t in TERM.findall(s):
                if t in vset:
                    c[t] += 1
        ok = sum(1 for t in vocab if c[t] >= a.min_occ)
        ok20 = sum(1 for t in vocab if c[t] >= 20)
        med = sorted(c[t] for t in vocab)[len(vocab) // 2]
        print(f"{n:>10,}  {100*ok/len(vocab):6.1f}%  {100*ok20/len(vocab):13.1f}%  {med:>13,}")


if __name__ == "__main__":
    main()
