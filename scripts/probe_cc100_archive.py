#!/usr/bin/env python3
"""Kho CC-100 GỐC (ja.txt.xz, 69,3GB ~ 417 TRIỆU câu) có lấp được thuật ngữ đang thiếu không?

Vì sao phải hỏi: `cc100_pick.txt` (5,2M) + `cc100_colloq.txt` (1,7M) mới chỉ là ~1,7%
của kho gốc. Kết luận "phải đi thu thập nguồn mới" ở bản trước được rút ra khi chỉ nhìn
phần đã lọc — chưa hề kiểm phần còn lại. Nếu 98% chưa đụng tới đã đủ lấp thì không cần
harvest gì mới, chỉ cần LỌC SÂU HƠN cái đang có.

Cách đo: giải nén tuần tự N câu đầu, đếm chính xác (Aho-Corasick) từng thuật ngữ, rồi
ngoại suy lên 417M theo tỷ lệ. Một từ cần >=60 lần trên toàn kho thì phải xuất hiện
>= 60*N/417M lần trong mẫu.

  python scripts/probe_cc100_archive.py --lines 20000000
"""
import argparse
import json
import lzma
import sys
from collections import Counter
from pathlib import Path

import ahocorasick

D = Path("D:/Bit-Translate-data")
TOTAL_EST = 417_000_000  # 69,3GiB / 400MB * 2,236,973 dòng


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(D / "raw" / "ja.txt.xz"))
    p.add_argument("--terms", default=str(D / "domain_terms.json"))
    p.add_argument("--cov", default="eval/term_coverage.json")
    p.add_argument("--lines", type=int, default=20_000_000)
    p.add_argument("--out", default="eval/cc100_archive_probe.json")
    a = p.parse_args()

    doms = json.loads(Path(a.terms).read_text(encoding="utf-8"))
    cov = json.loads(Path(a.cov).read_text(encoding="utf-8"))
    cur, pool, MIN = cov["counts_cur"], cov["counts_pool"], cov["min_occ"]
    all_terms = sorted({o["t"] for v in doms.values() for o in v})

    A = ahocorasick.Automaton()
    for t in all_terms:
        A.add_word(t, t)
    A.make_automaton()
    print(f"{len(all_terms):,} thuật ngữ | quét {a.lines:,} câu đầu của kho gốc", flush=True)

    arc = Counter()
    n = 0
    with lzma.open(a.src, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if len(s) < 10:
                continue
            n += 1
            for _, t in A.iter(s):
                arc[t] += 1
            if n % 2_000_000 == 0:
                print(f"  {n:,}", file=sys.stderr, flush=True)
            if n >= a.lines:
                break

    scale = TOTAL_EST / n
    print(f"\nquét {n:,} câu (= {100*n/TOTAL_EST:.1f}% kho) | hệ số ngoại suy x{scale:.0f}\n")

    # thuật ngữ còn thiếu SAU KHI đã tính corpus + kho đã tải
    missing = [t for t in all_terms if cur[t] + pool[t] < MIN]
    fixed = [t for t in missing if cur[t] + pool[t] + arc[t] * scale >= MIN]
    still = [t for t in missing if cur[t] + pool[t] + arc[t] * scale < MIN]
    zero = [t for t in still if arc[t] == 0]

    print(f"=== {len(missing):,} thuật ngữ đang thiếu (sau corpus + kho đã tải) ===")
    print(f"  {len(fixed):,} ({100*len(fixed)/len(missing):.0f}%) — kho GỐC lấp được nếu đào hết")
    print(f"  {len(still):,} ({100*len(still)/len(missing):.0f}%) — kho gốc VẪN không đủ")
    print(f"     trong đó {len(zero):,} từ không xuất hiện lần nào trong mẫu\n")

    # cần đào bao nhiêu câu? với mỗi từ: số câu kho gốc cần quét để đạt đủ 60 lần
    need_lines = []
    for t in missing:
        deficit = MIN - (cur[t] + pool[t])
        if arc[t] > 0:
            need_lines.append(deficit / (arc[t] / n))
    need_lines.sort()
    if need_lines:
        for q, lab in [(0.5, "50%"), (0.8, "80%"), (0.9, "90%"), (0.95, "95%")]:
            v = need_lines[min(int(q * len(need_lines)), len(need_lines) - 1)]
            print(f"  để lấp {lab} số từ thiếu: cần đào ~{v/1e6:,.0f}M câu kho gốc"
                  f"  ({100*min(v,TOTAL_EST)/TOTAL_EST:.0f}% kho)")

    # theo miền
    per = {}
    for name, items in doms.items():
        ts = [o["t"] for o in items]
        now = sum(1 for t in ts if cur[t] >= MIN)
        aft = sum(1 for t in ts if cur[t] + pool[t] >= MIN)
        arch = sum(1 for t in ts if cur[t] + pool[t] + arc[t] * scale >= MIN)
        per[name] = {"now": round(100 * now / len(ts)), "pool": round(100 * aft / len(ts)),
                     "archive": round(100 * arch / len(ts)),
                     "still": sum(1 for t in ts if cur[t] + pool[t] + arc[t] * scale < MIN)}

    print(f"\n{'miền':<26}{'giờ':>6}{'+kho tải':>10}{'+kho GỐC':>10}{'còn thiếu':>11}")
    print("-" * 65)
    for k, v in sorted(per.items(), key=lambda x: x[1]["archive"]):
        print(f"{k:<26}{v['now']:>5}%{v['pool']:>9}%{v['archive']:>9}%{v['still']:>11}")

    Path(a.out).write_text(json.dumps(
        {"scanned": n, "scale": scale, "per_domain": per,
         "counts_archive": {t: arc[t] for t in all_terms}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
