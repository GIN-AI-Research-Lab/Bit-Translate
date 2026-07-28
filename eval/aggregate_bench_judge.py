#!/usr/bin/env python3
"""Gộp điểm judge mù trên bench câu thật -> bảng so sánh + khoảng tin cậy.

Judge chấm từng panel và ghi ra <outdir>/judges/panel_N.json với dạng:
  {"<id>": {"A": 2, "B": 1, "C": 2}, ...}     # thang ĐÚNG NGHĨA 0/1/2
Script này giải mã nhãn A/B/C về tên hệ bằng panel_key.json rồi tổng hợp.

Thang điểm (chốt sau khi user thấy acc>=4 quá khắt khe):
  2 = ĐÚNG NGHĨA — người đọc hiểu đúng ý câu gốc, dù chữ nghĩa chưa mượt
  1 = đúng một phần — mất hoặc lệch một ý, còn dùng tạm được
  0 = sai nghĩa — hiểu sai, bịa, hoặc rơi nội dung chính

In kèm khoảng tin cậy 95% (Wilson) vì đây là điều quyết định KẾT LUẬN ĐƯỢC hay
không: n=50 cho sai số ±13 điểm, n=200 cho ±7. Hai hệ chỉ khác nhau thật khi
khoảng tin cậy của chênh lệch không chứa 0 (dùng McNemar cho dữ liệu bắt cặp).

  python eval/aggregate_bench_judge.py eval/judge_v4
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
D = Path(sys.argv[1])
key = json.load(open(D / "panel_key.json", encoding="utf-8"))
bench = {str(json.loads(l)["id"]): json.loads(l)
         for l in open(ROOT / "bench_new.jsonl", encoding="utf-8")}

# nạp mọi file điểm trong <outdir>/judges/
raw = {}
jd = D / "judges"
files = sorted(jd.glob("*.json")) if jd.exists() else []
if not files:
    sys.exit(f"chưa có file điểm nào trong {jd}\n"
             f"  judge chấm panel_N.jsonl rồi ghi {jd}/panel_N.json")
for f in files:
    for i, row in json.load(open(f, encoding="utf-8")).items():
        raw.setdefault(str(i), {}).update(row)

# giải mã A/B/C -> tên hệ
sc = defaultdict(dict)          # sc[id][system] = điểm
for i, row in raw.items():
    for L, v in row.items():
        s = key.get(i, {}).get(L)
        if s:
            sc[i][s] = v
syss = sorted({s for r in sc.values() for s in r})
print(f"đã chấm {len(sc)}/{len(key)} câu | hệ: {', '.join(syss)}\n")


def wilson(k, n):
    if not n:
        return 0.0, 0.0
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def table(ids, title):
    if not ids:
        return
    print(f"--- {title} (n={len(ids)}) ---")
    print(f"{'hệ':10} {'đúng nghĩa':>12} {'KTC 95%':>16} {'điểm TB /2':>12}")
    for s in syss:
        v = [sc[i][s] for i in ids if s in sc[i]]
        if not v:
            continue
        k = sum(x == 2 for x in v)
        lo, hi = wilson(k, len(v))
        print(f"{s:10} {100*k/len(v):11.0f}% {f'[{lo:.0f}-{hi:.0f}]':>16} "
              f"{sum(v)/len(v):12.2f}")
    print()


ids = sorted(sc, key=int)
table(ids, "TỔNG")
for L in ["short", "long"]:
    table([i for i in ids if bench[i]["len"] == L], L)

# So từng cặp bằng McNemar — dữ liệu BẮT CẶP (cùng câu, khác hệ) nên mạnh hơn
# nhiều so với so hai tỷ lệ độc lập.
if len(syss) >= 2:
    print("--- so từng cặp (McNemar, chỉ tính câu hai hệ khác kết quả) ---")
    for a in range(len(syss)):
        for b in range(a + 1, len(syss)):
            X, Y = syss[a], syss[b]
            n01 = n10 = 0
            for i in ids:
                if X not in sc[i] or Y not in sc[i]:
                    continue
                x, y = sc[i][X] == 2, sc[i][Y] == 2
                n01 += (not x) and y
                n10 += x and (not y)
            n = n01 + n10
            if n == 0:
                print(f"  {X} vs {Y}: giống hệt nhau")
                continue
            chi = (abs(n10 - n01) - 1) ** 2 / n
            p = math.erfc(math.sqrt(chi / 2))
            kl = "KHÁC BIỆT THẬT" if p < 0.05 else "chưa đủ bằng chứng"
            print(f"  {X} vs {Y}: {X} hơn ở {n10} câu, {Y} hơn ở {n01} câu "
                  f"-> p={p:.3f} ({kl})")
    print()

# lỗi còn lại theo đặc trưng, cho hệ đứng cuối bảng chữ cái v* (model của mình)
mine = [s for s in syss if s.startswith("v")] or syss
m = mine[-1]
bad = defaultdict(int)
for i in ids:
    if sc[i].get(m, 2) < 2:
        bad[bench[i]["feat"]] += 1
if bad:
    print(f"--- {m} còn sai ở đặc trưng nào ---")
    for k, v in sorted(bad.items(), key=lambda x: -x[1]):
        print(f"  {k:12} {v}")

out = D / "RESULT.json"
json.dump({"scores": sc, "n": len(sc)}, open(out, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"\n-> {out}")
