#!/usr/bin/env python3
"""Tách test-set glossary: giữ 10% thuật ngữ làm test (KHÔNG train) để đo
'tỉ lệ dịch đúng thuật ngữ' ở Vòng 1. Split cố định theo seed.

Đọc:  data/glossary/glossary_merged.csv  (ja,vi,en,source)
Ghi:  data/glossary/glossary_train.csv   (90% — dùng để train)
      data/glossary/glossary_test.csv    (10% — HOLD-OUT, chấm điểm)
      data/glossary/glossary_test_terms.txt  (danh sách ja term test — dùng để LỌC
        bỏ câu Gemini của các term này khỏi data train lúc binarize, tránh rò rỉ)
"""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
G = ROOT / "data" / "glossary"
FRAC = 0.10
R = random.Random(42)

rows = list(csv.reader((G / "glossary_merged.csv").open(encoding="utf-8-sig")))
hdr, rows = rows[0], rows[1:]

# gom theo ja term (một term có thể nhiều dòng vi) -> split theo TERM để test-set
# và train-set không dùng chung term nào
terms = sorted({r[0] for r in rows if r and r[0]})
R.shuffle(terms)
n_test = max(1, int(len(terms) * FRAC))
test_terms = set(terms[:n_test])

train = [r for r in rows if r[0] not in test_terms]
test = [r for r in rows if r[0] in test_terms]


def dump(path, data):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(data)


dump(G / "glossary_train.csv", train)
dump(G / "glossary_test.csv", test)
(G / "glossary_test_terms.txt").write_text("\n".join(sorted(test_terms)), encoding="utf-8")

print(f"tong term: {len(terms)} | test (hold-out): {len(test_terms)} term / {len(test)} dong")
print(f"train: {len(train)} dong")
print("-> glossary_train.csv, glossary_test.csv, glossary_test_terms.txt")
print("LUU Y: luc binarize, loc bo cau Gemini co term thuoc glossary_test_terms.txt.")
