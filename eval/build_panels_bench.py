#!/usr/bin/env python3
"""Dựng panel MÙ trên bench câu THẬT (bench_new.jsonl) — số hệ tuỳ ý, không có ref.

Khác build_judge_panels_custom.py: bộ đó chạy trên hardbench200 (có bản dịch tham
chiếu và đúng 4 hệ). Bench câu thật KHÔNG có ref — judge chấm thẳng từ câu nguồn
tiếng Nhật, nên panel chỉ gồm src + các bản dịch đã xáo thứ tự.

VÌ SAO CẦN CHẤM LẠI CẢ 200 CÂU: mốc cũ chỉ chấm 50 câu (v3 66% đúng nghĩa). Với
n=50, sai số 95% là ±13 điểm — không phân biệt được cải thiện +8 với nhiễu. n=200
kéo sai số xuống ±7. Ngoài ra hai hệ phải được chấm TRONG CÙNG MỘT PHIÊN: thang
điểm của judge không hiệu chuẩn được giữa các phiên khác nhau.

  python eval/build_panels_bench.py eval/judge_v4 v3=eval/bench_v3.jsonl \
      v4=eval/bench_v4.jsonl google=eval/bench_google.jsonl
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OUTDIR = Path(sys.argv[1])
pairs = [a.split("=", 1) for a in sys.argv[2:]]
assert 2 <= len(pairs) <= 6, f"cần 2-6 hệ, nhận {len(pairs)}"
OUTDIR.mkdir(parents=True, exist_ok=True)
PER_PANEL = 40


def load(p):
    p = Path(p)
    p = p if p.exists() else ROOT / p
    return {str(r["id"]): r for r in (json.loads(l) for l in open(p, encoding="utf-8"))}


base = load(ROOT / "bench_new.jsonl")
systems = {label: load(f) for label, f in pairs}

# Chỉ giữ câu mà MỌI hệ đều có bản dịch — nếu không thì so sánh lệch mẫu.
ids = [i for i in base if all(i in s for s in systems.values())]
ids.sort(key=int)
thieu = len(base) - len(ids)
if thieu:
    for lb, s in systems.items():
        m = [i for i in base if i not in s]
        if m:
            print(f"  CẢNH BÁO: {lb} thiếu {len(m)} câu (vd id {m[:5]})")

rng = random.Random(20260726)
letters = "ABCDEF"
rows, key = [], {}
for i in ids:
    b = base[i]
    order = list(systems)
    rng.shuffle(order)
    row = {"id": i, "len": b["len"], "feat": b["feat"], "src": b["src"]}
    key[i] = {}
    for L, s in zip(letters, order):
        row[L] = systems[s][i]["hyp"]
        key[i][L] = s
    rows.append(row)

npanel = (len(rows) + PER_PANEL - 1) // PER_PANEL
for p in range(npanel):
    chunk = rows[p * PER_PANEL:(p + 1) * PER_PANEL]
    out = OUTDIR / f"panel_{p}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chunk),
                   encoding="utf-8")
    print(f"  {out} ({len(chunk)} câu)")
(OUTDIR / "panel_key.json").write_text(
    json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n{len(rows)}/{len(base)} câu x {len(systems)} hệ: {', '.join(systems)}")
print(f"key: {OUTDIR / 'panel_key.json'}")
