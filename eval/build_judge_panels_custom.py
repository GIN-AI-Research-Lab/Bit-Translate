#!/usr/bin/env python3
"""Judge panel 4-way TỔNG QUÁT (không hardcode google/fable/haiku) — dùng để so
ứng viên thầy KD (PLAN_KD_JA2VI): vd gemini-lite + qwen-plus vs google + haiku,
không tốn 1 slot cho Fable (không phải ứng viên trong quyết định này).

  python eval/build_judge_panels_custom.py <outdir> LABEL1=file1.jsonl LABEL2=file2.jsonl \
         LABEL3=file3.jsonl LABEL4=file4.jsonl

Ghi <outdir>/panel_{0..4}.jsonl (40 câu/panel) + panel_key.json — cùng format
build_judge_panels_4way.py nên aggregate_judge_4way.py dùng thẳng được.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OUTDIR = Path(sys.argv[1])
pairs = [a.split("=", 1) for a in sys.argv[2:]]
assert len(pairs) == 4, f"cần đúng 4 hệ (A-D), nhận {len(pairs)}"
OUTDIR.mkdir(parents=True, exist_ok=True)


def load(p):
    return {r["id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8"))}


base = load(ROOT / "hardbench200.jsonl")
systems = {label: load(ROOT / f) if not Path(f).is_absolute() and not Path(f).exists()
                  else load(f) for label, f in pairs}
# fallback đơn giản: thử path tương đối tới ROOT trước, path đã cho sau
systems = {}
for label, f in pairs:
    p = Path(f)
    systems[label] = load(p if p.exists() else ROOT / f)

rng = random.Random(20260717)
rows, key = [], {}
for i in sorted(base):
    b = base[i]
    order = list(systems)
    rng.shuffle(order)
    letters = ["A", "B", "C", "D"]
    row = {"id": i, "domain": b["domain"], "dir": b["dir"], "src": b["src"], "ref": b["ref"]}
    key[i] = {}
    for L, s in zip(letters, order):
        row[L] = systems[s][i]["hyp"]
        key[i][L] = s
    rows.append(row)

for p in range(5):
    chunk = rows[p * 40:(p + 1) * 40]
    out = OUTDIR / f"panel_{p}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chunk), encoding="utf-8")
    print(out, len(chunk))
(OUTDIR / "panel_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
print("key:", OUTDIR / "panel_key.json")
print("systems:", list(systems))
