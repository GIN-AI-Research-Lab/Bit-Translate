#!/usr/bin/env python3
"""Dung de cham mu 4 he (110M step23000 / Google / Claude-Fable / Claude-Haiku)
tren hardbench200. Xao thu tu A-D deterministic theo id (seed co dinh).

  python eval/build_judge_panels_4way.py <haiku_hyp.jsonl> <outdir> [main.jsonl] [main_label]

main.jsonl/main_label (tùy chọn): thay hệ "110M" mặc định bằng model khác —
vd gate 292M (PLAN_RANKUP §1): ... eval/hardbench_292m_step18400.jsonl 292M

Ghi: <outdir>/panel_{0..4}.jsonl (40 cau/panel, fields: id domain dir src ref A B C D)
     <outdir>/panel_key.json  (id -> {A: system, ...})  -- KHONG dua cho trong tai
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent
HAIKU, OUTDIR = Path(sys.argv[1]), Path(sys.argv[2])
MAIN_FILE = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "hardbench_step23000.jsonl"
MAIN_LABEL = sys.argv[4] if len(sys.argv) > 4 else "110M"
OUTDIR.mkdir(parents=True, exist_ok=True)

def load(p):
    return {r["id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8"))}

base = load(ROOT / "hardbench200.jsonl")
systems = {
    MAIN_LABEL: load(MAIN_FILE),
    "google": load(ROOT / "hardbench_google.jsonl"),
    "fable": load(ROOT / "hardbench_claude.jsonl"),
    "haiku": load(HAIKU),
}

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
