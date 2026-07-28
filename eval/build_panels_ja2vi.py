#!/usr/bin/env python3
"""Dung panel mu CHI ja2vi (100 cau) cho 4 he: kd / google / haiku / fable.
Xao A-D deterministic theo id. Ghi panels/panel_{0..4}.jsonl (20 cau/panel) + key.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent
KD = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "hardbench_kd100m_step11000.jsonl"
# arg 2: thư mục ra. ĐỪNG ghi đè thư mục cũ — panel_key.json bị thay sẽ làm judge
# cũ không đối chiếu được nhãn hệ nữa.
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "judge_kd_ja2vi"
(OUT / "panels").mkdir(parents=True, exist_ok=True)


def load(p):
    return {r["id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8"))
            if r.get("dir") == "ja2vi"}


base = {r["id"]: r for r in (json.loads(l) for l in open(ROOT / "hardbench200.jsonl", encoding="utf-8"))
        if r["dir"] == "ja2vi"}
systems = {
    "kd": load(KD),
    "google": load(ROOT / "hardbench_google.jsonl"),
    "haiku": load(ROOT / "hardbench_haiku.jsonl"),
    "fable": load(ROOT / "hardbench_claude.jsonl"),
}

rng = random.Random(20260725)
rows, key = [], {}
for i in sorted(base):
    b = base[i]
    order = list(systems)
    rng.shuffle(order)
    row = {"id": i, "domain": b["domain"], "dir": b["dir"], "src": b["src"], "ref": b["ref"]}
    key[i] = {}
    for L, s in zip("ABCD", order):
        row[L] = systems[s][i]["hyp"]
        key[i][L] = s
    rows.append(row)

for p in range(5):
    chunk = rows[p * 20:(p + 1) * 20]
    (OUT / "panels" / f"panel_{p}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chunk), encoding="utf-8")
(OUT / "panels" / "panel_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Da dung 5 panel x 20 cau ja2vi -> {OUT/'panels'}")
print("4 he: kd / google / haiku / fable (xao mu)")
