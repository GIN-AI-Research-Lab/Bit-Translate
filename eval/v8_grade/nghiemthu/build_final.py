#!/usr/bin/env python3
"""Vong chot: v8_avg vs v7a truc tiep tren rand150 (no-regression cuoi)."""
import json, random, os

S = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(S, "refs")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


avg = {o["ja"]: o["vi"] for o in jl(os.path.join(S, "v8_step16750_v8avg.jsonl"))}
v7a = {o["ja"]: o["vi"] for o in jl(os.path.join(REF, "v7a_1200.jsonl"))}
gg = {o["ja"]: o["gg"] for o in jl(os.path.join(REF, "google_1200.jsonl"))}

random.seed(20260809 + 777)
old = json.load(open(os.path.join(REF, "grade_rand150_key.json"), encoding="utf-8"))
items, key = [], []
same = 0
for e in old:
    ja = e["ja"]
    if avg[ja] == v7a[ja]:
        same += 1
    heads = random.random() < 0.5  # heads -> A=v8_avg
    items.append({"id": e["id"], "ja": ja, "google_ref": gg.get(ja, ""),
                  "A": avg[ja] if heads else v7a[ja],
                  "B": v7a[ja] if heads else avg[ja]})
    key.append({"id": e["id"], "ja": ja,
                "A_sys": "avg" if heads else "v7a",
                "B_sys": "v7a" if heads else "avg",
                "identical": avg[ja] == v7a[ja]})
json.dump(key, open(os.path.join(S, "final_key_rand150.json"), "w",
                    encoding="utf-8"), ensure_ascii=False, indent=1)
from collections import Counter
print(f"rand150: {len(items)} cau ({same} giong het) | "
      f"A_sys={dict(Counter(k['A_sys'] for k in key))}")
n = len(items); sz = -(-n // 3)
for i in range(3):
    part = items[i * sz:(i + 1) * sz]
    json.dump(part, open(os.path.join(S, f"final_chunk_c{i+1}.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  chunk c{i+1}: {len(part)} cau")
