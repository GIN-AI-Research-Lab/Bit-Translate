#!/usr/bin/env python3
"""Panel 200b: v8avg-i2s vs v7a-i2s (ban dich da luu trong repo), A/B mu."""
import json, random, os

S = os.path.dirname(os.path.abspath(__file__))


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


v8 = {o["id"]: o for o in jl(os.path.join(S, "bench_200b_v8avgi2s.jsonl"))}
v7 = {o["id"]: o for o in jl(os.path.join(S, "refs", "bench_200b_v7ai2s.jsonl"))}
assert set(v8) == set(v7), "id lech giua 2 he"

random.seed(20260810 + 200)
items, key = [], []
same = 0
for i in sorted(v8, key=int):
    a, b = v8[i], v7[i]
    assert a["src"] == b["src"]
    if a["hyp"] == b["hyp"]:
        same += 1
    heads = random.random() < 0.5
    items.append({"id": int(i), "ja": a["src"],
                  "A": a["hyp"] if heads else b["hyp"],
                  "B": b["hyp"] if heads else a["hyp"]})
    key.append({"id": int(i), "ja": a["src"],
                "A_sys": "v8avg" if heads else "v7a",
                "B_sys": "v7a" if heads else "v8avg",
                "identical": a["hyp"] == b["hyp"],
                "len": a.get("len", ""), "diff": a.get("diff", ""),
                "feat": a.get("feat", "")})
json.dump(key, open(os.path.join(S, "b200_key.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
from collections import Counter
print(f"200 cau ({same} giong het) | A_sys={dict(Counter(k['A_sys'] for k in key))}")
sz = 50
for c in range(4):
    part = items[c * sz:(c + 1) * sz]
    json.dump(part, open(os.path.join(S, f"b200_chunk_c{c+1}.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  chunk c{c+1}: {len(part)}")
