#!/usr/bin/env python3
"""Dung bo cham mu TRUC TIEP: soup_a03 vs v8 (cung phien) tren dung 2 tap
err64 + rand150. A/B randomize, key rieng."""
import json, random, os

S = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(S, "refs")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


soup = {o["ja"]: o["vi"] for o in jl(os.path.join(S, "v8_step16750_soupA03.jsonl"))}
v8 = {o["ja"]: o["vi"] for o in jl(os.path.join(REF, "v8_final.jsonl"))}
gg = {o["ja"]: o["gg"] for o in jl(os.path.join(REF, "google_1200.jsonl"))}

random.seed(20260809 + 1)

for split, keyfile in [("err64", "grade_err64_key.json"),
                       ("rand150", "grade_rand150_key.json")]:
    old = json.load(open(os.path.join(REF, keyfile), encoding="utf-8"))
    items, key = [], []
    same = 0
    for e in old:
        ja = e["ja"]
        if soup[ja] == v8[ja]:
            same += 1  # hai ban giong het -> khoi cham, tinh tie san
        heads = random.random() < 0.5  # heads -> A=soup
        items.append({"id": e["id"], "ja": ja, "google_ref": gg.get(ja, ""),
                      "A": soup[ja] if heads else v8[ja],
                      "B": v8[ja] if heads else soup[ja]})
        key.append({"id": e["id"], "ja": ja,
                    "A_sys": "soup" if heads else "v8",
                    "B_sys": "v8" if heads else "soup",
                    "identical": soup[ja] == v8[ja],
                    "domain": e.get("domain", "")})
    json.dump(items, open(os.path.join(S, f"h2h_items_{split}.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(key, open(os.path.join(S, f"h2h_key_{split}.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    from collections import Counter
    print(f"{split}: {len(items)} cau ({same} cau 2 ban GIONG HET) | "
          f"A_sys={dict(Counter(k['A_sys'] for k in key))}")

# chia chunk cho 5 giam khao
for split, chunks in [("err64", 2), ("rand150", 3)]:
    items = json.load(open(os.path.join(S, f"h2h_items_{split}.json"), encoding="utf-8"))
    n = len(items); sz = -(-n // chunks)
    for i in range(chunks):
        part = items[i * sz:(i + 1) * sz]
        json.dump(part, open(os.path.join(S, f"h2h_chunk_{split}_c{i+1}.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  chunk {split}_c{i+1}: {len(part)} cau")
