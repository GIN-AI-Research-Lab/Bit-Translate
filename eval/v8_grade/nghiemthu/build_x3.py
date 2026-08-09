#!/usr/bin/env python3
"""Dung 2 panel cham mu cho v8x3_avg:
  Panel A: v8x3_avg vs v8_avg  (err64 + rand150) — chung ket 2 ung vien
  Panel B: v8x3_avg vs v7a    (rand150)          — thue regression vs baseline
"""
import json, random, os

S = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(S, "refs")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


x3 = {o["ja"]: o["vi"] for o in jl(os.path.join(S, "v8_step16750_x3avg.jsonl"))}
v8avg = {o["ja"]: o["vi"] for o in jl(os.path.join(S, "v8_step16750_v8avg.jsonl"))}
v7a = {o["ja"]: o["vi"] for o in jl(os.path.join(REF, "v7a_1200.jsonl"))}
gg = {o["ja"]: o["gg"] for o in jl(os.path.join(REF, "google_1200.jsonl"))}

random.seed(20260810)


def build(tag, sysA_name, sysA, sysB_name, sysB, splits):
    for split, keyfile, chunks in splits:
        old = json.load(open(os.path.join(REF, keyfile), encoding="utf-8"))
        items, key = [], []
        same = 0
        for e in old:
            ja = e["ja"]
            if sysA[ja] == sysB[ja]:
                same += 1
            heads = random.random() < 0.5
            items.append({"id": e["id"], "ja": ja, "google_ref": gg.get(ja, ""),
                          "A": sysA[ja] if heads else sysB[ja],
                          "B": sysB[ja] if heads else sysA[ja]})
            key.append({"id": e["id"], "ja": ja,
                        "A_sys": sysA_name if heads else sysB_name,
                        "B_sys": sysB_name if heads else sysA_name,
                        "identical": sysA[ja] == sysB[ja]})
        json.dump(key, open(os.path.join(S, f"{tag}_key_{split}.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)
        n = len(items); sz = -(-n // chunks)
        for i in range(chunks):
            part = items[i * sz:(i + 1) * sz]
            json.dump(part, open(os.path.join(S, f"{tag}_chunk_{split}_c{i+1}.json"),
                                 "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{tag}/{split}: {n} cau ({same} giong het), {chunks} chunk")


build("x3v8", "x3", x3, "v8avg", v8avg,
      [("err64", "grade_err64_key.json", 2), ("rand150", "grade_rand150_key.json", 3)])
build("x3base", "x3", x3, "v7a", v7a,
      [("rand150", "grade_rand150_key.json", 3)])
