#!/usr/bin/env python3
"""Giai ma 2 panel: x3v8 (v8x3_avg vs v8_avg) va x3base (v8x3_avg vs v7a)."""
import json, os, collections

S = os.path.dirname(os.path.abspath(__file__))


def load(p):
    return json.load(open(p, encoding="utf-8"))


def run(tag, split, chunks, name_a, name_b):
    key = {x["id"]: x for x in load(os.path.join(S, f"{tag}_key_{split}.json"))}
    verd = []
    for i in range(1, chunks + 1):
        p = os.path.join(S, f"{tag}_verdict_{split}_c{i}.json")
        if not os.path.exists(p):
            print(f"  !! THIEU {os.path.basename(p)}"); continue
        verd += load(p)
    ids = [v["id"] for v in verd]
    print(f"\n=== {tag}/{split} (n={len(verd)}, trung={len(ids)-len(set(ids))}) ===")
    win = collections.Counter(); qa = collections.Counter(); qb = collections.Counter()
    rows = []; noise = 0
    for v in verd:
        k = key.get(v["id"])
        if k is None:
            continue
        w = v["winner"]
        if k["identical"] and w != "tie":
            noise += 1
        sys_win = "tie" if w == "tie" else (k["A_sys"] if w == "A" else k["B_sys"])
        win[sys_win] += 1
        q1 = v["a_q"] if k["A_sys"] == name_a else v["b_q"]
        q2 = v["a_q"] if k["A_sys"] == name_b else v["b_q"]
        qa[q1] += 1; qb[q2] += 1
        rows.append({"id": v["id"], "ja": k["ja"], "winner": sys_win,
                     f"{name_a}_q": q1, f"{name_b}_q": q2,
                     "identical": k["identical"], "why": v.get("why", "")})
    print(f"  doi dau : {name_a} {win[name_a]} - {name_b} {win[name_b]}  (hoa {win['tie']})")
    print(f"  {name_a:6}: good {qa['good']} / ok {qa['ok']} / bad {qa['bad']}")
    print(f"  {name_b:6}: good {qb['good']} / ok {qb['ok']} / bad {qb['bad']}")
    if noise:
        print(f"  (noise: {noise} cau giong het khong tie)")
    json.dump(rows, open(os.path.join(S, f"{tag}_decoded_{split}.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    return rows


ra = run("x3v8", "err64", 2, "x3", "v8avg")
rb = run("x3v8", "rand150", 3, "x3", "v8avg")
rc = run("x3base", "rand150", 3, "x3", "v7a")

print("\n--- x3 BAD / doi thu khong bad (x3 thua o dau) ---")
for name, rows, other in (("x3v8/err64", ra, "v8avg_q"), ("x3v8/rand", rb, "v8avg_q"),
                          ("x3base/rand", rc, "v7a_q")):
    for x in rows:
        if x["x3_q"] == "bad" and x[other] != "bad":
            print(f"  [{name}] {x['ja'][:52]} | {x['why'][:85]}")
print("--- doi thu BAD / x3 khong bad (x3 cuu duoc gi) ---")
for name, rows, other in (("x3v8/err64", ra, "v8avg_q"), ("x3v8/rand", rb, "v8avg_q"),
                          ("x3base/rand", rc, "v7a_q")):
    for x in rows:
        if x[other] == "bad" and x["x3_q"] != "bad":
            print(f"  [{name}] {x['ja'][:52]} | {x['why'][:85]}")
