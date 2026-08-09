#!/usr/bin/env python3
"""Giai ma cham mu v8_avg vs v8-raw (cung phien)."""
import json, os, collections

S = os.path.dirname(os.path.abspath(__file__))


def load(p):
    return json.load(open(p, encoding="utf-8"))


def run(split, chunks):
    key = {x["id"]: x for x in load(os.path.join(S, f"avg_key_{split}.json"))}
    verd = []
    for i in range(1, chunks + 1):
        p = os.path.join(S, f"avg_verdict_{split}_c{i}.json")
        if not os.path.exists(p):
            print(f"  !! THIEU {os.path.basename(p)}"); continue
        verd += load(p)
    ids = [v["id"] for v in verd]
    print(f"\n=== {split} (n={len(verd)}, id trung={len(ids)-len(set(ids))}) ===")
    win = collections.Counter(); q_a = collections.Counter(); q_8 = collections.Counter()
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
        aq = v["a_q"] if k["A_sys"] == "avg" else v["b_q"]
        eq = v["a_q"] if k["A_sys"] == "v8" else v["b_q"]
        q_a[aq] += 1; q_8[eq] += 1
        rows.append({"id": v["id"], "ja": k["ja"], "winner": sys_win,
                     "avg_q": aq, "v8_q": eq, "identical": k["identical"],
                     "why": v.get("why", "")})
    print(f"  doi dau : v8_avg {win['avg']} - v8 {win['v8']}  (hoa {win['tie']})")
    print(f"  v8_avg  : good {q_a['good']} / ok {q_a['ok']} / bad {q_a['bad']}")
    print(f"  v8      : good {q_8['good']} / ok {q_8['ok']} / bad {q_8['bad']}")
    if noise:
        print(f"  (noise-check: {noise} cau giong het ma khong tie)")
    json.dump(rows, open(os.path.join(S, f"avg_decoded_{split}.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    return rows


r64 = run("err64", 2)
r150 = run("rand150", 3)

print("\n--- v8_avg BAD ma v8 khong bad ---")
for split, rows in (("err64", r64), ("rand150", r150)):
    for x in rows:
        if x["avg_q"] == "bad" and x["v8_q"] != "bad":
            print(f"  [{split}] {x['ja'][:55]} | {x['why'][:90]}")
print("--- v8 BAD ma v8_avg khong bad ---")
for split, rows in (("err64", r64), ("rand150", r150)):
    for x in rows:
        if x["v8_q"] == "bad" and x["avg_q"] != "bad":
            print(f"  [{split}] {x['ja'][:55]} | {x['why'][:90]}")
