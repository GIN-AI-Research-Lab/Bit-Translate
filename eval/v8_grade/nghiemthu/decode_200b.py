#!/usr/bin/env python3
"""Giai ma panel 200b: v8avg-i2s vs v7a-i2s, kem breakdown ngan/dai + kho."""
import json, os, collections

S = os.path.dirname(os.path.abspath(__file__))
key = {x["id"]: x for x in json.load(open(os.path.join(S, "b200_key.json"),
                                          encoding="utf-8"))}
verd = []
for i in (1, 2, 3, 4):
    p = os.path.join(S, f"b200_verdict_c{i}.json")
    if not os.path.exists(p):
        print(f"!! THIEU c{i}"); continue
    verd += json.load(open(p, encoding="utf-8"))
ids = [v["id"] for v in verd]
print(f"n={len(verd)}, trung={len(ids)-len(set(ids))}")


def agg(rows_filter, label):
    win = collections.Counter(); q8 = collections.Counter(); q7 = collections.Counter()
    n = 0
    for v in verd:
        k = key[v["id"]]
        if not rows_filter(k):
            continue
        n += 1
        w = v["winner"]
        sys_win = "tie" if w == "tie" else (k["A_sys"] if w == "A" else k["B_sys"])
        win[sys_win] += 1
        q8[v["a_q"] if k["A_sys"] == "v8avg" else v["b_q"]] += 1
        q7[v["a_q"] if k["A_sys"] == "v7a" else v["b_q"]] += 1
    print(f"\n{label} (n={n}):")
    print(f"  doi dau: v8avg {win['v8avg']} - v7a {win['v7a']} (hoa {win['tie']})")
    print(f"  v8avg  : good {q8['good']} / ok {q8['ok']} / bad {q8['bad']}")
    print(f"  v7a    : good {q7['good']} / ok {q7['ok']} / bad {q7['bad']}")


agg(lambda k: True, "TOAN BO 200")
agg(lambda k: k["len"] == "short", "CAU NGAN")
agg(lambda k: k["len"] == "long", "CAU DAI")
agg(lambda k: k["diff"] == "hard", "CAU KHO")

noise = sum(1 for v in verd
            if key[v["id"]]["identical"] and v["winner"] != "tie")
print(f"\nnoise-check: {noise} cau giong het ma khong tie")

rows = []
for v in verd:
    k = key[v["id"]]
    w = v["winner"]
    sys_win = "tie" if w == "tie" else (k["A_sys"] if w == "A" else k["B_sys"])
    rows.append({"id": v["id"], "ja": k["ja"], "winner": sys_win,
                 "v8avg_q": v["a_q"] if k["A_sys"] == "v8avg" else v["b_q"],
                 "v7a_q": v["a_q"] if k["A_sys"] == "v7a" else v["b_q"],
                 "len": k["len"], "diff": k["diff"], "feat": k["feat"],
                 "why": v.get("why", "")})
json.dump(rows, open(os.path.join(S, "b200_decoded.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("\n--- v8avg BAD ma v7a khong bad ---")
for x in rows:
    if x["v8avg_q"] == "bad" and x["v7a_q"] != "bad":
        print(f"  [{x['len']}/{x['diff']}] {x['ja'][:50]} | {x['why'][:85]}")
print("--- v7a BAD ma v8avg khong bad ---")
for x in rows:
    if x["v7a_q"] == "bad" and x["v8avg_q"] != "bad":
        print(f"  [{x['len']}/{x['diff']}] {x['ja'][:50]} | {x['why'][:85]}")
