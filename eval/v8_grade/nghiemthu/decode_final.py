#!/usr/bin/env python3
"""Giai ma vong chot: v8_avg vs v7a tren rand150."""
import json, os, collections

S = os.path.dirname(os.path.abspath(__file__))


def load(p):
    return json.load(open(p, encoding="utf-8"))


key = {x["id"]: x for x in load(os.path.join(S, "final_key_rand150.json"))}
verd = []
for i in (1, 2, 3):
    p = os.path.join(S, f"final_verdict_c{i}.json")
    if not os.path.exists(p):
        print(f"!! THIEU c{i}"); continue
    verd += load(p)
ids = [v["id"] for v in verd]
print(f"n={len(verd)}, id trung={len(ids)-len(set(ids))}")

win = collections.Counter(); q_a = collections.Counter(); q_7 = collections.Counter()
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
    vq = v["a_q"] if k["A_sys"] == "v7a" else v["b_q"]
    q_a[aq] += 1; q_7[vq] += 1
    rows.append({"id": v["id"], "ja": k["ja"], "winner": sys_win,
                 "avg_q": aq, "v7a_q": vq, "why": v.get("why", "")})

print(f"doi dau : v8_avg {win['avg']} - v7a {win['v7a']}  (hoa {win['tie']})")
print(f"v8_avg  : good {q_a['good']} / ok {q_a['ok']} / bad {q_a['bad']}")
print(f"v7a     : good {q_7['good']} / ok {q_7['ok']} / bad {q_7['bad']}")
if noise:
    print(f"(noise-check: {noise} cau giong het ma khong tie)")
json.dump(rows, open(os.path.join(S, "final_decoded_rand150.json"), "w",
                     encoding="utf-8"), ensure_ascii=False, indent=1)

print("\n--- REGRESSION: v8_avg bad ma v7a khong bad ---")
for x in rows:
    if x["avg_q"] == "bad" and x["v7a_q"] != "bad":
        print(f"  {x['ja'][:60]} | {x['why'][:100]}")
print("--- v7a bad ma v8_avg khong bad (avg tot hon baseline o dau) ---")
for x in rows:
    if x["v7a_q"] == "bad" and x["avg_q"] != "bad":
        print(f"  {x['ja'][:60]} | {x['why'][:100]}")
