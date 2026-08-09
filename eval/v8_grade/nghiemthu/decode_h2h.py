#!/usr/bin/env python3
"""Giai ma cham mu TRUC TIEP soup_a03 vs v8 (cung phien)."""
import json, os, collections

S = os.path.dirname(os.path.abspath(__file__))


def load(p):
    return json.load(open(p, encoding="utf-8"))


def run(split, chunks):
    key = {x["id"]: x for x in load(os.path.join(S, f"h2h_key_{split}.json"))}
    verd = []
    for i in range(1, chunks + 1):
        p = os.path.join(S, f"h2h_verdict_{split}_c{i}.json")
        if not os.path.exists(p):
            print(f"  !! THIEU {os.path.basename(p)}"); continue
        verd += load(p)
    ids = [v["id"] for v in verd]
    print(f"\n=== {split} (n={len(verd)}, id trung={len(ids)-len(set(ids))}) ===")
    win = collections.Counter(); q_s = collections.Counter(); q_8 = collections.Counter()
    rows = []
    ident_disagree = 0
    for v in verd:
        k = key[v["id"]]
        w = v["winner"]
        if k["identical"] and w != "tie":
            ident_disagree += 1  # 2 ban giong het ma giam khao khong tie -> noise check
        sys_win = "tie" if w == "tie" else (k["A_sys"] if w == "A" else k["B_sys"])
        win[sys_win] += 1
        sq = v["a_q"] if k["A_sys"] == "soup" else v["b_q"]
        eq = v["a_q"] if k["A_sys"] == "v8" else v["b_q"]
        q_s[sq] += 1; q_8[eq] += 1
        rows.append({"id": v["id"], "ja": k["ja"], "winner": sys_win,
                     "soup_q": sq, "v8_q": eq, "identical": k["identical"],
                     "why": v.get("why", ""), "domain": k.get("domain", "")})
    print(f"  doi dau : soup {win['soup']} - v8 {win['v8']}  (hoa {win['tie']})")
    print(f"  soup    : good {q_s['good']} / ok {q_s['ok']} / bad {q_s['bad']}")
    print(f"  v8      : good {q_8['good']} / ok {q_8['ok']} / bad {q_8['bad']}")
    if ident_disagree:
        print(f"  (noise-check: {ident_disagree} cau giong het ma khong tie)")
    json.dump(rows, open(os.path.join(S, f"h2h_decoded_{split}.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    return rows


r64 = run("err64", 2)
r150 = run("rand150", 3)

print("\n--- cau soup BAD ma v8 khong bad (soup thua o dau) ---")
for split, rows in (("err64", r64), ("rand150", r150)):
    for x in rows:
        if x["soup_q"] == "bad" and x["v8_q"] != "bad":
            print(f"  [{split}] {x['ja'][:55]} | {x['why'][:90]}")
print("--- cau v8 BAD ma soup khong bad (soup cuu duoc gi) ---")
for split, rows in (("err64", r64), ("rand150", r150)):
    for x in rows:
        if x["v8_q"] == "bad" and x["soup_q"] != "bad":
            print(f"  [{split}] {x['ja'][:55]} | {x['why'][:90]}")
