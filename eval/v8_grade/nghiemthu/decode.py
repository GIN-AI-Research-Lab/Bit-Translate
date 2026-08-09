#!/usr/bin/env python3
"""Gop verdict 5 giam khao + giai ma A/B bang key -> bang soup vs v7a.
In kem so sanh voi bang v8-vs-v7a goc (tu grade_merged.json)."""
import json, os, collections

S = os.path.dirname(os.path.abspath(__file__))
TAG = "a03"


def load(p):
    return json.load(open(p, encoding="utf-8"))


def run(split, chunks):
    key = {x["id"]: x for x in load(os.path.join(S, f"key_{TAG}_{split}.json"))}
    verd = []
    for i in range(1, chunks + 1):
        p = os.path.join(S, f"verdict_{split}_c{i}.json")
        if not os.path.exists(p):
            print(f"  !! THIEU {os.path.basename(p)}")
            continue
        verd += load(p)
    ids = [v["id"] for v in verd]
    dup = len(ids) - len(set(ids))
    win, q_soup, q_v7a, rows = collections.Counter(), collections.Counter(), collections.Counter(), []
    for v in verd:
        k = key.get(v["id"])
        if k is None:
            print(f"  !! id {v['id']} khong co trong key"); continue
        w = v["winner"]
        sys_win = "tie" if w == "tie" else (k["A_sys"] if w == "A" else k["B_sys"])
        win[sys_win] += 1
        sq = v["a_q"] if k["A_sys"] == "soup" else v["b_q"]
        vq = v["a_q"] if k["A_sys"] == "v7a" else v["b_q"]
        q_soup[sq] += 1; q_v7a[vq] += 1
        rows.append({"id": v["id"], "ja": k["ja"], "winner": sys_win,
                     "soup_q": sq, "v7a_q": vq, "why": v.get("why", ""),
                     "domain": k.get("domain", "")})
    print(f"\n=== {split}  (n={len(verd)}, id trung={dup}) ===")
    print(f"  doi dau : soup {win['soup']} - v7a {win['v7a']}  (hoa {win['tie']})")
    print(f"  soup    : good {q_soup['good']} / ok {q_soup['ok']} / bad {q_soup['bad']}")
    print(f"  v7a     : good {q_v7a['good']} / ok {q_v7a['ok']} / bad {q_v7a['bad']}")
    json.dump(rows, open(os.path.join(S, f"decoded_{TAG}_{split}.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    return rows


r64 = run("err64", 2)
r150 = run("rand150", 3)

print("\n" + "=" * 62)
print("SO SANH voi bang V8-vs-v7a goc (cung 2 tap, cung quy trinh):")
old = load(os.path.join(S, "grade_merged.json"))
for split, rows, okey in (("err64", r64, "err64"), ("rand150", r150, "rand150")):
    o = old[okey]
    ow = collections.Counter(x["winner"] for x in o)
    oq = collections.Counter(x["v8_q"] for x in o)
    nw = collections.Counter(x["winner"] for x in rows)
    nq = collections.Counter(x["soup_q"] for x in rows)
    print(f"\n{split}:")
    print(f"   v8   vs v7a : {ow['v8']:>3} - {ow['v7a']:<3} (hoa {ow['tie']:>3}) | "
          f"v8   bad={oq['bad']:>3} good={oq['good']:>3}")
    print(f"   soup vs v7a : {nw['soup']:>3} - {nw['v7a']:<3} (hoa {nw['tie']:>3}) | "
          f"soup bad={nq['bad']:>3} good={nq['good']:>3}")

reg = [x for x in r150 if x["soup_q"] == "bad" and x["v7a_q"] != "bad"]
print(f"\nREGRESSION THAT cua soup tren rand150: {len(reg)} cau (v8 goc: 7)")
for x in reg:
    print(f"  - {x['ja'][:60]} | {x['why'][:90]}")
