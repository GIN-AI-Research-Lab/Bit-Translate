# -*- coding: utf-8 -*-
"""Aggregate Opus-5 blind judge scores for bench_opus200b.

Joins judges/panel_*.json (blind scores keyed by A-D) with
../judge_200b/panel_key.json (letter -> system) and ../bench_opus200b.jsonl
(len/feat/diff metadata). Prints per-system tables and McNemar tests on 'use'.
"""
import json, glob, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = json.load(open(os.path.join(HERE, "..", "judge_200b", "panel_key.json"), encoding="utf-8"))
META = {}
for line in open(os.path.join(HERE, "..", "bench_opus200b.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    META[str(r["id"])] = r

scores = {}  # id -> system -> [acc, nat, use]
for f in sorted(glob.glob(os.path.join(HERE, "judges", "panel_*.json"))):
    for sid, cols in json.load(open(f, encoding="utf-8")).items():
        assert sid in KEY, f"id {sid} missing from key"
        scores[sid] = {KEY[sid][L]: v for L, v in cols.items()}

ids = sorted(scores, key=int)
print(f"n = {len(ids)} sentences x 4 systems judged")
systems = ["v7ai2s", "google", "gate", "v6"]

def pct(vals):
    return 100.0 * sum(vals) / len(vals) if vals else float("nan")

def table(subset, label):
    if not subset:
        return
    print(f"\n== {label} (n={len(subset)}) ==")
    print(f"{'system':8} {'acc==2':>7} {'nat==2':>7} {'use':>7}")
    for s in systems:
        a = pct([1 if scores[i][s][0] == 2 else 0 for i in subset])
        n = pct([1 if scores[i][s][1] == 2 else 0 for i in subset])
        u = pct([scores[i][s][2] for i in subset])
        print(f"{s:8} {a:6.1f}% {n:6.1f}% {u:6.1f}%")

table(ids, "TOTAL")
for ln in ["short", "long"]:
    table([i for i in ids if META[i]["len"] == ln], f"len={ln}")
for d in ["easy", "med", "hard"]:
    table([i for i in ids if META[i].get("diff") == d], f"diff={d}")

print("\n== by domain: use% ==")
domains = sorted({META[i]["feat"] for i in ids})
print(f"{'domain':14}" + "".join(f"{s:>9}" for s in systems))
for dom in domains:
    sub = [i for i in ids if META[i]["feat"] == dom]
    row = f"{dom:14}"
    for s in systems:
        row += f"{pct([scores[i][s][2] for i in sub]):8.0f}%"
    print(row + f"  (n={len(sub)})")

# McNemar exact (binomial) on 'use'
def mcnemar(s1, s2, subset):
    b = sum(1 for i in subset if scores[i][s1][2] == 1 and scores[i][s2][2] == 0)
    c = sum(1 for i in subset if scores[i][s1][2] == 0 and scores[i][s2][2] == 1)
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n * 2
    return b, c, min(p, 1.0)

print("\n== McNemar on 'use' (b = first-only-wins, c = second-only-wins) ==")
for pair in [("v7ai2s", "google"), ("v7ai2s", "v6"), ("v7ai2s", "gate"), ("gate", "v6")]:
    for label, subset in [("total", ids),
                          ("long", [i for i in ids if META[i]["len"] == "long"]),
                          ("short", [i for i in ids if META[i]["len"] == "short"])]:
        b, c, p = mcnemar(pair[0], pair[1], subset)
        print(f"{pair[0]:8} vs {pair[1]:8} [{label:5}]  +{b:<3} -{c:<3} p={p:.3f}")

# where all four systems fail (bench hardness / shared traps)
allfail = [i for i in ids if all(scores[i][s][2] == 0 for s in systems)]
print(f"\nall-4-fail sentences: {len(allfail)} -> {allfail}")
onlyv7 = [i for i in ids if scores[i]['v7ai2s'][2] == 1 and scores[i]['google'][2] == 0]
onlyg = [i for i in ids if scores[i]['v7ai2s'][2] == 0 and scores[i]['google'][2] == 1]
print(f"v7ai2s-beats-google ids: {onlyv7}")
print(f"google-beats-v7ai2s ids: {onlyg}")
