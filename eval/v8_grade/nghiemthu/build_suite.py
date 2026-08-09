#!/usr/bin/env python3
"""Dung eval/regression_suite_v1.jsonl — bo hoi quy TICH LUY.

Gom moi cau da biet BAD o it nhat mot version (v7a / v8 / v8_avg / soup / x3),
dedup theo ja. Moi record:
  ja        cau nguon
  sources   list nguon phat hien (err64 / rand150 / 200b / issues5)
  bad_in    list he da do duoc la bad tren cau nay
  trap      mo ta bay (tu why cua giam khao — cai gi de sai)
  family    "point" (kien thuc diem: quan ngu/da nghia/the/so lieu — KD-able)
            | "structure" (nang luc dien: dao vai/phu dinh/da menh de)
Phan loai family bang keyword tho tren why — REVIEW TAY truoc khi dung lam target KD.
"""
import json, os, re, collections

S = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(S, "refs")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load(p):
    return json.load(open(p, encoding="utf-8"))


suite = {}  # ja -> record


def add(ja, source, bad_in, trap):
    r = suite.setdefault(ja, {"ja": ja, "sources": [], "bad_in": [], "traps": []})
    if source not in r["sources"]:
        r["sources"].append(source)
    for b in bad_in:
        if b not in r["bad_in"]:
            r["bad_in"].append(b)
    if trap and trap not in r["traps"]:
        r["traps"].append(trap)


# 1) err64 — 64 cau v7a tung sai (bench 1200)
for e in jl(os.path.join(REF, "v7a_errors.jsonl")):
    add(e["ja"], "err64", ["v7a"], e.get("note") or e.get("why") or "")

# 2) grade_merged (phien goc TrTueN): cau bad cua v7a/v8 tren err64+rand150
gm = load(os.path.join(S, "grade_merged.json"))
for split in ("err64", "rand150"):
    for x in gm[split]:
        bad = [s for s, q in (("v8", x["v8_q"]), ("v7a", x["v7a_q"])) if q == "bad"]
        if bad:
            add(x["ja"], split, bad, x.get("why", ""))

# 3) cac phien cham cua toi: avg (vs v8), final (avg vs v7a), x3, 200b
for f, amap in [
    ("avg_decoded_err64.json", {"avg_q": "v8_avg", "v8_q": "v8"}),
    ("avg_decoded_rand150.json", {"avg_q": "v8_avg", "v8_q": "v8"}),
    ("final_decoded_rand150.json", {"avg_q": "v8_avg", "v7a_q": "v7a"}),
    ("x3v8_decoded_err64.json", {"x3_q": "v8x3", "v8avg_q": "v8_avg"}),
    ("x3v8_decoded_rand150.json", {"x3_q": "v8x3", "v8avg_q": "v8_avg"}),
    ("x3base_decoded_rand150.json", {"x3_q": "v8x3", "v7a_q": "v7a"}),
    ("b200_decoded.json", {"v8avg_q": "v8_avg", "v7a_q": "v7a"}),
]:
    p = os.path.join(S, f)
    if not os.path.exists(p):
        print(f"!! thieu {f}"); continue
    src = "200b" if "b200" in f else ("err64" if "err64" in f else "rand150")
    for x in load(p):
        bad = [name for q, name in amap.items() if x.get(q) == "bad"]
        if bad:
            add(x["ja"], src, bad, x.get("why", ""))

# family heuristic
STRUCT = re.compile(r"đảo|vai|chủ-vị|chủ ngữ|phủ định|nhân quả|cú pháp|gãy|đứt|lặp|rối|scope")
for r in suite.values():
    why = " ".join(r["traps"])
    r["family"] = "structure" if STRUCT.search(why) else "point"
    r["trap"] = (r["traps"][0][:160] if r["traps"] else "")
    del r["traps"]

rows = sorted(suite.values(), key=lambda r: (-len(r["bad_in"]), r["ja"]))
out = os.path.join(S, "regression_suite_v1.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for i, r in enumerate(rows):
        f.write(json.dumps({"id": i + 1, **r}, ensure_ascii=False) + "\n")

c_fam = collections.Counter(r["family"] for r in rows)
c_src = collections.Counter(s for r in rows for s in r["sources"])
print(f"suite v1: {len(rows)} cau -> {out}")
print(f"  family: {dict(c_fam)}")
print(f"  nguon : {dict(c_src)}")
print(f"  bad>=2 he: {sum(1 for r in rows if len(r['bad_in'])>=2)}")
