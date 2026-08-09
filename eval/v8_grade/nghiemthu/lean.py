#!/usr/bin/env python3
"""Tren cac cau QUAN TRONG da biet nhan (tu grade_merged.json vong V8),
soup bam theo v8 hay v7a? So khop chuoi chinh xac.

- 7 cau REGRESSION (v8 bad, v7a khong bad)  -> muon soup == v7a
- 31 cau v8 SUA DUOC (v8_q=good, v7a_q=bad) -> muon soup == v8
"""
import json, os

S = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(S, "refs")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


v7a = {o["ja"]: o["vi"] for o in jl(os.path.join(REF, "v7a_1200.jsonl"))}
v8 = {o["ja"]: o["vi"] for o in jl(os.path.join(REF, "v8_final.jsonl"))}
soups = {a: {o["ja"]: o["vi"] for o in jl(os.path.join(S, f"v8_step16750_soupA{a}.jsonl"))}
         for a in ("03", "05", "07")}

merged = json.load(open(os.path.join(S, "grade_merged.json"), encoding="utf-8"))

regress = [x for x in merged["rand150"] if x["v8_q"] == "bad" and x["v7a_q"] != "bad"]
fixed = [x for x in merged["err64"] if x["v8_q"] == "good" and x["v7a_q"] == "bad"]
print(f"n regression = {len(regress)} | n v8-fixed = {len(fixed)}\n")

hdr = f"{'alpha':>6} | {'REGRESSION: giong v7a (tot)':>28} | {'V8-FIXED: giong v8 (tot)':>26}"
print(hdr)
print("-" * len(hdr))
for a, sp in soups.items():
    r_v7 = sum(1 for x in regress if sp.get(x["ja"]) == v7a[x["ja"]])
    r_v8 = sum(1 for x in regress if sp.get(x["ja"]) == v8[x["ja"]])
    f_v8 = sum(1 for x in fixed if sp.get(x["ja"]) == v8[x["ja"]])
    f_v7 = sum(1 for x in fixed if sp.get(x["ja"]) == v7a[x["ja"]])
    print(f"  0.{a[1]}  | {r_v7:>2}/{len(regress)} giong v7a, {r_v8:>2} giong v8, "
          f"{len(regress)-r_v7-r_v8:>2} khac ca hai | "
          f"{f_v8:>2}/{len(fixed)} giong v8, {f_v7:>2} giong v7a, "
          f"{len(fixed)-f_v8-f_v7:>2} khac ca hai")

print("\n=== CHI TIET 7 CAU REGRESSION (alpha 0.3) ===")
sp = soups["03"]
for x in regress:
    ja = x["ja"]
    tag = "=v7a(OK)" if sp.get(ja) == v7a[ja] else ("=v8(xau)" if sp.get(ja) == v8[ja] else "MOI")
    print(f"\n[{tag}] {ja}")
    print(f"   v7a : {v7a[ja]}")
    print(f"   v8  : {v8[ja]}")
    print(f"   a0.3: {sp.get(ja)}")
    print(f"   why : {x['why'][:150]}")
