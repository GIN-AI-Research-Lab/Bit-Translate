#!/usr/bin/env python3
"""Dựng bộ chấm mù A/B v8 vs v7a: 64 câu lỗi + 150 câu random no-regression.
Randomize A/B từng câu (seed cố định), lưu key giải mã riêng."""
import json, random, os
S = "C:/Users/ADMINI~1/AppData/Local/Temp/claude/F--Project-Ai-Bit-Translate/cb22c430-19c2-41f3-9d50-82854664bfd4/scratchpad"

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

v8 = {o["ja"]: o["vi"] for o in load_jsonl(f"{S}/v8_final.jsonl")}
v7a = {o["ja"]: o["vi"] for o in load_jsonl(f"{S}/v7a_1200.jsonl")}
gg = {o["ja"]: o["gg"] for o in load_jsonl(f"{S}/google_1200.jsonl")}
errs = load_jsonl(f"{S}/v7a_errors.jsonl")
err_ja = {e["ja"] for e in errs}
err_dom = {e["ja"]: e.get("domain", "") for e in errs}

ja_all = [l.strip() for l in open(f"{S}/combined_1200.txt", encoding="utf-8") if l.strip()]

# set1 = 64 câu lỗi (theo thứ tự trong errors file, giữ id gốc từ ja_all)
id_of = {ja: i + 1 for i, ja in enumerate(ja_all)}
set1_ja = [e["ja"] for e in errs if e["ja"] in v8 and e["ja"] in v7a]

# set2 = 150 câu random từ phần còn lại (không thuộc 64), seed cố định
rest = [ja for ja in ja_all if ja not in err_ja and ja in v8 and ja in v7a]
random.seed(42)
set2_ja = random.sample(rest, 150)

def build(ja_list, tag):
    items, key = [], []
    for ja in ja_list:
        # coin flip: heads -> A=v8,B=v7a ; tails -> A=v7a,B=v8
        heads = random.random() < 0.5
        A = v8[ja] if heads else v7a[ja]
        B = v7a[ja] if heads else v8[ja]
        rid = id_of[ja]
        items.append({
            "id": rid,
            "ja": ja,
            "google_ref": gg.get(ja, ""),
            "A": A,
            "B": B,
        })
        key.append({"id": rid, "ja": ja,
                    "A_sys": "v8" if heads else "v7a",
                    "B_sys": "v7a" if heads else "v8",
                    "domain": err_dom.get(ja, "")})
    return items, key

s1_items, s1_key = build(set1_ja, "err64")
s2_items, s2_key = build(set2_ja, "rand150")

with open(f"{S}/grade_err64_items.json", "w", encoding="utf-8") as f:
    json.dump(s1_items, f, ensure_ascii=False, indent=1)
with open(f"{S}/grade_err64_key.json", "w", encoding="utf-8") as f:
    json.dump(s1_key, f, ensure_ascii=False, indent=1)
with open(f"{S}/grade_rand150_items.json", "w", encoding="utf-8") as f:
    json.dump(s2_items, f, ensure_ascii=False, indent=1)
with open(f"{S}/grade_rand150_key.json", "w", encoding="utf-8") as f:
    json.dump(s2_key, f, ensure_ascii=False, indent=1)

print(f"set1 (64 loi): {len(s1_items)} cau -> grade_err64_items.json")
print(f"set2 (random no-regression): {len(s2_items)} cau -> grade_rand150_items.json")
# san check phan bo A/B
from collections import Counter
print("set1 A_sys:", dict(Counter(k["A_sys"] for k in s1_key)))
print("set2 A_sys:", dict(Counter(k["A_sys"] for k in s2_key)))
