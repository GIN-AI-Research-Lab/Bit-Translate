#!/usr/bin/env python3
"""Dung bo cham mu A/B: SOUP vs v7a, tren DUNG 2 tap cua vong V8
(err64 + rand150) de so sanh truc tiep voi bang v8-vs-v7a.

Danh sach cau lay TU KEY FILE cu (khong random lai) -> cung tap, cung id.
A/B randomize lai bang seed rieng.

  python build_soup_grade.py <soup_translations.jsonl> <tag>
"""
import json, random, sys, os

S = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(S, "refs")

soup_path, TAG = sys.argv[1], sys.argv[2]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


soup = {o["ja"]: o["vi"] for o in load_jsonl(soup_path)}
v7a = {o["ja"]: o["vi"] for o in load_jsonl(os.path.join(REF, "v7a_1200.jsonl"))}
gg = {o["ja"]: o["gg"] for o in load_jsonl(os.path.join(REF, "google_1200.jsonl"))}

random.seed(20260809)

for split, keyfile in [("err64", "grade_err64_key.json"),
                       ("rand150", "grade_rand150_key.json")]:
    old = json.load(open(os.path.join(REF, keyfile), encoding="utf-8"))
    items, key = [], []
    miss = 0
    for e in old:
        ja = e["ja"]
        if ja not in soup or ja not in v7a:
            miss += 1
            continue
        heads = random.random() < 0.5          # heads -> A=soup
        items.append({"id": e["id"], "ja": ja, "google_ref": gg.get(ja, ""),
                      "A": soup[ja] if heads else v7a[ja],
                      "B": v7a[ja] if heads else soup[ja]})
        key.append({"id": e["id"], "ja": ja,
                    "A_sys": "soup" if heads else "v7a",
                    "B_sys": "v7a" if heads else "soup",
                    "domain": e.get("domain", "")})
    with open(os.path.join(S, f"items_{TAG}_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    with open(os.path.join(S, f"key_{TAG}_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=1)
    from collections import Counter
    print(f"{split}: {len(items)} cau (thieu {miss}) | A_sys={dict(Counter(k['A_sys'] for k in key))}")
