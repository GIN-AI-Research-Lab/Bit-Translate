#!/usr/bin/env python3
"""Gộp data Vòng 2: data/synthetic/vong2/out_*.jsonl -> vong2_pairs.{jsonl,ja,vi}

Lớp lọc cuối trước khi đóng gói (QUALITY_GATE lớp 1 + tái áp blacklist):
  - rule: rỗng / vế vi còn ký tự CJK (lẫn Nhật/Trung) / vế ja không có chữ Nhật /
    tỉ lệ độ dài bất thường / trùng (ja,vi)
  - blacklist: mọi ja nằm trong eval/vong2_review_blacklist.jsonl (Claude review)
In kiểm kê theo mode.
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
SYN = ROOT / "data" / "synthetic"
CJK = re.compile(r"[぀-ヿ㐀-鿿]")

bl = set()
blp = ROOT / "eval" / "vong2_review_blacklist.jsonl"
if blp.exists():
    for line in blp.open(encoding="utf-8"):
        try:
            bl.add(json.loads(line)["ja"].strip())
        except Exception:
            pass

seen = set()
rows = []
stats = Counter()
for f in sorted((SYN / "vong2").glob("out_*.jsonl")):
    for line in f.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        stats["read"] += 1
        try:
            p = json.loads(line)
        except json.JSONDecodeError:
            stats["bad_json"] += 1
            continue
        ja = unicodedata.normalize("NFC", (p.get("ja") or "").strip())
        vi = unicodedata.normalize("NFC", (p.get("vi") or "").strip())
        if not ja or not vi:
            stats["empty"] += 1
            continue
        if ja in bl:
            stats["blacklist"] += 1
            continue
        if CJK.search(vi):
            stats["cjk_in_vi"] += 1
            continue
        if not CJK.search(ja):
            stats["no_ja"] += 1
            continue
        r = len(vi) / max(len(ja), 1)
        if r < 0.35 or r > 8:
            stats["ratio"] += 1
            continue
        if (ja, vi) in seen:
            stats["dup"] += 1
            continue
        seen.add((ja, vi))
        rows.append({"ja": ja, "vi": vi, "src": p.get("src", "?")})
        stats["kept"] += 1

with (SYN / "vong2_pairs.jsonl").open("w", encoding="utf-8") as fl, \
     (SYN / "vong2_pairs.ja").open("w", encoding="utf-8") as fj, \
     (SYN / "vong2_pairs.vi").open("w", encoding="utf-8") as fv:
    for p in rows:
        fl.write(json.dumps(p, ensure_ascii=False) + "\n")
        fj.write(p["ja"] + "\n")
        fv.write(p["vi"] + "\n")

print("=== lọc cuối ===")
for k in ("read", "bad_json", "empty", "blacklist", "cjk_in_vi", "no_ja", "ratio", "dup", "kept"):
    print(f"  {k:<12} {stats[k]:,}")
print("=== kiểm kê theo mode ===")
for s, n in Counter(p["src"] for p in rows).most_common():
    print(f"  {s:<14} {n:,}")
print(f"-> data/synthetic/vong2_pairs.{{jsonl,ja,vi}} ({stats['kept']:,} cặp)")
