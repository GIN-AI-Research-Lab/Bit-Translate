#!/usr/bin/env python3
"""Gộp câu glossary đã sinh: glossary_sents.jsonl (cũ) + gen/out_*.jsonl (mới từ Copilot/khác)
-> khử trùng theo (ja,vi) -> ghi lại data/synthetic/glossary_sents.{jsonl,ja,vi}.

Chạy sau khi provider khác sinh xong các out_*.jsonl. An toàn chạy nhiều lần (idempotent).
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SYN = ROOT / "data" / "synthetic"
GEN = SYN / "gen"

seen, pairs = set(), []


def add(ja, vi, term):
    ja = (ja or "").strip()
    vi = (vi or "").strip()
    if not ja or not vi:
        return
    k = (ja, vi)
    if k in seen:
        return
    seen.add(k)
    pairs.append({"ja": ja, "vi": vi, "term": term})


def eat(path):
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            p = json.loads(line)
        except json.JSONDecodeError:
            continue
        before = len(pairs)
        add(p.get("ja"), p.get("vi"), p.get("term"))
        n += len(pairs) - before
    return n


# bản cũ trước (giữ ưu tiên), rồi các out_*.jsonl mới
old = SYN / "glossary_sents.jsonl"
if old.exists():
    print(f"[cũ] glossary_sents.jsonl: +{eat(old)}")
for f in sorted(GEN.glob("out_*.jsonl")):
    k = eat(f)
    if k:
        print(f"[mới] {f.name}: +{k}")

# ghi lại
(SYN / "glossary_sents.jsonl").write_text(
    "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
(SYN / "glossary_sents.ja").write_text("".join(p["ja"] + "\n" for p in pairs), encoding="utf-8")
(SYN / "glossary_sents.vi").write_text("".join(p["vi"] + "\n" for p in pairs), encoding="utf-8")
print(f"=== TỔNG: {len(pairs)} cặp câu (khử trùng) -> data/synthetic/glossary_sents.* ===")
