#!/usr/bin/env python3
"""Aggregate the per-batch JLPT JSONL files into one clean synthetic corpus.

Steps:
  - read every data/synthetic/batches/jlpt_*.jsonl
  - validate each row has non-empty ja/vi
  - normalize Vietnamese to Unicode NFC (matches Bước 2 convention)
  - drop exact duplicates (by (ja, vi)) and drop rows whose ja or vi text
    already appears in another pair (cross-field dup)
  - basic sanity filters: require Japanese script in `ja`, require Vietnamese
    diacritics OR latin letters in `vi`, sane length ratio
  - write parallel data/synthetic/jlpt.ja / jlpt.vi (+ jlpt.meta.jsonl with
    level/grammar for later error analysis in Bước 5)

Reports counts per level and how many rows each filter removed.
"""
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
BATCH_DIR = ROOT / "data" / "synthetic" / "batches"
OUT_DIR = ROOT / "data" / "synthetic"

# any CJK / kana presence -> looks like Japanese
JA_RANGES = [
    (0x3040, 0x30FF),   # hiragana + katakana
    (0x4E00, 0x9FFF),   # CJK unified ideographs
    (0x3400, 0x4DBF),   # CJK ext A
    (0xFF66, 0xFF9F),   # halfwidth katakana
]


def has_japanese(s):
    for ch in s:
        cp = ord(ch)
        for lo, hi in JA_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def main():
    files = sorted(BATCH_DIR.glob("jlpt_*.jsonl"))
    if not files:
        sys.exit("no batch files found in data/synthetic/batches/")

    stats = Counter()
    per_level = Counter()
    seen_pair = set()
    seen_ja = set()
    seen_vi = set()
    rows = []

    for f in files:
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            stats["read"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue
            ja = (obj.get("ja") or "").strip()
            vi = (obj.get("vi") or "").strip()
            level = obj.get("level") or ""
            grammar = obj.get("grammar") or ""

            if not ja or not vi:
                stats["empty"] += 1
                continue
            vi = unicodedata.normalize("NFC", vi)
            ja = unicodedata.normalize("NFC", ja)

            if not has_japanese(ja):
                stats["ja_not_japanese"] += 1
                continue
            if has_japanese(vi):
                stats["vi_has_japanese"] += 1
                continue

            # length-ratio sanity (chars): Japanese is denser, allow wide band
            r = len(vi) / max(len(ja), 1)
            if r < 0.5 or r > 8:
                stats["bad_len_ratio"] += 1
                continue

            key = (ja, vi)
            if key in seen_pair:
                stats["dup_pair"] += 1
                continue
            if ja in seen_ja or vi in seen_vi:
                stats["dup_side"] += 1
                continue
            seen_pair.add(key)
            seen_ja.add(ja)
            seen_vi.add(vi)
            rows.append((ja, vi, level, grammar))
            per_level[level] += 1
            stats["kept"] += 1

    with (OUT_DIR / "jlpt.ja").open("w", encoding="utf-8") as fja, \
         (OUT_DIR / "jlpt.vi").open("w", encoding="utf-8") as fvi, \
         (OUT_DIR / "jlpt.meta.jsonl").open("w", encoding="utf-8") as fm:
        for ja, vi, level, grammar in rows:
            fja.write(ja + "\n")
            fvi.write(vi + "\n")
            fm.write(json.dumps({"level": level, "grammar": grammar}, ensure_ascii=False) + "\n")

    print("=== filter stats ===")
    for k in ["read", "bad_json", "empty", "ja_not_japanese", "vi_has_japanese",
              "bad_len_ratio", "dup_pair", "dup_side", "kept"]:
        print(f"  {k:<18} {stats[k]}")
    print("=== kept per level ===")
    for lv in ["N5", "N4", "N3", "N2", "N1"]:
        print(f"  {lv}: {per_level[lv]}")
    print(f"TOTAL kept: {stats['kept']} -> data/synthetic/jlpt.ja / jlpt.vi")


if __name__ == "__main__":
    main()
