#!/usr/bin/env python3
"""Áp verdict Haiku review (+ phán xử lại của Fable) lên gloss harvest vòng 3.

Luật:
  1. ok=false lý do CHỈ về reading (romaji/format) → GIỮ mục, xoá trường reading
     (reading là metadata, prompt sinh câu không dùng).
  2. ok=false nằm trong RESCUE (Fable phán xử lại — Haiku yếu thành ngữ Việt,
     loại nhầm mục thật thông dụng) → GIỮ nguyên.
  3. ok=false còn lại → LOẠI.
Ghi: vong3_glosses_{ja,vi}_reviewed.jsonl, idiom_glosses_v3_reviewed.jsonl
   + POOL gộp cho gen: vong3_gloss_pool.jsonl (= 476 reviewed vòng 2 + 3 file trên).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
GEN = ROOT / "data" / "synthetic" / "gen"
RV = ROOT / "eval" / "review_v3"

# Fable phán xử lại 2026-07-18 (xem eval/review_v3/verdicts_gloss_vi.jsonl):
# mục thật + thông dụng + mapping đúng bị Haiku loại nhầm.
RESCUE = {"gloss_vi": {10, 50, 51, 53, 58, 90, 107, 150, 151, 159, 161, 185, 186, 187, 189, 205}}

READING_ONLY = re.compile(r"reading|romaji|romanji|hiragana", re.I)
NOT_READING = re.compile(r"dịch|vi sai|nghĩa|không tự nhiên|meme|phổ biến|bịa|tồn tại|typo|"
                         r"không khớp|lỗi thời|thiếu|JoJo|anime|từ cổ|ngôi|intensity", re.I)

SRC = [
    ("gloss_ja", GEN / "vong3_glosses_ja.jsonl", GEN / "vong3_glosses_ja_reviewed.jsonl"),
    ("gloss_vi", GEN / "vong3_glosses_vi.jsonl", GEN / "vong3_glosses_vi_reviewed.jsonl"),
    ("gloss_classic", GEN / "idiom_glosses.jsonl", GEN / "idiom_glosses_v3_reviewed.jsonl"),
]


def main():
    # POOL thứ tự CỐ ĐỊNH (todo đánh index theo đây, đừng đổi):
    #   [0 .. nVI)          mục gốc VIỆT (side=vi)     — mode idcv
    #   [nVI .. )           mục gốc NHẬT (side=ja): 476 vòng 2 + classic mới + slang JA mới — mode idc
    parts = {}
    old = GEN / "idiom_glosses_reviewed.jsonl"     # 476 vòng 2 — đã sạch
    parts["old"] = [json.loads(l) for l in old.open(encoding="utf-8") if l.strip()]
    for src, inf, outf in SRC:
        verdicts = {}
        vf = RV / f"verdicts_{src}.jsonl"
        if vf.exists():
            for line in vf.open(encoding="utf-8"):
                v = json.loads(line)
                verdicts[v["i"]] = v
        entries = [json.loads(l) for l in inf.open(encoding="utf-8") if l.strip()]
        kept, dropped, fixed = [], 0, 0
        for i, e in enumerate(entries):
            v = verdicts.get(i)
            if v is None or v.get("ok"):
                kept.append(e)
                continue
            reason = v.get("reason", "")
            if i in RESCUE.get(src, set()):
                kept.append(e)
                fixed += 1
            elif READING_ONLY.search(reason) and not NOT_READING.search(reason):
                e.pop("reading", None)
                kept.append(e)
                fixed += 1
            else:
                dropped += 1
        outf.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept),
                        encoding="utf-8")
        parts[src] = kept
        print(f"{src:14s} {len(entries)} -> giữ {len(kept)} (cứu {fixed}, loại {dropped})")
    for e in parts["gloss_vi"]:
        e["side"] = "vi"
    pool = parts["gloss_vi"] + parts["old"] + parts["gloss_classic"] + parts["gloss_ja"]
    poolf = GEN / "vong3_gloss_pool.jsonl"
    poolf.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in pool),
                     encoding="utf-8")
    n_vi = len(parts["gloss_vi"])
    n_new_ja = len(parts["gloss_classic"]) + len(parts["gloss_ja"])
    print(f"POOL {len(pool)} mục -> {poolf}")
    print(f"  layout: VI [0..{n_vi}) | JA cũ [{n_vi}..{n_vi+len(parts['old'])}) | "
          f"JA mới [{n_vi+len(parts['old'])}..{len(pool)})")
    n_lit = sum(1 for e in pool if e.get("lit"))
    print(f"  mục có nghĩa đen (ứng viên ct): {n_lit}")


if __name__ == "__main__":
    main()
