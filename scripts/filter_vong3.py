#!/usr/bin/env python3
"""Rule filter LỚP 1 cho data vòng 3 (trước Haiku review + LaBSE).

Đọc data/synthetic/vong3/out_*.jsonl →
  data/synthetic/vong3/filtered.jsonl  (đạt)
  data/synthetic/vong3/rejects.jsonl   (rớt, kèm lý do — Haiku review đọc mẫu file này để tune rule)
Luật: ja phải có kana/kanji + không dính chữ Việt có dấu; vi chủ yếu Latin (không sót
kana/kanji trừ tên riêng ngắn); tỉ lệ độ dài; dedup exact (ja,vi); loại câu rập khuôn
placeholder; loại mục nằm trong blacklist (eval/vong3_review_blacklist.jsonl — Haiku ghi).
In thống kê theo (src, m) để biết mode × provider nào bẩn.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
V3 = ROOT / "data" / "synthetic" / "vong3"
BLACKLIST = ROOT / "eval" / "vong3_review_blacklist.jsonl"

JA_CHARS = re.compile(r"[぀-ヿ一-鿿]")
VI_TONE = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)
PLACEHOLDER = re.compile(r"\.\.\.|…|<[a-z]+>|\[[a-z]+\]|XXX|N/A", re.I)


def ja_ratio(s):
    return len(JA_CHARS.findall(s)) / max(1, len(s))


def check(p):
    ja, vi = p["ja"].strip(), p["vi"].strip()
    if not (4 <= len(ja) <= 200):
        return "ja_len"
    if not (2 <= len(vi.split()) <= 60):
        return "vi_len"
    if ja_ratio(ja) < 0.3:
        return "ja_khong_phai_tieng_nhat"
    if VI_TONE.search(ja):
        return "ja_dinh_chu_viet"
    if len(JA_CHARS.findall(vi)) > 4:
        return "vi_sot_tieng_nhat"
    if not VI_TONE.search(vi):
        return "vi_khong_dau"          # câu Việt thật gần như luôn có dấu
    r = len(vi) / max(1, len(ja))
    if not (0.4 <= r <= 6.0):          # vi dài hơn ja là bình thường (kanji nén)
        return "ti_le_do_dai"
    if PLACEHOLDER.search(ja) or PLACEHOLDER.search(vi):
        return "placeholder"
    return None


def main():
    black = set()
    if BLACKLIST.exists():
        for line in BLACKLIST.open(encoding="utf-8"):
            try:
                black.add(json.loads(line)["ja"].strip())
            except Exception:
                pass
    seen, kept, rej = set(), [], []
    stats = Counter()
    for f in sorted(V3.glob("out_*.jsonl")):
        for line in f.open(encoding="utf-8"):
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            tag = (p.get("src", "?"), p.get("m", "?"))
            stats[("tong",) + tag] += 1
            why = check(p)
            # Haiku review 2026-07-18: batch GLM lỗi 10% sample -> bỏ cả batch (thay được bằng Qwen)
            if why is None and p.get("m") == "glm-4.7-flash":
                why = "bo_batch_glm"
            if why is None and p["ja"].strip() in black:
                why = "blacklist"
            if why is None and (p["ja"].strip(), p["vi"].strip()) in seen:
                why = "trung_lap"
            if why:
                p["reject"] = why
                rej.append(p)
                stats[(why,) + tag] += 1
            else:
                seen.add((p["ja"].strip(), p["vi"].strip()))
                kept.append(p)
                stats[("dat",) + tag] += 1
    (V3 / "filtered.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in kept), encoding="utf-8")
    (V3 / "rejects.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in rej), encoding="utf-8")
    print(f"ĐẠT {len(kept)} | RỚT {len(rej)}")
    bymp = Counter()
    for (kind, src, m), n in stats.items():
        if kind in ("tong", "dat"):
            bymp[(src, m, kind)] = n
    for (src, m) in sorted({(s, mm) for (s, mm, _) in bymp}):
        t, d = bymp.get((src, m, "tong"), 0), bymp.get((src, m, "dat"), 0)
        print(f"  {src:12s} {m:24s} {d}/{t} đạt ({100*d/max(1,t):.0f}%)")
    top = Counter()
    for (kind, src, m), n in stats.items():
        if kind not in ("tong", "dat"):
            top[kind] += n
    if top:
        print("  lý do rớt:", ", ".join(f"{k}={v}" for k, v in top.most_common(8)))


if __name__ == "__main__":
    sys.exit(main())
