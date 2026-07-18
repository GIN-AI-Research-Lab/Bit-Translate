#!/usr/bin/env python3
"""Sinh mẫu glossary/idiom-injection `[term=訳語]` bằng CODE — 0 token (PLAN §4.3.1, Opus G-6).

Mục đích: train model BIẾT DÙNG gợi ý app nhét vào đầu input lúc chạy
(app quét câu nguồn theo từ điển → chèn `[gấu=恋人]` → model dịch theo gợi ý).
Đây là lợi thế "tùy biến theo khách" mà Google/Haiku không có (PLAN_RANKUP §2.1#4).

Cách sinh: quét các cặp câu vòng 3 ĐÃ SẠCH (filtered.jsonl) tìm cặp mà câu nguồn
chứa mục từ trong pool 1.247 mục → tạo BẢN SAO có hint ở đầu vế nguồn:
  gốc JA:  {"ja": "[骨が折れる=vất vả cực nhọc] <câu ja>", "vi": <câu vi>, "dir": "ja2vi"}
  gốc VI:  {"vi": "[toang=詰んだ] <câu vi>", "ja": <câu ja>, "dir": "vi2ja"}
Data MỘT CHIỀU (field dir — mix cấm tự đảo). Bản KHÔNG hint chính là cặp gốc đã có
trong mix → model không lệ thuộc hint. Cap mỗi mục ≤6 mẫu, tổng ≤10k.
Chạy: python scripts/gen_glossary_inject.py → data/synthetic/vong3/glossary_inject.jsonl
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
GEN = ROOT / "data" / "synthetic" / "gen"
V3 = ROOT / "data" / "synthetic" / "vong3"
OUT = V3 / "glossary_inject.jsonl"
CAP_PER_ENTRY = 6
CAP_TOTAL = 10000
rng = random.Random(42)


def main():
    pool = [json.loads(l) for l in (GEN / "vong3_gloss_pool.jsonl").open(encoding="utf-8")]
    # mục gốc JA: hint cho chiều ja2vi (khóa quét = ja, giá trị = nghĩa vi)
    ja_entries = [(e["ja"].strip(), e["vi"].strip()) for e in pool
                  if e.get("side") != "vi" and len(e["ja"].strip()) >= 2]
    # mục gốc VI: hint cho chiều vi2ja (khóa quét = vi, giá trị = cách nói ja)
    vi_entries = [(e["vi"].strip(), e["ja"].strip()) for e in pool
                  if e.get("side") == "vi" and len(e["vi"].strip()) >= 2]
    pairs = [json.loads(l) for l in (V3 / "filtered.jsonl").open(encoding="utf-8")]

    used = {k: 0 for k, _ in ja_entries}
    used.update({k: 0 for k, _ in vi_entries})
    out, seen = [], set()
    rng.shuffle(pairs)
    for p in pairs:
        ja, vi = p["ja"].strip(), p["vi"].strip()
        if "[" in ja or "[" in vi:
            continue
        for term, gloss in ja_entries:
            if used[term] >= CAP_PER_ENTRY or term not in ja:
                continue
            rec = {"ja": f"[{term}={gloss}] {ja}", "vi": vi,
                   "src": "vong3_gi", "dir": "ja2vi"}
            k = (rec["ja"], rec["vi"])
            if k not in seen:
                seen.add(k)
                out.append(rec)
                used[term] += 1
            break   # 1 hint/câu — khớp cách app chèn từng term một
        else:
            for term, gloss in vi_entries:
                if used[term] >= CAP_PER_ENTRY or term.lower() not in vi.lower():
                    continue
                rec = {"vi": f"[{term}={gloss}] {vi}", "ja": ja,
                       "src": "vong3_gi", "dir": "vi2ja"}
                k = (rec["ja"], rec["vi"])
                if k not in seen:
                    seen.add(k)
                    out.append(rec)
                    used[term] += 1
                break
        if len(out) >= CAP_TOTAL:
            break
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
                   encoding="utf-8")
    n_ja = sum(1 for r in out if r["dir"] == "ja2vi")
    n_entry = sum(1 for v in used.values() if v > 0)
    print(f"GHI {len(out):,} mẫu inject ({n_ja} ja2vi / {len(out)-n_ja} vi2ja) "
          f"phủ {n_entry}/{len(used)} mục -> {OUT}")


if __name__ == "__main__":
    main()
