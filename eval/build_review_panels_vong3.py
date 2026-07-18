#!/usr/bin/env python3
"""Dựng panel cho Haiku review data vòng 3 (lớp 2, sau rule filter).

Nguồn + chiến lược (PROVIDERS.md: mode nào lỗi >5% thì đọc full):
  - GLOSS harvest (vong3_glosses_ja/vi.jsonl, idiom_glosses.jsonl mới): review FULL
    — từ điển là hạt giống, sai là dạy hư model hàng loạt.
  - Câu sinh (data/synthetic/vong3/filtered.jsonl): sample phân tầng theo (src, m)
    10% mỗi nhóm (min 20, max 60).
Ghi eval/review_v3/panel_*.jsonl (mỗi dòng 1 mục cần chấm, đánh số "i" để đối chiếu).
Chạy lại an toàn: xoá panel cũ trước khi ghi.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
GEN = ROOT / "data" / "synthetic" / "gen"
V3 = ROOT / "data" / "synthetic" / "vong3"
OUT = ROOT / "eval" / "review_v3"
OUT.mkdir(exist_ok=True)
CHUNK = 50
rng = random.Random(42)

for old in OUT.glob("panel_*.jsonl"):
    old.unlink()


def write_panels(name, items):
    for ci in range(0, len(items), CHUNK):
        chunk = items[ci:ci + CHUNK]
        f = OUT / f"panel_{name}_{ci // CHUNK:02d}.jsonl"
        f.write_text("".join(json.dumps({**it, "i": ci + j}, ensure_ascii=False) + "\n"
                             for j, it in enumerate(chunk)), encoding="utf-8")
    print(f"{name}: {len(items)} mục -> {(len(items) + CHUNK - 1) // CHUNK} panel")


# 1) gloss — full review
for gf, name in [(GEN / "vong3_glosses_ja.jsonl", "gloss_ja"),
                 (GEN / "vong3_glosses_vi.jsonl", "gloss_vi"),
                 (GEN / "idiom_glosses.jsonl", "gloss_classic")]:
    if gf.exists():
        items = [json.loads(l) for l in gf.open(encoding="utf-8") if l.strip()]
        write_panels(name, items)

# 2) câu sinh — sample phân tầng theo (src, m)
if (V3 / "filtered.jsonl").exists():
    groups = defaultdict(list)
    for line in (V3 / "filtered.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        groups[(p.get("src", "?"), p.get("m", "?"))].append(p)
    for (src, m), items in sorted(groups.items()):
        n = max(20, min(60, len(items) // 10))
        sample = rng.sample(items, min(n, len(items)))
        tag = f"sent_{src}_{re_m}" if (re_m := m.replace('/', '_').replace(':', '_')) else src
        write_panels(tag, sample)
