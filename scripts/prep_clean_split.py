#!/usr/bin/env python3
"""Tách kd_filtered.jsonl -> data/clean/{train,dev}.{ja,vi} (text thuần, thẳng hàng).

Dev holdout: ~3000 cặp deterministic (hash id % DEV_MOD == 0), để theo dõi loss
lúc train. FLORES/hardbench trong eval/ vẫn là bộ eval chất lượng riêng — dev này
chỉ là đồng hồ báo overfit.

  python scripts/prep_clean_split.py data/synthetic/kd_clean/kd_filtered.jsonl
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CLEAN = ROOT / "data" / "clean"
DEV_MOD = 3500   # ~10.5M / 3500 ≈ 3000 câu dev


def is_dev(idv):
    h = int(hashlib.blake2b(str(idv).encode(), digest_size=6).hexdigest(), 16)
    return h % DEV_MOD == 0


def main():
    global CLEAN
    inp = sys.argv[1] if len(sys.argv) > 1 else str(
        ROOT / "data" / "synthetic" / "kd_clean" / "kd_filtered.jsonl")
    # arg 2 (tuỳ chọn): thư mục ra — ổ E đầy nên vòng data mới ghi sang ổ D.
    #   python scripts/prep_clean_split.py D:/.../kd_filtered_niche.jsonl D:/.../clean_v2
    if len(sys.argv) > 2:
        CLEAN = Path(sys.argv[2])
    CLEAN.mkdir(parents=True, exist_ok=True)

    f = {
        ("train", "ja"): open(CLEAN / "train.ja", "w", encoding="utf-8"),
        ("train", "vi"): open(CLEAN / "train.vi", "w", encoding="utf-8"),
        ("dev", "ja"): open(CLEAN / "dev.ja", "w", encoding="utf-8"),
        ("dev", "vi"): open(CLEAN / "dev.vi", "w", encoding="utf-8"),
    }
    n = {"train": 0, "dev": 0}
    total = 0
    with open(inp, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            ja, vi = p.get("ja", "").strip(), p.get("vi", "").strip()
            if not ja or not vi or "\n" in ja or "\n" in vi:
                continue
            split = "dev" if is_dev(p.get("id")) else "train"
            f[(split, "ja")].write(ja + "\n")
            f[(split, "vi")].write(vi + "\n")
            n[split] += 1
            total += 1
            if total % 1_000_000 == 0:
                print(f"  ...{total:,} (train {n['train']:,} / dev {n['dev']:,})", flush=True)
    for h in f.values():
        h.close()
    print(f"\nXONG: train {n['train']:,} | dev {n['dev']:,} -> {CLEAN}")


if __name__ == "__main__":
    main()
