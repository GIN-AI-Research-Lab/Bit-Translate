#!/usr/bin/env python3
"""V7A buoc 1.1 — loc cau dev ro (dev.ja trung train.ja) khoi clean_v6.

Dedup theo blake2b(digest_size=8) tren cau .strip(), so voi TOAN BO train.ja 15,3M dong.
Ghi dev.ja/dev.vi SACH vao D:/Bit-Translate-data/clean_v7a/.
"""
import hashlib
from pathlib import Path

CLEAN_V6 = Path("D:/Bit-Translate-data/clean_v6")
OUT = Path("D:/Bit-Translate-data/clean_v7a")
OUT.mkdir(parents=True, exist_ok=True)


def h(s: str) -> bytes:
    return hashlib.blake2b(s.strip().encode("utf-8"), digest_size=8).digest()


train_hashes = set()
with open(CLEAN_V6 / "train.ja", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        train_hashes.add(h(line))
        if i % 2_000_000 == 0:
            print(f"  train.ja {i:,} dong...", flush=True)
print(f"train.ja: {i:,} dong, {len(train_hashes):,} hash duy nhat", flush=True)

dev_ja = (CLEAN_V6 / "dev.ja").read_text(encoding="utf-8").splitlines()
dev_vi = (CLEAN_V6 / "dev.vi").read_text(encoding="utf-8").splitlines()
assert len(dev_ja) == len(dev_vi), "dev ja/vi lech so dong"

kept_ja, kept_vi, n_leak = [], [], 0
for ja, vi in zip(dev_ja, dev_vi):
    if h(ja) in train_hashes:
        n_leak += 1
    else:
        kept_ja.append(ja)
        kept_vi.append(vi)

(OUT / "dev.ja").write_text("\n".join(kept_ja) + "\n", encoding="utf-8")
(OUT / "dev.vi").write_text("\n".join(kept_vi) + "\n", encoding="utf-8")
print(f"dev: {len(dev_ja):,} cap -> giu {len(kept_ja):,}, loai {n_leak:,} cap ro ({n_leak/len(dev_ja)*100:.2f}%)")
print(f"ghi {OUT}/dev.ja + dev.vi")
