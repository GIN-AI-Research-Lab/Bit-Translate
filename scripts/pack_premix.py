#!/usr/bin/env python3
"""Đóng gói premix data/bin thành .tar.zst — bằng python zstandard (máy không có zstd hệ thống).

Cấu trúc khớp bin_mix_vong2.tar.zst (cloud/restore giải nén bằng `tar -I zstd -xf`):
  data/bin/train.{tokens.u16,index.npy}  data/bin/dev.*  data/clean/flores/flores.*
Chạy: python scripts/pack_premix.py <tên_file_ra.tar.zst>
"""
import sys
import tarfile
from pathlib import Path

import zstandard

ROOT = Path(__file__).parent.parent
OUT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "bin_mix_vong3a.tar.zst")
FILES = [
    "data/bin/train.tokens.u16", "data/bin/train.index.npy",
    "data/bin/dev.tokens.u16", "data/bin/dev.index.npy",
    "data/clean/flores/flores.dev.ja", "data/clean/flores/flores.dev.vi",
    "data/clean/flores/flores.devtest.ja", "data/clean/flores/flores.devtest.vi",
]

cctx = zstandard.ZstdCompressor(level=10, threads=-1)
with open(OUT, "wb") as f, cctx.stream_writer(f) as zw, \
        tarfile.open(fileobj=zw, mode="w|") as tar:
    for rel in FILES:
        p = ROOT / rel
        if not p.exists():
            sys.exit(f"!! thiếu {rel}")
        tar.add(p, arcname=rel)
        print(f"  + {rel} ({p.stat().st_size / 2**20:.0f}MB)", flush=True)
print(f"XONG -> {OUT} ({OUT.stat().st_size / 2**20:.0f}MB)")
