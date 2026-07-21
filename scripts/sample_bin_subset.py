#!/usr/bin/env python3
"""Rút ngẫu nhiên K sequence từ một bộ binary (train.tokens.u16 + train.index.npy)
thành bộ mới nhỏ hơn — dùng làm "replay slice" cho clean-finetune (PLAN_KD §5):
giai đoạn finetune thuần KD cần một lát base nhỏ trộn kèm để chống catastrophic
forgetting, không cần (và không nên) vác cả 11,5M câu base theo.

  python scripts/sample_bin_subset.py <bin_in_dir> <bin_out_dir> <k> [--seed 20260721]
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin_in_dir")
    ap.add_argument("bin_out_dir")
    ap.add_argument("k", type=int)
    ap.add_argument("--seed", type=int, default=20260721)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    IN, OUT = Path(args.bin_in_dir), Path(args.bin_out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    toks = np.fromfile(IN / f"{args.split}.tokens.u16", dtype=np.uint16)
    idx = np.load(IN / f"{args.split}.index.npy")          # (N, 2): [len, tgt_start]
    offsets = np.concatenate([[0], np.cumsum(idx[:, 0])])
    print(f"nguồn: {len(idx):,} seq, {len(toks):,} token", flush=True)

    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(len(idx), size=args.k, replace=False))

    out_seqs = [toks[offsets[i]:offsets[i + 1]] for i in pick]
    out_toks = np.concatenate(out_seqs)
    out_idx = idx[pick]

    out_toks.tofile(OUT / f"{args.split}.tokens.u16")
    np.save(OUT / f"{args.split}.index.npy", out_idx)
    print(f"GHI: {len(out_idx):,} seq, {len(out_toks):,} token -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
