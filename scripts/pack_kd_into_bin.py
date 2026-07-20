#!/usr/bin/env python3
"""Tokenize corpus KD (JSONL {ja,vi,...} đã qua QC) theo ĐÚNG format binary của
premix (BOS + >>vie<< + ja_ids + EOS + vi_ids + EOS, khớp mix_and_binarize.py) rồi
NỐI vào 1 bộ train.tokens.u16/train.index.npy đã có (vd bản ja->vi-only từ
filter_bin_ja2vi.py) — dùng cho PLAN_KD_JA2VI §5 (train 100M chỉ ja->vi).

  python scripts/pack_kd_into_bin.py <kd.jsonl> <bin_in_dir> <bin_out_dir> [--oversample N]

--oversample N: lặp KD N lần khi nối (mặc định 1) — KD nhỏ so base (thường <1% khối
lượng), cần oversample để có trọng số đáng kể trong mix (tương tự new-frac ở
mix_and_binarize.py, nhưng đơn giản hoá vì premix ở đây chỉ 1 chiều/1 lớp).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).parent.parent
MAX_SEQ = 256

sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
BOS, EOS = sp.bos_id(), sp.eos_id()
VIE = sp.piece_to_id(">>vie<<")


def build_seq(ja, vi):
    ja_ids = sp.encode(ja.strip())
    vi_ids = sp.encode(vi.strip())
    seq = [BOS, VIE] + ja_ids + [EOS] + vi_ids + [EOS]
    if len(seq) > MAX_SEQ:
        return None
    tgt_start = 2 + len(ja_ids) + 1
    return seq, tgt_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kd_jsonl")
    ap.add_argument("bin_in_dir")
    ap.add_argument("bin_out_dir")
    ap.add_argument("--split", default="train")
    ap.add_argument("--oversample", type=int, default=1)
    args = ap.parse_args()

    IN, OUT = Path(args.bin_in_dir), Path(args.bin_out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    pairs = [json.loads(l) for l in Path(args.kd_jsonl).open(encoding="utf-8") if l.strip()]
    print(f"KD: {len(pairs):,} cặp, oversample x{args.oversample}", flush=True)

    new_seqs, new_idx = [], []
    too_long = 0
    for p in pairs:
        r = build_seq(p["ja"], p["vi"])
        if r is None:
            too_long += 1
            continue
        seq, tgt_start = r
        new_seqs.append(np.array(seq, dtype=np.uint16))
        new_idx.append((len(seq), tgt_start))
    if too_long:
        print(f"  bỏ {too_long} cặp dài > {MAX_SEQ} token", flush=True)

    new_toks_one = np.concatenate(new_seqs) if new_seqs else np.array([], dtype=np.uint16)
    new_idx_one = np.array(new_idx, dtype=np.int64)
    print(f"  KD sau tokenize: {len(new_idx_one):,} seq, {len(new_toks_one):,} token/lần", flush=True)

    base_toks = np.fromfile(IN / f"{args.split}.tokens.u16", dtype=np.uint16)
    base_idx = np.load(IN / f"{args.split}.index.npy")
    print(f"  base: {len(base_idx):,} seq, {len(base_toks):,} token", flush=True)

    all_toks = [base_toks] + [new_toks_one] * args.oversample
    all_idx = [base_idx] + [new_idx_one] * args.oversample
    out_toks = np.concatenate(all_toks)
    out_idx = np.concatenate(all_idx).astype(base_idx.dtype)

    out_toks.tofile(OUT / f"{args.split}.tokens.u16")
    np.save(OUT / f"{args.split}.index.npy", out_idx)
    n_kd_total = len(new_idx_one) * args.oversample
    print(f"GHI {args.split}: {len(out_idx):,} seq ({n_kd_total:,} KD, "
          f"{100*n_kd_total/len(out_idx):.2f}%), {len(out_toks):,} token -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
