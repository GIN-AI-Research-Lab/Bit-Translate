#!/usr/bin/env python3
"""Binarize CHỈ chiều ja->vi (pilot 100M) — mỗi cặp 1 sequence:
     [BOS] >>vie<< <ja> [EOS] <vi> [EOS]
Loss chỉ trên đoạn target (vi) từ tgt_start. Khác binarize.py (2 chiều) — bản này
1 chiều nên số seq = số cặp (không nhân đôi).

Đầu vào : data/clean/{split}.ja + {split}.vi (thẳng hàng)
Đầu ra  : data/bin/{split}.tokens.u16 (uint16) + {split}.index.npy (int32 [N,2])

Đổi thư mục (dùng cho curriculum pha 2 — tập nhỏ chất lượng cao):
  python scripts/binarize_ja2vi.py --clean data/clean_p2 --out data/bin_p2
"""
import argparse
import array
from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).parent.parent
CLEAN = ROOT / "data" / "clean"
OUTDIR = ROOT / "data" / "bin"
MAX_SEQ = 256
CHUNK = 100000

sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
BOS, EOS = sp.bos_id(), sp.eos_id()
VIE = sp.piece_to_id(">>vie<<")


def build_split(split):
    ja_lines = (CLEAN / f"{split}.ja").read_text(encoding="utf-8").splitlines()
    vi_lines = (CLEAN / f"{split}.vi").read_text(encoding="utf-8").splitlines()
    assert len(ja_lines) == len(vi_lines), "ja/vi lệch số dòng"
    n = len(ja_lines)
    print(f"[{split}] {n:,} cặp, ja->vi only, streaming tokenize...", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    index = array.array("i")
    n_skip = n_tok = 0
    with open(OUTDIR / f"{split}.tokens.u16", "wb") as fout:
        for s in range(0, n, CHUNK):
            ja_ids = sp.encode(ja_lines[s:s + CHUNK])
            vi_ids = sp.encode(vi_lines[s:s + CHUNK])
            buf = array.array("H")
            for j, v in zip(ja_ids, vi_ids):
                seq = [BOS, VIE] + j + [EOS] + v + [EOS]   # ja->vi
                if len(seq) > MAX_SEQ:
                    n_skip += 1
                    continue
                buf.extend(seq)
                index.append(len(seq))
                index.append(len(j) + 3)   # tgt_start = BOS,VIE,...ja...,EOS -> vi bắt đầu
            buf.tofile(fout)
            n_tok += len(buf)
            print(f"  [{split}] {min(s+CHUNK, n):,}/{n:,} | {n_tok:,} tokens", flush=True)

    idx = np.frombuffer(index, dtype=np.int32).reshape(-1, 2)
    np.save(OUTDIR / f"{split}.index.npy", idx)
    print(f"[{split}] GHI {idx.shape[0]:,} seq ({n_tok:,} token), bỏ {n_skip:,} câu >{MAX_SEQ}. "
          f"mean len {n_tok/max(idx.shape[0],1):.1f}", flush=True)
    return idx.shape[0], n_tok


def main():
    global CLEAN, OUTDIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(CLEAN), help="thư mục chứa {split}.ja/.vi")
    ap.add_argument("--out", default=str(OUTDIR), help="thư mục ghi bin")
    ap.add_argument("--splits", default="dev,train")
    a = ap.parse_args()
    CLEAN, OUTDIR = Path(a.clean), Path(a.out)
    print(f"specials: BOS={BOS} EOS={EOS} VIE={VIE} | {CLEAN} -> {OUTDIR}", flush=True)
    tot_seq = tot_tok = 0
    for split in a.splits.split(","):
        s, t = build_split(split)
        tot_seq += s
        tot_tok += t
    print(f"\ndone -> {OUTDIR}/ | tổng: {tot_seq:,} seq, {tot_tok:,} token")


if __name__ == "__main__":
    main()
