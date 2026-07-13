#!/usr/bin/env python3
"""Bước 4b — tokenize the clean pairs into decoder-only seq2seq training
sequences and pack to a compact binary (loaded fully into RAM at train time).

Each pair -> 2 directional examples (bidirectional model, one net):
  VI->JA: [BOS] >>jpn<< <vi> [EOS] <ja> [EOS]
  JA->VI: [BOS] >>vie<< <ja> [EOS] <vi> [EOS]
Loss is computed only on the target segment (tokens from tgt_start onward),
so `tgt_start` (index of the first target token) is stored per sequence.

Sequences longer than MAX_SEQ are skipped (p99 pair length is ~147 tokens, so
this drops very few). Output per split in data/bin/:
  <split>.tokens.u16   flat uint16 token stream
  <split>.index.npy    int32 [N,2] = (length, tgt_start) per sequence
"""
import array
from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).parent.parent
CLEAN = ROOT / "data" / "clean"
OUTDIR = ROOT / "data" / "bin"
MAX_SEQ = 256
ENC_BATCH = 20000

sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
BOS, EOS = sp.bos_id(), sp.eos_id()
VIE, JPN = sp.piece_to_id(">>vie<<"), sp.piece_to_id(">>jpn<<")


def build_split(split):
    """Streaming/chunked: encode CHUNK pairs at a time, append their token
    bytes to the output file, keep only the small (length, tgt_start) index in
    RAM. Avoids holding all ~10.8M tokenized sentences at once (that OOM-killed
    the earlier all-at-once version)."""
    ja_lines = (CLEAN / f"{split}.ja").read_text(encoding="utf-8").splitlines()
    vi_lines = (CLEAN / f"{split}.vi").read_text(encoding="utf-8").splitlines()
    assert len(ja_lines) == len(vi_lines)
    n_pairs = len(ja_lines)
    print(f"[{split}] {n_pairs:,} pairs, streaming tokenize...", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    index = array.array("i")   # flattened (length, tgt_start), ~86MB for train
    n_skip = 0
    n_tok = 0
    CHUNK = 100000

    with open(OUTDIR / f"{split}.tokens.u16", "wb") as fout:
        for s in range(0, n_pairs, CHUNK):
            ja_ids = sp.encode(ja_lines[s:s + CHUNK])
            vi_ids = sp.encode(vi_lines[s:s + CHUNK])
            buf = array.array("H")
            for j, v in zip(ja_ids, vi_ids):
                for tag, src, tgt in ((JPN, v, j), (VIE, j, v)):   # VI->JA, JA->VI
                    seq = [BOS, tag] + src + [EOS] + tgt + [EOS]
                    if len(seq) > MAX_SEQ:
                        n_skip += 1
                        continue
                    buf.extend(seq)
                    index.append(len(seq))
                    index.append(len(src) + 3)   # first target token index
            buf.tofile(fout)
            n_tok += len(buf)
            print(f"  [{split}] {min(s+CHUNK, n_pairs):,}/{n_pairs:,} pairs, "
                  f"{n_tok:,} tokens", flush=True)

    idx = np.frombuffer(index, dtype=np.int32).reshape(-1, 2)
    np.save(OUTDIR / f"{split}.index.npy", idx)
    print(f"[{split}] wrote {idx.shape[0]:,} sequences ({n_tok:,} tokens), "
          f"skipped {n_skip:,} over {MAX_SEQ}. mean len {n_tok/max(idx.shape[0],1):.1f}", flush=True)


def main():
    print(f"specials: BOS={BOS} EOS={EOS} VIE={VIE} JPN={JPN}", flush=True)
    for split in ("dev", "train"):
        build_split(split)
    print("done -> data/bin/", flush=True)


if __name__ == "__main__":
    main()
