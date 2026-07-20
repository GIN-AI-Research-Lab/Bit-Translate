#!/usr/bin/env python3
"""Lọc premix ĐÃ BINARIZE (data/bin/train.tokens.u16 + train.index.npy) chỉ giữ
sequence chiều JA->VI, phục vụ train 100M chỉ ja->vi (PLAN_KD_JA2VI §5 — dùng lại
toàn bộ corpus lịch sử base+vòng1/2/3a thay vì chỉ data KD nhỏ).

Format xác nhận từ scripts/mix_and_binarize.py: mỗi sequence =
  [BOS, tag] + src_ids + [EOS] + tgt_ids + [EOS]
tag = VIE (piece ">>vie<<") -> JA->VI (giữ); tag = JPN (">>jpn<<") -> VI->JA (bỏ).
idx[i] = [length, tgt_start] ; offset[i] = cumsum(length) — TAG luôn ở vị trí 1
(offset+1) của mỗi sequence bất kể chiều.

  python scripts/filter_bin_ja2vi.py <in_dir> <out_dir> [split]

in_dir: thư mục chứa <split>.tokens.u16 + <split>.index.npy (vd sau khi giải nén
premix). out_dir: ghi <split>.tokens.u16 + <split>.index.npy đã lọc (chỉ ja->vi).
split mặc định "train" (chạy lại cho "dev"/"test" nếu premix có).
"""
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).parent.parent
IN_DIR, OUT_DIR = Path(sys.argv[1]), Path(sys.argv[2])
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "train"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
VIE = sp.piece_to_id(">>vie<<")
JPN = sp.piece_to_id(">>jpn<<")
assert VIE != sp.unk_id() and JPN != sp.unk_id(), "tokenizer thiếu token >>vie<</>>jpn<<"


def main():
    idx = np.load(IN_DIR / f"{SPLIT}.index.npy")            # [N,2] length, tgt_start
    toks = np.fromfile(IN_DIR / f"{SPLIT}.tokens.u16", dtype=np.uint16)
    offsets = np.zeros(len(idx) + 1, dtype=np.int64)
    np.cumsum(idx[:, 0], out=offsets[1:])

    tags = toks[offsets[:-1] + 1].astype(np.int64)           # vị trí 1 mỗi sequence
    n_vie = int((tags == VIE).sum())
    n_jpn = int((tags == JPN).sum())
    n_other = len(idx) - n_vie - n_jpn
    print(f"[{SPLIT}] tổng {len(idx):,} seq | VIE(ja->vi)={n_vie:,} "
          f"JPN(vi->ja)={n_jpn:,} khác={n_other:,}", flush=True)
    if n_other:
        print(f"  CẢNH BÁO: {n_other} sequence có tag lạ ở vị trí 1 — kiểm tra lại "
              f"format trước khi tin kết quả (có thể lẫn record 1 chiều khác chuẩn).",
              flush=True)

    keep = np.where(tags == VIE)[0]
    new_toks = np.empty(int(idx[keep, 0].sum()), dtype=np.uint16)
    new_idx = np.empty((len(keep), 2), dtype=idx.dtype)
    pos = 0
    for j, i in enumerate(keep):
        L = int(idx[i, 0])
        new_toks[pos:pos + L] = toks[offsets[i]:offsets[i] + L]
        new_idx[j] = idx[i]
        pos += L
        if (j + 1) % 2_000_000 == 0:
            print(f"  {j+1:,}/{len(keep):,}", flush=True)

    new_toks.tofile(OUT_DIR / f"{SPLIT}.tokens.u16")
    np.save(OUT_DIR / f"{SPLIT}.index.npy", new_idx)
    print(f"[{SPLIT}] GIỮ {len(keep):,}/{len(idx):,} seq ({100*len(keep)/len(idx):.1f}%) "
          f"| {pos:,} token -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
