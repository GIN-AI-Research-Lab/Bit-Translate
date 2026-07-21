#!/usr/bin/env python3
"""Tokenize corpus >>fix<< (JSONL {ja, vi_wrong, vi_correct}) rồi NỐI vào 1 bộ
train.tokens.u16/train.index.npy đã có — dùng cho v2 multi-task (PLAN_KD_JA2VI
§5): model vừa dịch thuần vừa tự sửa lỗi, CÙNG 1 bộ trọng số.

Format sequence (tag mới >>fix<< = id do scripts/add_fix_token.py cấp, KHÔNG
phải >>vie<</>>jpn<<):
  [BOS, FIX_TAG] + ja_ids + [EOS] + vi_wrong_ids + [EOS] + vi_correct_ids + [EOS]
  "nguồn" = câu Nhật + bản nháp sai (ngăn nhau bởi EOS, khớp quy ước
  src+EOS+tgt+EOS của mix_and_binarize.py) ; "đích" (tính loss) = bản đã sửa.

  python scripts/pack_fix_into_bin.py <fix.jsonl> <bin_in_dir> <bin_out_dir> [--oversample N]

QUAN TRỌNG: tokenizer PHẢI đã chạy scripts/add_fix_token.py (vocab 32001) và
bin_in_dir phải là premix ĐÃ TOKENIZE bằng tokenizer gốc 32000 — token ID cũ
không đổi nên nối trực tiếp an toàn (xem scripts/add_fix_token.py).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).parent.parent
MAX_SEQ = 320  # fix task ghép JA+vi_wrong+vi_correct, dài hơn dịch thuần (256)

sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
BOS, EOS = sp.bos_id(), sp.eos_id()
FIX = sp.piece_to_id(">>fix<<")
assert FIX != sp.unk_id(), "tokenizer chưa có >>fix<< — chạy scripts/add_fix_token.py trước"


def build_seq(ja, vi_wrong, vi_correct):
    ja_ids = sp.encode(ja.strip())
    wrong_ids = sp.encode(vi_wrong.strip())
    correct_ids = sp.encode(vi_correct.strip())
    seq = [BOS, FIX] + ja_ids + [EOS] + wrong_ids + [EOS] + correct_ids + [EOS]
    if len(seq) > MAX_SEQ:
        return None
    tgt_start = 2 + len(ja_ids) + 1 + len(wrong_ids) + 1
    return seq, tgt_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fix_jsonl")
    ap.add_argument("bin_in_dir")
    ap.add_argument("bin_out_dir")
    ap.add_argument("--split", default="train")
    ap.add_argument("--oversample", type=int, default=1)
    args = ap.parse_args()

    IN, OUT = Path(args.bin_in_dir), Path(args.bin_out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    pairs = [json.loads(l) for l in Path(args.fix_jsonl).open(encoding="utf-8") if l.strip()]
    print(f"fix: {len(pairs):,} cặp, oversample x{args.oversample}", flush=True)

    new_seqs, new_idx, too_long = [], [], 0
    for p in pairs:
        r = build_seq(p["ja"], p["vi_wrong"], p["vi_correct"])
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
    print(f"  fix sau tokenize: {len(new_idx_one):,} seq, {len(new_toks_one):,} token/lần", flush=True)

    base_toks = np.fromfile(IN / f"{args.split}.tokens.u16", dtype=np.uint16)
    base_idx = np.load(IN / f"{args.split}.index.npy")
    print(f"  base: {len(base_idx):,} seq, {len(base_toks):,} token", flush=True)

    all_toks = [base_toks] + [new_toks_one] * args.oversample
    all_idx = [base_idx] + [new_idx_one] * args.oversample
    out_toks = np.concatenate(all_toks)
    out_idx = np.concatenate(all_idx).astype(base_idx.dtype)

    out_toks.tofile(OUT / f"{args.split}.tokens.u16")
    np.save(OUT / f"{args.split}.index.npy", out_idx)
    n_fix_total = len(new_idx_one) * args.oversample
    print(f"GHI {args.split}: {len(out_idx):,} seq ({n_fix_total:,} fix, "
          f"{100*n_fix_total/len(out_idx):.2f}%), {len(out_toks):,} token -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
