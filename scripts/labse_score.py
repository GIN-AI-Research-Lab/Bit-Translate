#!/usr/bin/env python3
"""Chấm LaBSE tổng quát (tham số hoá path, dùng lại logic labse_score_vong3.py)
cho lớp lọc 4 (semantic) — bắt lỗi rule+mắt thường sót: đảo nghĩa, dịch nghĩa đen
sai, hallucination (câu vẫn trôi chảy nhưng lệch nghĩa gốc).

  python scripts/labse_score.py <in.jsonl> <out_keep.jsonl> <out_reject.jsonl> [--thr 0.80]

in.jsonl: mỗi dòng {"ja":..., "vi":..., ...}. Giữ nguyên các field khác, thêm "labse".
Env: LABSE_DEVICE=cpu|cuda (mặc định cpu), LABSE_THREADS=4.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEV = os.environ.get("LABSE_DEVICE", "cpu")
torch.set_num_threads(int(os.environ.get("LABSE_THREADS", "4")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out_keep")
    ap.add_argument("out_reject")
    ap.add_argument("--thr", type=float, default=0.80)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in Path(args.inp).open(encoding="utf-8") if l.strip()]
    print(f"LaBSE {len(pairs):,} cặp | device={DEV} thr={args.thr}", flush=True)
    model = SentenceTransformer("sentence-transformers/LaBSE", device=DEV)
    model.max_seq_length = 128
    B = 64
    scores = np.zeros(len(pairs))
    ja = [p["ja"] for p in pairs]
    vi = [p["vi"] for p in pairs]
    for i in range(0, len(pairs), B):
        j = min(i + B, len(pairs))
        eja = model.encode(ja[i:j], batch_size=B, convert_to_numpy=True, normalize_embeddings=True)
        evi = model.encode(vi[i:j], batch_size=B, convert_to_numpy=True, normalize_embeddings=True)
        scores[i:j] = np.sum(eja * evi, axis=1)
        if (i // B) % 40 == 0:
            print(f"  {j:,}/{len(pairs):,}", flush=True)

    for lo in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        print(f"  >= {lo}: {int((scores >= lo).sum()):,} ({100*(scores >= lo).mean():.1f}%)", flush=True)

    keep = scores >= args.thr
    with open(args.out_keep, "w", encoding="utf-8") as fk, open(args.out_reject, "w", encoding="utf-8") as fr:
        for p, s, k in zip(pairs, scores, keep):
            p["labse"] = round(float(s), 4)
            (fk if k else fr).write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"GIỮ {int(keep.sum()):,} | LOẠI {int((~keep).sum()):,} (thr={args.thr}) "
          f"-> {args.out_keep} / {args.out_reject}", flush=True)


if __name__ == "__main__":
    main()
