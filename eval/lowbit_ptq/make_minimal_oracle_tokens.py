# -*- coding: utf-8 -*-
"""Tạo tokens.bin tối thiểu (chỉ prompt, KHÔNG cần hidden_final.npy/logits.npy) cho
qwen3_runner_tq33_fast.exe chạy chế độ `nocompare` (chỉ đo tốc độ, không so oracle — vì
oracle cũ thuộc model gen4_n4 khác, không áp dụng cho checkpoint PTQ mới này)."""
import glob
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

MODEL_DIR = glob.glob(r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*")[0]
OUT_DIR = r"D:\Bit-Translate-data\tq33_runner_ptq24\oracle"
PROMPT = "Xin chào, hôm nay bạn có khỏe không? Tôi muốn hỏi về thời tiết Hà Nội."


def main():
    from transformers import AutoTokenizer
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tok(PROMPT, return_tensors="pt").input_ids[0].numpy().astype(np.int32)
    print(f"prompt: {PROMPT!r} -> {len(ids)} token: {ids.tolist()}")
    with open(os.path.join(OUT_DIR, "tokens.bin"), "wb") as f:
        np.array([len(ids)], dtype=np.int32).tofile(f)
        ids.tofile(f)
    print(f"đã ghi {os.path.join(OUT_DIR, 'tokens.bin')}")


if __name__ == "__main__":
    main()
