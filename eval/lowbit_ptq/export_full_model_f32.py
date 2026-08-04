# -*- coding: utf-8 -*-
r"""TQ33 Runner — Giai đoạn 1, bước 1: dump TOÀN BỘ tensor của ckpt bake gen4_n4.pt
sang 1 file binary float32 (little-endian) phẳng + meta.json mô tả tên/shape/offset,
để qwen3_runner.c đọc thẳng bằng fread không cần parser phức tạp.

Phát hiện quan trọng (đã verify bằng inspect_ckpt.py + inspect_ckpt2.py, xem
RESEARCH_TQ33_RUNNER.md mục "Sai lệch so với giả định ban đầu"):
  - Bias KHÔNG phải 0 — mọi tensor .bias (196 cái, cả 7 loại linear x 28 layer) đều có
    giá trị thật (absmax ~0.005-0.038), PHẢI cộng vào. (Giả định --no-bias trong
    prep_hf_dir_from_ckpt.py SAI cho ckpt này — bias thật KHÔNG bị bake=0.)
  - "lm_head.weight" TỒN TẠI riêng trong ckpt (không giống giả định ban đầu là không có
    do tie_word_embeddings=true) NHƯNG giá trị KHÁC model.embed_tokens.weight (max abs
    diff 0.18) dù cả 2 đều = bản gốc Qwen3-0.6B pristine lúc from_pretrained (cũng khác
    nhau 0 lúc đó) — tie bị "vỡ" đâu đó trong pipeline (nghi do .to(dtype=...) tách
    Parameter). Test PPL thật trên dev.vi/dev.ja (resolve_lmhead.py) cho thấy dùng
    model.embed_tokens.weight làm output head cho PPL thấp hơn (296.7/575.8) so với
    dùng lm_head.weight (301.3/588.5) — VÀ khớp quy ước HF tie_weights() (tie OUTPUT
    theo INPUT, không phải ngược lại). => runner dùng model.embed_tokens.weight cho
    CẢ input lookup lẫn output logits. lm_head.weight vẫn được dump đầy đủ (để không bỏ
    sót tensor nào) nhưng KHÔNG dùng trong forward pass mặc định.
  - mlp.{gate,up,down}_proj CÓ bias dù kiến trúc Qwen3 gốc (HF) không có bias ở MLP —
    đây là bias riêng do LearnQLinear QAT thêm vào (deviation khỏi kiến trúc chuẩn),
    PHẢI cộng như linear thường.

Output: D:\Bit-Translate-data\tq33_runner\weights_f32\weights.bin (~3GB) + meta.json
        + meta_index.txt (format đơn giản cho C: 1 dòng/tensor "name ndim d0 d1 offset_f32 numel")
"""
import json
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

CKPT = r"D:\Bit-Translate-data\qat_ckpts\qat_gen4_n4.pt"
OUT_DIR = r"D:\Bit-Translate-data\tq33_runner\weights_f32"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"loading {CKPT} (mmap) ...")
    sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
    state = sd["state_dict"] if "state_dict" in sd else sd
    meta_ckpt = sd.get("meta", {})
    print(f"ckpt meta: {meta_ckpt}")

    keys = sorted(state.keys())
    print(f"tổng {len(keys)} tensor, dump toàn bộ ra float32 phẳng...")

    bin_path = os.path.join(OUT_DIR, "weights.bin")
    index = []
    offset_elems = 0
    n_bytes_total = 0
    with open(bin_path, "wb") as f:
        for k in keys:
            t = state[k]
            if not torch.is_tensor(t):
                print(f"  bỏ qua (không phải tensor): {k}")
                continue
            arr = t.detach().float().contiguous().numpy().astype("<f4")
            f.write(arr.tobytes())
            numel = arr.size
            index.append({
                "name": k,
                "shape": list(t.shape),
                "offset_f32": offset_elems,
                "numel": numel,
            })
            offset_elems += numel
            n_bytes_total += arr.nbytes

    print(f"đã ghi {n_bytes_total/1e9:.3f} GB -> {bin_path}")

    meta = {
        "ckpt": CKPT,
        "ckpt_meta": meta_ckpt,
        "n_tensors": len(index),
        "total_f32_elems": offset_elems,
        "total_bytes": n_bytes_total,
        "config": {
            "n_layer": 28, "hidden_size": 1024, "n_head": 16, "n_kv_head": 8,
            "head_dim": 128, "intermediate_size": 3072, "vocab_size": 151936,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings_config": True,
        },
        "output_head_tensor": "model.embed_tokens.weight",
        "output_head_note": ("lm_head.weight ton tai rieng trong ckpt nhung gia tri KHAC "
                              "model.embed_tokens.weight (da bi 'vo tie'); da verify bang PPL "
                              "that tren dev.vi/dev.ja - dung embed_tokens cho PPL thap hon va "
                              "khop quy uoc HF tie_weights(). Xem RESEARCH_TQ33_RUNNER.md."),
        "tensors": index,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    # index đơn giản cho C: name ndim dims... offset_f32 numel
    with open(os.path.join(OUT_DIR, "meta_index.txt"), "w", encoding="utf-8") as f:
        f.write(f"{len(index)}\n")
        for e in index:
            dims = " ".join(str(d) for d in e["shape"])
            f.write(f"{e['name']} {len(e['shape'])} {dims} {e['offset_f32']} {e['numel']}\n")

    print(f"meta.json + meta_index.txt ghi xong ({len(index)} tensor).")
    print(f"tổng elements={offset_elems}  ({offset_elems*4/1e9:.3f} GB float32)")


if __name__ == "__main__":
    main()
