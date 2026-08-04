# -*- coding: utf-8 -*-
"""Dựng thư mục HF từ ckpt bake của lab để convert_hf_to_gguf đọc thẳng:
- lọc state_dict về đúng schema HF (bỏ bias tự thêm, bỏ param phụ nếu có)
- lưu SHARDED model-NNNNN-of-MMMMM.safetensors + model.safetensors.index.json
  — máy chỉ ~47GB RAM, ckpt 30B ~60GB; đọc mmap=True (giống conversion/base.py dùng)
  nhưng ghi từng shard rồi giải phóng ngay, đỉnh RAM ~ 1 shard (mặc định 6GB) thay vì cả file.
  DÙNG SAFETENSORS chứ không phải .bin: đã kiểm chứng .bin-sharded + mmap qua đường
  conversion/base.py (self.lazy, dùng chung storage mmap nhiều part) SINH DỮ LIỆU SAI
  (87.9% byte lệch trên tensor thử — mmap của part cũ bị mất hiệu lực khi sang part sau).
  Safetensors là đường được test nhiều nhất trong llama.cpp (gguf.utility.SafetensorsLocal),
  mmap an toàn theo spec — đã kiểm chứng lossless 5/5 shard trên 0.6B trước khi dùng cho 30B.
- copy config/tokenizer từ HF hub (file nhỏ, không kéo safetensors gốc)
Dùng: python prep_hf_dir_from_ckpt.py --ckpt <pt> --model-id Qwen/Qwen3-30B-A3B --out <dir>
"""
import argparse
import gc
import json
import os
import shutil
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-gb", type=float, default=6.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # metadata nhỏ từ hub (config/tokenizer/generation) — TUYỆT ĐỐI không đụng weight file.
    # allow_patterns chỉ chặn TẢI MỚI; cache HF có thể đã có sẵn model.safetensors từ lần
    # tải khác trước đó trong session — nếu copy nguyên os.listdir(mdir) thì file weight gốc
    # lẫn vào thư mục out, và get_model_part_names() của convert script khớp CẢ "model.safetensors"
    # gốc lẫn shard của ta (cùng prefix "model"+suffix ".safetensors") -> nạp NHẦM weight FP16
    # gốc thay vì bake ternary của ta (đã bắt lỗi này bằng tay 1 lần — giờ lọc cứng cho chắc).
    mdir = snapshot_download(args.model_id,
                             allow_patterns=["config.json", "generation_config.json",
                                             "tokenizer*", "vocab*", "merges*",
                                             "special_tokens_map.json"])
    META_PATTERNS = ("config.json", "generation_config.json", "tokenizer", "vocab",
                     "merges", "special_tokens_map.json")
    for f in os.listdir(mdir):
        src = os.path.join(mdir, f)
        if not os.path.isfile(src):
            continue
        if f.endswith((".safetensors", ".bin", ".index.json", ".pt", ".gguf")):
            continue
        if not any(f.startswith(p) or f == p for p in META_PATTERNS):
            continue
        shutil.copy(src, os.path.join(args.out, f))
        print("meta:", f)

    # mmap=True: ckpt 60GB không cần load hết vào RAM (máy chỉ ~47GB) — tensor là view
    # lười trên file, đây là CÙNG kỹ thuật conversion/base.py dùng để đọc part sau này.
    print(f"mmap-load {args.ckpt} ...")
    sd = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)
    state = sd["state_dict"] if "state_dict" in sd else sd
    keys, dropped = [], []
    for k, v in state.items():
        if not torch.is_tensor(v):
            dropped.append(k)
            continue
        # schema HF qwen3(moe): không có bias ở linear; bias bake=0 (--no-bias) nên bỏ an toàn
        if k.endswith(".bias"):
            dropped.append(k)
            continue
        if any(t in k for t in (".log_s", ".u_vec", ".v_vec", "lora_")):
            dropped.append(k)
            continue
        keys.append(k)
    print(f"giữ {len(keys)} tensor, bỏ {len(dropped)} (bias/phụ):",
          dropped[:5], "..." if len(dropped) > 5 else "")

    # gom shard theo ngưỡng byte rồi ghi + giải phóng ngay (đỉnh RAM ~ 1 shard)
    budget = int(args.shard_gb * 1e9)
    shards, cur, cur_bytes = [], [], 0
    for k in keys:
        nbytes = state[k].numel() * state[k].element_size()
        if cur and cur_bytes + nbytes > budget:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(k)
        cur_bytes += nbytes
    if cur:
        shards.append(cur)

    n = len(shards)
    weight_map, total_size = {}, 0
    for i, shard_keys in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{n:05d}.safetensors"
        # safetensors đòi tensor contiguous + không alias storage nhau trong CÙNG file
        shard = {k: state[k].contiguous() for k in shard_keys}
        shard_bytes = sum(t.numel() * t.element_size() for t in shard.values())
        save_file(shard, os.path.join(args.out, name), metadata={"format": "pt"})
        for k in shard_keys:
            weight_map[k] = name
        total_size += shard_bytes
        print(f"  shard {i}/{n}: {len(shard_keys)} tensor, {shard_bytes/1e9:.2f} GB -> {name}")
        del shard
        gc.collect()

    with open(os.path.join(args.out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f)
    print(f"xong -> {args.out}  ({total_size/1e9:.1f} GB, {n} shard)")


if __name__ == "__main__":
    main()
