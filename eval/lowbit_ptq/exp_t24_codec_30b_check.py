# -*- coding: utf-8 -*-
"""Verify TQ33 codec (group-64 scale extraction) van lossless tren ckpt 30B
du du train o g256 (khac 0.6B train o g64) - kiem tra thuc nghiem truoc khi
encode hang loat, vi ly thuyet noi finer-subgroup-cua-constant la an toan
nhung PHAI do that (floating point co the bat ngo)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "e:/Bit-Translate/eval/lowbit_ptq")
from exp_t24_codec import analyze_and_codec
from safe_ckpt_reader import open_reader  # né bug torch.load(mmap=True) tren file 61GB nay

CKPT = "D:/Bit-Translate-data/pack_local/pytorch_model.bin"

def main():
    prefix, tensors_meta, extra, get, zf = open_reader(CKPT)
    print("meta.geo6:", extra.get("meta.geo6"), "| meta.tag:", extra.get("meta.tag"))
    keys = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.24.mlp.experts.63.down_proj.weight",
        "model.layers.47.mlp.experts.127.down_proj.weight",
        "model.layers.47.self_attn.o_proj.weight",
    ]
    all_ok = True
    for k in keys:
        if k not in tensors_meta:
            print(f"  [thieu key] {k}")
            continue
        W = get(k).float()
        r = analyze_and_codec(W)
        ok = r["lossless"] and r["viol"] == 0
        all_ok &= ok
        print(f"  {k:55s} shape={tuple(W.shape)} split={r['exact_split']} "
              f"lossless={r['lossless']} viol={r['viol']} nz-hist={r['hist']}")
    print("\nPHAN QUYET (30B, g256-native, extract g64):",
          "LOSSLESS TOAN BO - an toan encode hang loat" if all_ok else "CO VI PHAM - can dieu tra")
    zf.close()

if __name__ == "__main__":
    main()
