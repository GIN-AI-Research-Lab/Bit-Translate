#!/usr/bin/env python3
"""Bước 6 (part 2) — CPU inference from the packed 1.58-bit file.

Loads dist/model_1p58.npz, rebuilds the BitNetLM on CPU with the dequantized
(ternary) weights, and translates. Forward re-applies the same fake-quant as
training (RMSNorm -> 8-bit activation quant -> ternary matmul), so CPU outputs
match the trained model. Realizes CLAUDE.md §6: CPU, threads set to physical
cores, re-translation strategy is trivial (just call again).

Usage:
  cpu_translate.py --to jpn --text "Xin chào, bạn khỏe không?"
  cpu_translate.py --to vie --text "こんにちは、元気ですか？"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from bitnet import BitNetLM, BitNetConfig, BitLinear
import sentencepiece as spm

ROOT = Path(__file__).parent.parent
DIST = ROOT / "dist"


def unpack_ternary(packed, scale, shape):
    codes = np.empty(packed.size * 4, dtype=np.uint8)
    codes[0::4] = packed & 0b11
    codes[1::4] = (packed >> 2) & 0b11
    codes[2::4] = (packed >> 4) & 0b11
    codes[3::4] = (packed >> 6) & 0b11
    n = shape[0] * shape[1]
    w = (codes[:n].astype(np.float32) - 1.0) * float(scale)
    return torch.from_numpy(w.reshape(shape))


def load_cpu_model():
    cfg_d = json.loads((DIST / "config.json").read_text())
    cfg = BitNetConfig(vocab_size=cfg_d["vocab_size"], d_model=cfg_d["d_model"],
                       n_layers=cfg_d["n_layers"], n_heads=cfg_d["n_heads"],
                       d_ff=cfg_d["d_ff"], max_seq=cfg_d["max_seq"])
    model = BitNetLM(cfg).eval()
    blob = np.load(DIST / "model_1p58.npz")
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, BitLinear):
                w = unpack_ternary(blob[name + ".w_packed"], blob[name + ".w_scale"],
                                   tuple(mod.weight.shape))
                mod.weight.data.copy_(w)
                mod.norm.weight.data.copy_(torch.from_numpy(blob[name + ".norm"].astype(np.float32)))
        emb = torch.from_numpy(blob["embed.q"].astype(np.float32)) * \
              torch.from_numpy(blob["embed.scale"].astype(np.float32))[:, None]
        model.embed.weight.data.copy_(emb)
        model.norm.weight.data.copy_(torch.from_numpy(blob["final_norm"].astype(np.float32)))
    return model, cfg, cfg_d.get("step", -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", choices=["jpn", "vie"], required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--threads", type=int, default=6)  # physical cores (Ryzen 5 5600X)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--rep-penalty", type=float, default=1.0,
                    help=">1 (e.g. 1.3) giảm lặp từ; 1.0 = tắt")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
    model, cfg, step = load_cpu_model()
    model.freeze_for_inference()   # precompute ternary weights -> ~3x faster decode

    bos, eos = sp.bos_id(), sp.eos_id()
    tag = sp.piece_to_id(f">>{args.to}<<")
    ids = torch.tensor([[bos, tag] + sp.encode(args.text) + [eos]])
    with torch.inference_mode():   # KV-cache decode (see BitNetLM.generate_cached)
        out = model.generate_cached(ids, max_new_tokens=args.max_new,
                                    eos_id=eos, rep_penalty=args.rep_penalty)
    gen = out[0, ids.shape[1]:].tolist()
    if eos in gen:
        gen = gen[:gen.index(eos)]
    print(sp.decode(gen))


if __name__ == "__main__":
    main()
