#!/usr/bin/env python3
"""Bước 6 (part 1) — pack a trained checkpoint into a compact 1.58-bit file.

bitnet.cpp only runs its own exact BitNet arch/GGUF; our SubLN-in-BitLinear
model doesn't map onto it cleanly, so we ship a self-contained packed format +
a CPU runtime (cpu_translate.py) that realizes the size benefit CLAUDE.md §1
asks for (~25-50MB, CPU). Packing:
  - BitLinear weights -> ternary {-1,0,1} packed 4-per-byte + one fp16 scale
    (= mean|W|, the b1.58 dequant scale) per matrix.
  - embeddings (tied to lm_head) -> int8 per-row + fp16 row scale.
  - RMSNorm / final-norm weights -> fp16.
Reports the packed size. Output: dist/model_1p58.npz + dist/config.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from bitnet import BitNetLM, BitNetConfig

ROOT = Path(__file__).parent.parent
DIST = ROOT / "dist"


def pack_ternary(w):
    """w: [out,in] float tensor -> (uint8 packed 4-per-byte, fp16 scale)."""
    wf = w.float()
    scale = wf.abs().mean().clamp_(min=1e-5)
    q = (wf / scale).round().clamp_(-1, 1).to(torch.int8)  # {-1,0,1}
    codes = (q + 1).to(torch.uint8).reshape(-1)             # {0,1,2}
    pad = (-codes.numel()) % 4
    if pad:
        codes = torch.cat([codes, torch.zeros(pad, dtype=torch.uint8)])
    codes = codes.reshape(-1, 4)
    packed = (codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6))
    return packed.numpy(), np.float16(scale.item())


def pack_embed_int8(w):
    wf = w.float()
    scale = wf.abs().amax(dim=1, keepdim=True).clamp_(min=1e-5) / 127.0
    q = (wf / scale).round().clamp_(-127, 127).to(torch.int8)
    return q.numpy(), scale.squeeze(1).to(torch.float16).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "last.pt"))
    args = ap.parse_args()
    DIST.mkdir(exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = BitNetConfig(**ck["cfg"]) if "cfg" in ck else BitNetConfig()
    model = BitNetLM(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    blob = {}
    for name, mod in model.named_modules():
        from bitnet import BitLinear
        if isinstance(mod, BitLinear):
            packed, scale = pack_ternary(mod.weight.data)
            blob[name + ".w_packed"] = packed
            blob[name + ".w_scale"] = scale
            blob[name + ".norm"] = mod.norm.weight.data.to(torch.float16).numpy()
    q, s = pack_embed_int8(model.embed.weight.data)
    blob["embed.q"] = q
    blob["embed.scale"] = s
    blob["final_norm"] = model.norm.weight.data.to(torch.float16).numpy()

    np.savez(DIST / "model_1p58.npz", **blob)
    (DIST / "config.json").write_text(json.dumps({
        "vocab_size": cfg.vocab_size, "d_model": cfg.d_model, "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads, "d_ff": cfg.d_ff, "max_seq": cfg.max_seq,
        "step": ck.get("step", -1),
    }, indent=2))

    size = (DIST / "model_1p58.npz").stat().st_size
    print(f"packed step {ck.get('step',-1)} -> dist/model_1p58.npz  ({size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
