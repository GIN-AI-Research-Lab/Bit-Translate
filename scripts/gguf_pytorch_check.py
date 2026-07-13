#!/usr/bin/env python3
"""Nạp trọng số F16 từ GGUF ngược lại PyTorch BitNetLM và chạy generate().
Mục đích: phân định lỗi convert/runtime (llama.cpp) vs trọng số thật (train).
"""
import sys
from pathlib import Path
import numpy as np
import torch
from gguf import GGUFReader
import sentencepiece as spm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bitnet import BitNetLM, BitNetConfig

GGUF = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "dist" / "vija_f16.gguf")

sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
cfg = BitNetConfig(vocab_size=sp.get_piece_size())
model = BitNetLM(cfg).eval()

r = GGUFReader(GGUF)
g = {}
for t in r.tensors:
    arr = np.array(t.data, dtype=np.float32).reshape(tuple(reversed(t.shape)))
    g[t.name] = torch.from_numpy(arr)

def load(dst, name):
    w = g[name]
    assert tuple(w.shape) == tuple(dst.shape), f"{name}: {tuple(w.shape)} vs {tuple(dst.shape)}"
    dst.data.copy_(w)

sd = model.state_dict()
load(model.embed.weight, "token_embd.weight")
load(model.output_norm.weight, "output_norm.weight")
for i, blk in enumerate(model.blocks):
    p = f"blk.{i}."
    load(blk.attn.attn_norm.weight, p + "attn_norm.weight")
    load(blk.attn.wq.weight, p + "attn_q.weight")
    load(blk.attn.wk.weight, p + "attn_k.weight")
    load(blk.attn.wv.weight, p + "attn_v.weight")
    load(blk.attn.attn_sub_norm.weight, p + "attn_sub_norm.weight")
    load(blk.attn.wo.weight, p + "attn_output.weight")
    load(blk.ffn.ffn_norm.weight, p + "ffn_norm.weight")
    load(blk.ffn.gate.weight, p + "ffn_gate.weight")
    load(blk.ffn.up.weight, p + "ffn_up.weight")
    load(blk.ffn.ffn_sub_norm.weight, p + "ffn_sub_norm.weight")
    load(blk.ffn.down.weight, p + "ffn_down.weight")
print("nạp trọng số GGUF OK")

BOS, EOS = sp.bos_id(), sp.eos_id()
JPN, VIE = sp.piece_to_id(">>jpn<<"), sp.piece_to_id(">>vie<<")

def show_topk(prompt_ids, k=10):
    idx = torch.tensor([prompt_ids])
    logits, _ = model(idx)
    last = logits[0, -1]
    probs = torch.softmax(last.float(), -1)
    top = torch.topk(probs, k)
    print("  top-%d token kế tiếp:" % k)
    for p, t in zip(top.values.tolist(), top.indices.tolist()):
        print(f"    id={t:5d}  p={p:.3f}  '{sp.id_to_piece(t)}'")

tests = [(">>jpn<<", "Tôi thích mèo.", JPN), (">>vie<<", "猫が好きです。", VIE)]
for tag, src, tagid in tests:
    ids = [BOS, tagid] + sp.encode(src) + [EOS]
    print(f"\n=== {tag} {src}  (prompt ids={ids}) ===")
    show_topk(ids)
    out = model.generate(torch.tensor([ids]), max_new_tokens=40, eos_id=EOS, temperature=0.0)
    gen = out[0, len(ids):].tolist()
    print("  sinh ra ids:", gen)
    print("  decode:", repr(sp.decode(gen)))
