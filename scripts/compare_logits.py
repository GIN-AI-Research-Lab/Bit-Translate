#!/usr/bin/env python3
"""Nạp CÙNG bộ trọng số (f32 gguf) vào PyTorch, in top-8 token kế tiếp + chuỗi
greedy (bỏ qua EOS). So khớp với bitnet.cpp trên đúng file i2_s để xác minh
đường convert đúng số học (không phụ thuộc model đã train hay chưa)."""
import sys
from pathlib import Path
import numpy as np
import torch
from gguf import GGUFReader
import sentencepiece as spm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bitnet import BitNetLM, BitNetConfig

GGUF = sys.argv[1]
sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
cfg = BitNetConfig(vocab_size=sp.get_piece_size())
model = BitNetLM(cfg).eval()

r = GGUFReader(GGUF)
g = {t.name: torch.from_numpy(np.array(t.data, dtype=np.float32).reshape(tuple(reversed(t.shape)))) for t in r.tensors}
def L(dst, n): dst.data.copy_(g[n])
L(model.embed.weight, "token_embd.weight"); L(model.output_norm.weight, "output_norm.weight")
for i, b in enumerate(model.blocks):
    p = f"blk.{i}."
    L(b.attn.attn_norm.weight, p+"attn_norm.weight"); L(b.attn.wq.weight, p+"attn_q.weight"); L(b.attn.wk.weight, p+"attn_k.weight"); L(b.attn.wv.weight, p+"attn_v.weight"); L(b.attn.attn_sub_norm.weight, p+"attn_sub_norm.weight"); L(b.attn.wo.weight, p+"attn_output.weight")
    L(b.ffn.ffn_norm.weight, p+"ffn_norm.weight"); L(b.ffn.gate.weight, p+"ffn_gate.weight"); L(b.ffn.up.weight, p+"ffn_up.weight"); L(b.ffn.ffn_sub_norm.weight, p+"ffn_sub_norm.weight"); L(b.ffn.down.weight, p+"ffn_down.weight")

# freeze -> forward dùng ternary weight + 8bit act (mô phỏng i2_s inference)
model.freeze_for_inference()
BOS, EOS, JPN = sp.bos_id(), sp.eos_id(), sp.piece_to_id(">>jpn<<")
ids = [BOS, JPN] + sp.encode("Tôi thích mèo.") + [EOS]
print("prompt ids:", ids)
logits, _ = model(torch.tensor([ids]))
top = torch.topk(torch.softmax(logits[0, -1].float(), -1), 8)
print("PyTorch top-8 token kế tiếp:")
for pb, t in zip(top.values.tolist(), top.indices.tolist()):
    print(f"   id={t:5d} p={pb:.3f} '{sp.id_to_piece(t)}'")
# greedy 8 token, KHÔNG dừng ở EOS
out = model.generate(torch.tensor([ids]), max_new_tokens=8, eos_id=-999, temperature=0.0)
print("PyTorch greedy 8 token (bỏ EOS):", out[0, len(ids):].tolist())
