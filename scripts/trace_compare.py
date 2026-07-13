#!/usr/bin/env python3
"""Dump sum từng node layer-0 (giống tên ggml) từ forward PyTorch (no-quant) để
so với llama-eval-callback (Q8_0). Tìm điểm lệch graph."""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from gguf import GGUFReader
import sentencepiece as spm
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT/"src"))
from bitnet import BitNetLM, BitNetConfig, apply_rope

sp = spm.SentencePieceProcessor(model_file=str(ROOT/"tokenizer/spm_vija_32k.model"))
cfg = BitNetConfig(vocab_size=sp.get_piece_size()); m = BitNetLM(cfg).eval()
r = GGUFReader(str(ROOT/"dist/vija_1200_f16.gguf"))
g = {t.name: torch.from_numpy(np.array(t.data,dtype=np.float32).reshape(tuple(reversed(t.shape)))) for t in r.tensors}
def L(d,n): d.data.copy_(g[n])
L(m.embed.weight,'token_embd.weight'); L(m.output_norm.weight,'output_norm.weight')
for i,b in enumerate(m.blocks):
    p=f'blk.{i}.'
    L(b.attn.attn_norm.weight,p+'attn_norm.weight');L(b.attn.wq.weight,p+'attn_q.weight');L(b.attn.wk.weight,p+'attn_k.weight');L(b.attn.wv.weight,p+'attn_v.weight');L(b.attn.attn_sub_norm.weight,p+'attn_sub_norm.weight');L(b.attn.wo.weight,p+'attn_output.weight')
    L(b.ffn.ffn_norm.weight,p+'ffn_norm.weight');L(b.ffn.gate.weight,p+'ffn_gate.weight');L(b.ffn.up.weight,p+'ffn_up.weight');L(b.ffn.ffn_sub_norm.weight,p+'ffn_sub_norm.weight');L(b.ffn.down.weight,p+'ffn_down.weight')

m.freeze_for_inference()  # ternary weights, khớp Q8_0/i2_s
def S(name, t): print(f"  {name:18s} {t.float().sum().item():.6f}")

# CÙNG token mà eval-callback nạp (11588, inp_embd~0.0776)
tok = [11588]
idx = torch.tensor([tok])
eps = 1e-5
x = m.embed(idx); S("inp_embd", x)
blk = m.blocks[0]; a = blk.attn
def rms(z): return z*torch.rsqrt(z.float().pow(2).mean(-1,keepdim=True)+eps)
n0 = rms(x); S("norm-0", n0)
an = n0*a.attn_norm.weight; S("attn_norm-0", an)
B,T,_ = an.shape; H,Dh = a.n_heads, a.head_dim
q = a.wq(an); S("Qcur(raw)", q)
k = a.wk(an); v = a.wv(an)
qh = q.view(B,T,H,Dh).transpose(1,2); kh = k.view(B,T,H,Dh).transpose(1,2); vh = v.view(B,T,H,Dh).transpose(1,2)
cos,sin = m.rope_cos[:T], m.rope_sin[:T]
qr = apply_rope(qh,cos,sin); kr = apply_rope(kh,cos,sin)
S("Qcur(rope)", qr); S("Kcur(rope)", kr); S("Vcur", vh)
out = F.scaled_dot_product_attention(qr,kr,vh,is_causal=True)
out = out.transpose(1,2).contiguous().view(B,T,-1); S("kqv_out(pre-o)", out)
asn = rms(out)*a.attn_sub_norm.weight
o = a.wo(asn); S("kqv_out(o)", o)
ffn_inp = x + o; S("ffn_inp", ffn_inp)
f = blk.ffn
fn = rms(ffn_inp)*f.ffn_norm.weight; S("ffn_norm", fn)
gate = f.gate(fn); S("ffn_gate", gate)
gr = F.relu(gate); S("ffn_relu", gr)
gs = gr.square(); S("ffn_sqr(relu)", gs)
up = f.up(fn); S("ffn_up", up)
fo = gs*up; S("ffn_out", fo)
fsn = rms(fo)*f.ffn_sub_norm.weight
down = f.down(fsn); S("ffn_down", down)
lout = ffn_inp + down; S("l_out", lout)
