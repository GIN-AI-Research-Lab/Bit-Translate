#!/usr/bin/env python3
"""Test nhanh 1 checkpoint: load .pt -> freeze (ternary+8bit act, GIỐNG i2_s
inference) -> generate vài câu 2 chiều. Chạy CPU, không đụng GPU đang train.
Là bản xem trước trung thực cho bitnet.cpp (cùng phép lượng tử)."""
import sys
from pathlib import Path
import torch
import sentencepiece as spm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bitnet import BitNetLM, BitNetConfig

CKPT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "checkpoints" / "last.pt")
sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
cfg = BitNetConfig(vocab_size=sp.get_piece_size())
m = BitNetLM(cfg).eval()
ck = torch.load(CKPT, map_location="cpu")
m.load_state_dict(ck["model"])
print(f"loaded {CKPT} step={ck.get('step','?')} loss={ck.get('loss','?')}")
m.freeze_for_inference()

BOS, EOS = sp.bos_id(), sp.eos_id()
JPN, VIE = sp.piece_to_id(">>jpn<<"), sp.piece_to_id(">>vie<<")
tests = [
    (JPN, ">>jpn<<", "Tôi thích mèo."),
    (JPN, ">>jpn<<", "Xin chào, bạn khỏe không?"),
    (JPN, ">>jpn<<", "Hôm nay trời đẹp."),
    (VIE, ">>vie<<", "猫が好きです。"),
    (VIE, ">>vie<<", "おはようございます。"),
    (VIE, ">>vie<<", "これはテストです。"),
]
for tagid, tag, src in tests:
    ids = [BOS, tagid] + sp.encode(src) + [EOS]
    out = m.generate_cached(torch.tensor([ids]), max_new_tokens=48, eos_id=EOS)
    gen = out[0, len(ids):].tolist()
    if gen and gen[-1] == EOS:
        gen = gen[:-1]
    print(f"  {tag} {src!r}\n      -> {sp.decode(gen)!r}")
