#!/usr/bin/env python3
"""Dịch thử từ checkpoint .pt thô (chưa export GGUF) trên CPU.

Nạp {"model":state_dict,"cfg":...} vào BitNetLM, freeze (ternary + 8-bit act như
model thật), dịch list câu ja->vi. Input chuẩn hóa NFKC khớp lúc train.

  python scripts/translate_ckpt.py checkpoints/kd_step11000.pt
"""
import sys
import unicodedata
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from bitnet import BitNetLM, BitNetConfig
import sentencepiece as spm

ROOT = Path(__file__).parent.parent
torch.set_num_threads(6)

TESTS = [
    "猫が好きです。",
    "おはようございます。",
    "駅はどこですか？",
    "今日は天気がいいので、公園を散歩しようと思います。",
    "お忙しいところ恐れ入りますが、ご確認をお願いいたします。",
    "このAPIはJSONレスポンスを返します。",
    "彼は猫の手も借りたいほど忙しい。",
    "それは決して不可能ではない。",
    "雨が降っていたのに、彼は傘を持たずに出かけた。",
    "私が昨日買った本はとても面白かった。",
    "この製品は水冷式で高温に強い設計です。",
    "会議は来週の月曜日の午後三時から始まります。",
]


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/kd_step11000.pt"
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
    ck = torch.load(ckpt_path, map_location="cpu")
    c = ck["cfg"]
    cfg = BitNetConfig(vocab_size=c["vocab_size"], d_model=c["d_model"],
                       n_layers=c["n_layers"], n_heads=c["n_heads"],
                       d_ff=c["d_ff"], max_seq=c["max_seq"])
    model = BitNetLM(cfg).eval()
    model.load_state_dict(ck["model"])
    model.freeze_for_inference()
    print(f"Checkpoint step {ck.get('step','?')} | {model.num_params()/1e6:.1f}M "
          f"(d={cfg.d_model} L={cfg.n_layers})\n")

    bos, eos = sp.bos_id(), sp.eos_id()
    vie = sp.piece_to_id(">>vie<<")
    for ja in TESTS:
        norm = unicodedata.normalize("NFKC", ja).strip()
        ids = torch.tensor([[bos, vie] + sp.encode(norm) + [eos]])
        with torch.inference_mode():
            out = model.generate_cached(ids, max_new_tokens=120, eos_id=eos, rep_penalty=1.3)
        gen = out[0, ids.shape[1]:].tolist()
        if eos in gen:
            gen = gen[:gen.index(eos)]
        print(f"JA: {ja}")
        print(f"VI: {sp.decode(gen)}\n")


if __name__ == "__main__":
    main()
