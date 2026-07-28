#!/usr/bin/env python3
"""Dịch một file .txt (mỗi dòng một câu ja) bằng checkpoint .pt -> .jsonl.

Dùng để dựng bench theo MIỀN: mỗi miền một file txt, chạy cùng model, so tỷ lệ
dùng được giữa các miền. Đây là thước đo thay cho bench trộn lẫn — vòng 5 đã chứng
minh bench trộn lẫn giấu mất cải thiện trong miền.

  python scripts/translate_txt.py <ckpt> <in.txt> <out.jsonl> [--label X]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from bitnet import BitNetLM, BitNetConfig  # noqa: E402
from text_norm import normalize_for_model  # noqa: E402
import sentencepiece as spm  # noqa: E402

ROOT = Path(__file__).parent.parent
torch.set_num_threads(6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--label", default="")
    ap.add_argument("--max-new", type=int, default=200)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    c = ck["cfg"]
    cfg = BitNetConfig(vocab_size=c["vocab_size"], d_model=c["d_model"], n_layers=c["n_layers"],
                       n_heads=c["n_heads"], d_ff=c["d_ff"], max_seq=c["max_seq"])
    m = BitNetLM(cfg).eval()
    m.load_state_dict(ck["model"])
    m.freeze_for_inference()
    n_par = sum(p.numel() for p in m.parameters())
    print(f"[{a.label}] {c['n_layers']}L d{c['d_model']} | {n_par/1e6:.1f}M", flush=True)

    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
    bos, eos = sp.piece_to_id("<s>"), sp.piece_to_id("</s>")
    vie = sp.piece_to_id(">>vie<<")

    lines = [l.strip() for l in Path(a.src).open(encoding="utf-8") if l.strip()]
    rows = []
    for i, s in enumerate(lines):
        ids = torch.tensor([[bos, vie] + sp.encode(normalize_for_model(s)) + [eos]])
        with torch.inference_mode():
            out = m.generate_cached(ids, max_new_tokens=a.max_new, eos_id=eos, rep_penalty=1.0)
        gen = out[0, ids.shape[1]:].tolist()
        if eos in gen:
            gen = gen[:gen.index(eos)]
        rows.append({"id": str(i + 1), "src": s, "hyp": sp.decode(gen)})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(lines)}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                           encoding="utf-8")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
