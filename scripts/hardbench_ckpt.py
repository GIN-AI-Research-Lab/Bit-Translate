#!/usr/bin/env python3
"""Chạy hardbench ja2vi trên checkpoint .pt (CPU), chrF vs ref, so Google/Haiku.

  python scripts/hardbench_ckpt.py checkpoints/kd_step11000.pt            # greedy
  python scripts/hardbench_ckpt.py checkpoints/kd_step15000.pt --beam 5   # beam search
  python scripts/hardbench_ckpt.py <ckpt> --beam 5 --len-alpha 1.0 --min-new 4
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import sacrebleu

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from bitnet import BitNetLM, BitNetConfig
from text_norm import normalize_for_model
import sentencepiece as spm

ROOT = Path(__file__).parent.parent
torch.set_num_threads(6)


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    c = ck["cfg"]
    cfg = BitNetConfig(vocab_size=c["vocab_size"], d_model=c["d_model"], n_layers=c["n_layers"],
                       n_heads=c["n_heads"], d_ff=c["d_ff"], max_seq=c["max_seq"])
    m = BitNetLM(cfg).eval()
    m.load_state_dict(ck["model"])
    m.freeze_for_inference()
    return m, ck.get("step", "?")


def dom_agg(rows):
    agg = {}
    for r in rows:
        agg.setdefault(r["domain"], []).append(r["chrf"])
    return {k: sum(v) / len(v) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="checkpoints/kd_step11000.pt")
    ap.add_argument("--beam", type=int, default=0, help="0 = greedy (như cũ); 4-5 = beam search")
    ap.add_argument("--len-alpha", type=float, default=0.6, help="phạt độ dài GNMT")
    ap.add_argument("--min-new", type=int, default=0, help="chặn EOS trước N token")
    ap.add_argument("--out", default=None, help="file kết quả (mặc định theo chế độ decode)")
    a = ap.parse_args()
    path = a.ckpt
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
    bos, eos, vie = sp.bos_id(), sp.eos_id(), sp.piece_to_id(">>vie<<")
    model, step = load_model(path)
    how = "greedy" if a.beam <= 1 else f"beam={a.beam} len_alpha={a.len_alpha} min_new={a.min_new}"
    print(f"Model step {step} | chạy 100 câu ja2vi hardbench (CPU, {how})...\n", flush=True)

    probes = [json.loads(l) for l in (ROOT / "eval" / "hardbench200.jsonl").open(encoding="utf-8")
              if json.loads(l)["dir"] == "ja2vi"]
    rows = []
    for i, p in enumerate(probes):
        text = normalize_for_model(p["src"])
        ids = torch.tensor([[bos, vie] + sp.encode(text) + [eos]])
        with torch.inference_mode():
            if a.beam > 1:
                out = model.generate_beam(ids, max_new_tokens=160, eos_id=eos,
                                          beam=a.beam, len_alpha=a.len_alpha,
                                          min_new=a.min_new)
            else:
                out = model.generate_cached(ids, max_new_tokens=160, eos_id=eos, rep_penalty=1.0)
        gen = out[0, ids.shape[1]:].tolist()
        if eos in gen:
            gen = gen[:gen.index(eos)]
        hyp = sp.decode(gen)
        chrf = sacrebleu.sentence_chrf(hyp, [p["ref"]]).score
        rows.append({**p, "hyp": hyp, "chrf": round(chrf, 1)})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/100", flush=True)

    tag = "greedy" if a.beam <= 1 else f"beam{a.beam}"
    out_path = Path(a.out) if a.out else ROOT / "eval" / f"hardbench_kd100m_{step}_{tag}.jsonl"
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")

    ours = dom_agg(rows)
    goog = dom_agg([json.loads(l) for l in (ROOT / "eval" / "hardbench_google.jsonl").open(encoding="utf-8")
                    if json.loads(l)["dir"] == "ja2vi"])
    haik = dom_agg([json.loads(l) for l in (ROOT / "eval" / "hardbench_haiku.jsonl").open(encoding="utf-8")
                    if json.loads(l)["dir"] == "ja2vi"])

    print(f"\n=== hardbench ja2vi chrF (checkpoint step {step}) ===")
    print(f"{'domain':12} {'KD-100M':>8} {'Google':>7} {'Haiku':>7}  {'vs Google':>9}")
    print("-" * 50)
    for d in sorted(ours):
        diff = ours[d] - goog[d]
        print(f"{d:12} {ours[d]:8.1f} {goog[d]:7.1f} {haik[d]:7.1f}  {diff:+8.1f}")
    o = sum(r["chrf"] for r in rows) / len(rows)
    g = sum(ours[d] * 0 for d in ours)  # placeholder
    import statistics
    gall = statistics.mean(json.loads(l)["chrf"] for l in (ROOT / "eval" / "hardbench_google.jsonl").open(encoding="utf-8") if json.loads(l)["dir"] == "ja2vi")
    hall = statistics.mean(json.loads(l)["chrf"] for l in (ROOT / "eval" / "hardbench_haiku.jsonl").open(encoding="utf-8") if json.loads(l)["dir"] == "ja2vi")
    print("-" * 50)
    print(f"{'TỔNG TB':12} {o:8.1f} {gall:7.1f} {hall:7.1f}  {o-gall:+8.1f}")


if __name__ == "__main__":
    main()
