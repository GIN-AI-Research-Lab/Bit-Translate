#!/usr/bin/env python3
"""Bước 5 — evaluate a checkpoint with chrF (sacrebleu) both directions.

chrF is the primary metric (CLAUDE.md §5): character-level, robust for Japanese
without word segmentation. We translate FLORES-200 devtest (neutral benchmark)
and our held-out dev set in both VI->JA and JA->VI, greedily.

Greedy decode is length-grouped-batched (sentences sharing a tokenized source
length decode together, so the causal last-position stays aligned without a
padding mask). Prints chrF per direction; appends to eval/history.tsv.
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import sacrebleu

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from bitnet import BitNetLM, BitNetConfig
import sentencepiece as spm

ROOT = Path(__file__).parent.parent
CKPT = ROOT / "checkpoints"
EVALDIR = ROOT / "eval"


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = BitNetConfig(**ck["cfg"]) if "cfg" in ck else BitNetConfig()
    model = BitNetLM(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, ck.get("step", -1)


def autocast_ctx(device):
    import contextlib
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def translate(model, sp, srcs, tgt_lang, device, max_new=200, batch=64):
    bos, eos = sp.bos_id(), sp.eos_id()
    tag = sp.piece_to_id(f">>{tgt_lang}<<")
    prefixes = [[bos, tag] + sp.encode(s) + [eos] for s in srcs]
    order = sorted(range(len(srcs)), key=lambda i: len(prefixes[i]))
    out = [None] * len(srcs)

    groups = defaultdict(list)
    for i in order:
        groups[len(prefixes[i])].append(i)

    for L, members in groups.items():
        for b in range(0, len(members), batch):
            chunk = members[b:b + batch]
            idx = torch.tensor([prefixes[i] for i in chunk], device=device)
            done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
            gen = [[] for _ in chunk]
            for _ in range(max_new):
                with autocast_ctx(idx.device.type):
                    logits, _ = model(idx[:, -model.cfg.max_seq:])
                nxt = logits[:, -1, :].argmax(-1)
                for r in range(len(chunk)):
                    if not done[r]:
                        t = int(nxt[r])
                        if t == eos:
                            done[r] = True
                        else:
                            gen[r].append(t)
                if done.all():
                    break
                idx = torch.cat([idx, nxt[:, None]], dim=1)
            for r, i in enumerate(chunk):
                out[i] = sp.decode(gen[r])
    return out


def eval_direction(model, sp, src_file, ref_file, tgt_lang, device, n):
    srcs = Path(src_file).read_text(encoding="utf-8").splitlines()[:n]
    refs = Path(ref_file).read_text(encoding="utf-8").splitlines()[:n]
    hyps = translate(model, sp, srcs, tgt_lang, device)
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    return chrf, hyps, srcs, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT / "last.pt"))
    ap.add_argument("--n", type=int, default=500, help="sentences per direction (subset for speed)")
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu avoids contending with a running GPU training job")
    args = ap.parse_args()
    device = args.device
    if device == "cpu":
        torch.set_num_threads(6)
    EVALDIR.mkdir(exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
    model, step = load_model(args.ckpt, device)

    F = ROOT / "data" / "clean" / "flores"
    jobs = [
        ("FLORES vi->ja", F / "flores.devtest.vi", F / "flores.devtest.ja", "jpn"),
        ("FLORES ja->vi", F / "flores.devtest.ja", F / "flores.devtest.vi", "vie"),
        ("dev vi->ja", ROOT / "data/clean/dev.vi", ROOT / "data/clean/dev.ja", "jpn"),
        ("dev ja->vi", ROOT / "data/clean/dev.ja", ROOT / "data/clean/dev.vi", "vie"),
    ]
    t0 = time.time()
    results = {}
    for name, src, ref, tl in jobs:
        chrf, hyps, srcs, refs = eval_direction(model, sp, src, ref, tl, device, args.n)
        results[name] = chrf
        print(f"\n### {name}: chrF = {chrf:.2f}")
        for k in range(min(args.show, len(hyps))):
            print(f"  SRC: {srcs[k]}")
            print(f"  HYP: {hyps[k]}")
            print(f"  REF: {refs[k]}")

    line = f"step={step}\t" + "\t".join(f"{k}={v:.2f}" for k, v in results.items()) + \
           f"\t({time.time()-t0:.0f}s, n={args.n})"
    with (EVALDIR / "history.tsv").open("a") as f:
        f.write(line + "\n")
    print("\n" + line)


if __name__ == "__main__":
    main()
