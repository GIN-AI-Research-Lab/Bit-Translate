# -*- coding: utf-8 -*-
r"""TQ33 SPEED-100 — oracle PyTorch cho THÊM 2 prompt thật (1 VI dài hơn, 1 JA) để
validate nén int8 embed/lm_head trên nhiều câu, không chỉ 1 prompt 7 token cũ.
Logic model/monkey-patch bias/output-head GIỮ NGUYÊN ref_forward_pytorch.py (đã validate).

Output: D:\Bit-Translate-data\tq33_runner\oracle_p2\ , oracle_p3\
  (tokens.bin, hidden_l{i}.npy, hidden_final.npy, logits.npy, summary.txt — format y hệt oracle\)
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("OMP_NUM_THREADS", "5")
os.environ.setdefault("MKL_NUM_THREADS", "5")
torch.set_num_threads(5)

CKPT = r"D:\Bit-Translate-data\qat_ckpts\qat_gen4_n4.pt"
MDIR = (r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B"
        r"\snapshots\c1899de289a04d12100db370d81485cdf75e47ca")
BASE = r"D:\Bit-Translate-data\tq33_runner"

PROMPTS = {
    "oracle_p2": "Chiều nay trời mưa to, tôi phải mang ô đi làm và đường phố rất đông người.",
    "oracle_p3": "今日は天気がいいので、公園へ散歩に行きましょう。",
}

PROJS_ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
PROJS_MLP = ["gate_proj", "up_proj", "down_proj"]


def build_model():
    cfg = AutoConfig.from_pretrained(MDIR)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)
    model.eval()
    for blk in model.model.layers:
        for name in PROJS_ATTN:
            lin = getattr(blk.self_attn, name)
            lin.bias = nn.Parameter(torch.zeros(lin.weight.shape[0], dtype=lin.weight.dtype))
        for name in PROJS_MLP:
            lin = getattr(blk.mlp, name)
            lin.bias = nn.Parameter(torch.zeros(lin.weight.shape[0], dtype=lin.weight.dtype))
    return model, cfg


def main():
    print("loading ckpt...")
    sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
    state = sd["state_dict"] if "state_dict" in sd else sd
    model, cfg = build_model()
    tok = AutoTokenizer.from_pretrained(MDIR)

    body_sd = {k: v.float() for k, v in state.items()
               if k not in ("lm_head.weight", "model.embed_tokens.weight")}
    missing, unexpected = model.load_state_dict(body_sd, strict=False)
    assert set(missing) == {"model.embed_tokens.weight", "lm_head.weight"}, missing
    assert unexpected == [], unexpected
    gen_emb = state["model.embed_tokens.weight"].float()
    model.model.embed_tokens.weight.data = gen_emb.clone()
    model.lm_head.weight.data = gen_emb.clone()

    for dirname, prompt in PROMPTS.items():
        out_dir = os.path.join(BASE, dirname)
        os.makedirs(out_dir, exist_ok=True)
        captured, norm_out = {}, {}

        def make_hook(i):
            def hook(module, inp, out):
                captured[i] = out.detach().clone()
            return hook

        handles = [blk.register_forward_hook(make_hook(i))
                   for i, blk in enumerate(model.model.layers)]
        h_norm = model.model.norm.register_forward_hook(
            lambda m, i, o: norm_out.__setitem__("final", o.detach().clone()))

        ids = tok(prompt, return_tensors="pt").input_ids
        print(f"\n[{dirname}] {prompt!r} -> {ids.shape[1]} tokens: {ids[0].tolist()}")
        with torch.no_grad():
            out = model(input_ids=ids)
        logits = out.logits.detach()
        for h in handles:
            h.remove()
        h_norm.remove()

        tok_ids = np.asarray(ids[0].tolist(), dtype=np.int32)
        with open(os.path.join(out_dir, "tokens.bin"), "wb") as f:
            f.write(np.asarray([len(tok_ids)], dtype=np.int32).tobytes())
            f.write(tok_ids.tobytes())
        for i in range(cfg.num_hidden_layers):
            np.save(os.path.join(out_dir, f"hidden_l{i}.npy"),
                    captured[i][0].numpy().astype(np.float32))
        np.save(os.path.join(out_dir, "hidden_final.npy"),
                norm_out["final"][0].numpy().astype(np.float32))
        logits_arr = logits[0].numpy().astype(np.float32)
        np.save(os.path.join(out_dir, "logits.npy"), logits_arr)

        last = logits_arr[-1]
        top5 = np.argsort(-last)[:5]
        lines = [f"prompt: {prompt!r}", f"tokens: {tok_ids.tolist()}",
                 "top-5 next-token tai vi tri cuoi:"]
        for t in top5:
            lines.append(f"    id={t:6d}  logit={last[t]:.4f}  repr={tok.decode([int(t)])!r}")
        with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
