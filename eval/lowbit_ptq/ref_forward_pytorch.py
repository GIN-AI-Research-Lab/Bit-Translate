# -*- coding: utf-8 -*-
r"""TQ33 Runner — Giai đoạn 1, bước 2: forward pass THAM CHIẾU bằng PyTorch/transformers
thật (Qwen3ForCausalLM kiến trúc chuẩn + load_state_dict ckpt bake gen4_n4.pt), lưu lại
hidden state SAU MỖI LAYER (28 layer, RAW — trước final norm) + hidden state sau final
norm + logits, để qwen3_runner.c so khớp từng bước (debug được CHÍNH XÁC layer nào sai
nếu có lệch, thay vì chỉ biết "logits cuối sai").

Quyết định kỹ thuật quan trọng (xem export_full_model_f32.py đầu file để biết lý do):
  1. Monkey-patch bias vào q/k/v/o/gate/up/down proj TRƯỚC khi load_state_dict — kiến
     trúc Qwen3 chuẩn HF không có bias ở các linear này (attention_bias=false, MLP luôn
     bias=False), nhưng ckpt bake CÓ bias thật (do LearnQLinear luôn thêm bias riêng).
     Không patch trước -> load_state_dict(strict=False) sẽ ÂM THẦM BỎ QUA bias (đã kiểm
     chứng bị chính lỗi này lúc đầu, xem resolve_lmhead.py).
  2. Output head dùng model.embed_tokens.weight (không dùng lm_head.weight dù tồn tại
     riêng trong ckpt) — set thủ công SAU load_state_dict để tránh phụ thuộc thứ tự
     traversal khi model có tie_word_embeddings=True (2 Parameter cùng object lúc fresh
     construct, load_state_dict với 2 giá trị khác nhau cho cùng storage sẽ chỉ giữ lại
     giá trị nạp SAU CÙNG theo thứ tự duyệt module — không đáng tin cậy nếu không set
     tường minh lại).

Dùng: python ref_forward_pytorch.py
Output: D:\Bit-Translate-data\tq33_runner\oracle\
  tokens.bin        (int32: n_tok, rồi n_tok token id)
  hidden_l{i}.npy    i=0..27, RAW output sau layer i, shape [seq, 1024] float32
  hidden_final.npy  sau model.norm, shape [seq, 1024]
  logits.npy        shape [seq, 151936] float32
  summary.txt       token list, top-5 next-token dự đoán (người đọc được)
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
torch.set_num_threads(6)

CKPT = r"D:\Bit-Translate-data\qat_ckpts\qat_gen4_n4.pt"
MDIR = (r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B"
         r"\snapshots\c1899de289a04d12100db370d81485cdf75e47ca")
OUT_DIR = r"D:\Bit-Translate-data\tq33_runner\oracle"
PROMPT = "Xin chào, hôm nay"

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
    os.makedirs(OUT_DIR, exist_ok=True)
    print("loading ckpt...")
    sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
    state = sd["state_dict"] if "state_dict" in sd else sd

    print("building model + loading body weights...")
    model, cfg = build_model()
    tok = AutoTokenizer.from_pretrained(MDIR)

    body_sd = {k: v.float() for k, v in state.items()
               if k not in ("lm_head.weight", "model.embed_tokens.weight")}
    missing, unexpected = model.load_state_dict(body_sd, strict=False)
    assert set(missing) == {"model.embed_tokens.weight", "lm_head.weight"}, missing
    assert unexpected == [], unexpected

    gen_emb = state["model.embed_tokens.weight"].float()
    model.model.embed_tokens.weight.data = gen_emb.clone()
    model.lm_head.weight.data = gen_emb.clone()  # quyết định: dùng embed_tokens cho output head

    # --- hook per-layer RAW output (trước final norm) ---
    captured = {}

    def make_hook(i):
        def hook(module, inp, out):
            # Qwen3DecoderLayer.forward trả về tensor thẳng (không phải tuple)
            captured[i] = out.detach().clone()
        return hook

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(model.model.layers)]

    norm_out = {}

    def norm_hook(module, inp, out):
        norm_out["final"] = out.detach().clone()

    h_norm = model.model.norm.register_forward_hook(norm_hook)

    ids = tok(PROMPT, return_tensors="pt").input_ids
    seq_len = ids.shape[1]
    print(f"prompt: {PROMPT!r} -> {seq_len} tokens: {ids[0].tolist()}")
    for t in ids[0].tolist():
        print(f"    {t:6d}  {tok.decode([t])!r}")

    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits.detach()  # [1, seq, vocab]

    for h in handles:
        h.remove()
    h_norm.remove()

    assert len(captured) == cfg.num_hidden_layers
    assert "final" in norm_out

    # --- lưu ra file ---
    tok_ids = np.asarray(ids[0].tolist(), dtype=np.int32)
    with open(os.path.join(OUT_DIR, "tokens.bin"), "wb") as f:
        f.write(np.asarray([len(tok_ids)], dtype=np.int32).tobytes())
        f.write(tok_ids.tobytes())

    for i in range(cfg.num_hidden_layers):
        arr = captured[i][0].numpy().astype(np.float32)  # [seq, hidden]
        np.save(os.path.join(OUT_DIR, f"hidden_l{i}.npy"), arr)

    final_arr = norm_out["final"][0].numpy().astype(np.float32)
    np.save(os.path.join(OUT_DIR, "hidden_final.npy"), final_arr)

    logits_arr = logits[0].numpy().astype(np.float32)  # [seq, vocab]
    np.save(os.path.join(OUT_DIR, "logits.npy"), logits_arr)

    # top-5 next-token tại vị trí CUỐI (dự đoán token tiếp theo sau prompt)
    last_logits = logits_arr[-1]
    top5 = np.argsort(-last_logits)[:5]
    lines = [f"prompt: {PROMPT!r}", f"tokens: {tok_ids.tolist()}",
             f"decoded per-token: {[tok.decode([t]) for t in tok_ids.tolist()]}",
             "", "top-5 next-token tai vi tri cuoi:"]
    for t in top5:
        lines.append(f"    id={t:6d}  logit={last_logits[t]:.4f}  repr={tok.decode([int(t)])!r}")
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(f"\nda luu {cfg.num_hidden_layers} hidden_l*.npy + hidden_final.npy + logits.npy"
          f" + tokens.bin + summary.txt -> {OUT_DIR}")
    print(f"hidden_l0 shape={captured[0][0].shape}  logits shape={logits_arr.shape}")


if __name__ == "__main__":
    main()
