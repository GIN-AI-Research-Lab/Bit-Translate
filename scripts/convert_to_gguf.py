#!/usr/bin/env python3
"""Convert our BitNetLM checkpoint -> GGUF (arch "bitnet-b1.58") for bitnet.cpp.

Writes an F32 GGUF with the exact tensor names + metadata bitnet.cpp expects
(verified from the reference GGUF), plus the embedded SentencePiece tokenizer.
Then quantize — BẮT BUỘC single-thread (tham số cuối `1`):

    llama-quantize <out> <i2s> I2_S 1

CẢNH BÁO: bỏ số `1` (nthreads) là llama-quantize chạy đa luồng và GHI HỎNG
i2_s: ggml_quantize_chunk đặt mỗi chunk tại start_row*row_size (stride 4x quá
xa so với dữ liệu packed 2-bit), per-tensor scale bị ghi đè/lạc chỗ; runtime
đọc scale tại offset ne0*ne1/4 (ggml.c:12457) trúng vùng chưa ghi = 0.0
-> MỌI matmul ternary ra đúng 0 -> model câm. Pipeline chính chủ luôn truyền
nthreads=1 (setup_env.py). Chỉ i2_s dính lỗi này; Q8_0/F16 quantize an toàn.

With --ckpt: convert a trained model. Without: build an UNTRAINED model (for
validating the conversion+run path before committing to a long retrain).

Run in the bitnet venv (needs gguf + sentencepiece + torch).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import sentencepiece as spm
from gguf import GGUFWriter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bitnet import BitNetLM, BitNetConfig

# llama.cpp token-type enum
T_NORMAL, T_UNKNOWN, T_CONTROL, T_USERDEF, T_UNUSED, T_BYTE = 1, 2, 3, 4, 5, 6


def add_tokenizer(w, sp):
    n = sp.get_piece_size()
    tokens, scores, toktypes = [], [], []
    for i in range(n):
        p = sp.id_to_piece(i)
        tokens.append(p.encode("utf-8"))
        scores.append(sp.get_score(i))
        if sp.is_unknown(i):
            t = T_UNKNOWN
        elif sp.is_control(i):
            t = T_CONTROL
        elif sp.is_byte(i):
            t = T_BYTE
        elif sp.is_unused(i):
            t = T_UNUSED
        elif p in (">>vie<<", ">>jpn<<"):
            t = T_USERDEF
        else:
            t = T_NORMAL
        toktypes.append(t)
    w.add_tokenizer_model("llama")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(toktypes)
    w.add_bos_token_id(sp.bos_id())
    w.add_eos_token_id(sp.eos_id())
    w.add_unk_token_id(sp.unk_id())
    w.add_pad_token_id(sp.pad_id())
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint .pt; omit = untrained (for validation)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
    ap.add_argument("--f16", action="store_true",
                    help="linears F16 thay vì F32 -> file ~1/2 (dễ tải qua mạng cloud chậm). "
                         "i2_s sau khi quantize GIỐNG HỆT bản F32 (llama-quantize đọc về f32 nội bộ). "
                         "norms+embed VẪN F32 (kernel bitnet đòi src1 f32).")
    # Dims: mặc định None = tự đọc từ cfg lưu trong checkpoint (train.py lưu
    # "cfg": vars(cfg)); truyền cờ chỉ khi test dims lạ / ckpt không có cfg.
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--d-ff", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    ck = torch.load(args.ckpt, map_location="cpu") if args.ckpt else None
    ckcfg = (ck or {}).get("cfg") or {}
    def dim(key, arg, fallback):
        return arg if arg is not None else ckcfg.get(key, fallback)
    cfg = BitNetConfig(vocab_size=sp.get_piece_size(),
                       d_model=dim("d_model", args.d_model, 768),
                       d_ff=dim("d_ff", args.d_ff, 2048),
                       n_layers=dim("n_layers", args.n_layers, 12),
                       n_heads=dim("n_heads", args.n_heads, 12))
    model = BitNetLM(cfg).eval()
    if ck:
        model.load_state_dict(ck["model"])
        print(f"loaded {args.ckpt} step {ck.get('step', '?')} | "
              f"d={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads} ff={cfg.d_ff}")
    else:
        print("UNTRAINED model (validation mode)")

    # arch "bitnet-b1.58" -> llama.cpp build_bitnet_158() (squared-ReLU FFN),
    # the graph Microsoft's BitNet-b1.58-2B-4T uses and that runs correctly on
    # this build. (The older "bitnet"/build_bitnet SiLU graph is broken here.)
    w = GGUFWriter(args.out, "bitnet-b1.58")
    w.add_context_length(cfg.max_seq)
    w.add_embedding_length(cfg.d_model)
    w.add_block_count(cfg.n_layers)
    w.add_feed_forward_length(cfg.d_ff)
    w.add_head_count(cfg.n_heads)
    w.add_head_count_kv(cfg.n_heads)
    w.add_rope_dimension_count(cfg.d_model // cfg.n_heads)  # build_bitnet_158 asserts n_embd_head == n_rot
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-5)
    add_tokenizer(w, sp)

    # Everything F32, exactly like the reference (bitnet_b1_58 / 2B-4T) f32 GGUF
    # that llama-quantize turns into i2_s. norms+embed MUST be F32 (their outputs
    # feed the ternary matmul, which asserts f32 src1); linears are F32 too so the
    # i2_s quantizer sees full-precision master weights (no F16 rounding surprises)
    # — this intermediate f32 file is quantized to i2_s on the cloud, and only the
    # ~67MB i2_s is ever downloaded, so its size doesn't matter.
    def t32(t):
        return t.detach().float().numpy()

    def weight_quant_ternary(w):
        """Giải-lượng-tử ternary GIỐNG HỆT src/bitnet.py weight_quant: trả về
        {-m, 0, +m} với m=mean|w|. BẮT BUỘC: quantize_i2_s của bitnet.cpp gán
        ternary theo DẤU + scale=max|w| (KHÔNG tự chuẩn hoá absmean). Nếu lưu
        weight master thô (~0.02) thì phân bố sai -> model chạy ra 0 -> câm.
        Lưu dạng {-m,0,+m}: kernel sign+max khôi phục ĐÚNG ternary t và scale=m,
        tái tạo chính xác weight lúc train = t*m."""
        wf = w.detach().float()
        s = 1.0 / wf.abs().mean().clamp(min=1e-5)
        return (wf * s).round().clamp(-1, 1) / s

    def tlin(t):  # linear weights: áp weight_quant TRƯỚC, rồi F16 nếu --f16
        q = weight_quant_ternary(t)
        return q.to(torch.float16).numpy() if args.f16 else q.numpy()

    w.add_tensor("token_embd.weight", t32(model.embed.weight))
    w.add_tensor("output_norm.weight", t32(model.output_norm.weight))
    for i, blk in enumerate(model.blocks):
        a, f = blk.attn, blk.ffn
        pref = f"blk.{i}."
        w.add_tensor(pref + "attn_norm.weight", t32(a.attn_norm.weight))
        w.add_tensor(pref + "attn_q.weight", tlin(a.wq.weight))
        w.add_tensor(pref + "attn_k.weight", tlin(a.wk.weight))
        w.add_tensor(pref + "attn_v.weight", tlin(a.wv.weight))
        w.add_tensor(pref + "attn_sub_norm.weight", t32(a.attn_sub_norm.weight))
        w.add_tensor(pref + "attn_output.weight", tlin(a.wo.weight))
        w.add_tensor(pref + "ffn_norm.weight", t32(f.ffn_norm.weight))
        w.add_tensor(pref + "ffn_gate.weight", tlin(f.gate.weight))
        w.add_tensor(pref + "ffn_up.weight", tlin(f.up.weight))
        w.add_tensor(pref + "ffn_sub_norm.weight", t32(f.ffn_sub_norm.weight))
        w.add_tensor(pref + "ffn_down.weight", tlin(f.down.weight))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
