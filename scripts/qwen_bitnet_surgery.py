#!/usr/bin/env python3
"""Phẫu thuật Qwen2.5-0.5B-Instruct -> kiến trúc BitNet (bước 1-2 của PLAN_TERNARY_QWEN.md).

Thay 168 nn.Linear (q/k/v/o/gate/up/down x 24 block) bằng BitLinearB (BitLinear của
dự án + GIỮ BIAS fp32 — Qwen q/k/v CÓ bias, BitNet chuẩn không; bias không quantize),
chèn attn_sub_norm/ffn_sub_norm khởi tạo identity, rồi smoke-test forward + ppl.

KHÔNG train ở đây — train loop QAT-KD viết trên máy GPU (xem PLAN §1.4-1.5).
Master weight giữ NGUYÊN giá trị FP của Qwen: phép chiếu ternary chỉ xảy ra lúc
forward (nấc C), tuyệt đối không ghi đè weight bằng bản đã ternary hoá (nấc A — đã
chứng minh sập, eval/ternary_lab/).

  HF_HOME=D:/Bit-Translate-data/hf_cache PYTHONUTF8=1 python scripts/qwen_bitnet_surgery.py \
      --out D:/Bit-Translate-data/ternary_lab/qwen_bitnet_init.pt [--device cuda]
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bitnet import BitLinear, RMSNorm  # noqa: E402

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class BitLinearB(BitLinear):
    """BitLinear + bias fp32 tách riêng (không quantize bias)."""

    def __init__(self, lin: nn.Linear):
        super().__init__(lin.in_features, lin.out_features)
        with torch.no_grad():
            self.weight.copy_(lin.weight)          # master = FP gốc, KHÔNG ternary hoá
        if lin.bias is not None:
            self.bias_fp = nn.Parameter(lin.bias.detach().clone())
        else:
            self.bias_fp = None

    def forward(self, x, x_prequant=False):
        y = super().forward(x, x_prequant=x_prequant)
        return y if self.bias_fp is None else y + self.bias_fp


class SubNormed(nn.Module):
    """RMSNorm(identity-init) -> BitLinearB. Dùng cho o_proj và down_proj."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.sub_norm = RMSNorm(lin.in_features)   # weight khởi tạo = 1 (identity)
        self.proj = BitLinearB(lin)

    def forward(self, x):
        return self.proj(self.sub_norm(x))


def convert(model):
    n_lin = n_sub = 0
    for layer in model.model.layers:
        for holder, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj")),
                              (layer.mlp, ("gate_proj", "up_proj"))):
            for nm in names:
                setattr(holder, nm, BitLinearB(getattr(holder, nm)))
                n_lin += 1
        layer.self_attn.o_proj = SubNormed(layer.self_attn.o_proj)
        layer.mlp.down_proj = SubNormed(layer.mlp.down_proj)
        n_lin += 2
        n_sub += 2
    return n_lin, n_sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=None, help="lưu state_dict sau phẫu thuật")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    m.eval()
    tied = m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()
    print(f"{a.model}: {sum(p.numel() for p in m.parameters())/1e6:.1f}M | lm_head tied={tied}")

    n_lin, n_sub = convert(m)
    m.to(a.device)
    print(f"đã thay {n_lin} Linear -> BitLinearB, chèn {n_sub*24//2 if False else n_sub}x2... "
          f"({n_sub} SubNormed/layer-pair, tổng sub_norm = {n_sub})")

    # Smoke: forward + ppl 1 câu — KỲ VỌNG RÁC/ppl rất cao (chưa train, ternary bật).
    # Mục đích chỉ là: chạy không lỗi, shape đúng, loss hữu hạn.
    s = "来週の会議は資料が間に合わないので、日程を変更したいと思います。"
    ids = tok(s, return_tensors="pt").input_ids.to(a.device)
    t0 = time.time()
    with torch.no_grad():
        out = m(ids, labels=ids)
    print(f"smoke forward OK: loss={out.loss.item():.3f} "
          f"(ppl~{torch.exp(out.loss).item():.0f}; cao là ĐÚNG KỲ VỌNG trước QAT) "
          f"| {time.time()-t0:.1f}s | logits {tuple(out.logits.shape)}")
    assert torch.isfinite(out.loss), "loss không hữu hạn — kiểm sub_norm/activation quant"

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": m.state_dict(), "base": a.model,
                    "note": "master=FP goc, chua QAT"}, a.out)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
