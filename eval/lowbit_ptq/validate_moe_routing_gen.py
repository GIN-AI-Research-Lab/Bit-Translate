# -*- coding: utf-8 -*-
"""Giai doan A — sinh du lieu ngau nhien + oracle PyTorch cho thuat toan dieu phoi MoE
(softmax toan bo -> topk -> renormalize -> SwiGLU per-expert -> weighted sum), DUNG dung
kich thuoc routing that (n_expert=128, n_active=8) nhung hidden/moe_ffn nho (de nhanh — day
la test thuat toan dieu phoi, KHONG phai test kernel GEMV, kernel TQ33 da validate rieng roi
o 0.6B Giai doan 2).

Thuat toan oracle KHOP CHINH XAC llm_graph_context::build_moe_ffn (da doc truc tiep source
that, xem chu thich dau moe_common.h): softmax day du 128 expert -> topk 8 -> renormalize
weight -> silu(gate)*up moi expert -> tong co trong so. KHONG bias, KHONG clamp, KHONG scale
them (w_scale sentinel 0.0f = tat).

Output -> D:/Bit-Translate-data/tq33_30b/moe_validate/:
  meta.txt              "n_expert n_active hidden moe_ffn n_tokens"
  router_w.bin          float32 [n_expert, hidden]
  gate_w.bin            float32 [n_expert, moe_ffn, hidden]
  up_w.bin              float32 [n_expert, moe_ffn, hidden]
  down_w.bin            float32 [n_expert, hidden, moe_ffn]
  x.bin                 float32 [n_tokens, hidden]
  oracle_out.bin        float32 [n_tokens, hidden]
  oracle_idx.bin        int32   [n_tokens, n_active]   (thu tu GIAM DAN theo prob, nhu topk)
  oracle_weight.bin     float32 [n_tokens, n_active]
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

OUT = "D:/Bit-Translate-data/tq33_30b/moe_validate"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(1234)
np.random.seed(1234)

N_EXPERT = 128   # khop that (Qwen3-30B-A3B)
N_ACTIVE = 8     # khop that
HIDDEN = 32      # nho de nhanh — chi test THUAT TOAN dieu phoi, khong test GEMV lon
MOE_FFN = 48
N_TOKENS = 6     # vai token, seed khac nhau -> routing khac nhau moi token


def main():
    router_w = torch.randn(N_EXPERT, HIDDEN) * 0.3
    gate_w = torch.randn(N_EXPERT, MOE_FFN, HIDDEN) * 0.2
    up_w = torch.randn(N_EXPERT, MOE_FFN, HIDDEN) * 0.2
    down_w = torch.randn(N_EXPERT, HIDDEN, MOE_FFN) * 0.2
    x = torch.randn(N_TOKENS, HIDDEN) * 0.5

    # ---- oracle, dung DUNG thuat toan da xac nhan qua source (build_moe_ffn) ----
    logits = x @ router_w.T                      # [n_tokens, n_expert]
    probs = F.softmax(logits, dim=-1)             # softmax TREN TOAN BO 128 expert
    topv, topi = torch.topk(probs, k=N_ACTIVE, dim=-1)  # gia tri LON NHAT, GIAM DAN (mac dinh)
    wsum = topv.sum(dim=-1, keepdim=True).clamp(min=6.103515625e-5)
    weights = topv / wsum                          # renormalize (KHONG re-softmax)

    out = torch.zeros(N_TOKENS, HIDDEN)
    for t in range(N_TOKENS):
        for j in range(N_ACTIVE):
            e = int(topi[t, j].item())
            gate = x[t] @ gate_w[e].T               # [moe_ffn]
            up = x[t] @ up_w[e].T                    # [moe_ffn]
            h = F.silu(gate) * up                     # SwiGLU: silu(gate)*up
            o = h @ down_w[e].T                        # [hidden]
            out[t] += weights[t, j].item() * o

    # ---- ghi ra binary (little-endian, C-order, float32/int32) ----
    def w32(name, t):
        t.numpy().astype(np.float32).tofile(os.path.join(OUT, name))

    with open(os.path.join(OUT, "meta.txt"), "w") as f:
        f.write(f"{N_EXPERT} {N_ACTIVE} {HIDDEN} {MOE_FFN} {N_TOKENS}\n")
    w32("router_w.bin", router_w)
    w32("gate_w.bin", gate_w)
    w32("up_w.bin", up_w)
    w32("down_w.bin", down_w)
    w32("x.bin", x)
    w32("oracle_out.bin", out)
    topi.numpy().astype(np.int32).tofile(os.path.join(OUT, "oracle_idx.bin"))
    w32("oracle_weight.bin", weights)

    print(f"n_expert={N_EXPERT} n_active={N_ACTIVE} hidden={HIDDEN} moe_ffn={MOE_FFN} "
          f"n_tokens={N_TOKENS}")
    print("routing chon (topi) moi token:")
    for t in range(N_TOKENS):
        print(f"  token {t}: idx={topi[t].tolist()} weight={[round(w,4) for w in weights[t].tolist()]}")
    print(f"oracle_out[0][:5] = {out[0, :5].tolist()}")
    print(f"da ghi vao {OUT}")


if __name__ == "__main__":
    main()
