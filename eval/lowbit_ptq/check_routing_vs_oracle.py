# -*- coding: utf-8 -*-
"""Bo sung dieu tra cho RESEARCH_TQ33_OUTLIER_FIX.md Giai doan 2 — cau hoi quan trong: khi
rel-err hidden-state O LAYER SAU (33+) TANG sau khi fix, day co phai vi routing (chon expert)
cua BAN FIXED lech XA oracle HON ban UNFIXED khong, hay von di CA HAI da lech oracle o muc do
tuong duong (vi router nhay cam voi nhieu so hoc o cac expert gan-bang-nhau, DAC DIEM VON CO
cua he thong, KHONG PHAI do fix gay ra)?

Tinh oracle rieng (dung y het compute_oracle() trong validate_30b_layers_compare.py) nhung
GIU LAI ca idx routing tung layer, roi so voi idx cua CA dump truoc va SAU fix."""
import json
import sys
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
from safe_ckpt_reader import open_reader  # noqa: E402
from moe_common_py import moe_softmax, moe_topk_renorm  # noqa: E402

CKPT = "D:/Bit-Translate-data/pack_local/pytorch_model.bin"
TOKENS_PATH = "D:/Bit-Translate-data/tq33_30b/runner/oracle/tokens.bin"
N_LAYER = 48
HIDDEN = 2048
N_HEAD = 32
N_KV_HEAD = 4
HEAD_DIM = 128
N_ACTIVE = 8
ROPE_THETA = 1_000_000.0
RMS_EPS = 1e-6


def rmsnorm(x, w, eps=RMS_EPS):
    ss = x.pow(2).mean(dim=-1, keepdim=True)
    return w * (x * torch.rsqrt(ss + eps))


def rope_neox(x, pos, dim=HEAD_DIM, theta_base=ROPE_THETA):
    half = dim // 2
    j = torch.arange(half, dtype=torch.float32)
    freqs = theta_base ** (-2.0 * j / dim)
    angles = pos.float().unsqueeze(-1) * freqs.unsqueeze(0)
    cos, sin = torch.cos(angles).unsqueeze(1), torch.sin(angles).unsqueeze(1)
    x0, x1 = x[..., :half], x[..., half:]
    out = torch.empty_like(x)
    out[..., :half] = x0 * cos - x1 * sin
    out[..., half:] = x0 * sin + x1 * cos
    return out


def silu(x):
    return x * torch.sigmoid(x)


def main():
    prefix, tensors_meta, extra, get, zf = open_reader(CKPT)
    with open("D:/Bit-Translate-data/tq33_30b/corrupted_zeroed.json", encoding="utf-8") as f:
        corrupted_set = set(json.load(f)["tensors"])

    def get_z(name):
        t = get(name).float()
        return torch.zeros_like(t) if name in corrupted_set else t

    with open(TOKENS_PATH, "rb") as f:
        seq_len = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        tokens = np.frombuffer(f.read(4 * seq_len), dtype=np.int32).tolist()
    T = seq_len
    embed_w = get("model.embed_tokens.weight").float()
    hidden = embed_w[tokens]
    expert_cache = {}

    def get_expert(layer, e, kind):
        key = (layer, e, kind)
        if key not in expert_cache:
            expert_cache[key] = get_z(f"model.layers.{layer}.mlp.experts.{e}.{kind}_proj.weight")
        return expert_cache[key]

    oracle_idx = []
    pos = torch.arange(T, dtype=torch.float32)
    t0 = time.time()
    for l in range(N_LAYER):
        if l % 12 == 0:
            print(f"oracle layer {l} ({time.time()-t0:.0f}s)")
        ln_in = get(f"model.layers.{l}.input_layernorm.weight").float()
        ln_post = get(f"model.layers.{l}.post_attention_layernorm.weight").float()
        wq, wk, wv, wo = (get_z(f"model.layers.{l}.self_attn.{n}_proj.weight") for n in "qkvo")
        qn = get(f"model.layers.{l}.self_attn.q_norm.weight").float()
        kn = get(f"model.layers.{l}.self_attn.k_norm.weight").float()
        router_w = get(f"model.layers.{l}.mlp.gate.weight").float()

        cur = rmsnorm(hidden, ln_in)
        q = (cur @ wq.T).view(T, N_HEAD, HEAD_DIM)
        k = (cur @ wk.T).view(T, N_KV_HEAD, HEAD_DIM)
        v = (cur @ wv.T).view(T, N_KV_HEAD, HEAD_DIM)
        q, k = rmsnorm(q, qn), rmsnorm(k, kn)
        q, k = rope_neox(q, pos), rope_neox(k, pos)
        rep = N_HEAD // N_KV_HEAD
        k_rep, v_rep = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        scale = 1.0 / (HEAD_DIM ** 0.5)
        scores = torch.einsum("ihd,jhd->hij", q, k_rep) * scale
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn_w = F.softmax(scores, dim=-1)
        attn_out = torch.einsum("hij,jhd->ihd", attn_w, v_rep).reshape(T, N_HEAD * HEAD_DIM)
        o_out = attn_out @ wo.T
        ffn_inp = hidden + o_out
        cur2 = rmsnorm(ffn_inp, ln_post)
        router_logits = cur2 @ router_w.T
        probs = moe_softmax(router_logits)
        idx, weight = moe_topk_renorm(probs, N_ACTIVE)
        oracle_idx.append(idx.numpy().astype(np.int32))

        moe_out = torch.zeros(T, HIDDEN)
        for t in range(T):
            for j in range(N_ACTIVE):
                e = int(idx[t, j].item())
                wg, wu, wd = get_expert(l, e, "gate"), get_expert(l, e, "up"), get_expert(l, e, "down")
                h = silu(cur2[t] @ wg.T) * (cur2[t] @ wu.T)
                moe_out[t] += weight[t, j].item() * (h @ wd.T)
        hidden = ffn_inp + moe_out
    zf.close()

    def load_idx(path):
        with open(path, "rb") as f:
            hdr = np.frombuffer(f.read(16), dtype=np.int32)
            pl = hdr[0]
            out = []
            for l in range(N_LAYER):
                f.read(4 * pl * HIDDEN)
                idx = np.frombuffer(f.read(4 * pl * N_ACTIVE), dtype=np.int32).reshape(pl, N_ACTIVE)
                f.read(4 * pl * N_ACTIVE)
                out.append(idx)
            return out

    before = load_idx(r"D:\Bit-Translate-data\tq33_30b\runner\oracle\runner_dump_layers.bin")
    after = load_idx(r"D:\Bit-Translate-data\tq33_30b\runner\oracle\runner_dump_layers_fixed.bin")

    print(f"\n{'layer':>5} {'#pos lech(TRUOC vs oracle)':>28} {'#pos lech(SAU vs oracle)':>26}")
    tot_before = tot_after = 0
    for l in range(N_LAYER):
        nb = sum(1 for t in range(T) if set(before[l][t].tolist()) != set(oracle_idx[l][t].tolist()))
        na = sum(1 for t in range(T) if set(after[l][t].tolist()) != set(oracle_idx[l][t].tolist()))
        tot_before += nb
        tot_after += na
        marker = ""
        if na > nb:
            marker = "  SAU LECH NHIEU HON"
        elif na < nb:
            marker = "  SAU LECH IT HON"
        print(f"{l:5d} {nb:28d} {na:26d}{marker}")
    print(f"\nTONG so (layer,vi tri) lech tap-expert vs oracle: TRUOC={tot_before}/{N_LAYER*T}  "
          f"SAU={tot_after}/{N_LAYER*T}")


if __name__ == "__main__":
    main()
