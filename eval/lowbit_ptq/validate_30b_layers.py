# -*- coding: utf-8 -*-
"""Giai doan B.4 — oracle DOC LAP cho layer 0 va layer 1 cua Qwen3-30B-A3B, dung
safe_ckpt_reader doc TRUC TIEP vai tensor can tu ckpt goc (KHONG load het 61GB — chi doc
embed_tokens (622MB, 1 lan) + layer 0/1 weights (norm/attn/router) + CAC EXPERT THUC SU
duoc chon boi routing THAT (dong, fetch theo nhu cau, khong doc het 128 expert/layer)).

Thuat toan giong HET qwen3moe_runner_tq33.c (RMSNorm -> Q/K/V proj -> per-head QK-norm ->
RoPE NEOX -> GQA causal attention -> o_proj -> residual -> RMSNorm -> router (softmax full
128 -> top8 -> renorm, moe_common.h da validate Giai doan A) -> SwiGLU per-expert -> weighted
sum -> residual), tat ca float32, KHONG bias, KHONG clamp.

So voi output cua runner (`runner_dump_layers.bin`, ghi boi qwen3moe_runner_tq33.c ham
write_dump() — dump hidden+routing sau layer 0 va layer 1, TAI VI TRI POS = 0..prompt_len-1,
dung KV-cache that giong luc benchmark, KHONG phai batch rieng).

Ky vong: idx routing KHOP TUYET DOI (router khong luong tu hoa, dung DUNG input x nhu nhau
o ca 2 ben) — bat ky lech idx nao la BUG THAT (khong phai nhieu luong tu hoa). rel err hidden
state ky vong o TAM CO BAC voi 0.6B Giai doan 2 layer dau (~1e-2, tu TQ33 quantization noise
cua 7 GEMV: q/k/v/o/gate/up/down moi layer).
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from safe_ckpt_reader import open_reader  # noqa: E402
from moe_common_py import moe_softmax, moe_topk_renorm  # noqa: E402

CKPT = "D:/Bit-Translate-data/pack_local/pytorch_model.bin"
DUMP_PATH = "D:/Bit-Translate-data/tq33_30b/runner/oracle/runner_dump_layers.bin"
TOKENS_PATH = "D:/Bit-Translate-data/tq33_30b/runner/oracle/tokens.bin"

N_LAYER_VALIDATE = 48  # FULL model — ha tang da san sang (moi layer code giong het, chi khac
                        # trong so), va 0.6B da lam full 28 layer nen 30B cung lam full 48
                        # layer de dam bao muc do nghiem ngat tuong duong (xem RESEARCH_TQ33_
                        # RUNNER.md muc 2.2/3.3 — day la tinh than "khong suy dien tu vai layer").
HIDDEN = 2048
N_HEAD = 32
N_KV_HEAD = 4
HEAD_DIM = 128
N_EXPERT = 128
N_ACTIVE = 8
MOE_FFN = 768
ROPE_THETA = 1_000_000.0
RMS_EPS = 1e-6


def rmsnorm(x, w, eps=RMS_EPS):
    # x: [..., n], w: [n] -- KHOP CONG THUC C: dung double cho tong binh phuong, nhung o day
    # torch float32 la du (sai so khong dang ke o muc so sanh nay).
    ss = x.pow(2).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(ss + eps)
    return w * (x * inv)


def rope_neox(x, pos, dim=HEAD_DIM, theta_base=ROPE_THETA):
    # x: [T, n_head, dim], pos: [T] -- KHOP dung cong thuc C rope_neox (cap (j, j+half), NEOX)
    half = dim // 2
    j = torch.arange(half, dtype=torch.float32)
    freqs = theta_base ** (-2.0 * j / dim)               # [half]
    angles = pos.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [T, half]
    cos = torch.cos(angles).unsqueeze(1)                  # [T,1,half]
    sin = torch.sin(angles).unsqueeze(1)
    x0 = x[..., :half]
    x1 = x[..., half:]
    out = torch.empty_like(x)
    out[..., :half] = x0 * cos - x1 * sin
    out[..., half:] = x0 * sin + x1 * cos
    return out


def silu(x):
    return x * torch.sigmoid(x)


def main():
    prefix, tensors_meta, extra, get, zf = open_reader(CKPT)
    print(f"ckpt: {len(tensors_meta)} tensor")

    # QUAN TRONG (bug bat duoc va sua): safe_ckpt_reader.get() KHONG tu zero-hoa tensor CRC
    # sai — no CHI canh bao qua get.crc_warned, van tra ve byte THAT DOC DUOC (co the la du
    # lieu sai/rac, KHONG phai zero — da kiem truc tiep: model.layers.24.self_attn.q_proj
    # tra ve absmax=0.103, 2.8M/8.4M phan tu khac 0, RO RANG khong phai zero-fill). Trong khi
    # do bulk_encode_tq33_30b.py (script tao du lieu cho runner) CO zero-hoa tuong minh 70
    # tensor nay truoc khi encode (xem corrupted_zeroed.json). Neu oracle KHONG lam giong,
    # so sanh se lech GIA (tuong nhu bug o runner nhung thuc ra la oracle dung sai du lieu).
    # Sua: nap corrupted_zeroed.json, ep zero DUNG 70 tensor nay truoc khi dung — khop CHINH
    # XAC du lieu ma runner da doc.
    with open("D:/Bit-Translate-data/tq33_30b/corrupted_zeroed.json", encoding="utf-8") as f:
        corrupted_set = set(json.load(f)["tensors"])
    print(f"corrupted_zeroed.json: {len(corrupted_set)} tensor se bi EP VE ZERO (khop runner)")

    def get_z(name):
        t = get(name).float()
        if name in corrupted_set:
            print(f"  [corrupted->zero] {name} (absmax truoc khi zero: {t.abs().max().item():.4f})")
            t = torch.zeros_like(t)
        return t

    with open(TOKENS_PATH, "rb") as f:
        seq_len = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        tokens = np.frombuffer(f.read(4 * seq_len), dtype=np.int32).tolist()
    print(f"tokens ({seq_len}): {tokens}")
    T = seq_len

    print("doc embed_tokens (622MB, 1 lan)...")
    embed_w = get("model.embed_tokens.weight").float()   # [151936, 2048]
    hidden = embed_w[tokens]                               # [T, 2048]
    print(f"embed lookup xong, hidden.shape={tuple(hidden.shape)}")

    expert_cache = {}

    def get_expert(layer, e, kind):
        key = (layer, e, kind)
        if key not in expert_cache:
            name = f"model.layers.{layer}.mlp.experts.{e}.{kind}_proj.weight"
            expert_cache[key] = get_z(name)
        return expert_cache[key]

    dump_idx_all = []
    dump_weight_all = []
    hidden_after = []

    pos = torch.arange(T, dtype=torch.float32)
    t_start = time.time()

    for l in range(N_LAYER_VALIDATE):
        if l < 2 or l >= N_LAYER_VALIDATE - 2 or l % 8 == 0:
            print(f"\n=== layer {l} ({time.time()-t_start:.0f}s) ===")
        ln_in = get(f"model.layers.{l}.input_layernorm.weight").float()
        ln_post = get(f"model.layers.{l}.post_attention_layernorm.weight").float()
        wq = get_z(f"model.layers.{l}.self_attn.q_proj.weight")   # [4096,2048]
        wk = get_z(f"model.layers.{l}.self_attn.k_proj.weight")   # [512,2048]
        wv = get_z(f"model.layers.{l}.self_attn.v_proj.weight")   # [512,2048]
        wo = get_z(f"model.layers.{l}.self_attn.o_proj.weight")   # [2048,4096]
        qn = get(f"model.layers.{l}.self_attn.q_norm.weight").float()   # [128]
        kn = get(f"model.layers.{l}.self_attn.k_norm.weight").float()   # [128]
        router_w = get(f"model.layers.{l}.mlp.gate.weight").float()     # [128,2048]

        cur = rmsnorm(hidden, ln_in)                     # [T,2048]
        q = (cur @ wq.T).view(T, N_HEAD, HEAD_DIM)         # [T,32,128]
        k = (cur @ wk.T).view(T, N_KV_HEAD, HEAD_DIM)      # [T,4,128]
        v = (cur @ wv.T).view(T, N_KV_HEAD, HEAD_DIM)      # [T,4,128]

        q = rmsnorm(q, qn)
        k = rmsnorm(k, kn)
        q = rope_neox(q, pos)
        k = rope_neox(k, pos)

        # GQA: q head h dung kv head h // (N_HEAD/N_KV_HEAD)
        rep = N_HEAD // N_KV_HEAD
        k_rep = k.repeat_interleave(rep, dim=1)   # [T,32,128]
        v_rep = v.repeat_interleave(rep, dim=1)   # [T,32,128]

        scale = 1.0 / (HEAD_DIM ** 0.5)
        # scores[h,i,j] = q[i,h].k[j,h] * scale, causal j<=i
        scores = torch.einsum("ihd,jhd->hij", q, k_rep) * scale   # [32,T,T]
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_w = F.softmax(scores, dim=-1)                          # [32,T,T]
        attn_out = torch.einsum("hij,jhd->ihd", attn_w, v_rep)      # [T,32,128]
        attn_concat = attn_out.reshape(T, N_HEAD * HEAD_DIM)         # [T,4096]

        o_out = attn_concat @ wo.T                                    # [T,2048]
        ffn_inp = hidden + o_out

        cur2 = rmsnorm(ffn_inp, ln_post)                              # [T,2048]

        router_logits = cur2 @ router_w.T                             # [T,128]
        probs = moe_softmax(router_logits)                            # [T,128]
        idx, weight = moe_topk_renorm(probs, N_ACTIVE)                # [T,8],[T,8]

        moe_out = torch.zeros(T, HIDDEN)
        for t in range(T):
            for j in range(N_ACTIVE):
                e = int(idx[t, j].item())
                wg = get_expert(l, e, "gate")
                wu = get_expert(l, e, "up")
                wd = get_expert(l, e, "down")
                gate = cur2[t] @ wg.T
                up = cur2[t] @ wu.T
                h = silu(gate) * up
                o = h @ wd.T
                moe_out[t] += weight[t, j].item() * o

        hidden = ffn_inp + moe_out
        hidden_after.append(hidden.clone())
        dump_idx_all.append(idx.numpy().astype(np.int32))
        dump_weight_all.append(weight.numpy().astype(np.float32))
        dt = time.time() - t_start
        n_unique_experts = len({k[1] for k in expert_cache if k[0] == l})
        print(f"layer {l} xong ({dt:.0f}s tich luy, {n_unique_experts} expert rieng biet duoc fetch o layer nay):")
        if l < 2 or l >= N_LAYER_VALIDATE - 2:
            for t in range(T):
                print(f"  pos {t}: idx={idx[t].tolist()} w={[round(x,4) for x in weight[t].tolist()]}")

    zf.close()

    # ---------- so voi dump cua runner C ----------
    print(f"\n=== SO SANH VOI RUNNER (doc {DUMP_PATH}) ===")
    with open(DUMP_PATH, "rb") as f:
        hdr = np.frombuffer(f.read(16), dtype=np.int32)
        r_prompt_len, r_ndump, r_hidden, r_nactive = hdr.tolist()
        assert r_prompt_len == T, f"prompt_len runner={r_prompt_len} vs oracle T={T}"
        assert r_ndump == N_LAYER_VALIDATE and r_hidden == HIDDEN and r_nactive == N_ACTIVE
        idx_mismatch_total = 0
        n_reorder_total = n_boundary_total = 0
        all_boundary_ranks = []
        rel_err_by_layer = []
        for l in range(N_LAYER_VALIDATE):
            r_hidden_arr = np.frombuffer(f.read(4 * T * HIDDEN), dtype=np.float32).reshape(T, HIDDEN)
            r_idx_arr = np.frombuffer(f.read(4 * T * N_ACTIVE), dtype=np.int32).reshape(T, N_ACTIVE)
            r_weight_arr = np.frombuffer(f.read(4 * T * N_ACTIVE), dtype=np.float32).reshape(T, N_ACTIVE)

            o_hidden = hidden_after[l].numpy()
            o_idx = dump_idx_all[l]
            o_weight = dump_weight_all[l]

            err = np.sum((r_hidden_arr.astype(np.float64) - o_hidden.astype(np.float64)) ** 2)
            ref = np.sum(o_hidden.astype(np.float64) ** 2)
            rel = (err / (ref + 1e-30)) ** 0.5
            maxabs = np.max(np.abs(r_hidden_arr - o_hidden))

            idx_mismatch = int(np.sum(r_idx_arr != o_idx))
            idx_mismatch_total += idx_mismatch
            werr = np.sum((r_weight_arr - o_weight) ** 2)
            wref = np.sum(o_weight ** 2)
            wrel = (werr / (wref + 1e-30)) ** 0.5

            # phan loai TUNG VI TRI lech: "reorder" (CUNG TAP 8 expert, chi khac thu tu —
            # anh huong duy nhat la trong so nao gan cho expert nao) vs "boundary-flip" (TAP
            # KHAC — 1 expert bi thay boi 1 expert khac o RANH GIOI chon/khong chon, thuong la
            # hang thap nhat = trong so nho nhat). Day la cau hoi quan trong: neu MOI lech chi
            # la reorder hoac boundary-flip o hang thap, day la nhieu so hoc lan tai gia tri
            # gan-bang-nhau (KHONG phai bug); neu lech xay ra o hang 1-2 (trong so lon), do la
            # dau hieu dang lo ngai hon nhieu.
            n_reorder = n_boundary = 0
            boundary_ranks = []
            for t in range(T):
                if np.array_equal(r_idx_arr[t], o_idx[t]):
                    continue
                set_r, set_o = set(r_idx_arr[t].tolist()), set(o_idx[t].tolist())
                if set_r == set_o:
                    n_reorder += 1
                else:
                    n_boundary += 1
                    only_r = set_r - set_o
                    for e in only_r:
                        boundary_ranks.append(int(np.where(r_idx_arr[t] == e)[0][0]))
            n_reorder_total += n_reorder
            n_boundary_total += n_boundary
            all_boundary_ranks.extend(boundary_ranks)
            rel_err_by_layer.append(rel)

            print(f"\nlayer {l}:")
            print(f"  routing idx mismatch: {idx_mismatch} / {T * N_ACTIVE} vi tri-expert lech "
                  f"({n_reorder} vi tri CUNG TAP 8-chi-khac-thu-tu, {n_boundary} vi tri KHAC TAP "
                  f"[hang bi doi: {sorted(boundary_ranks)}, dem tu 0])")
            print(f"  routing weight rel err: {wrel:.3e}")
            print(f"  hidden state rel err (L2): {rel:.3e}   max abs diff: {maxabs:.4f}")

    n_pos_total = N_LAYER_VALIDATE * T
    from collections import Counter
    rank_hist = Counter(all_boundary_ranks)
    print(f"\n=== KET LUAN (sau khi sua bug corrupted-tensor cua chinh oracle nay) ===")
    print(f"tong so (layer,vi tri) co it nhat 1 lech: {n_reorder_total + n_boundary_total} / {n_pos_total}")
    print(f"  - CUNG TAP 8 expert, chi doi thu tu (KHONG anh huong TAP duoc chon, chi doi trong "
          f"so giua 2 expert gan-bang-nhau): {n_reorder_total}")
    print(f"  - KHAC TAP (1 expert o RANH GIOI bi thay): {n_boundary_total}")
    print(f"  - phan bo HANG (rank, dem tu 0=trong so lon nhat) cua expert bi thay trong cac vi "
          f"tri KHAC TAP: {dict(sorted(rank_hist.items()))}")
    print(f"  (neu phan bo don gan het o hang 5,6,7 (3 hang trong so NHO NHAT trong top-8) thi "
          f"day la nhieu so hoc o RANH GIOI chon/khong-chon, nhu ky vong tu TQ33 quant noise — "
          f"neu co o hang 0-2 (trong so LON) thi la dau hieu dang lo ngai hon)")
    print(f"rel err hidden state theo layer: layer0={rel_err_by_layer[0]:.2e}  "
          f"layer{N_LAYER_VALIDATE//2}={rel_err_by_layer[N_LAYER_VALIDATE//2]:.2e}  "
          f"layer{N_LAYER_VALIDATE-1}={rel_err_by_layer[-1]:.2e}")


if __name__ == "__main__":
    main()
