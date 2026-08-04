# -*- coding: utf-8 -*-
"""RESEARCH_TQ33_OUTLIER_FIX.md Giai doan 2.3 — so sanh rel err TRUOC/SAU outlier-channel fix,
CUNG 1 oracle F32 doc lap (tinh 1 LAN, tranh chay oracle 2 lan ton ~30s x2).

Day la BAN SAO cua validate_30b_layers.py (oracle giu NGUYEN VAN — file goc KHONG bi sua, van
con de chay doc lap neu can) — CHI khac phan cuoi: doc CA HAI dump (truoc = runner_dump_layers.bin
tu runner CHUA sua, sau = runner_dump_layers_fixed.bin tu runner DA sua outlier-channel isolation)
va in bang so sanh rel-err tung layer canh nhau, thay vi chi so 1 dump.

Ky vong (theo nhiem vu): rel err GIAM RO RET o cac layer co outlier manh (dac biet layer cuoi,
layer 47, bao cao truoc do thay 7.83e-2), KHONG duoc lam sai lech them o layer khac. Neu KHONG
giam hoac te hon, day la KET QUA HOP LE (khong phai loi) — bao cao trung thuc."""
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
DUMP_BEFORE = "D:/Bit-Translate-data/tq33_30b/runner/oracle/runner_dump_layers.bin"
DUMP_AFTER = "D:/Bit-Translate-data/tq33_30b/runner/oracle/runner_dump_layers_fixed.bin"
TOKENS_PATH = "D:/Bit-Translate-data/tq33_30b/runner/oracle/tokens.bin"

N_LAYER_VALIDATE = 48
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
    ss = x.pow(2).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(ss + eps)
    return w * (x * inv)


def rope_neox(x, pos, dim=HEAD_DIM, theta_base=ROPE_THETA):
    half = dim // 2
    j = torch.arange(half, dtype=torch.float32)
    freqs = theta_base ** (-2.0 * j / dim)
    angles = pos.float().unsqueeze(-1) * freqs.unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(1)
    sin = torch.sin(angles).unsqueeze(1)
    x0 = x[..., :half]
    x1 = x[..., half:]
    out = torch.empty_like(x)
    out[..., :half] = x0 * cos - x1 * sin
    out[..., half:] = x0 * sin + x1 * cos
    return out


def silu(x):
    return x * torch.sigmoid(x)


def compute_oracle():
    prefix, tensors_meta, extra, get, zf = open_reader(CKPT)
    print(f"ckpt: {len(tensors_meta)} tensor")

    with open("D:/Bit-Translate-data/tq33_30b/corrupted_zeroed.json", encoding="utf-8") as f:
        corrupted_set = set(json.load(f)["tensors"])
    print(f"corrupted_zeroed.json: {len(corrupted_set)} tensor se bi EP VE ZERO (khop runner)")

    def get_z(name):
        t = get(name).float()
        if name in corrupted_set:
            t = torch.zeros_like(t)
        return t

    with open(TOKENS_PATH, "rb") as f:
        seq_len = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        tokens = np.frombuffer(f.read(4 * seq_len), dtype=np.int32).tolist()
    print(f"tokens ({seq_len}): {tokens}")
    T = seq_len

    print("doc embed_tokens (622MB, 1 lan)...")
    embed_w = get("model.embed_tokens.weight").float()
    hidden = embed_w[tokens]

    expert_cache = {}

    def get_expert(layer, e, kind):
        key = (layer, e, kind)
        if key not in expert_cache:
            name = f"model.layers.{layer}.mlp.experts.{e}.{kind}_proj.weight"
            expert_cache[key] = get_z(name)
        return expert_cache[key]

    hidden_after = []
    pos = torch.arange(T, dtype=torch.float32)
    t_start = time.time()

    for l in range(N_LAYER_VALIDATE):
        if l % 8 == 0:
            print(f"  oracle layer {l} ({time.time()-t_start:.0f}s)...")
        ln_in = get(f"model.layers.{l}.input_layernorm.weight").float()
        ln_post = get(f"model.layers.{l}.post_attention_layernorm.weight").float()
        wq = get_z(f"model.layers.{l}.self_attn.q_proj.weight")
        wk = get_z(f"model.layers.{l}.self_attn.k_proj.weight")
        wv = get_z(f"model.layers.{l}.self_attn.v_proj.weight")
        wo = get_z(f"model.layers.{l}.self_attn.o_proj.weight")
        qn = get(f"model.layers.{l}.self_attn.q_norm.weight").float()
        kn = get(f"model.layers.{l}.self_attn.k_norm.weight").float()
        router_w = get(f"model.layers.{l}.mlp.gate.weight").float()

        cur = rmsnorm(hidden, ln_in)
        q = (cur @ wq.T).view(T, N_HEAD, HEAD_DIM)
        k = (cur @ wk.T).view(T, N_KV_HEAD, HEAD_DIM)
        v = (cur @ wv.T).view(T, N_KV_HEAD, HEAD_DIM)

        q = rmsnorm(q, qn)
        k = rmsnorm(k, kn)
        q = rope_neox(q, pos)
        k = rope_neox(k, pos)

        rep = N_HEAD // N_KV_HEAD
        k_rep = k.repeat_interleave(rep, dim=1)
        v_rep = v.repeat_interleave(rep, dim=1)

        scale = 1.0 / (HEAD_DIM ** 0.5)
        scores = torch.einsum("ihd,jhd->hij", q, k_rep) * scale
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_w = F.softmax(scores, dim=-1)
        attn_out = torch.einsum("hij,jhd->ihd", attn_w, v_rep)
        attn_concat = attn_out.reshape(T, N_HEAD * HEAD_DIM)

        o_out = attn_concat @ wo.T
        ffn_inp = hidden + o_out

        cur2 = rmsnorm(ffn_inp, ln_post)

        router_logits = cur2 @ router_w.T
        probs = moe_softmax(router_logits)
        idx, weight = moe_topk_renorm(probs, N_ACTIVE)

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

    zf.close()
    return T, hidden_after


def load_dump(path, T):
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(16), dtype=np.int32)
        r_prompt_len, r_ndump, r_hidden, r_nactive = hdr.tolist()
        assert r_prompt_len == T, f"prompt_len dump={r_prompt_len} vs oracle T={T}"
        assert r_ndump == N_LAYER_VALIDATE and r_hidden == HIDDEN and r_nactive == N_ACTIVE
        per_layer_hidden = []
        for l in range(N_LAYER_VALIDATE):
            r_hidden_arr = np.frombuffer(f.read(4 * T * HIDDEN), dtype=np.float32).reshape(T, HIDDEN)
            f.read(4 * T * N_ACTIVE)  # idx (khong dung o day, routing KHONG doi giua fixed/unfixed)
            f.read(4 * T * N_ACTIVE)  # weight
            per_layer_hidden.append(r_hidden_arr)
        return per_layer_hidden


def rel_err_per_layer(dump_hidden, oracle_hidden, T):
    out = []
    for l in range(N_LAYER_VALIDATE):
        o_hidden = oracle_hidden[l].numpy()
        r_hidden_arr = dump_hidden[l]
        err = np.sum((r_hidden_arr.astype(np.float64) - o_hidden.astype(np.float64)) ** 2)
        ref = np.sum(o_hidden.astype(np.float64) ** 2)
        rel = (err / (ref + 1e-30)) ** 0.5
        maxabs = float(np.max(np.abs(r_hidden_arr - o_hidden)))
        out.append((rel, maxabs))
    return out


def main():
    print("=== Tinh oracle F32 doc lap (1 LAN, dung cho ca 2 so sanh truoc/sau) ===")
    T, oracle_hidden = compute_oracle()

    print(f"\n=== Doc dump TRUOC fix: {DUMP_BEFORE} ===")
    dump_before = load_dump(DUMP_BEFORE, T)
    print(f"=== Doc dump SAU fix:   {DUMP_AFTER} ===")
    dump_after = load_dump(DUMP_AFTER, T)

    err_before = rel_err_per_layer(dump_before, oracle_hidden, T)
    err_after = rel_err_per_layer(dump_after, oracle_hidden, T)

    print(f"\n{'='*88}")
    print(f"{'layer':>5} | {'rel_err TRUOC':>14} {'maxabs TRUOC':>13} | "
          f"{'rel_err SAU':>12} {'maxabs SAU':>11} | {'ty le sau/truoc':>16}")
    print(f"{'='*88}")
    for l in range(N_LAYER_VALIDATE):
        rb, mb = err_before[l]
        ra, ma = err_after[l]
        ratio = ra / rb if rb > 1e-30 else float("nan")
        marker = "  <-- GIAM" if ratio < 0.95 else ("  ++ TANG" if ratio > 1.05 else "")
        print(f"{l:5d} | {rb:14.4e} {mb:13.4f} | {ra:12.4e} {ma:11.4f} | {ratio:14.4f}x{marker}")

    print(f"\n{'='*88}\nTOM TAT\n{'='*88}")
    rb_last, _ = err_before[-1]
    ra_last, _ = err_after[-1]
    print(f"Layer cuoi (47): rel_err TRUOC={rb_last:.4e}  SAU={ra_last:.4e}  "
          f"ty le={ra_last/rb_last:.4f}x  ({'GIAM' if ra_last < rb_last else 'TANG/khong doi'})")
    print(f"rel_err logits-tuong-duong (dung hidden lop cuoi lam proxy) TRUOC={rb_last:.4e} SAU={ra_last:.4e}")

    n_improved = sum(1 for l in range(N_LAYER_VALIDATE) if err_after[l][0] < err_before[l][0] * 0.95)
    n_worse = sum(1 for l in range(N_LAYER_VALIDATE) if err_after[l][0] > err_before[l][0] * 1.05)
    n_same = N_LAYER_VALIDATE - n_improved - n_worse
    print(f"So layer rel_err GIAM >=5%: {n_improved}/{N_LAYER_VALIDATE}")
    print(f"So layer rel_err TANG >=5%: {n_worse}/{N_LAYER_VALIDATE}")
    print(f"So layer khong doi ro ret: {n_same}/{N_LAYER_VALIDATE}")

    # ghi ra json de report doc lai de dua vao bang markdown
    out = {
        "T": T,
        "layers": [
            {"layer": l, "rel_err_before": err_before[l][0], "maxabs_before": err_before[l][1],
             "rel_err_after": err_after[l][0], "maxabs_after": err_after[l][1]}
            for l in range(N_LAYER_VALIDATE)
        ],
    }
    out_path = "D:/Bit-Translate-data/tq33_30b/genqa/rel_err_compare.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nda ghi so lieu day du -> {out_path}")


if __name__ == "__main__":
    main()
