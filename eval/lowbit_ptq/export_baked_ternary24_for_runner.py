# -*- coding: utf-8 -*-
r"""Đóng gói checkpoint ternary N:M 2:4 vừa bake thật (exp_ap_bake_ternary24_geo6.py) thành
đúng 3 thư mục mà qwen3_runner_tq33_fast.exe cần, TÁI DÙNG NGUYÊN logic encode đã verify
lossless trước đó (export_full_model_f32.py / export_tq33_all_linears.py / export_embed_int8.py)
— chỉ đổi nguồn đọc từ ckpt QAT gen4_n4.pt sang ckpt PTQ ternary24_seq_g64_baked.pt.

Output D:\Bit-Translate-data\tq33_runner_ptq24\:
  weights_f32\   -- dump f32 phẳng toàn bộ tensor (giống export_full_model_f32.py)
  tq33_packed\   -- 196 linear đóng gói TQ33 block64 (giống export_tq33_all_linears.py),
                    VERIFY LOSSLESS từng tensor trước khi ghi
  embed_int8\    -- embed_tokens int8 per-row (giống export_embed_int8.py, KHÔNG validate
                    qua oracle cũ — oracle đó thuộc model khác; validate bằng rel-err tái tạo)
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

CKPT = r"D:\Bit-Translate-data\qat_ckpts\ternary24_seq_g64_baked.pt"
OUT_ROOT = r"D:\Bit-Translate-data\tq33_runner_ptq24"
G_SCALE = 64
GROUP4 = 4
PROJS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ============================== 1) weights_f32 ==============================

def export_weights_f32(state, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    keys = sorted(state.keys())
    bin_path = os.path.join(out_dir, "weights.bin")
    index, offset_elems, n_bytes = [], 0, 0
    with open(bin_path, "wb") as f:
        for k in keys:
            t = state[k]
            if not torch.is_tensor(t):
                continue
            arr = t.detach().float().contiguous().numpy().astype("<f4")
            f.write(arr.tobytes())
            index.append({"name": k, "shape": list(t.shape), "offset_f32": offset_elems,
                           "numel": arr.size})
            offset_elems += arr.size
            n_bytes += arr.nbytes
    meta = {"ckpt": CKPT, "n_tensors": len(index), "total_f32_elems": offset_elems,
            "total_bytes": n_bytes,
            "config": {"n_layer": 28, "hidden_size": 1024, "n_head": 16, "n_kv_head": 8,
                       "head_dim": 128, "intermediate_size": 3072, "vocab_size": 151936,
                       "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
                       "tie_word_embeddings_config": True},
            "output_head_tensor": "model.embed_tokens.weight",
            "tensors": index}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(out_dir, "meta_index.txt"), "w", encoding="utf-8") as f:
        f.write(f"{len(index)}\n")
        for e in index:
            dims = " ".join(str(d) for d in e["shape"])
            f.write(f"{e['name']} {len(e['shape'])} {dims} {e['offset_f32']} {e['numel']}\n")
    log(f"weights_f32: {n_bytes/1e9:.3f} GB, {len(index)} tensor -> {out_dir}")
    return meta


# ============================== 2) TQ33 pack (196 linear) ==============================

def build_patterns():
    pats = []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for c in (-1, 0, 1):
                for d in (-1, 0, 1):
                    v = (a, b, c, d)
                    if sum(1 for x in v if x != 0) <= 2:
                        pats.append(v)
    assert len(pats) == 33
    return pats


PATS = build_patterns()
_LUT_B3 = torch.full((81,), -1, dtype=torch.int64)
for _pid, _p in enumerate(PATS):
    _b3 = sum((x + 1) * m for x, m in zip(_p, (27, 9, 3, 1)))
    _LUT_B3[_b3] = _pid
assert (_LUT_B3 >= 0).sum().item() == 33


def sanitize_24(t, Wv):
    nz = (t != 0).sum(dim=-1)
    bad = (nz > 2).nonzero(as_tuple=False).flatten()
    n_bad = bad.shape[0]
    if n_bad == 0:
        return t, 0
    t = t.clone()
    for idx in bad.tolist():
        mags = Wv[idx].abs()
        order = torch.argsort(mags, descending=True)
        keep = set(order[:2].tolist())
        for i in range(4):
            if i not in keep:
                t[idx, i] = 0
    return t, n_bad


def encode_tensor(W):
    R, C = W.shape
    assert C % G_SCALE == 0, f"C={C} khong chia het {G_SCALE}"
    nb = C // G_SCALE
    Wv = W.view(R, nb, G_SCALE)
    s = Wv.abs().amax(dim=2)
    s_exp = s.unsqueeze(2)
    t = torch.where(s_exp > 0, torch.round(Wv / s_exp.clamp(min=1e-12)),
                     torch.zeros_like(Wv)).clamp(-1, 1).to(torch.int8)

    tg = t.view(R, nb, G_SCALE // GROUP4, GROUP4)
    Wvg = Wv.view(R, nb, G_SCALE // GROUP4, GROUP4)
    tg_flat = tg.reshape(R * nb * (G_SCALE // GROUP4), GROUP4)
    Wvg_flat = Wvg.reshape(R * nb * (G_SCALE // GROUP4), GROUP4)
    tg_flat, n_bad = sanitize_24(tg_flat, Wvg_flat)
    tg = tg_flat.view(R, nb, G_SCALE // GROUP4, GROUP4)
    t = tg.view(R, nb, G_SCALE)

    base3 = ((tg.to(torch.int64) + 1) * torch.tensor([27, 9, 3, 1], dtype=torch.int64)).sum(-1)
    gid = _LUT_B3[base3]
    assert int((gid < 0).sum().item()) == 0, "pattern ngoai bang 33 sau sanitize!"

    gid_pairs = gid.view(R, nb, 8, 2)
    code = gid_pairs[..., 0] * 33 + gid_pairs[..., 1]
    assert int(code.max().item()) < 2048

    code_np = code.numpy().astype(object)
    acc = np.zeros((R, nb), dtype=object)
    for k in range(8):
        acc = acc | (code_np[:, :, k] << (11 * k))
    flat_acc = acc.reshape(-1)
    byte_blob = b"".join(int(v).to_bytes(11, "little") for v in flat_acc.tolist())
    code_bytes = np.frombuffer(byte_blob, dtype=np.uint8).reshape(R, nb, 11)

    s_np = s.numpy().astype(np.float32)
    uniq = np.unique(s_np)
    n_uniq = uniq.size
    assert n_uniq <= 256, f"tensor co {n_uniq} gia tri scale duy nhat > 256"
    table = np.zeros(256, dtype=np.float32)
    table[:n_uniq] = uniq
    val_to_idx = {float(v): i for i, v in enumerate(uniq.tolist())}
    sidx = np.vectorize(lambda v: val_to_idx[float(v)])(s_np).astype(np.uint8)

    packed = np.concatenate([code_bytes, sidx[:, :, None]], axis=2)
    return packed, table, n_uniq, n_bad, t, s


def verify_roundtrip(packed, table, t_ref, s_ref, R, C, nb):
    codes11 = packed[:, :, :11]
    sidx = packed[:, :, 11]
    flat = codes11.reshape(R * nb, 11)
    big = np.zeros(R * nb, dtype=object)
    for i in range(11):
        big += flat[:, i].astype(object) << (8 * i)
    codes = np.zeros((R * nb, 8), dtype=np.int64)
    for k in range(8):
        codes[:, k] = (big >> (11 * k)) & 0x7FF
    codes = codes.reshape(R, nb, 8)
    id0 = codes // 33
    id1 = codes % 33
    pats_t = torch.tensor(PATS, dtype=torch.int8)
    g0 = pats_t[torch.from_numpy(id0)]
    g1 = pats_t[torch.from_numpy(id1)]
    t_dec = torch.stack([g0, g1], dim=3).reshape(R, nb, 16, 4).reshape(R, nb, G_SCALE)
    table_t = torch.from_numpy(table)
    s_dec = table_t[torch.from_numpy(sidx.astype(np.int64))]
    W_dec = t_dec.float() * s_dec.unsqueeze(-1)
    W_ref = t_ref.float() * s_ref.unsqueeze(-1)
    return torch.equal(W_dec, W_ref)


def export_tq33_packed(state, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    index, byte_off, cb_off = [], 0, 0
    all_packed, all_tables = [], []
    t0 = time.time()
    total_bad = 0
    total_werr_num, total_werr_den = 0.0, 0.0
    for layer in range(28):
        for p in PROJS:
            key = f"model.layers.{layer}.{p}.weight"
            W = state[key].float()
            R, C = W.shape
            packed, table, n_uniq, n_bad, t_ref, s_ref = encode_tensor(W)
            nb = C // G_SCALE
            ok = verify_roundtrip(packed, table, t_ref, s_ref, R, C, nb)
            total_bad += n_bad
            if not ok:
                log(f"!!! ROUNDTRIP THAT BAI tai {key} — dung lai")
                sys.exit(1)
            W_dec = t_ref.float() * s_ref.unsqueeze(-1)
            total_werr_num += float((W_dec.view(R, C) - W).pow(2).sum())
            total_werr_den += float(W.pow(2).sum())
            index.append({"name": key, "R": R, "C": C, "nb": nb,
                          "byte_offset": byte_off, "cb_offset": cb_off})
            all_packed.append(packed.tobytes())
            all_tables.append(table.tobytes())
            byte_off += R * nb * 12
            cb_off += 256
    log(f"({time.time()-t0:.0f}s) tổng {total_bad} nhóm sanitize / 196 tensor; "
        f"werr pack-vs-baked = {(total_werr_num/max(total_werr_den,1e-12))**0.5*100:.4f}% "
        f"(kỳ vọng ~0 nếu bake đúng group=64)")

    with open(os.path.join(out_dir, "tq33_packed.bin"), "wb") as f:
        for b in all_packed:
            f.write(b)
    with open(os.path.join(out_dir, "tq33_codebooks.bin"), "wb") as f:
        for b in all_tables:
            f.write(b)
    with open(os.path.join(out_dir, "tq33_meta_index.txt"), "w", encoding="utf-8") as f:
        f.write(f"{len(index)}\n")
        for e in index:
            f.write(f"{e['name']} {e['R']} {e['C']} {e['nb']} {e['byte_offset']} {e['cb_offset']}\n")

    total_orig_elems = sum(e["R"] * e["C"] for e in index)
    log(f"tổng packed linear: {byte_off/1e6:.2f} MB ({byte_off*8/total_orig_elems:.3f} bpw thực tế "
        f"cho {total_orig_elems} trọng số) -> {out_dir}")
    return byte_off, total_orig_elems


# ============================== 3) embed_int8 ==============================

def export_embed_int8(state, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    W = state["model.embed_tokens.weight"].float().numpy()
    rows, cols = W.shape
    CHUNK = 16384
    Q = np.empty((rows, cols), dtype=np.int8)
    scales = np.empty(rows, dtype=np.float32)
    recon_rel = np.empty(rows, dtype=np.float64)
    for r0 in range(0, rows, CHUNK):
        r1 = min(r0 + CHUNK, rows)
        Wb = W[r0:r1]
        am = np.abs(Wb).max(axis=1)
        sc = np.where(am > 0, am / 127.0, 1.0).astype(np.float32)
        Qb = np.clip(np.rint(Wb / sc[:, None]), -127, 127).astype(np.int8)
        Q[r0:r1] = Qb
        scales[r0:r1] = sc
        rec = Qb.astype(np.float32) * sc[:, None]
        num = np.sqrt(np.sum((rec - Wb) ** 2, axis=1, dtype=np.float64))
        den = np.sqrt(np.sum(Wb.astype(np.float64) ** 2, axis=1)) + 1e-30
        recon_rel[r0:r1] = num / den
    log(f"embed int8 recon rel-L2: mean={recon_rel.mean():.4e} max={recon_rel.max():.4e}")
    Q.tofile(os.path.join(out_dir, "embed_int8.bin"))
    scales.tofile(os.path.join(out_dir, "embed_scales.bin"))
    with open(os.path.join(out_dir, "embed_int8_meta.txt"), "w") as f:
        f.write(f"{rows} {cols}\n")
    n_bytes = rows * cols + rows * 4
    log(f"embed_int8: {n_bytes/1e6:.2f} MB -> {out_dir}")
    return n_bytes


def inject_zero_bias(state):
    """Runner (viết cho ckpt QAT gen4_n4.pt có bias giả LearnQLinear) exit(1) nếu KHÔNG tìm
    thấy tensor .bias cho mỗi linear — model HF chuẩn (bake PTQ này) không có bias thật
    (Qwen3 bỏ bias, dùng QK-norm thay). Thêm bias=0 cho đủ tên runner cần, KHÔNG đổi toán học
    (cộng 0 = no-op) — chỉ để thỏa mãn giả định cấu trúc file của runner cũ."""
    added = 0
    for layer in range(28):
        for p in PROJS:
            key = f"model.layers.{layer}.{p}.bias"
            if key not in state:
                w_key = f"model.layers.{layer}.{p}.weight"
                out_dim = state[w_key].shape[0]
                state[key] = torch.zeros(out_dim, dtype=torch.float32)
                added += 1
    log(f"Đã thêm {added} tensor bias=0 (runner cần tên tensor này tồn tại, giá trị 0 = no-op)")
    return state


def main():
    log(f"Nạp checkpoint {CKPT}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    state = inject_zero_bias(state)
    log(f"meta: {ckpt.get('meta', {})}")

    meta = export_weights_f32(state, os.path.join(OUT_ROOT, "weights_f32"))
    packed_bytes, n_linear_weights = export_tq33_packed(state, os.path.join(OUT_ROOT, "tq33_packed"))
    embed_bytes = export_embed_int8(state, os.path.join(OUT_ROOT, "embed_int8"))

    # ---- tổng dung lượng thực tế nếu đóng gói thành 1 file ----
    extras_bytes = 0
    extras_names = []
    for e in meta["tensors"]:
        name = e["name"]
        if any(name == f"model.layers.{L}.{p}.weight" for L in range(28) for p in PROJS):
            continue  # đã trong tq33_packed
        if name == "model.embed_tokens.weight":
            continue  # thay bằng embed_int8 (đã tính riêng)
        if name == "lm_head.weight":
            continue  # KHÔNG dùng (tied, dùng embed_tokens cho output head)
        extras_bytes += int(np.prod(e["shape"])) * 2  # fp16 cho phần còn lại (norm/bias)
        extras_names.append(name)
    total_bytes = packed_bytes + embed_bytes + extras_bytes
    log(f"extras (norm+bias, fp16): {extras_bytes/1e6:.2f} MB, {len(extras_names)} tensor")
    log(f"\n=== TỔNG DUNG LƯỢNG THỰC TẾ (TQ33 linear + embed int8 + extras fp16) ===")
    log(f"  packed linear (196 tensor, 2:4 ternary): {packed_bytes/1e6:.2f} MB")
    log(f"  embed_tokens int8 (dùng chung input+output head): {embed_bytes/1e6:.2f} MB")
    log(f"  extras (norm/bias fp16): {extras_bytes/1e6:.2f} MB")
    log(f"  TỔNG: {total_bytes/1e6:.2f} MB")
    fp32_equiv = meta["total_bytes"]
    log(f"  so với FP32 gốc {fp32_equiv/1e6:.1f} MB: nén {fp32_equiv/total_bytes:.2f}x")

    with open(os.path.join(OUT_ROOT, "size_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"packed_linear_bytes": packed_bytes, "embed_int8_bytes": embed_bytes,
                   "extras_fp16_bytes": extras_bytes, "total_bytes": total_bytes,
                   "fp32_equiv_bytes": fp32_equiv,
                   "compression_ratio": fp32_equiv / total_bytes}, f, indent=2)


if __name__ == "__main__":
    main()
