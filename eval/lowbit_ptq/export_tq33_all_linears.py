# -*- coding: utf-8 -*-
r"""TQ33 Runner — Giai đoạn 2, bước 1: encode TOÀN BỘ 7 loại linear (q/k/v/o/gate/up/down)
x 28 layer = 196 tensor sang format TQ33 block64 (12 byte/64 trọng số = 1.5 bpw), mở rộng
exp_t24_export_bench.py (vốn chỉ làm 1 tensor) cho toàn model.

Format (khớp CHÍNH XÁC tq33_bench.c decode_block() — TÁI SỬ DỤNG y hệt, không đổi):
  block64 = 12 byte = [11 byte codes][1 byte scale-idx]
  codes = 8 codeword 11-bit LSB-first liền mạch (bit k*11..k*11+10 của stream 88-bit)
  codeword = id0*33 + id1  (id0,id1 = pattern-id của 2 nhóm-4 liên tiếp trong cặp)
  pattern-id: build_patterns() y hệt exp_t24_codec.py (thứ tự sinh a,b,c,d ∈{-1,0,1}³×⁴
  giữ nhóm ≤2 nonzero) — PHẢI khớp thứ tự build_tables() trong tq33_bench.c (cùng thuật
  toán sinh nested loop a ngoài, d trong, cùng range) để codeword id nhất quán 2 bên.
  scale-idx: index vào codebook ≤256 giá trị fp32 DUY NHẤT của tensor (per-tensor).

Xử lý vi phạm 2:4 hiếm gặp: đã phát hiện (inspect_ckpt2.py) 2 tensor có ĐÚNG 1 nhóm-4 vi
phạm (3 nonzero thay vì ≤2) trên tổng ~110M nhóm toàn model — model.layers.9.mlp.gate_proj
và model.layers.11.mlp.gate_proj. Xử lý: giữ 2 giá trị |W| lớn nhất trong nhóm, ép phần tử
nhỏ nhất về 0 (sanitize) trước khi encode — ảnh hưởng 4 trọng số / ~440M trọng số toàn
model, không đáng kể, nhưng nếu bỏ qua sẽ tạo codeword ngoài bảng 33 pattern (encode sai).

Output D:\Bit-Translate-data\tq33_runner\tq33_packed\:
  tq33_packed.bin     — concatenated [R,nb,12] byte cho từng tensor, theo thứ tự trong index
  tq33_codebooks.bin  — concatenated float32[256] codebook/tensor (pad 0 nếu <256 giá trị)
  tq33_meta_index.txt — "name R C nb byte_offset codebook_offset_floats" mỗi dòng (dòng đầu = count)
"""
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

CKPT = r"D:\Bit-Translate-data\qat_ckpts\qat_gen4_n4.pt"
OUT_DIR = r"D:\Bit-Translate-data\tq33_runner\tq33_packed"
G_SCALE = 64
GROUP = 4
PROJS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def build_patterns():
    """Y HET exp_t24_codec.py — PHẢI cùng thứ tự với tq33_bench.c build_tables()."""
    pats = []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for c in (-1, 0, 1):
                for d in (-1, 0, 1):
                    v = (a, b, c, d)
                    if sum(1 for x in v if x != 0) <= 2:
                        pats.append(v)
    assert len(pats) == 33, len(pats)
    return pats


PATS = build_patterns()
# lut_b3[base3] -> pattern id, base3 = sum((x+1)*mult) voi mult=(27,9,3,1).
# CHI 33/81 entry hop le (nhom <=2 nonzero) — 48 entry con lai (3-4 nonzero) DUNG la -1,
# se duoc assert o encode_tensor() rang DU LIEU THAT khong bao gio cham vao (sau sanitize).
_LUT_B3 = torch.full((81,), -1, dtype=torch.int64)
for _pid, _p in enumerate(PATS):
    _b3 = sum((x + 1) * m for x, m in zip(_p, (27, 9, 3, 1)))
    _LUT_B3[_b3] = _pid
assert (_LUT_B3 >= 0).sum().item() == 33, "so entry hop le phai dung 33"


def sanitize_24(t, Wv):
    """t, Wv: [N,4] (đã flatten toàn bộ nhóm-4 của tensor). Nếu nhóm nào >2 nonzero
    (hiếm — lỗi round-off ở biên khi 2 giá trị bake gần bằng s), giữ 2 |Wv| lớn nhất,
    ép phần còn lại về 0. Trả về t đã sửa + số nhóm bị sửa."""
    nz = (t != 0).sum(dim=-1)  # [N]
    bad = (nz > 2).nonzero(as_tuple=False).flatten()  # [K] chỉ số hàng vi phạm
    n_bad = bad.shape[0]
    if n_bad == 0:
        return t, 0
    t = t.clone()
    for idx in bad.tolist():
        mags = Wv[idx].abs()
        order = torch.argsort(mags, descending=True)  # idx giảm dần |W|
        keep = set(order[:2].tolist())
        for i in range(4):
            if i not in keep:
                t[idx, i] = 0
    return t, n_bad


def encode_tensor(W):
    """W: [R,C] float32. Trả (packed[R,nb,12] uint8, table[256] float32 (pad 0), n_uniq,
    n_bad_groups, max_abs_err_do_sanitize)."""
    R, C = W.shape
    assert C % G_SCALE == 0, f"C={C} khong chia het {G_SCALE}"
    nb = C // G_SCALE
    Wv = W.view(R, nb, G_SCALE)
    s = Wv.abs().amax(dim=2)  # [R, nb]
    s_exp = s.unsqueeze(2)
    t = torch.where(s_exp > 0, torch.round(Wv / s_exp.clamp(min=1e-12)),
                     torch.zeros_like(Wv)).clamp(-1, 1).to(torch.int8)  # [R, nb, 64]

    tg = t.view(R, nb, G_SCALE // GROUP, GROUP)          # [R, nb, 16, 4]
    Wvg = Wv.view(R, nb, G_SCALE // GROUP, GROUP)
    tg_flat = tg.reshape(R * nb * (G_SCALE // GROUP), GROUP)
    Wvg_flat = Wvg.reshape(R * nb * (G_SCALE // GROUP), GROUP)
    tg_flat, n_bad = sanitize_24(tg_flat, Wvg_flat)
    tg = tg_flat.view(R, nb, G_SCALE // GROUP, GROUP)

    # sai số do sanitize (chỉ khác 0 nếu n_bad>0)
    W_before_sanitize = (t.float() * s_exp).view(R, C)
    t = tg.view(R, nb, G_SCALE)
    W_after_sanitize = (t.float() * s_exp).view(R, C)
    max_err_sanitize = float((W_after_sanitize - W_before_sanitize).abs().max().item())

    # pattern id per nhom-4 qua LUT base-3
    base3 = ((tg.to(torch.int64) + 1) * torch.tensor([27, 9, 3, 1], dtype=torch.int64)).sum(-1)
    gid = _LUT_B3[base3]  # [R, nb, 16]
    assert int((gid < 0).sum().item()) == 0, "pattern ngoai bang 33 sau sanitize!"

    # ghep cap -> codeword 11-bit
    gid_pairs = gid.view(R, nb, 8, 2)
    code = gid_pairs[..., 0] * 33 + gid_pairs[..., 1]  # [R, nb, 8], 0..1088
    assert int(code.max().item()) < 2048

    # --- bit-pack 8 codeword 11-bit -> 11 byte (LSB-first) — vector hoa qua object dtype ---
    code_np = code.numpy().astype(object)  # [R, nb, 8] python int
    acc = np.zeros((R, nb), dtype=object)
    for k in range(8):
        acc = acc | (code_np[:, :, k] << (11 * k))
    flat_acc = acc.reshape(-1)
    byte_blob = b"".join(int(v).to_bytes(11, "little") for v in flat_acc.tolist())
    code_bytes = np.frombuffer(byte_blob, dtype=np.uint8).reshape(R, nb, 11)

    # --- codebook scale (<=256 gia tri duy nhat/tensor) ---
    s_np = s.numpy().astype(np.float32)
    uniq = np.unique(s_np)
    n_uniq = uniq.size
    assert n_uniq <= 256, (
        f"tensor co {n_uniq} gia tri scale duy nhat > 256 — codebook 1-byte KHONG du "
        f"(vi pham gia dinh 'scale tren luoi f8'). Can sua sang 2-byte scale-idx.")
    table = np.zeros(256, dtype=np.float32)
    table[:n_uniq] = uniq
    val_to_idx = {float(v): i for i, v in enumerate(uniq.tolist())}
    sidx = np.vectorize(lambda v: val_to_idx[float(v)])(s_np).astype(np.uint8)  # [R, nb]

    packed = np.concatenate([code_bytes, sidx[:, :, None]], axis=2)  # [R, nb, 12]

    return packed, table, n_uniq, n_bad, max_err_sanitize, t, s


def verify_roundtrip(packed, table, t_ref, s_ref, R, C, nb):
    """Giai ma lai bang Python (doc lap voi encode) roi so voi (t_ref, s_ref) -> phai
    LOSSLESS BIT-LEVEL (tru sai so sanitize da tinh rieng, o day t_ref/s_ref la BAN DA
    sanitize nen phai khop TUYET DOI)."""
    codes11 = packed[:, :, :11]
    sidx = packed[:, :, 11]
    # giai ma 8 codeword tu 11 byte (Python thuan, cham nhung chi de VERIFY 1 lan)
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
    pats_t = torch.tensor(PATS, dtype=torch.int8)  # [33,4]
    g0 = pats_t[torch.from_numpy(id0)]  # [R,nb,8,4] - nhom CHAN (2p) cua moi cap
    g1 = pats_t[torch.from_numpy(id1)]  # [R,nb,8,4] - nhom LE  (2p+1) cua moi cap
    # THU TU trong block64: 16 nhom-4 lien tiep = 8 cap (g0=nhom 2p, g1=nhom 2p+1).
    # Xep [pair, {g0,g1}, 4] roi flatten -> nhom 0,1,2,...,15 dung thu tu goc.
    t_dec = torch.stack([g0, g1], dim=3).reshape(R, nb, 16, 4).reshape(R, nb, G_SCALE)
    table_t = torch.from_numpy(table)
    s_dec = table_t[torch.from_numpy(sidx.astype(np.int64))]  # [R, nb]
    W_dec = t_dec.float() * s_dec.unsqueeze(-1)
    W_ref = t_ref.float() * s_ref.unsqueeze(-1)
    lossless = torch.equal(W_dec, W_ref)
    return lossless


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("loading ckpt (mmap)...")
    sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
    state = sd["state_dict"] if "state_dict" in sd else sd

    index = []
    byte_off = 0
    cb_off = 0
    all_packed = []
    all_tables = []
    t0 = time.time()
    total_bad = 0
    for layer in range(28):
        for p in PROJS:
            key = f"model.layers.{layer}.{p}.weight"
            W = state[key].float()
            R, C = W.shape
            packed, table, n_uniq, n_bad, max_err, t_ref, s_ref = encode_tensor(W)
            nb = C // G_SCALE
            ok = verify_roundtrip(packed, table, t_ref, s_ref, R, C, nb)
            total_bad += n_bad
            status = "OK" if ok else "LOI!!"
            flag = "" if n_bad == 0 else f"  (sanitize {n_bad} nhom, max_err={max_err:.4g})"
            print(f"  L{layer:2d} {p:22s} {R}x{C:5d}  n_uniq_scale={n_uniq:3d}  "
                  f"lossless={status}{flag}")
            if not ok:
                print(f"    !!! ROUNDTRIP THAT BAI tai {key} — DUNG LAI, xem lai encode.")
                sys.exit(1)

            index.append({"name": key, "R": R, "C": C, "nb": nb,
                          "byte_offset": byte_off, "cb_offset": cb_off,
                          "n_uniq_scale": n_uniq, "n_bad_groups": n_bad})
            all_packed.append(packed.tobytes())
            all_tables.append(table.tobytes())
            byte_off += R * nb * 12
            cb_off += 256

    print(f"\n({time.time()-t0:.0f}s) tong {total_bad} nhom bi sanitize / toan model (196 tensor)")

    with open(os.path.join(OUT_DIR, "tq33_packed.bin"), "wb") as f:
        for b in all_packed:
            f.write(b)
    with open(os.path.join(OUT_DIR, "tq33_codebooks.bin"), "wb") as f:
        for b in all_tables:
            f.write(b)
    with open(os.path.join(OUT_DIR, "tq33_meta_index.txt"), "w", encoding="utf-8") as f:
        f.write(f"{len(index)}\n")
        for e in index:
            f.write(f"{e['name']} {e['R']} {e['C']} {e['nb']} {e['byte_offset']} {e['cb_offset']}\n")

    total_packed_bytes = byte_off
    total_orig_elems = sum(e["R"] * e["C"] for e in index)
    print(f"\ntong packed: {total_packed_bytes/1e6:.2f} MB  "
          f"({total_packed_bytes*8/total_orig_elems:.3f} bpw thuc te, {total_orig_elems} trong so)")
    print(f"fp32 goc tuong ung: {total_orig_elems*4/1e6:.1f} MB")
    print(f"da ghi -> {OUT_DIR}")


if __name__ == "__main__":
    main()
