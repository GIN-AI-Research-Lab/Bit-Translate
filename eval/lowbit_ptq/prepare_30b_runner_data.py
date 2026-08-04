# -*- coding: utf-8 -*-
"""Tien xu ly du lieu TQ33-30B (D:/Bit-Translate-data/tq33_30b/) sang dinh dang runner C de
doc NHANH + DON GIAN — mirror dung pattern da chung minh o 0.6B (tq33_packed.bin +
tq33_codebooks.bin + tq33_meta_index.txt MOT file lon, thay vi 18624 file rieng le tung
tensor nhu bulk_encode_tq33_30b.py xuat ra). Ly do: C mo 18624 file rieng se cham/phuc tap
hon nhieu so voi 2 file lon + 1 index van ban doc bang fscanf (dung y het style 0.6B).

QUAN TRONG: linear_names trong bulk_encode_tq33_30b.py duoc .sort() THEO CHUOI (lexicographic)
truoc khi danh so file — nghia la "experts.10." dung TRUOC "experts.2." trong thu tu file
(so sanh chuoi, khong phai so). Script nay KHONG dua vao thu tu file .tq33 goc — doc manifest
theo TEN CHINH XAC (dict lookup O(1)) roi ghi ra theo thu tu CO DINH ma runner C mong doi:
  layer 0..47: [q_proj, k_proj, v_proj, o_proj, expert0.gate, expert0.up, expert0.down,
                expert1.gate, ..., expert127.down]

Output -> D:/Bit-Translate-data/tq33_30b/runner/:
  packed_30b.bin     -- noi tiep R*nb*12 byte cua ca 18624 tensor, THEO DUNG thu tu co dinh
  codebooks_30b.bin  -- noi tiep codebook float32 cua ca 18624 tensor (cung thu tu)
  linear_index.txt   -- "<count>\n" roi moi dong: name R C nb byte_offset cb_offset_float
  extras_index.txt   -- "<count>\n" roi moi dong: name shape_flat... offset_byte nbytes
                        (tro thang vao extras.bin CO SAN, khong copy — chi sap xep lai
                        thu tu doc de khop thu tu runner C mong doi)
"""
import json
import os
import time

OUT_DIR = "D:/Bit-Translate-data/tq33_30b"
RUNNER_DIR = os.path.join(OUT_DIR, "runner")
os.makedirs(RUNNER_DIR, exist_ok=True)

N_LAYER = 48
N_EXPERT = 128


def main():
    t0 = time.time()
    with open(os.path.join(OUT_DIR, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(OUT_DIR, "extras_manifest.json"), encoding="utf-8") as f:
        extras_manifest = json.load(f)
    print(f"manifest: {len(manifest)} linear tensors, extras: {len(extras_manifest)} tensors")

    # ---------- thu tu co dinh cho linear (TQ33) ----------
    order = []
    for l in range(N_LAYER):
        p = f"model.layers.{l}.self_attn."
        order.append(p + "q_proj.weight")
        order.append(p + "k_proj.weight")
        order.append(p + "v_proj.weight")
        order.append(p + "o_proj.weight")
        for e in range(N_EXPERT):
            pe = f"model.layers.{l}.mlp.experts.{e}."
            order.append(pe + "gate_proj.weight")
            order.append(pe + "up_proj.weight")
            order.append(pe + "down_proj.weight")
    assert len(order) == len(manifest) == 18624, f"mismatch: order={len(order)} manifest={len(manifest)}"
    missing = [n for n in order if n not in manifest]
    assert not missing, f"thieu {len(missing)} tensor trong manifest, vd: {missing[:5]}"

    packed_path = os.path.join(RUNNER_DIR, "packed_30b.bin")
    cb_path = os.path.join(RUNNER_DIR, "codebooks_30b.bin")
    idx_path = os.path.join(RUNNER_DIR, "linear_index.txt")

    byte_off = 0
    cb_off = 0  # don vi FLOAT (khop C: g_codebooks + cb_offset, float* arithmetic)
    lines = []
    with open(packed_path, "wb") as fp, open(cb_path, "wb") as fc:
        for i, name in enumerate(order):
            m = manifest[name]
            R, C = m["shape"]
            nb = m["nb"]
            src_path = os.path.join(OUT_DIR, m["file"])
            with open(src_path, "rb") as fsrc:
                n_cb_raw = fsrc.read(2)
                n_cb = int.from_bytes(n_cb_raw, "little")
                assert n_cb == m["codebook_n"], f"{name}: codebook_n header={n_cb} manifest={m['codebook_n']}"
                cb_bytes = fsrc.read(n_cb * 4)
                packed_bytes = fsrc.read(R * nb * 12)
                rest = fsrc.read()
                assert len(rest) == 0, f"{name}: con du du lieu thua {len(rest)} byte"
            fc.write(cb_bytes)
            fp.write(packed_bytes)
            lines.append(f"{name} {R} {C} {nb} {byte_off} {cb_off}")
            byte_off += len(packed_bytes)
            cb_off += n_cb
            if (i + 1) % 2000 == 0 or i == len(order) - 1:
                dt = time.time() - t0
                print(f"  [{i+1}/{len(order)}] {dt:.0f}s  packed={byte_off/1e9:.2f}GB  cb={cb_off*4/1e6:.1f}MB",
                      flush=True)

    with open(idx_path, "w", encoding="utf-8") as f:
        f.write(f"{len(order)}\n")
        f.write("\n".join(lines) + "\n")
    print(f"linear: packed={byte_off/1e9:.3f}GB codebook={cb_off*4/1e6:.2f}MB -> {RUNNER_DIR}")

    # ---------- extras index (KHONG copy du lieu, chi tro vao extras.bin co san) ----------
    extras_order = ["model.embed_tokens.weight"]
    for l in range(N_LAYER):
        p = f"model.layers.{l}."
        extras_order.append(p + "input_layernorm.weight")
        extras_order.append(p + "post_attention_layernorm.weight")
        extras_order.append(p + "self_attn.q_norm.weight")
        extras_order.append(p + "self_attn.k_norm.weight")
        extras_order.append(p + "mlp.gate.weight")
    extras_order.append("model.norm.weight")
    extras_order.append("lm_head.weight")
    missing_e = [n for n in extras_order if n not in extras_manifest]
    assert not missing_e, f"thieu extras: {missing_e[:5]}"
    assert len(extras_order) == len(extras_manifest), \
        f"extras_order={len(extras_order)} extras_manifest={len(extras_manifest)}"

    extras_idx_path = os.path.join(RUNNER_DIR, "extras_index.txt")
    with open(extras_idx_path, "w", encoding="utf-8") as f:
        f.write(f"{len(extras_order)}\n")
        for name in extras_order:
            m = extras_manifest[name]
            shape = m["shape"]
            shape_str = " ".join(str(s) for s in shape)
            f.write(f"{name} {len(shape)} {shape_str} {m['offset']} {m['nbytes']}\n")
    print(f"extras: {len(extras_order)} tensor -> {extras_idx_path} (tro vao extras.bin co san, khong copy)")
    print(f"TONG THOI GIAN: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
