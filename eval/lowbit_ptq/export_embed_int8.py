# -*- coding: utf-8 -*-
r"""TQ33 SPEED-100 — bước 1: lượng tử hoá int8 per-row cho `model.embed_tokens.weight`
(tensor này dùng làm CẢ input lookup LẪN output head/lm_head — quyết định đã chốt ở
RESEARCH_TQ33_RUNNER.md mục 1.1, KHÔNG dùng lm_head.weight).

Lý do: RESEARCH_TQ33_RUNNER.md mục 4.3 đo được embed+logits F32 (622MB đọc/token) chiếm
63-87% thời gian mỗi token — nút thắt lớn nhất, KHÔNG phải kernel TQ33. int8 per-row giảm
4x còn 155.6MB/token.

Kỹ thuật: per-row symmetric int8 — scale[v] = max|W[v,:]| / 127, Q[v,:] = round(W/scale)
clip ±127 (clip ±127 chứ không ±128 để kernel maddubs AVX2 không saturate: 127*127*2 =
32258 < 32767). Đọc thẳng từ weights_f32/weights.bin (dump f32 của ckpt, giá trị GIỐNG HỆT
đọc lại ckpt .pt — export_full_model_f32.py đã dump từ fp16 gốc sang f32).

Validation trong script này (trước khi đụng tới C):
  1. Sai số tái tạo per-row (rel L2) — phân bố mean/median/max.
  2. Logits trước/sau trên oracle THẬT (hidden_final.npy của prompt đã validate):
     (a) weight-int8, activation f32   (b) weight-int8 + activation int8 per-64-group
     (mô phỏng ĐÚNG kernel C sẽ làm) — so rel err + top-1/top-5 từng vị trí vs oracle.
  3. Sai số embedding lookup phía INPUT cho các token của prompt.

Output: D:\Bit-Translate-data\tq33_runner\embed_int8\
  embed_int8.bin    int8 [151936, 1024] row-major (155.58 MB)
  embed_scales.bin  f32  [151936]
  embed_int8_meta.txt  "rows cols"
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

WDIR = r"D:\Bit-Translate-data\tq33_runner\weights_f32"
ORACLE_DIR = r"D:\Bit-Translate-data\tq33_runner\oracle"
OUT_DIR = r"D:\Bit-Translate-data\tq33_runner\embed_int8"

CHUNK = 16384  # hàng/lần để không tạo temp f32 622MB


def rel_err(a, b):
    num = np.sqrt(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    den = np.sqrt(np.sum(b.astype(np.float64) ** 2)) + 1e-30
    return float(num / den)


def quantize_act_int8_g64(x):
    """int8 per-64-group y hệt quantize_x_int8 trong C (scale = max|x|/127, lrintf)."""
    n = x.shape[-1]
    assert n % 64 == 0
    xg = x.reshape(*x.shape[:-1], n // 64, 64)
    m = np.abs(xg).max(axis=-1, keepdims=True)
    sc = np.where(m > 0, m / 127.0, 1.0).astype(np.float32)
    q = np.rint(xg / sc).astype(np.int32)  # lrintf = round-half-even, np.rint giống
    return q, sc.squeeze(-1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(WDIR, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    ent = next(e for e in meta["tensors"] if e["name"] == "model.embed_tokens.weight")
    rows, cols = ent["shape"]
    print(f"model.embed_tokens.weight: [{rows}, {cols}]  offset_f32={ent['offset_f32']}")

    W = np.memmap(os.path.join(WDIR, "weights.bin"), dtype="<f4", mode="r",
                  offset=ent["offset_f32"] * 4, shape=(rows, cols))

    Q = np.empty((rows, cols), dtype=np.int8)
    scales = np.empty(rows, dtype=np.float32)
    recon_rel = np.empty(rows, dtype=np.float64)
    for r0 in range(0, rows, CHUNK):
        r1 = min(r0 + CHUNK, rows)
        Wb = np.asarray(W[r0:r1], dtype=np.float32)
        am = np.abs(Wb).max(axis=1)
        sc = np.where(am > 0, am / 127.0, 1.0).astype(np.float32)
        Qb = np.clip(np.rint(Wb / sc[:, None]), -127, 127).astype(np.int8)
        Q[r0:r1] = Qb
        scales[r0:r1] = sc
        rec = Qb.astype(np.float32) * sc[:, None]
        num = np.sqrt(np.sum((rec - Wb) ** 2, axis=1, dtype=np.float64))
        den = np.sqrt(np.sum(Wb.astype(np.float64) ** 2, axis=1)) + 1e-30
        recon_rel[r0:r1] = num / den

    print(f"[recon] rel L2 per-row: mean={recon_rel.mean():.4e} median={np.median(recon_rel):.4e} "
          f"max={recon_rel.max():.4e} (row {int(np.argmax(recon_rel))})")

    Q.tofile(os.path.join(OUT_DIR, "embed_int8.bin"))
    scales.tofile(os.path.join(OUT_DIR, "embed_scales.bin"))
    with open(os.path.join(OUT_DIR, "embed_int8_meta.txt"), "w") as f:
        f.write(f"{rows} {cols}\n")
    print(f"[ghi] embed_int8.bin ({rows*cols/1e6:.2f} MB) + embed_scales.bin -> {OUT_DIR}")

    # ---------- validation 2: logits trước/sau trên oracle hidden thật ----------
    X = np.load(os.path.join(ORACLE_DIR, "hidden_final.npy")).astype(np.float32)  # [seq,1024]
    L_ref = np.load(os.path.join(ORACLE_DIR, "logits.npy")).astype(np.float32)    # [seq,V]
    seq = X.shape[0]

    # (a) weight-int8, activation f32
    La = np.empty((seq, rows), dtype=np.float32)
    for r0 in range(0, rows, CHUNK):
        r1 = min(r0 + CHUNK, rows)
        La[:, r0:r1] = (X @ Q[r0:r1].astype(np.float32).T) * scales[r0:r1][None, :]

    # (b) weight-int8 + activation int8 per-64-group (mô phỏng kernel C)
    Xq, Xs = quantize_act_int8_g64(X)  # [seq,16,64] int32, [seq,16]
    Lb = np.zeros((seq, rows), dtype=np.float64)
    for r0 in range(0, rows, CHUNK):
        r1 = min(r0 + CHUNK, rows)
        Qb = Q[r0:r1].reshape(r1 - r0, cols // 64, 64).astype(np.int32)
        for g in range(cols // 64):
            acc = Xq[:, g, :].astype(np.float64) @ Qb[:, g, :].T.astype(np.float64)
            Lb[:, r0:r1] += acc * Xs[:, g][:, None]
        Lb[:, r0:r1] *= scales[r0:r1][None, :]
    Lb = Lb.astype(np.float32)

    for name, L in (("w-int8 only", La), ("w-int8 + act-int8/g64 (nhu kernel C)", Lb)):
        re = rel_err(L, L_ref)
        print(f"\n[logits {name}] rel_err toan cuc = {re:.4e}")
        n_top1 = n_top5 = 0
        for t in range(seq):
            t1r, t1 = int(np.argmax(L_ref[t])), int(np.argmax(L[t]))
            s5r = set(np.argsort(-L_ref[t])[:5].tolist())
            s5 = set(np.argsort(-L[t])[:5].tolist())
            n_top1 += (t1 == t1r)
            n_top5 += len(s5 & s5r)
            if t == seq - 1:
                print(f"  vi tri cuoi: top1 ref={t1r} int8={t1} "
                      f"{'KHOP' if t1 == t1r else 'LECH'}; top5 overlap={len(s5 & s5r)}/5")
        print(f"  top-1 khop {n_top1}/{seq} vi tri; top-5 overlap trung binh "
              f"{n_top5/seq:.2f}/5")

    # ---------- validation 3: embedding lookup phía input ----------
    with open(os.path.join(ORACLE_DIR, "tokens.bin"), "rb") as f:
        n_tok = np.fromfile(f, dtype=np.int32, count=1)[0]
        toks = np.fromfile(f, dtype=np.int32, count=n_tok)
    errs = []
    for t in toks:
        w = np.asarray(W[t], dtype=np.float32)
        rec = Q[t].astype(np.float32) * scales[t]
        errs.append(rel_err(rec, w))
    print(f"\n[input lookup] rel err embedding row cho {n_tok} token prompt: "
          f"{', '.join(f'{e:.3e}' for e in errs)}")
    print("\nXONG — neu top-1/top-5 khop va rel err ~1e-3 thi trien khai kernel C an toan.")


if __name__ == "__main__":
    main()
