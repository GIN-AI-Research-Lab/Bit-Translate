"""V8-x3 — dựng bin_x3 từ bin (mix KD ×8) NGAY TRÊN VOLUME: giữ base nguyên,
cắt đuôi KD từ ×8 xuống ×3 (giữ 3 bản đầu mỗi nhóm 8 bản liền kề).

Vì sao làm được ở mức binary: scripts/mix_and_binarize.py nối data mới vào CUỐI
base (`train.* = base.train.* + new`), mỗi record ja2vi sinh 1 seq rồi lặp
new_rep=8 bản LIỀN KỀ (vòng `for _ in range(r)`), nên đuôi 104.800 entry
= 13.100 nhóm × 8 bản giống hệt. Không cần v8_all.jsonl, không cần tokenize lại.

Kiểm tra chặt trước khi cắt (sai bất kỳ điều nào -> abort):
  - tổng seq = 15.986.646 = base 15.881.846 + 104.800 (số TONGKET_V8 §1)
  - đuôi chia hết nhóm 8, mỗi nhóm: (len,tstart) + TOKEN BYTES giống hệt nhau
  - token[1] của seq đuôi = id('>>vie<<') (KD là ja2vi)

Chạy:  MODAL_PROFILE=thaovyh2t modal run cloud/modal_rebin_v8x3.py::rebin_x3
"""
import modal

app = modal.App("vija-v8-rebin")

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy<2")
vol = modal.Volume.from_name("vija-v8-vol")

BASE_SEQ = 15_881_846      # bin_v7g phần train (TONGKET_V8 §1)
KD_SEQ = 104_800           # KD ×8
OLD_REP, NEW_REP = 8, 3
VIE_ID = 4                 # sp.piece_to_id('>>vie<<')


@app.function(image=image, volumes={"/persist": vol}, cpu=4.0, memory=16384,
              timeout=3600)
def rebin_x3():
    import numpy as np
    import os
    import shutil

    idx = np.load("/persist/bin/train.index.npy")
    toks = np.fromfile("/persist/bin/train.tokens.u16", dtype=np.uint16)
    total = idx.shape[0]
    print(f"index: {total:,} seq | tokens: {len(toks):,}")
    assert total == BASE_SEQ + KD_SEQ, f"tổng seq {total:,} != {BASE_SEQ + KD_SEQ:,} — bin không phải mix ×8 mong đợi, ABORT"
    assert idx[:, 0].sum() == len(toks), "index không khớp token stream, ABORT"

    offsets = np.zeros(total + 1, dtype=np.int64)
    np.cumsum(idx[:, 0], out=offsets[1:])
    tail = idx[BASE_SEQ:]
    assert KD_SEQ % OLD_REP == 0
    n_uniq = KD_SEQ // OLD_REP

    # kiểm cấu trúc nhóm-8: (len,tstart) + token bytes giống hệt trong nhóm
    for g in range(n_uniq):
        rows = tail[g * OLD_REP:(g + 1) * OLD_REP]
        assert (rows == rows[0]).all(), f"nhóm {g}: (len,tstart) lệch trong nhóm 8, ABORT"
        base_i = BASE_SEQ + g * OLD_REP
        s0 = toks[offsets[base_i]:offsets[base_i + 1]]
        assert s0[1] == VIE_ID, f"nhóm {g}: token[1]={s0[1]} != >>vie<<, ABORT"
        # so token bản 0 với bản cuối trong nhóm (đủ để bắt lệch layout)
        s7 = toks[offsets[base_i + OLD_REP - 1]:offsets[base_i + OLD_REP]]
        assert np.array_equal(s0, s7), f"nhóm {g}: token bytes lệch trong nhóm 8, ABORT"
    print(f"kiểm {n_uniq:,} nhóm ×8: OK (giống hệt trong nhóm, đều là ja2vi)")

    # dựng bin_x3
    os.makedirs("/persist/bin_x3", exist_ok=True)
    base_tok_end = int(offsets[BASE_SEQ])
    keep_rows = [idx[:BASE_SEQ]]
    with open("/persist/bin_x3/train.tokens.u16", "wb") as f:
        toks[:base_tok_end].tofile(f)                       # base nguyên vẹn
        for g in range(n_uniq):
            base_i = BASE_SEQ + g * OLD_REP
            seq = toks[offsets[base_i]:offsets[base_i + 1]]
            for _ in range(NEW_REP):
                seq.tofile(f)
            keep_rows.append(np.repeat(tail[g * OLD_REP:g * OLD_REP + 1],
                                       NEW_REP, axis=0))
    new_idx = np.vstack(keep_rows)
    np.save("/persist/bin_x3/train.index.npy", new_idx)
    for f in ("dev.tokens.u16", "dev.index.npy"):
        shutil.copyfile(f"/persist/bin/{f}", f"/persist/bin_x3/{f}")

    out_toks = np.fromfile("/persist/bin_x3/train.tokens.u16", dtype=np.uint16)
    assert new_idx[:, 0].sum() == len(out_toks), "bin_x3 index/token lệch, ABORT"
    vol.commit()
    print(f"XONG: bin_x3 = base {BASE_SEQ:,} + KD×{NEW_REP} {n_uniq * NEW_REP:,} "
          f"= {new_idx.shape[0]:,} seq | tokens {len(out_toks):,}")
