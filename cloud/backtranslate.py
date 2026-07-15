#!/usr/bin/env python3
"""Vòng 1 — Back-translation cứu chiều VI→JA (PLAN_BUOC5 §2.3).

Ý tưởng: muốn model GIỎI SINH tiếng Nhật thì TARGET khi train phải là câu Nhật
THẬT. Ta có sẵn câu JA in-domain thật (data/synthetic/ja_indomain_bt.ja, gom từ
Qiita). Dùng chính model (chiều mạnh ja->vi, chrF 41.7) dịch chúng ra VI tổng hợp
=> cặp (VI tổng hợp -> JA thật). Chỉ train chiều vi->ja với các cặp này.

Chạy trên GPU node (batch=1 vì generate_cached chỉ hỗ trợ 1 câu). ~86k câu:
GPU ~1.5-2.5h. RESUME được (đếm dòng bt.ja đã có). Lọc câu hỏng ngay tại đây.

Xuất (line-aligned, để mix_and_binarize gắn cờ vi->ja-only):
  data/synthetic/bt.ja   câu Nhật THẬT   (target khi train)
  data/synthetic/bt.vi   câu Việt do model sinh (source khi train)

Env:
  BT_CKPT   checkpoint (mặc định checkpoints/last.pt)
  BT_MAX    số câu tối đa xử lý (mặc định 0 = hết); dùng để chạy thử nhanh
  BT_MAXTOK max token sinh mỗi câu (mặc định 128)
  BT_DEVICE cuda | cpu (mặc định cuda nếu có)
"""
import os
import re
import sys
import time
from pathlib import Path

import torch
import sentencepiece as spm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bitnet import BitNetLM, BitNetConfig

SYN = ROOT / "data" / "synthetic"
SRC_JA = SYN / "ja_indomain_bt.ja"      # câu JA thật (input)
OUT_JA = SYN / "bt.ja"                    # câu JA thật (target train)
OUT_VI = SYN / "bt.vi"                    # câu VI model sinh (source train)

JA_CHARS = re.compile(r"[぀-ヿ㐀-鿿]")     # còn ký tự Nhật trong VI => model hỏng
CKPT = os.environ.get("BT_CKPT", str(ROOT / "checkpoints" / "last.pt"))
MAXTOK = int(os.environ.get("BT_MAXTOK", "128"))
BT_MAX = int(os.environ.get("BT_MAX", "0"))
DEVICE = os.environ.get("BT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def good(vi, ja):
    vi = vi.strip()
    if len(vi) < 2:
        return False
    if JA_CHARS.search(vi):                # model không dịch được -> bỏ
        return False
    r = len(vi) / max(len(ja), 1)          # tỉ lệ độ dài bất thường -> bỏ
    if r < 0.3 or r > 6:
        return False
    return True


def main():
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
    BOS, EOS = sp.bos_id(), sp.eos_id()
    VIE = sp.piece_to_id(">>vie<<")        # ja->vi: input mang thẻ >>vie<<

    cfg = BitNetConfig(vocab_size=sp.get_piece_size())
    m = BitNetLM(cfg).eval()
    ck = torch.load(CKPT, map_location="cpu")
    m.load_state_dict(ck["model"])
    m.to(DEVICE)                           # PHẢI .to() TRƯỚC freeze: _wq_frozen là attribute
    m.freeze_for_inference()               # thường (không phải buffer) nên .to() không đẩy nó
    #                                        lên GPU -> tính freeze SAU khi ở cuda mới khớp device.
    print(f"[BT] loaded {CKPT} step={ck.get('step','?')} | device={DEVICE}", flush=True)

    ja_all = [l.rstrip("\n") for l in SRC_JA.read_text(encoding="utf-8").splitlines() if l.strip()]
    if BT_MAX > 0:
        ja_all = ja_all[:BT_MAX]
    n = len(ja_all)

    done = 0
    if OUT_JA.exists():
        done = sum(1 for _ in OUT_JA.open(encoding="utf-8"))
        print(f"[BT] resume: đã có {done:,}/{n:,}", flush=True)

    fja = OUT_JA.open("a", encoding="utf-8", buffering=1)   # line-buffered: wc -l theo realtime
    fvi = OUT_VI.open("a", encoding="utf-8", buffering=1)
    kept, t0 = 0, time.time()
    print(f"[BT] bắt đầu generate {n-done:,} câu trên {DEVICE}... (in tiến độ mỗi 100 câu)", flush=True)
    try:
        for i in range(done, n):
            ja = ja_all[i]
            ids = [BOS, VIE] + sp.encode(ja) + [EOS]
            out = m.generate_cached(torch.tensor([ids], device=DEVICE),
                                    max_new_tokens=MAXTOK, eos_id=EOS)
            gen = out[0, len(ids):].tolist()
            if gen and gen[-1] == EOS:
                gen = gen[:-1]
            vi = sp.decode(gen).strip()
            # ghi cả cặp bị loại 1 dòng RỖNG ở vi để 2 file luôn thẳng hàng, rồi
            # mix bỏ dòng vi rỗng. Đơn giản & resume an toàn.
            if not good(vi, ja):
                vi = ""
            else:
                kept += 1
            fja.write(ja + "\n")
            fvi.write(vi + "\n")
            if (i + 1) % 100 == 0:
                dt = time.time() - t0
                rate = (i + 1 - done) / max(dt, 1e-9)
                eta = (n - i - 1) / max(rate, 1e-9) / 60
                print(f"[BT] {i+1:,}/{n:,} | giữ {kept:,} | {rate:.1f} câu/s | ETA {eta:.0f} phút",
                      flush=True)
    finally:
        fja.close()
        fvi.close()
    print(f"[BT] XONG: {n:,} câu, giữ được {kept:,} cặp vi->ja -> bt.ja/bt.vi", flush=True)


if __name__ == "__main__":
    main()
