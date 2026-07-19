#!/usr/bin/env python3
"""Vòng 1 — Back-translation cứu chiều VI→JA (PLAN_BUOC5 §2.3).

Ý tưởng: muốn model GIỎI SINH tiếng Nhật thì TARGET khi train phải là câu Nhật
THẬT. Ta có sẵn câu JA in-domain thật (data/synthetic/ja_indomain_bt.ja, gom từ
Qiita). Dùng chính model (chiều mạnh ja->vi, chrF 41.7) dịch chúng ra VI tổng hợp
=> cặp (VI tổng hợp -> JA thật). Chỉ train chiều vi->ja với các cặp này.

TỐC ĐỘ: generate BATCH theo nhóm câu CÙNG độ dài token (không cần padding nên
không phải sửa model) — GPU ăn ~BT_BATCH câu/lượt thay vì 1. Batch=1 cũ chỉ đạt
~3 câu/s vì vòng lặp Python mỗi token + sync GPU->CPU liên tục; bản batch đạt
~x30-60. Xử lý theo siêu-khối 4096 câu THEO THỨ TỰ GỐC rồi mới ghi ra file, nên
2 file output luôn thẳng hàng + resume theo số dòng vẫn đúng (kill giữa chừng
mất tối đa 1 siêu-khối compute, không hỏng file).

Xuất (line-aligned, để mix_and_binarize gắn cờ vi->ja-only):
  data/synthetic/bt.ja   câu Nhật THẬT   (target khi train)
  data/synthetic/bt.vi   câu Việt do model sinh (source khi train)

Env:
  BT_CKPT   checkpoint (mặc định checkpoints/last.pt)
  BT_MAX    số câu tối đa xử lý (mặc định 0 = hết); dùng để chạy thử nhanh
  BT_MAXTOK max token sinh mỗi câu (mặc định 128)
  BT_BATCH  cỡ batch (mặc định 128; hạ nếu OOM)
  BT_DEVICE cuda | cpu (mặc định cuda nếu có)
  BT_SRC / BT_OUT_JA / BT_OUT_VI  override đường dẫn (test)
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
SRC_JA = Path(os.environ.get("BT_SRC", SYN / "ja_indomain_bt.ja"))   # câu JA thật (input)
OUT_JA = Path(os.environ.get("BT_OUT_JA", SYN / "bt.ja"))            # câu JA thật (target train)
OUT_VI = Path(os.environ.get("BT_OUT_VI", SYN / "bt.vi"))            # câu VI model sinh (source)

JA_CHARS = re.compile(r"[぀-ヿ㐀-鿿]")     # còn ký tự Nhật trong VI => model hỏng
CKPT = os.environ.get("BT_CKPT", str(ROOT / "checkpoints" / "last.pt"))
MAXTOK = int(os.environ.get("BT_MAXTOK", "128"))
BT_MAX = int(os.environ.get("BT_MAX", "0"))
BT_BATCH = int(os.environ.get("BT_BATCH", "128"))
DEVICE = os.environ.get("BT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CHUNK = 4096                               # siêu-khối: ghi file mỗi lần xong 1 khối


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


@torch.no_grad()
def generate_batch(m, ids_list, max_new_tokens, eos_id, device):
    """Greedy + KV-cache cho MỘT batch câu CÙNG độ dài prompt (không padding).
    Trả về list[list[int]] token sinh ra (đã cắt tại EOS) theo đúng thứ tự vào."""
    B, L = len(ids_list), len(ids_list[0])
    caches = [None] * len(m.blocks)
    cur = torch.tensor(ids_list, device=device)          # (B, L) prefill
    pos = 0
    steps = []
    done = torch.zeros(B, dtype=torch.bool, device=device)
    # không vượt bảng RoPE (max_seq): prompt L + sinh tối đa max_seq-L-1
    max_new = min(max_new_tokens, m.cfg.max_seq - L - 1)
    for _ in range(max(max_new, 0)):
        T = cur.shape[1]
        x = m.embed(cur)
        cos, sin = m.rope_cos[pos:pos + T], m.rope_sin[pos:pos + T]
        for i, blk in enumerate(m.blocks):
            a, kv = m._attn_step(blk.attn, x, cos, sin, caches[i])
            x = x + a
            x = x + blk.ffn(x)
            caches[i] = kv
        logits = m.lm_head(m.output_norm(x)[:, -1, :])   # (B, V)
        nxt = logits.argmax(-1)                          # (B,)
        nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
        done = done | (nxt == eos_id)
        steps.append(nxt)
        pos += T
        cur = nxt.unsqueeze(1)
        if bool(done.all()):                             # 1 sync/token cho CẢ batch
            break
    if not steps:
        return [[] for _ in range(B)]
    gen = torch.stack(steps, dim=1).tolist()             # (B, n_steps)
    outs = []
    for row in gen:
        if eos_id in row:
            row = row[:row.index(eos_id)]
        outs.append(row)
    return outs


def main():
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
    BOS, EOS = sp.bos_id(), sp.eos_id()
    VIE = sp.piece_to_id(">>vie<<")        # ja->vi: input mang thẻ >>vie<<

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = BitNetConfig(**ck["cfg"]) if "cfg" in ck else BitNetConfig(vocab_size=sp.get_piece_size())
    m = BitNetLM(cfg).eval()
    m.load_state_dict(ck["model"])
    m.to(DEVICE)                           # PHẢI .to() TRƯỚC freeze: _wq_frozen là attribute
    m.freeze_for_inference()               # thường (không phải buffer) nên .to() không đẩy nó
    #                                        lên GPU -> tính freeze SAU khi ở cuda mới khớp device.
    print(f"[BT] loaded {CKPT} step={ck.get('step','?')} | device={DEVICE} | batch={BT_BATCH}",
          flush=True)

    ja_all = [l.rstrip("\n") for l in SRC_JA.read_text(encoding="utf-8").splitlines() if l.strip()]
    if BT_MAX > 0:
        ja_all = ja_all[:BT_MAX]
    n = len(ja_all)

    done_lines = 0
    if OUT_JA.exists():
        done_lines = sum(1 for _ in OUT_JA.open(encoding="utf-8"))
        print(f"[BT] resume: đã có {done_lines:,}/{n:,}", flush=True)

    fja = OUT_JA.open("a", encoding="utf-8", buffering=1)
    fvi = OUT_VI.open("a", encoding="utf-8", buffering=1)
    kept, t0 = 0, time.time()
    print(f"[BT] bắt đầu generate {n-done_lines:,} câu trên {DEVICE}...", flush=True)
    try:
        for c0 in range(done_lines, n, CHUNK):
            chunk = ja_all[c0:c0 + CHUNK]
            # tokenize + gom nhóm theo ĐÚNG độ dài prompt (batch không cần padding)
            enc = []
            by_len = {}
            for j, ja in enumerate(chunk):
                ids = [BOS, VIE] + sp.encode(ja) + [EOS]
                enc.append(ids)
                if len(ids) <= cfg.max_seq - 8:          # quá dài -> bỏ (ghi dòng rỗng)
                    by_len.setdefault(len(ids), []).append(j)
            vis = [""] * len(chunk)
            for L, idxs in sorted(by_len.items()):
                for b0 in range(0, len(idxs), BT_BATCH):
                    bidx = idxs[b0:b0 + BT_BATCH]
                    outs = generate_batch(m, [enc[j] for j in bidx], MAXTOK, EOS, DEVICE)
                    for j, gen in zip(bidx, outs):
                        vis[j] = sp.decode(gen).strip()
            # ghi CẢ khối theo thứ tự gốc (cặp loại -> dòng vi RỖNG, 2 file thẳng hàng)
            for j, ja in enumerate(chunk):
                vi = vis[j]
                if not good(vi, ja):
                    vi = ""
                else:
                    kept += 1
                fja.write(ja + "\n")
                fvi.write(vi + "\n")
            dt = time.time() - t0
            done_now = c0 + len(chunk) - done_lines
            rate = done_now / max(dt, 1e-9)
            eta = (n - c0 - len(chunk)) / max(rate, 1e-9) / 60
            print(f"[BT] {c0+len(chunk):,}/{n:,} | giữ {kept:,} | {rate:.1f} câu/s | ETA {eta:.0f} phút",
                  flush=True)
    finally:
        fja.close()
        fvi.close()
    print(f"[BT] XONG: {n:,} câu, giữ được {kept:,} cặp vi->ja -> bt.ja/bt.vi", flush=True)


if __name__ == "__main__":
    main()
