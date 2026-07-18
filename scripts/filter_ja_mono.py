#!/usr/bin/env python3
"""Lọc câu JA đơn ngữ harvest (Qiita/GitHub) trước khi làm vế đích BT.

Haiku review 300 mẫu (2026-07-18, eval/review_bt): 50.3% là rác — câu cụt, mảnh
danh sách/heading/mục lục, hàng bảng, khoảng trắng vỡ giữa từ (di chứng lột
inline-code). Round-trip/LaBSE lúc BT KHÔNG bắt được loại này (nguồn xấu sẵn)
nên phải lọc ở đây. Rule được CALIBRATE trên 300 nhãn Haiku (xem __main__ --calib).

Dùng: python scripts/filter_ja_mono.py           # lọc file mặc định -> .clean.ja
      python scripts/filter_ja_mono.py --calib   # đo precision/recall vs nhãn Haiku
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "data" / "synthetic" / "ja_indomain_bt.ja"
OUT = ROOT / "data" / "synthetic" / "ja_indomain_bt.clean.ja"

END_OK = re.compile(r"[。！？!?…」』]$")
JA = re.compile(r"[぀-ヿ㐀-鿿]")
HIRA = re.compile(r"[ぁ-ん]")
# heading/boilerplate/danh sách/bảng
BAD_PAT = re.compile(
    r"Advent Calendar|アドベントカレンダー|カレンダー\s*\d+日目|^第.{1,4}[章回話部]|──|—{2,}|"
    r"^[-・#*|>‣▪◦†]|^\d+[\.\)．]\s|^[①-⑳]|目次|^はじめに|続きを読む|^【[^】]*】$|"
    r"[|│┃]|^よかったら|いいね.*お願い")
# câu cụt: mở đầu bằng trợ từ/ngoặc đóng (mất vế trước) — không gồm と (という hợp lệ)
BAD_START = re.compile(r"^[のはがをにでへもや、。」』）)\]〜ー・:：]")
# khoảng trắng vỡ (di chứng lột inline-code/markdown): giữa chữ Nhật, quanh dấu câu,
# sau 、 (chỗ inline-code bị đục thành lỗ trống), " / " trần
BROKEN_SPACE = re.compile(r"[぀-ヿ㐀-鿿]\s+[、。]|[぀-ヿ㐀-鿿]\s+[぀-ヿ㐀-鿿]|、\s|\s/\s|（\s|\s）")
# ký hiệu trình bày (bảng/diagram/emoji-bullet)
JUNK_CHAR = re.compile(r"[→⇒⇔★☆✓✗✅❌⭕■□▲△▼▽●○◎‥]|:{2,}")


def ok(s):
    s = s.strip()
    if not (12 <= len(s) <= 160):
        return False
    if not END_OK.search(s):
        return False
    if not HIRA.search(s):          # không có hiragana = cụm danh từ/heading
        return False
    if BAD_PAT.search(s) or BAD_START.search(s):
        return False
    if BROKEN_SPACE.search(s) or JUNK_CHAR.search(s):
        return False
    # tỉ lệ ký tự Nhật tối thiểu (bỏ dòng chủ yếu là code/tên riêng Latin)
    if len(JA.findall(s)) < max(6, len(s) * 0.3):
        return False
    return True


def calib():
    """Đo filter trên 300 mẫu đã có nhãn Haiku (panel_mono_*.jsonl)."""
    RB = ROOT / "eval" / "review_bt"
    # chỉ số bị Haiku loại (ok=false), theo panel 00-04 ngày 2026-07-18
    rejects = set(
        [0, 1, 4, 5, 8, 10, 12, 16, 18, 19, 20, 26, 27, 28, 30, 33, 36, 38, 39, 40,
         42, 45, 47, 51, 54, 57, 58, 59,
         60, 61, 63, 66, 67, 69, 70, 71, 72, 73, 76, 80, 83, 84, 85, 89, 93, 94, 95,
         96, 97, 99, 101, 102, 103, 106, 112, 113, 117,
         122, 123, 124, 128, 130, 133, 134, 137, 139, 140, 141, 143, 145, 147, 149,
         150, 151, 153, 155, 156, 158, 159, 161, 162, 163, 170, 171, 172, 174, 175,
         181, 182, 185, 186, 189, 190, 191, 192, 193, 194, 196, 197, 199, 201, 202,
         203, 204, 205, 206, 208, 209, 211, 212, 214, 216, 219, 220, 221, 222, 223,
         224, 226, 228, 233, 235, 238, 239,
         241, 243, 245, 249, 251, 252, 253, 255, 257, 258, 262, 263, 265, 267, 268,
         272, 273, 277, 278, 279, 280, 281, 286, 287, 295, 296, 298])
    items = {}
    for f in sorted(RB.glob("panel_mono_*.jsonl")):
        for line in f.open(encoding="utf-8"):
            o = json.loads(line)
            items[o["i"]] = o["ja"]
    tp = fn = fp = tn = 0     # positive = filter LOẠI
    miss_bad, kill_good = [], []
    for i, s in items.items():
        bad_truth = i in rejects
        killed = not ok(s)
        if bad_truth and killed:
            tp += 1
        elif bad_truth and not killed:
            fn += 1
            miss_bad.append(s)
        elif not bad_truth and killed:
            fp += 1
            kill_good.append(s)
        else:
            tn += 1
    print(f"filter bắt rác: {tp}/{tp+fn} ({100*tp/(tp+fn):.0f}% recall) | "
          f"giết oan câu tốt: {fp}/{fp+tn} ({100*fp/(fp+tn):.0f}%)")
    print("--- rác LỌT (mẫu) ---")
    for s in miss_bad[:6]:
        print(" ", s[:80])
    print("--- tốt bị GIẾT OAN (mẫu) ---")
    for s in kill_good[:6]:
        print(" ", s[:80])


def main():
    lines = [l.strip() for l in SRC.open(encoding="utf-8") if l.strip()]
    kept = [s for s in lines if ok(s)]
    OUT.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"{len(lines):,} -> GIỮ {len(kept):,} ({100*len(kept)/len(lines):.1f}%) -> {OUT}")


if __name__ == "__main__":
    calib() if "--calib" in sys.argv else main()
