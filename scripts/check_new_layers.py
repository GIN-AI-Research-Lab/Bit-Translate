#!/usr/bin/env python3
"""Đo xem các block MỚI chèn vào (grow_depth) có thực sự học hay vẫn nằm im ở identity.

Vì sao cần: block mới khởi tạo zero-init `attn.wo` + `ffn.down` nên đóng góp ban đầu
= 0. Gradient qua chúng lúc đầu nhỏ, model có thể "lười" và giữ chúng gần identity —
tức là thêm tham số nhưng KHÔNG dùng. Đây là chỉ số quyết định:

  - Nếu norm hai projection đó của block mới vẫn ~0 sau train ⇒ model không cần thêm
    chỗ chứa ⇒ DUNG LƯỢNG KHÔNG PHẢI NÚT THẮT, đừng scale tiếp, dồn vào data.
  - Nếu chúng lên cỡ tương đương block cũ ⇒ model đã dùng hết chỗ mới ⇒ scale tiếp
    có cơ sở.

  python scripts/check_new_layers.py D:/.../v3_avg.pt --new 2,5,8,11,14,17
  # --new = vị trí block mới, in ra bởi grow_depth.py (dòng "thứ tự block")
"""
import argparse
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--new", default="2,5,8,11,14,17",
                    help="chỉ số block mới, phân tách bằng dấu phẩy")
    a = ap.parse_args()
    new = {int(x) for x in a.new.split(",") if x.strip()}

    sd = torch.load(a.ckpt, map_location="cpu")["model"]
    nL = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))

    rows = []
    for i in range(nL):
        wo = sd.get(f"blocks.{i}.attn.wo.weight")
        dn = sd.get(f"blocks.{i}.ffn.down.weight")
        if wo is None or dn is None:
            continue
        rows.append((i, i in new, wo.float().norm().item(), dn.float().norm().item()))

    old_wo = [r[2] for r in rows if not r[1]]
    old_dn = [r[3] for r in rows if not r[1]]
    ref_wo = sum(old_wo) / max(1, len(old_wo))
    ref_dn = sum(old_dn) / max(1, len(old_dn))

    print(f"{'block':>6} {'loại':>5} {'|attn.wo|':>11} {'|ffn.down|':>12} "
          f"{'% so block cũ':>15}")
    print("-" * 56)
    for i, is_new, w, d in rows:
        pct = 100 * (w / ref_wo + d / ref_dn) / 2
        print(f"{i:6d} {'MỚI' if is_new else 'cũ':>5} {w:11.2f} {d:12.2f} "
              f"{pct:14.1f}%")

    nw = [r for r in rows if r[1]]
    if nw:
        avg = sum(100 * (w / ref_wo + d / ref_dn) / 2 for _, _, w, d in nw) / len(nw)
        print("-" * 56)
        print(f"Block mới trung bình đạt {avg:.1f}% độ lớn của block cũ.")
        if avg < 5:
            print("=> Block mới VẪN GẦN IDENTITY. Model không dùng chỗ chứa mới.")
            print("   KẾT LUẬN: dung lượng KHÔNG phải nút thắt — dừng scale, dồn vào data.")
        elif avg < 40:
            print("=> Block mới có học nhưng đóng góp còn nhỏ. Cân nhắc train thêm step")
            print("   trước khi kết luận về dung lượng.")
        else:
            print("=> Block mới đã hoạt động thực sự. Scale tiếp (18L->24L) có cơ sở.")


if __name__ == "__main__":
    main()
