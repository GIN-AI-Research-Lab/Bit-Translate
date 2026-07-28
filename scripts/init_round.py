#!/usr/bin/env python3
"""Biến một checkpoint bất kỳ thành ĐIỂM KHỞI ĐẦU của vòng train mới.

Vì sao cần: train.py resume bằng `step = ck["step"]`, mà `avg_ckpt.py` ghi step là
CHUỖI ("avg3(7000-8000)") -> resume thẳng từ file avg sẽ nổ TypeError khi so sánh
step < max_steps. Ngoài ra optimizer state của vòng cũ không còn đúng với corpus
mới (moment ước lượng trên phân bố data cũ) nên bỏ luôn cho sạch.

  python scripts/init_round.py D:/Bit-Translate-data/checkpoints_v3/v3_avg.pt \
      -o D:/Bit-Translate-data/checkpoints_v4/last.pt
"""
import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--step", type=int, default=0)
    a = p.parse_args()

    ck = torch.load(a.src, map_location="cpu")
    cfg = ck["cfg"]
    nparam = sum(v.numel() for v in ck["model"].values())
    print(f"nguồn : {a.src}")
    print(f"  step={ck.get('step')} | {cfg['n_layers']}L d{cfg['d_model']} "
          f"| {nparam/1e6:.1f}M tham số | {'CÓ' if 'opt' in ck else 'không'} optimizer")

    out = {"model": ck["model"], "cfg": cfg, "step": a.step,
           "from": f"{Path(a.src).name}@{ck.get('step')}"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    print(f"-> {a.out} (step={a.step}, không optimizer state)")


if __name__ == "__main__":
    main()
