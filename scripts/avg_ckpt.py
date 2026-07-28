#!/usr/bin/env python3
"""Checkpoint averaging — trung bình master weights của N checkpoint cuối.

Trick MT cổ điển (Vaswani 2017 §5.4, Marian/fairseq đều dùng): trung bình vài
checkpoint cuối làm phẳng nhiễu SGD quanh cực tiểu, thường được +0,3-1 chrF
MIỄN PHÍ — không train thêm, không đổi kiến trúc.

Với BitNet: average trên MASTER WEIGHTS FP (thứ đang lưu trong .pt), rồi mới ép
ternary lúc inference như bình thường. Đúng thứ tự — average sau khi quantize là
sai (trung bình của {-1,0,1} không còn ternary).

  python scripts/avg_ckpt.py checkpoints/kd_step{11000,12000,13000,14000,15000}.pt \
      -o checkpoints/kd_avg5.pt
"""
import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="các file .pt để trung bình")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    paths = [Path(p) for p in a.ckpts]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit("thiếu file: " + ", ".join(str(p) for p in missing))

    acc, cfg, steps, n = None, None, [], 0
    for p in paths:
        ck = torch.load(p, map_location="cpu")
        sd = ck["model"]
        steps.append(ck.get("step", "?"))
        if acc is None:
            cfg = ck["cfg"]
            acc = {k: v.detach().to(torch.float64).clone() if v.is_floating_point()
                   else v.detach().clone() for k, v in sd.items()}
        else:
            if set(sd) != set(acc):
                raise SystemExit(f"{p.name}: state_dict khác khoá — không average được")
            for k, v in sd.items():
                if acc[k].is_floating_point():
                    acc[k] += v.detach().to(torch.float64)
        n += 1
        print(f"  + {p.name} (step {steps[-1]})", flush=True)

    ref = torch.load(paths[0], map_location="cpu")["model"]
    out = {}
    for k, v in acc.items():
        out[k] = (v / n).to(ref[k].dtype) if v.is_floating_point() else v

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": out, "cfg": cfg, "step": f"avg{n}({steps[0]}-{steps[-1]})"}, a.out)
    print(f"\nĐã trung bình {n} checkpoint -> {a.out}")


if __name__ == "__main__":
    main()
