#!/usr/bin/env python3
"""Checkpoint averaging — trung bình trọng số N checkpoint (mẹo NMT chuẩn, thường
+0.2-0.5 chrF so bản cuối; Vaswani 2017 dùng avg 5 bản cuối).

  python scripts/avg_checkpoints.py --out checkpoints/avg.pt \
         checkpoints/step30500.pt checkpoints/step31000.pt ...

Chỉ lấy state_dict "model" (bỏ opt), cộng dồn fp64 rồi chia — an toàn số học.
Tensor không phải float (nếu có) lấy từ bản ĐẦU. Giữ "cfg" + "step" của bản CUỐI
để convert_to_gguf.py tự đọc dims như checkpoint thường.
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("ckpts", nargs="+", help="các checkpoint .pt (thứ tự step tăng dần)")
    args = ap.parse_args()
    assert len(args.ckpts) >= 2, "cần >=2 checkpoint để trung bình"

    acc, meta, first_sd = None, None, None
    for i, p in enumerate(args.ckpts):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        meta = {"step": ck.get("step", 0), "cfg": ck.get("cfg")}  # bản cuối thắng
        print(f"  [{i+1}/{len(args.ckpts)}] {p} (step {meta['step']})", flush=True)
        if acc is None:
            first_sd = {k: v for k, v in sd.items() if not v.is_floating_point()}
            acc = {k: v.to(torch.float64) for k, v in sd.items() if v.is_floating_point()}
            keys = set(sd)
        else:
            assert set(sd) == keys, f"{p}: state_dict lệch key so bản đầu"
            for k in acc:
                acc[k] += sd[k].to(torch.float64)
        del ck, sd

    n = len(args.ckpts)
    out_sd = {k: (v / n).to(torch.float32) for k, v in acc.items()}
    out_sd.update(first_sd)
    torch.save({"model": out_sd, "step": meta["step"], "cfg": meta["cfg"]}, args.out)
    print(f"OK: avg {n} bản -> {args.out} (step ghi = {meta['step']}, cfg giữ nguyên)")


if __name__ == "__main__":
    main()
