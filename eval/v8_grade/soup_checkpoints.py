#!/usr/bin/env python3
"""Model-soup: trộn trọng số v7a_avg (step 8750) va v8 (step 16750).
Chay duoc tren CPU (khong can GPU) — chi la trung binh co trong so state_dict.

Dung:
  python soup_checkpoints.py --v7a path/v7a_avg.pt --v8 path/v8_step16750.pt \
      --alpha 0.5 --out path/soup_a0.5.pt
  # alpha = trong so cho v8; 0.5 = trung binh deu; 0.3 = nghieng ve v7a; 0.7 = nghieng v8

Sau khi tao soup, eval lai 1199 cau (tren Modal L40S cho nhanh, hoac CPU cham):
  modal run cloud/modal_train_v8.py::eval_1200 --ckpt soup_a0.5.pt --tag soupA05
  (nho upload soup*.pt len volume checkpoints_v8/ truoc)

Goi y: quet alpha 0.3 / 0.5 / 0.7, cham lai tap-64 + tap-150, chon alpha
an-ca-hai-dau (giu cu sua 64 ma khong regression pho thong).
"""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7a", required=True, help="checkpoint v7a_avg (step 8750)")
    ap.add_argument("--v8", required=True, help="checkpoint v8 (step 16750)")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="trong so cho v8 (0..1). out = (1-a)*v7a + a*v8")
    ap.add_argument("--out", required=True, help="duong dan luu soup")
    args = ap.parse_args()

    a = args.alpha
    assert 0.0 <= a <= 1.0
    print(f">> soup: (1-{a})*v7a + {a}*v8")
    c7 = torch.load(args.v7a, map_location="cpu")
    c8 = torch.load(args.v8, map_location="cpu")
    s7 = c7["model"] if "model" in c7 else c7
    s8 = c8["model"] if "model" in c8 else c8

    assert set(s7.keys()) == set(s8.keys()), "hai checkpoint khac kien truc!"
    out = {}
    for k in s8.keys():
        t7, t8 = s7[k], s8[k]
        if t8.dtype.is_floating_point:
            out[k] = (1 - a) * t7.float() + a * t8.float()
            out[k] = out[k].to(t8.dtype)
        else:
            # buffer nguyen (vd position ids) -> lay cua v8
            out[k] = t8.clone()

    # giu cfg cua v8 (v7a_avg co the thieu cfg)
    cfg = c8.get("cfg", c7.get("cfg"))
    saved = {"model": out, "cfg": cfg, "step": c8.get("step", 0),
             "from": f"soup (1-{a})*v7a + {a}*v8"}
    torch.save(saved, args.out)
    print(f">> luu {args.out} (cfg={'co' if cfg else 'THIEU'}, "
          f"{sum(v.numel() for v in out.values())/1e6:.1f}M tham so)")


if __name__ == "__main__":
    main()
