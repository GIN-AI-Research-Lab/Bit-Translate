#!/usr/bin/env python3
"""Mở rộng embedding của checkpoint đã train thêm N hàng (khớp tokenizer vừa
thêm token mới qua add_fix_token.py — vocab 32000 -> 32001). embed.weight và
lm_head.weight TIED (cùng 1 tensor, xem src/bitnet.py) nên chỉ cần mở 1 chỗ.

  python scripts/expand_checkpoint_vocab.py --ckpt <in.pt> --out <out.pt> [--new-vocab 32001]

KHÔNG giữ optimizer state (shape cũ không khớp vocab mới — train.py đã sửa để
tự bỏ qua opt state không khớp, dùng optimizer mới tinh cho hàng embedding mới
lẫn toàn bộ mô hình ở v2, hợp lý vì đây coi như khởi động lại pha train mới).
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--new-vocab", type=int, default=32001)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]
    # BUG ĐÃ SỬA: embed.weight và lm_head.weight TIED trong model đang chạy (cùng
    # nn.Parameter, xem src/bitnet.py) nhưng torch.save() state_dict() LƯU THÀNH
    # 2 KEY RIÊNG (không phải cùng 1 tensor object khi load lại) -> chỉ mở
    # "embed.weight" mà quên "lm_head.weight" gây lỗi thật: "size mismatch for
    # lm_head.weight: ... torch.Size([32000, 768])" khi resume train.py.
    KEYS = ["embed.weight", "lm_head.weight"]
    for key in KEYS:
        assert key in sd, f"không thấy {key} trong checkpoint — kiểm tra lại tên tham số"
    old = sd[KEYS[0]]
    old_vocab, d_model = old.shape
    add = args.new_vocab - old_vocab
    assert add >= 0, f"new-vocab ({args.new_vocab}) < vocab hiện tại ({old_vocab})"
    if add == 0:
        print(f"vocab đã là {old_vocab}, không cần mở rộng.")
    else:
        extra = torch.empty(add, d_model)
        torch.nn.init.normal_(extra, mean=0.0, std=0.02)  # khớp init gốc (xem src/bitnet.py)
        new = torch.cat([old, extra], dim=0)
        print(f"embed.weight/lm_head.weight: {tuple(old.shape)} -> {tuple(new.shape)} "
              f"(+{add} hàng, init N(0,0.02), CÙNG giá trị cho cả 2 key vì tied)")
        for key in KEYS:
            sd[key] = new.clone()

    cfg = dict(ck.get("cfg", {}))
    cfg["vocab_size"] = args.new_vocab

    torch.save({"model": sd, "step": ck.get("step", 0), "cfg": cfg}, args.out)
    print(f"GHI (KHÔNG kèm optimizer state — train.py sẽ tự khởi động optimizer mới) -> {args.out}")


if __name__ == "__main__":
    main()
