#!/usr/bin/env python3
"""Mở rộng ĐỘ SÂU của checkpoint BitNet: 12 layer -> 18/20 layer, GIỮ NGUYÊN chất lượng.

Kỹ thuật: block expansion (LLaMA Pro 2024). Chèn thêm block mới, zero-init hai
projection cuối (attn.wo, ffn.down) => attn(x)=0 và ffn(x)=0 => block mới là
IDENTITY (x = x + 0 + 0). Model sau khi mở rộng cho output Y HỆT model gốc, rồi
train tiếp để layer mới học dần.

Vì sao dùng được với BitNet: weight_quant() có clamp(min=1e-5) ở mẫu số nên
weight toàn 0 vẫn ra 0 chứ không NaN (đã kiểm chứng).

Vì sao hướng SÂU chứ không RỘNG: (a) giữ d_model=768 nên kế thừa được toàn bộ
embedding + 12 layer đã train — đổi d_model thì phải train lại từ đầu; (b) fail
còn lại nặng nhất là câu dài / cấu trúc lồng nhau, vốn thiên về độ sâu.

  python scripts/grow_depth.py D:/.../v2_avg3.pt --layers 18 -o D:/.../grow18.pt
  # kiểm tra tính identity (bắt buộc):
  python scripts/grow_depth.py D:/.../v2_avg3.pt --layers 18 -o ... --verify
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from bitnet import BitNetLM, BitNetConfig  # noqa: E402


def build_cfg(c, n_layers=None, ls=0.0):
    return BitNetConfig(vocab_size=c["vocab_size"], d_model=c["d_model"],
                        n_layers=n_layers or c["n_layers"], n_heads=c["n_heads"],
                        d_ff=c["d_ff"], max_seq=c["max_seq"], label_smoothing=ls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--layers", type=int, required=True, help="số layer đích (vd 18, 20)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--verify", action="store_true",
                    help="kiểm tra model mới cho output Y HỆT model cũ (nên luôn bật)")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    c = dict(ck["cfg"])
    old_L = c["n_layers"]
    if a.layers <= old_L:
        raise SystemExit(f"--layers ({a.layers}) phải > số layer hiện có ({old_L})")
    n_new = a.layers - old_L

    # Chèn block mới RẢI ĐỀU chứ không dồn cuối: giữ được cấu trúc phân tầng
    # (tầng thấp học từ vựng, tầng cao học ngữ nghĩa) thay vì nhồi hết lên trên.
    # vd 12->18: chèn sau mỗi 2 block cũ.
    step = old_L / n_new
    insert_after = {int((i + 1) * step) - 1 for i in range(n_new)}
    while len(insert_after) < n_new:            # tránh trùng khi làm tròn
        insert_after.add(max(insert_after) + 1)
    insert_after = sorted(insert_after)[:n_new]

    sd = ck["model"]
    new_sd, new_idx = {}, 0
    plan = []
    for old_i in range(old_L):
        for k, v in sd.items():
            p = f"blocks.{old_i}."
            if k.startswith(p):
                new_sd[f"blocks.{new_idx}." + k[len(p):]] = v.clone()
        plan.append(f"{old_i}->{new_idx}")
        new_idx += 1
        if old_i in insert_after:
            # block MỚI: copy toàn bộ từ block vừa chép (giữ thống kê weight hợp lý),
            # rồi ZERO hai projection cuối -> identity.
            for k, v in sd.items():
                p = f"blocks.{old_i}."
                if k.startswith(p):
                    suf = k[len(p):]
                    new_sd[f"blocks.{new_idx}." + suf] = torch.zeros_like(v) if (
                        suf.startswith("attn.wo.") or suf.startswith("ffn.down.")) else v.clone()
            plan.append(f"NEW@{new_idx}")
            new_idx += 1

    for k, v in sd.items():                      # embed / output_norm / lm_head
        if not k.startswith("blocks."):
            new_sd[k] = v.clone()

    c["n_layers"] = a.layers
    model = BitNetLM(build_cfg(c))
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if unexpected:
        raise SystemExit(f"key thừa: {list(unexpected)[:5]}")
    if missing:
        print(f"CẢNH BÁO: thiếu {len(missing)} key (vd {missing[:3]}) — sẽ dùng init mặc định")

    old_p = BitNetLM(build_cfg(dict(ck["cfg"]))).num_params()
    print(f"{old_L}L ({old_p/1e6:.1f}M) -> {a.layers}L ({model.num_params()/1e6:.1f}M)")
    print("  thứ tự block:", " ".join(plan))

    if a.verify:
        old_model = BitNetLM(build_cfg(dict(ck["cfg"]))).eval()
        old_model.load_state_dict(ck["model"])
        model.eval()
        torch.manual_seed(0)
        x = torch.randint(0, c["vocab_size"], (2, 24))
        with torch.inference_mode():
            lo, _ = old_model(x)
            ln, _ = model(x)
        d = (lo - ln).abs().max().item()
        print(f"  VERIFY: sai lệch logits tối đa = {d:.3e}", end=" ")
        if d < 1e-4:
            print("-> ĐẠT (block mới là identity)")
        else:
            raise SystemExit("-> HỎNG: block mới KHÔNG identity, đừng train từ file này")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": {
        "vocab_size": c["vocab_size"], "d_model": c["d_model"], "n_layers": a.layers,
        "n_heads": c["n_heads"], "d_ff": c["d_ff"], "max_seq": c["max_seq"]},
        "step": 0, "from": f"{a.ckpt} (grow {old_L}L->{a.layers}L)"}, a.out)
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
