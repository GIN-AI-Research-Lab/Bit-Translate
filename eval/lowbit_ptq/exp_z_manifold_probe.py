# -*- coding: utf-8 -*-
"""
Exp Z (H2-probe rẻ-trước) — đo CHIỀU NỘI TẠI của manifold activation tại input MLP
(block giữa, Qwen3-0.6B). H2 (response-surface layer) chỉ sống nếu activation thật
nằm trên manifold thấp chiều (bảng tra 64-128 chiều phủ được).

Kỳ vọng ghi trước: dim@95% > 200 -> H2 CHẾT không cần xây bảng (xác suất ~65%).
"""
import glob
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
sys.path.insert(0, r"e:\Bit-Translate\eval\lowbit_ptq")
from exp_r_qat_lite import CAL_CODE, CAL_EN, CAL_MATH, CAL_ZH, read_lines  # noqa: E402

MODEL_GLOB = r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*"
DVI = r"D:\Bit-Translate-data\clean_v7g\dev.vi"
DJA = r"D:\Bit-Translate-data\clean_v7g\dev.ja"
BLOCK = 14


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mdir = glob.glob(MODEL_GLOB)[0]
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32).eval()
    texts = (read_lines(DVI, 300) + read_lines(DJA, 300)
             + CAL_EN * 4 + CAL_CODE * 4 + CAL_ZH * 4 + CAL_MATH * 4)

    feats = []
    blk = model.model.layers[BLOCK]

    def hook(mod, inp):
        feats.append(inp[0].detach().reshape(-1, inp[0].shape[-1]))
    h = blk.mlp.gate_proj.register_forward_pre_hook(hook)
    with torch.no_grad():
        for s in texts:
            ids = tok(s, return_tensors="pt", truncation=True, max_length=96).input_ids
            if ids.shape[1] >= 4:
                model(ids)
    h.remove()
    X = torch.cat(feats, 0)                          # [N, 1024]
    print(f"thu {X.shape[0]} vector activation (block {BLOCK}, dim {X.shape[1]})")
    X = X - X.mean(0, keepdim=True)
    # PCA qua SVD trên mẫu (subsample nếu quá lớn)
    if X.shape[0] > 20000:
        X = X[torch.randperm(X.shape[0])[:20000]]
    S = torch.linalg.svdvals(X)
    var = S.pow(2)
    cum = var.cumsum(0) / var.sum()
    for thr in (0.90, 0.95, 0.99):
        k = int((cum < thr).sum().item()) + 1
        print(f"dim @ {int(thr*100)}% phương sai: {k}")
    k95 = int((cum < 0.95).sum().item()) + 1
    verdict = ("MANIFOLD THẤP CHIỀU — H2 đáng xây bảng thử" if k95 <= 128 else
               ("VÙNG XÁM (128-200) — cân nhắc" if k95 <= 200 else
                "CHIỀU CAO — H2 CHẾT (đúng nhánh ~65% kỳ vọng ghi trước)"))
    print(f"PHÁN QUYẾT probe: {verdict}")


if __name__ == "__main__":
    main()
