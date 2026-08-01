# -*- coding: utf-8 -*-
"""
Exp F — "KHÔNG TRAIN TOÀN MODEL" đi được tới đâu? Layer-wise reconstruction cho ternary.

Đây là hiện thân của trực giác "bố trí thông minh thay vì làm tròn ngây thơ" (AdaRound/BRECQ/
GPTQ), áp cho TOÀN Qwen3-0.6B — nhưng KHÔNG backprop toàn model, KHÔNG thầy, KHÔNG GPU:
mỗi ma trận linear được tối ưu ĐỘC LẬP để đầu ra khớp đầu ra gốc, trên ít activation mẫu.

So 3 mức:
  (1) FP32                         — trần trên.
  (2) PTQ ternary absmean (ngây thơ) — làm tròn, kỳ vọng sụp.
  (3) Layer-wise reconstruction     — tối ưu từng ma trận (STE) khớp output. Đây là "không train".

QUAN TRỌNG (trung thực): reconstruction dùng activation của model FP cho MỌI lớp (teacher
forcing kiểu AdaRound) → bỏ qua tích lũy sai số qua depth → kết quả là CHẶN TRÊN LẠC QUAN.
Nếu ngay cả chặn trên này vẫn sụp PPL thì kết luận cứng: không-train-toàn-model không tới
được ternary. Nếu nó ổn, cần thêm bản sequential (BRECQ) để xác nhận.
"""
import glob
import io
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
torch.set_num_threads(5)

MODEL_DIR = glob.glob(r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*")[0]
DEV_VI = r"D:\Bit-Translate-data\clean_v7g\dev.vi"
DEV_JA = r"D:\Bit-Translate-data\clean_v7g\dev.ja"
GROUP = 32
RECON_STEPS = 150
LR = 1.5e-3
N_CALIB_LINES = 40      # số câu (xen vi/ja) lấy activation calibration
N_EVAL_LINES = 16       # số câu mỗi ngôn ngữ để đo PPL
MAX_TOK = 96


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def read_lines(path, n):
    out = []
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line)
            if len(out) >= n:
                break
    return out


def ternary_g32(W):
    """absmean ternary per group-32 theo cột (giữ shape, đã dequant)."""
    R, C = W.shape
    pad = (GROUP - C % GROUP) % GROUP
    Wp = F.pad(W, (0, pad)) if pad else W
    Wg = Wp.view(R, -1, GROUP)
    scale = Wg.abs().mean(dim=2, keepdim=True).clamp(min=1e-8)
    Wt = (torch.round(Wg / scale).clamp(-1, 1) * scale).view(R, -1)[:, :C]
    return Wt


def ste_ternary(W):
    return W + (ternary_g32(W) - W).detach()


@torch.no_grad()
def eval_ppl(model, tok, lines, tag):
    total_nll, total_tok = 0.0, 0
    for s in lines:
        ids = tok(s, return_tensors="pt", truncation=True, max_length=MAX_TOK).input_ids
        if ids.shape[1] < 2:
            continue
        out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        total_nll += out.loss.item() * n
        total_tok += n
    ppl = float(np.exp(total_nll / max(total_tok, 1)))
    log(f"    PPL [{tag}] = {ppl:.2f}  ({total_tok} token)")
    return ppl


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"Nạp Qwen3-0.6B FP32 từ {MODEL_DIR}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    model.eval()

    # gom các nn.Linear trong decoder layers (bỏ lm_head/embedding)
    linears = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "layers." in name:
            linears.append((name, mod))
    log(f"Số ma trận linear sẽ xử lý: {len(linears)} (giữ FP: embedding tied, lm_head, mọi norm)")

    vi = read_lines(DEV_VI, N_CALIB_LINES // 2)
    ja = read_lines(DEV_JA, N_CALIB_LINES // 2)
    calib = [x for pair in zip(vi, ja) for x in pair]
    eval_vi = read_lines(DEV_VI, N_EVAL_LINES)[N_CALIB_LINES:] or read_lines(DEV_VI, 200)[-N_EVAL_LINES:]
    eval_ja = read_lines(DEV_JA, N_EVAL_LINES)[N_CALIB_LINES:] or read_lines(DEV_JA, 200)[-N_EVAL_LINES:]

    # ---- (1) baseline FP32 ----
    log("=== (1) FP32 baseline ===")
    ppl_fp_vi = eval_ppl(model, tok, eval_vi, "FP vi")
    ppl_fp_ja = eval_ppl(model, tok, eval_ja, "FP ja")

    # ---- thu activation đầu vào mỗi linear (teacher-forcing từ model FP) ----
    log("=== Thu activation calibration (1 lượt forward FP) ===")
    acts = {name: [] for name, _ in linears}
    handles = []

    def mk_hook(nm):
        def hook(mod, inp):
            acts[nm].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float16))
        return hook

    for name, mod in linears:
        handles.append(mod.register_forward_pre_hook(mk_hook(name)))
    with torch.no_grad():
        for s in calib:
            ids = tok(s, return_tensors="pt", truncation=True, max_length=MAX_TOK).input_ids
            model(ids)
    for h in handles:
        h.remove()
    for nm in acts:
        acts[nm] = torch.cat(acts[nm], dim=0)  # [N_tok, in]
    tot_mb = sum(v.numel() * 2 for v in acts.values()) / 1024 / 1024
    log(f"    activation lưu: {tot_mb:.0f} MB (float16)")

    # lưu bản trọng số gốc để dựng 2 kịch bản
    orig = {name: mod.weight.data.clone() for name, mod in linears}

    # ---- (2) PTQ ternary ngây thơ ----
    log("=== (2) PTQ ternary absmean (làm tròn ngây thơ) ===")
    with torch.no_grad():
        for name, mod in linears:
            mod.weight.data = ternary_g32(orig[name])
    ppl_ptq_vi = eval_ppl(model, tok, eval_vi, "PTQ vi")
    ppl_ptq_ja = eval_ppl(model, tok, eval_ja, "PTQ ja")

    # ---- (3) layer-wise reconstruction ----
    log(f"=== (3) Layer-wise reconstruction ({RECON_STEPS} step/ma trận, CPU) ===")
    t0 = time.time()
    err_before, err_after = [], []
    for i, (name, mod) in enumerate(linears):
        W0 = orig[name]
        X = acts[name].to(torch.float32)          # [N, in]
        target = (X @ W0.t()).detach()            # output gốc [N, out]
        Wfp = W0.clone().requires_grad_(True)
        opt = torch.optim.Adam([Wfp], lr=LR)
        with torch.no_grad():
            e0 = ((X @ ternary_g32(W0).t()) - target).norm() / target.norm()
        # track-best: KHÔNG BAO GIỜ nhận kết quả tệ hơn điểm khởi đầu round-to-nearest.
        # Bước 0 có Wfp==W0 nên tự ghi nhận chính điểm round-to-nearest làm mốc.
        best_loss = float("inf")
        best_W = ternary_g32(W0).detach().clone()
        for _ in range(RECON_STEPS):
            opt.zero_grad()
            Wq_step = ste_ternary(Wfp)           # forward = giá trị ternary (STE)
            pred = X @ Wq_step.t()
            loss = (pred - target).pow(2).mean()
            if loss.item() < best_loss:          # loss này CHÍNH LÀ loss của bản ternary
                best_loss = loss.item()
                best_W = ternary_g32(Wfp).detach().clone()
            loss.backward()
            opt.step()
        with torch.no_grad():
            e1 = ((X @ best_W.t()) - target).norm() / target.norm()
            mod.weight.data = best_W
        err_before.append(float(e0)); err_after.append(float(e1))
        acts[name] = None
        if (i + 1) % 28 == 0 or i == len(linears) - 1:
            log(f"    [{i+1:3d}/{len(linears)}] out-err TB: PTQ {np.mean(err_before)*100:.1f}% "
                f"-> recon {np.mean(err_after)*100:.1f}%  ({time.time()-t0:.0f}s)")
    ppl_rec_vi = eval_ppl(model, tok, eval_vi, "RECON vi")
    ppl_rec_ja = eval_ppl(model, tok, eval_ja, "RECON ja")

    print("\n" + "=" * 72)
    print("KẾT QUẢ — 'KHÔNG TRAIN TOÀN MODEL' ĐI ĐƯỢC TỚI ĐÂU (ternary ~1.58-bit)")
    print("=" * 72)
    print(f"{'':26s}{'PPL vi':>12s}{'PPL ja':>12s}")
    print(f"{'(1) FP32':26s}{ppl_fp_vi:12.1f}{ppl_fp_ja:12.1f}")
    print(f"{'(2) PTQ làm tròn':26s}{ppl_ptq_vi:12.1f}{ppl_ptq_ja:12.1f}")
    print(f"{'(3) Layer-wise recon':26s}{ppl_rec_vi:12.1f}{ppl_rec_ja:12.1f}")
    print("=" * 72)
    print(f"Sai số OUTPUT trung bình/lớp: PTQ {np.mean(err_before)*100:.1f}% -> recon {np.mean(err_after)*100:.1f}%")
    print("LƯU Ý: recon dùng activation FP mọi lớp (teacher-forcing) => CHẶN TRÊN LẠC QUAN.")
    print("Thực tế (sai số tích lũy qua 28 lớp) sẽ TỆ HƠN con số recon này.")


if __name__ == "__main__":
    main()
