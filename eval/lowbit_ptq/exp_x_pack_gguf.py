# -*- coding: utf-8 -*-
"""
Exp X — ĐÓNG GÓI 30B ternary+LoRA thành GGUF TQ2_0 chạy thật trên CPU (pha C).

Đường đi: ckpt bake exp_v (ternary g64-f8 + bias + LoRA exp_w)
  1) merge LoRA vào W (bf16)              -> giữ công KD trong trọng số
  2) refit DENSE-ternary g256-f16          -> nằm ĐÚNG lưới TQ2_0 (zeros của mask giữ nguyên)
  3) bias -> 0 (GGUF Qwen3MoE không bias)  -> THUẾ đo được, in PPL từng bước
  4) đo PPL 6 miền (stream) trước/sau      -> biết trả bao nhiêu chất lượng cho tính-chạy-được
  5) save HF fp16 (giá trị t×s) + convert_hf_to_gguf + llama-quantize TQ2_0
Kết quả: /vol/out/qwen3-30b-a3b-ternKD-TQ2_0.gguf (~8.5GB) — tải về máy B chạy llama.cpp.
"""
import argparse
import io
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root")
from exp_r_qat_lite import (CODE_EVAL, EN_EVAL, MATH_EVAL, ZH_EVAL, read_lines)  # noqa: E402
from exp_v_s1_stream import stream_sweep  # noqa: E402
from exp_w_lora_kd import to_bias_linears  # noqa: E402

OUT_DIR = os.environ.get("EXPR_OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
CONVERT_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@torch.no_grad()
def refit_g256(W, G=256):
    """Refit dense-ternary scale f16 per-256 (3-iter Lloyd như LearnQLinear init, KHÔNG mask).
    Trả weight đã bake (t*s) nằm đúng lưới TQ2_0."""
    R, C = W.shape
    pad = (G - C % G) % G
    Wp = torch.nn.functional.pad(W.float(), (0, pad)) if pad else W.float()
    Wv = Wp.view(R, -1, G)
    s = Wv.abs().mean(2, keepdim=True).clamp(min=1e-8)
    for _ in range(3):
        t = torch.round(Wv / s).clamp(-1, 1)
        num = (Wv * t).sum(2, keepdim=True)
        den = (t * t).sum(2, keepdim=True).clamp(min=1e-8)
        s = (num / den).abs().clamp(min=1e-8)
    s = s.half().float()                     # lưới f16 của TQ2_0
    q = (torch.round(Wv / s).clamp(-1, 1) * s).view(R, -1)[:, :C]
    return q.to(W.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--ckpt", default="/vol/out/expv_Qwen3-30B-A3B_2x4.pt")
    ap.add_argument("--lora", default="/vol/out/expw_lora_30b.pt")
    ap.add_argument("--dev-vi", default="/root/qat_data/dev.vi")
    ap.add_argument("--dev-ja", default="/root/qat_data/dev.ja")
    ap.add_argument("--outdir", default="/vol/out/pack30b")
    ap.add_argument("--gguf-name", default="qwen3-30b-a3b-ternKD-TQ2_0.gguf")
    ap.add_argument("--eval", type=int, default=1)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    log("nạp skeleton + ckpt ternary...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    to_bias_linears(model)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = sd["state_dict"] if "state_dict" in sd else sd
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected: {unexpected[:5]}"

    # ---- 1) merge LoRA vào W ----
    if args.lora and os.path.exists(args.lora):
        lo = torch.load(args.lora, map_location="cpu", weights_only=False)
        lora = lo["lora"] if "lora" in lo else lo
        log(f"merge LoRA ({lo.get('meta', {})})...")
        n_merged = 0
        msd = dict(model.named_parameters())
        for k in list(lora.keys()):
            if not k.endswith(".A"):
                continue
            base = k[:-2]                     # "...q_proj" (module path của LoRALinear)
            A = lora[k].float()               # [r, in]
            B = lora[base + ".B"].float()     # [out, r]
            wkey = base + ".base.weight"      # tên khi LoRALinear bọc — trong model THƯỜNG là base+".weight"
            pw = msd.get(base + ".weight")
            if pw is None:
                log(f"  !! không thấy {base}.weight — bỏ")
                continue
            pw.data.add_((B @ A).to(pw.dtype))
            n_merged += 1
        log(f"merge {n_merged} LoRA module vào trọng số")

    # ---- chuẩn bị bộ đo ----
    def build_probe_seqs():
        dev_vi = read_lines(args.dev_vi, 400)[-16:]
        dev_ja = read_lines(args.dev_ja, 400)[-16:]
        probes = {"vi": dev_vi, "ja": dev_ja, "en": EN_EVAL, "code": CODE_EVAL,
                  "zh": ZH_EVAL, "math": MATH_EVAL}
        seqs, dom = [], {}
        for name, lines in probes.items():
            dom[name] = []
            for s_ in lines:
                ids = tok(s_, return_tensors="pt", truncation=True,
                          max_length=(160 if name == "code" else 96)).input_ids
                if ids.shape[1] >= 2:
                    dom[name].append(len(seqs))
                    seqs.append(ids)
        return seqs, dom

    results = {}
    if args.eval:
        seqs, dom = build_probe_seqs()
        _, _, nll = stream_sweep(model, seqs, dev, nll_domains=dom)
        results["merged_g64_bias"] = nll
        log("PPL sau merge (g64+bias): " + " / ".join(f"{k} {v:.1f}" for k, v in nll.items()))

    # ---- 2)+3) refit g256 + bias->0 ----
    log("refit dense-ternary g256-f16 (lưới TQ2_0) + bias->0...")
    t0 = time.time()
    for blk in model.model.layers:
        moe = hasattr(blk.mlp, "experts")
        subs = [blk.self_attn] + (list(blk.mlp.experts) if moe else [blk.mlp])
        for sub in subs:
            for name, lin in list(sub.named_children()):
                if isinstance(lin, nn.Linear) and "norm" not in name and name != "gate":
                    lin.weight.data = refit_g256(lin.weight.data)
                    if lin.bias is not None:
                        lin.bias.data.zero_()
    log(f"refit xong ({time.time()-t0:.0f}s)")
    if args.eval:
        _, _, nll2 = stream_sweep(model, seqs, dev, nll_domains=dom)
        results["tq2native_nobias"] = nll2
        log("PPL TQ2-native (g256, no-bias): "
            + " / ".join(f"{k} {v:.1f}" for k, v in nll2.items()))
        v100 = {"vi100": [], "ja100": []}
        vseqs = []
        for nm, path in (("vi100", args.dev_vi), ("ja100", args.dev_ja)):
            for s_ in read_lines(path, 200)[-100:]:
                ids = tok(s_, return_tensors="pt", truncation=True, max_length=96).input_ids
                if ids.shape[1] >= 2:
                    v100[nm].append(len(vseqs))
                    vseqs.append(ids)
        _, _, nv = stream_sweep(model, vseqs, dev, nll_domains=v100)
        results["val100"] = nv
        log(f"val-100 TQ2-native: vi {nv['vi100']:.1f} | ja {nv['ja100']:.1f}")

    # ---- 4) save HF fp16 KHÔNG bias (đúng skeleton gốc) ----
    log("gỡ bias + save HF fp16...")
    for blk in model.model.layers:
        moe = hasattr(blk.mlp, "experts")
        subs = [blk.self_attn] + (list(blk.mlp.experts) if moe else [blk.mlp])
        for sub in subs:
            for name, lin in list(sub.named_children()):
                if isinstance(lin, nn.Linear) and lin.bias is not None \
                        and "norm" not in name and name != "gate":
                    nl = nn.Linear(lin.in_features, lin.out_features, bias=False,
                                   dtype=torch.float16)
                    nl.weight.data = lin.weight.data.to(torch.float16)
                    setattr(sub, name, nl)
    hf_dir = os.path.join(args.outdir, "hf")
    model.half().save_pretrained(hf_dir)
    tok.save_pretrained(hf_dir)
    del model
    log(f"HF fp16: {hf_dir}")

    # ---- 5) convert -> GGUF f16 -> quantize TQ2_0 ----
    conv = os.path.join(args.outdir, "convert_hf_to_gguf.py")
    if not os.path.exists(conv):
        urllib.request.urlretrieve(CONVERT_URL, conv)
    f16 = os.path.join(args.outdir, "model-f16.gguf")
    log("convert_hf_to_gguf -> f16...")
    r = subprocess.run([sys.executable, conv, hf_dir, "--outfile", f16, "--outtype", "f16"],
                       capture_output=True, text=True, timeout=7200)
    if r.returncode != 0 or not os.path.exists(f16):
        print(r.stdout[-1500:], r.stderr[-3000:])
        sys.exit("convert FAIL")
    log(f"f16 gguf: {os.path.getsize(f16)/1e9:.1f} GB")
    tq2 = os.path.join(args.outdir, args.gguf_name)
    log("llama-quantize TQ2_0...")
    r = subprocess.run(["/root/llama.cpp/build/bin/llama-quantize", f16, tq2, "TQ2_0", "8"],
                       capture_output=True, text=True, timeout=7200)
    if r.returncode != 0 or not os.path.exists(tq2):
        print(r.stdout[-1500:], r.stderr[-3000:])
        sys.exit("quantize FAIL")
    log(f"TQ2_0 gguf: {os.path.getsize(tq2)/1e9:.2f} GB")
    os.remove(f16)

    rj = os.path.join(OUT_DIR, "exp_x_results.json")
    results["gguf"] = {"path": tq2, "gb": round(os.path.getsize(tq2) / 1e9, 2)}
    with io.open(rj, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log("EXP X XONG")


if __name__ == "__main__":
    main()
