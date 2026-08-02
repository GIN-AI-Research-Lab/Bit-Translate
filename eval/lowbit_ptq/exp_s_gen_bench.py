# -*- coding: utf-8 -*-
"""
Exp S — GATE HÀNH VI trên máy B (CPU): sinh văn bản vi/ja/en/code + PPL 4 miền
cho từng checkpoint đã persist (1.56 QAT-v3, O1/O2/O3 S1).

LƯU Ý TRUNG THỰC VỀ TỐC ĐỘ: đây là mô phỏng HF fp32 (trọng số đã dequant) —
tok/s in ra KHÔNG phải tốc độ format thật. Tốc độ THẬT đo riêng bằng llama-bench
trên format đóng gói (TQ/IQ) — xem bảng bench kèm theo.
"""
import glob
import io
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "5")
os.environ.setdefault("MKL_NUM_THREADS", "5")
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_r_qat_lite import EN_EVAL, CODE_EVAL, eval_ppl, read_lines, LIN_PATHS  # noqa

sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(5)

MODEL_DIR = glob.glob(r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*")[0]
CKPT_DIR = r"D:\Bit-Translate-data\qat_ckpts"
DEV_VI = r"D:\Bit-Translate-data\clean_v7g\dev.vi"
DEV_JA = r"D:\Bit-Translate-data\clean_v7g\dev.ja"
OUT_MD = r"e:\Bit-Translate\eval\lowbit_ptq\exp_s_report.md"
OUT_JSON = r"e:\Bit-Translate\eval\lowbit_ptq\exp_s_results.json"

CKPTS = [
    ("FP32 (baseline)", None, 16.0),
    ("1.56bpw QAT-v3", "qat_lite_n3_v3.pt", 1.56),
    ("1.02bpw S1", "qat_s1_o1.pt", 1.02),
    ("0.70bpw S1", "qat_s1_o2.pt", 0.70),
    ("0.62bpw S1", "qat_s1_o3.pt", 0.62),
]

PROMPTS = {
    "vi": ["Hà Nội là thủ đô của nước",
           "Hôm nay trời đẹp nên tôi quyết định",
           "Trí tuệ nhân tạo là công nghệ"],
    "ja": ["東京は日本の",
           "今日は天気がいいので、",
           "日本で一番高い山は"],
    "en": ["The capital of France is",
           "Machine learning is a field of",
           "Once upon a time, there was a"],
    "code": ["def factorial(n):\n    ",
             "# reverse a string\ndef reverse_string(s):\n    ",
             "import math\n\ndef circle_area(radius):\n    "],
}
GEN_TOK = 60


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    log("Nạp FP32 + giữ bản gốc trong RAM")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    pristine = {k: v.clone() for k, v in model.state_dict().items()}

    eval_vi = read_lines(DEV_VI, 400)[-16:]
    eval_ja = read_lines(DEV_JA, 400)[-16:]

    def to_bias_linears():
        for blk in model.model.layers:
            for sub, name in LIN_PATHS:
                parent = getattr(blk, sub)
                lin = getattr(parent, name)
                if lin.bias is None:
                    nl = nn.Linear(lin.in_features, lin.out_features, bias=True)
                    nl.weight.data = lin.weight.data
                    nl.bias.data = torch.zeros(lin.out_features)
                    setattr(parent, name, nl)

    def to_plain():
        model.load_state_dict(pristine, strict=False)  # sau khi thay lại Linear không bias
        for blk in model.model.layers:
            for sub, name in LIN_PATHS:
                parent = getattr(blk, sub)
                lin = getattr(parent, name)
                if lin.bias is not None:
                    nl = nn.Linear(lin.in_features, lin.out_features, bias=False)
                    nl.weight.data = lin.weight.data
                    setattr(parent, name, nl)
        model.load_state_dict(pristine, strict=True)

    report = ["# Exp S — Gate hành vi 4 miền trên máy B (CPU)\n",
              "> tok/s dưới đây là MÔ PHỎNG HF fp32 — tốc độ format thật xem bảng llama-bench.\n"]
    results = {}

    for label, fname, bpw in CKPTS:
        log(f"=== {label} ===")
        if fname is None:
            to_plain()
        else:
            path = os.path.join(CKPT_DIR, fname)
            if not os.path.exists(path):
                log(f"    THIẾU {path} — bỏ qua")
                continue
            to_plain()
            to_bias_linears()
            sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
            model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
        model.eval()

        ppl = {
            "vi": eval_ppl(model, tok, eval_vi, "cpu"),
            "ja": eval_ppl(model, tok, eval_ja, "cpu"),
            "en": eval_ppl(model, tok, EN_EVAL, "cpu"),
            "code": eval_ppl(model, tok, CODE_EVAL, "cpu", max_tok=160),
        }
        log(f"    PPL vi {ppl['vi']:.1f} / ja {ppl['ja']:.1f} / en {ppl['en']:.1f} / code {ppl['code']:.1f}")

        report.append(f"\n## {label} (bpw {bpw})\n")
        report.append(f"PPL: vi **{ppl['vi']:.1f}** · ja **{ppl['ja']:.1f}** · "
                      f"en **{ppl['en']:.1f}** · code **{ppl['code']:.1f}**\n")
        gens, n_tok, t_gen = {}, 0, 0.0
        for dom, plist in PROMPTS.items():
            gens[dom] = []
            for p in plist:
                ids = tok(p, return_tensors="pt").input_ids
                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=GEN_TOK, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
                dt = time.time() - t0
                new = out[0][ids.shape[1]:]
                txt = tok.decode(new, skip_special_tokens=True)
                gens[dom].append(txt)
                n_tok += len(new)
                t_gen += dt
                report.append(f"\n**[{dom}]** `{p!r}`\n\n> {txt.strip()[:400]}\n")
        sim_tps = n_tok / max(t_gen, 1e-9)
        report.append(f"\n*tok/s mô phỏng fp32: {sim_tps:.1f}*\n")
        log(f"    sinh {n_tok} token, {sim_tps:.1f} tok/s (mô phỏng)")
        results[label] = {"bpw": bpw, "ppl": ppl, "sim_tps": round(sim_tps, 1), "gens": gens}

        with io.open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with io.open(OUT_MD, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
    log(f"XONG. Báo cáo: {OUT_MD}")


if __name__ == "__main__":
    main()
