# -*- coding: utf-8 -*-
"""
Exp A — Thang PTQ thấp-bit THẬT trên Qwen3-0.6B, đo PPL bằng llama.cpp b10199.

Quy trình (tuần tự, máy memory-bound — không chạy song song):
  1. Dựng calib.txt (imatrix) từ train.vi + train.ja; test_vi/test_ja từ dev (held-out).
  2. llama-imatrix trên Q8_0 + calib.txt.
  3. Quantize từ Q8_0 (--allow-requantize; Q8_0 sai số <0.1% nên chấp nhận được làm nguồn)
     xuống: IQ1_S, IQ1_M, TQ1_0, TQ2_0, IQ2_XXS, IQ2_XS, IQ2_S, Q2_K, IQ3_XXS + Q4_K_M tham chiếu.
  4. llama-perplexity từng bản trên test_vi.txt và test_ja.txt.
  5. llama-cli 4 câu hành vi trên các bản đáng chú ý.
Kết quả: exp_a_results.json + log đầy đủ trong thư mục làm việc.
"""
import io
import json
import os
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

BIN = r"D:\Bit-Translate-data\llama_cpp_new"
SRC_Q8 = r"D:\Bit-Translate-data\models\Qwen3-0.6B-Q8_0.gguf"
REF_Q4 = r"D:\Bit-Translate-data\ref_models\Qwen3-0.6B-Q4_K_M.gguf"
TRAIN_VI = r"D:\Bit-Translate-data\clean_v7g\train.vi"
TRAIN_JA = r"D:\Bit-Translate-data\clean_v7g\train.ja"
DEV_VI = r"D:\Bit-Translate-data\clean_v7g\dev.vi"
DEV_JA = r"D:\Bit-Translate-data\clean_v7g\dev.ja"
WORK = r"D:\Bit-Translate-data\ptq_ladder"
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_a_results.json")

THREADS = "6"
CALIB_BYTES_PER_LANG = 200_000   # ~200KB mỗi thứ tiếng cho imatrix
TEST_BYTES_PER_LANG = 150_000    # ~150KB mỗi thứ tiếng cho PPL

LADDER = ["IQ1_S", "IQ1_M", "TQ1_0", "TQ2_0", "IQ2_XXS", "IQ2_XS", "IQ2_S", "Q2_K", "IQ3_XXS", "Q4_K_M"]
BEHAV_MODELS = ["TQ2_0", "IQ2_XXS", "IQ2_S", "Q2_K", "Q4_K_M"]
BEHAV_PROMPTS = [
    "Thủ đô của Nhật Bản là gì? Trả lời ngắn gọn. /no_think",
    "Dịch câu sau sang tiếng Nhật: Sáng nay trời rất đẹp. /no_think",
    "12 + 35 bằng bao nhiêu? /no_think",
    "日本の首都はどこですか？短く答えてください。 /no_think",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(args, log_name, timeout=3600):
    """Chạy lệnh, ghi toàn bộ output ra file log, trả (rc, text)."""
    log_path = os.path.join(WORK, log_name)
    t0 = time.time()
    p = subprocess.run(args, capture_output=True, timeout=timeout)
    text = (p.stdout or b"").decode("utf-8", "replace") + "\n" + (p.stderr or b"").decode("utf-8", "replace")
    with io.open(log_path, "w", encoding="utf-8") as f:
        f.write(" ".join(str(a) for a in args) + "\n\n" + text)
    log(f"  {log_name}: rc={p.returncode}, {time.time()-t0:.0f}s")
    return p.returncode, text


def take_slice(src, n_bytes, skip_lines=0):
    out, size = [], 0
    with io.open(src, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
            line = line.strip()
            if not line:
                continue
            out.append(line)
            size += len(line.encode("utf-8")) + 1
            if size >= n_bytes:
                break
    return out


def build_texts():
    calib = os.path.join(WORK, "calib.txt")
    test_vi = os.path.join(WORK, "test_vi.txt")
    test_ja = os.path.join(WORK, "test_ja.txt")
    if all(os.path.exists(p) for p in (calib, test_vi, test_ja)):
        log("Text calib/test đã có, dùng lại.")
        return calib, test_vi, test_ja
    # calib: xen kẽ vi/ja từ TRAIN (bỏ 100k dòng đầu để tránh trùng miền đầu file)
    vi = take_slice(TRAIN_VI, CALIB_BYTES_PER_LANG, skip_lines=100_000)
    ja = take_slice(TRAIN_JA, CALIB_BYTES_PER_LANG, skip_lines=100_000)
    inter = []
    for a, b in zip(vi, ja):
        inter.append(a)
        inter.append(b)
    with io.open(calib, "w", encoding="utf-8") as f:
        f.write("\n".join(inter))
    # test: dev held-out
    with io.open(test_vi, "w", encoding="utf-8") as f:
        f.write("\n".join(take_slice(DEV_VI, TEST_BYTES_PER_LANG)))
    with io.open(test_ja, "w", encoding="utf-8") as f:
        f.write("\n".join(take_slice(DEV_JA, TEST_BYTES_PER_LANG)))
    log(f"Đã dựng calib ({os.path.getsize(calib)//1024}KB), test_vi ({os.path.getsize(test_vi)//1024}KB), test_ja ({os.path.getsize(test_ja)//1024}KB)")
    return calib, test_vi, test_ja


def main():
    os.makedirs(WORK, exist_ok=True)
    results = {"source": SRC_Q8, "note": "quantize từ Q8_0 (--allow-requantize), imatrix vi+ja", "quants": {}}

    calib, test_vi, test_ja = build_texts()

    # 1) imatrix
    imat = os.path.join(WORK, "imatrix_q8.gguf")
    if not os.path.exists(imat):
        rc, _ = run([os.path.join(BIN, "llama-imatrix.exe"), "-m", SRC_Q8, "-f", calib,
                     "-o", imat, "-t", THREADS, "-c", "512"], "imatrix.log", timeout=5400)
        if rc != 0 or not os.path.exists(imat):
            log("imatrix THẤT BẠI — dừng.")
            sys.exit(1)
    else:
        log("imatrix đã có, dùng lại.")

    # 2) quantize + 3) perplexity
    for qtype in LADDER:
        out_gguf = os.path.join(WORK, f"qwen3-0.6b-{qtype}.gguf")
        entry = {"file": out_gguf}
        if qtype == "Q4_K_M" and os.path.exists(REF_Q4):
            out_gguf = REF_Q4
            entry = {"file": out_gguf, "note": "bản tải sẵn ref_models"}
        elif not os.path.exists(out_gguf):
            rc, text = run([os.path.join(BIN, "llama-quantize.exe"), "--allow-requantize",
                            "--imatrix", imat, SRC_Q8, out_gguf, qtype, THREADS],
                           f"quantize_{qtype}.log", timeout=1800)
            if rc != 0 or not os.path.exists(out_gguf):
                entry["error"] = f"quantize rc={rc}"
                results["quants"][qtype] = entry
                log(f"  {qtype}: quantize LỖI — bỏ qua")
                continue
        entry["size_mb"] = round(os.path.getsize(out_gguf) / 1024 / 1024, 1)

        for lang, test_file in (("vi", test_vi), ("ja", test_ja)):
            rc, text = run([os.path.join(BIN, "llama-perplexity.exe"), "-m", out_gguf,
                            "-f", test_file, "-t", THREADS, "-c", "512"],
                           f"ppl_{qtype}_{lang}.log", timeout=3600)
            m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)(?:\s*\+/-\s*([0-9.]+))?", text)
            entry[f"ppl_{lang}"] = float(m.group(1)) if m else None
            if m is None:
                entry[f"ppl_{lang}_error"] = text[-500:]
        results["quants"][qtype] = entry
        log(f"  {qtype}: size={entry.get('size_mb')}MB ppl_vi={entry.get('ppl_vi')} ppl_ja={entry.get('ppl_ja')}")
        with io.open(RESULTS, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # 4) hành vi
    for qtype in BEHAV_MODELS:
        entry = results["quants"].get(qtype)
        if not entry or "error" in entry:
            continue
        answers = []
        for i, prompt in enumerate(BEHAV_PROMPTS):
            rc, text = run([os.path.join(BIN, "llama-cli.exe"), "-m", entry["file"],
                            "-p", prompt, "-n", "80", "-t", THREADS, "--single-turn",
                            "--no-display-prompt", "--temp", "0"],
                           f"behav_{qtype}_{i}.log", timeout=600)
            tail = text.strip().splitlines()
            answers.append(" ".join(tail[-8:])[-400:] if tail else "(rỗng)")
        entry["behavior"] = answers

    with io.open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"XONG. Kết quả: {RESULTS}")


if __name__ == "__main__":
    main()
