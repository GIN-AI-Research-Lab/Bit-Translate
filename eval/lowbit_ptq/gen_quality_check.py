# -*- coding: utf-8 -*-
"""Sinh van ban THAT bang sampling (temperature/top-p, khong greedy) tren vai prompt
chat/suy luan/code, dung runner TQ33 30B-A3B da validate + tokenizer that cua Qwen3
(dung chung vocab giua 0.6B va 30B-A3B, da xac nhan trong RESEARCH_TQ33_RUNNER_30B.md)."""
import json
import os
import struct
import subprocess
import sys

from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

TOK_DIR = "D:/Bit-Translate-data/pack_local/smoke06b"   # tokenizer Qwen3 (chung vocab 0.6B/30B)
RUNNER_DIR = "D:/Bit-Translate-data/tq33_30b/runner"
EXTRAS_BIN = "D:/Bit-Translate-data/tq33_30b/extras.bin"
EXE = "e:/Bit-Translate/eval/lowbit_ptq/qwen3moe_runner_tq33.exe"
WORK = "D:/Bit-Translate-data/tq33_30b/genqa"

PROMPTS = [
    ("chat", "Xin chào! Bạn có thể giới thiệu ngắn gọn về bản thân không?"),
    ("reasoning", "Một quả táo giá 5000 đồng, một quả cam giá 7000 đồng. "
                  "Tôi mua 3 quả táo và 2 quả cam. Hãy tính tổng số tiền tôi phải trả, "
                  "giải thích từng bước."),
    ("coding", "Write a Python function `is_prime(n)` that returns True if n is a prime "
               "number and False otherwise. Include a short docstring."),
]

N_GEN = 100
TEMPERATURE = 0.7
TOP_P = 0.9
SEED = 12345
THREADS = 8


def write_tokens_bin(path, ids):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(ids)))
        f.write(struct.pack(f"<{len(ids)}i", *ids))


def read_tokens_bin(path):
    with open(path, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        ids = struct.unpack(f"<{n}i", f.read(4 * n))
    return list(ids)


def main():
    os.makedirs(WORK, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    print(f"tokenizer vocab_size={tok.vocab_size}")

    results = []
    for name, user_msg in PROMPTS:
        messages = [{"role": "user", "content": user_msg}]
        prompt_ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        print(f"\n=== [{name}] prompt ({len(prompt_ids)} token) ===\n{user_msg}\n")

        pbin = f"{WORK}/{name}_prompt.bin"
        obin = f"{WORK}/{name}_out.bin"
        write_tokens_bin(pbin, prompt_ids)

        cmd = [EXE, "gen", RUNNER_DIR, EXTRAS_BIN, pbin, obin,
               str(N_GEN), str(TEMPERATURE), str(TOP_P), str(SEED), str(THREADS)]
        print("chay:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(r.stdout[-2000:])
        if r.returncode != 0:
            print("LOI runner:", r.stderr[-2000:])
            continue

        all_ids = read_tokens_bin(obin)
        gen_ids = all_ids[len(prompt_ids):]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        full_text = tok.decode(all_ids, skip_special_tokens=False)
        print(f"\n--- [{name}] VAN BAN SINH RA (chi phan model, {len(gen_ids)} token) ---")
        print(gen_text)
        print("--- (het) ---\n")
        results.append({"domain": name, "prompt": user_msg, "n_gen_tokens": len(gen_ids),
                         "generated_text": gen_text, "full_decoded": full_text})

    with open(f"{WORK}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDa luu ket qua -> {WORK}/results.json")


if __name__ == "__main__":
    main()
