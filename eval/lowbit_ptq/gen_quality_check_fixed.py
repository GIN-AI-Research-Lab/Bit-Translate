# -*- coding: utf-8 -*-
"""RESEARCH_TQ33_OUTLIER_FIX.md Giai doan 3 — BAN SAO cua gen_quality_check.py (file goc KHONG
sua, van con nguyen de doi chieu) — CHI khac: goi che do CLI "genfix" (BAT outlier-channel fix)
thay vi "gen", va ghi ra results_fixed.json (ten khac results.json goc) de 2 ban ket qua TRUOC/
SAU cung ton tai canh nhau, so sanh truc tiep duoc. CUNG 3 prompt, CUNG seed=12345, CUNG
temp=0.7/top_p=0.9, CUNG N_GEN=100 — moi dieu kien GIONG HET ban goc, chi khac runner co fix
hay khong, de so sanh cong bang."""
import json
import os
import struct
import subprocess
import sys

from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

TOK_DIR = "D:/Bit-Translate-data/pack_local/smoke06b"
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
        print(f"\n=== [{name}] prompt ({len(prompt_ids)} token) — outlier-fix BAT ===\n{user_msg}\n")

        pbin = f"{WORK}/{name}_prompt_fix.bin"
        obin = f"{WORK}/{name}_out_fix.bin"
        write_tokens_bin(pbin, prompt_ids)

        cmd = [EXE, "genfix", RUNNER_DIR, EXTRAS_BIN, pbin, obin,
               str(N_GEN), str(TEMPERATURE), str(TOP_P), str(SEED), str(THREADS)]
        print("chay:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(r.stdout[-2500:])
        if r.returncode != 0:
            print("LOI runner:", r.stderr[-2000:])
            continue

        all_ids = read_tokens_bin(obin)
        gen_ids = all_ids[len(prompt_ids):]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        full_text = tok.decode(all_ids, skip_special_tokens=False)
        print(f"\n--- [{name}] VAN BAN SINH RA (outlier-fix BAT, chi phan model, {len(gen_ids)} token) ---")
        print(gen_text)
        print("--- (het) ---\n")
        results.append({"domain": name, "prompt": user_msg, "n_gen_tokens": len(gen_ids),
                         "generated_text": gen_text, "full_decoded": full_text})

    with open(f"{WORK}/results_fixed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDa luu ket qua -> {WORK}/results_fixed.json")


if __name__ == "__main__":
    main()
