# -*- coding: utf-8 -*-
"""Giai doan 1 (RESEARCH_TQ33_OUTLIER_FIX.md) — chuan bi 1 chuoi token THAT (khong phai ngau
nhien) de chay qua qwen3moe_runner_tq33.exe che do `stats`, phuc vu do outlier-channel THAT
tren Qwen3-30B-A3B (chua biet chi so kenh cu the cho 30B — hidden=2048, khac 0.6B hidden=1024
noi bao cao cu do duoc dim~48/52 — PHAI do lai, khong doan).

Dung 20 dong dau cua val100_vi.txt (cau tieng Viet that, da co san trong repo, dung cho iq_compare
truoc do) -> ghep boi \n -> tokenize bang dung tokenizer Qwen3 (chung vocab 0.6B/30B-A3B, da xac
nhan trong RESEARCH_TQ33_RUNNER_30B.md muc 3.4) -> ra ~470 token (vua du "vai tram token" theo
yeu cau nhiem vu, con du margin duoi MAX_POS=768 cua runner)."""
import struct

from transformers import AutoTokenizer

TOK_DIR = "D:/Bit-Translate-data/pack_local/smoke06b"
VAL_FILE = "D:/Bit-Translate-data/iq_compare/val100_vi.txt"
N_LINES = 20
OUT_PATH = "D:/Bit-Translate-data/tq33_30b/genqa/stats_prompt.bin"


def main():
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    with open(VAL_FILE, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    text = "\n".join(lines[:N_LINES])
    ids = tok.encode(text)
    print(f"dung {N_LINES} dong dau cua {VAL_FILE}")
    print(f"tong {len(ids)} token (MAX_POS runner = 768, con du margin)")
    print(f"5 token dau: {ids[:5]}  5 token cuoi: {ids[-5:]}")

    with open(OUT_PATH, "wb") as f:
        f.write(struct.pack("<i", len(ids)))
        f.write(struct.pack(f"<{len(ids)}i", *ids))
    print(f"da ghi -> {OUT_PATH}")


if __name__ == "__main__":
    main()
