#!/usr/bin/env python3
"""Rule filter LỚP 1 + chuẩn hóa NFKC cho corpus KD full 11.88M (streaming).

Tái dùng logic check() đã hiệu chỉnh của filter_kd_corpus.py (PILOT 2026-07-20),
nhưng:
  - STREAMING: đọc/ghi từng dòng, bộ nhớ không đổi (11.88M dòng, file 2.5GB).
  - Chuẩn hóa NFKC (khớp nmt_nfkc của tokenizer — text_norm.py) cho cả ja & vi.
  - SỬA báo động giả `・`: ký tự bullet U+30FB nằm trong dải katakana nên bị đếm
    nhầm là "chữ Nhật lọt vào VI" — loại nó khỏi phép đếm (đã thấy thật trong mẫu).
  - dedup bằng digest 8-byte (tiết kiệm RAM so với giữ nguyên chuỗi).

  python scripts/filter_kd_full.py \
      data/synthetic/kd_clean/kd_gemini3_final.jsonl \
      data/synthetic/kd_clean/kd_filtered.jsonl \
      data/synthetic/kd_clean/kd_rejects.jsonl
"""
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter

JA_CHARS = re.compile(r"[぀-ヿ一-鿿]")
BULLET = re.compile(r"[・·]")          # loại khỏi phép đếm chữ-Nhật-lọt-VI (bullet hợp lệ)
VI_TONE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)
PLACEHOLDER = re.compile(r"<think>|</think>|^N/A$|^XXX$", re.I)
REFUSAL = re.compile(
    r"^(xin lỗi,? tôi không thể|i cannot|i'm sorry|as an ai|"
    r"tôi không thể (dịch|giúp))", re.I)
ZH_LEAK = re.compile(r"[前辈们儿嗯呢吧啊哦]")


def norm(text):
    """NFKC + dồn khoảng trắng (khớp text_norm.normalize_for_model)."""
    t = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", t).strip()


def ja_ratio(s):
    return len(JA_CHARS.findall(s)) / max(1, len(s))


def check(ja, vi):
    if not ja or not vi:
        return "rong"
    if len(vi) < 2:
        return "vi_qua_ngan"
    if REFUSAL.search(vi):
        return "tu_choi"
    if ja_ratio(ja) < 0.3:
        return "ja_khong_phai_tieng_nhat"
    # đếm chữ Nhật lọt vào VI, BỎ QUA bullet ・/· (báo động giả)
    if len(JA_CHARS.findall(BULLET.sub("", vi))) > 4:
        return "vi_sot_tieng_nhat"
    if ZH_LEAK.search(vi):
        return "vi_ro_han_gian_the"
    if not VI_TONE.search(vi):
        return "vi_khong_dau"
    r = len(vi) / max(1, len(ja))
    # Chặn dưới luôn áp (VI quá ngắn vs JA = bị cụt/mất nội dung).
    # Chặn trên CHỈ áp khi JA không quá ngắn: kanji đậm đặc (vd 水冷高温型) nở ra
    # tiếng Việt dài >5× là BÌNH THƯỜNG, không phải lỗi — đừng loại nhầm.
    if r < 0.3:
        return "ti_le_do_dai"
    if len(ja) > 8 and r > 5.0:
        return "ti_le_do_dai"
    if PLACEHOLDER.search(vi):
        return "placeholder"
    return None


def main():
    if len(sys.argv) < 4:
        print("dùng: filter_kd_full.py <in.jsonl> <out_keep.jsonl> <out_reject.jsonl>")
        sys.exit(1)
    inp, out_keep, out_rej = sys.argv[1], sys.argv[2], sys.argv[3]

    seen = set()
    stats = Counter()
    n = 0
    with open(inp, "r", encoding="utf-8") as fin, \
         open(out_keep, "w", encoding="utf-8") as fk, \
         open(out_rej, "w", encoding="utf-8") as fr:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                stats["json_hong"] += 1
                continue
            n += 1
            ja = norm(p.get("ja", ""))
            vi = norm(p.get("vi", ""))
            why = check(ja, vi)
            if why is None:
                h = hashlib.blake2b((ja + "\t" + vi).encode("utf-8"), digest_size=8).digest()
                if h in seen:
                    why = "trung_lap"
                else:
                    seen.add(h)
            if why:
                stats[why] += 1
                p["reject"] = why
                fr.write(json.dumps(p, ensure_ascii=False) + "\n")
            else:
                stats["dat"] += 1
                out = {"id": p.get("id"), "ja": ja, "vi": vi}
                fk.write(json.dumps(out, ensure_ascii=False) + "\n")
            if n % 1_000_000 == 0:
                print(f"  ...{n:,} dòng | đạt {stats['dat']:,}", flush=True)

    kept = stats["dat"]
    rej = n - kept
    print(f"\n=== RULE FILTER XONG ===")
    print(f"Tổng đọc : {n:,}")
    print(f"ĐẠT      : {kept:,} ({100*kept/max(1,n):.2f}%)")
    print(f"RỚT      : {rej:,} ({100*rej/max(1,n):.2f}%)")
    print("Lý do rớt:")
    for k, v in stats.most_common():
        if k != "dat":
            print(f"  {k:26s} {v:>10,}")


if __name__ == "__main__":
    main()
