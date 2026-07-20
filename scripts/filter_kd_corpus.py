#!/usr/bin/env python3
"""Rule filter LỚP 1 cho corpus KD Đợt 1 (dịch câu thật, không sinh từ seed —
khác vòng 3a nên không cần đọc gloss). Bắt: copy-through/rò ngôn ngữ, tỉ lệ độ
dài bất thường, dedup, artifact cụ thể đã thấy trong judge audit (PLAN_KD_JA2VI):
leftover <think>, placeholder "Anh/chị" kiểu X/Y chưa chọn, ký tự Hán lạ lẫn vào
vế Việt, câu rỗng/quá ngắn do model từ chối.

  python scripts/filter_kd_corpus.py [--all]   # mặc định: chỉ data/synthetic/kd/raw_*.jsonl

Ghi data/synthetic/kd/filtered.jsonl (đạt) + data/synthetic/kd/rejects.jsonl (rớt,
kèm lý do — dùng để tune rule / lấy mẫu review lớp 3).
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
KD = ROOT / "data" / "synthetic" / "kd"

JA_CHARS = re.compile(r"[぀-ヿ一-鿿]")
VI_TONE = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)
# PILOT 2026-07-20 đã calibrate: bỏ check "..."/"anh/chị" — corpus nguồn là blog kỹ
# thuật (Qiita) đầy code/JSON/công thức toán ("$V[X]=E[...]$", "{"data":[...]}"") và
# lời nói ngập ngừng tự nhiên ("い、いや…") -> "..." KHÔNG phải placeholder ở đây.
# "anh/chị" là cách xưng hô kính ngữ chuẩn khi không rõ giới tính, KHÔNG phải lỗi.
# Chỉ giữ marker rõ ràng là artifact model (thẻ HTML/XML lạ, N/A, XXX literal).
PLACEHOLDER = re.compile(r"<think>|</think>|^N/A$|^XXX$", re.I)
REFUSAL = re.compile(
    r"^(xin lỗi,? tôi không thể|i cannot|i'm sorry|as an ai|"
    r"tôi không thể (dịch|giúp))", re.I)
# ký tự Hán KHÔNG phải kanji thường gặp trong câu Nhật (giản thể TQ lẫn vào do model
# đôi lúc trộn ngôn ngữ) — heuristic: một số simplified-only glyph hay gặp trong lỗi
ZH_LEAK = re.compile(r"[前辈们儿嗯呢吧啊哦]")


def ja_ratio(s):
    return len(JA_CHARS.findall(s)) / max(1, len(s))


def check(ja, vi):
    ja, vi = ja.strip(), vi.strip()
    if not ja or not vi:
        return "rong"
    if len(vi) < 2:
        return "vi_qua_ngan"
    if REFUSAL.search(vi):
        return "tu_choi"
    if ja_ratio(ja) < 0.3:
        return "ja_khong_phai_tieng_nhat"
    if len(JA_CHARS.findall(vi)) > 4:
        return "vi_sot_tieng_nhat"
    if ZH_LEAK.search(vi):
        return "vi_ro_han_gian_the"
    if not VI_TONE.search(vi):
        return "vi_khong_dau"
    r = len(vi) / max(1, len(ja))
    if not (0.3 <= r <= 5.0):
        return "ti_le_do_dai"
    if PLACEHOLDER.search(vi):
        return "placeholder"
    return None


def main():
    seen, kept, rej = set(), [], []
    stats = Counter()
    files = sorted(KD.glob("raw_*.jsonl"))
    assert files, f"không có file data/synthetic/kd/raw_*.jsonl — chạy gen_kd_corpus.py trước"
    for f in files:
        m_name = f.stem.replace("raw_", "")
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats[("tong", m_name)] += 1
            why = check(p["ja"], p["vi"])
            key = (p["ja"].strip(), p["vi"].strip())
            if why is None and key in seen:
                why = "trung_lap"
            if why:
                p["reject"] = why
                rej.append(p)
                stats[(why, m_name)] += 1
            else:
                seen.add(key)
                kept.append(p)
                stats[("dat", m_name)] += 1

    (KD / "filtered.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in kept), encoding="utf-8")
    (KD / "rejects.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in rej), encoding="utf-8")

    print(f"ĐẠT {len(kept):,} | RỚT {len(rej):,} (tổng {len(kept)+len(rej):,})")
    models = sorted({m for (_, m) in stats})
    for m in models:
        t = stats.get(("tong", m), 0)
        d = stats.get(("dat", m), 0)
        print(f"  {m:28s} {d:,}/{t:,} đạt ({100*d/max(1,t):.1f}%)")
    top = Counter()
    for (kind, m), n in stats.items():
        if kind not in ("tong", "dat"):
            top[kind] += n
    if top:
        print("  lý do rớt:", ", ".join(f"{k}={v}" for k, v in top.most_common(10)))


if __name__ == "__main__":
    sys.exit(main())
