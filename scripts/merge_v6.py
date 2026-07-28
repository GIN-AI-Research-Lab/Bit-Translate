#!/usr/bin/env python3
"""Gộp corpus vòng 6 = corpus v5 + KD mới (đào theo MẶT TRẬN LỖI + thuật ngữ).

Khác vòng 5: vòng 5 chọn câu theo CHỦ ĐỀ (Quốc hội, CC-100 khẩu ngữ). Vòng 6 chọn
theo NGUYÊN NHÂN LỖI đo được (`eval/error_taxonomy_v5.json`): cấu trúc câu 37%,
nghĩa từ thường 22%, ẩn chủ ngữ 10% — còn thuật ngữ chỉ 3%.

KHÔNG oversample gì cả. Lý do: bộ đào đã lấy đúng số câu cần cho mỗi thuật ngữ
((60 − đang có) × 1,3), nên nhân bản thêm sẽ vượt ngưỡng và lấn chỗ của phần khác.
Vòng 5 oversample ×3 chỉ vì quán ngữ SINH ra ít hơn nhu cầu — vòng này không có
tình huống đó.

  python scripts/merge_v6.py
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from filter_kd_full import check, norm  # noqa: E402

D = Path("D:/Bit-Translate-data")


def dig(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=str(D / "kd_v5_merged.jsonl"))
    p.add_argument("--new", nargs="+", default=[str(D / "raw" / "kd_v6.jsonl")])
    p.add_argument("--out", default=str(D / "kd_v6_merged.jsonl"))
    a = p.parse_args()

    seen = set()
    n_base = n_new = n_drop = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        with open(a.base, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen.add(dig(o.get("ja", "")))
                fo.write(line + "\n")
                n_base += 1
        print(f"corpus v5 : {n_base:,}", flush=True)

        for src in a.new:
            sp = Path(src)
            if not sp.exists():
                print(f"  BỎ QUA {sp.name}")
                continue
            got = 0
            # errors="ignore": file KD bị cắt giữa lúc 120 luồng ghi -> có byte hỏng
            for line in sp.open(encoding="utf-8", errors="ignore"):
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ja, vi = norm(o.get("ja", "")), norm(o.get("vi", ""))
                if not ja or not vi or check(ja, vi):
                    n_drop += 1
                    continue
                h = dig(ja)
                if h in seen:
                    n_drop += 1
                    continue
                seen.add(h)
                fo.write(json.dumps({"id": f"v6_{n_new}", "ja": ja, "vi": vi},
                                    ensure_ascii=False) + "\n")
                n_new += 1
                got += 1
            print(f"  + {sp.name}: {got:,}", flush=True)

    tot = n_base + n_new
    print(f"\n=== GỘP XONG ===")
    print(f"v5 cũ   : {n_base:,} ({100*n_base/tot:.1f}%)")
    print(f"KD vòng6: {n_new:,} ({100*n_new/tot:.1f}%)")
    print(f"loại QC : {n_drop:,}")
    print(f"TỔNG    : {tot:,}  -> {a.out}")


if __name__ == "__main__":
    main()
