#!/usr/bin/env python3
"""Gộp corpus vòng 5 = corpus v4 + KD mới (Quốc hội/CC-100) + quán ngữ sinh (×N).

Khác merge_niche_corpus.py: vòng 5 phần thêm chủ yếu là DATA THẬT đã KD (không
oversample), chỉ quán ngữ sinh mới nhân bản. Vòng 4 ngược lại — phần thêm gần như
toàn tổng hợp, và bench câu thật cho thấy cải thiện không khái quát.

Oversample quán ngữ: đo được 3.168/4.651 quán ngữ xuất hiện <50 lần trong corpus
v4 + phần chọn v5 (2.098 mục = 0 lần). Nhân ×3 để mỗi mục đạt ~60 lần — ngưỡng lấy
từ vòng 4 (thuật ngữ 78 lần/corpus thì được vá, <=5 lần thì dịch bậy).

  python scripts/merge_v5.py
  # -> D:/Bit-Translate-data/kd_v5_merged.jsonl
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
    p.add_argument("--base", default=str(ROOT / "data" / "synthetic" / "kd_clean" / "kd_filtered.jsonl"))
    p.add_argument("--out", default=str(D / "kd_v5_merged.jsonl"))
    p.add_argument("--idiom-oversample", type=int, default=3)
    a = p.parse_args()

    seen = set()
    n_base = n_new = n_idi = n_drop = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    with open(a.out, "w", encoding="utf-8") as fo:
        # 1) corpus v4 nguyên vẹn — model đã học, giữ để chống quên
        with open(a.base, encoding="utf-8") as f:
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
        print(f"corpus v4      : {n_base:,}", flush=True)

        # 2) KD mới (data THẬT: Quốc hội + CC-100) — KHÔNG oversample.
        # LỌC theo v5_planC.txt: kd_v5c.jsonl có lẫn ~81k cặp "giai đoạn 2" do
        # kd_forever.sh tự chuyển nguồn nhưng ghi cùng file. Phần đó là Quốc hội
        # ngoài danh sách chọn — nhận vào sẽ đẩy tỷ lệ Quốc hội vượt mức 19% đã
        # tính. Giữ lại làm vốn cho vòng 6, KHÔNG đưa vào corpus v5.
        planc = set()
        pc = D / "raw" / "v5_planC.txt"
        if pc.exists():
            for line in pc.open(encoding="utf-8"):
                s = line.strip()
                if s:
                    planc.add(s)
        # kd_v5 / kd_v5b dịch từ v5_todo(2) — cũng nằm trong tập đã chọn v5_select
        sel = set()
        sp = D / "raw" / "v5_select.txt"
        if sp.exists():
            for line in sp.open(encoding="utf-8"):
                s = line.strip()
                if s:
                    sel.add(s)
        keep = planc | sel if planc or sel else None
        n_outside = 0

        for src in [D / "raw" / "kd_v5.jsonl", D / "raw" / "kd_v5b.jsonl",
                    D / "raw" / "kd_v5c.jsonl", D / "raw" / "kd_cc100.jsonl"]:
            if not src.exists():
                print(f"  bỏ qua {src.name} (không có)")
                continue
            got = 0
            # errors="ignore": file KD bị kill giữa lúc 60 luồng ghi -> có byte
            # hỏng. Đã vá riêng, nhưng để đây cho chắc — một byte lạ không được
            # phép làm sập cả chuỗi gộp 13 triệu dòng.
            for line in src.open(encoding="utf-8", errors="ignore"):
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ja, vi = norm(o.get("ja", "")), norm(o.get("vi", ""))
                if keep is not None and o.get("ja", "").strip() not in keep:
                    n_outside += 1
                    continue
                if check(ja, vi):
                    n_drop += 1
                    continue
                h = dig(ja)
                if h in seen:
                    n_drop += 1
                    continue
                seen.add(h)
                fo.write(json.dumps({"id": f"v5_{n_new}", "ja": ja, "vi": vi},
                                    ensure_ascii=False) + "\n")
                n_new += 1
                got += 1
            print(f"  + {src.name}: {got:,}", flush=True)

        # 3) quán ngữ sinh × oversample — phần DUY NHẤT nhân bản
        idi = D / "gen_niche" / "idiomgap.jsonl"
        if idi.exists():
            rows = []
            for line in idi.open(encoding="utf-8"):
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ja, vi = norm(o.get("ja", "")), norm(o.get("vi", ""))
                if check(ja, vi):
                    continue
                h = dig(ja)
                if h in seen:
                    continue
                seen.add(h)
                rows.append((ja, vi))
            for k in range(a.idiom_oversample):
                for ja, vi in rows:
                    fo.write(json.dumps({"id": f"idi_{n_idi}", "ja": ja, "vi": vi},
                                        ensure_ascii=False) + "\n")
                    n_idi += 1
            print(f"  + quán ngữ: {len(rows):,} câu ×{a.idiom_oversample} = {n_idi:,}",
                  flush=True)

    tot = n_base + n_new + n_idi
    print(f"\n=== GỘP XONG ===")
    print(f"v4 cũ      : {n_base:,} ({100*n_base/tot:.1f}%)")
    print(f"KD mới THẬT: {n_new:,} ({100*n_new/tot:.1f}%)")
    print(f"Quán ngữ   : {n_idi:,} ({100*n_idi/tot:.1f}%)")
    print(f"Loại QC/trùng: {n_drop:,}")
    print(f"TỔNG       : {tot:,}")
    print(f"Output     : {a.out}")


if __name__ == "__main__":
    main()
