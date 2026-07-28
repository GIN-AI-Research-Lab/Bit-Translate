#!/usr/bin/env python3
"""Danh sách TỪ MƯỢN KATAKANA đang thiếu — thay cho bộ lọc "katakana dài" (sai).

Vì sao bộ lọc cũ sai: `[ァ-ヶ]{7,}` chỉ bắt katakana DÀI, mà dài ≠ hiếm. Nó lấy về
ホルモンバランス, ファイルダウンロード — toàn từ phổ thông corpus đã thừa. Trong khi
lỗi thật đo được là マジックテープ→"băng dính ma thuật" (phải là Velcro): thiếu ÁNH XẠ
của một từ mượn CỤ THỂ, không phải thiếu văn bản có katakana.

⇒ Katakana phải vá y như thuật ngữ: liệt kê từng từ, đếm, tìm từ dưới ngưỡng.
Không cần LLM — trích thẳng từ vựng katakana có thật trong text rồi đếm.

  python scripts/build_katakana_gap.py
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

D = Path("D:/Bit-Translate-data")
KATA = re.compile(r"[ァ-ヴー]{3,}")
# đuôi/thành phần ghép quá phổ thông -> không phải "từ" cần học riêng
STOP = set("""センター サービス システム データ ページ サイト メール ネット グループ
コース タイプ ポイント アップ チェック スタート レベル テーマ メンバー パターン
ケース スペース サイズ エリア デザイン イメージ スタイル シリーズ セット バランス
メニュー コメント ブログ コンテンツ ユーザー クリック ダウンロード アドレス""".split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(D / "kd_v5_merged.jsonl"))
    ap.add_argument("--ref", default=str(D / "raw" / "cc100_pick.txt"))
    ap.add_argument("--ref-lines", type=int, default=5_000_000)
    ap.add_argument("--min-occ", type=int, default=60)
    ap.add_argument("--min-ref", type=int, default=8,
                    help="phải gặp >=N lần trong text thật mới đáng học")
    ap.add_argument("--out", default=str(D / "katakana_gap.json"))
    a = ap.parse_args()

    print("đếm katakana trong corpus v5...", flush=True)
    cur = Counter()
    with open(a.corpus, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                s = json.loads(line)["ja"]
            except Exception:
                continue
            for w in KATA.findall(s):
                cur[w] += 1

    print("đếm katakana trong text thật đối chứng...", flush=True)
    ref = Counter()
    n = 0
    with open(a.ref, encoding="utf-8", errors="ignore") as f:
        for line in f:
            n += 1
            for w in KATA.findall(line):
                ref[w] += 1
            if n >= a.ref_lines:
                break

    # từ ĐÁNG học = xuất hiện đủ nhiều ngoài đời NHƯNG dưới ngưỡng trong corpus
    gap = {w: {"cur": cur[w], "ref": c} for w, c in ref.items()
           if c >= a.min_ref and cur[w] < a.min_occ and w not in STOP and len(w) >= 3}
    tot_ref = sum(1 for w, c in ref.items() if c >= a.min_ref and w not in STOP)

    print(f"\ntừ vựng katakana: {len(cur):,} trong corpus | {len(ref):,} trong text thật")
    print(f"đủ phổ biến ngoài đời (>={a.min_ref} lần/{n:,} dòng): {tot_ref:,}")
    print(f"trong đó CHƯA ĐỦ trong corpus (<{a.min_occ} lần): {len(gap):,} "
          f"({100*len(gap)/max(tot_ref,1):.0f}%)")

    top = sorted(gap.items(), key=lambda x: -x[1]["ref"])
    print(f"\n--- 40 từ hay gặp ngoài đời nhất mà corpus thiếu ---")
    for i in range(0, 40, 5):
        print("  " + "  ".join(f"{w}({d['cur']}/{d['ref']})" for w, d in top[i:i + 5]))

    Path(a.out).write_text(json.dumps(
        {"min_occ": a.min_occ, "n_ref_lines": n, "gap": gap},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {a.out}  (số trong ngoặc = corpus/text thật)")


if __name__ == "__main__":
    main()
