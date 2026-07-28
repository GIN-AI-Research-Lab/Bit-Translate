#!/usr/bin/env python3
"""Đo MẬT ĐỘ từng KỸ NĂNG trong corpus, để biết "lặp bao nhiêu thì đủ".

KỸ NĂNG khác THUẬT NGỮ ở chỗ lặp:
  - Thuật ngữ = ghi nhớ một mục -> cần CHÍNH mục đó xuất hiện >=60 lần.
  - Kỹ năng   = học một khuôn -> cần NHIỀU VÍ DỤ KHÁC NHAU của khuôn đó.
    Lặp lại một câu 60 lần không dạy được khuôn; 60 câu khác nhau thì có.
  ⇒ với kỹ năng, đơn vị đo là % CORPUS chứa khuôn, không phải số lần/mục.

MỐC HIỆU CHUẨN lấy từ chính dự án: keigo đã ĐẠT NGANG GOOGLE (80% vs 80%) sau khi
được bơm data có mục tiêu. Mật độ keigo lúc đó = mốc "đủ để học xong một kỹ năng".
Ngược lại slang chỉ 0,32% corpus -> 45% vs Google 90%. Hai điểm này neo thang đo.

  python scripts/skill_density.py
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

D = Path("D:/Bit-Translate-data")

# Mỗi kỹ năng = (regex, số lần khớp tối thiểu trong MỘT câu để tính là "có khuôn")
SKILL = {
    # --- MẶT TRẬN 1: cấu trúc / mệnh đề lồng (37% lỗi chính, v5 50% vs GG 95%) ---
    "menh_de_long": (
        r"という|ということ|とされ|と思われ|と考えられ|ような|ように|ための|における|"
        r"に対する|に関する|といった|であって|ものの|にもかかわらず", 2),
    "cau_che_nominal": (r"のは.{0,30}(です|だ|である)|ことが|ことを|ことに|もので|わけで", 2),

    # --- MẶT TRẬN 2: nghĩa từ thông thường (22%) -> đo qua từ đa nghĩa hay sai ---
    "tu_da_nghia": (
        r"微妙|適当|結構|やばい|大丈夫|いい加減|さすが|せっかく|わざと|かえって|"
        r"あいにく|なかなか|いちおう|とりあえず|まさか|さすがに", 1),

    # --- MẶT TRẬN 3: slang / khẩu ngữ (v5 45% vs GG 90%) ---
    "slang": (
        r"ヤバ|マジ|ウケ|エモい|しんどい|ガチ|それな|草[。ｗw]|うざ|きも|ワンチャン|"
        r"推し|バズ|エグい|きつ[いね]|だる[いい]|めんど|むずい|やらかし|ドン引き|"
        r"あざす|パない|ノリ|イケてる|ダサ|ウザ|チャラ|ぼっち|リア充|沼|尊い", 1),
    "khau_ngu_suong_sa": (
        r"だよ|だね|じゃん|でしょ|よね|っす|ちゃう|とく[。、]|やっぱ|めっちゃ|"
        r"すげ|ほんと[にう]|なんか|てか|だって|もん[。ね]|かも[。ね]", 2),

    # --- MẶT TRẬN 4: ẩn chủ ngữ (10% lỗi chính, v5 65% vs GG 85%) ---
    # xấp xỉ: câu có vị ngữ nhưng KHÔNG có đại từ/danh từ chỉ người làm chủ ngữ
    "an_chu_ngu": (r"^(?![^。]{0,40}(私|僕|俺|彼|彼女|あなた|君|我々|皆|人々|"
                   r"[一-龥]{2,4}(さん|氏|君|様|大臣|議員)))"
                   r"[^。]{12,}(ます|ました|です|でした|ている|ていた)[。、]", 1),

    # --- MẶT TRẬN 5: phủ định / thái độ (v5 65% vs GG 85%) ---
    "phu_dinh_kho": (
        r"ないわけ|なくはない|ざるを得な|ないこともない|ずして|まい[。、]|かねない|"
        r"にほかならな|とは限らな|わけではな|どころか|ばかりか|しかない|ずには", 1),

    # --- MẶT TRẬN 6: số / niên hiệu (v5 70% vs GG 95%) ---
    "so_nien_hieu": (
        r"(令和|平成|昭和|大正|明治)\s*[〇一二三四五六七八九十百元\d]+\s*年|"
        r"[〇一二三四五六七八九十]{1,4}(万|億|兆)|\d+(?:\.\d+)?\s*(%|％|倍|割)", 1),

    # --- MẶT TRẬN 7: katakana ngoại lai (v5 75% vs GG 95%) ---
    "katakana_dai": (r"[ァ-ヶー]{6,}", 1),

    # --- MỐC HIỆU CHUẨN: keigo — kỹ năng ĐÃ HỌC XONG (ngang Google) ---
    "keigo_MOC": (r"いらっしゃ|おっしゃ|ご覧|なさい(ます|ませ)|くださいます|存じ|伺い|"
                  r"申し上げ|拝見|恐れ入り|いただけます|申し(ます|上げ)|いたし(ます|まし)|"
                  r"参り(ます|まし)|おり(ます|まし)|承知", 1),
    # --- MỐC ĐỐI CHỨNG: idiom — cả v5 lẫn Google đều kém (60/65) ---
    "idiom_MOC": (r"気を付|手を貸|骨が折れ|目を通|腹が立|水に流|念のため|一石二鳥|"
                  r"猫の手|棚から|馬の耳|石の上|骨身|顔が広|耳が痛|口が軽", 1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(D / "kd_v5_merged.jsonl"))
    ap.add_argument("--total", type=int, default=13144706)
    ap.add_argument("--out", default="eval/skill_density.json")
    a = ap.parse_args()

    pats = {k: (re.compile(v[0]), v[1]) for k, v in SKILL.items()}
    cnt = Counter()
    n = 0
    with open(a.src, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                s = json.loads(line)["ja"]
            except Exception:
                continue
            n += 1
            for k, (p, need) in pats.items():
                if need == 1:
                    if p.search(s):
                        cnt[k] += 1
                else:
                    if len(p.findall(s)) >= need:
                        cnt[k] += 1

    # điểm bench đã đo, để đối chiếu mật độ <-> chất lượng
    SCORE = {"keigo_MOC": (80, 80), "slang": (45, 90), "menh_de_long": (50, 95),
             "an_chu_ngu": (65, 85), "phu_dinh_kho": (65, 85), "so_nien_hieu": (70, 95),
             "katakana_dai": (75, 95), "idiom_MOC": (60, 65)}

    print(f"corpus {n:,} câu\n")
    print(f"{'kỹ năng':<22}{'số câu':>12}{'% corpus':>10}{'  v5':>6}{'  GG':>5}{'  chênh':>8}")
    print("-" * 66)
    res = {}
    for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
        pct = 100 * v / n
        sc = SCORE.get(k)
        s1 = f"{sc[0]}%" if sc else "-"
        s2 = f"{sc[1]}%" if sc else "-"
        gap = f"{sc[0]-sc[1]:+d}" if sc else "-"
        res[k] = {"n": v, "pct": round(pct, 2), "v5": sc[0] if sc else None,
                  "gg": sc[1] if sc else None}
        print(f"{k:<22}{v:>12,}{pct:>9.2f}%{s1:>6}{s2:>5}{gap:>8}")

    Path(a.out).write_text(json.dumps({"n": n, "skills": res}, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
