#!/usr/bin/env python3
"""Bản đồ miền (domain map) cho corpus ja→vi — ĐO chứ không đoán.

Ba trục ĐỘC LẬP nhau, một câu có thể trúng nhiều nhãn ở cả ba:

  A. CHỦ ĐỀ (topic)        — câu nói VỀ cái gì
  B. THỂ VĂN (register)    — viết/nói theo lối nào (kính ngữ, văn nói, điều luật...)
  C. HIỆN TƯỢNG (phenom)   — cấu trúc gây gãy dịch máy (số đếm, niên hiệu, ẩn chủ ngữ...)

Trộn ba trục vào một danh sách "domain" phẳng là sai lầm cũ: một câu biên bản Quốc hội
vừa là chính trị (A), vừa là kính ngữ (B), vừa đầy niên hiệu + trợ số từ (C). Bơm thêm
biên bản Quốc hội thì cả ba cùng lên — nên không biết cái nào mới là thứ đang vá.

  python scripts/domain_map.py --sample 500000
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

D = Path("D:/Bit-Translate-data")

# --------------------------------------------------------------------------- A
# CHỦ ĐỀ. Từ khoá chọn theo nguyên tắc: phải là từ HIẾM ngoài miền của nó, nếu
# không sẽ bắt nhầm (vd "問題" xuất hiện khắp nơi -> vô dụng làm chỉ báo).
TOPIC = {
    "chinh_tri_hanh_chinh": r"内閣|国会|衆議院|参議院|与党|野党|法案|閣議|大臣|省庁|自治体|選挙|議員|行政|条例|政策|答弁|委員会|首相|官房|社会保障|年金制度|生活保護|福祉制度",
    "phap_luat_tu_phap": r"裁判|判決|訴訟|被告|原告|弁護士|検察|刑法|民法|憲法|条文|契約書|違法|起訴|控訴|判例|法廷|有罪|無罪|損害賠償",
    "kinh_te_tai_chinh": r"経済|金融|株価|投資|決算|売上高|利益率|GDP|景気|為替|日銀|証券|債券|融資|財政|税制|市場|企業買収|上場|配当|仮想通貨|暗号資産|ビットコイン|マイニング|トレーディング|投資家",
    "y_te_suc_khoe": r"医療|病院|患者|診断|治療|症状|手術|看護|薬剤|感染|ワクチン|疾患|臨床|処方|検査結果|医師|介護|リハビリ|副作用|アレルギー|ホルモン|動物病院|獣医|ペットの健康|医薬品",
    "cntt_phan_mem": r"ソフトウェア|プログラム|データベース|サーバー|アルゴリズム|コード|API|クラウド|セキュリティ|ネットワーク|アプリ|開発環境|バグ|実装|エンジニア|AI|機械学習|ブロックチェーン|マザーボード|ハードウェア|ディスプレイ|BIOS|Exif|ドライバー更新",
    "cong_nghe_che_tao": r"製造|工場|機械|設備|部品|生産ライン|品質管理|加工|溶接|金型|エンジン|材料|強度|設計図|自動化|ロボット|工程",
    "khoa_hoc_tu_nhien": r"実験|物理|化学|生物|遺伝子|分子|細胞|宇宙|観測|理論|仮説|素粒子|進化|生態系|元素|反応式|論文",
    "giao_duc": r"学校|生徒|教師|授業|大学|入試|教育|学習|カリキュラム|試験|学生|講義|部活|教科書|進学|塾|卒業",
    "the_thao": r"試合|選手|優勝|チーム|監督|得点|リーグ|オリンピック|野球|サッカー|練習|大会|記録更新|決勝|敗退",
    "giai_tri_van_hoa": r"映画|音楽|ドラマ|俳優|歌手|アニメ|漫画|ゲーム|アイドル|ライブ|舞台|作品|監督|声優|バンド|小説家",
    "am_thuc": r"料理|レシピ|食材|味付け|調理|美味しい|焼く|煮る|炒め|食感|グルメ|デザート|惣菜|栄養素|食生活|食べ放題|栄養価",
    "du_lich": r"旅行|観光|ホテル|旅館|空港|新幹線|温泉|名所|宿泊|ツアー|土産|観光地|絶景|旅程|チェックイン",
    "thoi_trang_lam_dep": r"ファッション|化粧|コーデ|美容|髪型|スキンケア|ブランド|服装|メイク|ネイル|着こなし|コスメ",
    "bat_dong_san_xay_dung": r"不動産|物件|賃貸|マンション|住宅|建築|施工|間取り|土地|リフォーム|耐震|建設|工事|坪|内装",
    "nong_lam_ngu": r"農業|漁業|林業|農家|収穫|作物|漁港|水産|畜産|農地|栽培|養殖|米作|農林水産|品種",
    "moi_truong_nang_luong": r"環境|温暖化|再生可能|排出|原発|電力|太陽光|リサイクル|廃棄物|脱炭素|エネルギー|CO2|生物多様性",
    "quan_su_an_ninh": r"防衛|自衛隊|安全保障|軍事|ミサイル|基地|同盟|安保|テロ|兵器|演習|国防",
    "ton_giao_triet_hoc": r"宗教|仏教|神道|神社仏閣|信仰心|哲学者|哲学的|祈りを捧げ|教義|禅宗|聖書|キリスト教|イスラム教|創価学会|お経|念仏|出家",
    "lich_su": r"歴史|時代|戦国|江戸|明治|幕府|遺跡|古代|中世|王朝|史料|考古|年表|併合条約|大日本帝国|植民地支配|世界大戦",
    "tam_ly_quan_he": r"気持ち|感情|ストレス|不安|人間関係|恋愛|友達|悩み|心理|性格|共感|孤独|自信|出会い系|マッチング|カウンセリング|メンタルヘルス|心の病",
    "gia_dinh_nuoi_day": r"子育て|育児|家族|親子|赤ちゃん|保育園|夫婦|妊娠|出産|家事|兄弟|しつけ|ペット|飼い主|愛犬|愛猫",
    "giao_thong_oto": r"自動車|運転|道路|交通|渋滞|鉄道|バス|免許|車両|高速道路|事故|駐車|燃費|EV",
    "lao_dong_nhan_su": r"就職|転職|採用|人事|給与|残業|労働|職場|上司|部下|退職|昇進|求人|面接|働き方",
    "sinh_hoat_thuong_ngay": r"買い物に行く|掃除機|洗濯物|ご近所|自炊|一人暮らし|引っ越し|ゴミ出し|節約術|片付け|日用品",
}

# --------------------------------------------------------------------------- B
# THỂ VĂN. Đây là trục bị bỏ quên nhất — corpus có thể phủ đủ chủ đề mà vẫn dịch
# cứng vì toàn văn viết trang trọng, không có văn nói.
REGISTER = {
    "keigo_ton_kinh": r"いらっしゃ|おっしゃ|ご覧|なさい(ます|ませ)|お[ぁ-ん]+になり|くださいます|存じ|伺い|申し上げ|拝見|恐れ入り|いただけます",
    "kenjougo_khiem_nhuong": r"申し(ます|上げ)|いたし(ます|まし)|参り(ます|まし)|おり(ます|まし)|承知|拝",
    "desu_masu_lich_su": r"(です|ます|ました|ません|でしょう|ましょう)[。、！？]",
    "da_dearu_van_viet": r"(である|であった|だった|ではない|という|られる)[。]",
    "van_noi_suong_sa": r"(だよ|だね|かな|じゃん|でしょ|よね|っす|さ[。、]|わけ|んだ[。よね]|ちゃう|とく[。、]|やっぱ|めっちゃ|すごく)",
    "slang_tre": r"ヤバ|マジ|ウケる|エモい|しんどい|ガチ|それな|草[。ｗw]|うざ|きも|ムリゲー|ワンチャン|推し",
    "phuong_ngu": r"(やねん|せやな|ちゃう[ん。]|あかん|やで|とっとと|だべ|だっぺ|ばい[。、]|けん[。、]|なんしよ)",
    "menh_lenh_huong_dan": r"(してください|しましょう|すること[。、]|しなければならない|禁止|注意して|手順|まず.{0,12}次に|以下の通り)",
    "dieu_luat_hop_dong": r"第[〇一二三四五六七八九十百\d]+条|前項|当該|ものとする|に基づき|準用|但し書|甲は|乙は|本契約",
    "tin_tuc_bao_chi": r"(と述べた|と語った|明らかにした|によると|同社|同氏|関係者によ|発表した|判明した)",
    "quang_cao_catchcopy": r"(新登場|今なら|限定|無料|お得|キャンペーン|人気No|話題の|注目の|おすすめ)",
    "hoi_thoai_doi_dap": r"^[「『]|[」』]$|って言っ|と聞い|なんて言|え[?？]|うん[、。]|はい[、。]",
    "van_hoc_tu_su": r"(だった。|のだった|ような気がした|のである。|しれない。|見つめ|呟い|微笑)",
    "sns_ngan": r"[#＃][^\s]{2,}|www$|笑$|w{2,}$|！！|？？",
}

# --------------------------------------------------------------------------- C
# HIỆN TƯỢNG NGÔN NGỮ gây gãy dịch. Trục này quyết định "dùng được hay không"
# nhiều hơn cả chủ đề — đo được ở vòng 4-5: sai niên hiệu / sai số / sai tên riêng
# là thứ làm câu thành vô dụng dù phần còn lại dịch đúng.
PHENOM = {
    "nien_hieu": r"(令和|平成|昭和|大正|明治)\s*[〇一二三四五六七八九十百元\d]+\s*年",
    "so_lon_kanji": r"[〇一二三四五六七八九十]{1,4}(万|億|兆|千)",
    "tro_so_tu": r"[0-9〇一二三四五六七八九十百千]+(人|匹|台|冊|枚|本|個|軒|件|回|度|割|倍|名|品|箱|杯|足|頭|羽|隻|棟|条|号|項)",
    "don_vi_do_luong": r"\d+(?:\.\d+)?\s*(km|cm|mm|kg|mg|ml|ℓ|リットル|平方|坪|畳|度|%|％|円|ドル|ユーロ|億円|万円)",
    "ngay_gio": r"\d{1,2}月\d{1,2}日|\d{1,2}時\d{1,2}分|来年度|今年度|昨年度|上半期|四半期",
    "ten_nguoi_chuc_danh": r"[一-龥]{2,4}(大臣|議員|社長|会長|教授|知事|市長|町長|部長|課長|先生|さん|氏|君|様)(?![ぁ-ん])",
    "ten_dia_danh": r"(東京|大阪|京都|北海道|沖縄|名古屋|福岡|横浜|神戸|仙台|広島|札幌)|[一-龥]{2,3}(県|市|町|村|区|島|川|山|湾|港)(?![ぁ-ん])",
    "ten_to_chuc": r"[一-龥ァ-ヶ]{2,8}(株式会社|協会|委員会|機構|財団|組合|連盟|省|庁|大学|銀行|グループ)",
    "quan_ngu_thanh_ngu": r"(気を付|手を貸|骨が折れ|目を通|腹が立|水に流|念のため|一石二鳥|猫の手|棚から|馬の耳|石の上)",
    "onomatope": r"(ドキドキ|ワクワク|キラキラ|ゆっくり|しっかり|はっきり|ぼんやり|ぐっすり|バタバタ|ペコペコ|ガタガタ|ふわふわ|さっぱり|すっきり)",
    "katakana_ngoai_lai": r"[ァ-ヶー]{5,}",
    "viet_tat_chu_cai": r"\b[A-Z]{2,6}\b",
    "phu_dinh_kep": r"ない(わけ|こと|はず)は(ない|ありません)|なくはない|ざるを得な",
    "the_bi_dong_sai_khien": r"(させられ|られてい|せざる|させていただ)",
    "cho_nhan_yarimorai": r"(てもらい|ていただき|てくださ|てあげ|てくれ)",
    "menh_de_quan_he_dai": r"[ぁ-んァ-ヶ一-龥]{25,}(という|ような|ための|における|に関する)[一-龥]{2,}",
    "trich_dan_long": r"「[^」]{10,}」と|『[^』]{10,}』",
    "cau_hoi": r"[?？]|(ですか|ますか|のか)[。、]?$",
    "liet_ke_dai": r"(、[^、]{2,12}){4,}",
}

LEN_TIER = [(0, 20, "1_rat_ngan"), (20, 40, "2_ngan"), (40, 80, "3_vua"),
            (80, 140, "4_dai"), (140, 10 ** 9, "5_rat_dai")]

AXES = [("A_chu_de", TOPIC), ("B_the_van", REGISTER), ("C_hien_tuong", PHENOM)]
COMPILED = [(nm, {k: re.compile(v) for k, v in d.items()}) for nm, d in AXES]


def classify(ja):
    out = {}
    for axis, pats in COMPILED:
        out[axis] = [k for k, p in pats.items() if p.search(ja)]
    return out


def tier(n):
    for lo, hi, nm in LEN_TIER:
        if lo <= n < hi:
            return nm
    return "5_rat_dai"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(D / "kd_v5_merged.jsonl"))
    ap.add_argument("--sample", type=int, default=500000)
    ap.add_argument("--total", type=int, default=13144706)
    ap.add_argument("--out", default="eval/domain_map_v5.json")
    a = ap.parse_args()

    keep = a.sample / max(a.total, 1)
    rnd = random.Random(7)
    cnt = {axis: Counter() for axis, _ in AXES}
    tiers = Counter()
    none_hit = Counter()
    n = 0

    with open(a.src, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if rnd.random() > keep:
                continue
            try:
                ja = json.loads(line)["ja"]
            except Exception:
                continue
            n += 1
            tiers[tier(len(ja))] += 1
            c = classify(ja)
            for axis, labs in c.items():
                if not labs:
                    none_hit[axis] += 1
                for l in labs:
                    cnt[axis][l] += 1
            if n % 100000 == 0:
                print(f"  ...{n:,}", file=sys.stderr, flush=True)

    scale = a.total / n if n else 0
    print(f"\n=== BẢN ĐỒ MIỀN corpus v5 — mẫu {n:,} / {a.total:,} câu ===\n")
    print("ĐỘ DÀI (ký tự ja)")
    for _, _, nm in LEN_TIER:
        v = tiers[nm]
        print(f"  {nm:<12} {100*v/n:5.1f}%  ~{int(v*scale):>10,}")

    res = {"n_sample": n, "total": a.total, "tiers": dict(tiers)}
    for axis, _ in AXES:
        print(f"\n{axis}   (trống: {100*none_hit[axis]/n:.1f}%)")
        rows = sorted(cnt[axis].items(), key=lambda x: -x[1])
        for k, v in rows:
            print(f"  {k:<28} {100*v/n:6.2f}%  ~{int(v*scale):>11,}")
        res[axis] = {k: int(v * scale) for k, v in rows}
        res[axis + "_pct"] = {k: round(100 * v / n, 3) for k, v in rows}
        res[axis + "_none_pct"] = round(100 * none_hit[axis] / n, 2)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
