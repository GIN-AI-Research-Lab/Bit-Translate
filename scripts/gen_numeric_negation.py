#!/usr/bin/env python3
"""Sinh data số liệu + phủ định bằng CODE template — 0 token LLM (PLAN_BUOC5 §4.1, Opus G-4).

Trị 2 lỗi:
  1. SỐ: quy đổi 万/億 (10万円 = 100.000 yên, KHÔNG phải 10.000!), loại từ (本/枚/台…),
     niên hiệu (令和6年=2024), ngày giờ, %, version/mã ticket (copy-through).
  2. PHỦ ĐỊNH: cặp tối thiểu khẳng định/phủ định (mất 「〜ていません」) + mẫu N2-N1
     lắt léo (わけではない, しか〜ない, ざるを得ない…) — bản dịch giữ ĐÚNG chiều logic.

Mỗi khung câu (frame) × nhiều bộ số ngẫu nhiên (seed cố định) → ~12k cặp, dedup.
Data 2 chiều bình thường (không field dir). JA dùng digit halfwidth (khớp NFKC của SPM).
Chạy: python scripts/gen_numeric_negation.py   → data/synthetic/vong3/numeric_negation.jsonl
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "synthetic" / "vong3" / "numeric_negation.jsonl"
rng = random.Random(20260718)
K = 600      # số lần quay mỗi frame (dedup sẽ cắt)
CAP_NUM = 11000   # trần phần số liệu sau dedup


def vnum(n):
    """100000 -> '100.000' (kiểu VN)."""
    return f"{n:,}".replace(",", ".")


def man(n):          # n万 -> số VN
    return vnum(n * 10_000)


def oku(n):          # n億 -> 'x triệu/tỷ'
    v = n * 100_000_000
    if v >= 1_000_000_000:
        t = v / 1_000_000_000
        return (f"{t:.1f}".replace(".", ",").rstrip("0").rstrip(",")) + " tỷ"
    return f"{v // 1_000_000} triệu"


def reiwa(n):
    return 2018 + n     # 令和1 = 2019


def heisei(n):
    return 1988 + n     # 平成1 = 1989


WEEKDAYS = [("月", "Hai"), ("火", "Ba"), ("水", "Tư"), ("木", "Năm"), ("金", "Sáu"), ("土", "Bảy")]
ITEMS_HON = [("ビール", "chai bia"), ("ペン", "cây bút"), ("傘", "cây ô"), ("ワイン", "chai rượu vang")]
ITEMS_MAI = [("コピー", "bản photo"), ("チケット", "tấm vé"), ("シャツ", "chiếc áo sơ mi"), ("書類", "tờ tài liệu")]
ITEMS_DAI = [("パソコン", "chiếc máy tính"), ("プリンター", "chiếc máy in"), ("サーバー", "con server"), ("車", "chiếc ô tô")]
ITEMS_SATSU = [("マニュアル", "cuốn tài liệu hướng dẫn"), ("本", "cuốn sách"), ("ノート", "cuốn sổ")]
ITEMS_HIKI = [("猫", "con mèo"), ("犬", "con chó"), ("金魚", "con cá vàng")]
PROJ = ["ABC", "VJP", "DEV", "OPS", "QA"]

# ==== FRAMES SỐ LIỆU: (hàm sinh (ja, vi)) ====


def F(fn):
    FRAMES.append(fn)
    return fn


FRAMES = []

# --- tiền 万/億 ---
F(lambda: (lambda n: (f"この案件の予算は{n}万円です。", f"Ngân sách của dự án này là {man(n)} yên."))(rng.randint(3, 500)))
F(lambda: (lambda n: (f"修理代は{n}万円かかりました。", f"Tiền sửa hết {man(n)} yên."))(rng.randint(1, 60)))
F(lambda: (lambda n: (f"月給は{n}万円からスタートします。", f"Lương tháng khởi điểm từ {man(n)} yên."))(rng.randint(18, 60)))
F(lambda: (lambda n: (f"あの会社の売上は{n}億円を超えたそうです。", f"Nghe nói doanh thu của công ty đó đã vượt {oku(n)} yên."))(rng.randint(1, 300)))
F(lambda: (lambda n: (f"来年度の投資額は{n}億円になる見込みです。", f"Số tiền đầu tư năm tài khóa tới dự kiến là {oku(n)} yên."))(rng.randint(1, 120)))
F(lambda: (lambda n, m: (f"見積もりは{n}万円でしたが、実際は{m}万円かかりました。", f"Báo giá là {man(n)} yên nhưng thực tế tốn {man(m)} yên."))(rng.randint(5, 200), rng.randint(5, 300)))
F(lambda: (lambda n: (f"このアパートの家賃は月{n}万円です。", f"Tiền thuê căn hộ này là {man(n)} yên một tháng."))(rng.randint(4, 25)))
F(lambda: (lambda n: (f"{n}万ドンあれば足りますか？", f"{man(n)} đồng thì đủ không?"))(rng.randint(5, 900)))

# --- loại từ ---
F(lambda: (lambda n, it: (f"{it[0]}を{n}本買ってきてください。", f"Mua giúp tôi {n} {it[1]} nhé."))(rng.randint(2, 12), rng.choice(ITEMS_HON)))
F(lambda: (lambda n, it: (f"{it[0]}を{n}枚印刷しておきました。", f"Tôi đã in sẵn {n} {it[1]} rồi."))(rng.randint(2, 30), rng.choice(ITEMS_MAI)))
F(lambda: (lambda n, it: (f"新しい{it[0]}を{n}台導入する予定です。", f"Dự định trang bị thêm {n} {it[1]} mới."))(rng.randint(2, 20), rng.choice(ITEMS_DAI)))
F(lambda: (lambda n, it: (f"{it[0]}を{n}冊送っていただけますか。", f"Anh/chị gửi giúp {n} {it[1]} được không?"))(rng.randint(2, 10), rng.choice(ITEMS_SATSU)))
F(lambda: (lambda n, it: (f"うちは{it[0]}を{n}匹飼っています。", f"Nhà tôi nuôi {n} {it[1]}."))(rng.randint(1, 5), rng.choice(ITEMS_HIKI)))
F(lambda: (lambda n: (f"会議室に{n}人集まっています。", f"Có {n} người đang tập trung ở phòng họp."))(rng.randint(3, 40)))

# --- niên hiệu / năm ---
F(lambda: (lambda n: (f"令和{n}年に入社しました。", f"Tôi vào công ty năm {reiwa(n)}."))(rng.randint(1, 8)))
F(lambda: (lambda n: (f"このシステムは令和{n}年から稼働しています。", f"Hệ thống này vận hành từ năm {reiwa(n)}."))(rng.randint(1, 8)))
F(lambda: (lambda n: (f"平成{n}年生まれです。", f"Tôi sinh năm {heisei(n)}."))(rng.randint(1, 31)))
F(lambda: (lambda n, m: (f"契約は令和{n}年{m}月までです。", f"Hợp đồng có hiệu lực đến tháng {m} năm {reiwa(n)}."))(rng.randint(5, 9), rng.randint(1, 12)))

# --- ngày giờ ---
F(lambda: (lambda m, d, w: (f"次の定例会議は{m}月{d}日（{w[0]}）です。", f"Cuộc họp định kỳ tới là thứ {w[1]} ngày {d} tháng {m}."))(rng.randint(1, 12), rng.randint(1, 28), rng.choice(WEEKDAYS)))
F(lambda: (lambda h, mi: (f"打ち合わせは午後{h}時{mi}分からです。", f"Buổi trao đổi bắt đầu lúc {h} giờ {mi} chiều."))(rng.randint(1, 5), rng.choice([15, 30, 45])))
F(lambda: (lambda h: (f"打ち合わせは午後{h}時からです。", f"Buổi trao đổi bắt đầu lúc {h} giờ chiều."))(rng.randint(1, 6)))
F(lambda: (lambda h: (f"面接は午前{h}時半の予定です。", f"Buổi phỏng vấn dự kiến lúc {h} giờ rưỡi sáng."))(rng.randint(8, 11)))
F(lambda: (lambda h: (f"毎朝{h}時に出勤しています。", f"Sáng nào tôi cũng đi làm lúc {h} giờ."))(rng.randint(6, 9)))
F(lambda: (lambda m, d: (f"締め切りは{m}月{d}日の23時59分です。", f"Hạn chót là 23 giờ 59 phút ngày {d} tháng {m}."))(rng.randint(1, 12), rng.randint(1, 28)))

# --- % / version / mã (copy-through) ---
F(lambda: (lambda n: (f"売上が前年比{n}%増加しました。", f"Doanh thu tăng {n}% so với năm trước."))(rng.randint(2, 45)))
F(lambda: (lambda n: (f"今なら全品{n}%オフです。", f"Bây giờ toàn bộ sản phẩm đang giảm {n}%."))(rng.choice([5, 10, 15, 20, 30, 50, 70])))
F(lambda: (lambda n: (f"CPU使用率が{n}%を超えています。", f"Mức sử dụng CPU đang vượt {n}%."))(rng.choice([70, 80, 85, 90, 95, 99])))
F(lambda: (lambda a, b, c: (f"バージョン{a}.{b}.{c}をリリースしました。", f"Đã phát hành phiên bản {a}.{b}.{c}."))(rng.randint(1, 9), rng.randint(0, 20), rng.randint(0, 9)))
F(lambda: (lambda a, b, c: (f"v{a}.{b}.{c}で修正済みです。", f"Đã sửa ở bản v{a}.{b}.{c}."))(rng.randint(1, 9), rng.randint(0, 20), rng.randint(0, 9)))
F(lambda: (lambda p, n: (f"チケット{p}-{n}を確認してもらえますか。", f"Anh/chị xem giúp ticket {p}-{n} được không?"))(rng.choice(PROJ), rng.randint(100, 9999)))
F(lambda: (lambda p, n: (f"{p}-{n}はもうクローズしました。", f"{p}-{n} đã đóng rồi."))(rng.choice(PROJ), rng.randint(100, 9999)))

# ==== FRAMES PHỦ ĐỊNH: cặp tối thiểu — mỗi lần gọi trả list 2 cặp ====

NEG_PAIRS = [
    ("原因が見つかりました。", "Đã tìm ra nguyên nhân.",
     "原因はまだ見つかっていません。", "Vẫn chưa tìm ra nguyên nhân."),
    ("レポートはもう提出しました。", "Tôi đã nộp báo cáo rồi.",
     "レポートはまだ提出していません。", "Tôi vẫn chưa nộp báo cáo."),
    ("この機能はもう使えます。", "Tính năng này dùng được rồi.",
     "この機能はまだ使えません。", "Tính năng này vẫn chưa dùng được."),
    ("彼から返事が来ました。", "Anh ấy đã trả lời rồi.",
     "彼からまだ返事が来ていません。", "Anh ấy vẫn chưa trả lời."),
    ("問題は解決しました。", "Vấn đề đã được giải quyết.",
     "問題は解決していません。", "Vấn đề vẫn chưa được giải quyết."),
    ("お金は全部払いました。", "Tôi đã trả hết tiền rồi.",
     "お金はまだ全部は払っていません。", "Tôi vẫn chưa trả hết tiền."),
    ("テストは全部通りました。", "Test đã pass hết.",
     "テストが全部は通っていません。", "Test vẫn chưa pass hết."),
    ("会議の資料は準備できています。", "Tài liệu họp đã chuẩn bị xong.",
     "会議の資料はまだ準備できていません。", "Tài liệu họp vẫn chưa chuẩn bị xong."),
]
NEG_N2N1 = [
    ("行きたくないわけではないんですが、時間がないんです。",
     "Không phải là tôi không muốn đi, chỉ là không có thời gian."),
    ("彼の気持ちが分からなくもないです。",
     "Cũng không phải là tôi không hiểu cảm giác của anh ấy."),
    ("できないことはないですが、時間がかかります。",
     "Không phải là không làm được, nhưng sẽ mất thời gian."),
    ("今日中に終わらせるしかないですね。",
     "Đành phải làm xong trong hôm nay thôi."),
    ("財布には500円しかありません。", "Trong ví chỉ còn đúng 500 yên."),
    ("この件は部長に報告せざるを得ません。",
     "Chuyện này đành phải báo cáo với trưởng phòng thôi."),
    ("高いからといって、品質がいいとは限らないです。",
     "Đắt không có nghĩa là chất lượng chắc chắn tốt."),
    ("急いだからといって、間に合うとは限りません。",
     "Vội thì vội chứ chưa chắc đã kịp."),
    ("笑わずにはいられなかったです。", "Tôi đã không thể nào nhịn cười được."),
    ("心配で、確認せずにはいられません。",
     "Lo quá nên không kiểm tra thì không chịu được."),
    ("嫌いというわけではなく、ただ苦手なだけです。",
     "Không phải là ghét, chỉ là không giỏi món đó thôi."),
    ("全員が賛成しているわけではありません。",
     "Không phải tất cả mọi người đều tán thành."),
    ("休みたくても休めない状況です。",
     "Tình hình là muốn nghỉ cũng không nghỉ được."),
    ("この量は一人では食べきれません。",
     "Chỗ này một người ăn không hết nổi."),
    ("涙をこらえきれませんでした。", "Tôi đã không kìm nổi nước mắt."),
    ("もう彼を待つ必要はありません。", "Không cần phải đợi anh ấy nữa."),
    ("明日は来なくてもいいですよ。", "Mai không đến cũng được đấy."),
    ("そこまでしなければならないんですか？", "Phải làm đến mức đó cơ à?"),
]


def main():
    seen, out = set(), []
    for _ in range(K):
        for fr in FRAMES:
            ja, vi = fr()
            if (ja, vi) not in seen:
                seen.add((ja, vi))
                out.append({"ja": ja, "vi": vi, "src": "vong3_num"})
    if len(out) > CAP_NUM:
        rng.shuffle(out)
        out = out[:CAP_NUM]
    for a_ja, a_vi, n_ja, n_vi in NEG_PAIRS:
        for p in [{"ja": a_ja, "vi": a_vi}, {"ja": n_ja, "vi": n_vi}]:
            if (p["ja"], p["vi"]) not in seen:
                seen.add((p["ja"], p["vi"]))
                out.append({**p, "src": "vong3_neg"})
    for ja, vi in NEG_N2N1:
        if (ja, vi) not in seen:
            seen.add((ja, vi))
            out.append({"ja": ja, "vi": vi, "src": "vong3_neg"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in out),
                   encoding="utf-8")
    n_num = sum(1 for p in out if p["src"] == "vong3_num")
    n_neg = len(out) - n_num
    print(f"GHI {len(out):,} cặp (số liệu {n_num:,} + phủ định {n_neg}) -> {OUT}")


if __name__ == "__main__":
    main()
