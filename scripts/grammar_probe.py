#!/usr/bin/env python3
"""Bộ DÒ NGỮ PHÁP — đo tự động model hiểu đúng cấu trúc hay không, không cần judge.

Vì sao cần bộ này bên cạnh bench judge: bench chỉ nói "câu này sai", không nói SAI Ở
ĐÂU. Đọc tay 17 câu v3 dịch sai trên bench câu thật cho thấy chỉ ~1/3 là lỗi THUẬT
NGỮ (長掌筋, 中央海嶺, ゼロ・サム・ゲーム), phần lớn còn lại là lỗi CẤU TRÚC:

  [106] 乗ってもらいました -> "được một chiếc Porsche đón nhận"  (đảo ngược hướng もらう)
  [115] 15世紀から19世紀   -> "thế kỷ 15 và 19"                  (mất から〜まで)
  [102] 言いたくなるところです -> mất hẳn sắc thái "dễ khiến ta muốn nói"
  [105] 100倍効率化        -> bỏ luôn "gấp 100 lần"              (rơi nội dung câu dài)

Đếm tần suất trong corpus KHÔNG phát hiện được loại lỗi này: もらう xuất hiện đầy
corpus, model vẫn dịch sai HƯỚNG. Nên cách đo đúng là DÒ HÀNH VI bằng cặp tối thiểu:
cùng một nội dung, chỉ đổi đúng một dấu hiệu ngữ pháp, rồi kiểm tra bản dịch có
đổi theo không.

Chấm tự động: mỗi probe khai báo `must` (regex PHẢI khớp bản dịch) và `must_not`
(regex KHÔNG được khớp). Không dùng LLM judge -> chạy lại sau mỗi vòng train miễn phí.

  python scripts/grammar_probe.py D:/Bit-Translate-data/checkpoints_v3/v3_avg.pt --label v3
  # -> eval/probe_v3.jsonl + bảng điểm theo nhóm ngữ pháp
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# ---------------------------------------------------------------------------
# Mỗi probe: (nhóm, câu Nhật, regex PHẢI có, regex KHÔNG được có, ghi chú)
# Viết theo CẶP TỐI THIỂU: các câu trong cùng một cụm chỉ khác nhau dấu hiệu ngữ pháp.
# ---------------------------------------------------------------------------
P = []


def add(group, ja, must, must_not="", note=""):
    P.append({"id": len(P) + 1, "group": group, "ja": ja,
              "must": must, "must_not": must_not, "note": note})


# --- 1. HƯỚNG CHO/NHẬN: あげる / くれる / もらう -----------------------------
# Lỗi đã bắt được ở bench câu thật [106]: 乗ってもらいました -> dịch ngược hướng.
add("benefactive", "私は田中さんに本を貸してあげました",
    r"(cho|giúp)\s+(anh|chị|ông|bà)?\s*Tanaka", r"Tanaka\s+(cho|đã cho)\s+tôi",
    "tôi CHO Tanaka mượn")
add("benefactive", "田中さんが私に本を貸してくれました",
    r"Tanaka\s+.{0,20}(cho|đã cho)\s+tôi", r"tôi\s+(cho|đã cho)\s+.{0,12}Tanaka",
    "Tanaka CHO tôi mượn")
add("benefactive", "私は田中さんに本を貸してもらいました",
    r"(tôi\s+.{0,25}(được|nhờ)|Tanaka\s+.{0,20}cho\s+tôi)", r"tôi\s+cho\s+.{0,12}Tanaka",
    "tôi ĐƯỢC Tanaka cho mượn")
add("benefactive", "部長に資料を確認していただきました",
    r"(được|nhờ).{0,40}(trưởng phòng|giám đốc)", r"tôi\s+(đã\s+)?(kiểm tra|xác nhận)\s+cho",
    "tôi ĐƯỢC trưởng phòng xác nhận")
add("benefactive", "先生が説明してくださいました",
    r"(thầy|cô|giáo viên).{0,30}(giải thích|đã giải thích)", r"tôi\s+đã\s+giải thích",
    "thầy giải thích CHO tôi")
add("benefactive", "私が先生に説明してさしあげました",
    r"tôi.{0,30}giải thích", r"(thầy|cô).{0,20}giải thích\s+cho\s+tôi",
    "tôi giải thích cho thầy")

# --- 2. BỊ ĐỘNG / SAI KHIẾN / SAI KHIẾN-BỊ ĐỘNG ------------------------------
add("voice", "彼は部長に叱られました", r"(bị|được).{0,30}(mắng|khiển trách|la)",
    r"anh ấy.{0,15}(mắng|khiển trách)\s+(trưởng phòng|sếp)", "anh ấy BỊ mắng")
add("voice", "部長は彼を叱りました", r"(trưởng phòng|sếp).{0,20}(mắng|khiển trách|la)",
    r"(bị|được).{0,20}(mắng|khiển trách)", "trưởng phòng MẮNG anh ấy")
add("voice", "部長は彼に報告書を書かせました",
    r"(bắt|bảo|yêu cầu|sai|cho|để).{0,30}viết",
    r"trưởng phòng.{0,10}(đã\s+)?viết\s+báo cáo\s*[.．]?$",
    "trưởng phòng BẮT anh ấy viết")
add("voice", "彼は部長に報告書を書かされました",
    r"(bị\s+.{0,30}(bắt|buộc)|(buộc|đành|bắt) phải viết|phải viết)", "",
    "anh ấy BỊ BẮT viết")
add("voice", "会議は来週に延期されました", r"(bị|được).{0,20}(hoãn|dời|lùi)", "",
    "cuộc họp BỊ hoãn")

# --- 3. PHẠM VI / GIỚI HẠN: から〜まで, 〜以上, 〜以内 ------------------------
# Lỗi bench [115]: 15世紀から19世紀 -> "thế kỷ 15 và 19"
add("range", "この展示は15世紀から19世紀までの作品を扱います",
    r"(từ|thế kỷ)\s*15.{0,20}(đến|tới)\s*(thế kỷ\s*)?19", r"15\s+và\s+19",
    "TỪ 15 ĐẾN 19, không phải '15 và 19'")
add("range", "受付は9時から17時までです", r"9\D{0,12}(đến|tới)\D{0,8}17", r"9\s+và\s+17",
    "từ 9 đến 17 giờ")
add("range", "参加者は50人以上でした", r"(trên|hơn|từ)\s*50", r"(dưới|ít hơn)\s*50",
    "TRÊN 50 người")
add("range", "参加者は50人以下でした", r"((dưới|không quá|tối đa|ít hơn)\s*50|50\s*(người\s*)?trở xuống)",
    r"(trên|hơn)\s*50", "DƯỚI 50 người")
add("range", "三日以内に返信してください", r"(trong|trong vòng|nội)\s*(vòng\s*)?(3|ba)\s*ngày",
    r"sau\s*(3|ba)\s*ngày", "TRONG VÒNG 3 ngày")

# --- 4. TÌNH THÁI CUỐI CÂU ---------------------------------------------------
# Lỗi bench [102]: 言いたくなるところです -> mất sắc thái
add("modality", "そう言いたくなるところです", r"(dễ|khiến|muốn|chực).{0,25}(nói|bảo)", "",
    "'dễ khiến ta muốn nói', không phải khẳng định")
add("modality", "彼は来るはずです", r"(chắc|hẳn|lẽ ra|đáng lẽ|theo dự kiến)", r"^\s*Anh ấy sẽ đến\s*[.．]?\s*$",
    "HẲN LÀ đến (suy đoán có căn cứ)")
add("modality", "彼は来るかもしれません", r"(có thể|có lẽ|biết đâu)", r"(chắc chắn|nhất định)",
    "CÓ THỂ đến")
add("modality", "彼は来るに違いありません", r"(chắc chắn|nhất định|không thể sai)", r"có thể\s+là",
    "CHẮC CHẮN đến")
add("modality", "この方法は失敗しかねません", r"(có thể|có nguy cơ|dễ).{0,20}(thất bại|hỏng)",
    r"không thể\s+thất bại", "CÓ NGUY CƠ thất bại")
add("modality", "参加せざるを得ませんでした", r"(buộc phải|đành phải|không thể không|phải)",
    r"(không\s+tham gia|đã từ chối)", "BUỘC PHẢI tham gia")
add("modality", "今出かけるところです", r"(sắp|vừa định|đang chuẩn bị)", "",
    "SẮP ra ngoài (chưa đi)")
add("modality", "今出かけたところです", r"(vừa|vừa mới)", r"sắp", "VỪA MỚI ra ngoài (đã đi)")

# --- 5. PHỦ ĐỊNH KÉP / PHỦ ĐỊNH PHẠM VI -------------------------------------
add("negation", "行かないわけではありません",
    r"(không phải\s+(là\s+)?(\w+\s+){0,3}không|vẫn|không hẳn)",
    r"^\s*(Tôi\s+)?không đi", "KHÔNG PHẢI LÀ không đi")
add("negation", "誰も来ませんでした", r"(không\s+(có\s+)?ai|chẳng\s+(có\s+)?ai)", r"(mọi người|ai cũng)\s+đã đến",
    "KHÔNG AI đến")
add("negation", "全員が来たわけではありません", r"(không phải\s+(tất cả|ai cũng)|không phải\s+toàn bộ)",
    r"^\s*(Tất cả|Mọi người).{0,15}(không đến|đã không)", "KHÔNG PHẢI TẤT CẢ đều đến")
add("negation", "彼しか来ませんでした", r"(chỉ|duy nhất).{0,20}(anh ấy|cậu ấy)", "",
    "CHỈ MỖI anh ấy đến")
add("negation", "彼は来なかったのではないかと思います", r"(không.{0,20}(đến|tới))", "",
    "e rằng anh ấy đã KHÔNG đến")

# --- 6. ĐIỀU KIỆN: と / ば / たら / なら --------------------------------------
add("conditional", "ボタンを押すと画面が変わります", r"(khi|hễ|cứ|thì).{0,40}(thay đổi|chuyển|đổi)",
    r"(nếu\s+bạn\s+muốn)", "HỄ nhấn LÀ đổi (quy luật)")
add("conditional", "雨が降ったら中止します", r"(nếu|trong trường hợp).{0,25}mưa", "",
    "NẾU mưa thì hủy")
add("conditional", "行くなら早く準備してください", r"(nếu|đã).{0,20}(đi|định đi)", "",
    "NẾU (đã định) đi thì chuẩn bị sớm")
add("conditional", "もっと早く言ってくれればよかったのに",
    r"(giá|phải chi|lẽ ra|ước gì)", r"^\s*Nếu\s+.{0,30}\s+thì\s+tốt\s*[.．]?\s*$",
    "GIÁ MÀ nói sớm hơn — tiếc nuối, việc đã không xảy ra")

# --- 7. NHƯỢNG BỘ / TƯƠNG PHẢN ----------------------------------------------
add("concessive", "努力したものの結果は出ませんでした", r"(tuy|dù|mặc dù|nhưng)", "",
    "TUY đã cố gắng NHƯNG không có kết quả")
add("concessive", "雨にもかかわらず試合は行われました", r"(dù|mặc dù|bất chấp)", "",
    "BẤT CHẤP mưa")
add("concessive", "残念ながらもう席がありません", r"(rất tiếc|tiếc là|đáng tiếc)", "",
    "RẤT TIẾC là hết chỗ")
add("concessive", "とはいえ問題が残っています", r"(tuy vậy|dù vậy|mặc dù vậy|nói vậy)", "",
    "TUY VẬY vẫn còn vấn đề")

# --- 8. TRÍCH DẪN / TRUYỀN ĐẠT ----------------------------------------------
add("quotative", "彼は来ないそうです", r"(nghe nói|nghe bảo|hình như.{0,10}nói|được biết)",
    r"^\s*Anh ấy\s+không\s+đến\s*[.．]?\s*$", "NGHE NÓI anh ấy không đến")
add("quotative", "彼は来なさそうです", r"(có vẻ|trông|dường như)", r"nghe nói",
    "CÓ VẺ anh ấy không đến (nhìn mà đoán)")
add("quotative", "会議は中止とのことです", r"(nghe nói|được biết|theo.{0,15}(thông báo|thì))", "",
    "ĐƯỢC BIẾT là hủy họp")
add("quotative", "彼によると価格が上がるらしいです", r"(theo|nghe).{0,20}(anh ấy|cậu ấy)", "",
    "THEO anh ấy thì giá sẽ tăng")

# --- 9. KÍNH NGỮ: ai làm gì cho ai ------------------------------------------
add("keigo", "社長がお見えになりました", r"(giám đốc|chủ tịch).{0,25}(đến|tới|có mặt)",
    r"tôi.{0,15}(đến|gặp)", "GIÁM ĐỐC đã đến (tôn kính -> chủ ngữ là giám đốc)")
add("keigo", "明日伺います", r"(tôi|chúng tôi).{0,25}(đến|ghé|thăm|qua)",
    r"(quý khách|anh|chị).{0,10}(đến|ghé)", "TÔI sẽ đến (khiêm nhường -> chủ ngữ là tôi)")
add("keigo", "資料を拝見しました", r"(tôi|chúng tôi).{0,25}(xem|đọc|nhận)",
    r"(quý khách|anh|chị).{0,10}xem", "TÔI đã xem (khiêm nhường)")
add("keigo", "ご覧になりましたか",
    r"(quý khách|anh|chị|ông|bà|ngài|bạn|cậu).{0,25}(xem|đã xem)",
    r"^\s*Tôi\s+đã\s+xem", "NGÀI đã xem chưa (tôn kính)")
add("keigo", "お待たせして申し訳ございません", r"(xin lỗi|thành thật xin lỗi|rất xin lỗi)", "",
    "xin lỗi vì đã để chờ")

# --- 10. LƯỢC CHỦ NGỮ TRONG HỘI THOẠI ---------------------------------------
add("zeropron", "「明日来ますか」「はい、行きます」",
    r"(vâng|dạ|có).{0,30}(tôi|mình)?.{0,10}(đi|đến|sẽ)", "", "trả lời: TÔI sẽ đi")
add("zeropron", "手伝ってくれてありがとう", r"(cảm ơn).{0,30}(giúp|hỗ trợ)", "",
    "cảm ơn vì ĐÃ GIÚP TÔI")
add("zeropron", "先に帰ってもいいですか", r"(tôi|mình|em).{0,25}(về|đi về)",
    r"(bạn|anh|chị).{0,10}(về trước|có thể về)", "TÔI về trước được không")
add("zeropron", "先に帰ってもいいですよ", r"(bạn|anh|chị|em|cậu).{0,25}(về|đi về)",
    r"^\s*Tôi\s+", "BẠN về trước cũng được")

# --- 11. SỐ LIỆU / ĐƠN VỊ (dễ rơi trong câu dài) ----------------------------
# Lỗi bench [105]: 100倍効率化 -> bỏ luôn "gấp 100 lần"
add("number", "この方法で作業を100倍効率化できます", r"(100|một trăm).{0,15}lần", "",
    "GẤP 100 LẦN — không được bỏ")
add("number", "売上は前年比で3割減少しました", r"(30\s*%|ba mươi phần trăm|3\s*phần\s*10|30 phần trăm)",
    r"(3\s*%(?!\d))", "3割 = 30%, KHÔNG phải 3%")
add("number", "会場には約1万5千人が集まりました", r"(15[.,．]?000|15 000|1[.,．]?5 vạn|mười lăm nghìn|15 nghìn)",
    r"1[.,．]?000\D", "1万5千 = 15.000")
add("number", "気温は零下5度まで下がりました", r"(âm|-|dưới không).{0,8}5", r"(?<![âm\-])\b5 độ C? trên",
    "ÂM 5 độ")


# --- 12. CÙNG ĐIỂM NGỮ PHÁP ĐÓ NHƯNG NHÚNG TRONG CÂU DÀI --------------------
# PHÉP THỬ QUYẾT ĐỊNH của cả bộ probe. v3 làm đúng gần hết các câu NGẮN ở trên
# (48/54) trong khi trên bench câu THẬT nó sai 43% ở nhóm câu dài. Hai con số đó
# chỉ dung hòa được nếu vấn đề KHÔNG phải "model không biết ngữ pháp" mà là "model
# mất dấu cấu trúc khi câu dài ra".
#
# Nhóm probe này giữ NGUYÊN lõi ngữ nghĩa của bản ngắn, chỉ bọc thêm mệnh đề phụ và
# thông tin gây nhiễu. Nếu ngắn ĐÚNG mà dài SAI trên cùng một điểm ngữ pháp thì
# nút thắt là DUNG LƯỢNG/CHÚ Ý, và câu trả lời là tăng model hoặc tăng data câu dài
# — chứ không phải bơm thêm data dạy đúng điểm ngữ pháp đó (model đã biết rồi).
add("long_benefactive",
    "先週の打ち合わせで話題に出た件ですが、資料が見つからなくて困っていたところ、"
    "同じ部署の田中さんに古いファイルを貸してもらいました",
    r"((tôi|đã|và)\s+.{0,40}(được|nhờ|mượn được)|được\s+.{0,25}cho\s+mượn|(Tanaka|anh ấy)\s+.{0,25}cho\s+tôi)",
    r"tôi\s+(đã\s+)?cho\s+.{0,15}Tanaka\s+mượn", "TÔI ĐƯỢC Tanaka cho mượn (dài)")
add("long_benefactive",
    "資料の準備が間に合わないという話になったので、私のほうから田中さんに"
    "去年の古いファイルを貸してあげました",
    r"tôi\s+.{0,40}cho\s+(anh\s+|chị\s+)?Tanaka",
    r"Tanaka\s+.{0,25}cho\s+tôi\s+mượn", "TÔI CHO Tanaka mượn (dài)")
add("long_range",
    "今回の企画展はテーマを絞っておらず、ヨーロッパ各地の美術館から借りた"
    "15世紀から19世紀までの作品を、時代順ではなく主題ごとに並べて展示します",
    r"(từ\s*)?(thế kỷ\s*)?15.{0,25}(đến|tới)\s*(thế kỷ\s*)?19", r"15\s+và\s+19",
    "TỪ thế kỷ 15 ĐẾN 19 (dài)")
add("long_number",
    "手作業でやっていた確認工程を見直したところ、単純な繰り返し作業を"
    "自動化するだけで、全体の処理を100倍効率化できることが分かりました",
    r"(100|một trăm)\s*lần", "", "GẤP 100 LẦN — không được rơi (dài)")
add("long_number",
    "昨年から続いていた原材料費の高騰と円安の影響が重なり、"
    "今期の売上は前年比で3割減少しました",
    r"(30\s*%|30 phần trăm|ba mươi phần trăm)", r"3\s*%(?!\d)", "3割 = 30% (dài)")
add("long_negation",
    "何度も誘っていただいて本当にありがたいのですが、"
    "予定が立て込んでいるだけで、決して行きたくないというわけではありません",
    r"((không phải|không có nghĩa)\s+(là\s+)?(\w+\s+){0,4}không|không hẳn|đâu phải)",
    r"^\s*(Tôi\s+)?không\s+muốn\s+đi\s*[.．]?\s*$", "KHÔNG PHẢI LÀ không muốn đi (dài)")
add("long_negation",
    "案内は全員に送ったはずなのですが、当日は雪の影響で交通が乱れたこともあり、"
    "参加を予定していた人が全員来たわけではありません",
    r"(không phải\s+(tất cả|ai cũng|toàn bộ)|không phải\s+là\s+(tất cả|ai cũng))",
    "", "KHÔNG PHẢI TẤT CẢ đều đến (dài)")
add("long_modality",
    "この二つの特徴だけを並べて見ると、世界チャンピオン大会レベルの対局は"
    "自動化からかけ離れたものだと言いたくなるところです",
    r"(dễ|khiến|muốn|chực|có xu hướng).{0,30}(nói|bảo|kết luận|cho rằng)", "",
    "'dễ khiến ta muốn nói' — không được thành khẳng định (dài)")
add("long_modality",
    "スケジュールにはまだ余裕があるように見えますが、"
    "レビューの回数を減らして進めると、後の工程で品質の問題が起きかねません",
    r"(có thể|có nguy cơ|dễ).{0,30}(xảy ra|phát sinh|gặp|nảy sinh|dẫn đến)",
    r"không thể\s+(xảy ra|phát sinh)", "CÓ NGUY CƠ phát sinh (dài)")
add("long_voice",
    "納期が近いのに担当者が急に休むことになったため、"
    "私は部長に週末の報告書を書かされました",
    r"(bị\s+.{0,35}(bắt|buộc|yêu cầu)|(buộc|đành) phải viết|phải viết)",
    r"tôi\s+(đã\s+)?bắt\s+trưởng phòng", "TÔI BỊ BẮT viết (dài)")
add("long_voice",
    "先方との調整に時間がかかっているという連絡があり、"
    "来月に予定していた会議は再来週まで延期されました",
    r"(bị|được).{0,25}(hoãn|dời|lùi)", "", "cuộc họp BỊ hoãn (dài)")
add("long_quotative",
    "朝から何度か連絡を入れているのですが返事がなく、"
    "同じチームの人の話では、彼は今日は来ないそうです",
    r"(nghe nói|nghe bảo|theo lời|được biết|theo.{0,20}thì)", "",
    "NGHE NÓI anh ấy không đến (dài)")
add("long_keigo",
    "本日はお忙しい中お集まりいただきありがとうございます。"
    "先ほどお配りした資料について、もうご覧になりましたでしょうか",
    r"(quý vị|quý khách|anh|chị|ông|bà|ngài|bạn|các bạn|mọi người).{0,30}(xem|đã xem)",
    r"^\s*Tôi\s+đã\s+xem", "QUÝ VỊ đã xem chưa (dài)")
add("long_zeropron",
    "スタンフォード大学にRevsプログラムというものがあると聞いていたので、"
    "せっかくの機会だと思ってジョンをそこへ連れて行き、"
    "1960年型のポルシェに乗ってもらいました",
    r"(cho|để|mời).{0,25}(John|anh ấy|cậu ấy).{0,25}(ngồi|lái|lên|đi|cưỡi|thử)",
    r"(tôi|chúng tôi).{0,20}(được|bị).{0,20}(Porsche|xe)",
    "cho John NGỒI LÊN xe — không phải 'tôi được xe đón nhận' (dài)")


# --- 13. THANG ĐỘ DÀI: cùng điểm ngữ pháp ở 3 mức ngắn / vừa / rất dài -------
# Kết quả đợt đầu (v3: ngắn 52/54 = 96%, dài 11/14 = 79%) cho thấy model BIẾT ngữ
# pháp nhưng mất dấu khi câu dài. n=14 còn ít nên nhóm này thêm mức TRUNG GIAN và
# mức RẤT DÀI cho đúng ba điểm đã gãy (phạm vi / trích dẫn / hướng cho-nhận) cộng
# ba điểm còn giữ được, để biết ngưỡng gãy nằm ở đâu chứ không chỉ biết có gãy.
add("mid_range", "ヨーロッパ各地から借りた15世紀から19世紀までの作品を展示します",
    r"(từ\s*)?(thế kỷ\s*)?15.{0,25}(đến|tới)\s*(thế kỷ\s*)?19", r"15\s+và\s+19",
    "TỪ 15 ĐẾN 19 (vừa)")
add("mid_quotative", "何度か連絡したのですが、彼は今日は来ないそうです",
    r"(nghe nói|nghe bảo|theo lời|được biết|hình như)", "", "NGHE NÓI (vừa)")
add("mid_benefactive", "資料が見つからず困っていたので、田中さんに貸してもらいました",
    r"(tôi\s+.{0,35}(được|nhờ|mượn được)|Tanaka\s+.{0,25}cho\s+tôi)",
    r"tôi\s+(đã\s+)?cho\s+.{0,15}Tanaka\s+mượn", "TÔI ĐƯỢC cho mượn (vừa)")
add("mid_number", "作業を見直した結果、全体の処理を100倍効率化できました",
    r"(100|một trăm)\s*lần", "", "GẤP 100 LẦN (vừa)")
add("mid_negation", "予定が立て込んでいるだけで、行きたくないわけではありません",
    r"(không phải\s+(là\s+)?(\w+\s+){0,4}không|không hẳn|đâu phải)", "",
    "KHÔNG PHẢI LÀ không muốn đi (vừa)")
add("mid_voice", "担当者が急に休んだため、私は部長に報告書を書かされました",
    r"(bị\s+.{0,35}(bắt|buộc|yêu cầu)|(buộc|đành) phải viết|phải viết)", "",
    "TÔI BỊ BẮT viết (vừa)")

add("xlong_range",
    "今回の企画展は特定の流派に絞らない方針で構成されており、ロンドンやパリ、"
    "マドリードの美術館から特別に借り受けた15世紀から19世紀までの油彩作品を、"
    "制作年代の順ではなく主題ごとにまとめ、各室の入口に解説パネルを添えて展示します",
    r"(từ\s*)?(thế kỷ\s*)?15.{0,25}(đến|tới)\s*(thế kỷ\s*)?19", r"15\s+và\s+19",
    "TỪ 15 ĐẾN 19 (rất dài)")
add("xlong_quotative",
    "朝から何度も電話とメールで連絡を入れているのですが一向に返事がなく、"
    "先ほど廊下で会った同じチームの人の話では、体調を崩したとかで"
    "彼は今日は来ないそうです",
    r"(nghe nói|nghe bảo|theo lời|được biết|theo.{0,25}thì)", "",
    "NGHE NÓI anh ấy không đến (rất dài)")
add("xlong_benefactive",
    "先週の打ち合わせで話題に出た古い案件の件で、社内のどこを探しても資料が"
    "見つからず困り果てていたのですが、たまたま廊下で会った同じ部署の田中さんに"
    "事情を話したところ、当時のファイルを貸してもらいました",
    r"(tôi\s+.{0,45}(được|nhờ|mượn được)"
    r"|(Tanaka|anh ấy|cậu ấy)\s+.{0,30}cho\s+tôi\s+(mượn|dùng))",
    r"tôi\s+(đã\s+)?cho\s+.{0,15}Tanaka\s+mượn", "TÔI ĐƯỢC Tanaka cho mượn (rất dài)")
add("xlong_number",
    "これまで担当者が目視で一件ずつ確認していた工程を洗い出し、"
    "判断が要らない単純な繰り返し作業だけを切り出して自動化した結果、"
    "人員を増やさずに全体の処理を100倍効率化できることが分かりました",
    r"(100|một trăm)\s*lần", "", "GẤP 100 LẦN (rất dài)")
add("xlong_negation",
    "毎回声をかけていただいて本当にありがたく思っておりますし、"
    "皆さんとお会いするのを楽しみにしていないわけでもないのですが、"
    "今期は別件の予定が立て込んでいるだけで、決して行きたくないというわけではありません",
    r"(không phải\s+(là\s+)?(\w+\s+){0,4}không|không hẳn|đâu phải)", "",
    "KHÔNG PHẢI LÀ không muốn đi (rất dài)")
add("xlong_voice",
    "月末の納期が迫っていた上に、資料をまとめていた担当者が家庭の事情で"
    "急に一週間休むことになってしまったため、代わりに手が空いていた"
    "私は部長に週末の報告書を書かされました",
    r"(bị\s+.{0,35}(bắt|buộc|yêu cầu)|(buộc|đành) phải viết|phải viết)", "",
    "TÔI BỊ BẮT viết (rất dài)")


# ---------------------------------------------------------------------------
def score(hyp, pr):
    ok_must = bool(re.search(pr["must"], hyp, re.I)) if pr["must"] else True
    ok_not = not re.search(pr["must_not"], hyp, re.I) if pr["must_not"] else True
    return ok_must and ok_not, ok_must, ok_not


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", help="đường dẫn .pt; bỏ trống = chỉ in bộ probe")
    ap.add_argument("--label", default="model")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-new", type=int, default=180)
    a = ap.parse_args()

    if not a.ckpt:
        g = defaultdict(int)
        for pr in P:
            g[pr["group"]] += 1
        print(f"{len(P)} probe / {len(g)} nhóm")
        for k, v in sorted(g.items(), key=lambda x: -x[1]):
            print(f"  {k:14s} {v:>3}")
        return

    # dùng lại đúng đường dịch của bench (cùng chuẩn hóa, cùng rep_penalty) để
    # điểm probe so sánh được với điểm bench
    from translate_bench import run_ckpt
    hyps = run_ckpt([{"src": pr["ja"]} for pr in P], a.ckpt)

    out = Path(a.out or ROOT / "eval" / f"probe_{a.label}.jsonl")
    agg = defaultdict(lambda: [0, 0])
    fails = []
    with out.open("w", encoding="utf-8") as f:
        for pr, hyp in zip(P, hyps):
            ok, om, on = score(hyp, pr)
            agg[pr["group"]][0] += ok
            agg[pr["group"]][1] += 1
            f.write(json.dumps({**pr, "hyp": hyp, "ok": ok,
                                "thieu_must": not om, "dinh_must_not": not on},
                               ensure_ascii=False) + "\n")
            if not ok:
                fails.append((pr, hyp, om, on))

    tot_ok = sum(v[0] for v in agg.values())
    print(f"\n=== DÒ NGỮ PHÁP: {a.label} — {tot_ok}/{len(P)} "
          f"({100*tot_ok/len(P):.0f}%) ===")

    # Bảng THEO ĐỘ DÀI là bảng đáng đọc nhất: cùng điểm ngữ pháp, chỉ khác độ dài
    # câu. Nếu ngắn cao mà dài thấp thì nút thắt là DUNG LƯỢNG, không phải kiến thức.
    tier = defaultdict(lambda: [0, 0])
    for pr in P:
        g = pr["group"]
        t = ("2.vừa" if g.startswith("mid_") else
             "3.dài" if g.startswith("long_") else
             "4.rất dài" if g.startswith("xlong_") else "1.ngắn")
        ok = score(hyps[pr["id"] - 1], pr)[0]
        tier[t][0] += ok
        tier[t][1] += 1
    print("  --- theo ĐỘ DÀI CÂU (cùng các điểm ngữ pháp) ---")
    for t in sorted(tier):
        ok, n = tier[t]
        print(f"  {t:12s} {ok:>2}/{n:<3} {100*ok/n:5.0f}%")
    print("  --- theo NHÓM NGỮ PHÁP ---")
    for k in sorted(agg, key=lambda x: agg[x][0] / agg[x][1]):
        ok, n = agg[k]
        print(f"  {k:18s} {ok:>2}/{n:<2} {100*ok/n:5.0f}%")
    print(f"\n--- {len(fails)} probe TRƯỢT ---")
    for pr, hyp, om, on in fails:
        why = "thiếu dấu hiệu bắt buộc" if not om else "dính mẫu cấm (dịch ngược/sai)"
        print(f"[{pr['group']}] {pr['ja']}")
        print(f"   -> {hyp}")
        print(f"   ({why}; đúng phải là: {pr['note']})")
    print(f"\nGhi: {out}")


if __name__ == "__main__":
    main()
