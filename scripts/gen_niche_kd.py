#!/usr/bin/env python3
"""Sinh data KD NICHE ja->vi (thành ngữ / keigo) qua REST Gemini, song song nhiều key.

Mục đích: enrich 2 domain FAIL nặng nhất của pilot KD-100M (thành ngữ 9/10 fail,
keigo 7/10 fail) để test giả thuyết "nút thắt là DATA coverage, không phải dung
lượng". Xem HANDOFF_KD100M.md mục 5 & 8.

Hai mode:
  idiom  — seed từ data/synthetic/gen/idiom_glosses.jsonl (1223 quán ngữ đã harvest,
           ưu tiên bản _reviewed nếu có). Mỗi request: N idiom x K câu, trong 1 ngữ
           cảnh (scene) ngẫu-định để câu không lặp khuôn.
  keigo  — seed = tổ hợp (pattern keigo x tình huống). Có nhóm pattern CONTRAST
           (させていただく vs していただく ...) nhắm trực diện lỗi ĐẢO HƯỚNG kính ngữ.

QC tái dùng check() của filter_kd_full.py + kiểm tra riêng theo mode (idiom phải
thật sự xuất hiện trong câu ja; keigo phải có marker kính ngữ).

Chạy (Windows):
  set PYTHONUTF8=1
  python scripts/gen_niche_kd.py idiom --target 30000
  python scripts/gen_niche_kd.py keigo --target 80000

Resume: cứ chạy lại cùng lệnh — file out được append, dedup theo câu ja đã có.
"""
import argparse
import asyncio
import json
import os
import queue
import random
import re
import sys
import threading
import time
import unicodedata
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from filter_kd_full import check, norm  # noqa: E402

OUTDIR = ROOT / "data" / "synthetic" / "gen_niche"
GLOSS = ROOT / "data" / "synthetic" / "gen" / "idiom_glosses.jsonl"
GLOSS_REVIEWED = ROOT / "data" / "synthetic" / "gen" / "idiom_glosses_reviewed.jsonl"

# ---------------------------------------------------------------- seed: idiom
# Ngữ cảnh để cùng 1 idiom sinh ra câu khác nhau qua các vòng (tránh sinh lặp).
SCENES_IDIOM = [
    "hội thoại thân mật giữa bạn bè",
    "hội thoại trong công ty giữa đồng nghiệp",
    "cấp dưới báo cáo/xin ý kiến cấp trên (giọng lịch sự vừa)",
    "tin nhắn LINE/chat ngắn, khẩu ngữ",
    "bài viết blog cá nhân kể chuyện",
    "đoạn tiểu thuyết/truyện có tự thuật nội tâm",
    "bài báo/tin tức giọng viết trung tính",
    "phỏng vấn tuyển dụng hoặc phỏng vấn báo chí",
    "email nội bộ trao đổi công việc",
    "phàn nàn/bực mình về việc ở nhà hoặc chỗ làm",
    "kể lại chuyện đã xảy ra trong quá khứ",
    "tư vấn/khuyên bảo người khác",
    "bình luận về thể thao, phim, giải trí",
    "trao đổi chuyện tiền bạc, chi tiêu, giá cả",
    "hội thoại gia đình (vợ chồng, cha mẹ - con)",
    "thuyết trình/họp trình bày kế hoạch",
]

# --------------------------------------------------------------- seed: keigo
# Nhóm 1: 謙譲語 — người NÓI tự hạ mình (chủ thể = mình/bên mình).
KEIGO_HUMBLE = [
    "〜させていただく (xin phép được làm — chủ thể là NGƯỜI NÓI)",
    "〜いたします / お〜いたします (tôi/chúng tôi xin làm)",
    "伺う・お伺いする (tôi xin hỏi / tôi xin đến)",
    "拝見する・拝聴する (tôi xin được xem/nghe)",
    "申し上げる・ご連絡差し上げる (tôi xin nói/xin liên hệ)",
    "存じております・存じ上げません (tôi biết / tôi không biết — khiêm nhường)",
    "かしこまりました・承知いたしました (tôi đã hiểu, xin vâng)",
    "頂戴する・いただく (tôi xin nhận)",
    "〜ております (thể khiêm nhường của 〜ている, chủ thể là bên mình)",
    "お目にかかる・お世話になっております (chào hỏi kinh doanh khiêm nhường)",
    "恐縮ながら〜させていただきたく存じます (văn viết email rất khiêm)",
]
# Nhóm 2: 尊敬語 — tôn NGƯỜI NGHE / người thứ ba (chủ thể = đối phương).
KEIGO_RESPECT = [
    "いらっしゃる・おいでになる (chủ thể là ĐỐI PHƯƠNG)",
    "ご覧になる・お聞きになる (đối phương xem/nghe)",
    "おっしゃる (đối phương nói)",
    "なさる・される (đối phương làm)",
    "お〜になる (vd お帰りになる — đối phương làm)",
    "ご〜ください・お〜ください (mời/xin đối phương làm)",
    "〜ていただけますか・〜ていただけますでしょうか (nhờ ĐỐI PHƯƠNG làm)",
    "〜くださいますようお願い申し上げます (văn viết trang trọng nhờ đối phương)",
    "ご〜いただき、ありがとうございます (cảm ơn vì đối phương đã làm)",
    "召し上がる・お召しになる (đối phương ăn/mặc)",
]
# Nhóm 3: クッション言葉 — mở đầu giảm nhẹ.
KEIGO_CUSHION = [
    "恐れ入りますが〜 (dạ xin phép, cho tôi hỏi/nhờ...)",
    "差し支えなければ〜 (nếu không có gì bất tiện...)",
    "お手数ですが〜 (mong anh/chị chịu khó...)",
    "申し訳ございませんが〜 (thành thật xin lỗi nhưng...)",
    "あいにくですが〜・せっかくですが〜 (rất tiếc là...)",
    "念のため申し添えますと〜 (xin nói thêm cho chắc...)",
]
# Nhóm 4: CONTRAST — cặp câu gần y hệt nhưng ĐẢO chủ thể. Nhắm trực diện lỗi
# "検討させていただく -> mong quý khách xem xét" (dịch ngược hướng kính ngữ).
KEIGO_CONTRAST = [
    "CẶP ĐỐI CHIẾU: 〜させていただきます (NGƯỜI NÓI làm) vs 〜していただきます (ĐỐI PHƯƠNG làm)",
    "CẶP ĐỐI CHIẾU: ご確認いたします (tôi kiểm tra) vs ご確認いただけますか (nhờ anh/chị kiểm tra)",
    "CẶP ĐỐI CHIẾU: 検討させていただきます (bên tôi sẽ xem xét) vs ご検討ください (mong quý vị xem xét)",
    "CẶP ĐỐI CHIẾU: お送りいたします (tôi gửi) vs お送りいただけますか (nhờ anh/chị gửi)",
    "CẶP ĐỐI CHIẾU: 説明させていただきます (tôi xin giải thích) vs ご説明いただけますか (nhờ anh/chị giải thích)",
    "CẶP ĐỐI CHIẾU: 伺います (tôi xin hỏi/đến) vs お聞きになりますか (anh/chị có hỏi không)",
    "CẶP ĐỐI CHIẾU: 拝見しました (tôi đã xem) vs ご覧になりましたか (anh/chị đã xem chưa)",
    "CẶP ĐỐI CHIẾU: お待ちいたします (tôi sẽ đợi) vs お待ちいただけますか (nhờ anh/chị đợi)",
    "CẶP ĐỐI CHIẾU: 訪問させていただきます (tôi xin đến thăm) vs ご訪問いただけますか (mời anh/chị đến)",
    "CẶP ĐỐI CHIẾU: 対応いたします (bên tôi xử lý) vs ご対応いただきありがとうございます (cảm ơn bên anh/chị đã xử lý)",
]
SCENES_KEIGO = [
    "điện thoại doanh nghiệp (nhận/gọi, chuyển máy, nhắn lại)",
    "email gửi khách hàng ngoài công ty",
    "xin lỗi khi có sự cố/hàng lỗi/chậm giao",
    "phỏng vấn tuyển dụng (ứng viên và nhà tuyển dụng)",
    "họp với đối tác, đàm phán điều kiện hợp đồng",
    "quầy lễ tân/tiếp khách đến công ty",
    "thông báo nội bộ, báo cáo cấp trên",
    "xin hoãn deadline hoặc xin nghỉ phép",
    "nhà hàng, khách sạn phục vụ khách",
    "bệnh viện, phòng khám tiếp bệnh nhân",
    "ngân hàng, thủ tục hành chính, cơ quan công quyền",
    "giáo viên trao đổi với phụ huynh",
    "giới thiệu/bán sản phẩm cho khách",
    "hỗ trợ kỹ thuật IT helpdesk cho người dùng",
    "chào hỏi, giới thiệu bản thân, danh thiếp lần đầu gặp",
    "gửi báo giá, hối thúc thanh toán, xác nhận đơn hàng",
    "mời tham dự sự kiện, cảm ơn sau sự kiện",
    "từ chối lời đề nghị một cách lịch sự",
]

# ---------------------------------------------------------- seed: slang
# Corpus chỉ có 0,32% slang mà hardbench chrF slang = 26,0 (THẤP NHẤT, và là
# domain duy nhất THUA Google -1,1). Đây là lỗ hổng data rõ nhất sau idiom/keigo.
SLANG_CATS = [
    "ネットスラング hiện đại (草、それな、わかる、尊い、推し、沼、語彙力、しんどい)",
    "若者言葉 (めっちゃ、ワンチャン、ガチ、エグい、バイブス、まじ、やばい)",
    "略語・縮約 khi chat (りょ、おつ、とりあえず→とりあえ、あざす、なるはや)",
    "SNS/chat: バズる、リプ、DM、垢、フォロバ、匂わせ、実況、リアタイ",
    "khẩu ngữ rút gọn trong nói (〜っす、〜じゃん、〜だろ、〜てか、〜みたいな、〜的な)",
    "cảm xúc bực/chê (うざい、きもい、だるい、ムカつく、萎える、めんどい)",
    "cảm xúc khen/hype (神、優勝、ヤバい(褒め)、鬼〜、爆速、映える)",
    "khẩu ngữ công sở không chính thức (バッファ、アサイン、ペンディング、巻き取る、ざっくり)",
    "slang game/anime/otaku (廃課金、ガチャ、詰み、初見、伏線、作画、供給)",
    "khẩu ngữ gia đình/bạn bè thân (〜っしょ、〜わ、〜っての、マジで？、うそでしょ)",
    "slang mô tả người (陽キャ、陰キャ、天然、ドS、コミュ障、意識高い系)",
    "cách nói phóng đại/trend (〜すぎる、〜しか勝たん、〜案件、〜ガチャ、〜みが深い)",
]

# ------------------------------------------------------- seed: phủ định
# Corpus chỉ 0,06% mẫu phủ định phức — mà "đảo nghĩa/polarity" là nhóm nguyên
# nhân fail #4 của pilot. Dùng lại kỹ thuật CẶP ĐỐI CHIẾU đã hiệu quả ở keigo.
NEG_PATTERNS = [
    "CẶP ĐỐI CHIẾU: 〜わけではない (không phải là...) vs 〜わけだ (tức là, hóa ra)",
    "CẶP ĐỐI CHIẾU: 〜ないとは言えない (không thể nói là không) vs 〜とは言えない (không thể nói là)",
    "CẶP ĐỐI CHIẾU: 〜なくはない (cũng không phải là không) vs 〜はない (không có)",
    "CẶP ĐỐI CHIẾU: 〜とは限らない (không hẳn/không nhất thiết) vs 〜に限る (chỉ có... mới)",
    "CẶP ĐỐI CHIẾU: 〜ないわけにはいかない (không thể không làm) vs 〜わけにはいかない (không thể làm)",
    "CẶP ĐỐI CHIẾU: 〜てもらえますか (nhờ làm giúp) vs 〜てもらえませんか (CŨNG là nhờ, KHÔNG phải phủ định)",
    "CẶP ĐỐI CHIẾU: 〜ないでください (xin đừng) vs 〜てください (xin hãy)",
    "CẶP ĐỐI CHIẾU: 〜しかない (chỉ còn cách) vs 〜だけではない (không chỉ có)",
    "CẶP ĐỐI CHIẾU: 〜どころではない (không phải lúc) vs 〜どころか (chứ đừng nói)",
    "CẶP ĐỐI CHIẾU: 〜かねない (có nguy cơ xảy ra) vs 〜かねる (khó mà làm được)",
    "二重否定: 〜ないではいられない / 〜ずにはいられない (không thể không...)",
    "二重否定: 〜なしには〜ない (không có... thì không thể)",
    "〜ざるを得ない (buộc phải, không thể không)",
    "〜わけがない・〜はずがない (không thể nào có chuyện)",
    "〜ないことはない (không phải là không thể)",
    "〜どころか〜ない (chẳng những không... mà còn)",
    "反語 (câu hỏi tu từ mang nghĩa phủ định: 〜だろうか、〜ものか)",
    "部分否定 vs 全体否定: 全部〜ない (không cái nào) vs 全部が〜わけではない (không phải tất cả)",
]
# Nhóm RIÊNG: câu NHỜ VẢ mang hình thức phủ định — lượt sinh đầu chỉ ra 3%
# (598/18.457) dù đây là loại model dễ dịch ngược nhất (nhờ vả -> từ chối).
# Nhân trọng số khi trộn để bù.
NEG_REQUEST = [
    "câu NHỜ VẢ hình thức phủ định: 〜てもらえませんか / 〜ていただけませんか (VẪN là nhờ vả lịch sự)",
    "câu MỜI hình thức phủ định: 〜ませんか / 〜ないか (rủ rê, KHÔNG phải từ chối)",
    "câu XIN PHÉP hình thức phủ định: 〜てもかまいませんか / 〜てはいけませんか",
    "câu ĐỀ NGHỊ nhẹ dạng phủ định: 〜たらどうでしょうか / 〜ないほうがいいのでは",
    "câu XÁC NHẬN dạng phủ định: 〜ではありませんか / 〜じゃないですか (nhấn mạnh, không phủ định)",
    "ĐỐI CHIẾU: 〜てもらえませんか (nhờ vả) vs 〜てもらえないんです (nêu khó khăn/không được giúp)",
    "ĐỐI CHIẾU: 〜ませんか (rủ) vs 〜ません (phủ định thật)",
]

# ----------------------------------------------------- seed: katakana
# Lỗi thật của pilot: リハーサル→"real", TBD→"TMD". Từ ngoại lai/viết tắt hỏng.
KATA_CATS = [
    "katakana IT/phần mềm (デプロイ、リファクタ、マージ、ロールバック、キャッシュ)",
    "katakana business/họp (アジェンダ、リスケ、コンセンサス、ドラフト、フィックス)",
    "viết tắt tiếng Anh trong công sở Nhật (TBD、ASAP、KPI、PoC、MTG、ROI、FYI、NDA)",
    "katakana sản xuất/kỹ thuật (トルク、キャリブレーション、シーケンス、バルブ)",
    "katakana y tế/sức khỏe (カルテ、オペ、リハビリ、アレルギー、サプリ)",
    "katakana tài chính/kế toán (キャッシュフロー、ヘッジ、ポートフォリオ、レバレッジ)",
    "katakana marketing (ペルソナ、コンバージョン、リード、インプレッション)",
    "katakana thời trang/mỹ phẩm (シルエット、コーデ、トレンド、ファンデ)",
    "katakana ẩm thực (ソムリエ、パティシエ、アラカルト、テイクアウト)",
    "katakana thể thao (リハーサル、フォーメーション、ルーティン、コンディション)",
    "katakana xây dựng/bất động sản (リノベ、ゼネコン、テナント、レイアウト)",
    "katakana xe/logistics (サスペンション、ロジ、トレーサビリティ、リコール)",
    "katakana hay bị dịch sai vì giống từ Anh khác (リハーサル、リハビリ、レジュメ、モラル)",
    "wasei-eigo (từ Anh kiểu Nhật): サラリーマン、ワンオペ、コンセント、マンション、ペーパードライバー",
]

# --------------------------------------------------------- seed: câu dài
# caudai chrF 43,0 (Haiku 48,2) và beam search KHÔNG cứu được (+0,4) -> lỗi model
# chứ không phải decode. Corpus chỉ 7,6% câu >=80 ký tự.
LONG_STRUCTS = [
    "câu có mệnh đề định ngữ dài bổ nghĩa cho danh từ (修飾節) + kết luận ở cuối",
    "câu điều kiện + nghịch ý: 〜すれば〜が、実際には〜",
    "câu có 〜ものの / 〜にもかかわらず nối 2 ý trái ngược",
    "câu nêu lý do dài rồi mới ra kết luận (〜ため、〜ことから、〜以上)",
    "câu liệt kê 3-4 hạng mục rồi tổng kết",
    "câu có trích dẫn lời người khác lồng trong câu (〜と説明されていたが)",
    "câu thời gian nhiều tầng (〜してから〜するまでの間に、〜)",
    "câu có chủ ngữ ẩn đổi giữa các mệnh đề (zero-pronoun nhiều tầng)",
    "câu so sánh phức (〜に比べて〜のほうが〜だけに)",
    "câu email công việc dài, nhiều điều kiện và đề nghị",
    "câu tin tức/báo cáo dài có số liệu và nguyên nhân-kết quả",
    "câu kể chuyện dài có cảm xúc và diễn biến thời gian",
]
LONG_SCENES = [
    "báo cáo công việc", "tin tức thời sự", "email khách hàng", "hồi ký/kể chuyện",
    "bài phân tích kinh tế", "biên bản họp", "hướng dẫn kỹ thuật",
    "bài blog cảm nghĩ", "thư xin lỗi khách hàng", "đánh giá sản phẩm dài",
]

# ------------------------------------------------- seed: hội thoại & họp
# hop 3,6 · hoithoai 3,3 — chưa từng bơm riêng. Không phải yếu nhất nhưng còn
# cách Haiku (3,9 / 3,6) và là domain người dùng gặp nhiều nhất.
CONV_SCENES = [
    "họp dự án: chốt hạng mục, phân công, deadline",
    "họp review tiến độ: báo cáo chậm trễ, nêu rủi ro",
    "brainstorm ý tưởng, phản biện lịch sự",
    "họp với khách hàng: nghe yêu cầu, xác nhận phạm vi",
    "họp 1:1 với cấp trên: xin ý kiến, phản hồi",
    "trao đổi nhanh ở chỗ ngồi (không trang trọng)",
    "chat nhóm công việc: hỏi han, nhắc việc, xác nhận",
    "hội thoại đời thường: rủ ăn trưa, hỏi cuối tuần",
    "hội thoại gia đình: phân công việc nhà, kế hoạch",
    "hội thoại bạn bè: kể chuyện công ty, than thở",
    "hội thoại với hàng xóm, người quen sơ",
    "hội thoại mua bán, hỏi đường, đặt lịch hẹn",
    "phỏng vấn/onboarding nhân viên mới",
    "xử lý bất đồng ý kiến trong nhóm mà vẫn giữ hoà khí",
]
CONV_ACTS = [
    "đề nghị và phản hồi đề nghị", "từ chối khéo", "xin lỗi và giải thích",
    "nhờ vả rồi cảm ơn", "xác nhận lại thông tin cho chắc",
    "ngắt lời lịch sự để bổ sung ý", "chốt kết luận và giao việc",
    "hỏi lại khi chưa hiểu", "đưa ý kiến trái chiều", "than phiền nhẹ rồi tự trấn an",
]

# ------------------------------------------------ seed: zero-pronoun
# zeropronoun 3,3. Tiếng Nhật lược chủ ngữ; dịch sai = đảo người thực hiện.
ZERO_PATTERNS = [
    "câu lược chủ ngữ mà người làm là NGƯỜI NÓI (phải dịch ra 'tôi/em/mình')",
    "câu lược chủ ngữ mà người làm là NGƯỜI NGHE (phải dịch ra 'anh/chị/cậu')",
    "câu lược chủ ngữ mà người làm là NGƯỜI THỨ BA đã nhắc trước đó",
    "câu có 〜てあげる/〜てくれる/〜てもらう — hướng cho-nhận quyết định ai làm cho ai",
    "câu có 〜ておく/〜てしまう + chủ ngữ ẩn",
    "câu bị động lược tác nhân (〜られる) — phải suy ai chịu tác động",
    "câu sai khiến 〜させる lược cả người sai và người bị sai",
    "chuỗi 2-3 câu liên tiếp đổi chủ thể giữa chừng mà không nhắc lại tên",
    "câu hỏi lược chủ ngữ (もう食べた?、行く?) — phải suy hỏi ai",
    "câu trong email/chat công việc lược chủ ngữ theo thói quen",
    "câu có kính ngữ nên chủ thể suy ra TỪ DẠNG động từ (いらっしゃる vs 参る)",
    "câu kể chuyện lược chủ ngữ nhưng đổi người giữa các mệnh đề",
]

# ------------------------------------------- seed: thuật ngữ CHUYÊN NGÀNH
# Vì sao cần dù vòng 2 đã có mode `katakana`: mode đó toàn katakana ĐỜI THƯỜNG
# (マンション, サラリーマン, コンセント). Bench câu thật (TED) cho thấy model gãy ở
# katakana KHOA HỌC và tên riêng phiên âm — đo được v3 chỉ 2,50 ở nhóm này:
#   チック→"Tsim Shake" · 中央海嶺→"đường biển chính" · ゼロ・サム・ゲーム→"game không có Sam"
#   長掌筋→"cơ bắp tay" · フラーレン→lộn cấu trúc · オイラー(Euler)→"đã đọc trên biểu đồ"
# Lỗi KHÔNG phải ngữ pháp mà là ĐOÁN BỪA khi gặp từ lạ.
SCI_FIELDS = [
    "y học lâm sàng (カルテ、オペ、既往歴、予後、寛解、対症療法)",
    "giải phẫu người (長掌筋、前頭葉、胸骨、腱、靭帯、脊髄)",
    "thần kinh học & tâm thần (チック、前駆衝動、トゥレット症候群、双極性障害、幻肢、認知行動療法)",
    "sinh học phân tử & di truyền (塩基配列、大腸菌、遺伝子発現、転写、プラスミド)",
    "hóa học & vật liệu (高分子化合物、フラーレン、触媒、重合、結晶構造)",
    "vật lý hạt & vũ trụ (素粒子、粒子加速器、重力波、真空管、時空の歪み)",
    "thiên văn (恒星、超新星、赤方偏移、系外惑星、ブラックホール)",
    "địa chất & hải dương (中央海嶺、プレートテクトニクス、堆積層、熱水噴出孔)",
    "khí hậu & môi trường (エコシステム、生物多様性、炭素循環、緩和策、閾値)",
    "toán học (オイラー路、素因数、位相幾何、行列、確率分布)",
    "thống kê & phương pháp nghiên cứu (有意差、標本、交絡因子、対照群、再現性)",
    "khoa học máy tính & AI (アフォーダンス、強化学習、過学習、勾配、推論)",
    "kinh tế học (ゼロ・サム・ゲーム、限界効用、インフレ率、需給曲線、外部性)",
    "tài chính doanh nghiệp (資本金、キャッシュフロー、時価総額、増資、減価償却)",
    "luật & hành chính (憲法上、判例、控訴、行政指導、条例)",
    "xã hội học & nhân học (カースト制、階層移動、フィールドワーク、エスノグラフィー)",
    "chính trị học (ファシズム、ナショナリズム、ポピュリズム、国民投票、三権分立)",
    "y tế công cộng & dịch tễ (疫学、罹患率、集団免疫、コホート研究)",
    "kỹ thuật cơ khí & sản xuất (トルク、公差、疲労破壊、シーケンス制御)",
    "kỹ thuật điện & điện tử (インピーダンス、半導体、整流、実装)",
    "xây dựng & kiến trúc (耐震設計、基礎工事、ゼネコン、荷重)",
    "nông nghiệp & thực phẩm (品種改良、収量、発酵、キャッサバ、残留農薬)",
    "nghệ thuật & lịch sử mỹ thuật (自画像、遠近法、印象派、修復、キュレーター)",
    "ngôn ngữ học (音韻、形態素、統語、コーパス、語用論)",
]
SCI_TYPES = [
    "KATAKANA khoa học (từ ngoại lai chuyên ngành) — phải dịch ĐÚNG thuật ngữ Việt, "
    "KHÔNG phiên âm bừa, KHÔNG đoán theo mặt chữ giống tiếng Anh khác",
    "TÊN RIÊNG nước ngoài phiên âm katakana (nhà khoa học, địa danh, tổ chức) — "
    "phải khôi phục ĐÚNG tên gốc: オイラー=Euler, アインシュタイン=Einstein, "
    "ダーウィン=Darwin, ケイマン諸島=quần đảo Cayman",
    "THUẬT NGỮ KANJI chuyên ngành — dịch đúng khái niệm, không tách chữ đoán nghĩa "
    "(中央海嶺 = sống núi giữa đại dương, KHÔNG phải 'đường biển giữa')",
    "VIẾT TẮT khoa học/kỹ thuật (DNA, RNA, GDP, CPU, MRI, PCR, AI, IoT)",
    "SỐ và ĐƠN VỊ khoa học (2の67乗 = 2 mũ 67, 10万塩基 = 100.000 base, "
    "ppm, μm, 摂氏, 光年) — phải giữ đúng con số và đơn vị",
    "định nghĩa/giải thích một khái niệm chuyên ngành cho người ngoài ngành",
]
SCI_SCENES = [
    "bài giảng TED hoặc thuyết trình khoa học phổ thông",
    "bài báo khoa học tóm tắt cho báo chí",
    "trao đổi giữa hai chuyên gia trong phòng thí nghiệm",
    "tài liệu kỹ thuật/hướng dẫn vận hành",
    "bài viết Wikipedia giải thích khái niệm",
    "phỏng vấn nhà nghiên cứu trên truyền hình",
    "báo cáo phân tích gửi ban lãnh đạo",
    "sách giáo khoa đại học",
]

NEG_MARKER = re.compile(r"ない|ぬ|ず|ません|なく|まい|ざる|どころ|限らな|わけが|はずが")
KATA_RUN = re.compile(r"[ァ-ヶー]{3,}")
ASCII_ABBR = re.compile(r"\b[A-Z]{2,6}\b")

KEIGO_MARKER = re.compile(
    # ま[すしせ] phủ ます/ました/ません; で[すし]ょ phủ でしょう/でしょうか
    r"いただ|くださ|ま[すしせ]|で[すし]ょ|ございま|でござ|いらっしゃ|おっしゃ|なさ|"
    r"伺|拝見|拝聴|申し上げ|存じ|承知|かしこま|頂戴|恐れ入|恐縮|差し支え|賜|"
    r"お手数|申し訳|召し上が|ご覧|差し上げ|おいで|お越し|ご足労|参りま|おりま")

PROMPT_IDIOM = """Bạn là dịch giả Nhật-Việt kỳ cựu, viết data huấn luyện cho model dịch.

Với MỖI quán ngữ dưới đây, viết {k} câu tiếng Nhật TỰ NHIÊN có dùng quán ngữ đó
(được phép chia thể/thời cho hợp câu), trong ngữ cảnh: {scene}.
Sau đó dịch sang tiếng Việt.

Quán ngữ (kèm nghĩa bóng tham chiếu):
{items}

QUY TẮC BẮT BUỘC:
- Bản dịch Việt phải theo NGHĨA BÓNG, tuyệt đối KHÔNG dịch theo mặt chữ
  (油を売る = "la cà lười biếng", KHÔNG phải "bán dầu").
- Ưu tiên dùng thành ngữ/cách nói tương đương của tiếng Việt khi có
  (猫に小判 -> "đàn gảy tai trâu"); nếu không có thì diễn đạt trôi chảy tự nhiên.
- Câu Nhật dài 15-45 ký tự, có ngữ cảnh cụ thể (tên người/việc/nơi), KHÔNG phải
  câu mẫu trống rỗng. Đa dạng thể văn (です/ます, だ/である, khẩu ngữ) theo ngữ cảnh.
- Câu Việt phải đọc như người Việt viết, KHÔNG cứng, KHÔNG dịch từng chữ.
- Mỗi câu dùng quán ngữ theo một tình huống KHÁC nhau.

CHỈ trả về JSON array, không markdown, không giải thích:
[{{"ja":"<câu Nhật>","vi":"<câu Việt>","idiom":"<quán ngữ đã dùng>"}}]"""

PROMPT_KEIGO = """Bạn là dịch giả Nhật-Việt kỳ cựu, viết data huấn luyện cho model dịch.

Viết {k} cặp câu Nhật-Việt về giao tiếp KÍNH NGỮ (敬語) tiếng Nhật.
Tình huống: {scene}
Mẫu kính ngữ phải dùng: {pattern}

QUY TẮC BẮT BUỘC (đây là chỗ model hay dịch sai nhất — phải cực chính xác):
- Giữ ĐÚNG CHỦ THỂ hành động dù tiếng Nhật lược chủ ngữ. 謙譲語 = NGƯỜI NÓI /
  bên mình làm -> tiếng Việt "tôi/chúng tôi xin...", "bên tôi sẽ...", "cho phép
  tôi...". 尊敬語 = ĐỐI PHƯƠNG làm -> "anh/chị/quý vị vui lòng...", "mong quý
  công ty...". KHÔNG được đảo ngược hướng.
  Ví dụ sai điển hình: 検討させていただきます nghĩa là "chúng tôi sẽ xem xét",
  DỊCH SAI thành "mong quý khách xem xét".
- Tiếng Việt phải lịch sự TỰ NHIÊN theo văn phong Việt (dạ/vâng, xin phép, kính
  mong, quý công ty, anh/chị...), KHÔNG bịa kính ngữ máy móc, KHÔNG bê nguyên
  cấu trúc Nhật.
- Câu Nhật dài 15-60 ký tự, ngữ cảnh cụ thể (tên công ty/người/việc), như trích
  từ hội thoại hoặc email thật.
- Đa dạng: câu hỏi, câu đề nghị, câu xin lỗi, câu cảm ơn, câu thông báo.

Trường "actor": ai thực hiện hành động chính — "speaker" (người nói/bên mình),
"listener" (đối phương), "third" (người thứ ba).

CHỈ trả về JSON array, không markdown, không giải thích:
[{{"ja":"<câu Nhật>","vi":"<câu Việt>","actor":"speaker|listener|third","pattern":"<mẫu kính ngữ đã dùng>"}}]"""


PROMPT_SLANG = """Bạn là dịch giả Nhật-Việt kỳ cựu, viết data huấn luyện cho model dịch.

Viết {k} cặp câu Nhật-Việt dùng KHẨU NGỮ/SLANG tiếng Nhật thuộc nhóm: {cat}
Bối cảnh: {scene}

QUY TẮC BẮT BUỘC:
- Câu Nhật phải là khẩu ngữ THẬT người Nhật đang dùng (chat, SNS, nói chuyện),
  KHÔNG phải văn viết trang trọng. Được dùng 笑/w/絵文字 nếu tự nhiên.
- Bản dịch Việt phải dùng khẩu ngữ VIỆT tương đương cùng sắc thái (vd めっちゃ美味い
  -> "ngon dã man"; それな -> "đúng thế thật"; ワンチャンある -> "vẫn có khả năng đấy").
  KHÔNG dịch sang tiếng Việt trang trọng — sai sắc thái là sai dịch.
- KHÔNG bịa slang tiếng Việt lạ; dùng cách nói giới trẻ Việt thật sự nói.
- Câu Nhật 8-40 ký tự, có ngữ cảnh cụ thể. Đa dạng: khen, chê, than, rủ rê, đùa.

CHỈ trả về JSON array, không markdown:
[{{"ja":"<câu Nhật>","vi":"<câu Việt>","slang":"<từ slang đã dùng>"}}]"""

PROMPT_NEG = """Bạn là dịch giả Nhật-Việt kỳ cựu, viết data huấn luyện cho model dịch.

Viết {k} cặp câu Nhật-Việt về mẫu PHỦ ĐỊNH/POLARITY tiếng Nhật: {pattern}
Bối cảnh: {scene}

QUY TẮC BẮT BUỘC (đây là lỗi nghiêm trọng nhất của model — phải cực chính xác):
- Giữ ĐÚNG CHIỀU khẳng định/phủ định. Model hay dịch NGƯỢC: câu nhờ vả thành câu
  từ chối, "không phải là không thích" thành "không thích".
- Với 二重否定 (phủ định hai lần) phải ra nghĩa KHẲNG ĐỊNH có sắc thái dè dặt
  (〜なくはない -> "cũng không phải là không...", "kể ra thì cũng có").
- 〜わけではない = phủ định MỘT PHẦN ("không phải là...") ≠ 〜ない (phủ định hoàn toàn).
- 〜てもらえませんか là câu NHỜ VẢ lịch sự, KHÔNG phải câu phủ định.
- Nếu pattern là CẶP ĐỐI CHIẾU: viết theo cặp liền nhau, cùng chủ đề, chỉ khác
  mẫu ngữ pháp, để thấy rõ nghĩa đảo ngược.
- Câu Nhật 15-50 ký tự, ngữ cảnh cụ thể.

Trường "polarity": "affirm" (ý cuối là khẳng định), "negate" (ý cuối là phủ định),
"partial" (phủ định một phần), "request" (câu nhờ vả dạng phủ định).

CHỈ trả về JSON array, không markdown:
[{{"ja":"<câu Nhật>","vi":"<câu Việt>","polarity":"affirm|negate|partial|request","pattern":"<mẫu>"}}]"""

PROMPT_KATA = """Bạn là dịch giả Nhật-Việt kỳ cựu, viết data huấn luyện cho model dịch.

Viết {k} cặp câu Nhật-Việt chứa TỪ NGOẠI LAI/KATAKANA thuộc nhóm: {cat}
Bối cảnh: {scene}

QUY TẮC BẮT BUỘC:
- Dịch ĐÚNG NGHĨA từ katakana, KHÔNG phiên âm bừa, KHÔNG đoán theo mặt chữ giống
  tiếng Anh. Lỗi thật cần tránh: リハーサル (buổi tổng duyệt) bị dịch thành "real";
  TBD (chưa quyết định) bị dịch thành "TMD"; リハビリ là "phục hồi chức năng"
  chứ không phải "rehearsal".
- Với wasei-eigo (từ Anh kiểu Nhật) phải dịch theo nghĩa NHẬT: マンション = "căn hộ
  chung cư" (không phải "biệt thự"); コンセント = "ổ điện"; ワンオペ = "làm một mình".
- Viết tắt tiếng Anh (KPI, MTG, ASAP...) giữ nguyên hoặc dịch nghĩa tùy thói quen
  người Việt trong ngành, KHÔNG bịa chữ viết tắt Việt.
- Câu Nhật 15-50 ký tự, ngữ cảnh công việc/đời sống cụ thể, mỗi câu 1-3 từ katakana.

CHỈ trả về JSON array, không markdown:
[{{"ja":"<câu Nhật>","vi":"<câu Việt>","term":"<từ katakana chính>"}}]"""

PROMPT_LONG = """Bạn là dịch giả Nhật-Việt kỳ cựu, viết data huấn luyện cho model dịch.

Viết {k} cặp câu Nhật-Việt là CÂU DÀI PHỨC HỢP.
Cấu trúc yêu cầu: {struct}
Bối cảnh: {scene}

QUY TẮC BẮT BUỘC:
- Câu Nhật DÀI 80-150 ký tự, có NHIỀU MỆNH ĐỀ lồng nhau, KHÔNG phải hai câu ngắn
  ghép bằng dấu chấm.
- Bản dịch Việt phải giữ ĐỦ mọi thông tin và ĐÚNG quan hệ logic giữa các mệnh đề
  (nguyên nhân, nhượng bộ, điều kiện, thời gian). Đây là chỗ model hay sụp cấu
  trúc: bỏ mệnh đề, đảo quan hệ, hoặc chèn nội dung không có trong câu gốc.
- Tiếng Việt được TÁCH thành 2-3 câu nếu tự nhiên hơn — nhưng KHÔNG được mất ý.
- Chủ ngữ ẩn phải giải đúng; nếu câu đổi chủ thể giữa các mệnh đề thì bản dịch
  phải nói rõ ai làm gì.
- Nội dung cụ thể, có tên người/công ty/số liệu, như trích từ văn bản thật.

CHỈ trả về JSON array, không markdown:
[{{"ja":"<câu Nhật dài>","vi":"<câu Việt>"}}]"""


PROMPT_CONV = """Bạn là dịch giả Nhật-Việt kỳ cựu, viết data huấn luyện cho model dịch.

Viết {k} cặp câu Nhật-Việt là HỘI THOẠI THẬT.
Bối cảnh: {scene}
Hành vi giao tiếp: {act}

QUY TẮC BẮT BUỘC:
- Viết như lời NÓI thật, không phải văn viết. Có thể có 〜ね、〜よ、〜けど、〜んです.
- Chọn ĐÚNG ngôi xưng tiếng Việt theo quan hệ và mức lịch sự của câu Nhật:
  cấp trên-cấp dưới (anh/chị - em), đồng nghiệp ngang hàng (mình/cậu), bạn thân
  (tao/mày hoặc tớ/cậu), với khách (chúng tôi/quý khách). Sai ngôi xưng = sai dịch.
- Giữ đúng mức trang trọng: だ体 → suồng sã; です・ます → lịch sự vừa; 敬語 → trang trọng.
- Câu Nhật 10-45 ký tự, nội dung cụ thể (có tên việc/người/thời gian).
- Đa dạng: câu hỏi, câu đáp, câu nhờ, câu từ chối, câu chốt việc.

CHỈ trả về JSON array, không markdown:
[{{"ja":"<câu Nhật>","vi":"<câu Việt>","reg":"casual|polite|keigo"}}]"""

PROMPT_ZERO = """Bạn là dịch giả Nhật-Việt kỳ cựu, viết data huấn luyện cho model dịch.

Viết {k} cặp câu Nhật-Việt LƯỢC CHỦ NGỮ (zero-pronoun) — dạng: {pattern}
Bối cảnh: {scene}

QUY TẮC BẮT BUỘC (đây là lỗi model hay mắc — phải cực chính xác):
- Câu Nhật PHẢI lược chủ ngữ như người Nhật thật nói (KHÔNG viết 私は/あなたは).
- Bản dịch tiếng Việt PHẢI NÊU RÕ ai làm gì, vì tiếng Việt không lược được như vậy.
  Suy chủ thể từ: dạng kính ngữ, hướng cho-nhận (〜てくれる = người khác làm cho tôi;
  〜てあげる = tôi làm cho người khác), ngữ cảnh câu trước.
- Lỗi điển hình cần tránh: đảo người thực hiện (câu "tôi đã đưa cho anh" bị dịch
  thành "anh đã đưa cho tôi").
- Nếu là chuỗi câu đổi chủ thể giữa chừng, bản dịch phải làm rõ từng người.

Trường "who": ai thực hiện hành động chính — "speaker" | "listener" | "third".

CHỈ trả về JSON array, không markdown:
[{{"ja":"<câu Nhật lược chủ ngữ>","vi":"<câu Việt nêu rõ chủ thể>","who":"speaker|listener|third"}}]"""


PROMPT_SCI = """Bạn là dịch giả Nhật-Việt chuyên ngành, viết data huấn luyện cho model dịch.

Viết {k} cặp câu Nhật-Việt có THUẬT NGỮ CHUYÊN NGÀNH.
Lĩnh vực: {field}
Loại thuật ngữ cần có: {type}
Bối cảnh: {scene}

QUY TẮC BẮT BUỘC (đây là chỗ model đang gãy nặng nhất — phải cực chính xác):
- Dịch ĐÚNG thuật ngữ tiếng Việt đang được dùng trong ngành. Nếu tiếng Việt quen
  dùng nguyên từ tiếng Anh (DNA, CPU, fullerene) thì giữ nguyên, KHÔNG phiên âm.
- TUYỆT ĐỐI KHÔNG đoán bừa khi từ nghe lạ. Lỗi thật cần tránh:
  チック (tic, máy giật cơ) bị dịch thành "Tsim Shake";
  中央海嶺 (sống núi giữa đại dương) thành "đường biển chính";
  ゼロ・サム・ゲーム (trò chơi có tổng bằng không) thành "game không có Sam";
  長掌筋 (cơ gan tay dài) thành "cơ bắp tay";
  オイラー (Euler) bị hiểu thành động từ "đọc".
- Tên riêng phiên âm katakana phải khôi phục về TÊN GỐC (アインシュタイン = Einstein).
- Số và đơn vị phải giữ NGUYÊN GIÁ TRỊ (2の67乗 = 2 mũ 67; 10万 = 100.000).
- Câu Nhật 20-90 ký tự, là câu THẬT trong ngữ cảnh trên, không phải câu mẫu trống.
- Câu Việt phải đọc trôi chảy như người trong ngành viết, KHÔNG dịch cứng.

Trường "term": thuật ngữ chính trong câu. Trường "vi_term": bản dịch của thuật ngữ đó.

CHỈ trả về JSON array, không markdown:
[{{"ja":"<câu Nhật>","vi":"<câu Việt>","term":"<thuật ngữ Nhật>","vi_term":"<thuật ngữ Việt>"}}]"""


def load_keys():
    keys = []
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*gemini_key_(\d+)\s*=\s*(\S+)", line)
            if m:
                keys.append(m.group(2))
    if not keys and os.environ.get("GEMINI_API_KEY"):
        keys = [os.environ["GEMINI_API_KEY"]]
    if not keys:
        sys.exit("Không tìm thấy gemini_key_* trong .env")
    return keys


def load_idioms():
    """Gộp glosses: bản _reviewed ghi đè nghĩa vi của bản gốc."""
    items, order = {}, []
    for f in (GLOSS, GLOSS_REVIEWED):
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            ja, vi = (o.get("ja") or "").strip(), (o.get("vi") or "").strip()
            if not ja or not vi:
                continue
            if ja not in items:
                order.append(ja)
            items[ja] = vi
    if not items:
        sys.exit(f"Không có seed idiom — chạy scripts/gen_idiom_harvest.py trước ({GLOSS})")
    return [(ja, items[ja]) for ja in order]


def idiom_stem(idiom):
    """Gốc idiom để kiểm tra xuất hiện trong câu đã chia thể.

    Cắt đuôi động từ/tính từ cuối (売る -> 売, 高い -> 高) để khớp cả 売っていた,
    高くない... Nếu cắt xong quá ngắn thì giữ nguyên.
    """
    s = re.sub(r"[〜~\s]", "", idiom)
    if len(s) > 2 and re.search(r"[るうくすつぬぶむぐいだたずじえ]$", s):
        s = s[:-1]
    return s if len(s) >= 2 else re.sub(r"[〜~\s]", "", idiom)


def _lcs_len(a, b):
    """Độ dài chuỗi con LIÊN TIẾP dài nhất chung của a và b (a, b đều ngắn)."""
    best, prev = 0, [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def idiom_in(idiom, ja):
    """Quán ngữ có thật sự xuất hiện trong câu ja?

    3 tầng, nới dần: (1) stem khớp trực tiếp; (2) với tục ngữ 2 vế (…、…) chỉ cần
    một vế khớp — người Nhật hay nói nửa đầu; (3) khớp chuỗi con liên tiếp dài
    (idiom bị chia thể/chèn trợ từ) — cần >=5 ký tự và >=50% độ dài idiom.
    """
    core = re.sub(r"[〜~\s]", "", idiom)
    if not core:
        return False
    if idiom_stem(idiom) in ja:
        return True
    for part in re.split(r"[、，,。]", core):
        if len(part) >= 4 and idiom_stem(part) in ja:
            return True
    m = _lcs_len(core, ja)
    return m >= 5 and m >= 0.5 * len(core)


# Rác quan sát thật: model đôi khi nhét ghi chú tiếng Việt vào cuối câu ja
# ("... Quán ngữ đã dùng: 蝉時雨.") — 0,05% mẫu. Dấu tiếng Việt trong câu Nhật
# LUÔN là rác, còn Latin trơn thì hợp lệ (AIツール, SNS, LINE, ABC商事).
VI_TONE_STRICT = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)
# Lookaround phải chặn CẢ CHỮ SỐ: nếu không, "CO2" khớp "co", "LA3" khớp "la"…
# -> câu khoa học hợp lệ (CO2, PM2.5) bị loại oan. Đã gặp thật ở mode sciterm.
VI_WORD_IN_JA = re.compile(
    r"(?<![A-Za-z0-9])(cua|la|va|co|khong|nguoi|duoc|mot|trong|quan ngu|thanh ngu)"
    r"(?![A-Za-z0-9])", re.I)


def qc(mode, ja, vi, rec):
    """Trả về lý do rớt, hoặc None nếu đạt."""
    why = check(ja, vi)
    if why:
        return why
    n = len(ja)
    if n < 10:
        return "ja_qua_ngan"
    if n > 200:
        return "ja_qua_dai"
    if VI_TONE_STRICT.search(ja) or VI_WORD_IN_JA.search(ja):
        return "ja_lan_tieng_viet"
    if mode == "idiom":
        idiom = (rec.get("idiom") or "").strip()
        if not idiom:
            return "thieu_idiom_field"
        if not idiom_in(idiom, ja):
            return "idiom_khong_xuat_hien"
    elif mode == "keigo":
        if not KEIGO_MARKER.search(ja):
            return "khong_co_marker_keigo"
        if (rec.get("actor") or "") not in ("speaker", "listener", "third"):
            return "actor_khong_hop_le"
    elif mode == "negation":
        if not NEG_MARKER.search(ja):
            return "khong_co_marker_phu_dinh"
        if (rec.get("polarity") or "") not in ("affirm", "negate", "partial", "request"):
            return "polarity_khong_hop_le"
    elif mode == "katakana":
        # cần katakana thật (>=3 ký tự liền) HOẶC viết tắt ASCII in hoa
        if not (KATA_RUN.search(ja) or ASCII_ABBR.search(ja)):
            return "khong_co_katakana"
    elif mode == "long":
        if n < 60:
            return "chua_du_dai"
    elif mode == "sciterm":
        term = (rec.get("term") or "").strip()
        if not term:
            return "thieu_term"
        if not (rec.get("vi_term") or "").strip():
            return "thieu_vi_term"
        # Chấp nhận nếu term có trong câu HOẶC câu tự nó mang dấu hiệu chuyên ngành.
        # (Kiểm tra term-in-ja đơn thuần loại oan ~15%: model hay ghi term ở dạng gốc
        #  khác dạng chia trong câu, hoặc ghi khái niệm thay vì đúng chuỗi ký tự.)
        core = re.sub(r"[〜~\s（）()・]", "", term)
        if core and core in ja:
            return None
        if KATA_RUN.search(ja) or ASCII_ABBR.search(ja):
            return None
        if re.search(r"[一-鿿]{3,}", ja):      # cụm kanji dài = thuật ngữ hán-việt
            return None
        return "khong_co_dau_hieu_thuat_ngu"
    elif mode == "conv":
        if (rec.get("reg") or "") not in ("casual", "polite", "keigo"):
            return "reg_khong_hop_le"
    elif mode == "zeropron":
        if (rec.get("who") or "") not in ("speaker", "listener", "third"):
            return "who_khong_hop_le"
        # câu Nhật phải THẬT SỰ lược chủ ngữ, không được viết 私は/あなたは
        if re.search(r"(私|わたし|僕|ぼく|俺|おれ|あなた|君)は", ja):
            return "khong_luoc_chu_ngu"
    # slang: không check marker (danh sách slang là mở, chặn cứng sẽ loại oan)
    return None


def parse_array(txt):
    """Bóc JSON array; chịu được ```json fence và đuôi bị cắt."""
    t = re.sub(r"^```[a-z]*\s*|```\s*$", "", txt.strip(), flags=re.I | re.M)
    try:
        o = json.loads(t)
        return o if isinstance(o, list) else []
    except json.JSONDecodeError:
        pass
    i = t.find("[")
    if i < 0:
        return []
    # cắt tới object hoàn chỉnh cuối cùng rồi đóng array
    j = t.rfind("}")
    if j < i:
        return []
    try:
        return json.loads(t[i:j + 1] + "]")
    except json.JSONDecodeError:
        return []


class Gen:
    def __init__(self, args, keys):
        self.a = args
        self.keys = keys
        self.lock = threading.Lock()
        self.seen = set()
        self.kept = 0
        self.rej = {}
        self.calls = 0
        self.errs = 0
        self.t0 = time.time()
        od = Path(args.outdir) if args.outdir else OUTDIR
        od.mkdir(parents=True, exist_ok=True)
        self.outf = od / f"{args.mode}.jsonl"
        self.rejf = od / f"{args.mode}_rejects.jsonl"
        if self.outf.exists():  # resume: nạp câu ja đã có
            for line in self.outf.open(encoding="utf-8"):
                try:
                    self.seen.add(json.loads(line)["ja"])
                except Exception:  # noqa: BLE001
                    pass
            self.kept = len(self.seen)
            print(f"[resume] đã có {self.kept:,} cặp trong {self.outf.name}", flush=True)
        self.fo = self.outf.open("a", encoding="utf-8")
        self.fr = self.rejf.open("a", encoding="utf-8")

    # -------------------------------------------------- tasks
    def tasks(self):
        """Sinh task vô hạn, mỗi task = 1 request."""
        rng = random.Random(1234)
        if self.a.mode == "idiom":
            idioms = load_idioms()
            print(f"[seed] {len(idioms)} quán ngữ", flush=True)
            g = self.a.group
            rnd = 0
            while True:
                order = list(range(len(idioms)))
                rng.shuffle(order)
                for s in range(0, len(order), g):
                    chunk = [idioms[i] for i in order[s:s + g]]
                    if not chunk:
                        continue
                    yield {"items": chunk, "scene": SCENES_IDIOM[(rnd + s // g) % len(SCENES_IDIOM)]}
                rnd += 1
        elif self.a.mode == "keigo":
            # trộn 4 nhóm pattern, CONTRAST được lấy đậm hơn (nhắm lỗi đảo hướng)
            pats = (KEIGO_HUMBLE + KEIGO_RESPECT + KEIGO_CUSHION
                    + KEIGO_CONTRAST * 3)
            print(f"[seed] {len(pats)} pattern x {len(SCENES_KEIGO)} tình huống", flush=True)
            combos = [(p, s) for p in pats for s in SCENES_KEIGO]
            while True:
                rng.shuffle(combos)
                for p, s in combos:
                    yield {"pattern": p, "scene": s}
        elif self.a.mode == "slang":
            print(f"[seed] {len(SLANG_CATS)} nhóm slang x {len(SCENES_IDIOM)} bối cảnh",
                  flush=True)
            combos = [(c, s) for c in SLANG_CATS for s in SCENES_IDIOM]
            while True:
                rng.shuffle(combos)
                for c, s in combos:
                    yield {"cat": c, "scene": s}
        elif self.a.mode == "negation":
            # NEG_REQUEST nhân 3: lượt đầu chỉ ra 3% loại "nhờ vả dạng phủ định"
            # dù đây là loại model dễ dịch ngược nhất.
            pats = NEG_PATTERNS + NEG_REQUEST * 3
            print(f"[seed] {len(pats)} pattern phủ định x {len(SCENES_KEIGO)} bối cảnh",
                  flush=True)
            combos = [(p, s) for p in pats for s in SCENES_KEIGO]
            while True:
                rng.shuffle(combos)
                for p, s in combos:
                    yield {"pattern": p, "scene": s}
        elif self.a.mode == "katakana":
            print(f"[seed] {len(KATA_CATS)} nhóm katakana x {len(SCENES_KEIGO)} bối cảnh",
                  flush=True)
            combos = [(c, s) for c in KATA_CATS for s in SCENES_KEIGO]
            while True:
                rng.shuffle(combos)
                for c, s in combos:
                    yield {"cat": c, "scene": s}
        elif self.a.mode == "conv":
            print(f"[seed] {len(CONV_SCENES)} bối cảnh x {len(CONV_ACTS)} hành vi",
                  flush=True)
            combos = [(s, ac) for s in CONV_SCENES for ac in CONV_ACTS]
            while True:
                rng.shuffle(combos)
                for s, ac in combos:
                    yield {"scene": s, "act": ac}
        elif self.a.mode == "zeropron":
            print(f"[seed] {len(ZERO_PATTERNS)} dạng x {len(CONV_SCENES)} bối cảnh",
                  flush=True)
            combos = [(p, s) for p in ZERO_PATTERNS for s in CONV_SCENES]
            while True:
                rng.shuffle(combos)
                for p, s in combos:
                    yield {"pattern": p, "scene": s}
        elif self.a.mode == "sciterm":
            print(f"[seed] {len(SCI_FIELDS)} lĩnh vực x {len(SCI_TYPES)} loại thuật ngữ"
                  f" x {len(SCI_SCENES)} bối cảnh", flush=True)
            combos = [(f, t, s) for f in SCI_FIELDS for t in SCI_TYPES for s in SCI_SCENES]
            while True:
                rng.shuffle(combos)
                for f, t, sc in combos:
                    yield {"field": f, "type": t, "scene": sc}
        else:  # long
            print(f"[seed] {len(LONG_STRUCTS)} cấu trúc x {len(LONG_SCENES)} bối cảnh",
                  flush=True)
            combos = [(st, s) for st in LONG_STRUCTS for s in LONG_SCENES]
            while True:
                rng.shuffle(combos)
                for st, s in combos:
                    yield {"struct": st, "scene": s}

    def build_prompt(self, t):
        m, k = self.a.mode, self.a.per_req
        if m == "idiom":
            items = "\n".join(f"- {ja} (nghĩa: {vi})" for ja, vi in t["items"])
            return PROMPT_IDIOM.format(k=max(1, self.a.per_item), scene=t["scene"], items=items)
        if m == "keigo":
            return PROMPT_KEIGO.format(k=k, scene=t["scene"], pattern=t["pattern"])
        if m == "slang":
            return PROMPT_SLANG.format(k=k, cat=t["cat"], scene=t["scene"])
        if m == "negation":
            return PROMPT_NEG.format(k=k, pattern=t["pattern"], scene=t["scene"])
        if m == "katakana":
            return PROMPT_KATA.format(k=k, cat=t["cat"], scene=t["scene"])
        if m == "conv":
            return PROMPT_CONV.format(k=k, scene=t["scene"], act=t["act"])
        if m == "zeropron":
            return PROMPT_ZERO.format(k=k, pattern=t["pattern"], scene=t["scene"])
        if m == "sciterm":
            return PROMPT_SCI.format(k=k, field=t["field"], type=t["type"], scene=t["scene"])
        return PROMPT_LONG.format(k=k, struct=t["struct"], scene=t["scene"])

    # -------------------------------------------------- worker
    def worker(self, wid, key, q):
        client = genai.Client(api_key=key)
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=self.a.temp,
            max_output_tokens=self.a.max_tokens)
        backoff = 2.0
        while True:
            try:
                t = q.get_nowait()
            except queue.Empty:
                return
            if self.done():
                return
            try:
                r = client.models.generate_content(
                    model=self.a.model, contents=self.build_prompt(t), config=cfg)
                txt = r.text or ""
                backoff = 2.0
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                with self.lock:
                    self.errs += 1
                if re.search(r"RESOURCE_EXHAUSTED|429|1011|UNAVAILABLE|503|500|INTERNAL", msg):
                    time.sleep(min(backoff, 60))
                    backoff *= 2
                else:
                    print(f"  [w{wid}] LỖI {msg[:110]}", flush=True)
                    time.sleep(2)
                continue
            self.absorb(t, parse_array(txt))

    def absorb(self, t, recs):
        keep, rej = [], []
        for o in recs:
            if not isinstance(o, dict):
                continue
            ja, vi = norm(str(o.get("ja") or "")), norm(str(o.get("vi") or ""))
            why = qc(self.a.mode, ja, vi, o)
            if why:
                rej.append((why, ja, vi))
                continue
            keep.append((ja, vi, o))
        with self.lock:
            self.calls += 1
            n0 = self.kept
            for ja, vi, o in keep:
                if ja in self.seen:
                    self.rej["trung_lap"] = self.rej.get("trung_lap", 0) + 1
                    continue
                self.seen.add(ja)
                out = {"ja": ja, "vi": vi, "src": self.a.mode, "scene": t.get("scene", "")}
                m = self.a.mode
                if m == "idiom":
                    out["idiom"] = (o.get("idiom") or "").strip()
                elif m == "keigo":
                    out["actor"] = o.get("actor")
                    out["pattern"] = (o.get("pattern") or t["pattern"])[:60]
                elif m == "negation":
                    out["polarity"] = o.get("polarity")
                    out["pattern"] = (o.get("pattern") or t["pattern"])[:60]
                elif m == "slang":
                    out["slang"] = (o.get("slang") or "").strip()[:40]
                elif m == "katakana":
                    out["term"] = (o.get("term") or "").strip()[:40]
                elif m == "conv":
                    out["reg"] = o.get("reg")
                    out["act"] = t.get("act", "")[:40]
                elif m == "sciterm":
                    out["term"] = (o.get("term") or "").strip()[:40]
                    out["vi_term"] = (o.get("vi_term") or "").strip()[:60]
                    out["field"] = t.get("field", "")[:30]
                elif m == "zeropron":
                    out["who"] = o.get("who")
                    out["pattern"] = (t.get("pattern") or "")[:60]
                self.fo.write(json.dumps(out, ensure_ascii=False) + "\n")
                self.kept += 1
            for why, ja, vi in rej:
                self.rej[why] = self.rej.get(why, 0) + 1
                self.fr.write(json.dumps({"reject": why, "ja": ja, "vi": vi},
                                         ensure_ascii=False) + "\n")
            if self.calls % 20 == 0 or self.kept - n0 == 0:
                self.fo.flush()
                self.fr.flush()
            if self.calls % 10 == 0:
                el = time.time() - self.t0
                rate = (self.kept - self.start_kept) / max(1e-9, el / 60)
                print(f"  {self.calls} req | {self.kept:,}/{self.a.target:,} cặp "
                      f"| +{rate:.0f}/phút | lỗi {self.errs} | {el/60:.1f}p", flush=True)

    def done(self):
        return self.kept >= self.a.target

    # ---------------------------------------------- backend LIVE (asyncio)
    # Vì sao Live nhanh hơn REST: quota Live tính theo SESSION ĐỒNG THỜI, không
    # phải request/phút — đã chứng minh chịu được 11,88M câu ở pha KD. Đo thực
    # tế: 15 cặp/9,1s/session => 120 session ~ 12.000 cặp/phút (REST flash-lite
    # chỉ ~2.400/phút rồi bị RPM chặn).
    # LƯU Ý: model chỉ nhận response_modalities=["AUDIO"] (TEXT bị 1007), text
    # lấy qua output_audio_transcription — ĐÃ KIỂM CHỨNG transcription giữ NGUYÊN
    # JSON (dấu {, ", ```) và kanji chính xác, tức là text gốc model sinh chứ
    # không phải ASR của audio. Nên sinh câu Nhật qua đường này là an toàn.
    async def live_worker(self, wid, key, q, gate):
        client = genai.Client(api_key=key)
        cfg = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=types.AudioTranscriptionConfig())
        cm = sess = None
        turns = 0
        # So le khoi dong: 120 session mo cung luc -> bao handshake (CONN_GATE là
        # thứ chính chống bão, cái này chỉ rải đều thêm).
        await asyncio.sleep(wid * 0.05)

        async def close():
            nonlocal cm, sess, turns
            if cm is not None:
                try:
                    await cm.__aexit__(None, None, None)
                except BaseException:  # noqa: BLE001
                    pass
            cm = sess = None
            turns = 0

        async def open_():
            """Mở session có CỔNG: chặn bão handshake khi 120 luồng cùng mở."""
            nonlocal cm, sess, turns
            async with gate:
                c = client.aio.live.connect(model=self.a.model, config=cfg)
                s = await c.__aenter__()
            cm, sess, turns = c, s, 0

        while not self.done():
            try:
                t = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            tries = t.get("_try", 0)
            try:
                if sess is None or turns >= self.a.turns_per_session:
                    await close()
                    await open_()
                await sess.send_client_content(turns=types.Content(
                    role="user", parts=[types.Part(text=self.build_prompt(t))]))
                buf = ""
                async for r in sess.receive():
                    sc = r.server_content
                    if not sc:
                        continue
                    ot = getattr(sc, "output_transcription", None)
                    if ot and ot.text:
                        buf += ot.text
                    if sc.turn_complete:
                        break
                turns += 1
                self.absorb(t, parse_array(buf))
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    self.errs += 1
                    n = self.errs
                if n % 50 == 1:
                    print(f"  [live w{wid}] {str(e)[:110]}", flush=True)
                await close()
                # ĐẨY TASK VỀ HÀNG ĐỢI: handshake timeout là chuyện thường khi
                # nhiều session, không requeue là mất trắng request đó.
                if tries + 1 < self.a.max_attempts:
                    t["_try"] = tries + 1
                    q.put_nowait(t)
                await asyncio.sleep(min(1.5 * (tries + 1), 10))

        await close()

    async def run_live(self):
        nreq = self.a.max_requests or int(
            (self.a.target - self.kept) / max(1, self.a.per_req) * 2.5 + 20)
        q = asyncio.Queue()
        gen = self.tasks()
        for _ in range(nreq):
            q.put_nowait(next(gen))
        nw = self.a.workers_per_key * len(self.keys)
        gate = asyncio.Semaphore(self.a.handshake)
        print(f"[run] backend=LIVE model={self.a.model} mode={self.a.mode} "
              f"target={self.a.target:,} | {len(self.keys)} key x "
              f"{self.a.workers_per_key} = {nw} session | {nreq} request xếp hàng\n",
              flush=True)
        await asyncio.gather(*[
            self.live_worker(i, self.keys[i % len(self.keys)], q, gate)
            for i in range(nw)])

    def finish(self):
        self.fo.flush()
        self.fo.close()
        self.fr.flush()
        self.fr.close()
        el = (time.time() - self.t0) / 60
        print(f"\n=== XONG ({self.a.mode}) ===")
        print(f"Cặp đạt   : {self.kept:,} (+{self.kept-self.start_kept:,} phiên này)")
        print(f"Request    : {self.calls} | lỗi API {self.errs} | {el:.1f} phút")
        if self.rej:
            print(f"Rớt QC     : {sum(self.rej.values()):,}")
            for k, v in sorted(self.rej.items(), key=lambda x: -x[1]):
                print(f"  {k:24s} {v:>7,}")
        print(f"Output     : {self.outf}")

    def run(self):
        self.start_kept = self.kept
        if self.done():
            print(f"Đã đủ {self.kept:,} >= target {self.a.target:,}. Không làm gì.")
            return
        if self.a.backend == "live":
            asyncio.run(self.run_live())
            self.finish()
            return
        nreq = self.a.max_requests or int(
            (self.a.target - self.kept) / max(1, self.a.per_req) * 2.5 + 20)
        q = queue.Queue()
        gen = self.tasks()
        for _ in range(nreq):
            q.put(next(gen))
        nw = self.a.workers_per_key * len(self.keys)
        print(f"[run] mode={self.a.mode} target={self.a.target:,} | {len(self.keys)} key "
              f"x {self.a.workers_per_key} = {nw} luồng | {nreq} request đã xếp hàng\n",
              flush=True)
        ths = []
        for i in range(nw):
            th = threading.Thread(target=self.worker, args=(i, self.keys[i % len(self.keys)], q),
                                  daemon=True)
            th.start()
            ths.append(th)
            time.sleep(0.05)
        for th in ths:
            th.join()
        self.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["idiom", "keigo", "slang", "negation",
                                    "katakana", "long", "conv", "zeropron", "sciterm"])
    p.add_argument("--outdir", default=None,
                   help="thư mục ghi output (mặc định data/synthetic/gen_niche). "
                        "Ổ E hay đầy -> dùng /d/Bit-Translate-data/gen_niche")
    p.add_argument("--backend", choices=["live", "rest"], default="live",
                   help="live = Live API (nhanh, quota theo session — MẶC ĐỊNH); "
                        "rest = generate_content (bị RPM/RPD chặn)")
    p.add_argument("--turns-per-session", type=int, default=8,
                   help="live: số request trên 1 session trước khi mở lại (giữ context nhỏ)")
    p.add_argument("--handshake", type=int, default=12,
                   help="live: số handshake đồng thời tối đa (chống bão bắt tay)")
    p.add_argument("--max-attempts", type=int, default=4,
                   help="live: số lần thử lại 1 request khi session lỗi")
    p.add_argument("--target", type=int, default=30000, help="tổng số cặp muốn có (kể cả đã có)")
    # Mặc định theo backend (đặt None để biết user có tự chỉnh hay không):
    #   live → gemini-3.1-flash-live-preview (đã dùng cho KD 11,88M câu)
    #   rest → gemini-flash-lite-latest = THẦY CHÍNH đã audit (PROVIDERS.md, acc 4.90).
    # ĐỪNG dùng gemini-flash-latest cho REST: alias tới gemini-3.6-flash, free tier
    # chỉ 20 request/NGÀY (GenerateRequestsPerDayPerProjectPerModel-FreeTier).
    p.add_argument("--model", default=None)
    p.add_argument("--workers-per-key", type=int, default=None,
                   help="live: mặc định 12 (=120 session như KD); rest: mặc định 2")
    p.add_argument("--group", type=int, default=None, help="idiom: số quán ngữ mỗi request")
    p.add_argument("--per-item", type=int, default=5, help="idiom: số câu mỗi quán ngữ")
    p.add_argument("--per-req", type=int, default=None, help="keigo: số cặp mỗi request")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--max-requests", type=int, default=0, help="0 = tự tính theo target")
    a = p.parse_args()
    live = a.backend == "live"
    if a.model is None:
        a.model = "gemini-3.1-flash-live-preview" if live else "gemini-flash-lite-latest"
    if a.workers_per_key is None:
        a.workers_per_key = 12 if live else 2
    # Live: request nhỏ hơn (15 cặp ~9s/session) để không giữ session quá lâu.
    if a.group is None:
        a.group = 3 if live else 5
    if a.per_req is None:
        a.per_req = 15 if live else 25
    if a.mode == "idiom":
        a.per_req = a.group * a.per_item
    Gen(a, load_keys()).run()


if __name__ == "__main__":
    main()
