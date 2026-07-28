#!/usr/bin/env python3
"""Sinh DANH SÁCH THUẬT NGỮ ĐỘC LẬP cho từng miền, để đo độ phủ corpus KHÔNG dùng regex.

VÌ SAO PHẢI LÀM LẠI: bản đo cũ (`domain_map.py`) đếm câu bằng regex từ khoá do tôi tự
viết -> đo tay thấy precision 60-100% tuỳ miền, recall còn tệ hơn (>20/35 câu bị bỏ
sót thực ra CÓ thuộc miền). Số đếm đó không đủ tin để quyết định rót nguồn lực.

CÁCH MỚI, tránh đúng hai lỗi đó:
  - Danh sách thuật ngữ do Gemini sinh (độc lập với người viết bộ lọc) -> hết thiên
    lệch "miền nào tôi nghĩ ra nhiều từ khoá hơn thì trông đầy đặn hơn".
  - Đếm SỐ LẦN XUẤT HIỆN của từng thuật ngữ bằng khớp chuỗi chính xác (Aho-Corasick),
    KHÔNG phân loại câu -> không còn khái niệm precision/recall của bộ phân loại.
  - Đo đúng thứ model cần: một thuật ngữ phải gặp >=60 lần thì model mới dịch đúng
    (ngưỡng đo được ở vòng 4: 78 lần thì được vá, <=5 lần thì bịa).

Chia 3 TẦNG để thấy ĐỘ SÂU phủ, không chỉ bề rộng: corpus có thể phủ 100% từ cơ bản
của một miền mà 0% từ chuyên sâu — lúc đó model dịch trôi câu phổ thông nhưng vỡ ngay
khi gặp văn bản thật của ngành.

  python scripts/gen_domain_terms.py            # -> D:/Bit-Translate-data/domain_terms.json
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).parent.parent
OUT = Path("D:/Bit-Translate-data/domain_terms.json")

# 50 miền chủ đề. Tiêu chí TÁCH: thuật ngữ phải RỜI NHAU. "Y tế lâm sàng" và "dược -
# sinh học" tách vì từ vựng khác hẳn; "nông nghiệp" và "thuỷ sản - lâm nghiệp" cũng vậy.
#
# ĐÃ BỎ khỏi trục chủ đề (so với bản 24 miền cũ): "sinh hoạt thường ngày" — đo lại thấy
# nó KHÔNG phải miền thuật ngữ mà là THỂ LOẠI/giọng văn (nhật ký, chuyện vặt), từ vựng
# của nó chính là từ vựng lõi xuất hiện ở mọi nơi. Đếm bằng từ khoá cho ra 70k rồi
# 10,5k tuỳ cách viết regex — vô nghĩa. Chuyển sang trục B (thể văn).
DOMAINS = {
    "y_te_lam_sang": "y tế lâm sàng: bệnh lý, triệu chứng, chẩn đoán, phẫu thuật, chuyên khoa",
    "duoc_sinh_hoc": "dược phẩm và công nghệ sinh học: thuốc, hoạt chất, thử nghiệm lâm sàng, phân tử",
    "dieu_duong_cham_soc": "điều dưỡng và chăm sóc người cao tuổi: kaigo, chăm sóc tại nhà, phân cấp chăm sóc",
    "y_te_cong_cong": "y tế công cộng và dịch tễ học: dịch bệnh, tiêm chủng, kiểm dịch, thống kê sức khoẻ",
    "phap_luat_tu_phap": "pháp luật và tư pháp: tố tụng, hợp đồng, điều luật, toà án",
    "hanh_chinh_chinh_sach": "hành chính công và chính sách: thủ tục hành chính, ngân sách địa phương, quy hoạch",
    "chinh_tri_ngoai_giao": "chính trị và ngoại giao: quốc hội, bầu cử, hiệp định, quan hệ quốc tế",
    "kinh_te_vi_mo": "kinh tế vĩ mô: GDP, lạm phát, chính sách tiền tệ, chu kỳ kinh tế",
    "tai_chinh_dau_tu": "tài chính và đầu tư: chứng khoán, trái phiếu, quỹ, M&A, phái sinh",
    "ke_toan_thue": "kế toán, thuế và kiểm toán: báo cáo tài chính, khấu hao, quyết toán thuế",
    "bao_hiem": "bảo hiểm: hợp đồng bảo hiểm, bồi thường, phí bảo hiểm, bảo hiểm nhân thọ/phi nhân thọ",
    "cntt_phan_mem": "CNTT và phần mềm: lập trình, cơ sở dữ liệu, cloud, bảo mật, AI",
    "ban_dan_phan_cung": "bán dẫn và phần cứng máy tính: chip, wafer, bo mạch, linh kiện điện tử",
    "vien_thong_mang": "viễn thông và mạng: 5G, cáp quang, giao thức, nhà mạng, băng thông",
    "co_khi_che_tao": "cơ khí và chế tạo: gia công, dung sai, khuôn, dây chuyền, kiểm soát chất lượng",
    "oto_xe_co": "ô tô và xe cộ: động cơ, hộp số, xe điện, phụ tùng, đăng kiểm",
    "xay_dung_kien_truc": "xây dựng và kiến trúc: kết cấu, thi công, vật liệu xây dựng, chống động đất",
    "bat_dong_san": "bất động sản: mua bán nhà đất, cho thuê, môi giới, định giá, chung cư",
    "dien_nang_luong": "điện và năng lượng: phát điện, lưới điện, hạt nhân, tái tạo, pin",
    "hoa_chat_vat_lieu": "hoá chất và vật liệu: polymer, hợp kim, xúc tác, chất bán dẫn hữu cơ",
    "moi_truong_khi_hau": "môi trường và khí hậu: phát thải, ô nhiễm, đa dạng sinh học, tái chế",
    "nong_nghiep": "nông nghiệp: trồng trọt, chăn nuôi, phân bón, giống cây, nông cơ",
    "thuy_san_lam_nghiep": "thuỷ sản và lâm nghiệp: đánh bắt, nuôi trồng thuỷ sản, rừng, gỗ, ngư cụ",
    "thuc_pham_che_bien": "thực phẩm và chế biến: an toàn thực phẩm, phụ gia, bảo quản, dây chuyền thực phẩm",
    "am_thuc_nha_hang": "ẩm thực và nhà hàng: món ăn, nguyên liệu nấu, kỹ thuật nấu, kinh doanh ăn uống",
    "ban_le_thuong_mai": "bán lẻ và thương mại: cửa hàng, tồn kho, khuyến mãi, thương mại điện tử, POS",
    "logistics_van_tai": "logistics và vận tải: kho bãi, chuỗi cung ứng, vận chuyển, thông quan",
    "hang_khong_vu_tru": "hàng không và vũ trụ: máy bay, sân bay, vệ tinh, tên lửa đẩy, hàng không dân dụng",
    "duong_sat_hang_hai": "đường sắt và hàng hải: tàu hoả, ga, tàu biển, cảng, tuyến vận tải biển",
    "du_lich_khach_san": "du lịch và khách sạn: lưu trú, tour, điểm tham quan, lữ hành, onsen",
    "giao_duc": "giáo dục: trường học, chương trình học, thi cử, sư phạm, giáo dục đại học",
    "the_thao": "thể thao: các môn thể thao, thi đấu, huấn luyện, giải đấu, vận động viên",
    "giai_tri_dien_anh": "giải trí và điện ảnh: phim, truyền hình, diễn viên, sản xuất, showbiz",
    "anime_manga_game": "anime, manga và game: thể loại, nhân vật, đồng nhân, gacha, thanh niên hâm mộ",
    "am_nhac": "âm nhạc: nhạc cụ, thể loại nhạc, sáng tác, biểu diễn, ngành thu âm",
    "van_hoc_nghe_thuat": "văn học và nghệ thuật: thể loại văn học, thủ pháp, hội hoạ, thư pháp, bảo tàng",
    "thoi_trang_my_pham": "thời trang và mỹ phẩm: quần áo, chất liệu, trang điểm, chăm sóc da, thương hiệu",
    "ton_giao": "tôn giáo: Phật giáo, Thần đạo, Kitô giáo, nghi lễ, cơ sở thờ tự",
    "triet_hoc_tu_tuong": "triết học và tư tưởng: trường phái, khái niệm triết học, đạo đức học, logic",
    "lich_su": "lịch sử: các thời kỳ Nhật Bản và thế giới, nhân vật lịch sử, sự kiện, sử liệu",
    "toan_ly": "toán học và vật lý: đại số, giải tích, cơ học, lượng tử, thống kê toán",
    "hoa_sinh_hoc": "hoá học và sinh học: phản ứng, nguyên tố, tế bào, gen, tiến hoá",
    "thien_van_dia_chat": "thiên văn và khoa học trái đất: thiên thể, quan trắc, địa chất, khí tượng, đại dương",
    "tam_ly_hoc": "tâm lý học: nhận thức, hành vi, trị liệu, rối loạn tâm lý, phát triển",
    "xa_hoi_nhan_khau": "xã hội học và nhân khẩu: dân số, già hoá, đô thị hoá, giai tầng, khảo sát xã hội",
    "quan_su_quoc_phong": "quân sự và quốc phòng: lực lượng, khí tài, chiến lược, an ninh quốc gia",
    "canh_sat_hinh_su": "cảnh sát và hình sự: điều tra, tội phạm, bắt giữ, hiện trường, phòng chống tội phạm",
    "thien_tai_phong_chong": "thiên tai và phòng chống: động đất, sóng thần, bão, sơ tán, cứu hộ",
    "phuc_loi_an_sinh": "phúc lợi và an sinh xã hội: lương hưu, trợ cấp, người khuyết tật, hỗ trợ sinh hoạt",
    "nuoi_day_con": "nuôi dạy con: mang thai, sinh nở, nhà trẻ, phát triển trẻ, giáo dục gia đình",
    "thu_y_thu_cung": "thú y và thú cưng: chó mèo, bệnh vật nuôi, thức ăn cho thú, phòng khám thú y",
    "nhan_su_lao_dong": "nhân sự và lao động: tuyển dụng, lương thưởng, luật lao động, đánh giá, nghỉ việc",
    "marketing_quang_cao": "marketing và quảng cáo: thương hiệu, chiến dịch, SEO, KPI, nghiên cứu thị trường",
    "quan_tri_khoi_nghiep": "quản trị và khởi nghiệp: chiến lược, gọi vốn, quản lý dự án, tổ chức doanh nghiệp",
}

PROMPT = """Bạn là chuyên gia thuật ngữ tiếng Nhật. Liệt kê thuật ngữ tiếng Nhật ĐẶC TRƯNG cho lĩnh vực sau:

LĨNH VỰC: {desc}

Yêu cầu NGHIÊM NGẶT:
1. Chia đúng 3 tầng, mỗi tầng {k} từ:
   - tier 1 (cơ bản): từ mà BẤT KỲ văn bản nào thuộc lĩnh vực này cũng dùng
   - tier 2 (trung cấp): từ chuyên môn thường gặp trong bài viết/tin tức về lĩnh vực
   - tier 3 (chuyên sâu): thuật ngữ chuyên ngành thật sự, chỉ người trong ngành dùng
2. Mỗi từ phải là từ tiếng Nhật THẬT, viết đúng chính tả (kanji hoặc katakana), dài 2-10 ký tự.
3. TUYỆT ĐỐI KHÔNG dùng từ chung chung có thể xuất hiện ở mọi lĩnh vực khác
   (ví dụ xấu: 問題, 思想, システム, 情報, 内容, 会社, 方法, メニュー, 天気).
   Từ phải ĐẶC TRƯNG — đọc từ đó là đoán ra ngay lĩnh vực.
4. Không lặp từ. Không giải thích. Không thêm chữ nào ngoài JSON.

Trả về DUY NHẤT một JSON array:
[{{"t":"心筋梗塞","tier":1}}, {{"t":"経皮的冠動脈形成術","tier":3}}, ...]"""


def load_keys():
    keys = []
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*gemini_key_(\d+)\s*=\s*(\S+)", line)
        if m:
            keys.append(m.group(2))
    if not keys:
        sys.exit("Không tìm thấy gemini_key_* trong .env")
    return keys


def parse_array(txt):
    t = re.sub(r"^```[a-z]*\s*|```\s*$", "", txt.strip(), flags=re.I | re.M)
    try:
        o = json.loads(t)
        return o if isinstance(o, list) else []
    except json.JSONDecodeError:
        pass
    i, j = t.find("["), t.rfind("}")
    if i < 0 or j < i:
        return []
    try:
        return json.loads(t[i:j + 1] + "]")
    except json.JSONDecodeError:
        return []


OK_TERM = re.compile(r"^[一-龥ぁ-んァ-ヶー々〆]{2,10}$")


async def one(sem, client, model, name, desc, k, res):
    async with sem:
        for attempt in range(3):
            try:
                r = await client.aio.models.generate_content(
                    model=model, contents=PROMPT.format(desc=desc, k=k),
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", temperature=0.4))
                rows = parse_array(r.text or "")
                out, seen = [], set()
                for o in rows:
                    if not isinstance(o, dict):
                        continue
                    t = str(o.get("t", "")).strip()
                    try:
                        tier = int(o.get("tier", 0))
                    except (TypeError, ValueError):
                        continue
                    if tier in (1, 2, 3) and OK_TERM.match(t) and t not in seen:
                        seen.add(t)
                        out.append({"t": t, "tier": tier})
                if len(out) >= k:
                    res[name] = out
                    print(f"  {name:26} {len(out):>3} từ", flush=True)
                    return
            except Exception as e:
                if attempt == 2:
                    print(f"  {name:26} LỖI {type(e).__name__}: {str(e)[:90]}", flush=True)
                await asyncio.sleep(3 + attempt * 5)
        res.setdefault(name, [])


async def main_async(a):
    keys = load_keys()
    print(f"{len(keys)} key | {len(DOMAINS)} miền | {a.per_tier}x3 từ/miền\n")
    res = {}
    sem = asyncio.Semaphore(a.concurrency)
    tasks = []
    for i, (name, desc) in enumerate(DOMAINS.items()):
        client = genai.Client(api_key=keys[i % len(keys)])
        tasks.append(one(sem, client, a.model, name, desc, a.per_tier * 3, res))
    await asyncio.gather(*tasks)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemini-flash-latest")
    p.add_argument("--per-tier", type=int, default=40)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()

    res = asyncio.run(main_async(a))

    # Thuật ngữ xuất hiện ở >2 miền thì KHÔNG đặc trưng -> loại. Đây là chốt chặn cho
    # đúng lỗi đã mắc lần trước (từ khoá nghĩa rộng khớp bừa khắp nơi).
    from collections import Counter
    freq = Counter(o["t"] for v in res.values() for o in v)
    dropped = {t for t, c in freq.items() if c > 2}
    clean = {k: [o for o in v if o["t"] not in dropped] for k, v in res.items()}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(len(v) for v in clean.values())
    empty = [k for k, v in clean.items() if len(v) < 30]
    print(f"\n{n:,} thuật ngữ / {len(clean)} miền  (loại {len(dropped)} từ trùng >2 miền)")
    if empty:
        print(f"⚠ miền thiếu từ (<30): {', '.join(empty)}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
