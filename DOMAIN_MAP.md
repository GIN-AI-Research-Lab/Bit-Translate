# BẢN ĐỒ MIỀN v2 — đo bằng ĐỘ PHỦ THUẬT NGỮ (2026-07-28)

> Trả lời: *"54 miền nào đang thiếu, thiếu bao nhiêu, bù kiểu gì?"*
> **Bản v1 (27/07) dùng regex đếm câu — đã BỊ BÁC, xem §1.** Bản này đếm chính xác
> 16.922 thuật ngữ bằng Aho-Corasick trên 13,1M câu corpus + 11M kho đã tải + 20M mẫu
> kho gốc. Không còn khái niệm precision/recall của bộ phân loại.
>
> Script: `gen_domain_terms.py` → `term_coverage.py` → `probe_cc100_archive.py`
> Dữ liệu: `D:/domain_terms.json`, `eval/term_coverage.json`, `eval/cc100_archive_probe.json`

---

## 1. Review bản v1: cái gì sai, cái gì đứng vững

| Khẳng định ở v1 | Phán quyết |
|---|---|
| "24 miền chủ đề" | **Chia quá thô.** Thực tế 54 miền có thuật ngữ rời nhau. `sinh_hoat_thuong_ngay` không phải miền thuật ngữ — nó là THỂ LOẠI, từ vựng của nó chính là từ vựng lõi ở mọi nơi; đã bỏ khỏi trục A. |
| **"~85% chỗ thiếu đã có sẵn trong kho ổ D, chỉ cần lọc + KD dịch"** | **SAI ở mức quan trọng nhất.** Đúng khi đếm CÂU, sai khi đếm THUẬT NGỮ. Đổ toàn bộ 11M câu kho đã tải vào chỉ nâng độ phủ trung bình 40,8% → 55,5%, và **0/54 miền** đạt 85%. Kho đã tải RỘNG nhưng NÔNG: thêm câu nói về nông nghiệp, không thêm thuật ngữ nông nghiệp mới. |
| "Chỉ 6-7 miền thiếu nguồn" | **Sai.** Theo độ phủ thuật ngữ, **53/54 miền dưới ngưỡng**. Chỉ `cntt_phan_mem` (74%) gần đạt. |
| "Target 750k/450k/220k câu mỗi miền" | **Bỏ.** Đường cong cũ đo độ phủ của chính những từ phổ biến nhất TRONG corpus → vòng tròn luẩn quẩn. Thay bằng target theo % thuật ngữ đạt ≥60 lần, dùng danh sách từ độc lập. |
| Khung 3 trục (chủ đề / thể văn / hiện tượng) | **Đứng vững** — số liệu mới củng cố thêm. |
| Ngưỡng ≥60 lần/thuật ngữ | **Đứng vững** — vẫn là con số vận hành. |
| Hiện tượng C (niên hiệu, tên riêng) gắn với miền, phải rải đều | **Đứng vững** — nhưng tên riêng cần chiến lược KHÁC hẳn, xem §5. |

**Vì sao v1 sai:** đếm "số câu có nhắc tới nông nghiệp" ≠ đo "model có biết từ vựng nông
nghiệp không". Kho có thừa câu loại một, thiếu trầm trọng loại hai.

---

## 2. Con số thực: ba mức dữ liệu

Độ phủ = % thuật ngữ của miền đạt **≥60 lần** trong corpus (ngưỡng đo được ở vòng 4:
78 lần thì model dịch đúng, ≤5 lần thì bịa).

| Mức dữ liệu | Độ phủ TB | Đạt ≥70% | Đạt ≥85% |
|---|---:|---:|---:|
| Corpus v5 hiện tại (13,1M câu) | **40,8%** | 1/54 | 0/54 |
| + toàn bộ kho đã tải (11M câu chưa dùng) | **55,5%** | 6/54 | 0/54 |
| + kho CC-100 GỐC (`ja.txt.xz`, 417M câu) | **77,3%** | **42/54** | **18/54** |

### Ngưỡng cần đạt: ~70-75% (suy ra từ quan hệ đo được)

Ba miền đã dịch thử thật 40 câu/miền và tự chấm — khớp đơn điệu với độ phủ đo mới:

| Miền | Độ phủ | Lỗi thật |
|---|---:|---:|
| cntt_phan_mem | 74% | **2,5%** |
| y_te_lam_sang | 51% | **15%** |
| nong_nghiep | 27% | **25%** |

Nội suy tuyến tính: `lỗi% ≈ 38 − 0,48 × độ phủ%` → **để lỗi ≤5% cần độ phủ ≥70%**.
(Chỉ 3 điểm, và "lỗi" là tôi tự chấm — dùng để định hướng, không phải hằng số vật lý.)

### Phát hiện quan trọng nhất: ĐỘ SÂU, không phải bề rộng

| Tầng | Số từ | Corpus v5 | + kho đã tải | Không hề xuất hiện |
|---|---:|---:|---:|---:|
| tier1 — cơ bản (患者, 農家, 料理) | 5.694 | 73,6% | **85,3%** | 1,0% |
| tier2 — trung cấp (báo chí, bài viết ngành) | 5.930 | 34,2% | **54,4%** | 6,6% |
| tier3 — chuyên sâu (経皮的冠動脈形成術, 陽イオン交換容量) | 5.703 | 10,7% | **22,8%** | 30,3% |

⇒ **Model xử lý được văn bản đời thường về mọi chủ đề, lung lay ở văn báo chí/chuyên
ngành, và vỡ hẳn ở văn bản chuyên môn thật.** Đây chính xác là trải nghiệm "chỗ thì dịch
ổn chỗ thì không" — không phải miền nào tốt miền nào tệ, mà là **TẦNG nào cũng tệ như
nhau ở mọi miền**.

---

## 3. 54 miền — độ phủ theo ba mức

Sắp theo mức cuối (sau khi đào hết kho gốc), thấp nhất trước.

### 🔴 Nhóm khó nhất — kho gốc cũng chỉ đưa lên 54-62%

| Miền | Giờ | +kho tải | +kho gốc | Còn thiếu |
|---|---:|---:|---:|---:|
| hoa_chat_vat_lieu | 35% | 44% | **54%** | 154 |
| kinh_te_vi_mo | 32% | 45% | **55%** | 147 |
| co_khi_che_tao | 34% | 40% | **56%** | 152 |
| thuy_san_lam_nghiep | 25% | 36% | **57%** | 152 |
| bao_hiem | 18% | 29% | **60%** | 140 |
| thien_van_dia_chat | 28% | 41% | **60%** | 139 |
| xa_hoi_nhan_khau | 28% | 44% | **61%** | 130 |
| nong_nghiep | 27% | 40% | **62%** | 129 |

### 🟡 Nhóm giữa — kho gốc đưa lên 65-79%

thuc_pham_che_bien 65% · y_te_cong_cong 66% · ban_dan_phan_cung 67% · vien_thong_mang 69%
· duoc_sinh_hoc 70% · dien_nang_luong 72% · canh_sat_hinh_su 72% · logistics_van_tai 73%
· van_hoc_nghe_thuat 73% · xay_dung_kien_truc 74% · triet_hoc_tu_tuong 74% · hoa_sinh_hoc 74%
· hang_khong_vu_tru 75% · thien_tai_phong_chong 76% · toan_ly 77% · tam_ly_hoc 77%
· marketing_quang_cao 77% · nuoi_day_con 78% · ke_toan_thue 79% · du_lich_khach_san 79%
· quan_su_quoc_phong 79% · quan_tri_khoi_nghiep 79%

### 🟢 Nhóm kho gốc lấp gần đủ — 81-92%

moi_truong_khi_hau 81% · anime_manga_game 81% · dieu_duong_cham_soc 82% · duong_sat_hang_hai 82%
· ban_le_thuong_mai 83% · am_thuc_nha_hang 84% · phap_luat_tu_phap 85% · y_te_lam_sang 85%
· tai_chinh_dau_tu 85% · thu_y_thu_cung 85% · hanh_chinh_chinh_sach 86% · oto_xe_co 86%
· cntt_phan_mem 87% · am_nhac 87% · lich_su 88% · the_thao 89% · nhan_su_lao_dong 89%
· thoi_trang_my_pham 90% · chinh_tri_ngoai_giao 91% · bat_dong_san 91% · giao_duc 91%
· giai_tri_dien_anh 92% · ton_giao 92% · phuc_loi_an_sinh 92%

**Chú ý miền nhảy vọt nhờ kho gốc:** bao_hiem 18→60, lich_su 30→88, ton_giao 38→92,
bat_dong_san 36→91, phuc_loi_an_sinh 48→92. Đây là các miền **dân sự phổ thông** — biên bản
Quốc hội và lát cắt CC-100 đã lọc không có, nhưng web tiếng Nhật nói chung thì đầy.

---

## 3b. ĐÃ ĐÀO XONG KHO GỐC (2026-07-28) — số thật thay ngoại suy

`xz -dc | mine_rare_terms_mp.py --workers 10` — **392,8 triệu câu, 428 giây** (78% CPU).

| | Kết quả |
|---|---|
| Câu chọn được (chứa thuật ngữ dưới ngưỡng) | **216.179** |
| Thuật ngữ được lấp | 3.921/7.882 (**50%**) |
| Vẫn 0 lần trong CẢ 393M câu | **1.024 từ** → bắt buộc SINH |
| Độ phủ TB | 55,5% → **77,8%** |
| Miền đạt ≥70% | 6/54 → **44/54** |
| **Kiểm đồng đều kho** | **lệch chuẩn/TB = 5,6% → ĐỀU**, ngoại suy từ mẫu đầu file là tin được |

Ngoại suy trước đó dự đoán 77,3% — thực tế 77,8%. Khớp.

**Bài học tốc độ (đã đo, đừng lặp lại sai lầm):**
| Cách | Tốc độ | Toàn kho 69GB |
|---|---|---|
| `lzma` Python + `pyahocorasick` | ~4-7 MB/s | ~2,8 giờ |
| `xz -dc` + `pyahocorasick` | 26 MB/s | ~100 phút |
| `xz -dc` + `ahocorasick_rs` (Rust) | 60 MB/s | ~35 phút |
| **`xz -dc` + Rust + 10 tiến trình** | **~160 MB/s** | **7 phút** |

GPU/CUDA KHÔNG giúp: đây là đi bộ trên máy trạng thái (truy cập bộ nhớ bất quy tắc,
rẽ nhánh theo ký tự), không thiếu FLOP. Nút thắt lần lượt là: giải nén → binding
Python của AC → giải mã UTF-8 một luồng. Cả ba đều là CPU/bộ nhớ, không phải tính toán.

---

## 3c. "54 miền đã đủ chưa?" — KẾT HỢP hai cách kiểm

Không tin cách nào một mình:
- **Cách A (LLM)**: hỏi Gemini 3 lần độc lập → danh sách ứng viên. Yếu: có thể bịa miền
  nghe hợp lý nhưng thực tế hiếm.
- **Cách B (đếm)**: đếm thuật ngữ thật trong text thật. Yếu: chỉ đo được cái đã nghĩ ra.
- **Kết hợp**: A sinh ứng viên → B phán quyết bằng số.

**⚠️ Bẫy đã mắc và sửa:** so cả 3 tầng thì ứng viên bị thiệt, vì 166 từ phổ biến nhất
đã bị loại do trùng 54 miền cũ → danh sách còn lại dồn về đuôi hiếm. Phán quyết ban đầu
sai ở 4 miền. **So riêng tier1 (từ cơ bản) mới công bằng** — mốc: tier1 của 54 miền cũ
phủ **74,9%**.

| Miền ứng viên | Phủ tier1 | Tần suất ngoài đời | Phán quyết |
|---|---:|---:|---|
| onsen_sento | 27% | 15 | 🔴 thiếu nặng, nhưng ít gặp |
| **kankonsosai** (hiếu hỉ) | **37%** | 18 | 🔴 **THIẾU THẬT** |
| **co_bac_giai_tri** (pachinko, keiba) | **46%** | 31 | 🔴 **THIẾU THẬT** |
| **van_hoa_truyen_thong** (kabuki, trà đạo) | **50%** | 45 | 🔴 **THIẾU THẬT** |
| **vo_dao** (judo, kendo, sumo) | **57%** | 69 | 🔴 **THIẾU THẬT, hay gặp** |
| **le_hoi_thuong_nien** (tết, obon, matsuri) | **58%** | 49 | 🔴 **THIẾU THẬT** |
| **van_hoa_mang** (tiếng lóng mạng) | **65%** | **164** | 🟡 thiếu vừa, RẤT hay gặp |
| so_thich_ngoai_troi | 73% | **510** | 🟢 đã gần mốc |
| boi_toan_tam_linh | 76% | 95 | 🟢 đạt mốc |
| nhap_canh_luu_tru | **91%** | 188 | 🟢 vượt mốc — đã phủ tốt |

⇒ **54 miền CHƯA đủ, thiếu 6-7 miền — toàn về VĂN HOÁ VÀ ĐỜI SỐNG NHẬT**, không phải
kỹ thuật. Taxonomy ban đầu lệch về hướng chuyên môn/công nghiệp. Cách B độc lập (lấy
cụm kanji phổ biến nhất trong 1,2M dòng web trừ đi 16.922 từ đã có) KHÔNG lộ ra miền kỹ
thuật nào thiếu — phần chuyên môn đã phủ đủ rộng.

**Lưu ý về `nhap_canh_luu_tru`:** tier1 phủ 91% (rất tốt, nhờ biên bản Quốc hội bàn
nhiều về chính sách nhập cư), NHƯNG tier3 vẫn 0 lần cho 出入国在留庁, 資格外許可,
優良監理団体, 認定送出機関 — đúng những từ người Việt ở Nhật gặp trên giấy tờ thật.
Đây là vấn đề TẦNG, không phải vấn đề miền.

---

## 4. Trả lời "bù kiểu gì": kho gốc CC-100 chưa đào là câu trả lời chính

`D:/Bit-Translate-data/raw/ja.txt.xz` = **14,8 GB nén / 69,3 GB giải nén / ~417 triệu câu.**
Đã khai thác: `cc100_pick.txt` 5,2M + `cc100_colloq.txt` 1,7M = **~7M = 1,7%.**
**98,3% kho nằm im trên đĩa, chưa từng được lọc.**

### Chi phí đào (đo trên mẫu 20M câu = 4,8% kho, ngoại suy ×21)

| Mục tiêu | Cần đào |
|---|---|
| Lấp 50% thuật ngữ đang thiếu | ~98M câu (24% kho) |
| Lấp 80% | ~520M câu — **vượt cả kho** |
| Lấp 90-95% | 1.040-1.180M câu — **không tồn tại đủ** |

⇒ Đào hết kho đưa TB lên **77,3%**, còn lại **4.073 thuật ngữ** (24% số đang thiếu) mà
**không lượng web tiếng Nhật nào lấp nổi** — chủ yếu tier3 siêu hiếm (2.949 từ không xuất
hiện lần nào trong 20M câu). Nhóm này chỉ có thể **SINH** bằng Gemini.

### Quy trình đề xuất — chi phí thật

| Bước | Việc | Chi phí |
|---|---|---|
| 1 | Quét tuần tự toàn bộ 417M câu kho gốc, **chọn ra** câu chứa thuật ngữ đang dưới ngưỡng, giới hạn ~60 câu/thuật ngữ | **~5-6 giờ CPU nội bộ, $0** |
| 2 | Ước ~250-400k câu được chọn (tổng thiếu 375.065 lượt ÷ ~1,3 thuật ngữ/câu) | — |
| 3 | KD dịch số câu đó bằng Gemini (73 câu/s đo được) | **~1-1,5 giờ, $0** (quota TPM) |
| 4 | Sinh bổ sung cho ~2.949 từ 0 lần: ~20 câu/từ ×3 oversample | ~59k câu, **vài giờ, $0** |
| 5 | Gộp + oversample ×3 phần hiếm → corpus ~14,3M (+9%) | — |
| 6 | Train vòng 6 | **~$15** (còn $16,5) |

**Điểm mấu chốt:** cách này chỉ thêm **~9% dung lượng corpus** nhưng vá 50-80% lỗ hổng
thuật ngữ — đòn bẩy cao hơn hẳn vòng 5 (thêm 2,45M câu đại trà). Vì chọn câu **theo thuật
ngữ đang thiếu**, không phải theo chủ đề.

⚠️ **Cảnh báo phương pháp:** mẫu 20M lấy từ ĐẦU file, chưa xác minh CC-100 có xáo trộn
đều không. Nếu file sắp theo domain/thời gian thì ngoại suy ×21 sẽ lệch. Rẻ để kiểm:
quét thêm một lát ở giữa file rồi so phân bố trước khi cam kết 5-6 giờ đào.

---

## 5. Tên riêng: KHÔNG phải bài toán thêm data — cần đổi cách nghĩ

Đo trên corpus v5 (13,1M câu), tên riêng là lớp từ **MỞ** nên đếm "bao nhiêu câu có tên"
(cách làm ở v1) là vô nghĩa. Đếm **bao nhiêu tên khác nhau, mỗi tên gặp mấy lần**:

| Loại | Tên riêng biệt | ≥60 lần | ≥5 lần | **Đúng 1 lần** |
|---|---:|---:|---:|---:|
| Người | 14.497 | 343 (2,4%) | 2.344 | **8.890 (61%)** |
| Địa danh | 20.664 | 309 (1,5%) | 3.357 | **12.221 (59%)** |
| Tổ chức | 22.470 | 246 (1,1%) | 2.909 | **13.533 (60%)** |

**~60% tên riêng chỉ xuất hiện ĐÚNG MỘT LẦN.** Model không có cơ hội học → đoán bừa theo
âm Hán-Việt. Đúng bằng chứng đã thấy: 西村→"Tây Thôn", 茨城→"Trim", 興和→"ông Kowa".

*(Bộ regex bắt tên có nhiễu — 内閣総理, 患者 lọt vào. Nhưng HÌNH DẠNG phân bố (60% xuất
hiện một lần) mới là tín hiệu, và nhiễu không đổi được hình dạng đó.)*

### Vì sao ngưỡng ≥60 KHÔNG áp dụng được ở đây

Với lớp từ đóng (thuật ngữ), mục tiêu là **ghi nhớ** → cần mỗi từ ≥60 lần.
Với lớp từ mở (tên riêng), không bao giờ phủ hết được → mục tiêu phải là **học QUY TẮC**
kanji→romaji. Mà quy tắc thì học từ **ĐỘ ĐA DẠNG**, không phải từ tần suất: 10.000 tên
mỗi tên 3 lần dạy quy tắc tốt hơn 500 tên mỗi tên 60 lần, vì các tên dùng chung ký tự
(西村/西川/中村/木村 đều có 村=mura).

**Bằng chứng model ĐÃ học được một phần quy tắc:** v5 dịch đúng 林大臣→"Bộ trưởng Hayashi",
越前市長→"Thị trưởng Echizen" (bench held-out §0-SEPTIES). Nó gãy ở tên hiếm hơn, tức quy
tắc có nhưng chưa đủ chắc.

⇒ **Hành động ngược với trực giác:** với tên riêng, ưu tiên **nhiều tên khác nhau, tần
suất thấp** — tức KHÔNG oversample, KHÔNG lọc bỏ tên hiếm. Và đo bằng thước khác: số cặp
(ký tự kanji → cách đọc) được phủ, không phải số lần mỗi tên xuất hiện.

---

## 6. Mục tiêu "không bao giờ có câu nào không dùng được"

Nói thẳng: **không hệ dịch máy nào đạt được mức tuyệt đối này**, kể cả Google/DeepL —
bench 200 câu TED đo được Google 87%, tức Google cũng hỏng 13%. Nhưng con số có ý nghĩa
hơn là mục tiêu tuyệt đối:

- Hiện tại: độ phủ TB 40,8% → theo quan hệ đo được, lỗi ~18% ở miền trung bình, tệ nhất ~25%
- Sau khi đào kho gốc: TB 77,3% → **lỗi ~1-5% ở 42/54 miền**
- 12 miền còn lại (nhóm 🔴🟡 đầu bảng) sẽ dừng ở 54-70% → lỗi ~5-12%, cần sinh bổ sung

Đó là mức "tốt, dùng được" mà `CLAUDE.md` §3 đã đặt ra từ đầu — và lần này có đường đi
định lượng để tới.

---

## 7. Thứ tự làm

1. **Kiểm mẫu giữa file `ja.txt.xz`** (30 phút) — xác nhận ngoại suy ×21 trước khi cam kết.
2. **Quét toàn kho gốc, chọn câu theo thuật ngữ thiếu** (5-6 giờ CPU, $0) — đây là việc
   có đòn bẩy cao nhất trong toàn bộ kế hoạch.
3. **KD dịch phần chọn** (~1-1,5 giờ, $0).
4. **Sinh cho 2.949 từ không tồn tại trong web** (vài giờ, $0), oversample ×3.
5. **Rải hiện tượng C** (niên hiệu, số, ngày tháng) đều qua mọi miền khi chọn câu ở bước 2 —
   ép tối thiểu 5-10% câu mỗi miền chứa chúng (bài học §0-SEPTIES: hiện tượng C gắn với miền).
6. **Tên riêng: giữ nguyên đuôi dài**, không lọc bỏ, không oversample (§5).
7. **Train vòng 6** (~$15) → bench TRONG từng miền vừa bơm (bài học [[bench-do-sai-mien]]),
   không dùng bench trộn lẫn.

---

## 8. Việc người quyết

- **Đào tới đâu?** 24% kho lấp 50% lỗ hổng; 100% kho lấp 80%. Đào hết tốn 5-6 giờ CPU
  (miễn phí, chạy nền) — khuyến nghị đào hết luôn, nhưng chiếm CPU nhiều giờ.
- **tier3 có đáng đuổi không?** Đưa tier3 từ 23% lên 70% tốn phần lớn ngân sách sinh, mà
  152M tham số có thể không đủ chỗ nhớ 5.703 thuật ngữ siêu chuyên. Đề xuất: **vòng 6 nhắm
  tier1+tier2 lên ≥90%, để tier3 ở mức kho gốc cho sẵn** — rồi đo xem lỗi thực tế còn bao nhiêu.
- **12 miền nhóm 🔴🟡** (hoá chất, kinh tế vĩ mô, cơ khí, thuỷ-lâm, bảo hiểm, thiên văn,
  xã hội học, nông nghiệp...): kho gốc không lấp nổi. Chọn: chấp nhận lỗi ~10% ở các miền
  này, hay đầu tư sinh riêng cho chúng?
