# HỒ SƠ KỸ THUẬT — dự án model dịch ja→vi 152M tham số, BitNet 1.58-bit

Bạn là chuyên gia machine translation độc lập. Hồ sơ dưới đây là số liệu ĐO THẬT của
một dự án đang chạy. Người phân tích duy nhất từ trước tới giờ là một AI khác, và nó
đã sai ít nhất 2 lần. Việc của bạn là tìm chỗ chuỗi lập luận SAI, không phải xác nhận nó.

## 0. Bối cảnh cố định (không đổi được)

- Model: Transformer encoder-decoder 18 lớp, d_model 768, **152M tham số**, weight ternary
  (BitNet b1.58). Train from-scratch. Chỉ tối ưu chiều ja→vi.
- Data train: dịch máy do LLM thầy sinh (knowledge distillation) từ câu tiếng Nhật đơn ngữ
  đào từ CC-100 (392,8M câu). Thầy: gemini-flash-lite, qwen-plus.
- Phần cứng train: 1 GPU L40S thuê giờ. Ngân sách CỰC HẠN CHẾ (~$8-16/vòng).
- Mỗi vòng = bơm thêm data mới + train tiếp từ checkpoint vòng trước.

## 1. Cách đo chất lượng (bench chính)

200 câu tiếng Nhật thật (trích TED/OpenSubtitles), **cân bằng nhân tạo: 10 đặc trưng ×
đúng 20 câu**, 100 câu ngắn / 100 câu dài. Không có bản dịch tham chiếu.
Chấm bằng LLM-judge MÙ (các hệ xáo thành A/B/C), thang **0/1/2 về ĐÚNG NGHĨA**:
2 = người đọc hiểu đúng ý; 1 = mất/lệch một ý; 0 = sai nghĩa.
**Con số % = tỉ lệ câu đạt đúng 2 điểm.**

Kết quả (cùng một phiên chấm, n=20 mỗi đặc trưng):

| đặc trưng | v4 | v5 | Google | KTC 95% của v5 (n=20) |
|---|---:|---:|---:|---|
| clause | 55% | 50% | 95% | [30–70] |
| idiom | 55% | 60% | 65% | [39–78] |
| katakana | 60% | 75% | 95% | [53–89] |
| keigo | 75% | 80% | 80% | [58–92] |
| negation | 75% | 65% | 85% | [43–82] |
| number | 65% | 70% | 95% | [48–85] |
| plain | 75% | 70% | 90% | [48–85] |
| question | 60% | 75% | 90% | [53–89] |
| slang | 50% | 45% | 90% | [26–66] |
| zeropron | 70% | 65% | 85% | [43–82] |
| **TỔNG (n=200)** | **64%** | **66%** | **87%** | [59–72] |

## 2. Phân loại nguyên nhân lỗi

Người đọc tay **59 câu mà v5 chấm dưới 2 điểm NHƯNG Google chấm 2 điểm** (tức lỗi
"đuổi kịp được", loại bỏ câu khó với mọi hệ). Mỗi câu gán 1 nguyên nhân CHÍNH + các
nguyên nhân phụ:

| nguyên nhân | là CHÍNH | % | xuất hiện BẤT KỲ | % |
|---|---:|---:|---:|---:|
| tu_vung_thong_thuong | 13 | 22% | 25 | 42% |
| cau_truc_cau | 22 | 37% | 25 | 42% |
| van_phong_tu_nhien | 0 | 0% | 15 | 25% |
| an_chu_ngu | 6 | 10% | 14 | 24% |
| them_bot_nghia | 3 | 5% | 11 | 19% |
| thuat_ngu_chuyen_nganh | 2 | 3% | 9 | 15% |
| katakana_ngoai_lai | 3 | 5% | 7 | 12% |
| quan_ngu_thanh_ngu | 5 | 8% | 5 | 8% |
| ten_rieng | 1 | 2% | 4 | 7% |
| phu_dinh_modality | 3 | 5% | 3 | 5% |
| bi_dong_sai_khien_thu_nhan | 1 | 2% | 1 | 2% |

Số nguyên nhân trên một câu hỏng: 1 nguyên nhân = 11 câu (19%),
2 nguyên nhân = 36 câu (61%), 3 nguyên nhân = 12 câu
(20%). Trung bình 2.0 nguyên nhân/câu.

Số câu mà một mặt trận là nguyên nhân **DUY NHẤT** (tức sửa xong mặt trận đó là câu
lên 2 điểm): cấu trúc câu 2, từ vựng thông thường
1, ẩn chủ ngữ 2,
quán ngữ 3, thuật ngữ chuyên ngành
0.

Ví dụ thật (câu nguồn → bản dịch v5 → bản dịch Google):

- **quan_ngu_thanh_ngu**: `途方に暮れましたが 大きく手を広げることにしました`
  - v5 (1đ): Dù hoàn toàn bế tắc nhưng tôi đã quyết định mở rộng tầm mắt ra.
  - google (1đ): Tôi cảm thấy hụt hẫng nhưng vẫn quyết định rẽ nhánh.
  - ghi chú người đọc: Model dịch sai quán ngữ '手を広げる' (mở rộng phạm vi/rẽ nhánh hoạt động) thành 'mở rộng tầm mắt'.

- **an_chu_ngu**: `製作費8千万ドルですから まあ1カ月でペイします`
  - v5 (1đ): Chi phí sản xuất là 80 triệu đô la, nên chắc là tôi sẽ trả trong một tháng.
  - google (2đ): Nó tiêu tốn 80 triệu đô la để thực hiện, vì vậy nó sẽ trả hết sau một tháng.
  - ghi chú người đọc: Model tự ý thêm chủ ngữ 'tôi' khi câu ẩn chủ ngữ và hiểu sai từ Katakana ペイする (hoàn vốn/thu hồi vốn) thành 'trả'.

- **tu_vung_thong_thuong**: `そのやり方は微妙で なかなか気がつきません`
  - v5 (1đ): Cách làm đó hơi khó hiểu và tôi mãi không nhận ra.
  - google (2đ): Phương pháp này tinh vi và khó nhận thấy.
  - ghi chú người đọc: Bản dịch dịch sai từ '微妙' thành 'hơi khó hiểu' thay vì 'tinh vi', đồng thời tự ý thêm chủ ngữ 'tôi' vào câu diễn đạt chung.

- **cau_truc_cau**: `ハッピーなまま 一生が終わればいいですけれども`
  - v5 (0đ): Mong rằng hạnh phúc sẽ kết thúc trọn đời.
  - google (2đ): Tôi ước gì tôi có thể kết thúc cuộc đời mình một cách hạnh phúc.
  - ghi chú người đọc: Model phân tích sai cấu trúc ngữ pháp khi coi 'hạnh phúc' là chủ ngữ của hành động 'kết thúc' thay vì diễn đạt 'cuộc đời kết thúc trong trạng thái hạnh phúc'.

## 3. Mật độ "kỹ năng" trong corpus train

Đo bằng regex trên câu tiếng Nhật: % câu trong corpus có chứa khuôn của từng kỹ năng.
Giả thuyết mà dự án đang dùng: **keigo đạt 4,0% corpus → điểm 80% = NGANG Google**, nên
4% được lấy làm "mốc đủ để học xong một kỹ năng".

| kỹ năng | corpus v5 | corpus v6 | bội | điểm v5 | Google |
|---|---:|---:|---:|---:|---:|
| phu_dinh_kho | 0.43% | 2.66% | 6.25× | 65% | 85% |
| slang | 0.63% | 3.24% | 5.17× | 45% | 90% |
| khau_ngu_suong_sa | 0.42% | 0.63% | 1.49× | – | – |
| tu_da_nghia | 0.87% | 1.15% | 1.32× | – | – |
| cau_che_nominal | 1.80% | 2.31% | 1.28× | – | – |
| menh_de_long | 5.52% | 6.11% | 1.11× | 50% | 95% |
| an_chu_ngu | 22.52% | 24.14% | 1.07× | 65% | 85% |
| katakana_dai | 23.23% | 24.24% | 1.04× | 75% | 95% |
| keigo_MOC | 4.00% | 3.91% | 0.98× | 80% | 80% |
| so_nien_hieu | 3.00% | 2.83% | 0.94× | 70% | 95% |
| idiom_MOC | 1.24% | 1.10% | 0.88× | 60% | 65% |

Corpus v5 = 13.144.706 cặp câu. Corpus v6 = 15.307.817 (thêm 2.163.111 câu đào theo
nguyên nhân lỗi ở §2). Độ phủ thuật ngữ 54 miền: 55,5% → 77,8%.

## 4. Đường cong loss của v6 (đang train) so với v5 (đã xong)

v6 khởi đầu bằng cách nạp trung bình 7 checkpoint cuối của v5 rồi train tiếp trên
corpus v6, LR cosine 8e-5 → warmup 200 step → giảm dần, tổng 16.400 step
(= 2,5 epoch, 2,15 tỉ token trên corpus 858M token).

| cửa sổ step | loss v6 | loss v5 |
|---|---:|---:|
| 0–1000 | 2.1393 | 2.1574 |
| 1000–2000 | 2.1465 | 2.1490 |
| 2000–3000 | 2.1407 | 2.1383 |
| 3000–4000 | 2.1350 | 2.1278 |
| 4000–5000 | 2.1278 | 2.1185 |
| 5000–6000 | 2.1214 | 2.1086 |
| 6000–7000 | 2.1147 | 2.0806 |
| 7000–8000 | 2.1003 | 2.0733 |
| 8000–9000 | 2.0813 | 2.0653 |
| 9000–10000 | 2.0752 | 2.0599 |
| 10000–11000 | 2.0702 | 2.0516 |

v5 dừng ở step 14.000 (hết tiền, KHÔNG hội tụ) với loss 500 step cuối = 2,0190 và
độ dốc −0,006/1000 step. v6 hiện ở step 11520.
Loss của v5_avg trên corpus v6 lúc mới nạp (step 10, LR≈0) = 2,0690.

## 5. Kiểm giữa chừng: checkpoint step 9000 của v6

chrF (n-gram ký tự, so 1 bản tham chiếu do LLM soạn) trên bộ 100 câu khó:

| miền | v5 (cuối, đã trung bình ckpt) | v6 step9000 (chưa anneal) | Δ | data v6 bơm vào |
|---|---:|---:|---:|---|
| caudai (câu dài) | 48,8 | 52,0 | +3,2 | – |
| slang | 31,4 | 34,0 | +2,6 | bơm 5,2× mật độ |
| zeropronoun | 47,1 | 49,5 | +2,4 | 1,07× |
| hoithoai | 44,8 | 46,6 | +1,8 | – |
| thanhngu | 41,7 | 43,2 | +1,5 | 0,88× (loãng đi) |
| it_deep | 48,4 | 49,0 | +0,6 | – |
| hop | 43,0 | 43,3 | +0,3 | – |
| solieu | 74,2 | 73,3 | −0,9 | 0,94× |
| nguphap | 51,1 | 49,5 | −1,6 | – |
| **keigo** | **46,5** | **39,9** | **−6,6** | **không bơm, 0,98×** |
| TỔNG TB | 47,7 | 48,0 | +0,3 | |

Cùng bộ này Google được 36,1 và Claude Haiku 41,5 — tức chrF nói model đang hơn Google
+11,9, trong khi LLM-judge ở §1 nói Google hơn model 21 điểm phần trăm.
Trên FLORES-200 (bộ công khai, tham chiếu do người dịch chuyên nghiệp), chiều ja→vi:
Google 53,5 · Haiku 51,1 · model dự án 42,3.

## 6. Lịch sử: những gì đã thử và KẾT QUẢ THẬT

- **Vòng 3**: bơm data tổng hợp theo chủ đề tự nghĩ ra → hardbench +18 điểm nhưng bench
  câu thật ĐỨNG YÊN.
- **Vòng 5**: bơm 2,45M câu biên bản Quốc hội → bench TED đứng yên (66% vs v4 64%,
  p=0,755) NHƯNG câu Quốc hội held-out nhảy 14% → 95%.
- **Giả thuyết "KD bị pha loãng"** (train lại với tỉ lệ KD đậm 35%): +0,05 điểm rồi tụt
  vì overfit → BỊ BÁC.
- **Beam search**: vô ích. **Trung bình 5-7 checkpoint cuối**: +0,7 chrF, miễn phí.
- **Mở rộng độ sâu 12→18 lớp bằng cách chèn block identity** (thay vì train lại từ đầu):
  rẻ hơn 3×, giữ nguyên chất lượng.

## 7. Kế hoạch vòng 7 hiện tại (nguyên văn)


# KẾ HOẠCH VÒNG 7 — quyết theo KẾT QUẢ v6, không quyết trước

> Viết 2026-07-28 lúc v6 đang chạy. Câu hỏi cần trả lời: **vòng sau thêm data hay
> train thêm step trên data hiện có?** Không đoán bây giờ — dưới đây là LUẬT QUYẾT
> ĐỊNH dựa trên số liệu v6 sẽ có.

## 0. Trạng thái khi viết

| | |
|---|---|
| Corpus v6 | 15,39M cặp (v5 13,14M + 2,20M đào theo nguyên nhân lỗi) |
| Train v6 | 16.400 step ≈ 2,75 epoch ≈ $16,4 |
| Ngân sách sau v6 | **$0** — cần credit mới cho v7 |
| Mốc phải vượt | bench TED: v5 66%, Google 87% |

---

## 1. LUẬT QUYẾT ĐỊNH: đọc loss cuối v6

Lấy từ `logs/KETQUA_V6.txt` hoặc `D:/Bit-Translate-data/train_v6.log`, tính độ dốc
1000 step cuối (script `skill_density.py` có sẵn đoạn tính, hoặc chạy tay).

| Độ dốc loss/1000 step cuối | Chẩn đoán | Vòng 7 làm gì |
|---|---|---|
| **< −0,005** (còn giảm rõ) | **Thiếu STEP**, không thiếu data | Train tiếp trên chính bin_v6, **+12.000 → 16.000 step** (tổng ~5 epoch). KHÔNG thêm data. |
| **−0,005 … −0,002** | Đang bão hoà | Train tiếp **+8.000 step**, đồng thời chuẩn bị data cho v8 |
| **> −0,002** (phẳng) | **Đã vắt hết data hiện có** | Bắt buộc thêm data mới. Xem §3. |

Mốc tham chiếu: v5 dừng ở step 14.000 với độ dốc **−0,006/1000** → đó là trạng thái
"còn ăn được", và đúng là vòng 6 vẫn tiếp tục học. Nếu v6 kết thúc cũng ở mức đó thì
**train tiếp là rẻ nhất**: không tốn công gom data, không tốn quota KD, chỉ tốn GPU.

### Vì sao "train tiếp" có thể thắng "thêm data"
Corpus v6 có 15,39M cặp × ~50 token = ~770M token. Ở 2,75 epoch model mới thấy
~2,1 tỷ token. Chinchilla cho 152M tham số gợi ý ~3 tỷ token là tối ưu tính toán —
mà model nhỏ trên tác vụ hẹp thường cần **nhiều hơn** mức đó, không ít hơn.
⇒ **Nhiều khả năng v6 vẫn còn dưới điểm bão hoà.**

---

## 2. LUẬT QUYẾT ĐỊNH: đọc kết quả từng MẶT TRẬN

Bench theo đặc trưng (`eval/judge_v6/`) so với mốc v5. Mỗi mặt trận cho một tín hiệu
KHÁC NHAU về việc bơm data có tác dụng không:

| Mặt trận | Mốc v5 | Google | Data v6 đã bơm | Nếu KHÔNG nhích thì kết luận gì |
|---|---:|---:|---|---|
| mệnh đề lồng | 50% | 95% | 400k văn nói nhiều mệnh đề | Không phải thiếu văn nói → nghi **giới hạn dung lượng thật**, cân nhắc 200M+ |
| slang | 45% | 90% | 450k (0,63%→~3%) | Bác luôn giả thuyết "đói data" — cần đổi cách (sinh có ngữ cảnh) |
| nghĩa từ thường | — | — | 512 từ × 200 câu | Cách này chưa có tiền lệ; không ăn thì bỏ, đừng lặp |
| ẩn chủ ngữ | 65% | 85% | 350k câu truy hồi được | Xác nhận "đã 22,5% mà vẫn hỏng" là vấn đề KIẾN TRÚC (thiếu ngữ cảnh câu trước) |
| phủ định | 65% | 85% | 350k (0,43%→~2,6%) | — |
| katakana | 75% | 95% | 9.000 từ mượn | Nếu ăn → xác nhận cách "vá theo từ" đúng cho lớp từ mở |
| keigo | 80% | 80% | không bơm | **Mốc đối chứng**: phải giữ nguyên. Tụt = quên kiến thức cũ |

**Quy tắc chung:** mặt trận nào bơm data mà KHÔNG nhích thì **đừng bơm tiếp cùng loại
ở vòng 7** — đó là bài học đắt nhất của cả dự án (vòng 3 bơm data tổng hợp theo chủ đề
tự nghĩ ra, hardbench +18 điểm mà bench thật đứng yên).

---

## 3. NẾU phải thêm data — thứ tự ưu tiên đã tính sẵn

Chỉ làm khi §1 kết luận "phẳng". Nguồn còn lại, đã đo:

| Nguồn | Còn bao nhiêu | Ghi chú |
|---|---:|---|
| Sinh cho từ 0 lần | 1.024 từ × ~20 câu ×3 = 61k câu | Không đào được — cả 393M câu web không có |
| 7 miền văn hoá thiếu | ~2.300 từ | kankonsosai 37%, cờ bạc 46%, văn hoá truyền thống 50%, võ đạo 57%, lễ hội 58% (mốc: 74,9%) |
| Katakana ngoài top 9.000 | 35.000 từ nữa | 44.142 từ dưới ngưỡng, mới lấy 9.000 |
| Tên riêng | 10k họ + 1.741 địa danh + 9k ga | **Chưa làm bao giờ**; 60% tên riêng trong corpus chỉ xuất hiện 1 lần |
| Thuật ngữ 12 miền đáy | ~1.500 từ | kinh_te_vi_mo 53%, hoa_chat 56%, co_khi 57% |

**Ưu tiên 1 nếu thêm data: TÊN RIÊNG.** Là lớp từ mở duy nhất chưa động tới, có danh
sách CÔNG KHAI và ĐÓNG (10.000 họ phủ ~96% dân số Nhật), và lỗi của nó làm câu vô dụng
hoàn toàn (西村→"Tây Thôn"). Cách làm khác thuật ngữ: cần **nhiều tên khác nhau, tần
suất thấp** để học QUY TẮC kanji→romaji, KHÔNG oversample.

---

## 4. Ngân sách và tài khoản

Đã dùng hết $16,5 của `trituekstns`. Các profile Modal còn:
`nguyentuanngai`, `free30` (workspace nguyenda190497), `tritue12` (trituenguyen97).
Đổi bằng `modal profile activate <tên>`; volume và checkpoint là **theo tài khoản**,
nên phải upload lại `bin_v6` + checkpoint sang volume của account mới (~40 phút).

Ước chi phí v7 theo phương án:
- Train tiếp 12.000 step trên bin_v6: **~$12**, không tốn quota KD
- Thêm data rồi train: ~$12 GPU + vài giờ KD (miễn phí, quota TPM)

---

## 5. Việc phải làm ngay khi v6 xong (đã tự động hoá)

`scripts/auto_eval_v6.sh` chạy nền, tự làm:
1. Theo dõi tốc độ step tới lúc ổn định, in ETA
2. Trung bình 7 checkpoint cuối → `v6_avg.pt`
3. Dò ngữ pháp 80 phép thử + dịch bench 200 câu + hardbench
4. **Bench TRONG MIỀN**: 3 miền `probe_*` + 100 câu Quốc hội held-out

Kết quả gom ở **`logs/KETQUA_V6.txt`**.

**Người phải làm tay** (không tự động được vì cần chấm mù):
```bash
python eval/build_panels_bench.py eval/judge_v6 v5=eval/bench_v5.jsonl \
    v6=eval/bench_v6.jsonl google=eval/bench_google.jsonl
```
⚠️ Nhắc lại bài học vòng 5: **KHÔNG kết luận chỉ bằng bench TED.** Văn thuyết trình
phổ thông gần như không có thuật ngữ chuyên ngành, nên nó giấu mất phần cải thiện
trong miền. Phải xem cả `eval/dom6_*.jsonl` và `eval/bench_held_v6.jsonl`.


---

## 8. Các kết luận mà AI phân tích trước đã rút ra — HÃY CÔNG PHÁ TỪNG CÁI

1. "Nút thắt thật là CẤU TRÚC CÂU (37% nguyên nhân chính), không phải thuật ngữ (3%).
   Cả một tuần đo độ phủ thuật ngữ và đào 393M câu chỉ nhắm vào 3% nguyên nhân."
2. "Mốc mật độ 4% là điều kiện CẦN nhưng KHÔNG ĐỦ: mệnh đề lồng đã có 5,52% ở v5 —
   cao hơn mốc keigo — mà vẫn chỉ 50% vs Google 95%. Nên với mặt trận đã trên 4%, bơm
   thêm data cùng loại là vô ích."
3. "81% câu hỏng có ≥2 nguyên nhân, và sửa trọn một mặt trận chỉ cứu được 2-3 câu trên
   59. Đó là lý do mọi vòng bơm data đều thấy hardbench nhảy mà bench thật đứng yên."
4. "Vì vậy vòng 7 nên TRAIN THÊM STEP trên corpus v6 (dốc loss còn −0,0199/1000, mới
   2,5 epoch trong khi Chinchilla gợi ý ~3 tỉ token cho 152M) chứ không bơm data mới.
   Ưu tiên 'tên riêng' trong PLAN_V7 §3 là sai vì tên riêng chỉ 2% nguyên nhân."
5. "keigo tụt 6,6 chrF là quên kiến thức cũ (catastrophic forgetting), khớp với việc
   keigo là kỹ năng DUY NHẤT bị loãng mật độ (4,00% → 3,91%)."
6. "chrF không dùng được để kết luận: nó nói model hơn Google +11,9 trong khi judge nói
   kém 21 điểm; trên FLORES thì đảo dấu thành −11,2."

## 9. Câu hỏi cho bạn

Trả lời NGẮN GỌN, có cấu trúc, ưu tiên chỗ bạn thấy lập luận sai:

**A.** Kết luận nào trong §8 là SAI hoặc thiếu bằng chứng? Chỉ rõ số liệu nào bác nó.
**B.** Nút thắt thật của model 152M này là gì: thiếu step, thiếu data, thiếu dung lượng
tham số, thiếu ngữ cảnh, hay lỗi ở chính cách ĐO? Xếp hạng có lý do.
**C.** Có sai sót phương pháp nào nghiêm trọng mà cả dự án chưa nhận ra? (nghĩ tới: data
train là output của LLM thầy nên có trần chất lượng; bench cân bằng nhân tạo 10×20;
n=20 mỗi đặc trưng; judge không hiệu chuẩn giữa phiên; khởi đầu vòng mới bằng checkpoint
đã trung bình)
**D.** Với ngân sách ~$9 (≈9.000 step L40S) cho vòng 7, việc CỤ THỂ nào có kỳ vọng cao
nhất? Nêu 1 phương án chính + 1 phương án dự phòng, kèm cách biết sớm là nó đang ăn.
