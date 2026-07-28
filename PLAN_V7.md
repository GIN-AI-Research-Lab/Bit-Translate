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
