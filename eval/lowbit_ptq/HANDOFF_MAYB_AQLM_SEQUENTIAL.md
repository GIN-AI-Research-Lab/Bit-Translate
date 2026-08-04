# HANDOFF — đóng câu hỏi AQLM vs scalar-ternary+SEQUENTIAL (từ Máy A, cho Máy B)

**Ngày**: 2026-08-04 · **Từ**: phiên Claude máy A (desktop, đang chạy PPL full-model nốt job
cũ, không rảnh tay code thêm) · **Cho**: phiên chạy trên máy B (laptop, CPU, không cần GPU —
toàn bộ việc dưới đây là PTQ thuần, giống mọi việc khác trong nhánh lowbit gần đây).

**Đọc trước**: `RESEARCH_AQLM_CODEBOOK_PTQ.md` mục "5b. ĐỐI CHIẾU với phiên song song trên
Máy B" — đây là lý do việc này tồn tại. Tóm tắt 3 dòng: phiên máy A (AQLM trên OLMoE-1B-7B,
KHÔNG sequential) đo AQLM thắng ternary N:M. Phiên máy B (VQ trên Qwen3-0.6B, CÓ sequential)
đo VQ thua ternary+SEQUENTIAL ~10×. Hai kết quả KHÔNG tương đương vì thiếu đúng 1 biến:
chưa ai test AQLM+sequential CÙNG model để so trực tiếp.

## 0. Việc cần làm

Port sequential/BRECQ-lite (đòn lớn nhất lab từng đo, Bài 8: "SEQUENTIAL ăn ~10×") vào AQLM,
so với scalar ternary+sequential — CÙNG model, CÙNG bpw, CÙNG phương pháp đo — để dứt điểm
câu hỏi "AQLM có thật sự tốt hơn ternary hay chỉ là ternary chưa được tối ưu hết".

**Có 2 lựa chọn model, cả hai đều hợp lệ, chọn cái bạn thấy tiện hơn:**
- **Tiếp tục trên OLMoE-1B-7B** (dùng lại toàn bộ hạ tầng máy A đã viết: `exp_al/am/an_*.py`,
  `exp_aj_generalization.py`) — cần tải GGUF Q4_K_M (~4,2GB, xem lệnh §2) vì file này KHÔNG
  commit vào git (data, không phải code).
- **Chạy trên Qwen3-0.6B** (dùng lại hạ tầng có sẵn của chính máy B:
  `exp_ad_vq_codebook.py`/`exp_r_qat_lite.py`/`s1_sequential`) — có thể tiện hơn vì model +
  data + oracle đã có sẵn tại máy B từ trước, không cần tải gì thêm. **Khuyến nghị hướng này**
  vì tái dùng được `s1_sequential`/`exp_k_fixpack_sequential.py` gần như nguyên vẹn, chỉ cần
  đổi quantizer bên trong từ scalar-ternary sang AQLM.

## 1. Thiết kế cụ thể (nếu chọn Qwen3-0.6B — khuyến nghị)

1. Lấy `s1_sequential` (hoặc `exp_k_fixpack_sequential.py`) làm khung — đây đã là vòng lặp
   block-wise: mỗi block nhận input đã lượng tử của block trước, tối ưu khớp output FP.
2. Thay bước quantize BÊN TRONG mỗi block: hiện đang gọi `q_ternary`/tương tự (scalar+mask) —
   đổi sang gọi AQLM (`aqlm_err`/`aqlm_quantize`-style, xem `exp_am_aqlm_full.py` máy A đã
   viết, copy logic k-means+beam sang) để tái tạo trọng số bằng codebook thay vì scalar+mask.
3. Đo PPL vi/ja held-out CÙNG cách máy B đã đo VQ+SEQ (6,044 @1,587bpw) và scalar+SEQUENTIAL
   (594,3 @1,94bpw) — so 3 số CÙNG bảng: scalar+SEQ, VQ/AQLM+SEQ (máy B đã có), AQLM+SEQ
   (việc mới, dùng đúng recipe k-means+beam của máy A thay recipe VQ cũ của máy B nếu khác).

   ⚠️ Lưu ý: máy B đã tự làm VQ+SEQUENTIAL rồi (kết quả 6,044, TỆ hơn scalar 10×) — có thể
   đây ĐÃ LÀ câu trả lời và không cần làm lại! Đọc kỹ `RESEARCH_VQ_CODEBOOK_PTQ.md` của chính
   máy B trước — nếu recipe VQ đó ĐÃ dùng k-means+beam-search giống máy A (không phải biến thể
   khác), thì câu hỏi ĐÃ ĐÓNG: AQLM/VQ thua scalar+sequential, điểm mạnh của AQLM (thắng ở
   Bài 5b của máy A) CHỈ tồn tại khi CHƯA có sequential. Nếu recipe VQ của máy B khác cách
   gán/refine của máy A (ví dụ không có beam-search, hoặc beam width khác) thì đáng thử lại
   với ĐÚNG recipe máy A trước khi kết luận.

## 2. Nếu chọn OLMoE (cần tải thêm)

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='allenai/OLMoE-1B-7B-0924-GGUF', filename='olmoe-1b-7b-0924-q4_k_m.gguf', local_dir='<duong-dan-may-B>')"
```

Sau đó sửa `GGUF_PATH` trong `exp_ak_ppl_real.py`/`exp_aj_generalization.py` cho khớp đường
dẫn máy B (máy A dùng `E:/hf_gguf_cache/...` — máy B không có ổ này, tự chọn đường dẫn phù hợp
theo `CLAUDE.md` §2 máy B).

## 3. Tiêu chí trả lời (ghi trước, đúng kỷ luật lab)

| Kết quả | Ý nghĩa |
|---|---|
| AQLM+SEQ thắng scalar+SEQ | AQLM là cải tiến thật, kể cả với đòn mạnh nhất đã biết — đáng đầu tư kernel/đóng gói |
| AQLM+SEQ thua scalar+SEQ, nhưng gap < 2× | Không rõ ràng — cần thêm seed/tinh chỉnh trước khi kết luận |
| AQLM+SEQ thua scalar+SEQ, gap ~10× (giống máy B đã đo VQ) | Đóng dứt điểm: AQLM chỉ thắng khi CHƯA có sequential; sequential là đòn không chuyển giao sang codebook. Cập nhật `RESEARCH_AQLM_CODEBOOK_PTQ.md` §5b, xóa mục "chưa giải quyết" |

## 4. Không đụng

File của máy A (`exp_al/am/an/ag/ah/ai/aj/ak_*.py`) và máy B (`exp_ad/ae/af_vq_*.py`) —
việc này tạo file MỚI (`exp_ao_*.py` hoặc tên tương ứng máy B đang dùng), so sánh KẾT QUẢ với
cả hai, không sửa code cũ.
