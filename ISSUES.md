# ISSUES — vấn đề đã xác định, CHƯA giải quyết

> Mỗi issue phải có: số đo, cách kiểm lại, và cái gì đã thử/đã loại. Không ghi phỏng đoán.

---

## #1 — Câu dài: vị trí token >160 gần như chưa được luyện (MỞ)

**Phát hiện 2026-07-28, sau vòng 6.**

### Số đo

Độ phủ vị trí trong `bin_v6` (15.284.511 chuỗi, dài trung bình 56,2 token):

| vị trí token | số chuỗi phủ tới đó | % corpus |
|---:|---:|---:|
| 32 | 9.579.413 | 62,67% |
| 64 | 4.342.143 | 28,41% |
| 96 | 2.484.702 | 16,26% |
| 128 | 1.481.337 | 9,69% |
| 160 | 734.086 | 4,80% |
| 192 | 268.404 | 1,76% |
| 224 | 58.740 | **0,38%** |
| 256 | 685 | **0,00%** |

`max_seq = 256` là trần **cứng** (bảng RoPE có 256 hàng) cho **input + output CỘNG LẠI**, vì
`generate_cached` giữ cả hai trong một chuỗi. Tỉ lệ token VI/JA đo trên 15,28M cặp:
**1,12** (trung vị), 1,41 (p90). Với ~2,05 ký tự/token ở văn phong Quốc hội, câu Nhật quá
**~245 ký tự** sẽ bị cắt output.

### Không phải khan hiếm nguồn — là chưa bao giờ chọn theo độ dài

`kokkai_ja.txt`, 2 triệu dòng đầu: **7,9% câu ≥150 ký tự, 1,3% ≥200 ký tự**
(158.646 và 25.144 câu). CC-100 còn 392,8M câu. Các script đào (`mine_skills2.py`,
`mine_rare.py`) chọn theo **KHUÔN MẪU** (mệnh đề lồng, slang, phủ định), **chưa lần nào
chọn theo ĐỘ DÀI**.

### Đã đo được gì về chất lượng thực tế

Trên 3 câu Quốc hội dài nhất (198–209 ký tự, 12–14 dấu `、`, không có `。` ở giữa):
`v6_avg` dịch **trọn cả ba, không bị cắt** (tổng 197/218/224 token, vừa dưới 256).
So với Gemini trên cùng 3 câu: v6 **rơi từ và sai thuật ngữ**
(`税理士/会計士` → "kế toán thuế / kế toán tăng"; mất `融資だと`, mất `税収減`),
Gemini đúng cả. ⇒ **v6 không gãy vì độ dài, mà kém vì năng lực.**

⚠️ Bench 200 câu **KHÔNG đo được** issue này: câu "dài" nhất trong đó chỉ **72–77 ký tự**,
còn Quốc hội held-out là 130–218. Muốn theo dõi issue này phải dùng `bench_held_*.jsonl`.

### Bốn hướng, xếp theo giá trị/chi phí

1. **Ghép cặp ngắn thành cặp dài** — nối 3-4 cặp có sẵn bằng `、`/`そして`. **$0, không tốn
   quota KD**, nguồn vô hạn. Sửa được độ phủ vị trí + dạy "câu dài thì dừng ở đâu".
   KHÔNG dạy được cấu trúc lồng thật (câu ghép không có phụ thuộc chéo mệnh đề).
2. **Đào theo độ dài rồi KD** — cách duy nhất dạy cấu trúc lồng thật. Câu dài tốn ~4×
   token/cặp, nên 200k cặp dài ≈ 800k cặp ngắn về quota. 200k đủ để nhân 4× độ phủ vị trí 224.
3. **Tách câu lúc inference** (cách Google làm) — không train lại. Rủi ro: tiếng Nhật đặt
   động từ CUỐI câu, vị ngữ cuối chi phối các mệnh đề trước; cắt ngang là mất mối đó.
   **Phải đo trước khi tin** — kiểm rẻ trên 120 câu `heldout_kokkai.txt`.
4. **Nới `max_seq` bằng RoPE interpolation — ĐỪNG LÀM TRƯỚC.** Cửa sổ 256 hiện tại còn chưa
   lấp đầy (vị trí 224 mới 0,38%); nới lên 512 chỉ làm vùng chưa luyện rộng thêm.
   Chỉ xét sau khi (1) hoặc (2) đã lấp vùng 160–256.

### Đã sửa (không phải giải quyết issue, chỉ hết crash)

`src/bitnet.py generate_cached()` giờ dừng gọn khi `pos + T > max_seq` thay vì nổ
`RuntimeError: size of tensor a (0) must match tensor b (768)`. Lỗi cũ tiềm ẩn ở **mọi**
caller (`translate_bench.py`, `hardbench_ckpt.py`, `translate_txt.py`, demo) — chỉ chưa lộ
vì input ngắn hơn. Nguyên nhân crash là **`--max-new 200` + prompt 102 token = 302 > 256**,
tức độ dài SINH RA, không phải độ dài đọc vào.

### Cách kiểm lại issue

```bash
python scripts/translate_txt.py <ckpt> D:/Bit-Translate-data/raw/heldout_kokkai.txt \
    eval/bench_held_vN.jsonl --label vN
```
Rồi đếm % câu có output bị cắt (không kết thúc bằng dấu câu), và so độ phủ vị trí bằng
đoạn đo trong mục "Số đo" ở trên.
