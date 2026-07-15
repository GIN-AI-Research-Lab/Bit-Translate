# Tiếp tục sinh câu glossary trên provider khác (GitHub Copilot / khác)

## Bối cảnh
Sinh **data huấn luyện dịch máy Việt–Nhật** từ glossary. Mỗi "batch" = 30 thuật ngữ.
Đã sinh **47,296 cặp câu / 267 batch**. Còn **330 batch**. *(cập nhật 2026-07-15 — Copilot đã chạy tới batch 266)*

## Trạng thái file
- **Đã sinh (giữ nguyên, đừng ghi đè):** `data/synthetic/glossary_sents.jsonl` (+ `.ja` / `.vi`) — 47,296 dòng `{"ja","vi","term"}`.
- **Input còn lại:** `data/synthetic/gen/batch_NNN.json` — mỗi file là mảng JSON `[{ja, vi, en}, ...]` (30 term).
- **Danh sách batch CẦN làm:** `data/synthetic/gen/todo_remaining.json` — mảng **330 số** (hiện bắt đầu từ `[267,268,269,...]`). Chỉ làm các số này.
- **Output đã có:** `data/synthetic/gen/out_NNN.jsonl` (Copilot ghi tới nay) — đừng làm lại các batch đã có out.

## Việc cần làm
Với **mỗi** số `N` trong `todo_remaining.json`:
1. Đọc `data/synthetic/gen/batch_{N:03d}.json` (vd N=78 → `batch_078.json`).
2. Với **mỗi thuật ngữ** trong file, sinh **6 cặp câu** (câu Nhật tự nhiên + bản dịch Việt trung thực).
3. Ghi kết quả ra `data/synthetic/gen/out_{N:03d}.jsonl` — mỗi dòng 1 JSON: `{"ja":"...","vi":"...","term":"<từ ja>"}`.

## Prompt/tiêu chuẩn chất lượng (dùng y hệt để nhất quán)
> Với mỗi thuật ngữ, viết 6 cặp câu (JA tự nhiên + VI dịch trung thực) **DÙNG thuật ngữ đúng nghĩa** trong câu.
> - JA: tiếng Nhật bản ngữ tự nhiên. VI: tiếng Việt tự nhiên, khớp nghĩa CẢ câu (không chỉ từ khoá).
> - Đa dạng: trần thuật / câu hỏi / **nhờ-mệnh lệnh lịch sự** (〜てください, 〜ていただけますか); trộn **thể lịch sự + khẩu ngữ**.
> - Ngữ cảnh: IT/phần mềm, văn phòng/nghiệp vụ, đời thường.
> - **Giữ katakana** nếu term là katakana (vd `ライブラリ` giữ nguyên trong câu JA).
> - Câu 5–25 từ, KHÔNG lặp khuôn.

## Sau khi Copilot làm xong: gộp lại (chạy 1 lệnh)
```bash
python scripts/merge_glossary_sents.py
```
Script gộp `glossary_sents.jsonl` (bản cũ) + tất cả `gen/out_*.jsonl` (bản mới) → khử trùng theo (ja,vi) → ghi lại `data/synthetic/glossary_sents.{jsonl,ja,vi}` và in tổng.

## Kiểm tra tiến độ bất cứ lúc nào
```bash
ls data/synthetic/gen/out_*.jsonl | wc -l    # số batch đã có output
```

## Ghi chú
- **Model gợi ý:** loại rẻ/nhanh (Copilot GPT-4o-mini / tương đương) là đủ — đây là sinh câu ví dụ, không cần model mạnh nhất. Tiết kiệm quota.
- Không cần lọc LaBSE ở bước này — lọc chất lượng chạy sau trên GPU node (script `scripts/clean_stage3_labse.py`).
- Ngoài ra còn **17,951 cặp term-level trực tiếp** ở `data/synthetic/glossary_direct.{ja,vi}` (miễn phí, đã có).
