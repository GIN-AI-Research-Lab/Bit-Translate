# Luồng điều tra MỚI: convert sang kiến trúc nhẹ-cực-đoan giữ bản chất (04/08)

Goal (user): kiến trúc mới giữ token/suy luận/in-out/đa luồng/trọng số nhưng NHẸ hơn mọi thứ
từng làm, chạy CPU+RAM máy B; test trước trên Qwen3-0.6B; đích chất lượng fp16/fp32.
Chuẩn trung thực: "chưa từng có" không kiểm chứng được — đây là các TỔ HỢP cơ chế chưa thấy
trong văn liệu, mỗi cái kèm phép thử giết-được-trong-một-buổi, $0 (CPU local).

## H1 — Gauge-canonical folding (gấp layer đẳng cấu qua chuẩn hóa gauge)
- Cơ chế: đưa mọi matrix về dạng chính tắc bằng ĐÚNG bộ gauge exact của lab (perm + diagonal)
  → đo khoảng cách cặp xuyên-layer → gấp cụm gần nhau: 1 bản weight + per-layer (perm idx + diag).
- Vì sao có thể mới: văn liệu chỉ có tied-layers (train từ đầu) & model-merging (giữa model);
  chuẩn-hóa-gauge hậu kỳ để gấp NỘI BỘ model = chưa thấy. Vũ khí riêng: gauge toolkit đã kiểm chứng.
- Test 0.6B (CPU ~30ph): canonical 28 bộ MLP → ma trận khoảng cách → gấp thử cặp gần nhất + PPL.
- Kỳ vọng ghi trước: 70% khoảng cách ~random (chết rẻ); 30% có cụm ≥2-4 layer gấp được.
- Nếu sống: scaling (gấp trong expert MoE 128 cái/block là mỏ vàng), + fine-align bằng
  Procrustes-trên-gauge, + correction nhỏ per-instance.
- **KẾT QUẢ (04/08, exp_y_gauge_fold_probe): H1 CHẾT** — real mean 1.4305 vs null 1.4347
  (98.4-99.7% null trên mọi cặp). Không có đẳng cấu xuyên-layer dưới gauge perm+diag.
  Tín hiệu phụ: cặp gần nhất toàn layer LIỀN KỀ (depth-smoothness ~0.3%, không khai thác được).
  Hệ quả: trọng số các layer là thông tin độc lập thật sự — folding hậu kỳ không có cửa;
  nhất quán với entropy-cao đã đo. Còn biến thể CHƯA thử: fold 128 expert TRONG 1 block MoE
  (cùng phân bố input — xác suất đẳng cấu cao hơn xuyên-layer) — để dành khi quay lại MoE.

## H2 — Response-surface layer (lưu HÀNH VI lớp, bỏ hẳn trọng số)
- Cơ chế: x → proj z (≈64 chiều học) → bảng/nội suy f(z) → proj ngược. Lưu bảng, không lưu W.
  Cược vào intrinsic-dimension thấp của activation manifold thật.
- Vì sao có thể mới: LUT-network/memorization có ở mạng nhỏ; áp cho LAYER của LLM pretrained
  như một phép CONVERT + đo PPL = gần như trống.
- Test 0.6B: thay 1 lớp mlp giữa bằng bảng fit trên 5k câu → PPL vs size bảng.
- Kỳ vọng ghi trước: 80% fail (off-manifold); sống 1 lớp = có đường scaling đáng đào.
- **KẾT QUẢ (04/08, exp_z_manifold_probe — probe rẻ-trước): H2 CHẾT** — intrinsic dim của
  activation tại mlp block-14: **645/1024 @95%** (894 @99%) — gần full-rank, không có manifold
  thấp chiều để tra bảng. Đóng hồ sơ không cần xây bảng.

## TỔNG KẾT LUỒNG (04/08): H1 ✗ + H2 ✗ + entropy-trọng-số-cực-đại (đo từ trước) hội tụ:
thông tin transformer đã train là ĐẶC — không đường tắt biểu diễn nào còn giấu. Free-lunch
chỉ tồn tại ở tầng FORMAT (gauge/perm/mask — đã vét). Con đường kiến trúc-mới còn sống: H3
(không đấu R(D), tái phân bổ chi phí theo token) + capacity (model to) + compute (KD).

## H3 — Hot-core fp16 + cold-overlay bit-thấp, cổng theo entropy (kiến trúc 2 tầng vật lý)
- Cơ chế: lõi ~100-200M fp16 in-RAM xử mặc định; overlay (nén sâu, mmap SSD) chỉ kích hoạt
  khi entropy/uncertainty lõi vượt ngưỡng → trả chi phí theo TOKEN KHÓ.
- Họ hàng xa: speculative/cascade — nhưng đảo vai (to = cấp cứu, không verify). Convert-only.
- Test 0.6B: lõi distill ~100M (tái dùng recipe v7a) + overlay 0.6B-ternary; đo % token lõi
  tự xử, chất lượng hỗn hợp, tốc độ hiệu dụng.
- Đây là ứng viên khả thi nhất cho goal "chạy máy B, chất lượng fp16 phần lớn thời gian".

Thứ tự: H1 (tối nay, xen kẽ ca đêm chuỗi native) → H2 → H3 (dự án tuần).
