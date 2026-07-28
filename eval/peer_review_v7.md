<!-- nvidia/nemotron-3-ultra-550b-a55b:free | in 2936 out 2022 -->

### A. Chỗ sai / thiếu bằng chứng trong §6

| # | Khẳng định §6 | Số liệu bác / thiếu | Ghi chú |
|---|---|---|---|
| 1 | *“Khoảng cách lớn nhất với Google là câu dài (−23, p=0,001)”* → **ưu tiên data dài** | **Bảng §1**: `clause` (−25), `zeropron` (−25), `katakana` (−20) **cũng lớn bằng/hơn** gap dài (−23). `slang` (−20) dù data 5,17×. **Kết luận “gap lớn nhất = câu dài” sai số liệu**. |
| 2 | *“Nguyên nhân đo được là vị trí 224 chỉ 0,38% corpus”* → **nghịch lý vị trí token** | **§3**: p99 tổng token = 220, **chỉ 0,10% vượt 256**. Vị trí 224 phủ 0,38% chuỗi **đủ cho 99,9% phân phối thực**. Không có bằng chứng model thất thủ do *vị trí* chứ không phải *cấu trúc lồng* (xem §5: 37% cấu trúc câu, 81% đa nguyên nhân). |
| 3 | *“LR-restart còn ăn: v6 khởi đầu từ v5_avg, dev 2,18→2,09”* | **§2**: v6 có **+14,1% data mới**. Không có ablation tách LR-restart vs data mới → **không có bằng chứng** LR-restart tự thân có tác dụng. |
| 4 | *“Toàn bộ tiền dồn vào GPU vì data $0”* | **§4**: “Vòng 3 bơm data tổng hợp: hardbench +18 mà bench thật ĐỨNG YÊN”, “Vòng 5 bơm 2,45M Quốc hội: TED đứng yên”. **Lịch sử dự án chứng minh thêm data $0 KHÔNG đảm bảo ăn benchmark**. §6 bỏ qua tiền lệ này. |
| 5 | *“Kiểm sớm: dev loss < 2,0761 ở step 3000”* | **§2**: Δdev/1k step từ 12k→16k chỉ **−0,0025**. Từ 2,0896 xuống 2,0761 cần **−0,0135** ≈ **5.400 step** ở tốc độ hiện tại. **Ngưỡng 3.000 step vô lý so với đường cong thực**. |
| 6 | *“Ghép cặp ngắn thành cặp dài” giải quyết gap dài* | **§5**: 81% câu hỏng ≥2 nguyên nhân. Ghép câu ngắn **không tạo ra mệnh đề lồng, zeropron, slang, keigo** — những nguyên nhân chính. **Không có bằng chứng** ghép câu ngắn cải thiện các đặc trưng này. |

---

### B. Phương án kỳ vọng cao hơn với $9 (9.000 step)

| Phương án | Cơ chế | Cách biết sớm (≤3.000 step) | Ước lượng gain so với §6 |
|---|---|---|---|
| **1. Fine-tune chỉ trên held-out Quốc hội (4.435 cặp dev + 2M dòng kokkai ≥150 chars)** | Domain dài, cấu trúc lồng, zeropron thật — chính phân phối v6 thua Google nhất. | Dev loss Quốc hội giảm >0,02 **hoặc** chrF held-out tăng >1,5. | +5–8% accuracy dài (học cấu trúc thật thay vì ghép giả). |
| **2. Continued pre-training (denoising / span corruption) trên 15,3M cặp JA đơn ngữ** | BitNet 1.58 cần dung lượng model cho ngữ pháp JA; 152M quá nhỏ để học đồng thời dịch + ngôn ngữ. | Perplexity JA monolingual giảm >0,15 / dev loss dịch giảm song song. | Giảm lỗi `clause`, `zeropron`, `katakana` (lỗi kiến thức ngôn ngữ). |
| **3. LoRA rank 32 trên decoder-only (Qwen2-0.5B distillation logits)** | 152M enc-dec bị giới hạn attention heads; LoRA decoder-only khai thác LM mạnh hơn cho generation. | Val loss LoRA giảm nhanh hơn baseline enc-dec (so sánh cùng step). | Bypass bottleneck enc-dec 152M; tận dụng knowledge distillation thực sự. |

> **Chỉ làm 1 trong 3**, không chia nhỏ $9. Ưu tiên **Phương án 1** (rẻ nhất, trực diện lỗi dài, có held-out sẵn để early-stop).

---

### C. Rủi ro “ghép cặp ngắn thành cặp dài” chưa lộ số liệu

1. **Phân phối vị trí token giả mạo**: Ghép 3–4 câu ngắn bằng `、`/`そして` tạo ra chuỗi **dài về token nhưng shallow về depth cây phụ thuộc**. §5 xác nhận 37% lỗi do *cấu trúc câu lồng* — ghép ngang không tạo lồng.
2. **Zero-pronoun ngữ cảnh đoạn**: Ghép câu đơn mất chủ ngữ/đại từ thứ 2 trở đi → model học **không có zeropron** tại vị trí nối. §5: `zeropron` gap −25, mật độ data 1,07× → model đã thấy zeropron nhưng **chưa học giải quyết**; ghép câu làm giảm thêm tín hiệu 감독.
3. **Noise alignment**: Nối phần Việt “tương ứng” giả định alignment 1-1 ở mức câu. Sai alignment tại ranh nối → **label noise** lan gradient.
4. **Catastrophic forgetting short**: 9.000 step chỉ trên data ghép (hoặc mix) làm lệch phân phối về ngắn — §1 v6 đã thắng Google ở ngắn (81% vs 78%), dễ tuột.

**Không có số liệu nào trong hồ sơ đo lường depth câu, tỷ lệ zeropron tại ranh nối, hoặc alignment quality của cặp ghép.**

---

### D. Sai sót phương pháp nghiêm trọng chưa nhận ra

1. **Benchmark 200 câu không đo domain thực**: §4 minh chứng Quốc hội held-out nhảy 14→95% mà TED đứng yên. **Bench 200 câu (10 đặc trưng × 20) không đại diện cho miền dài thật** (clause, zeropron, keigo). Tối ưu bench này = overfit proxy sai.
2. **Model quá nhỏ cho BitNet 1.58 + enc-dec**: 152M ternary weights ≈ 30–40M effective FP16 params. **Không đủ dung lượng** học đồng thời: ngữ pháp JA, ngữ pháp Vi, alignment, copy number/katakana, discourse. Mọi đòn data/LR chỉ bám saturation.
3. **Không có verification set tách biệt cho early-stop**: Dùng dev loss (cùng phân phối train) và 120 câu Quốc hội (cùng domain bơm) → **selection bias**. Cần một tập *out-of-domain* (ví dụ: news, sub, legal khác) để chốt checkpoint.
4. **Bỏ qua inference-time compute**: §4 “Beam search vô ích, oracle +5,2 chrF”. Với model 152M, **generation diversity thấp** → beam/search không giúp. Nên đầu tư **speculative decoding / n-gram rerank** thay vì train thêm.
5. **BitNet 1.58 quantization-aware training chưa xác minh ổn định**: Không có log weight distribution, gradient norm, activation clipping qua các LR restart. Có thể **weight collapse** ở step muộn khiến dev loss giả giảm (flatter loss landscape chứ không phải better generalization).