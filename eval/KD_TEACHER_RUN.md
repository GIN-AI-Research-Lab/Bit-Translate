# Chạy thầy Sonnet 5 — pilot audit-300 rồi scale (PLAN_V8 §T3→§T4)

> User chốt 2026-08-08: thầy = **Sonnet 5 toàn bộ**. Harness: `scripts/gen_teacher_sonnet.py`.
> Glossary canonical: `eval/glossary_v8_candidates.csv` (dùng làm gợi ý + kiểm PIN_VI).

## Điều kiện (bạn tự làm — Claude không xử lý được credential)

```bash
pip install anthropic          # chưa cài trên Máy A
export ANTHROPIC_API_KEY=sk-ant-...   # hoặc: ant auth login
```

## Bước 1 — Pilot audit 300 câu (LUẬT dự án: audit thầy TRƯỚC khi bung)

Tập audit đã dựng sẵn: `eval/pilot300_teacher_audit.ja` (300 câu = 64 câu lỗi v7a +
phủ đều 6 miền; **audit-only, KHÔNG được đưa vào KD training** vì trùng bench test).

```bash
python scripts/gen_teacher_sonnet.py \
  --src eval/pilot300_teacher_audit.ja \
  --out eval/pilot300_sonnet.jsonl \
  --effort low
# in ra tổng token + ~chi phí (giá intro Sonnet 5 $2/$10). Pilot 300 câu ≈ $0.2-0.4.
```

Sau khi có `pilot300_sonnet.jsonl`: **Opus 4.8 chấm mù** bản dịch Sonnet vs JA nguồn
(acc+nat, đúng phương pháp bench 1200) → quyết Sonnet có **đạt làm thầy** không.
Gate đạt: acc==2 ≥ ~95%, nat==2 ≥ ~90%, và trên riêng 64 câu lỗi v7a phải khá rõ.
Nếu fail → xem lại prompt/effort (bump `--effort medium` cho idiom/phủ định) rồi audit lại.

## Bước 2 — Scale KD (SAU khi pilot đạt + SAU T0 chuyển corpus về Máy A)

Nguồn câu JA cho KD (chưa có trên Máy A — chờ T0):
- Bucket ③ đồng âm/katakana: **mine** corpus 11.88M theo glossary + homograph list.
- Bucket ①② phủ định/idiom: **sinh** (minimal-pair + idiom×6) — dùng `gen_via_api.py`
  hoặc sinh riêng, rồi cho Sonnet dịch chuẩn hoá.

```bash
# ví dụ khi đã có nguồn:
python scripts/gen_teacher_sonnet.py --src data/synthetic/v8/bucket3_src.ja \
  --out data/synthetic/v8/bucket3_kd.jsonl --effort low
python scripts/gen_teacher_sonnet.py --src data/synthetic/v8/bucket12_src.ja \
  --out data/synthetic/v8/bucket12_kd.jsonl --effort medium   # idiom/phủ định
```

Sau đó: LaBSE≥0.55 gate → dedup tuyệt đối với train + bench/dev → `gen_glossary_inject.py`
sinh mẫu `[term=訳語]` → binarize → train +8-12k step 18L (PLAN_V8 §T5-T7).

## Đang chờ bạn

- [ ] `pip install anthropic` + cấp `ANTHROPIC_API_KEY`
- [ ] Duyệt cột `register` (7 dòng `AUDIENCE`) trong `eval/glossary_v8_candidates.csv`
- [ ] T0: chuyển corpus 11.88M + `glossary_merged.csv` laptop→Máy A (mở khoá mine ③)
