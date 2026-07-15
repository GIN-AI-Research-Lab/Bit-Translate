#!/usr/bin/env python3
"""Phase 2: LLM code-switching + synthetic 指摘事項-style Q&A (khách Nhật <-> dev Việt).
Sinh HỢP PHÁP từ kịch bản IT (KHÔNG dùng file mật). OpenAI-compatible (Gemini 2-key).

Task = "<mode>:<seed_idx>", mode in {cs, qa}. Đọc GEN_TODO (json list task).
Ghi: data/synthetic/phase2/out_<mode>_<idx>.jsonl  (JSONL {ja,vi})
Merge dùng chung merge_glossary_sents.py? KHÔNG — file này gộp riêng (xem cuối).
"""
import json
import os
import re
import time
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "synthetic" / "phase2"
OUT.mkdir(exist_ok=True)
MODEL = os.environ.get("GEN_MODEL", "gemini-flash-lite-latest")
MIN_GAP = 60.0 / float(os.environ.get("GEN_RPM", "15"))
client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
                api_key=os.environ["OPENAI_API_KEY"])

SEEDS = [
    "bug đăng nhập trên môi trường staging", "database quá tải làm hệ thống chậm",
    "thay đổi spec (仕様変更) giữa dự án", "deadline (納期) bị trễ, xin gia hạn",
    "deploy fail phải rollback", "kết quả unit test không đạt", "feedback trong code review",
    "API trả lỗi 500", "xin quyền truy cập môi trường production", "kế hoạch release phiên bản mới",
    "dời lịch họp với khách", "báo cáo tiến độ tuần", "lỗi hiển thị ở giao diện",
    "xung đột khi merge nhánh", "điều tra nguyên nhân sự cố (障害)",
    "khách hỏi về estimate (見積) tính năng mới", "xác nhận nội dung 基本設計",
    "báo lỗi tồn đọng (課題) chưa xử lý", "kiểm thử tích hợp (結合テスト) phát sinh lỗi",
    "khách yêu cầu thêm chức năng ngoài spec",
    "báo cáo daily standup", "review pull request bị conflict", "hỏi cách cấu hình môi trường",
    "khách phàn nàn performance chậm", "giải thích lý do chậm tiến độ", "đề xuất phương án kỹ thuật",
    "xác nhận môi trường test với khách", "báo lỗi security cần vá gấp", "hướng dẫn khách cách reproduce bug",
    "thảo luận cách thiết kế database", "xin xác nhận trước khi deploy production", "báo cáo kết quả kiểm thử",
    "trao đổi về refactor code cũ", "hỏi về tài liệu 詳細設計 còn thiếu", "xử lý ticket khách báo",
    "onboarding thành viên mới vào dự án", "thảo luận version library cần nâng cấp",
    "báo cáo incident môi trường production", "xin thêm thời gian cho task khó", "confirm bàn giao (検収) cuối dự án",
]

CS_PROMPT = """Sinh 18 cặp câu DATA dịch máy Việt-Nhật, chủ đề: {seed}.
Câu tiếng Việt phải theo văn phong DÂN IT VIỆT trộn từ tiếng Anh tự nhiên (deploy, merge, commit, bug, review, log, server, API, staging, production, pull request...). Câu tiếng Nhật là bản dịch tự nhiên, thuật ngữ katakana chuẩn.
CHỈ trả JSONL, mỗi dòng {{"ja":"...","vi":"..."}}. Không markdown, không giải thích."""

QA_PROMPT = """Viết một đoạn HỘI THOẠI công việc IT giữa KHÁCH HÀNG NHẬT và KỸ SƯ/BrSE VIỆT, chủ đề: {seed}.
~16 lượt thoại xen kẽ. Mỗi lượt: câu gốc (khách nói tiếng Nhật lịch sự/keigo; dev đáp tiếng Nhật) + bản dịch tiếng Việt chuẩn văn phong công sở (xưng hô anh/chị hợp lý). Đúng thuật ngữ nghiệp vụ (指摘/対応/納期/仕様変更/障害...).
CHỈ trả JSONL, mỗi dòng {{"ja":"<câu tiếng Nhật>","vi":"<bản dịch tiếng Việt>"}}. Không markdown, không giải thích."""


def gen(task):
    parts = task.split(":")
    mode, idx = parts[0], parts[1]   # task: "cs:3" hoặc "cs:3:r2" (lặp để đa dạng)
    seed = SEEDS[int(idx)]
    prompt = (CS_PROMPT if mode == "cs" else QA_PROMPT).format(seed=seed)
    r = client.chat.completions.create(model=MODEL, temperature=0.8, max_tokens=8000,
                                       messages=[{"role": "user", "content": prompt}])
    txt = re.sub(r"^```[a-z]*\n?|```$", "", (r.choices[0].message.content or "").strip(), flags=re.M)
    out = []
    for line in txt.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            p = json.loads(line)
            if p.get("ja") and p.get("vi"):
                out.append({"ja": p["ja"], "vi": p["vi"], "src": f"phase2_{mode}"})
        except json.JSONDecodeError:
            pass
    return out


def main():
    todo = json.loads((ROOT / "data" / "synthetic" / "gen" / os.environ["GEN_TODO"]).read_text())
    todo = [t for t in todo if not (OUT / f"out_{t.replace(':','_')}.jsonl").exists()]
    print(f"phase2 [{MODEL}] còn {len(todo)} task", flush=True)
    last, done = 0.0, 0
    for t in todo:
        gap = MIN_GAP - (time.time() - last)
        if gap > 0:
            time.sleep(gap)
        last = time.time()
        try:
            pairs = gen(t)
        except Exception as e:  # noqa: BLE001
            print(f"  {t}: LỖI {str(e)[:70]}", flush=True)
            continue
        if pairs:
            (OUT / f"out_{t.replace(':','_')}.jsonl").write_text(
                "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
            done += 1
            print(f"  {t}: +{len(pairs)} cặp ({done}/{len(todo)})", flush=True)
    print(f"XONG {done} task.", flush=True)


if __name__ == "__main__":
    main()
