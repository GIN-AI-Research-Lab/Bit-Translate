#!/usr/bin/env python3
"""Bổ sung glossary BrSE/Comtor (quy trình phát triển + quản lý dự án IT Nhật-Việt).
Đây là từ vựng NGHỀ THẬT mà MS Terminology (UI phần mềm) hay thiếu:
工程 phát triển, quản lý dự án, xử lý lỗi, kiểm thử...

Append vào data/glossary/glossary_merged.csv (source=brse), khử trùng theo (ja,vi).
"""
import csv
from pathlib import Path

G = Path(__file__).parent.parent / "data" / "glossary" / "glossary_merged.csv"

# ja, vi, en
BRSE = [
    # --- Quy trình phát triển (開発工程) ---
    ("要件定義", "định nghĩa yêu cầu", "requirements definition"),
    ("基本設計", "thiết kế cơ bản", "basic design"),
    ("詳細設計", "thiết kế chi tiết", "detailed design"),
    ("製造", "lập trình", "implementation"),
    ("単体テスト", "kiểm thử đơn vị", "unit test"),
    ("結合テスト", "kiểm thử tích hợp", "integration test"),
    ("総合テスト", "kiểm thử tổng hợp", "system test"),
    ("システムテスト", "kiểm thử hệ thống", "system test"),
    ("受入テスト", "kiểm thử nghiệm thu", "acceptance test"),
    ("運用", "vận hành", "operation"),
    ("保守", "bảo trì", "maintenance"),
    ("本番環境", "môi trường sản xuất", "production environment"),
    ("検証環境", "môi trường kiểm thử", "staging environment"),
    ("ステージング環境", "môi trường staging", "staging environment"),
    ("開発環境", "môi trường phát triển", "development environment"),
    # --- Quản lý dự án (プロジェクト管理) ---
    ("仕様変更", "thay đổi đặc tả", "specification change"),
    ("仕様書", "tài liệu đặc tả", "specification document"),
    ("要件", "yêu cầu", "requirement"),
    ("課題", "vấn đề tồn đọng", "issue"),
    ("障害", "sự cố", "incident"),
    ("不具合", "lỗi", "defect"),
    ("指摘", "chỉ ra lỗi", "point out"),
    ("指摘事項", "hạng mục chỉ ra", "pointed-out items"),
    ("対応", "xử lý", "handling"),
    ("対応中", "đang xử lý", "in progress"),
    ("未対応", "chưa xử lý", "not handled"),
    ("対応済み", "đã xử lý", "handled"),
    ("納期", "thời hạn giao hàng", "deadline"),
    ("納品", "bàn giao", "delivery"),
    ("検収", "nghiệm thu", "acceptance inspection"),
    ("工数", "khối lượng công việc", "man-hours"),
    ("見積", "ước lượng", "estimate"),
    ("見積書", "bảng báo giá", "quotation"),
    ("進捗", "tiến độ", "progress"),
    ("進捗報告", "báo cáo tiến độ", "progress report"),
    ("議事録", "biên bản họp", "meeting minutes"),
    ("打ち合わせ", "buổi họp", "meeting"),
    ("定例会", "họp định kỳ", "regular meeting"),
    ("承認", "phê duyệt", "approval"),
    ("レビュー", "rà soát", "review"),
    ("コードレビュー", "rà soát mã", "code review"),
    ("リスク", "rủi ro", "risk"),
    ("スケジュール", "lịch trình", "schedule"),
    ("マイルストーン", "cột mốc", "milestone"),
    ("要員", "nhân sự", "personnel"),
    ("体制", "cơ cấu tổ chức", "team structure"),
    ("稼働", "công suất làm việc", "workload"),
    ("案件", "dự án", "project"),
    # --- Vai trò ---
    ("ブリッジエンジニア", "kỹ sư cầu nối", "bridge engineer"),
    ("コムトール", "phiên dịch IT", "IT interpreter"),
    # --- Kỹ thuật thường gặp ---
    ("負荷", "tải", "load"),
    ("過負荷", "quá tải", "overload"),
    ("ソースコード", "mã nguồn", "source code"),
    ("リストア", "khôi phục", "restore"),
    ("権限", "quyền", "permission"),
    ("認証", "xác thực", "authentication"),
    ("仕様", "đặc tả", "specification"),
    ("帳票", "báo biểu", "report form"),
    ("項目", "trường dữ liệu", "field"),
    ("再現", "tái hiện", "reproduce"),
    ("原因", "nguyên nhân", "cause"),
    ("影響範囲", "phạm vi ảnh hưởng", "impact scope"),
    ("暫定対応", "xử lý tạm thời", "temporary fix"),
    ("恒久対応", "xử lý triệt để", "permanent fix"),
    ("再発防止", "ngăn tái phát", "recurrence prevention"),
    ("エビデンス", "bằng chứng kiểm thử", "test evidence"),
    ("テストケース", "ca kiểm thử", "test case"),
    ("テスト仕様書", "tài liệu kiểm thử", "test specification"),
    ("デグレード", "lỗi hồi quy", "degrade"),
    ("リグレッションテスト", "kiểm thử hồi quy", "regression test"),
    ("リファクタリング", "tái cấu trúc mã", "refactoring"),
    ("手順書", "tài liệu quy trình", "procedure manual"),
    ("運用手順", "quy trình vận hành", "operation procedure"),
    ("要件漏れ", "sót yêu cầu", "missing requirement"),
    ("横展開", "áp dụng mở rộng", "horizontal deployment"),
    ("切り戻し", "khôi phục lại bản cũ", "rollback"),
    ("リリース手順書", "tài liệu quy trình phát hành", "release procedure"),
    ("QA", "đảm bảo chất lượng", "quality assurance"),
    ("フィードバック", "phản hồi", "feedback"),
    ("エスカレーション", "báo cáo lên cấp trên", "escalation"),
    ("優先度", "độ ưu tiên", "priority"),
    ("重要度", "mức độ quan trọng", "severity"),
]

rows = list(csv.reader(G.open(encoding="utf-8-sig")))
hdr, rows = rows[0], rows[1:]
seen = {(r[0], r[1]) for r in rows}
added = 0
for ja, vi, en in BRSE:
    if (ja, vi) not in seen:
        rows.append([ja, vi, en, "brse"])
        seen.add((ja, vi))
        added += 1
with G.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    w.writerows(rows)
print(f"BrSE: +{added} cặp mới (bỏ {len(BRSE)-added} đã trùng) | tổng glossary: {len(rows)}")
