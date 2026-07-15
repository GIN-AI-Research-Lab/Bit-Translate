#!/usr/bin/env python3
"""Copy-through augmentation (sinh bằng CODE, 0 token LLM).

Dạy model: gặp chuỗi "lạ" (version, mã ticket, URL, tên file, commit hash, lệnh,
biến môi trường, số điện thoại...) thì GIỮ NGUYÊN VĂN khi dịch, KHÔNG phiên âm/bịa.
Trị lớp lỗi `ライブラリ→Librali`, `システム→cysteinm`.

Mỗi cặp: chuỗi {X} xuất hiện Y HỆT ở cả câu VI lẫn câu JA.
Xuất: data/synthetic/copythrough.{ja,vi,jsonl}
"""
import json
import random
import string
from pathlib import Path

R = random.Random(20260715)
OUT = Path(__file__).parent.parent / "data" / "synthetic"

# (câu VI, câu JA) — {X} là chuỗi giữ nguyên; đánh dấu loại pool phù hợp
TEMPLATES = [
    ("Hãy kiểm tra ticket {X}.", "チケット{X}を確認してください。", "ticket"),
    ("Bug {X} đã được sửa xong.", "バグ{X}は修正済みです。", "ticket"),
    ("Task {X} đã hoàn thành.", "タスク{X}が完了しました。", "ticket"),
    ("Phiên bản {X} vừa được phát hành.", "バージョン{X}がリリースされました。", "ver"),
    ("Vui lòng cập nhật lên phiên bản {X}.", "バージョン{X}にアップデートしてください。", "ver"),
    ("Lỗi này xuất hiện từ bản {X}.", "このエラーは{X}から発生しています。", "ver"),
    ("Mở file {X} lên giúp tôi.", "ファイル{X}を開いてください。", "file"),
    ("File {X} bị thiếu.", "ファイル{X}が見つかりません。", "file"),
    ("Chỉnh sửa trong {X} rồi lưu lại.", "{X}を編集して保存してください。", "file"),
    ("Lỗi xảy ra ở module {X}.", "モジュール{X}でエラーが発生しました。", "ident"),
    ("Hàm {X} trả về null.", "関数{X}がnullを返します。", "ident"),
    ("Xem tài liệu tại {X}.", "ドキュメントは{X}にあります。", "url"),
    ("Gọi API tới {X}.", "{X}のAPIを呼び出します。", "url"),
    ("Commit {X} đã được merge.", "コミット{X}がマージされました。", "hash"),
    ("Deploy bản {X} vào chiều nay.", "今日の午後に{X}をデプロイします。", "ver"),
    ("Chạy lệnh `{X}` để cài đặt.", "`{X}`コマンドを実行してインストールしてください。", "cmd"),
    ("Server {X} đang bảo trì.", "サーバー{X}はメンテナンス中です。", "host"),
    ("Pull request {X} cần được review.", "プルリクエスト{X}のレビューをお願いします。", "ticket"),
    ("Tài khoản {X} đã bị khoá.", "アカウント{X}はロックされました。", "user"),
    ("Nhánh {X} đã được tạo.", "ブランチ{X}が作成されました。", "branch"),
    ("Khởi động lại container {X}.", "コンテナ{X}を再起動してください。", "host"),
    ("Biến môi trường {X} chưa được đặt.", "環境変数{X}が設定されていません。", "env"),
    ("Kiểm tra log ở {X}.", "{X}のログを確認してください。", "file"),
    ("Số điện thoại liên hệ là {X}.", "連絡先の電話番号は{X}です。", "phone"),
    ("Mã đơn hàng {X} đã được xác nhận.", "注文番号{X}が確認されました。", "order"),
    ("Endpoint {X} trả về lỗi 500.", "エンドポイント{X}が500エラーを返します。", "path"),
    ("Cài package {X} trước đã.", "先にパッケージ{X}をインストールしてください。", "pkg"),
]


def versions(n):
    out = []
    for _ in range(n):
        s = f"v{R.randint(0,9)}.{R.randint(0,20)}.{R.randint(0,30)}"
        if R.random() < 0.3:
            s += R.choice(["-rc1", "-beta", "-alpha", ".1", "-hotfix"])
        out.append(s)
    return out


def tokens(n, kind):
    out = []
    for _ in range(n):
        if kind == "ticket":
            out.append(f"{R.choice(['JIRA','PROJ','BUG','TASK','ABC','DEV','SUP'])}-{R.randint(1,9999)}")
        elif kind == "hash":
            out.append("".join(R.choice("0123456789abcdef") for _ in range(R.choice([7, 8, 10]))))
        elif kind == "url":
            out.append(f"https://{R.choice(['example.com','api.internal','docs.company.jp','github.com/org/repo'])}/{R.choice(['api','v1','docs','users',''])}")
        elif kind == "path":
            out.append(f"/{R.choice(['api','v1','v2'])}/{R.choice(['users','login','health','orders','items'])}")
        elif kind == "file":
            out.append(R.choice(["config.yaml", "main.py", "index.html", ".env", "docker-compose.yml",
                                  "README.md", "app.js", "settings.json", "Dockerfile", "pom.xml"]))
        elif kind == "cmd":
            out.append(R.choice(["npm install", "git pull", "docker build .", "pip install -r requirements.txt",
                                 "kubectl apply -f .", "yarn dev", "make build", "mvn package"]))
        elif kind == "host":
            out.append(R.choice(["web-01", "db-prod", "redis-cache", "api-gateway", "worker-03", "nginx-lb"]))
        elif kind == "env":
            out.append(R.choice(["DATABASE_URL", "API_KEY", "NODE_ENV", "PORT", "SECRET_KEY", "REDIS_HOST"]))
        elif kind == "branch":
            out.append(f"{R.choice(['feature','hotfix','release','bugfix'])}/{R.choice(['login','crash','1.2','payment','ui'])}")
        elif kind == "user":
            out.append(R.choice(["admin", "user_042", "tanaka.t", "nguyenvan.a", "svc-bot", "guest01"]))
        elif kind == "phone":
            out.append(R.choice([f"0{R.randint(2,9)}-{R.randint(1000,9999)}-{R.randint(1000,9999)}",
                                 f"090-{R.randint(1000,9999)}-{R.randint(1000,9999)}"]))
        elif kind == "order":
            out.append(f"{R.choice(['ORD','PO','INV'])}-{R.randint(2020,2026)}-{R.randint(1,9999):04d}")
        elif kind == "ident":
            out.append(R.choice(["getUserData", "parseConfig", "handleRequest", "AuthService", "PaymentModule", "renderView"]))
        elif kind == "pkg":
            out.append(R.choice(["react", "numpy", "express", "@types/node", "requests", "lodash", "pandas"]))
        else:
            out.append("".join(R.choice(string.ascii_letters + string.digits) for _ in range(R.randint(4, 10))))
    return out


PER = 300   # số biến thể mỗi template
pairs, seen = [], set()
for vi_t, ja_t, kind in TEMPLATES:
    xs = versions(PER) if kind == "ver" else tokens(PER, kind)
    for x in xs:
        vi, ja = vi_t.replace("{X}", x), ja_t.replace("{X}", x)
        if (ja, vi) in seen:
            continue
        seen.add((ja, vi))
        pairs.append({"ja": ja, "vi": vi, "keep": x})

R.shuffle(pairs)
with open(OUT / "copythrough.ja", "w", encoding="utf-8") as fj, \
     open(OUT / "copythrough.vi", "w", encoding="utf-8") as fv, \
     open(OUT / "copythrough.jsonl", "w", encoding="utf-8") as fl:
    for p in pairs:
        fj.write(p["ja"] + "\n")
        fv.write(p["vi"] + "\n")
        fl.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"copy-through: {len(pairs)} cặp -> data/synthetic/copythrough.{{ja,vi,jsonl}}")
