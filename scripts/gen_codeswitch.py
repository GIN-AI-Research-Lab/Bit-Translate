#!/usr/bin/env python3
"""Code-switching VI-EN <-> JA (sinh bằng CODE, 0 token API — chạy song song thoải mái).

Dân IT Việt trộn từ EN vào câu Việt: "deploy lên staging", "merge PR", "fix bug".
Model train trên VI sạch dễ vỡ khi gặp EN xen giữa -> data này dạy nó xử lý.
Vế JA dùng thuật ngữ katakana/tiếng Nhật tự nhiên.

Xuất: data/synthetic/codeswitch.{ja,vi,jsonl}
"""
import json
import random
from pathlib import Path

R = random.Random(20260716)
OUT = Path(__file__).parent.parent / "data" / "synthetic"

# EN (giữ trong câu Việt) <-> JA (thuật ngữ Nhật)
VERBS = [("deploy", "デプロイ"), ("merge", "マージ"), ("commit", "コミット"), ("push", "プッシュ"),
         ("review", "レビュー"), ("build", "ビルド"), ("test", "テスト"), ("release", "リリース"),
         ("backup", "バックアップ"), ("restart", "再起動"), ("debug", "デバッグ"), ("deploy lại", "再デプロイ")]
NOUNS = [("bug", "バグ"), ("server", "サーバー"), ("API", "API"), ("database", "データベース"),
         ("log", "ログ"), ("branch", "ブランチ"), ("pull request", "プルリクエスト"),
         ("source code", "ソースコード"), ("task", "タスク"), ("ticket", "チケット"),
         ("staging", "ステージング環境"), ("production", "本番環境"), ("container", "コンテナ"),
         ("unit test", "単体テスト"), ("release note", "リリースノート")]

VERB_T = [
    ("Chiều nay mình sẽ {T} cái này.", "今日の午後にこれを{J}します。"),
    ("{T} giúp mình cái này nhé.", "これを{J}してくれる?"),
    ("Khi nào {T} xong thì báo mình.", "{J}が終わったら教えてね。"),
    ("Nhớ {T} trước khi về nhé.", "帰る前に{J}するのを忘れないでね。"),
    ("Mình vừa {T} xong rồi.", "さっき{J}し終わったよ。"),
    ("Cần {T} lại cái này một lần nữa.", "これをもう一度{J}する必要があります。"),
    ("Ai {T} cái này vậy?", "誰がこれを{J}したの?"),
    ("Anh {T} lên staging giúp em với.", "ステージングに{J}していただけますか。"),
]
NOUN_T = [
    ("Cái {T} này lỗi rồi.", "この{J}がおかしいです。"),
    ("{T} bị chậm quá.", "{J}が遅すぎます。"),
    ("Kiểm tra {T} giúp mình với.", "{J}を確認してください。"),
    ("{T} đang bảo trì.", "{J}はメンテナンス中です。"),
    ("Mình không truy cập được {T}.", "{J}にアクセスできません。"),
    ("Cái {T} này ai phụ trách vậy?", "この{J}は誰が担当ですか?"),
    ("{T} này cần update lại.", "この{J}を更新する必要があります。"),
]

pairs, seen = [], set()
for pool, tmpls in ((VERBS, VERB_T), (NOUNS, NOUN_T)):
    for en, ja_t in pool:
        for vi_tmpl, ja_tmpl in tmpls:
            vi = vi_tmpl.replace("{T}", en)
            ja = ja_tmpl.replace("{J}", ja_t)
            if (ja, vi) in seen:
                continue
            seen.add((ja, vi))
            pairs.append({"ja": ja, "vi": vi, "en_term": en})

R.shuffle(pairs)
with open(OUT / "codeswitch.ja", "w", encoding="utf-8") as fj, \
     open(OUT / "codeswitch.vi", "w", encoding="utf-8") as fv, \
     open(OUT / "codeswitch.jsonl", "w", encoding="utf-8") as fl:
    for p in pairs:
        fj.write(p["ja"] + "\n")
        fv.write(p["vi"] + "\n")
        fl.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"code-switch (code): {len(pairs)} cặp -> data/synthetic/codeswitch.{{ja,vi,jsonl}}")
