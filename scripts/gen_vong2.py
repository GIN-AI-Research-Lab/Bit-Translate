#!/usr/bin/env python3
"""Vòng 2 — sinh data hội thoại/họp/keigo/công thức văn hoá (PLAN_BUOC5 §3.1).

7 mode (taxonomy từ eval/analysis_javi48_vong1.md):
  pb  phrasebook công thức văn hoá — dịch theo CHỨC NĂNG (お疲れ様≠"bạn đã mệt")
  ht  hội thoại đời thường + slang (やばい・まじ・それな…)
  hop họp/công sở — MỖI nội dung 2 biến thể register (keigo + thể thường)
  ps  persona xưng hô VI đúng quan hệ (sếp↔nhân viên, bạn thân, khách hàng…)
  dn  từ đa nghĩa tương phản (大丈夫/結構/いい… mỗi nghĩa bản dịch khác)
  pk  phủ định kép (ないことはない/なくもない… — Vòng 1 dịch NGƯỢC nghĩa)
  g2  glossary đợt 2: term dev-workflow hiếm (リベース/カナリア…)

Task = "<mode>:<idx>" hoặc "<mode>:<idx>:r2" (lặp đa dạng). GEN_TODO = file json list task
trong data/synthetic/gen/. Ghi data/synthetic/vong2/out_<task>.jsonl, resume theo file có sẵn.
Env: OPENAI_API_KEY, OPENAI_BASE_URL (mặc định Gemini), GEN_MODEL, GEN_RPM, GEN_TODO.
"""
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "synthetic" / "vong2"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("GEN_MODEL", "gemini-flash-lite-latest")
MIN_GAP = 60.0 / float(os.environ.get("GEN_RPM", "15"))
client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL",
                                        "https://generativelanguage.googleapis.com/v1beta/openai/"),
                api_key=os.environ["OPENAI_API_KEY"])

PB_FORMULAS = [
    "お疲れ様です", "お世話になっております", "よろしくお願いいたします", "恐れ入りますが",
    "お手数をおかけしますが", "ご査収ください", "取り急ぎご連絡まで", "ご確認のほどお願いします",
    "お先に失礼します", "ごちそうさまでした", "いただきます", "お邪魔します",
    "お大事に", "ご愁傷様です", "おめでとうございます", "お久しぶりです",
    "頑張ってください", "お気をつけて", "ご苦労様です", "承知いたしました",
    "かしこまりました", "少々お待ちください", "お待たせしました", "失礼いたします",
    "ご遠慮なく", "遠慮しておきます", "結構です", "大丈夫です",
    "申し訳ございません", "とんでもないです", "こちらこそ", "お陰様で",
    "せっかくですが", "あいにくですが", "差し支えなければ", "念のため",
    "ご都合はいかがでしょうか", "お時間いただけますか", "また改めてご連絡します", "ひとまず以上です",
    "引き続きよろしくお願いします", "ご協力ありがとうございました", "何卒よろしくお願い申し上げます", "ご検討ください",
    "お返事お待ちしております", "ご迷惑をおかけしました", "助かりました", "気にしないでください",
]
HT_SCENARIOS = [
    "rủ nhau đi ăn tối sau giờ làm", "hẹn giờ gặp cuối tuần rồi đổi giờ", "gọi món ở quán ăn Nhật",
    "hỏi đường đến ga tàu", "nói chuyện thời tiết nóng lạnh", "hỏi thăm sức khỏe đồng nghiệp ốm",
    "khoe mua đồ giảm giá", "than phiền tàu đông giờ cao điểm", "bàn phim/drama đang hot",
    "rủ chơi game tối nay", "kể chuyện ngủ quên trễ làm", "than deadline nhiều việc mệt",
    "hỏi mượn đồ (sạc, ô, bút)", "chia tiền bữa nhậu", "kể chuyện du lịch cuối tuần",
    "bàn chuyện chuyển nhà", "hỏi quán cà phê ngon gần công ty", "nói về sở thích chạy bộ/gym",
    "khen đồng nghiệp cắt tóc mới", "bàn kế hoạch nghỉ lễ", "than thở giá cả tăng",
    "nhờ trông hộ đồ một lát", "hỏi wifi mật khẩu quán", "kể chuyện nuôi mèo/chó",
    "rủ đi karaoke", "từ chối khéo lời rủ đi nhậu", "hỏi ý kiến chọn quà sinh nhật",
    "bàn chuyện học tiếng Nhật/tiếng Việt", "kể chuyện bị mưa ướt", "hẹn đi khám răng đổi lịch",
    "than nóng máy lạnh hỏng", "khoe ảnh đi chơi",
]
HOP_SCENARIOS = [
    "mở đầu cuộc họp định kỳ", "chốt và kết thúc cuộc họp", "xin dời lịch họp",
    "báo cáo tiến độ theo phần trăm", "xin gia hạn deadline", "xác nhận biên bản họp",
    "xin phép vắng mặt buổi họp", "đề xuất phương án mới trong họp", "phản đối ý kiến một cách lịch sự",
    "hỏi lại điểm chưa rõ trong họp", "phân công việc sau họp", "xin ý kiến phê duyệt sếp",
    "báo cáo sự cố cho cấp trên", "nhắc khéo deadline sắp tới", "cảm ơn sau khi được hỗ trợ",
    "giới thiệu thành viên mới trong họp", "chia sẻ màn hình và trình bày", "hẹn lịch họp với khách",
    "xin lỗi khách vì phản hồi chậm", "xác nhận yêu cầu của khách", "thông báo nghỉ phép",
    "bàn giao công việc trước khi nghỉ", "hỏi tình trạng xử lý ticket", "tổng kết quý và kế hoạch",
]
PS_RELATIONS = [
    ("sếp Nhật nói với nhân viên Việt trẻ", "sếp gọi 'em/cậu', xưng 'tôi/anh'"),
    ("nhân viên Việt trẻ nói với sếp Nhật", "xưng 'em', gọi 'anh/sếp' (lịch sự)"),
    ("hai đồng nghiệp ngang hàng thân nhau", "xưng 'mình/tớ-cậu' hoặc 'ông-tôi' suồng sã"),
    ("senpai nói với kouhai trong team", "xưng 'anh/chị', gọi 'em'"),
    ("kouhai nói với senpai", "xưng 'em', gọi 'anh/chị'"),
    ("hai người bạn thân từ đại học", "xưng 'tao-mày' hoặc 'tớ-cậu' tuỳ mức thân"),
    ("nhân viên nói với khách hàng", "xưng 'em/chúng tôi', gọi 'anh/chị/quý khách'"),
    ("khách hàng nói với vendor", "xưng 'tôi/bên tôi', gọi 'anh/chị/bên bạn'"),
]
DN_TRAPS = [
    "大丈夫", "結構", "いい", "やばい", "適当", "微妙", "甘い", "うるさい", "厳しい", "重い",
    "軽い", "固い", "熱い", "冷たい", "高い", "落ちる", "上がる", "回す", "切る", "通す",
    "流す", "飛ばす", "詰める", "投げる", "拾う", "叩く", "走る", "立てる", "持つ", "見る",
    "ジョブ", "タスク", "ケース", "パス", "キー", "ドライバー", "ホスト", "クライアント", "サーバー", "イメージ",
    "đá", "cơm", "được", "nhà", "ăn", "chạy", "đứng", "cứng", "mềm", "nặng",
    "sáng", "tối", "khô", "ướt", "chín", "sống", "bạc", "vàng", "đường", "chỉ",
]
PK_PATTERNS = [
    "〜ないことはない (không phải là không…)", "〜なくもない (cũng không hẳn là không…)",
    "〜ないわけではない (không có nghĩa là không…)", "〜というわけではない (không phải là…)",
    "〜しか〜ない (chỉ… mà thôi)", "〜ざるを得ない (đành phải…)",
    "〜ずにはいられない (không thể không…)", "〜とは限らない (chưa chắc đã…)",
    "〜ないとも言えない (cũng không thể nói là không…)", "〜なければならない vs 〜なくてもいい",
    "〜きれない (không… hết nổi)", "まだ〜ていない vs もう〜ない",
]
G2_TERMS = [
    ("リベース", "rebase"), ("チェリーピック", "cherry-pick"), ("ホットフィックス", "hotfix"),
    ("ロールバック", "rollback"), ("ロールアウト", "rollout"), ("カナリアリリース", "canary release"),
    ("ブルーグリーンデプロイ", "blue-green deployment"), ("フィーチャーフラグ", "feature flag"),
    ("技術的負債", "technical debt"), ("リファクタリング", "refactoring"), ("リンター", "linter"),
    ("継続的インテグレーション", "CI"), ("アーティファクト", "artifact"), ("コンテナ", "container"),
    ("オーケストレーション", "orchestration"), ("ミドルウェア", "middleware"), ("ウェブフック", "webhook"),
    ("ペイロード", "payload"), ("冪等性", "idempotency"), ("競合状態", "race condition"),
    ("デッドロック", "deadlock"), ("排他制御", "mutex/exclusive control"), ("キャッシュ無効化", "cache invalidation"),
    ("シャーディング", "sharding"), ("レプリケーション", "replication"), ("フェイルオーバー", "failover"),
    ("サーキットブレーカー", "circuit breaker"), ("レートリミット", "rate limit"), ("バックプレッシャー", "backpressure"),
    ("可観測性", "observability"), ("分散トレーシング", "distributed tracing"), ("負荷試験", "load testing"),
    ("疎通確認", "connectivity check"), ("切り戻し", "revert/rollback"), ("横展開", "horizontal rollout"),
    ("デグレ", "regression"), ("バグチケット", "bug ticket"), ("スプリント", "sprint"),
    ("ベロシティ", "velocity"), ("レトロスペクティブ", "retrospective"), ("ペアプロ", "pair programming"),
    ("モブプロ", "mob programming"), ("スタブ", "stub"), ("モック", "mock"),
    ("カバレッジ", "coverage"), ("静的解析", "static analysis"), ("脆弱性診断", "vulnerability scan"),
    ("ペネトレーションテスト", "penetration test"), ("監査ログ", "audit log"), ("権限昇格", "privilege escalation"),
]

COMMON = """CHỈ trả JSONL, mỗi dòng {{"ja":"...","vi":"..."}}. Không markdown, không giải thích, không đánh số.
Tiếng Việt TỰ NHIÊN như người Việt nói (không dịch word-by-word); tiếng Nhật bản ngữ tự nhiên."""

PROMPTS = {
    "pb": "Công thức giao tiếp Nhật: 「{seed}」. Viết 22 cặp câu: câu Nhật CHỨA công thức này trong ngữ cảnh đa dạng (email công việc, chat nội bộ, họp, nói miệng) + bản dịch Việt theo CHỨC NĂNG giao tiếp (điều người Việt sẽ NÓI trong tình huống đó, TUYỆT ĐỐI không dịch chữ). Ví dụ お疲れ様です đầu email → \"Chào anh/chị\". Phủ cả câu formula đứng một mình lẫn nằm trong câu dài.\n" + COMMON,
    "ht": "Viết 22 cặp câu hội thoại đời thường Nhật-Việt, tình huống: {seed}. Văn nói tự nhiên cả 2 bên (Nhật: thể thường/khẩu ngữ như 〜じゃん、〜っけ、まじ、やばい、それな khi hợp; Việt: khẩu ngữ trẻ tự nhiên). Câu 2 vế kiểu \"Hôm nay rảnh không? Đi ăn nhé\" càng tốt. Xen 4-6 câu có slang Nhật thông dụng.\n" + COMMON,
    "hop": "Tình huống họp/công sở: {seed}. Viết 11 NỘI DUNG, mỗi nội dung 2 biến thể tiếng Nhật: (a) keigo lịch sự (丁寧語/謙譲語 đúng chuẩn công sở) và (b) thể thường (nói với đồng nghiệp thân). Cả 2 biến thể đều kèm bản dịch Việt phù hợp register (lịch sự ↔ thân mật). Tổng 22 cặp.\n" + COMMON,
    "ps": "Hội thoại công sở IT Nhật-Việt. Quan hệ: {seed}. QUY TẮC XƯNG HÔ tiếng Việt: {rule}. Viết 22 cặp câu (lượt thoại đơn) đúng quan hệ đó — tiếng Nhật đúng register (keigo nếu nói với cấp trên/khách, thể thường nếu ngang hàng/thân), tiếng Việt xưng hô ĐÚNG QUY TẮC trên (cấm dùng 'tôi/bạn' đều tăm tắp).\n" + COMMON,
    "dn": "Từ đa nghĩa: 「{seed}」. Liệt kê các nghĩa/cách dùng khác nhau (kể cả nghĩa lóng, nghĩa trong IT nếu có), mỗi nghĩa viết 5-6 cặp câu Nhật-Việt mà bản dịch của từ này KHÁC HẲN nhau giữa các nghĩa (chứng minh ngữ cảnh đổi nghĩa). Tổng ~20 cặp.\n" + COMMON,
    "pk": "Mẫu ngữ pháp phủ định tiếng Nhật: {seed}. Viết 22 cặp câu dùng mẫu này trong ngữ cảnh đời thường + công việc IT. Bản dịch Việt phải giữ ĐÚNG chiều khẳng định/phủ định và sắc thái (đây là mẫu máy dịch hay dịch NGƯỢC nghĩa — bản dịch phải chuẩn tuyệt đối về logic).\n" + COMMON,
    "g2": "Thuật ngữ dev: {seed}. Với MỖI thuật ngữ viết 5 cặp câu Nhật-Việt ngữ cảnh làm việc thật (code review, vận hành, sự cố, daily). Giữ nguyên dạng katakana/kanji ở vế Nhật; vế Việt dùng term tiếng Anh như dân IT Việt nói (rebase, hotfix...) khi tự nhiên hơn dịch nghĩa. Tổng ~20-25 cặp.\n" + COMMON,
}


def build_prompt(mode, idx):
    if mode == "pb":
        return PROMPTS["pb"].format(seed=PB_FORMULAS[idx])
    if mode == "ht":
        return PROMPTS["ht"].format(seed=HT_SCENARIOS[idx])
    if mode == "hop":
        return PROMPTS["hop"].format(seed=HOP_SCENARIOS[idx])
    if mode == "ps":
        rel, rule = PS_RELATIONS[idx % len(PS_RELATIONS)]
        return PROMPTS["ps"].format(seed=rel, rule=rule)
    if mode == "dn":
        return PROMPTS["dn"].format(seed=DN_TRAPS[idx])
    if mode == "pk":
        return PROMPTS["pk"].format(seed=PK_PATTERNS[idx])
    if mode == "g2":
        terms = G2_TERMS[idx * 5:(idx + 1) * 5]
        seed = ", ".join(f"{ja} ({en})" for ja, en in terms)
        return PROMPTS["g2"].format(seed=seed)
    raise ValueError(mode)


def gen(task):
    parts = task.split(":")
    mode, idx = parts[0], int(parts[1])
    prompt = build_prompt(mode, idx)
    r = client.chat.completions.create(model=MODEL, temperature=0.85, max_tokens=8000,
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
                out.append({"ja": p["ja"], "vi": p["vi"], "src": f"vong2_{mode}"})
        except json.JSONDecodeError:
            pass
    return out


def main():
    todo = json.loads((ROOT / "data" / "synthetic" / "gen" / os.environ["GEN_TODO"]).read_text())
    todo = [t for t in todo if not (OUT / f"out_{t.replace(':','_')}.jsonl").exists()]
    print(f"vong2 [{MODEL}] còn {len(todo)} task", flush=True)
    last, done, total = 0.0, 0, 0
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
            total += len(pairs)
            print(f"  {t}: +{len(pairs)} ({done}/{len(todo)}, tổng {total})", flush=True)
    print(f"XONG {done} task, {total} cặp.", flush=True)


if __name__ == "__main__":
    main()
