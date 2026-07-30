# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH KIỂM THỬ TRỰC TIẾP 10 CÂU TIẾNG NHẬT KHÁC NHAU VỚI BITPLANE LOCAL SERVER
Gửi trực tiếp 10 HTTP POST Requests tới http://localhost:8000/v1/chat/completions
và in ra câu trả lời thực tế do Server Bitplane Package phản hồi.
"""
import os
import sys
import time
import requests

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

SERVER_URL = "http://localhost:8000/v1/chat/completions"

JP_SENTENCES = [
    {
        "id": 1,
        "jp": "おはようございます。今日も一日頑張りましょう。",
        "mean": "Chào buổi sáng! Hôm nay chúng ta cùng cố gắng nhé."
    },
    {
        "id": 2,
        "jp": "お腹が空きました。一緒に昼ごはんを食べに行きませんか。",
        "mean": "Tôi đói rồi. Chúng ta cùng đi ăn trưa nhé?"
    },
    {
        "id": 3,
        "jp": "すみません、この近くに駅はありますか。",
        "mean": "Xin lỗi, gần đây có ga tàu nào không?"
    },
    {
        "id": 4,
        "jp": "日本語の勉強はとても面白いですが、漢字が難しいです。",
        "mean": "Học tiếng Nhật rất thú vị nhưng Hán tự (Kanji) khá khó."
    },
    {
        "id": 5,
        "jp": "週末は友達と映画を見たり、買い物に行ったりしました。",
        "mean": "Cuối tuần tôi đã đi xem phim và đi mua sắm cùng bạn bè."
    },
    {
        "id": 6,
        "jp": "来週から新しいプロジェクトが始まります。",
        "mean": "Từ tuần sau dự án mới sẽ bắt đầu."
    },
    {
        "id": 7,
        "jp": "この料理はとても美味しいですね。レシピを教えてください。",
        "mean": "Món ăn này ngon quá! Cho tôi xin công thức nhé."
    },
    {
        "id": 8,
        "jp": "明日の天気は晴れのち曇りでしょう。",
        "mean": "Thời tiết ngày mai dự báo sẽ có nắng sau đó chuyển mây."
    },
    {
        "id": 9,
        "jp": "どうぞお体に気をつけてお過ごしください。",
        "mean": "Xin giữ gìn sức khỏe cẩn thận nhé."
    },
    {
        "id": 10,
        "jp": "ありがとうございました。またお会いしましょう。",
        "mean": "Cảm ơn bạn rất nhiều! Hẹn gặp lại nhé."
    }
]

def run_10_jp_test():
    print("=" * 80)
    print("🚀 THỬ NGHIỆM TRỰC TIẾP 10 CÂU TIẾNG NHẬT KHÁC NHAU VỚI LOCAL SERVER PORT 8000")
    print("=" * 80)

    for item in JP_SENTENCES:
        prompt_text = f"Dịch câu tiếng Nhật này ra tiếng Việt: 「{item['jp']}」"
        
        payload = {
            "model": "laguna-s2.1-i158",
            "messages": [
                {"role": "user", "content": prompt_text}
            ],
            "max_tokens": 128
        }
        
        start_t = time.time()
        try:
            res = requests.post(SERVER_URL, json=payload, timeout=5)
            elapsed = time.time() - start_t
            
            if res.status_code == 200:
                answer = res.json()["choices"][0]["message"]["content"]
                print(f"📌 Câu {item['id']}: 「{item['jp']}」")
                print(f"   ⏱️ Thời gian phản hồi Server: {elapsed:.4f}s")
                print(f"   💬 Phản hồi thực tế từ Bitplane Server Port 8000:")
                print(f"   {answer}")
            else:
                print(f"❌ Câu {item['id']}: Lỗi Server Status Code {res.status_code}")
        except Exception as e:
            print(f"❌ Câu {item['id']}: Không thể kết nối Server: {e}")

        print("-" * 80)

    print("🎉 ĐÃ HOÀN THÀNH TẤT CẢ 10 CÂU TEST TRỰC TIẾP VỚI LOCAL SERVER!")
    print("=" * 80)

if __name__ == "__main__":
    run_10_jp_test()
