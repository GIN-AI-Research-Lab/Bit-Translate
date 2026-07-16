#!/usr/bin/env bash
# Demo UI dịch re-translate VI⇄JA (BitNet 1.58-bit, CPU).
# Chạy trong WSL:  bash demo/run_demo.sh [model.gguf]
# Rồi mở trình duyệt Windows:  http://localhost:8765
set -u
MODEL="${1:-$HOME/vija-test/vija19000_i2s.gguf}"
SERVER_BIN="${LLAMA_SERVER:-$HOME/BitNet-test/build/bin/llama-server}"
THREADS="${THREADS:-6}"
cd "$(dirname "$0")"

[ -f "$MODEL" ]      || { echo "Không thấy model: $MODEL"; exit 1; }
[ -x "$SERVER_BIN" ] || { echo "Không thấy llama-server: $SERVER_BIN (build target llama-server trước)"; exit 1; }

pkill -f "llama-server .*--port 8080" 2>/dev/null && sleep 1
"$SERVER_BIN" -m "$MODEL" --host 127.0.0.1 --port 8080 -t "$THREADS" -c 256 --parallel 1 \
  > /tmp/llama-server.log 2>&1 &
echo "llama-server PID $! (log: /tmp/llama-server.log)"

for i in $(seq 1 60); do
  curl -s http://127.0.0.1:8080/health >/dev/null 2>&1 && break
  sleep 0.5
done
curl -s http://127.0.0.1:8080/health >/dev/null 2>&1 || { echo "llama-server không lên, xem /tmp/llama-server.log"; exit 1; }
echo "llama-server sẵn sàng."

exec python3 server.py --port 8765
