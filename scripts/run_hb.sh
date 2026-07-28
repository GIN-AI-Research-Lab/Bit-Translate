#!/bin/bash
cd /mnt/e/Bit-Translate
pkill -f llama-server 2>/dev/null
sleep 1
/home/tuent/BitNet-test/build/bin/llama-server -m dist/kd100m_i2s.gguf --host 127.0.0.1 --port 8080 -t 6 -c 320 --parallel 1 >/tmp/kdsrv.log 2>&1 &
SRV=$!
for i in $(seq 1 45); do
  if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then echo "SERVER_UP_${i}s"; sleep 2; break; fi
  sleep 1
done
python3 scripts/wsl_hb_client.py eval/hardbench_ja2vi.jsonl eval/hardbench_kd100m_i2s_hyps.jsonl
echo CLIENT_DONE
kill $SRV 2>/dev/null
tail -6 /tmp/kdsrv.log
