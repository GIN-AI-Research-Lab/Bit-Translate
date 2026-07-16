#!/usr/bin/env bash
# Chạy sinh data trên BẤT KỲ provider nào có key trong .env — không phụ thuộc Gemini.
#
#   bash scripts/gen_any.sh <provider> <todo.json> [script]
#   ví dụ: bash scripts/gen_any.sh dashscope vong2_r2_key1.json
#          bash scripts/gen_any.sh zhipu     vong2_id_key1.json scripts/gen_vong2.py
#
# provider: gemini1 gemini2 dashscope zhipu siliconflow deepseek openrouter
# Ghi đè model/RPM: GEN_MODEL=... GEN_RPM=... bash scripts/gen_any.sh ...
# Mọi script gen (gen_vong2.py, gen_via_api.py, gen_idiom_harvest.py) đều đọc
# OPENAI_BASE_URL / OPENAI_API_KEY / GEN_MODEL / GEN_RPM / GEN_TODO — chuẩn OpenAI-compat.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROVIDER=${1:?provider}; TODO=${2:?todo.json}; SCRIPT=${3:-scripts/gen_vong2.py}
PY=${PY:-$HOME/vija-venv/bin/python}
getkey(){ grep "^$1=" .env | cut -d= -f2- | tr -d "\r\n "; }

case "$PROVIDER" in
  gemini1)     KEY=$(getkey gemini_key_1); BASE="https://generativelanguage.googleapis.com/v1beta/openai/"
               MODEL=${GEN_MODEL:-gemini-flash-lite-latest}; RPM=${GEN_RPM:-14} ;;
  gemini2)     KEY=$(getkey gemini_key_2); BASE="https://generativelanguage.googleapis.com/v1beta/openai/"
               MODEL=${GEN_MODEL:-gemini-flash-lite-latest}; RPM=${GEN_RPM:-14} ;;
  # DashScope QUỐC TẾ: 1M in + 1M out / MỖI model (90 ngày) -> đổi GEN_MODEL là được quota mới:
  # qwen-plus, qwen-turbo, qwen-max, qwen-flash... JA-VI thuộc top tốt nhất.
  dashscope)   KEY=$(getkey DASHSCOPE_API_KEY); BASE="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
               MODEL=${GEN_MODEL:-qwen-plus}; RPM=${GEN_RPM:-30} ;;
  # Zhipu: GLM-4.7-Flash free vĩnh viễn, ~1 req/s -> RPM 50 an toàn.
  zhipu)       KEY=$(getkey ZHIPU_API_KEY); BASE="https://open.bigmodel.cn/api/paas/v4"
               MODEL=${GEN_MODEL:-glm-4.7-flash}; RPM=${GEN_RPM:-50} ;;
  # SiliconFlow: always-free Qwen3-8B (30 RPM) + ¥14 credit cho model lớn hơn.
  siliconflow) KEY=$(getkey SILICONFLOW_API_KEY); BASE="https://api.siliconflow.com/v1"
               MODEL=${GEN_MODEL:-Qwen/Qwen3-8B}; RPM=${GEN_RPM:-25} ;;
  # DeepSeek: 5M token 30 ngày. deepseek-chat bị khai tử 24/07/2026 -> dùng ID V4.
  deepseek)    KEY=$(getkey DEEPSEEK_API_KEY); BASE="https://api.deepseek.com"
               MODEL=${GEN_MODEL:-deepseek-v4-flash}; RPM=${GEN_RPM:-60} ;;
  openrouter)  KEY=$(getkey OPENROUTER_API_KEY); BASE="https://openrouter.ai/api/v1"
               MODEL=${GEN_MODEL:-qwen/qwen3-coder:free}; RPM=${GEN_RPM:-15} ;;
  *) echo "provider lạ: $PROVIDER"; exit 1 ;;
esac
[ -n "$KEY" ] || { echo "!! Chưa có key cho $PROVIDER trong .env — đăng ký rồi điền vào."; exit 1; }

echo "[gen_any] $PROVIDER | model=$MODEL | rpm=$RPM | todo=$TODO | script=$SCRIPT"
OPENAI_API_KEY=$KEY OPENAI_BASE_URL=$BASE GEN_MODEL=$MODEL GEN_RPM=$RPM GEN_TODO=$TODO \
GEN_REASONING=${GEN_REASONING:-} exec $PY -u "$SCRIPT"
