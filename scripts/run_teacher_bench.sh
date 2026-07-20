#!/usr/bin/env bash
# Tuyển THẦY cho KD (PLAN_KD_JA2VI Đợt 1): chạy hardbench200 qua các ứng viên
# free-quota song song. Key đọc từ .env. Resume an toàn — chạy lại là tiếp.
#   bash scripts/run_teacher_bench.sh
# Log: eval/tb_<model>.log ; kết quả: eval/hardbench_<model>.jsonl
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
PY=${PY:-$HOME/vija-venv/bin/python}

run() { # base key model rpm
  OPENAI_BASE_URL=$1 OPENAI_API_KEY=$2 MODEL=$3 LABEL=$3 RPM=$4 \
    "$PY" eval/run_hardbench_api.py > "eval/tb_$3.log" 2>&1 &
}

DS=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
GM=https://generativelanguage.googleapis.com/v1beta/openai
run "$DS" "$DASHSCOPE_API_KEY" qwen-max   30
run "$DS" "$DASHSCOPE_API_KEY" qwen-plus  30
run "$DS" "$DASHSCOPE_API_KEY" qwen-turbo 30
run "$DS" "$DASHSCOPE_API_KEY" qwen-flash 30
run https://open.bigmodel.cn/api/paas/v4 "$ZHIPU_API_KEY" glm-4.7-flash 40
run "$GM" "$gemini_key_1" gemini-2.5-flash 8
run "$GM" "$gemini_key_1" gemini-flash-lite-latest 10
wait

echo "=== chrF TB (so mốc: google ja2vi 36.1 / haiku 34.6 trên cùng đề) ==="
"$PY" - <<'EOF'
import json, glob
for f in sorted(glob.glob("eval/hardbench_*.jsonl")):
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    by = {}
    for r in rows:
        by.setdefault(r["dir"], []).append(r["chrf"])
    s = "  ".join(f"{d}: {sum(v)/len(v):5.1f} (n={len(v)})" for d, v in sorted(by.items()))
    print(f"{f.split('hardbench_')[1][:-6]:28} {s}")
EOF
