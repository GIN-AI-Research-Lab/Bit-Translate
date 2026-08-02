# -*- coding: utf-8 -*-
"""
Exp R — QAT-LITE END-TO-END (nấc cuối của cái núm PTQ->QAT). Thiết kế cho MÁY A (3060 Ti 8GB),
chạy được cả CPU (chậm) và có --smoke để kiểm end-to-end trước khi bàn giao.

2 GIAI ĐOẠN:
  S1  Dựng lại trạng thái kỷ lục (công thức exp_q trên N3: gauge + Wanda + sequential 2-pass
      + scale/bias học được + norm co-tune + budget). GPU: ~10-15 phút.
  S2  E2E KD-STE: mở khóa TOÀN BỘ 196 wrapper (Wfp+scale+bias) + mọi norm, tối ưu
      KL(teacher_FP || student_ternary) trên text train, STE xuyên suốt. Đây là "train" thật
      nhưng nhẹ: ~1500 step x batch 8 x 128 tok ~ 1.5M token, ~30-60 phút trên 3060 Ti.
      VRAM (bf16 master + Adam bf16 + teacher bf16): ~5.5GB — vừa 8GB.

Kỳ vọng ghi trước: 400 (vi, mốc exp_q N3) -> vùng 150-300. KHÔNG hứa chạm FP 69.
Persist: lưu state_dict đã bake + kết quả (vá lỗ hổng "chưa từng lưu model").

Chạy máy A (WSL2):  python exp_r_qat_lite.py --device cuda --steps 1500 \
    --train-vi <path>/train.vi --train-ja <path>/train.ja --dev-vi <path>/dev.vi --dev-ja <path>/dev.ja
Smoke máy B:        python exp_r_qat_lite.py --smoke
"""
import argparse
import glob
import io
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
random.seed(0)

DEF_MODEL = r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*"
DEF_VI = r"D:\Bit-Translate-data\clean_v7g\train.vi"
DEF_JA = r"D:\Bit-Translate-data\clean_v7g\train.ja"
DEF_DVI = r"D:\Bit-Translate-data\clean_v7g\dev.vi"
DEF_DJA = r"D:\Bit-Translate-data\clean_v7g\dev.ja"
OUT_DIR = os.environ.get("EXPR_OUT_DIR", r"e:\Bit-Translate\eval\lowbit_ptq")

LIN_PATHS = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
             ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]

# Cổng eval đa miền (quan sát catastrophic forgetting — KHÔNG dùng để chọn best):
# bộ câu cố định để so TƯƠNG ĐỐI trước/sau train, không phải benchmark tuyệt đối.
EN_EVAL = [
    "The industrial revolution transformed the way people lived and worked across Europe.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
    "The committee announced that the annual budget would be reviewed next quarter.",
    "Despite heavy rain, the marathon continued as scheduled through the city center.",
    "Quantum computers exploit superposition to perform certain calculations faster.",
    "The novel explores themes of memory, loss, and the passage of time.",
    "Scientists discovered a new species of deep-sea fish near the volcanic vents.",
    "The central bank raised interest rates to curb rising inflation.",
    "Ancient trade routes connected distant civilizations across mountains and deserts.",
    "The software update includes security patches and performance improvements.",
    "Renewable energy sources now account for a growing share of electricity production.",
    "The museum's new exhibition features artifacts from the Bronze Age.",
]
CODE_EVAL = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    "for i in range(10):\n    if i % 2 == 0:\n        print(f'even: {i}')",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)\n    def pop(self):\n        return self.items.pop()",
    "import json\nwith open('config.json') as f:\n    config = json.load(f)\nprint(config.get('name', 'default'))",
    "const result = numbers.filter(x => x > 0).map(x => x * 2).reduce((a, b) => a + b, 0);",
    "SELECT name, COUNT(*) as total FROM orders GROUP BY name HAVING COUNT(*) > 5 ORDER BY total DESC;",
    "try:\n    value = int(user_input)\nexcept ValueError:\n    value = 0\nfinally:\n    print(value)",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    return quicksort([x for x in arr[1:] if x < pivot]) + [pivot] + quicksort([x for x in arr[1:] if x >= pivot])",
]
NORM_PATHS = [("", "input_layernorm"), ("", "post_attention_layernorm"),
              ("self_attn", "q_norm"), ("self_attn", "k_norm")]

# Calib bổ sung cho chế độ mix4 (TÁCH BIỆT với EN_EVAL/CODE_EVAL — không nhiễm gate)
CAL_EN = [
    "The weather forecast predicts light rain throughout the weekend in the northern regions.",
    "Global supply chains continue to adapt to shifting trade policies and consumer demand.",
    "The research team published their findings on protein folding in a peer-reviewed journal.",
    "Students are encouraged to submit applications before the deadline at the end of March.",
    "The city council approved funding for new public transportation infrastructure.",
    "Historical records suggest the settlement was established during the twelfth century.",
    "Engineers tested the bridge design under simulated earthquake conditions.",
    "The documentary examines the daily lives of fishermen along the coastal villages.",
    "Investors reacted cautiously to the quarterly earnings report released on Monday.",
    "The recipe calls for two cups of flour, a pinch of salt, and fresh rosemary.",
    "Volunteers spent the morning cleaning debris from the riverbank after the flood.",
    "The orchestra rehearsed the symphony's final movement late into the evening.",
    "Migration patterns of arctic birds have shifted noticeably over the past decade.",
    "The startup raised funding to expand its cloud security platform across Asia.",
    "Local farmers reported higher yields after adopting drip irrigation techniques.",
    "The novel's protagonist returns to her hometown to confront a long-buried secret.",
    "Regulators proposed new guidelines for labeling genetically modified products.",
    "The hiking trail winds through dense forest before reaching the alpine meadow.",
    "Technicians upgraded the observatory's telescope with a more sensitive detector.",
    "The museum acquired a rare collection of manuscripts from a private donor.",
    "Negotiations between the two companies stalled over intellectual property terms.",
    "The marathon route passes several historic landmarks in the old quarter.",
    "Analysts expect the housing market to stabilize by the second quarter next year.",
    "The workshop teaches participants how to restore antique wooden furniture.",
]
CAL_CODE = [
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
    "class LinkedList:\n    def __init__(self):\n        self.head = None\n    def append(self, value):\n        node = Node(value)\n        if not self.head:\n            self.head = node",
    "async function fetchUser(id) {\n  const res = await fetch(`/api/users/${id}`);\n  if (!res.ok) throw new Error('failed');\n  return res.json();\n}",
    "import os\nfor root, dirs, files in os.walk('.'):\n    for f in files:\n        if f.endswith('.log'):\n            os.remove(os.path.join(root, f))",
    "CREATE TABLE users (\n  id SERIAL PRIMARY KEY,\n  email VARCHAR(255) UNIQUE NOT NULL,\n  created_at TIMESTAMP DEFAULT NOW()\n);",
    "fn main() {\n    let numbers: Vec<i32> = (1..=10).filter(|n| n % 2 == 0).collect();\n    println!(\"{:?}\", numbers);\n}",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
    "const debounce = (fn, ms) => {\n  let timer;\n  return (...args) => {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn(...args), ms);\n  };\n};",
    "with open('data.csv') as f:\n    reader = csv.DictReader(f)\n    rows = [row for row in reader if float(row['score']) > 0.5]",
    "public static int gcd(int a, int b) {\n    while (b != 0) {\n        int t = b;\n        b = a % b;\n        a = t;\n    }\n    return a;\n}",
    "git rebase -i HEAD~3\n# pick abc123 fix parser\n# squash def456 cleanup\n# reword ghi789 update docs",
    "import numpy as np\nA = np.random.randn(64, 64)\nU, S, Vt = np.linalg.svd(A)\nrank = np.sum(S > 1e-10)",
    "SELECT u.name, COUNT(o.id) AS orders\nFROM users u LEFT JOIN orders o ON o.user_id = u.id\nWHERE o.created_at > '2025-01-01'\nGROUP BY u.name;",
    "try {\n  const data = JSON.parse(raw);\n  cache.set(key, data, { ttl: 3600 });\n} catch (e) {\n  logger.warn('invalid payload', e);\n}",
    "def flatten(nested):\n    for item in nested:\n        if isinstance(item, list):\n            yield from flatten(item)\n        else:\n            yield item",
    "#!/bin/bash\nfor f in *.png; do\n  convert \"$f\" -resize 50% \"small_$f\"\ndone",
]
# Probe zh/toán (đo độ ĐỀU đa miền — không dùng chọn best, không trùng calib):
ZH_EVAL = [
    "今天天气很好，我们去公园散步吧。",
    "这家公司的季度收入比去年同期增长了百分之十五。",
    "人工智能正在改变医疗、教育和交通等许多行业。",
    "他每天早上六点起床，先跑步半小时再吃早饭。",
    "这本小说讲述了一个家庭三代人的故事。",
    "请把这份报告翻译成英文，明天上午发给客户。",
    "长城是中国古代最伟大的建筑工程之一。",
    "手机没电了，你带充电器了吗？",
    "科学家发现了一种新的方法来降低电池的生产成本。",
    "如果明天下雨，运动会就推迟到下个星期举行。",
]
MATH_EVAL = [
    "Tom has 3 boxes with 12 apples each. He gives away 8 apples. 3 x 12 = 36, and 36 - 8 = 28. The answer is 28.",
    "Solve for x: 2x + 5 = 17. Subtract 5 from both sides: 2x = 12. Divide by 2: x = 6.",
    "The area of a rectangle is length times width. A = 7 * 4 = 28 square meters.",
    "A train travels 60 km per hour for 2.5 hours. Distance = speed * time = 60 * 2.5 = 150 km.",
    "The probability of rolling a 6 on a fair die is 1/6, so in 60 rolls we expect 60 / 6 = 10 sixes.",
    "12% of 250 is 0.12 * 250 = 30.",
    "The derivative of x^2 + 3x is 2x + 3, so at x = 2 the slope is 7.",
    "Sarah saves $15 per week. After 8 weeks she has 15 * 8 = $120, enough for the $110 bike.",
]
# Calib bổ sung cho mixw (TÁCH BIỆT probe — tránh nhiễm cổng đo):
CAL_ZH = [
    "北京是中国的首都，也是政治和文化中心。",
    "我昨天在书店买了两本关于历史的书。",
    "这个软件可以自动备份你的照片和文件。",
    "经济学家预测明年的通货膨胀率会下降。",
    "妈妈做的饺子比饭馆的好吃多了。",
    "地铁站离我家只有五分钟的路程。",
    "他大学毕业以后在一家互联网公司工作。",
    "气候变化导致极端天气越来越频繁。",
    "请在会议开始前把材料打印好。",
    "这部电影的结局出乎所有人的意料。",
    "健康的生活方式包括均衡饮食和适量运动。",
    "老师让学生们用中文写一篇关于家乡的作文。",
    "新的高铁线路把两个城市之间的时间缩短到三个小时。",
    "网上购物虽然方便，但是也要注意保护个人信息。",
    "博物馆里陈列着几千年前的青铜器和陶器。",
    "公司决定在东南亚开设三家新的分公司。",
    "冬天来了，山上的树木都被雪覆盖了。",
    "他学了五年中文，现在说得非常流利。",
    "医生建议病人多喝水，好好休息。",
    "这个问题很复杂，我们需要更多的时间讨论。",
]
CAL_MATH = [
    "A shop sells pens for $2 each. Buying 15 pens costs 15 * 2 = $30. With a $5 discount the total is $25.",
    "If 3 workers finish a job in 8 days, 6 workers finish it in 8 * 3 / 6 = 4 days.",
    "The sum of angles in a triangle is 180 degrees. If two angles are 50 and 60, the third is 70.",
    "Simplify: (2/3) + (1/6) = (4/6) + (1/6) = 5/6.",
    "A circle with radius 5 has area pi * 25, approximately 78.5.",
    "John reads 40 pages a day. A 320-page book takes 320 / 40 = 8 days.",
    "Solve: x^2 - 9 = 0, so x = 3 or x = -3.",
    "The average of 4, 8, and 12 is (4 + 8 + 12) / 3 = 8.",
    "A car uses 6 liters per 100 km. For 250 km it needs 6 * 2.5 = 15 liters.",
    "Convert 3/4 to a percentage: 3/4 = 0.75 = 75%.",
    "The next number in 2, 4, 8, 16 is 32 because each term doubles.",
    "20% off a $50 shirt saves $10, so you pay $40.",
    "If f(x) = 3x - 2 then f(4) = 10.",
    "A box holds 24 cans arranged in 4 rows, so each row has 24 / 4 = 6 cans.",
]
CAL_CHAT = [
    "<|im_start|>user\nXin chào, bạn có thể giúp tôi dịch câu này sang tiếng Nhật không?<|im_end|>\n<|im_start|>assistant\nTất nhiên rồi! Bạn hãy gửi câu cần dịch nhé.<|im_end|>",
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nThe capital of France is Paris.<|im_end|>",
    "<|im_start|>user\n明日の会議は何時からですか。<|im_end|>\n<|im_start|>assistant\n明日の会議は午前10時からです。<|im_end|>",
    "<|im_start|>user\nWrite a Python function to reverse a string.<|im_end|>\n<|im_start|>assistant\ndef reverse_string(s):\n    return s[::-1]<|im_end|>",
    "<|im_start|>user\n请推荐三本适合初学者的编程书。<|im_end|>\n<|im_start|>assistant\n我推荐《Python编程：从入门到实践》、《笨办法学Python》和《代码大全》。<|im_end|>",
    "<|im_start|>user\nExplain photosynthesis in one sentence.<|im_end|>\n<|im_start|>assistant\nPhotosynthesis is the process by which plants use sunlight to convert water and carbon dioxide into glucose and oxygen.<|im_end|>",
    "<|im_start|>user\nTính 17 nhân 23 bằng bao nhiêu?<|im_end|>\n<|im_start|>assistant\n17 nhân 23 bằng 391.<|im_end|>",
    "<|im_start|>user\nGive me a haiku about autumn.<|im_end|>\n<|im_start|>assistant\nCrimson leaves drifting /\nover the quiet river —\nautumn holds its breath.<|im_end|>",
]
HARD_BLOCKS = {2: 2.0, 26: 2.0, 27: 2.0, 0: 1.5, 1: 1.5, 3: 1.25}
EASY_RANGE = set(range(8, 21))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def read_lines(path, n, skip=0):
    out = []
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, ln in enumerate(f):
            if i < skip:
                continue
            ln = ln.strip()
            if ln:
                out.append(ln)
            if len(out) >= n:
                break
    return out


def to_f8(s):
    try:
        return s.to(torch.float8_e4m3fn).to(s.dtype)
    except Exception:
        sign = torch.sign(s)
        a = s.abs().clamp(min=2 ** -9)
        e = torch.floor(torch.log2(a))
        return sign * (2 ** e) * (torch.round(a / (2 ** e) * 8) / 8)


def to_f16(s):
    return s.to(torch.float16).to(s.dtype)


def wanda_nm_mask(W, xnorm, N, M):
    R, C = W.shape
    imp = W.abs() * xnorm[None, :].clamp(min=1e-8)
    pad = (M - C % M) % M
    A = F.pad(imp, (0, pad)) if pad else imp
    g = A.view(R, -1, M)
    kth = g.kthvalue(M - N + 1, dim=2, keepdim=True).values
    return (g >= kth).view(R, -1)[:, :C]


def inv_softplus(y):
    return y + torch.log(-torch.expm1(-y))


class LearnQLinear(nn.Module):
    def __init__(self, lin, mask, G=64, use_bias=True, sdtype="f8"):
        super().__init__()
        W0 = lin.weight.data
        self.Wfp = nn.Parameter(W0.clone())
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.zeros(W0.shape[0], dtype=W0.dtype, device=W0.device)) \
            if use_bias else None
        self.sdtype = sdtype
        R, C = W0.shape
        self.R, self.C, self.G = R, C, G
        self.pad = (G - C % G) % G
        self.register_buffer("maskf", mask.to(W0.dtype))
        with torch.no_grad():
            Wv, Mv = self._views(W0)
            Wm = Wv * Mv
            cnt = Mv.sum(2, keepdim=True).clamp(min=1)
            s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-6)
            for _ in range(3):
                t = torch.round(Wm / s).clamp(-1, 1) * Mv
                num = (Wm * t).sum(2, keepdim=True)
                den = (t * t).sum(2, keepdim=True).clamp(min=1e-8)
                s = (num / den).abs().clamp(min=1e-6)
        self.raw_s = nn.Parameter(inv_softplus(s.float()).to(W0.dtype))

    def _views(self, W):
        Wp = F.pad(W, (0, self.pad)) if self.pad else W
        Mp = F.pad(self.maskf, (0, self.pad)) if self.pad else self.maskf
        return Wp.view(self.R, -1, self.G), Mp.view(self.R, -1, self.G)

    def quant(self):
        s = F.softplus(self.raw_s).clamp(min=4e-3)
        qs = to_f16(s) if self.sdtype == "f16" else to_f8(s)
        s_q = s + (qs - s).detach()
        s_q = s_q.clamp(min=1e-6)
        Wv, Mv = self._views(self.Wfp)
        t = (torch.round((Wv * Mv) / s_q.detach()).clamp(-1, 1) * Mv).detach()
        return (t * s_q).reshape(self.R, -1)[:, :self.C]

    def forward(self, x):
        q = self.quant()
        Wq = q + self.Wfp - self.Wfp.detach()
        return F.linear(x, Wq, self.bias)


def hadamard(n, device, dtype):
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


@torch.no_grad()
def apply_gauge(model, rot=False):
    cfg = model.config
    n_q, n_kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    rep = n_q // n_kv
    for blk in model.model.layers:
        Wd = blk.mlp.down_proj.weight.data
        Wu = blk.mlp.up_proj.weight.data
        c = Wd.abs().mean(dim=0).clamp(min=1e-8)
        d = c / torch.exp(torch.log(c).mean())
        Wu.mul_(d[:, None])
        Wd.div_(d[None, :])
        Wv = blk.self_attn.v_proj.weight.data
        Wo = blk.self_attn.o_proj.weight.data
        co = Wo.abs().mean(dim=0).clamp(min=1e-8).view(n_q, hd)
        m = torch.ones(n_kv * hd, device=Wv.device, dtype=Wv.dtype)
        for kv in range(n_kv):
            qs = [kv * rep + r for r in range(rep)]
            m[kv * hd:(kv + 1) * hd] = torch.exp(sum(torch.log(co[q]) for q in qs) / rep)
        m = m / torch.exp(torch.log(m).mean())
        Wv.mul_(m[:, None])
        for kv in range(n_kv):
            for r in range(rep):
                q = kv * rep + r
                Wo[:, q * hd:(q + 1) * hd] /= m[kv * hd:(kv + 1) * hd][None, :]
        if rot:  # incoherence MIỄN PHÍ: xoay Hadamard per-head, o bù H^T -> attention bất biến
            H = hadamard(hd, Wv.device, Wv.dtype)
            for kv in range(n_kv):
                Wv[kv * hd:(kv + 1) * hd, :] = H @ Wv[kv * hd:(kv + 1) * hd, :]
                for r in range(rep):
                    q = kv * rep + r
                    Wo[:, q * hd:(q + 1) * hd] = Wo[:, q * hd:(q + 1) * hd] @ H.t()


def balanced_idx(imp, M):
    """Chia bài round-robin theo importance giảm dần vào các nhóm M liền kề
    (nhóm cuối có thể ngắn nếu C không chia hết M — khớp padding của wanda_nm_mask)."""
    C = imp.numel()
    G = (C + M - 1) // M
    cap = [M] * G
    if C % M:
        cap[-1] = C % M
    srt = imp.argsort(descending=True).tolist()
    slots = [[] for _ in range(G)]
    g = 0
    for ch in srt:
        while len(slots[g]) >= cap[g]:
            g = (g + 1) % G
        slots[g].append(ch)
        g = (g + 1) % G
    return torch.tensor([c for s in slots for c in s], dtype=torch.long, device=imp.device)


@torch.no_grad()
def apply_perm_gauge(model, xnorm, M):
    """Hoán vị kênh CHÍNH XÁC TUYỆT ĐỐI (gauge) để kênh quan trọng rải đều các nhóm M
    của mask N:M — khác rotation (bị bác): giữ nguyên thống kê từng kênh, chỉ đổi cách chia nhóm.
    3 họ: (a) hidden toàn cục -> input q/k/v/gate/up; (b) intermediate từng block -> input down;
    (c) head-dim v<->o theo kv-head -> input o. RoPE/QK-norm không bị đụng (chỉ hoán vị INPUT).
    Sửa xnorm tại chỗ cho khớp thứ tự mới."""
    cfg = model.config
    n_q, n_kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    rep = n_q // n_kv
    layers = model.model.layers
    emb = model.model.embed_tokens.weight
    # (a) hidden toàn cục: importance = tổng xnorm (chuẩn hóa per-layer) của mọi consumer
    imp = torch.zeros(cfg.hidden_size, device=emb.device)
    for b in range(len(layers)):
        for sub, name in LIN_PATHS:
            if name in ("o_proj", "down_proj"):
                continue
            xn = xnorm[f"model.layers.{b}.{sub}.{name}"]
            imp += xn / xn.mean().clamp(min=1e-8)
    pi = balanced_idx(imp, M)
    tied = model.lm_head.weight.data_ptr() == emb.data_ptr()
    emb.data = emb.data[:, pi].contiguous()
    if not tied:
        model.lm_head.weight.data = model.lm_head.weight.data[:, pi].contiguous()
    model.model.norm.weight.data = model.model.norm.weight.data[pi]
    for b, blk in enumerate(layers):
        blk.input_layernorm.weight.data = blk.input_layernorm.weight.data[pi]
        blk.post_attention_layernorm.weight.data = blk.post_attention_layernorm.weight.data[pi]
        for sub, name in LIN_PATHS:
            lin = getattr(getattr(blk, sub), name)
            key = f"model.layers.{b}.{sub}.{name}"
            if name in ("o_proj", "down_proj"):
                lin.weight.data = lin.weight.data[pi, :].contiguous()
            else:
                lin.weight.data = lin.weight.data[:, pi].contiguous()
                xnorm[key] = xnorm[key][pi]
    for b, blk in enumerate(layers):
        # (b) intermediate: gate/up hàng <-> down cột (silu elementwise -> exact)
        kd = f"model.layers.{b}.mlp.down_proj"
        pj = balanced_idx(xnorm[kd], M)
        blk.mlp.gate_proj.weight.data = blk.mlp.gate_proj.weight.data[pj, :].contiguous()
        blk.mlp.up_proj.weight.data = blk.mlp.up_proj.weight.data[pj, :].contiguous()
        blk.mlp.down_proj.weight.data = blk.mlp.down_proj.weight.data[:, pj].contiguous()
        xnorm[kd] = xnorm[kd][pj]
        # (c) v<->o theo kv-head: 1 hoán vị dùng chung cho rep q-head (attention là scalar/head
        #     nên không trộn kênh -> exact); biên nhóm thẳng hàng vì hd % M == 0 hoặc nhóm cuối ngắn
        ko = f"model.layers.{b}.self_attn.o_proj"
        xo = xnorm[ko]
        Wv = blk.self_attn.v_proj.weight.data
        Wo = blk.self_attn.o_proj.weight.data
        xo_new = xo.clone()
        for kv in range(n_kv):
            agg = sum(xo[(kv * rep + r) * hd:(kv * rep + r + 1) * hd] for r in range(rep))
            ph = balanced_idx(agg, M)
            Wv[kv * hd:(kv + 1) * hd, :] = Wv[kv * hd:(kv + 1) * hd, :][ph, :]
            for r in range(rep):
                q = kv * rep + r
                Wo[:, q * hd:(q + 1) * hd] = Wo[:, q * hd:(q + 1) * hd][:, ph]
                xo_new[q * hd:(q + 1) * hd] = xo[q * hd:(q + 1) * hd][ph]
        xnorm[ko] = xo_new


@torch.no_grad()
def eval_ppl(model, tok, lines, dev, max_tok=96):
    nll, ntok = 0.0, 0
    for s in lines:
        ids = tok(s, return_tensors="pt", truncation=True, max_length=max_tok).input_ids.to(dev)
        if ids.shape[1] < 2:
            continue
        nll += model(ids, labels=ids).loss.float().item() * (ids.shape[1] - 1)
        ntok += ids.shape[1] - 1
    return float(np.exp(nll / max(ntok, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model-glob", default=DEF_MODEL)
    ap.add_argument("--train-vi", default=DEF_VI)
    ap.add_argument("--train-ja", default=DEF_JA)
    ap.add_argument("--dev-vi", default=DEF_DVI)
    ap.add_argument("--dev-ja", default=DEF_DJA)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--calib", type=int, default=60, help="số câu cho S1 (sequential)")
    ap.add_argument("--train-sents", type=int, default=8000, help="số câu cho S2 (KD)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--train-skip", type=int, default=200_000, help="số dòng bỏ qua đầu file train")
    ap.add_argument("--freeze-scales", type=int, default=0, help="1 = đóng băng raw_s ở S2 (kiểu EfficientQAT)")
    ap.add_argument("--freeze-norms", type=int, default=0, help="1 = đóng băng norm ở S2")
    ap.add_argument("--nm-n", type=int, default=2, help="N của mask N:M (2:4=1.56bpw, 1:4=1.02, 1:8=0.70, 1:10=0.62)")
    ap.add_argument("--nm-m", type=int, default=4)
    ap.add_argument("--fast", type=int, default=0, help="1 = S2 batch thật + bf16 autocast (v3)")
    # F0 — chế độ TQ2_0-native: dense (không mask), scale f16 per-256, KHÔNG bias
    # -> baked weights nằm ĐÚNG lưới TQ2_0 -> llama-quantize TQ2_0 tái tạo lossless
    ap.add_argument("--dense", type=int, default=0, help="1 = bỏ mask N:M (dense ternary)")
    ap.add_argument("--sgroup", type=int, default=64, help="cỡ nhóm scale (64 | 256=TQ2_0-native)")
    ap.add_argument("--no-bias", type=int, default=0, help="1 = không bias (bắt buộc cho export GGUF)")
    # S1-floor attack
    ap.add_argument("--calib-mode", default="vija", choices=["vija", "mix4", "mixw"],
                    help="mix4 = vi/ja/en/code; mixw = theo căn cước Qwen3 (thêm zh/toán/chat)")
    ap.add_argument("--s1-passes", type=int, default=2, help="2=(70,30) | 3=(100,50,30)")
    ap.add_argument("--best-metric", default="geo2", choices=["geo2", "geo4"],
                    help="geo4 = chọn best theo cả 4 miền (bảo vệ en/code khi KD thuần vi/ja)")
    ap.add_argument("--rot-gauge", type=int, default=0,
                    help="1 = Hadamard-128 per-head trong gauge v<->o (incoherence MIỄN PHÍ, exact)")
    ap.add_argument("--perm-gauge", type=int, default=0,
                    help="1 = hoán vị kênh EXACT để kênh quan trọng rải đều nhóm M (cứu mask N:M)")
    ap.add_argument("--save-ckpt", type=int, default=1, help="0 = không lưu .pt (screening)")
    ap.add_argument("--tag", default="", help="nhãn riêng cho record trong results.json")
    # v4 — gói train-polish (mặc định tắt để giữ so sánh được với v3)
    ap.add_argument("--cosine", type=int, default=0, help="1 = cosine decay sau warmup")
    ap.add_argument("--kd-temp", type=float, default=1.0, help="nhiệt độ KD (2.0 = dark knowledge)")
    ap.add_argument("--ce-w", type=float, default=0.0, help="trọng số CE trộn thêm (vd 0.1)")
    ap.add_argument("--ema", type=float, default=0.0, help="decay EMA (vd 0.999); 0 = tắt")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "qat_lite_n3.pt"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.calib, args.train_sents, args.batch = 12, 64, 2
        args.steps = min(args.steps, 6)   # giữ --steps 0 (nhánh screening S1-only)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(5)
    log(f"device={dev} steps={args.steps} smoke={args.smoke}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    mdir = glob.glob(args.model_glob)[0]
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32).to(dev).eval()

    dev_vi = read_lines(args.dev_vi, 400)[-16:]
    dev_ja = read_lines(args.dev_ja, 400)[-16:]
    ppl_fp = (eval_ppl(model, tok, dev_vi, dev), eval_ppl(model, tok, dev_ja, dev))
    log(f"FP32: vi {ppl_fp[0]:.1f} / ja {ppl_fp[1]:.1f}")

    # ===== S1: gauge + sanity =====
    apply_gauge(model, rot=bool(args.rot_gauge))
    ppl_g = eval_ppl(model, tok, dev_vi, dev)
    log(f"FP sau gauge: vi {ppl_g:.1f}")
    if abs(ppl_g - ppl_fp[0]) / ppl_fp[0] > 0.01:
        log("!!! gauge sai — abort")
        sys.exit(1)

    vi_c = read_lines(args.dev_vi, args.calib // 2)
    ja_c = read_lines(args.dev_ja, args.calib // 2)
    calib = [x for pr in zip(vi_c, ja_c) for x in pr]
    if args.calib_mode == "mix4":
        calib = calib[: args.calib] + CAL_EN + CAL_CODE   # vi/ja + en + code (tách biệt gate)
        random.Random(0).shuffle(calib)
        log(f"calib mix4: {len(calib)} câu (vi/ja {min(len(calib)-40, args.calib)} + en 24 + code 16)")
    elif args.calib_mode == "mixw":
        # Theo căn cước Qwen3 (en/zh/code/math ~ mix pretraining) + mục đích dùng (vi/ja) + chat
        k = 4 if args.smoke else None
        calib = (vi_c[:(k or 20)] + ja_c[:(k or 20)] + CAL_EN[:k] + CAL_CODE[:k]
                 + CAL_ZH[:k] + CAL_MATH[:k] + CAL_CHAT[:k])
        random.Random(0).shuffle(calib)
        log(f"calib mixw (căn cước Qwen3): {len(calib)} câu"
            " (vi 20 + ja 20 + en 24 + code 16 + zh 20 + math 14 + chat 8)")
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and "layers." in n]
    xn_acc, handles = {n: None for n, _ in linears}, []
    def mk(nm_):
        def h(mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            v = x.pow(2).sum(0)
            xn_acc[nm_] = v if xn_acc[nm_] is None else xn_acc[nm_] + v
        return h
    for n, m in linears:
        handles.append(m.register_forward_pre_hook(mk(n)))
    calib_ids = []
    with torch.no_grad():
        for s in calib:
            ids = tok(s, return_tensors="pt", truncation=True, max_length=96).input_ids.to(dev)
            if ids.shape[1] >= 4:
                calib_ids.append(ids)
                model(ids)
    for h in handles:
        h.remove()
    xnorm = {n: xn_acc[n].sqrt() for n in xn_acc}
    if args.perm_gauge:
        apply_perm_gauge(model, xnorm, args.nm_m)
        ppl_p = eval_ppl(model, tok, dev_vi, dev)
        log(f"FP sau perm-gauge: vi {ppl_p:.1f}")
        if abs(ppl_p - ppl_fp[0]) / ppl_fp[0] > 0.01:
            log("!!! perm-gauge sai — abort")
            sys.exit(1)
    orig = {n: m.weight.data.clone() for n, m in linears}

    layers = model.model.layers
    NB = len(layers)
    ropes, H_fp = [], [[] for _ in range(NB + 1)]
    with torch.no_grad():
        for ids in calib_ids:
            h = model.model.embed_tokens(ids)
            pos = torch.arange(ids.shape[1], device=dev)[None]
            cos, sin = model.model.rotary_emb(h, pos)
            ropes.append((cos, sin))
            for b, blk in enumerate(layers):
                H_fp[b].append(h.clone())
                h = blk(h, position_embeddings=(cos, sin))
            H_fp[NB].append(h.clone())
    NC = len(calib_ids)
    eval_sub = list(range(0, NC, max(1, NC // 12)))[:12]

    wrapped = []
    for b, blk in enumerate(layers):
        for sub, name in LIN_PATHS:
            parent = getattr(blk, sub)
            lin = getattr(parent, name)
            key = f"model.layers.{b}.{sub}.{name}"
            m_ = (torch.ones_like(orig[key]) if args.dense
                  else wanda_nm_mask(orig[key], xnorm[key], args.nm_n, args.nm_m))
            w = LearnQLinear(lin, m_, G=args.sgroup, use_bias=not args.no_bias,
                             sdtype=("f16" if args.sgroup == 256 else "f8"))
            setattr(parent, name, w)
            wrapped.append((b, parent, name, lin, w))

    t0 = time.time()
    pass_steps = (6, 4) if args.smoke else ((100, 50, 30) if args.s1_passes >= 3 else (70, 30))
    for p_idx, base_steps in enumerate(pass_steps):
        H_q = [H_fp[0][s].clone() for s in range(NC)]
        for b, blk in enumerate(layers):
            f = HARD_BLOCKS.get(b, 0.75 if b in EASY_RANGE else 1.0)
            steps = max(4, int(round(base_steps * f)))
            mods = [w for (bb, _, _, _, w) in wrapped if bb == b]
            norm_ws = []
            for sub, name in NORM_PATHS:
                parent = getattr(blk, sub) if sub else blk
                norm_ws.append(getattr(parent, name).weight)
            for nw_ in norm_ws:
                nw_.requires_grad_(True)
            opt = torch.optim.Adam([
                {"params": [w.Wfp for w in mods], "lr": 1e-3},
                {"params": [w.raw_s for w in mods], "lr": 5e-3},
                {"params": [w.bias for w in mods if w.bias is not None], "lr": 5e-4},
                {"params": norm_ws, "lr": 5e-4},
            ])

            def eval_block():
                with torch.no_grad():
                    return sum(F.mse_loss(blk(H_q[s], position_embeddings=ropes[s]),
                                          H_fp[b + 1][s]).item() for s in eval_sub)

            best = eval_block()
            best_state = ([w.Wfp.detach().clone() for w in mods],
                          [w.raw_s.detach().clone() for w in mods],
                          [w.bias.detach().clone() for w in mods if w.bias is not None],
                          [nw_.detach().clone() for nw_ in norm_ws])
            for step in range(steps):
                idx = random.sample(range(NC), min(6, NC))
                opt.zero_grad()
                loss = sum(F.mse_loss(blk(H_q[s], position_embeddings=ropes[s]), H_fp[b + 1][s])
                           for s in idx) / len(idx)
                loss.backward()
                opt.step()
                if step % 10 == 9 or step == steps - 1:
                    v = eval_block()
                    if v < best:
                        best = v
                        best_state = ([w.Wfp.detach().clone() for w in mods],
                                      [w.raw_s.detach().clone() for w in mods],
                                      [w.bias.detach().clone() for w in mods if w.bias is not None],
                                      [nw_.detach().clone() for nw_ in norm_ws])
            with torch.no_grad():
                for w, Wb, Sb, Bb in zip(mods, best_state[0], best_state[1], best_state[2]):
                    w.Wfp.data, w.raw_s.data, w.bias.data = Wb, Sb, Bb
                for nw_, nb_ in zip(norm_ws, best_state[3]):
                    nw_.data = nb_
            for nw_ in norm_ws:
                nw_.requires_grad_(False)
            with torch.no_grad():
                for s in range(NC):
                    H_q[s] = blk(H_q[s], position_embeddings=ropes[s]).detach()
        log(f"S1 pass {p_idx+1} xong ({time.time()-t0:.0f}s)")
    del H_fp, H_q, ropes
    pv = eval_ppl(model, tok, dev_vi, dev)
    log(f"S1 (dựng lại ~exp_q): PPL vi {pv:.1f}  (mốc lab: 400.1)")

    # ===== S2: E2E KD-STE =====
    if args.steps > 0:
        log("S2: nạp teacher FP...")
        teacher = AutoModelForCausalLM.from_pretrained(
            mdir, dtype=(torch.bfloat16 if dev == "cuda" else torch.float32)).to(dev).eval()
        # teacher cùng gauge (exact). KHÔNG cần cùng perm: KD chỉ so LOGIT (bất biến hoán vị).
        apply_gauge(teacher, rot=bool(args.rot_gauge))
        for p in teacher.parameters():
            p.requires_grad_(False)
        tr_vi = read_lines(args.train_vi, args.train_sents // 2, skip=args.train_skip)
        tr_ja = read_lines(args.train_ja, args.train_sents // 2, skip=args.train_skip)
        train_txt = [x for pr in zip(tr_vi, tr_ja) for x in pr]
        log(f"S2 data: {len(train_txt)} câu")
    else:
        teacher, train_txt = None, []   # screening S1-only: khỏi tốn RAM/thời gian teacher

    params_w = [w.Wfp for (_, _, _, _, w) in wrapped]
    params_s = [w.raw_s for (_, _, _, _, w) in wrapped]
    params_b = [w.bias for (_, _, _, _, w) in wrapped if w.bias is not None]
    norm_all = []
    for b, blk in enumerate(layers):
        for sub, name in NORM_PATHS:
            parent = getattr(blk, sub) if sub else blk
            norm_all.append(getattr(parent, name).weight)
    norm_all.append(model.model.norm.weight)
    groups = [{"params": params_w, "lr": args.lr}, {"params": params_b, "lr": args.lr}]
    trainable = params_w + params_b
    if not args.freeze_scales:
        groups.append({"params": params_s, "lr": args.lr * 2})
        trainable += params_s
    if not args.freeze_norms:
        groups.append({"params": norm_all, "lr": args.lr * 0.5})
        trainable += norm_all
    for p in trainable:
        p.requires_grad_(True)
    log(f"S2 trainable: W+bias{' +scales' if not args.freeze_scales else ''}"
        f"{' +norms' if not args.freeze_norms else ''} | lr {args.lr}")
    opt = torch.optim.Adam(groups)
    # regression guard step-0: S2 không bao giờ được phép bàn giao trạng thái tệ hơn S1.
    # Best theo GEO-MEAN(vi, ja) — tránh hy sinh ja để đẹp vi (bài học run 1: ja nổ 145k mà vi hồi).
    v0 = eval_ppl(model, tok, dev_vi, dev)
    j0 = eval_ppl(model, tok, dev_ja, dev)
    e0 = eval_ppl(model, tok, EN_EVAL, dev)
    c0 = eval_ppl(model, tok, CODE_EVAL, dev, max_tok=160)
    z0 = eval_ppl(model, tok, ZH_EVAL, dev)
    m0 = eval_ppl(model, tok, MATH_EVAL, dev)
    best_score = (v0 * j0 * e0 * c0) ** 0.25 if args.best_metric == "geo4" else math.sqrt(v0 * j0)
    best_vi = v0
    best_sd = {k: t.detach().clone() for k, t in model.state_dict().items()}
    s1_probe = {"vi": v0, "ja": j0, "en": e0, "code": c0, "zh": z0, "math": m0}
    log(f"S2 step-0 (mốc S1): PPL vi {v0:.1f} / ja {j0:.1f} | en {e0:.1f} / code {c0:.1f}"
        f" | zh {z0:.1f} / math {m0:.1f}")
    nan_cnt = 0
    eval_every = max(2, args.steps // 12)
    base_lrs = [g["lr"] for g in opt.param_groups]
    WARMUP = max(1, min(100, args.steps // 10))
    ema = [p.detach().clone() for p in trainable] if args.ema > 0 else None

    def ema_swap():
        for p, e in zip(trainable, ema):
            tmp = p.data.clone()
            p.data.copy_(e)
            e.copy_(tmp)
    t1 = time.time()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_ac = args.fast and dev == "cuda"
    for step in range(args.steps):
        fac = min(1.0, (step + 1) / WARMUP)
        if args.cosine and step >= WARMUP:
            prog = (step - WARMUP) / max(1, args.steps - WARMUP)
            fac = 0.5 * (1 + math.cos(math.pi * prog))
        for g, bl in zip(opt.param_groups, base_lrs):
            g["lr"] = bl * fac
        batch = random.sample(train_txt, min(args.batch, len(train_txt)))
        opt.zero_grad()
        loss_acc = 0.0
        if args.fast:
            # v3: batch THẬT (pad + attention_mask), KL chỉ tính trên token thật, bf16 autocast
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.seq)
            ids = enc.input_ids.to(dev)
            am = enc.attention_mask.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
                with torch.no_grad():
                    tl = teacher(ids, attention_mask=am).logits
                sl = model(ids, attention_mask=am).logits
            T = args.kd_temp
            lt = F.log_softmax(tl.float() / T, -1)
            ls = F.log_softmax(sl.float() / T, -1)
            kl_tok = (lt.exp() * (lt - ls)).sum(-1)          # [B, T]
            m = am.float()
            loss = (kl_tok * m).sum() / m.sum().clamp(min=1) * (T * T)
            if args.ce_w > 0:                                 # v4: neo đuôi phân bố bằng CE thật
                m2 = (am[:, 1:] * am[:, :-1]).float()
                ce_tok = F.cross_entropy(sl.float()[:, :-1].transpose(1, 2), ids[:, 1:],
                                         reduction="none")
                loss = loss + args.ce_w * (ce_tok * m2).sum() / m2.sum().clamp(min=1)
            loss.backward()
            loss_acc = loss.item()
        else:
            for s in batch:
                ids = tok(s, return_tensors="pt", truncation=True, max_length=args.seq).input_ids.to(dev)
                if ids.shape[1] < 4:
                    continue
                with torch.no_grad():
                    tl = teacher(ids).logits.float()
                sl = model(ids).logits.float()
                # FIX 02/08: chia theo SỐ TOKEN (batchmean chia theo batch=1 -> loss to gấp ~seq_len)
                loss = F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1),
                                log_target=True, reduction="sum") / ids.shape[1]
                (loss / len(batch)).backward()
                loss_acc += loss.item() / len(batch)
        if not math.isfinite(loss_acc):
            nan_cnt += 1
            log(f"  step {step}: loss NaN/inf — bỏ step ({nan_cnt})")
            opt.zero_grad()
            if nan_cnt >= 3:
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                nan_cnt = 0
            continue
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for p, e in zip(trainable, ema):
                    e.mul_(args.ema).add_(p.data, alpha=1 - args.ema)
        if step % eval_every == eval_every - 1 or step == args.steps - 1:
            if ema is not None:
                ema_swap()                      # eval + chọn best trên trọng số EMA (mượt STE)
            v = eval_ppl(model, tok, dev_vi, dev)
            j = eval_ppl(model, tok, dev_ja, dev)
            en = eval_ppl(model, tok, EN_EVAL, dev)
            cd = eval_ppl(model, tok, CODE_EVAL, dev, max_tok=160)
            sc = (v * j * en * cd) ** 0.25 if args.best_metric == "geo4" else math.sqrt(v * j)
            log(f"  step {step+1}/{args.steps}: KL/token {loss_acc:.4f} | vi {v:.1f} / ja {j:.1f} | geo {sc:.1f} | en {en:.1f} / code {cd:.1f} ({time.time()-t1:.0f}s)")
            if sc < best_score:
                best_score = sc
                best_vi = v
                best_sd = {k: t.detach().clone() for k, t in model.state_dict().items()}
            if ema is not None:
                ema_swap()
    if best_sd is not None:
        model.load_state_dict(best_sd)

    # bake + persist + eval cuối
    with torch.no_grad():
        for b, parent, name, lin, w in wrapped:
            new_lin = nn.Linear(w.C, w.R, bias=w.use_bias).to(dev)
            new_lin.weight.data = w.quant().detach()
            if w.use_bias:
                new_lin.bias.data = w.bias.detach()
            setattr(parent, name, new_lin)
    pv, pj = eval_ppl(model, tok, dev_vi, dev), eval_ppl(model, tok, dev_ja, dev)
    pen = eval_ppl(model, tok, EN_EVAL, dev)
    pcd = eval_ppl(model, tok, CODE_EVAL, dev, max_tok=160)
    pzh = eval_ppl(model, tok, ZH_EVAL, dev)
    pma = eval_ppl(model, tok, MATH_EVAL, dev)
    big_vi = eval_ppl(model, tok, read_lines(args.dev_vi, 200)[-100:], dev)
    # bpw trung thực theo bậc (payload*keep + entropy mask + scale f8/g64) — dense: không mask
    BPW_NM = {(2, 4): 1.566, (1, 4): 1.023, (1, 8): 0.699, (1, 10): 0.616}
    sb = (16.0 if args.sgroup == 256 else 8.0) / args.sgroup
    bpw = round(math.log2(3) + sb, 3) if args.dense else BPW_NM.get((args.nm_n, args.nm_m), -1.0)
    log(f"=> EXP R QAT-lite: PPL vi {pv:.1f} / ja {pj:.1f} | en {pen:.1f} / code {pcd:.1f}"
        f" | zh {pzh:.1f} / math {pma:.1f} | validation 100 câu vi: {big_vi:.1f}")
    if args.save_ckpt:
        torch.save({"state_dict": {k: v.half().cpu() for k, v in model.state_dict().items()},
                    "meta": {"config": f"nm{args.nm_n}:{args.nm_m}+gauge+QATlite",
                             "tag": args.tag, "bpw": bpw, "calib_mode": args.calib_mode,
                             "perm_gauge": args.perm_gauge, "ppl_vi": pv, "ppl_ja": pj,
                             "ppl_vi_100c": big_vi, "steps": args.steps}}, args.out)
        log(f"Đã lưu model bake: {args.out}")
    rj = os.path.join(OUT_DIR, "exp_r_results.json")
    out = {}
    if os.path.exists(rj):
        try:
            with io.open(rj, "r", encoding="utf-8") as f:
                out = json.load(f)
        except Exception:
            out = {}
    key = args.tag if args.tag else f"exp_r[steps={args.steps},smoke={args.smoke}]"
    out[key] = {
        "bpw": bpw, "nm": f"{args.nm_n}:{args.nm_m}", "calib_mode": args.calib_mode,
        "perm_gauge": args.perm_gauge, "steps": args.steps,
        "ppl_vi": pv, "ppl_ja": pj, "ppl_en": pen, "ppl_code": pcd,
        "ppl_zh": pzh, "ppl_math": pma, "ppl_vi_100c": big_vi,
        "s1_probe": s1_probe, "fp_vi": ppl_fp[0], "fp_ja": ppl_fp[1]}
    with io.open(rj, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("=" * 72)
    print(f"EXP R — nm {args.nm_n}:{args.nm_m} ({bpw} bpw) calib={args.calib_mode}"
          f" perm={args.perm_gauge} steps={args.steps}")
    print(f"  vi {pv:9.1f}  ja {pj:9.1f}  en {pen:9.1f}  code {pcd:8.1f}"
          f"  zh {pzh:9.1f}  math {pma:8.1f} | 100c vi: {big_vi:.1f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
