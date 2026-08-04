"""SUB-BIT mới (exp_ab: rollout-KD × bit-curriculum) trên Modal — GPU L40S.

⚠️ KHÔNG TỰ CHẠY: cả 4 ví Modal đang CẠN (04/08). Mọi lệnh dưới đây chỉ chạy sau khi
user nạp credit VÀ xác nhận. File này là CHUẨN BỊ (theo RESEARCH_NOVEL_SUBBIT_PLAN.md).

Thứ tự khuyến nghị (rẻ → đắt, có cổng hủy giữa chừng):
  P0  probe_p0    (~$1)   : divergence + battery trên ckpt GEN4 CÓ SẴN trên volume
                            → nếu ckpt 1.56bpw KHÔNG loop và battery khá: hủy hết, xong.
                            → nếu loop + KL-entry cao: cơ chế xác nhận, mua tiếp arms.
  A1  arm base    (~$5)   : 2:4 thẳng 4000 bước (mốc, GEN4-class + battery đầy đủ)
  A2  arm roll    (~$8-9) : 2:4 + rollout-KD (đo RIÊNG tác dụng rollout)
      → cổng: roll thua base về battery/loop → hủy A3/A4 rollout, chỉ chạy curr.
  A3  arm curr    (~$6)   : curriculum 1:32→2:4 (đo RIÊNG tác dụng curriculum)
  A4  arm cr      (~$9-10): gộp cả hai
  Tổng trần: ~$30 + $2 gate = $32; đường lean (P0+A1+A2): ~$15.

Chạy (SAU khi user xác nhận):
  set MODAL_PROFILE=trituekstns
  python -m modal run --detach cloud/modal_qat_subbit.py::probe_p0
  python -m modal run --detach cloud/modal_qat_subbit.py::arms_all
  python -m modal run cloud/modal_qat_subbit.py::status
"""
import modal

app = modal.App("qat-subbit-ab")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==4.57.6", "numpy", "accelerate", "huggingface_hub",
                 "datasets")
    .add_local_file("eval/lowbit_ptq/exp_r_qat_lite.py", "/root/exp_r_qat_lite.py")
    .add_local_file("eval/lowbit_ptq/exp_ab_subbit_curriculum.py",
                    "/root/exp_ab_subbit_curriculum.py")
    .add_local_file("eval/lowbit_ptq/exp_ac_divergence_probe.py",
                    "/root/exp_ac_divergence_probe.py")
    .add_local_dir("cloud/qat_data", "/root/qat_data")
)

vol = modal.Volume.from_name("qat-lite-vol", create_if_missing=True)


def _prep_kd_mix():
    """Stream en/code/zh/math về /vol/kd_mix (bản sao modal_qat_lite — cache, idempotent)."""
    import io
    import os
    srcs = [
        ("en", "HuggingFaceFW/fineweb-edu", None, "text", 12_000),
        ("code", "codeparrot/codeparrot-clean", None, "content", 8_000),
        ("zh", "HuggingFaceFW/fineweb-2", "cmn_Hani", "text", 8_000),
        ("math", "open-web-math/open-web-math", None, "text", 5_000),
    ]
    os.makedirs("/vol/kd_mix", exist_ok=True)
    from datasets import load_dataset
    for name, repo, cfg, field, n in srcs:
        path = f"/vol/kd_mix/kd_{name}.txt"
        if os.path.exists(path) and os.path.getsize(path) > 100_000:
            print(f"kd_mix: {name} đã cache", flush=True)
            continue
        try:
            ds = load_dataset(repo, cfg, split="train", streaming=True)
            cnt = 0
            with io.open(path + ".tmp", "w", encoding="utf-8") as f:
                for row in ds:
                    t = (row.get(field) or "").strip()
                    if name == "code":
                        if len(t) < 200:
                            continue
                        t = t[:600].replace("\\", "\\\\").replace("\n", "\\n")
                    else:
                        t = t.replace("\n", " ")
                        if len(t) < 40:
                            continue
                        t = t[:400]
                    f.write(t + "\n")
                    cnt += 1
                    if cnt >= n:
                        break
            os.replace(path + ".tmp", path)
            print(f"kd_mix: {name} = {cnt} dòng", flush=True)
        except Exception as e:
            print(f"kd_mix: {name} LỖI {type(e).__name__}: {e}", flush=True)


ARMS = {
    # tag: (ladder, stage_steps, rollout)  — 4 arm CÙNG tổng 4000 bước, cùng recipe GEN4
    "ab_base": ("2:4", "4000", 0),
    "ab_roll": ("2:4", "4000", 1),
    "ab_curr": ("1:32,1:16,1:8,1:4,2:4", "400,400,600,800,1800", 0),
    "ab_cr":   ("1:32,1:16,1:8,1:4,2:4", "400,400,600,800,1800", 1),
}


def _run_arm(tag, mdir, highway=0):
    import subprocess
    ladder, steps, roll = ARMS[tag]
    cmd = ["python", "/root/exp_ab_subbit_curriculum.py", "--device", "cuda",
           "--model-glob", mdir,
           "--train-vi", "/root/qat_data/train_slice.vi",
           "--train-ja", "/root/qat_data/train_slice.ja",
           "--dev-vi", "/root/qat_data/dev.vi",
           "--dev-ja", "/root/qat_data/dev.ja",
           "--kd-en", "/vol/kd_mix/kd_en.txt", "--kd-code", "/vol/kd_mix/kd_code.txt",
           "--kd-zh", "/vol/kd_mix/kd_zh.txt", "--kd-math", "/vol/kd_mix/kd_math.txt",
           "--ladder", ladder, "--stage-steps", steps, "--rollout", str(roll),
           "--highway", str(highway),
           "--batch", "16", "--seq", "128", "--lr", "2e-05",
           "--kd-temp", "2.0", "--ce-w", "0.1", "--perm-gauge", "1",
           "--tag", tag, "--out", f"/vol/out/{tag}.pt"]
    print("RUN:", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=2 * 3600)
def probe_p0(ckpts: str = "qat_gen4_n4.pt,qat_gen4_o1.pt"):
    """P0 (~$1, 20-30 phút): divergence probe + behavior battery trên ckpt GEN4 CÓ SẴN.
    Cổng quyết định TRƯỚC khi mua arms:
      - self-KL entry (bucket 0-8) >> TF-KL + loop-rate cao @0.7 → cơ chế exposure/attractor
        xác nhận → mua A1/A2.
      - battery đã khá + không loop → coherence không phải nút thắt ở 0.6B@1.56 → dừng,
        xem lại mục tiêu (có thể chuyển thẳng 30B).
    """
    import os
    import subprocess
    import sys

    import torch
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/vol/hf"
    mdir = snapshot_download("Qwen/Qwen3-0.6B")
    os.makedirs("/vol/out", exist_ok=True)
    os.environ["EXPR_OUT_DIR"] = "/vol/out"
    # (a) divergence: FP control + từng ckpt
    r = subprocess.run(["python", "/root/exp_ac_divergence_probe.py", "--device", "cuda",
                        "--model-glob", mdir, "--dev-vi", "/root/qat_data/dev.vi",
                        "--arms", "fp,int4", "--tag", "p0_controls"])
    print(f"controls exit={r.returncode}", flush=True)
    for ck in ckpts.split(","):
        ck = ck.strip()
        if not os.path.exists(f"/vol/out/{ck}"):
            print(f"P0: {ck} KHÔNG có trên volume — bỏ", flush=True)
            continue
        r = subprocess.run(["python", "/root/exp_ac_divergence_probe.py", "--device", "cuda",
                            "--model-glob", mdir, "--dev-vi", "/root/qat_data/dev.vi",
                            "--arms", "ckpt", "--ckpt", f"/vol/out/{ck}",
                            "--tag", f"p0_{ck}"])
        print(f"{ck} divergence exit={r.returncode}", flush=True)
    # (b) battery trên ckpt (sinh THẬT greedy + t0.7)
    sys.path.insert(0, "/root")
    import torch.nn as nn
    from exp_ab_subbit_curriculum import behavior_battery
    from exp_r_qat_lite import LIN_PATHS
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32).to("cuda").eval()
    print("=== battery FP (mốc trần) ===", flush=True)
    behavior_battery(model, tok, "cuda")
    for blk in model.model.layers:
        for sub, name in LIN_PATHS:
            parent = getattr(blk, sub)
            lin = getattr(parent, name)
            nl = nn.Linear(lin.in_features, lin.out_features, bias=True)
            nl.weight.data = lin.weight.data.clone()
            nl.bias.data.zero_()
            setattr(parent, name, nl.to("cuda"))
    for ck in ckpts.split(","):
        ck = ck.strip()
        p = f"/vol/out/{ck}"
        if not os.path.exists(p):
            continue
        sd = torch.load(p, map_location="cpu", weights_only=False)
        state = {k: v.float() for k, v in sd["state_dict"].items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, f"unexpected: {unexpected[:5]}"
        print(f"=== battery {ck} | meta {sd.get('meta', {})} ===", flush=True)
        behavior_battery(model, tok, "cuda")
    vol.commit()
    print("P0 XONG — đọc exp_ac_results.json trên volume + log battery ở trên", flush=True)


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=6 * 3600)
def arm(tag: str = "ab_base", highway: int = 0):
    """Chạy MỘT arm (xem ARMS). ~$5 (base/curr) | ~$8-10 (roll/cr)."""
    import os

    from huggingface_hub import snapshot_download
    os.environ["HF_HOME"] = "/vol/hf"
    mdir = snapshot_download("Qwen/Qwen3-0.6B")
    os.makedirs("/vol/out", exist_ok=True)
    os.environ["EXPR_OUT_DIR"] = "/vol/out"
    _prep_kd_mix()
    rc = _run_arm(tag, mdir, highway)
    vol.commit()
    print(f"ARM {tag} exit={rc}")
    return rc


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=20 * 3600)
def arms_all(order: str = "ab_base,ab_roll,ab_curr,ab_cr", highway: int = 0):
    """Cả 4 arm TUẦN TỰ trong 1 container (bài học Bài 13: không phóng rời từ driver
    local). IDEMPOTENT: arm có tag trong exp_ab_results.json thì bỏ qua — preempt xong
    chạy lại không đốt lại cell đã xong. Tổng ~$28 ± 8."""
    import json
    import os

    from huggingface_hub import snapshot_download
    os.environ["HF_HOME"] = "/vol/hf"
    mdir = snapshot_download("Qwen/Qwen3-0.6B")
    os.makedirs("/vol/out", exist_ok=True)
    os.environ["EXPR_OUT_DIR"] = "/vol/out"
    _prep_kd_mix()
    done = {}
    rj = "/vol/out/exp_ab_results.json"
    if os.path.exists(rj):
        with open(rj, encoding="utf-8") as f:
            done = json.load(f)
    fails = []
    for tag in order.split(","):
        tag = tag.strip()
        if tag in done:
            print(f"SKIP {tag} (đã có kết quả)", flush=True)
            continue
        print(f"==== ARM {tag} ====", flush=True)
        rc = _run_arm(tag, mdir, highway)
        if rc != 0:
            fails.append(tag)
        vol.commit()
    print(f"ARMS XONG — fail: {fails if fails else 'không'}", flush=True)


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=3600)
def gate_ab(ckpt: str = "ab_cr.pt"):
    """Gate lại 1 ckpt exp_ab: battery đầy đủ + divergence (~$0.7)."""
    import os
    import subprocess
    import sys

    import torch
    import torch.nn as nn
    from huggingface_hub import snapshot_download
    os.environ["HF_HOME"] = "/vol/hf"
    mdir = snapshot_download("Qwen/Qwen3-0.6B")
    os.environ["EXPR_OUT_DIR"] = "/vol/out"
    r = subprocess.run(["python", "/root/exp_ac_divergence_probe.py", "--device", "cuda",
                        "--model-glob", mdir, "--dev-vi", "/root/qat_data/dev.vi",
                        "--arms", "ckpt", "--ckpt", f"/vol/out/{ckpt}",
                        "--tag", f"gate_{ckpt}"])
    print(f"divergence exit={r.returncode}", flush=True)
    sys.path.insert(0, "/root")
    from exp_ab_subbit_curriculum import behavior_battery
    from exp_r_qat_lite import LIN_PATHS
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32).to("cuda").eval()
    for blk in model.model.layers:
        for sub, name in LIN_PATHS:
            parent = getattr(blk, sub)
            lin = getattr(parent, name)
            nl = nn.Linear(lin.in_features, lin.out_features, bias=True)
            nl.weight.data = lin.weight.data.clone()
            nl.bias.data.zero_()
            setattr(parent, name, nl.to("cuda"))
    sd = torch.load(f"/vol/out/{ckpt}", map_location="cpu", weights_only=False)
    state = {k: v.float() for k, v in sd["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected: {unexpected[:5]}"
    print(f"GATE {ckpt} | meta {sd.get('meta', {})}", flush=True)
    behavior_battery(model, tok, "cuda", log_all=False)
    vol.commit()


@app.function(image=image, volumes={"/vol": vol}, timeout=120)
def status():
    import io as _io
    import json
    import os
    for p in ("/vol/out/exp_ab_results.json", "/vol/out/exp_ac_results.json"):
        if os.path.exists(p):
            print(f"==== {p} ====")
            with _io.open(p, "r", encoding="utf-8") as f:
                print(json.dumps(json.load(f), ensure_ascii=False, indent=2))
        else:
            print(f"{p}: chưa có")
