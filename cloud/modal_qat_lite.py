"""QAT-lite e2e (exp_r) trên Modal — GPU L40S, tài khoản tuent1997 (profile trituekstns).

Chạy:
  set MODAL_PROFILE=trituekstns
  python -m modal run --detach cloud/modal_qat_lite.py::train
  python -m modal run cloud/modal_qat_lite.py::status
Tải checkpoint về (khi xong):
  python -m modal volume get qat-lite-vol out/qat_lite_n3.pt D:\\Bit-Translate-data\\qat_lite_n3.pt

Chi phí ước: L40S ~$1.95/h; S1 ~5-8 phút + S2 1500 step ~20-35 phút => ~$1-1.5/run.
"""
import modal

app = modal.App("qat-lite-n3")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==4.57.6", "numpy", "accelerate", "huggingface_hub")
    .add_local_file("eval/lowbit_ptq/exp_r_qat_lite.py", "/root/exp_r_qat_lite.py")
    .add_local_dir("cloud/qat_data", "/root/qat_data")
)

vol = modal.Volume.from_name("qat-lite-vol", create_if_missing=True)


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=3 * 3600)
def train(steps: int = 1500, train_sents: int = 8000, lr: float = 2e-4,
          freeze_scales: int = 0, freeze_norms: int = 0,
          nm_n: int = 2, nm_m: int = 4, batch: int = 8, out_name: str = "qat_lite_n3.pt",
          fast: int = 0, cosine: int = 0, kd_temp: float = 1.0, ce_w: float = 0.0,
          ema: float = 0.0, dense: int = 0, sgroup: int = 64, no_bias: int = 0,
          calib_mode: str = "vija", s1_passes: int = 2, rot_gauge: int = 0,
          best_metric: str = "geo2", perm_gauge: int = 0, save_ckpt: int = 1,
          tag: str = ""):
    import os
    import subprocess
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/vol/hf"
    mdir = snapshot_download("Qwen/Qwen3-0.6B")
    os.makedirs("/vol/out", exist_ok=True)
    os.environ["EXPR_OUT_DIR"] = "/vol/out"
    cmd = [
        "python", "/root/exp_r_qat_lite.py",
        "--device", "cuda",
        "--model-glob", mdir,
        "--train-vi", "/root/qat_data/train_slice.vi",
        "--train-ja", "/root/qat_data/train_slice.ja",
        "--dev-vi", "/root/qat_data/dev.vi",
        "--dev-ja", "/root/qat_data/dev.ja",
        "--train-skip", "0",
        "--steps", str(steps),
        "--train-sents", str(train_sents),
        "--lr", str(lr),
        "--freeze-scales", str(freeze_scales),
        "--freeze-norms", str(freeze_norms),
        "--nm-n", str(nm_n),
        "--nm-m", str(nm_m),
        "--batch", str(batch),
        "--fast", str(fast),
        "--cosine", str(cosine),
        "--kd-temp", str(kd_temp),
        "--ce-w", str(ce_w),
        "--ema", str(ema),
        "--dense", str(dense),
        "--sgroup", str(sgroup),
        "--no-bias", str(no_bias),
        "--calib-mode", calib_mode,
        "--s1-passes", str(s1_passes),
        "--rot-gauge", str(rot_gauge),
        "--best-metric", best_metric,
        "--perm-gauge", str(perm_gauge),
        "--save-ckpt", str(save_ckpt),
        "--tag", tag,
        "--out", f"/vol/out/{out_name}",
    ]
    print("RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    vol.commit()
    print(f"exit={r.returncode}")
    return r.returncode


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=3 * 3600)
def screen_s1():
    """Screening S1-v2: sàn PTQ 4 bậc × {mix4, mixw, ±perm-gauge}, steps=0 (không KD, không ckpt).
    Full-square ở 2:4 và 1:4 (quy trách nhiệm từng mảnh); bậc sâu 1:8/1:10 chỉ mốc + full-stack.
    11 run × ~4-5 phút L40S ≈ $1.7. Kết quả dồn vào /vol/out/exp_r_results.json theo tag."""
    import os
    import subprocess
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/vol/hf"
    mdir = snapshot_download("Qwen/Qwen3-0.6B")
    os.makedirs("/vol/out", exist_ok=True)
    os.environ["EXPR_OUT_DIR"] = "/vol/out"
    cfgs = []
    for n, m in ((2, 4), (1, 4)):
        for cm, pg in (("mix4", 0), ("mixw", 0), ("mix4", 1), ("mixw", 1)):
            if (n, m) == (2, 4) and cm == "mix4" and pg == 0:
                continue  # mốc đã có từ gen3_n3 (444/2207/1448/142)
            cfgs.append((n, m, cm, pg))
    for n, m in ((1, 8), (1, 10)):
        cfgs.append((n, m, "mix4", 0))   # mốc đối chứng
        cfgs.append((n, m, "mixw", 1))   # full-stack
    fails = []
    for n, m, cm, pg in cfgs:
        tag = f"s1v2[{n}:{m},{cm},perm{pg}]"
        print(f"==== SCREEN {tag} ====", flush=True)
        cmd = ["python", "/root/exp_r_qat_lite.py", "--device", "cuda",
               "--model-glob", mdir,
               "--train-vi", "/root/qat_data/train_slice.vi",
               "--train-ja", "/root/qat_data/train_slice.ja",
               "--dev-vi", "/root/qat_data/dev.vi",
               "--dev-ja", "/root/qat_data/dev.ja",
               "--train-skip", "0", "--steps", "0", "--save-ckpt", "0",
               "--nm-n", str(n), "--nm-m", str(m),
               "--calib-mode", cm, "--perm-gauge", str(pg), "--tag", tag]
        r = subprocess.run(cmd)
        print(f"exit={r.returncode}", flush=True)
        if r.returncode != 0:
            fails.append(tag)
        vol.commit()
    print(f"SCREEN S1-V2 XONG — fail: {fails if fails else 'không'}")


@app.function(image=image, volumes={"/vol": vol}, timeout=120)
def status():
    import io as _io
    import json
    import os
    p = "/vol/out/exp_r_results.json"
    if not os.path.exists(p):
        print("chưa có kết quả")
        return
    with _io.open(p, "r", encoding="utf-8") as f:
        print(json.dumps(json.load(f), ensure_ascii=False, indent=2))
    ck = "/vol/out/qat_lite_n3.pt"
    if os.path.exists(ck):
        print(f"checkpoint: {ck} ({os.path.getsize(ck)/1024/1024:.0f} MB)")
