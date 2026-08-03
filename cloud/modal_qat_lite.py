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
    .pip_install("torch", "transformers==4.57.6", "numpy", "accelerate", "huggingface_hub",
                 "datasets")
    .add_local_file("eval/lowbit_ptq/exp_r_qat_lite.py", "/root/exp_r_qat_lite.py")
    .add_local_file("eval/lowbit_ptq/exp_v_s1_stream.py", "/root/exp_v_s1_stream.py")
    .add_local_file("eval/lowbit_ptq/exp_w_lora_kd.py", "/root/exp_w_lora_kd.py")
    .add_local_dir("cloud/qat_data", "/root/qat_data")
)


def _prep_kd_mix():
    """Stream en/code/zh/math về /vol/kd_mix (MỘT lần, cache) — server-side, 0 byte 4G."""
    import io
    import os
    srcs = [
        ("en", "HuggingFaceFW/fineweb-edu", None, "text", 12_000),
        # the-stack-smol GATED; smollm python-edu chỉ có blob_id (0 dòng).
        # codeparrot-clean: ĐÃ KIỂM CHỨNG qua datasets-server — code inline field "content"
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
                    t = (row.get(field) or "").strip().replace("\n", " ")
                    if len(t) < 40:
                        continue
                    f.write(t[:400] + "\n")
                    cnt += 1
                    if cnt >= n:
                        break
            os.replace(path + ".tmp", path)
            print(f"kd_mix: {name} = {cnt} dòng", flush=True)
        except Exception as e:  # nguồn hỏng thì bỏ qua — exp_r tự dồn về vi/ja
            print(f"kd_mix: {name} LỖI {type(e).__name__}: {e}", flush=True)
    # code_ml: GIỮ newline/indent (escape \n) — calib code cho MoE cần đúng phân bố cấu trúc
    path = "/vol/kd_mix/kd_code_ml.txt"
    if not (os.path.exists(path) and os.path.getsize(path) > 100_000):
        try:
            ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
            cnt = 0
            with io.open(path + ".tmp", "w", encoding="utf-8") as f:
                for row in ds:
                    raw = (row.get("content") or "").strip()
                    if len(raw) < 200:
                        continue
                    t = raw[:600].replace("\\", "\\\\").replace("\n", "\\n")
                    f.write(t + "\n")
                    cnt += 1
                    if cnt >= 4_000:
                        break
            os.replace(path + ".tmp", path)
            print(f"kd_mix: code_ml = {cnt} mẫu (newline giữ nguyên)", flush=True)
        except Exception as e:
            print(f"kd_mix: code_ml LỖI {type(e).__name__}: {e}", flush=True)

vol = modal.Volume.from_name("qat-lite-vol", create_if_missing=True)

# Image cho pha C (đóng gói GGUF): thêm toolchain build llama-quantize
image_pack = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==4.57.6", "numpy", "accelerate", "huggingface_hub",
                 "datasets", "sentencepiece", "gguf")
    .apt_install("git", "cmake", "build-essential")
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /root/llama.cpp",
        "cmake -S /root/llama.cpp -B /root/llama.cpp/build"
        " -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release",
        "cmake --build /root/llama.cpp/build --target llama-quantize -j 8",
    )
    .add_local_file("eval/lowbit_ptq/exp_r_qat_lite.py", "/root/exp_r_qat_lite.py")
    .add_local_file("eval/lowbit_ptq/exp_v_s1_stream.py", "/root/exp_v_s1_stream.py")
    .add_local_file("eval/lowbit_ptq/exp_w_lora_kd.py", "/root/exp_w_lora_kd.py")
    .add_local_file("eval/lowbit_ptq/exp_x_pack_gguf.py", "/root/exp_x_pack_gguf.py")
    .add_local_dir("cloud/qat_data", "/root/qat_data")
)


@app.function(image=image_pack, gpu="L40S", volumes={"/vol": vol}, timeout=4 * 3600,
              memory=147_456, cpu=8)
def pack_30b():
    """Pha C — đóng gói ternary+LoRA thành GGUF TQ2_0 chạy thật (exp_x)."""
    import os
    import subprocess
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/vol/hf"
    snapshot_download("Qwen/Qwen3-30B-A3B")
    os.makedirs("/vol/out", exist_ok=True)
    os.environ["EXPR_OUT_DIR"] = "/vol/out"
    r = subprocess.run(["python", "/root/exp_x_pack_gguf.py",
                        "--dev-vi", "/root/qat_data/dev.vi",
                        "--dev-ja", "/root/qat_data/dev.ja"])
    vol.commit()
    print(f"exit={r.returncode}")
    return r.returncode


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=3 * 3600)
def train(steps: int = 1500, train_sents: int = 8000, lr: float = 2e-4,
          freeze_scales: int = 0, freeze_norms: int = 0,
          nm_n: int = 2, nm_m: int = 4, batch: int = 8, out_name: str = "qat_lite_n3.pt",
          fast: int = 0, cosine: int = 0, kd_temp: float = 1.0, ce_w: float = 0.0,
          ema: float = 0.0, dense: int = 0, sgroup: int = 64, no_bias: int = 0,
          calib_mode: str = "vija", s1_passes: int = 2, rot_gauge: int = 0,
          best_metric: str = "geo2", perm_gauge: int = 0, save_ckpt: int = 1,
          tag: str = "", awq_alpha: float = 0.0, mask_signal: str = "wanda",
          snip_beta: float = 0.5, calib_seq: int = 96, nm_profile: str = "uniform",
          lowrank: int = 0, cascade2: str = ""):
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
        "--awq-alpha", str(awq_alpha),
        "--mask-signal", mask_signal,
        "--snip-beta", str(snip_beta),
        "--calib-seq", str(calib_seq),
        "--nm-profile", nm_profile,
        "--lowrank", str(lowrank),
        "--cascade2", cascade2,
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


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=2 * 3600)
def screen_a():
    """Đợt A — screening S1 các đòn CHƯA dùng, so với mốc s1v2 (1:4 mixw+perm geo6 732;
    1:8 mixw+perm geo6 1310). 6 ô × ~4-6 phút L40S ≈ $1.2. steps=0, không ckpt."""
    import os
    import subprocess
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/vol/hf"
    mdir = snapshot_download("Qwen/Qwen3-0.6B")
    os.makedirs("/vol/out", exist_ok=True)
    os.environ["EXPR_OUT_DIR"] = "/vol/out"
    # (tag, nm_n, nm_m, calib_mode, perm, awq_alpha, mask_signal, calib_seq)
    cells = [
        ("A1-japatch[1:4]", 1, 4, "mixwj", 1, 0.0, "wanda", 96),
        ("A2-awq25[1:4]", 1, 4, "mixw", 1, 0.25, "wanda", 96),
        ("A2-awq50[1:4]", 1, 4, "mixw", 1, 0.5, "wanda", 96),
        ("A3-snip[1:4]", 1, 4, "mixw", 1, 0.0, "snip", 96),
        ("A4-big[1:4]", 1, 4, "mixwbig", 1, 0.0, "wanda", 96),
        ("A5-seq192[1:8]", 1, 8, "mixw", 1, 0.0, "wanda", 192),
    ]
    fails = []
    for tag, n, m, cm, pg, aw, ms, cs in cells:
        print(f"==== SCREEN {tag} ====", flush=True)
        cmd = ["python", "/root/exp_r_qat_lite.py", "--device", "cuda",
               "--model-glob", mdir,
               "--train-vi", "/root/qat_data/train_slice.vi",
               "--train-ja", "/root/qat_data/train_slice.ja",
               "--dev-vi", "/root/qat_data/dev.vi",
               "--dev-ja", "/root/qat_data/dev.ja",
               "--train-skip", "0", "--steps", "0", "--save-ckpt", "0",
               "--nm-n", str(n), "--nm-m", str(m), "--calib-mode", cm,
               "--perm-gauge", str(pg), "--awq-alpha", str(aw),
               "--mask-signal", ms, "--calib-seq", str(cs), "--tag", tag]
        r = subprocess.run(cmd)
        print(f"exit={r.returncode}", flush=True)
        if r.returncode != 0:
            fails.append(tag)
        vol.commit()
    print(f"SCREEN DOT-A XONG — fail: {fails if fails else 'không'}")


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=6 * 3600)
def gen4(steps: int = 5000, lowrank_sub1: int = 8):
    """GEN4 — 4 bậc TUẦN TỰ trong 1 container (app chết là chuỗi chết theo, không mồ côi):
    S1-v2 thắng cuộc (mixwj + perm) + v4 + KD-mix + chọn best geo6 + absorber cho <1bpw.
    steps 5000 (thay 6000) để vừa budget ~$5. Ckpt Ở LẠI volume (chế độ 4G)."""
    import os
    import subprocess
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/vol/hf"
    mdir = snapshot_download("Qwen/Qwen3-0.6B")
    os.makedirs("/vol/out", exist_ok=True)
    os.environ["EXPR_OUT_DIR"] = "/vol/out"
    _prep_kd_mix()
    tiers = [
        ("n4", 2, 4, 0),
        ("o1", 1, 4, 0),
        ("o2", 1, 8, lowrank_sub1),
        ("o3", 1, 10, lowrank_sub1),
    ]
    fails = []
    for name, n, m, lr_ in tiers:
        print(f"==== GEN4 {name} (nm {n}:{m}, lowrank {lr_}) ====", flush=True)
        cmd = ["python", "/root/exp_r_qat_lite.py", "--device", "cuda",
               "--model-glob", mdir,
               "--train-vi", "/root/qat_data/train_slice.vi",
               "--train-ja", "/root/qat_data/train_slice.ja",
               "--dev-vi", "/root/qat_data/dev.vi",
               "--dev-ja", "/root/qat_data/dev.ja",
               "--train-skip", "0",
               "--steps", str(steps), "--train-sents", "60000", "--lr", "2e-05",
               "--freeze-scales", "1", "--freeze-norms", "1", "--batch", "16",
               "--fast", "1", "--cosine", "1", "--kd-temp", "2.0", "--ce-w", "0.1",
               "--ema", "0.999", "--nm-n", str(n), "--nm-m", str(m),
               "--calib-mode", "mixwj", "--perm-gauge", "1", "--best-metric", "geo6",
               "--kd-mix", "1",
               "--kd-en", "/vol/kd_mix/kd_en.txt", "--kd-code", "/vol/kd_mix/kd_code.txt",
               "--kd-zh", "/vol/kd_mix/kd_zh.txt", "--kd-math", "/vol/kd_mix/kd_math.txt",
               "--lowrank", str(lr_),
               "--tag", f"gen4[{n}:{m}]", "--out", f"/vol/out/qat_gen4_{name}.pt"]
        r = subprocess.run(cmd)
        print(f"exit={r.returncode}", flush=True)
        if r.returncode != 0:
            fails.append(name)
        vol.commit()
    print(f"GEN4 XONG CA 4 BAC — fail: {fails if fails else 'không'}")


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=4 * 3600)
def gen41(steps: int = 5000):
    """GEN4.1 — rerun 2 bậc dưới-1-bit với code-KD ĐÃ SỬA (codeparrot-clean).
    Chạy trên tài khoản MỚI (tritue12/trituenguyen97) — volume tự dựng từ đầu."""
    import os
    import subprocess
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/vol/hf"
    mdir = snapshot_download("Qwen/Qwen3-0.6B")
    os.makedirs("/vol/out", exist_ok=True)
    os.environ["EXPR_OUT_DIR"] = "/vol/out"
    _prep_kd_mix()
    fails = []
    for name, n, m in (("o2", 1, 8), ("o3", 1, 10)):
        print(f"==== GEN4.1 {name} (nm {n}:{m}) ====", flush=True)
        cmd = ["python", "/root/exp_r_qat_lite.py", "--device", "cuda",
               "--model-glob", mdir,
               "--train-vi", "/root/qat_data/train_slice.vi",
               "--train-ja", "/root/qat_data/train_slice.ja",
               "--dev-vi", "/root/qat_data/dev.vi",
               "--dev-ja", "/root/qat_data/dev.ja",
               "--train-skip", "0",
               "--steps", str(steps), "--train-sents", "60000", "--lr", "2e-05",
               "--freeze-scales", "1", "--freeze-norms", "1", "--batch", "16",
               "--fast", "1", "--cosine", "1", "--kd-temp", "2.0", "--ce-w", "0.1",
               "--ema", "0.999", "--nm-n", str(n), "--nm-m", str(m),
               "--calib-mode", "mixwj", "--perm-gauge", "1", "--best-metric", "geo6",
               "--kd-mix", "1",
               "--kd-en", "/vol/kd_mix/kd_en.txt", "--kd-code", "/vol/kd_mix/kd_code.txt",
               "--kd-zh", "/vol/kd_mix/kd_zh.txt", "--kd-math", "/vol/kd_mix/kd_math.txt",
               "--lowrank", "0",
               "--tag", f"gen4.1[{n}:{m}]", "--out", f"/vol/out/qat_gen41_{name}.pt"]
        r = subprocess.run(cmd)
        print(f"exit={r.returncode}", flush=True)
        if r.returncode != 0:
            fails.append(name)
        vol.commit()
    print(f"GEN4.1 XONG — fail: {fails if fails else 'không'}")


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=5 * 3600,
              memory=147_456, cpu=8)
def s1_30b(model_id: str = "Qwen/Qwen3-30B-A3B", nm_n: int = 2, nm_m: int = 4,
           steps_block: int = 60, smoke: int = 0, tag: str = "", save: int = 1,
           cal_scale: int = 1, ja_share: int = 32, code_ml: int = 0):
    """Exp V — S1-only streaming cho 30B-A3B: model bf16 ở CPU RAM 144GB, L40S cầm từng block.
    ~2.5-3h/bậc. Ckpt bake bf16 (~61GB) Ở LẠI volume."""
    import os
    import subprocess
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/vol/hf"
    snapshot_download(model_id)
    os.makedirs("/vol/out", exist_ok=True)
    os.environ["EXPR_OUT_DIR"] = "/vol/out"
    if code_ml:
        _prep_kd_mix()   # bảo đảm kd_code_ml.txt (code giữ newline) có mặt
    tag = tag or f"expv[{model_id.split('/')[-1]},{nm_n}:{nm_m}]"
    code_path = "/vol/kd_mix/kd_code_ml.txt" if code_ml else "/vol/kd_mix/kd_code.txt"
    cmd = ["python", "/root/exp_v_s1_stream.py", "--model-id", model_id,
           "--dev-vi", "/root/qat_data/dev.vi", "--dev-ja", "/root/qat_data/dev.ja",
           "--nm-n", str(nm_n), "--nm-m", str(nm_m), "--steps-block", str(steps_block),
           "--cal-scale", str(cal_scale), "--ja-share", str(ja_share),
           "--kd-en", "/vol/kd_mix/kd_en.txt", "--kd-code", code_path,
           "--kd-zh", "/vol/kd_mix/kd_zh.txt", "--kd-math", "/vol/kd_mix/kd_math.txt",
           "--tag", tag, "--save", str(save),
           "--out", f"/vol/out/expv_{model_id.split('/')[-1]}_{nm_n}x{nm_m}.pt"]
    if smoke:
        cmd.append("--smoke")
    print("RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    vol.commit()
    print(f"exit={r.returncode}")
    return r.returncode


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=2 * 3600)
def screen_b():
    """Đợt B — phân bổ bit thông minh: absorber SVD / guard6 / 2:8 / cascade.
    Mốc so: 1:4 mixw+perm 732 geo6 @1.023 | 1:8 mixw+perm 1310 @0.699 | 1:10 1919 @0.616.
    B1b là ô đối đầu TRỰC TIẾP: 1:10+absorber @0.700 vs 1:8 thuần @0.699."""
    import os
    import subprocess
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/vol/hf"
    mdir = snapshot_download("Qwen/Qwen3-0.6B")
    os.makedirs("/vol/out", exist_ok=True)
    os.environ["EXPR_OUT_DIR"] = "/vol/out"
    # (tag, nm_n, nm_m, profile, lowrank, cascade2)
    cells = [
        ("B1a-lr8[1:8]", 1, 8, "uniform", 8, ""),
        ("B1b-lr8[1:10]", 1, 10, "uniform", 8, ""),
        ("B2-guard6", 1, 8, "guard6", 0, ""),
        ("B3a-2of8", 2, 8, "uniform", 0, ""),
        ("B3b-casc[1:8+1:32]", 1, 8, "uniform", 0, "1:32"),
    ]
    fails = []
    for tag, n, m, prof, lr_, c2 in cells:
        print(f"==== SCREEN {tag} ====", flush=True)
        cmd = ["python", "/root/exp_r_qat_lite.py", "--device", "cuda",
               "--model-glob", mdir,
               "--train-vi", "/root/qat_data/train_slice.vi",
               "--train-ja", "/root/qat_data/train_slice.ja",
               "--dev-vi", "/root/qat_data/dev.vi",
               "--dev-ja", "/root/qat_data/dev.ja",
               "--train-skip", "0", "--steps", "0", "--save-ckpt", "0",
               "--nm-n", str(n), "--nm-m", str(m), "--calib-mode", "mixw",
               "--perm-gauge", "1", "--nm-profile", prof,
               "--lowrank", str(lr_), "--cascade2", c2, "--tag", tag]
        r = subprocess.run(cmd)
        print(f"exit={r.returncode}", flush=True)
        if r.returncode != 0:
            fails.append(tag)
        vol.commit()
    print(f"SCREEN DOT-B XONG — fail: {fails if fails else 'không'}")


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=3600)
def screen_b2():
    """B1 bản SỬA (absorber đích Wfp−Q thay vì orig−Q) — 2 ô, so cặp với B1a/B1b bản lỗi."""
    import os
    import subprocess
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/vol/hf"
    mdir = snapshot_download("Qwen/Qwen3-0.6B")
    os.makedirs("/vol/out", exist_ok=True)
    os.environ["EXPR_OUT_DIR"] = "/vol/out"
    cells = [("B1a2-lr8fix[1:8]", 1, 8, 8, ""),
             ("B1b2-lr8fix[1:10]", 1, 10, 8, ""),
             ("B3b-casc[1:8+1:32]", 1, 8, 0, "1:32")]   # ô bị preempt ở screen_b, chạy bù
    for tag, n, m, lr_, c2 in cells:
        print(f"==== SCREEN {tag} ====", flush=True)
        r = subprocess.run(["python", "/root/exp_r_qat_lite.py", "--device", "cuda",
                            "--model-glob", mdir,
                            "--train-vi", "/root/qat_data/train_slice.vi",
                            "--train-ja", "/root/qat_data/train_slice.ja",
                            "--dev-vi", "/root/qat_data/dev.vi",
                            "--dev-ja", "/root/qat_data/dev.ja",
                            "--train-skip", "0", "--steps", "0", "--save-ckpt", "0",
                            "--nm-n", str(n), "--nm-m", str(m), "--calib-mode", "mixw",
                            "--perm-gauge", "1", "--lowrank", str(lr_),
                            "--cascade2", c2, "--tag", tag])
        print(f"exit={r.returncode}", flush=True)
        vol.commit()
    print("SCREEN B2 XONG")


@app.function(image=image, gpu=["H100", "A100-80GB"], volumes={"/vol": vol},
              timeout=5 * 3600, memory=112_640, cpu=8)
def lora_kd_30b(ckpt: str = "expv_Qwen3-30B-A3B_2x4.pt", steps: int = 2000,
                rank: int = 8, tag: str = "", lr: float = 3e-5,
                lora_scope: str = "attn+down",
                tlogits: str = "tlogits_30b_v2.pt"):
    """(b) — LoRA-KD trên nền ternary exp_v: pha T cache top-64 logits teacher (1 lượt A100),
    pha S train LoRA r nhỏ trên student đóng băng + grad checkpointing. ~2.5h/$8."""
    import os
    import subprocess
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/vol/hf"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # chống phân mảnh 80GB
    snapshot_download("Qwen/Qwen3-30B-A3B")
    os.makedirs("/vol/out", exist_ok=True)
    os.environ["EXPR_OUT_DIR"] = "/vol/out"
    _prep_kd_mix()
    tag = tag or f"expw[30B,r{rank},s{steps}]"
    cmd = ["python", "/root/exp_w_lora_kd.py", "--model-id", "Qwen/Qwen3-30B-A3B",
           "--ckpt", f"/vol/out/{ckpt}",
           "--train-vi", "/root/qat_data/train_slice.vi",
           "--train-ja", "/root/qat_data/train_slice.ja",
           "--dev-vi", "/root/qat_data/dev.vi", "--dev-ja", "/root/qat_data/dev.ja",
           "--kd-en", "/vol/kd_mix/kd_en.txt", "--kd-code", "/vol/kd_mix/kd_code_ml.txt",
           "--kd-zh", "/vol/kd_mix/kd_zh.txt", "--kd-math", "/vol/kd_mix/kd_math.txt",
           "--steps", str(steps), "--rank", str(rank), "--lr", str(lr),
           "--lora-scope", lora_scope,
           "--tlogits", f"/vol/out/{tlogits}",
           "--tag", tag, "--out", "/vol/out/expw_lora_30b.pt"]
    print("RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    vol.commit()
    print(f"exit={r.returncode}")
    return r.returncode


@app.function(image=image, gpu="A100-80GB", volumes={"/vol": vol}, timeout=3600,
              memory=112_640, cpu=8)
def gate_30b(ckpt: str = "expv_Qwen3-30B-A3B_2x4.pt", max_new: int = 48):
    """Gate hành vi cho 30B ternary (exp_v bake): PPL nhanh + sinh 8 prompt 6 miền."""
    import os
    import sys

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root")
    from exp_w_lora_kd import to_bias_linears

    os.environ["HF_HOME"] = "/vol/hf"
    mid = "Qwen/Qwen3-30B-A3B"
    snapshot_download(mid)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(
        mid, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    to_bias_linears(model)
    sd = torch.load(f"/vol/out/{ckpt}", map_location="cpu", weights_only=False)
    state = sd["state_dict"] if "state_dict" in sd else sd
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected: {unexpected[:5]}"
    model.to(dev)
    print(f"GATE30B {ckpt} | meta: {sd.get('meta', {})}", flush=True)
    PROMPTS = [
        ("vi", "Hà Nội là thủ đô của Việt Nam, nổi tiếng với"),
        ("vi", "Hôm nay trời mưa nên tôi quyết định"),
        ("ja", "日本の四季の中で、私が一番好きなのは"),
        ("ja", "東京駅から新幹線に乗って"),
        ("en", "The most important thing about learning a new language is"),
        ("code", "# Python function to check if a number is prime\ndef is_prime(n):\n    "),
        ("zh", "北京的秋天很美，特别是"),
        ("math", "Q: A box has 12 apples. Tom takes 5. How many are left?\nA:"),
    ]
    for dom, pr in PROMPTS:
        ids = tok(pr, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"[{dom}] -> {txt!r}", flush=True)
    print("GATE30B XONG", flush=True)


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=3600)
def gate_all(max_new: int = 48):
    """Gate hành vi CẢ 4 ckpt gen4 trong MỘT container (1 lần nạp model, 4 lần đổ state)."""
    import os
    import sys

    import torch
    import torch.nn as nn
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root")
    from exp_r_qat_lite import (CODE_EVAL, EN_EVAL, LIN_PATHS, MATH_EVAL,
                                ZH_EVAL, eval_ppl, read_lines)

    os.environ["HF_HOME"] = "/vol/hf"
    mdir = snapshot_download("Qwen/Qwen3-0.6B")
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32).to(dev).eval()
    for blk in model.model.layers:
        for sub, name in LIN_PATHS:
            parent = getattr(blk, sub)
            lin = getattr(parent, name)
            nl = nn.Linear(lin.in_features, lin.out_features, bias=True)
            nl.weight.data = lin.weight.data.clone()
            nl.bias.data.zero_()
            setattr(parent, name, nl.to(dev))
    dev_vi = read_lines("/root/qat_data/dev.vi", 400)[-16:]
    dev_ja = read_lines("/root/qat_data/dev.ja", 400)[-16:]
    PROMPTS = [
        ("vi", "Hà Nội là thủ đô của Việt Nam, nổi tiếng với"),
        ("vi", "Hôm nay trời mưa nên tôi quyết định"),
        ("ja", "日本の四季の中で、私が一番好きなのは"),
        ("ja", "東京駅から新幹線に乗って"),
        ("en", "The most important thing about learning a new language is"),
        ("code", "# Python function to check if a number is prime\ndef is_prime(n):\n    "),
        ("zh", "北京的秋天很美，特别是"),
        ("math", "Q: A box has 12 apples. Tom takes 5. How many are left?\nA:"),
    ]
    for ckpt in ("qat_gen4_n4.pt", "qat_gen4_o1.pt", "qat_gen4_o2.pt", "qat_gen4_o3.pt"):
        p = f"/vol/out/{ckpt}"
        if not os.path.exists(p):
            print(f"GATE {ckpt}: KHÔNG CÓ trên volume — bỏ", flush=True)
            continue
        sd = torch.load(p, map_location="cpu", weights_only=False)
        state = {k: v.float() for k, v in sd["state_dict"].items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        print(f"===== GATE {ckpt} | meta: {sd.get('meta', {})}", flush=True)
        with torch.no_grad():
            rows = [("vi", eval_ppl(model, tok, dev_vi, dev)),
                    ("ja", eval_ppl(model, tok, dev_ja, dev)),
                    ("en", eval_ppl(model, tok, EN_EVAL, dev)),
                    ("code", eval_ppl(model, tok, CODE_EVAL, dev, max_tok=160)),
                    ("zh", eval_ppl(model, tok, ZH_EVAL, dev)),
                    ("math", eval_ppl(model, tok, MATH_EVAL, dev))]
        print("PPL: " + " | ".join(f"{k} {v:.1f}" for k, v in rows), flush=True)
        for dom, pr in PROMPTS:
            ids = tok(pr, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            print(f"[{dom}] -> {txt!r}", flush=True)
    print("GATE_ALL XONG", flush=True)


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=1800)
def gate(ckpt: str = "qat_gen3_n3.pt", max_new: int = 48):
    """Gate hành vi NGAY TRÊN MODAL (không tải ckpt về máy — tiết kiệm 4G):
    PPL 6 miền + sinh thử mỗi miền từ ckpt trong /vol/out. Chỉ trả TEXT."""
    import os
    import sys

    import torch
    import torch.nn as nn
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root")
    from exp_r_qat_lite import (CODE_EVAL, EN_EVAL, LIN_PATHS, MATH_EVAL,
                                ZH_EVAL, eval_ppl, read_lines)

    os.environ["HF_HOME"] = "/vol/hf"
    mdir = snapshot_download("Qwen/Qwen3-0.6B")
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32).to(dev).eval()
    for blk in model.model.layers:  # ckpt bake có bias=True -> phẫu thuật Linear như exp_s
        for sub, name in LIN_PATHS:
            parent = getattr(blk, sub)
            lin = getattr(parent, name)
            nl = nn.Linear(lin.in_features, lin.out_features, bias=True)
            nl.weight.data = lin.weight.data.clone()
            nl.bias.data.zero_()
            setattr(parent, name, nl.to(dev))
    sd = torch.load(f"/vol/out/{ckpt}", map_location="cpu", weights_only=False)
    state = {k: v.float() for k, v in sd["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    print(f"GATE {ckpt} | meta: {sd.get('meta', {})}", flush=True)

    dev_vi = read_lines("/root/qat_data/dev.vi", 400)[-16:]
    dev_ja = read_lines("/root/qat_data/dev.ja", 400)[-16:]
    with torch.no_grad():
        rows = [("vi", eval_ppl(model, tok, dev_vi, dev)),
                ("ja", eval_ppl(model, tok, dev_ja, dev)),
                ("en", eval_ppl(model, tok, EN_EVAL, dev)),
                ("code", eval_ppl(model, tok, CODE_EVAL, dev, max_tok=160)),
                ("zh", eval_ppl(model, tok, ZH_EVAL, dev)),
                ("math", eval_ppl(model, tok, MATH_EVAL, dev))]
    print("PPL: " + " | ".join(f"{k} {v:.1f}" for k, v in rows), flush=True)

    PROMPTS = [
        ("vi", "Hà Nội là thủ đô của Việt Nam, nổi tiếng với"),
        ("vi", "Hôm nay trời mưa nên tôi quyết định"),
        ("ja", "日本の四季の中で、私が一番好きなのは"),
        ("ja", "東京駅から新幹線に乗って"),
        ("en", "The most important thing about learning a new language is"),
        ("code", "# Python function to check if a number is prime\ndef is_prime(n):\n    "),
        ("zh", "北京的秋天很美，特别是"),
        ("math", "Q: A box has 12 apples. Tom takes 5. How many are left?\nA:"),
    ]
    for dom, pr in PROMPTS:
        ids = tok(pr, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"[{dom}] {pr!r}\n    -> {txt!r}", flush=True)
    print("GATE XONG", flush=True)


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
