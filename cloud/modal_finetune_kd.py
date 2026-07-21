"""Clean-finetune thử nghiệm — kiểm tra giả thuyết "KD bị pha loãng" (2026-07-21).

Bối cảnh: v3/v4 trộn KD sạch vào base 11,5M câu -> KD chỉ chiếm 0,84% mix, tín
hiệu gần như mất. Judge v3: acc ja→vi 2,08 (hardbench) / 2,45 (FLORES đơn giản)
— thua 292M (2,42) và thua xa Google (4,05/4,83). Rerank probe xác nhận chất
lượng KHÔNG giấu trong weights (oracle chỉ +5,2 chrF) -> phải đổi cách NẠP
kiến thức, không phải cách lấy ra.

Thí nghiệm này: finetune NGẮN từ checkpoint v4 trên mix mà KD chiếm trọng số
thật: 120k base replay (chống catastrophic forgetting) + 32.390 KD x2 (35%) +
2.721 fix x2 (2,9%) = 190.222 seq, 7,2M token (`bin_ja2vi_ft.tar.gz`).
+300 step (15500->15800) ≈ 5,5 epoch trên mix này — mỗi câu KD được thấy ~11
lần. RỦI RO OVERFIT CAO CHỦ ĐÍCH (bài học wave0): vì vậy milestone mỗi 100
step, PHẢI judge từng milestone (15600/15700/15800) rồi mới chọn, KHÔNG lấy
mù bản cuối. LR thấp 3e-5 -> 1e-5, warmup 20.

Đọc kết quả: judge milestone tốt nhất so v4-final —
  - Nhảy rõ (>= +0.3 acc): giả thuyết pha loãng ĐÚNG -> công thức = pretrain
    to + clean-finetune; lúc đó mới đáng bàn mua/sinh thêm KD quy mô lớn.
  - Không nhích/tụt: 100M không hấp thụ nổi kể cả khi KD đậm đặc -> dừng đổ
    công vào data cho 100M, chuyển câu hỏi sang 292M / glossary-inference.

Chuẩn bị 1 lần (SAU KHI v4 xong + backup final lên autosave-100m-v4):
  python3 -m modal run cloud/modal_finetune_kd.py::setup
Train:
  python3 -m modal run --detach cloud/modal_finetune_kd.py::train
  python3 -m modal run cloud/modal_finetune_kd.py::status
"""
import modal

app = modal.App("vija-100m-ft")
REPO = "trituenguyen97/Bit-Translate"
PREMIX_TAG = "train-assets-ja2vi-100m"
V4_BACKUP_TAG = "autosave-100m-v4"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "zstd")
    .run_commands(
        "curl -fsSL https://github.com/cli/cli/releases/download/v2.63.0/gh_2.63.0_linux_amd64.tar.gz "
        "| tar xz -C /tmp && cp /tmp/gh_2.63.0_linux_amd64/bin/gh /usr/local/bin/gh")
    .pip_install("torch", "numpy<2", "sentencepiece", "sacrebleu")
    .add_local_dir("src", "/root/bt/src")
    .add_local_dir("scripts", "/root/bt/scripts")
    .add_local_dir("tokenizer", "/root/bt/tokenizer")
)

vol = modal.Volume.from_name("vija-100m-ft-vol", create_if_missing=True)
gh = modal.Secret.from_name("github-token")

DIMS = "--d-model 768 --n-layers 12 --n-heads 12 --d-ff 2048 --vocab-size 32001"
# save-every 50 + milestone mỗi 100: run ngắn, phải soi được từng chặng.
TRAIN_ARGS = ("--max-tokens 8192 --grad-accum 16 --compile --max-seq 384 --pad-multiple 32 "
              "--max-steps 15800 --lr-anchor 15500 --lr 3e-5 --min-lr 1e-5 --warmup 20 "
              "--save-every 50 --milestone-every 100 --log-every 10")


@app.function(image=image, volumes={"/persist": vol}, secrets=[gh], timeout=3600)
def setup():
    """Tải mix finetune + ghép checkpoint cuối của v4 làm seed (idempotent)."""
    import os
    import subprocess

    def sh(c):
        print(">>", c, flush=True)
        subprocess.run(c, shell=True, check=True)

    os.makedirs("/persist/bin", exist_ok=True)
    os.makedirs("/persist/checkpoints", exist_ok=True)
    env = f'GH_TOKEN={os.environ["GH_TOKEN"]}'

    if not os.path.exists("/persist/bin/train.tokens.u16"):
        sh(f"cd /tmp && {env} gh release download {PREMIX_TAG} -R {REPO} "
           f"--pattern bin_ja2vi_ft.tar.gz --clobber")
        sh("tar -C /persist -xzf /tmp/bin_ja2vi_ft.tar.gz && rm /tmp/bin_ja2vi_ft.tar.gz")
        print("data OK:", os.listdir("/persist/bin"), flush=True)
    else:
        print("data đã có, bỏ qua.", flush=True)

    if not os.path.exists("/persist/checkpoints/last.pt"):
        sh(f"cd /tmp && {env} gh release download {V4_BACKUP_TAG} -R {REPO} --pattern 'v4_a*' --clobber")
        sh("cd /tmp && cat v4_a* > /persist/checkpoints/last.pt && rm -f v4_a*")
        sz = os.path.getsize("/persist/checkpoints/last.pt")
        print(f"checkpoint seed (v4 step15500) OK: {sz//2**20}MB", flush=True)
    else:
        print("checkpoint đã có, bỏ qua.", flush=True)

    vol.commit()
    print("SETUP XONG — Volume sẵn sàng cho ::train.", flush=True)


@app.function(image=image, volumes={"/persist": vol}, secrets=[gh], gpu="L40S",
              cpu=8.0, memory=16384, timeout=3 * 3600)
def train():
    """Resume checkpoint v4 (step15500) -> 15800, mix KD đậm đặc 35%."""
    import os
    import subprocess
    import threading
    import time

    assert os.path.exists("/persist/checkpoints/last.pt"), "chưa setup! chạy ::setup trước."
    assert os.path.exists("/persist/bin/train.tokens.u16"), "chưa setup! chạy ::setup trước."
    os.makedirs("/root/bt/data", exist_ok=True)
    for src, dst in [("/persist/bin", "/root/bt/data/bin"),
                     ("/persist/checkpoints", "/root/bt/checkpoints")]:
        if os.path.islink(dst) or os.path.exists(dst):
            subprocess.run(["rm", "-rf", dst])
        os.symlink(src, dst)

    env = dict(os.environ)
    env["BITNET_OPT"] = "off"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = f"cd /root/bt && python3 scripts/train.py {DIMS} {TRAIN_ARGS}"
    print(">> FINETUNE 100M (KD 35%):", cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)

    def backup_release(tag):
        if not os.path.exists("/persist/checkpoints/last.pt"):
            return False
        try:
            subprocess.run("cd /persist/checkpoints && rm -f ft_a? && split -b 1900m last.pt ft_",
                           shell=True, check=True, timeout=600)
            gh_env = f'GH_TOKEN={os.environ["GH_TOKEN"]}'
            r = subprocess.run(
                f"cd /persist/checkpoints && {gh_env} gh release upload autosave-100m-ft "
                "-R " + REPO + " --clobber $(ls ft_a?)",
                shell=True, timeout=900, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"[{tag}] backup OK", flush=True)
                return True
            if "release not found" in (r.stderr or "").lower():
                r2 = subprocess.run(
                    f"cd /persist/checkpoints && {gh_env} gh release create autosave-100m-ft "
                    "$(ls ft_a?) -R " + REPO + " --title '100M clean-finetune autosave' "
                    "--notes 'auto backup'",
                    shell=True, timeout=900, capture_output=True, text=True)
                if r2.returncode == 0:
                    print(f"[{tag}] backup release CREATED", flush=True)
                    return True
                print(f"[{tag}] create fail: {r2.stderr[-300:]}", flush=True)
            print(f"[{tag}] backup FAIL: {r.stderr[-300:]}", flush=True)
            return False
        except Exception as e:
            print(f"[{tag}] backup lỗi: {e}", flush=True)
            return False
        finally:
            subprocess.run("rm -f /persist/checkpoints/ft_a?", shell=True)

    stop = threading.Event()

    def guardian():
        last_up = 0
        while not stop.wait(300):
            try:
                vol.commit()
                print("[guardian] Volume committed", flush=True)
            except Exception as e:
                print("[guardian] commit lỗi:", e, flush=True)
            if time.time() - last_up > 600:
                last_up = time.time()
                backup_release("guardian")

    threading.Thread(target=guardian, daemon=True).start()
    rc = proc.wait()
    stop.set()
    vol.commit()
    print(f"FINETUNE kết thúc rc={rc}. Volume đã commit.", flush=True)

    ok = False
    for attempt in range(1, 4):
        ok = backup_release("final")
        if ok:
            break
        print(f"[final] lần {attempt}/3 fail, chờ 30s...", flush=True)
        time.sleep(30)
    if not ok:
        print("[final] CẢNH BÁO: backup fail — checkpoint vẫn trong Volume, "
              "gọi lại ::train để backup lại.", flush=True)

    log = "/persist/checkpoints/train.log"
    if os.path.exists(log):
        tail = subprocess.run(["tail", "-5", log], capture_output=True, text=True).stdout
        print("train.log:\n" + tail, flush=True)


@app.function(image=image, volumes={"/persist": vol}, timeout=120)
def status():
    import os
    import subprocess
    vol.reload()
    log = "/persist/checkpoints/train.log"
    print("==== 100M clean-finetune (KD 35%) 15500 -> 15800 ====")
    if os.path.exists(log):
        out = subprocess.run(f"grep -E '^step ' {log} | tail -8", shell=True,
                             capture_output=True, text=True).stdout.strip()
        print(out or "(chưa có dòng step — compile warmup)")
    else:
        print("(chưa có train.log)")
    ms = [s for s in range(15600, 15801, 100)
          if os.path.exists(f"/persist/checkpoints/step{s}.pt")]
    print(f"milestone có: {ms}")


@app.local_entrypoint()
def main():
    print("modal run cloud/modal_finetune_kd.py::setup  (1 lần, SAU KHI v4 xong)  "
          "->  --detach ::train  ->  ::status")
