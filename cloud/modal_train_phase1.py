"""Train Phase 1 (292M, vòng 3a) trên Modal — GPU A10 24GB, bf16, compile.

Chuẩn bị 1 lần:
  1. Tạo secret GH token:  TK=$(gh auth token); .venv/bin/python -m modal secret create github-token GH_TOKEN=$TK
  2. Nạp Volume (tải data vòng 3a + checkpoint step25000 — ~4GB, chạy 1 lần):
       .venv/bin/python -m modal run cloud/modal_train_phase1.py::setup
Train (resume step25000 -> +5000 step trên vòng 3a; gọi lại nếu timeout, tự resume):
       .venv/bin/python -m modal run cloud/modal_train_phase1.py::train
Xem log/tiến độ realtime ở dashboard modal.com hoặc terminal.

Cơ chế chống mất: train.py lưu checkpoints/last.pt mỗi 100 step -> Volume; hàm này
commit Volume mỗi ~5' + upload last.pt lên release autosave-scale300m mỗi ~10'.
Modal container tắt/timeout -> gọi lại train, tự resume từ Volume (không tải lại 4GB).
Khi training loop kết thúc (chạm max-steps hoặc crash), hàm CHỜ backup release cuối
xác nhận OK (thử tối đa 3 lần, cách nhau 30s) rồi mới return -> container mới tắt hẳn
và ngừng tính phí GPU. Nếu cả 3 lần đều fail, checkpoint vẫn an toàn trong Volume
(không mất) nhưng release sẽ tạm chưa có bản mới nhất — gọi lại ::train hoặc ::status
để thử backup lại.
"""
import modal

app = modal.App("vija-phase1")
REPO = "trituenguyen97/Bit-Translate"

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

vol = modal.Volume.from_name("vija-vol", create_if_missing=True)
gh = modal.Secret.from_name("github-token")

# 292M dims + config Phase 1 (resume step25000 -> +5000 step trên vòng 3a, LR restart)
DIMS = "--d-model 1152 --n-layers 16 --n-heads 18 --d-ff 3072"
PHASE1 = ("--max-tokens 4096 --grad-accum 32 --compile "  # đã xác nhận hội tụ 5.55s/step trên A10G (warmup ~45 step)
          "--max-steps 30000 --lr-anchor 25000 --lr 1.2e-4 --min-lr 1.5e-5 --warmup 200 "
          "--save-every 60 --milestone-every 2000 --log-every 5")   # lưu last.pt mỗi ~4.5'


@app.function(image=image, volumes={"/persist": vol}, secrets=[gh], timeout=1800)
def setup():
    """Tải data vòng 3a + checkpoint step25000 vào Volume (idempotent)."""
    import os
    import subprocess

    def sh(c):
        print(">>", c, flush=True)
        subprocess.run(c, shell=True, check=True)

    os.makedirs("/persist/bin", exist_ok=True)
    os.makedirs("/persist/checkpoints", exist_ok=True)
    env = f'GH_TOKEN={os.environ["GH_TOKEN"]}'

    # 1) data vòng 3a premix (tar chứa data/bin/*) -> /persist/bin/*
    if not os.path.exists("/persist/bin/train.tokens.u16"):
        sh(f"cd /tmp && {env} gh release download train-assets-vong3a -R {REPO} "
           f"--pattern bin_mix_vong3a.tar.zst --clobber")
        sh("tar --strip-components=1 -C /persist -I zstd -xf /tmp/bin_mix_vong3a.tar.zst && rm /tmp/bin_mix_vong3a.tar.zst")
        print("data OK:", os.listdir("/persist/bin")[:6], flush=True)
    else:
        print("data đã có, bỏ qua tải.", flush=True)

    # 2) checkpoint step mới nhất (chọn bộ .step cao hơn, như restore_300m.sh)
    if not os.path.exists("/persist/checkpoints/last.pt"):
        sh(f"cd /tmp && {env} gh release download autosave-scale300m -R {REPO} --pattern 'last_*.step' --clobber")
        sa = int(open("/tmp/last_a.step").read().split()[0]) if os.path.exists("/tmp/last_a.step") else -1
        sb = int(open("/tmp/last_b.step").read().split()[0]) if os.path.exists("/tmp/last_b.step") else -1
        which = "a" if sa >= sb else "b"
        step = max(sa, sb)
        print(f"bộ mới nhất: last_{which} (step {step})", flush=True)
        sh(f"cd /tmp && {env} gh release download autosave-scale300m -R {REPO} --pattern 'last_{which}.part_*' --clobber")
        sh(f"cat /tmp/last_{which}.part_* > /persist/checkpoints/last.pt && rm /tmp/last_{which}.part_*")
        sz = os.path.getsize("/persist/checkpoints/last.pt")
        assert sz > 3_000_000_000, f"checkpoint chỉ {sz}B (<3GB), bộ hỏng?"
        print(f"checkpoint OK: {sz//2**20}MB (step {step})", flush=True)
    else:
        print("checkpoint đã có, bỏ qua tải.", flush=True)

    vol.commit()
    print("SETUP XONG — Volume vija-vol sẵn sàng.", flush=True)


@app.function(image=image, volumes={"/persist": vol}, secrets=[gh], gpu="L40S",
              cpu=8.0, memory=16384, timeout=8 * 3600)   # L40S ~2.8s/step (thử; nếu preempt như A100 -> đổi lại "A10G")
def train():
    """Resume step25000 -> train vòng 3a. Commit Volume + backup release định kỳ."""
    import os
    import subprocess
    import threading
    import time

    assert os.path.exists("/persist/checkpoints/last.pt"), "chưa setup! chạy ::setup trước."
    # symlink data/bin + checkpoints (train.py đọc ROOT/data/bin, ROOT/checkpoints)
    os.makedirs("/root/bt/data", exist_ok=True)
    for src, dst in [("/persist/bin", "/root/bt/data/bin"), ("/persist/checkpoints", "/root/bt/checkpoints")]:
        if os.path.islink(dst) or os.path.exists(dst):
            subprocess.run(["rm", "-rf", dst])
        os.symlink(src, dst)

    env = dict(os.environ)
    env["BITNET_OPT"] = "off"                 # code gốc đã kiểm chứng trên cloud
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # OK trên Linux thật (native, không WSL2)

    cmd = f"cd /root/bt && python3 scripts/train.py {DIMS} {PHASE1}"
    print(">> TRAIN:", cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)

    def backup_release(tag):
        """Split last.pt -> p_aa/p_ab, upload release, check exit THẬT. Trả True/False."""
        if not os.path.exists("/persist/checkpoints/last.pt"):
            return False
        try:
            r = subprocess.run(
                "cd /persist/checkpoints && rm -f p_* && split -b 1900m last.pt p_ && "
                f'GH_TOKEN={os.environ["GH_TOKEN"]} gh release upload autosave-scale300m '
                f"-R {REPO} --clobber p_aa p_ab",
                shell=True, timeout=900, capture_output=True, text=True)
            subprocess.run("rm -f /persist/checkpoints/p_*", shell=True)
            if r.returncode == 0:
                print(f"[{tag}] backup release OK", flush=True)
                return True
            print(f"[{tag}] backup release FAIL: {r.stderr[-300:]}", flush=True)
            return False
        except Exception as e:
            print(f"[{tag}] backup lỗi: {e}", flush=True)
            return False

    # nền: commit Volume mỗi 5', upload last.pt lên release mỗi 10' (chống mất khi timeout)
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

    g = threading.Thread(target=guardian, daemon=True)
    g.start()
    rc = proc.wait()
    stop.set()
    vol.commit()
    print(f"TRAIN kết thúc rc={rc}. Volume đã commit.", flush=True)

    # backup CUỐI đảm bảo release có checkpoint mới nhất TRƯỚC KHI container tắt
    # (chỉ dừng tính phí khi upload xác nhận OK, hoặc hết số lần thử — không treo vô hạn)
    ok = False
    for attempt in range(1, 4):
        ok = backup_release("final")
        if ok:
            break
        print(f"[final] thử lại lần {attempt}/3 thất bại, chờ 30s...", flush=True)
        time.sleep(30)
    if ok:
        print("[final] release ĐÃ CÓ checkpoint mới nhất — an toàn để dừng.", flush=True)
    else:
        print("[final] CẢNH BÁO: backup release thất bại sau 3 lần — "
              "checkpoint vẫn còn nguyên trong Volume (không mất), nhưng release CHƯA cập nhật. "
              "Gọi lại ::train hoặc ::status để thử backup lại.", flush=True)

    # xem đã tới đích chưa
    log = "/persist/checkpoints/train.log"
    if os.path.exists(log):
        tail = subprocess.run(["tail", "-3", log], capture_output=True, text=True).stdout
        print("train.log:\n" + tail, flush=True)


@app.function(image=image, volumes={"/persist": vol}, timeout=120)
def status():
    """Xem tiến độ train từ Volume (giống watch.sh nhưng cho Modal).
       .venv/bin/python -m modal run cloud/modal_train_phase1.py::status"""
    import os
    import subprocess

    vol.reload()  # lấy trạng thái commit mới nhất (train commit mỗi 5' -> trễ tối đa ~5')
    log = "/persist/checkpoints/train.log"
    ck = "/persist/checkpoints/last.pt"
    print("==== TIẾN ĐỘ PHASE 1 (292M, vòng 3a) — từ Volume ====")
    if os.path.exists(log):
        out = subprocess.run(f"grep -E '^step ' {log} | tail -8", shell=True,
                             capture_output=True, text=True).stdout.strip()
        print(out or "(chưa có dòng step nào — đang compile warmup ~2-3')")
    else:
        print("(chưa có train.log — chưa chạy train hoặc chưa commit lần đầu)")
    if os.path.exists(ck):
        import time
        age = (time.time() - os.path.getmtime(ck)) / 60
        print(f"\nlast.pt: {os.path.getsize(ck) // 2**20}MB, cập nhật {age:.0f}' trước")
    print("(số này trễ tối đa ~5' so thực tế do commit định kỳ; muốn realtime xem terminal ::train hoặc dashboard)")


@app.local_entrypoint()
def main():
    print("Dùng: modal run cloud/modal_train_phase1.py::setup  (1 lần)  ->  ::train  |  ::status (xem tiến độ)")
