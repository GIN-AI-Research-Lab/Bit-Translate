"""Back-translation vòng 3b trên Modal (L40S): dịch pool JA mono 841k câu sạch
bằng checkpoint 292M step30000 (chiều ja->vi mạnh) -> sinh cặp (VI tổng hợp, JA
thật) để train chiều vi->ja. Tái dùng cloud/backtranslate.py (đã có KV-cache
batched generate + freeze_for_inference, nhanh hơn nhiều so với generate() thô).

Chuẩn bị 1 lần (đã có sẵn secret github-token + Volume vija-vol từ Phase 1):
    .venv/bin/python -m modal run cloud/modal_backtranslate.py::test    # thử nhanh 2000 câu, đo tốc độ
Chạy full (sau khi ::test cho tốc độ ổn):
    .venv/bin/python -m modal run --detach cloud/modal_backtranslate.py::run
Xem tiến độ:
    .venv/bin/python -m modal run cloud/modal_backtranslate.py::status
"""
import modal

app = modal.App("vija-backtranslate")
REPO = "trituenguyen97/Bit-Translate"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "zstd")
    .run_commands(
        "curl -fsSL https://github.com/cli/cli/releases/download/v2.63.0/gh_2.63.0_linux_amd64.tar.gz "
        "| tar xz -C /tmp && cp /tmp/gh_2.63.0_linux_amd64/bin/gh /usr/local/bin/gh")
    .pip_install("torch", "numpy<2", "sentencepiece")
    .add_local_dir("src", "/root/bt/src")
    .add_local_dir("cloud", "/root/bt/cloud")
    .add_local_dir("tokenizer", "/root/bt/tokenizer")
    .add_local_file("data/synthetic/ja_indomain_bt.clean.ja",
                     "/root/bt/data/synthetic/ja_indomain_bt.clean.ja")
)

vol = modal.Volume.from_name("vija-vol", create_if_missing=True)
gh = modal.Secret.from_name("github-token")

BT_TAG = "bt-vong3b"  # release chứa bt_vong3b.ja / bt_vong3b.vi (backup định kỳ)


def _ensure_ckpt():
    import os
    import subprocess
    os.makedirs("/persist/checkpoints", exist_ok=True)
    ckpt = "/persist/checkpoints/last_30000.pt"
    if os.path.exists(ckpt) and os.path.getsize(ckpt) > 3_000_000_000:
        return ckpt
    env = f'GH_TOKEN={os.environ["GH_TOKEN"]}'
    subprocess.run(f"cd /tmp && {env} gh release download autosave-scale300m -R {REPO} "
                    f"--pattern 'p_a*' --clobber", shell=True, check=True)
    subprocess.run(f"cat /tmp/p_aa /tmp/p_ab > {ckpt} && rm -f /tmp/p_aa /tmp/p_ab",
                    shell=True, check=True)
    sz = __import__("os").path.getsize(ckpt)
    assert sz > 3_000_000_000, f"checkpoint chỉ {sz}B, hỏng?"
    return ckpt


def _backup_release(tag):
    import os
    import subprocess
    ja, vi = "/persist/bt_vong3b.ja", "/persist/bt_vong3b.vi"
    if not (os.path.exists(ja) and os.path.exists(vi)):
        return False
    env = f'GH_TOKEN={os.environ["GH_TOKEN"]}'
    # snapshot trước khi upload: backtranslate.py liên tục append 2 file này (mỗi CHUNK),
    # upload thẳng file đang ghi dở dính race "request body larger than content length".
    # Copy sang thư mục riêng GIỮ NGUYÊN basename (asset name = basename, không dùng cú
    # pháp path#label vì đó chỉ đổi display label, không đổi tên file thật trên release).
    subprocess.run("rm -rf /tmp/bt_snap && mkdir -p /tmp/bt_snap && "
                   "cp /persist/bt_vong3b.ja /persist/bt_vong3b.vi /tmp/bt_snap/", shell=True)
    # kiểm tra release đã tồn tại chưa để chọn đúng lệnh (upload sai release-not-found lại
    # rơi vào create trùng tag => fail chồng fail)
    exists = subprocess.run(f"{env} gh release view {BT_TAG} -R {REPO}",
                             shell=True, capture_output=True, text=True).returncode == 0
    verb = (f"gh release upload {BT_TAG} bt_vong3b.ja bt_vong3b.vi -R {REPO} --clobber") if exists else (
            f"gh release create {BT_TAG} bt_vong3b.ja bt_vong3b.vi -R {REPO} "
            f"--title 'BT vong 3b (ja mono -> vi synth)' --notes 'auto backup'")
    r = subprocess.run(f"cd /tmp/bt_snap && {env} {verb}", shell=True, timeout=900,
                        capture_output=True, text=True)
    subprocess.run("rm -rf /tmp/bt_snap", shell=True)
    if r.returncode == 0:
        print(f"[{tag}] backup release OK", flush=True)
        return True
    print(f"[{tag}] backup release FAIL: {r.stderr[-300:]}", flush=True)
    return False


@app.function(image=image, volumes={"/persist": vol}, secrets=[gh], gpu="L40S",
              cpu=8.0, memory=16384, timeout=1800)
def test(limit: int = 2000, batch: int = 256):
    """Chạy thử nhanh (mặc định 2000 câu) để đo tốc độ thật trước khi chạy full."""
    import os
    import subprocess
    import time

    ckpt = _ensure_ckpt()
    for f in ("/tmp/test.ja", "/tmp/test.vi"):
        if os.path.exists(f):
            os.remove(f)
    env = dict(os.environ)
    env.update({
        "BT_CKPT": ckpt,
        "BT_SRC": "/root/bt/data/synthetic/ja_indomain_bt.clean.ja",
        "BT_OUT_JA": "/tmp/test.ja",
        "BT_OUT_VI": "/tmp/test.vi",
        "BT_MAX": str(limit),
        "BT_BATCH": str(batch),
        "BT_DEVICE": "cuda",
    })
    t0 = time.time()
    subprocess.run("cd /root/bt && python3 cloud/backtranslate.py", shell=True, env=env, check=True)
    dt = time.time() - t0
    n = sum(1 for _ in open("/tmp/test.ja", encoding="utf-8"))
    print(f"\n=== TEST: {n} câu trong {dt:.0f}s ({n/dt:.1f} câu/s) ===", flush=True)
    print(f"Ước tính 841k câu: {841000/(n/dt)/3600:.1f} giờ", flush=True)


@app.function(image=image, volumes={"/persist": vol}, secrets=[gh], gpu="L40S",
              cpu=8.0, memory=16384, timeout=8 * 3600)
def run(batch: int = 256, maxtok: int = 128):
    """Chạy full 841k câu, tự resume nếu gọi lại (backtranslate.py resume theo số dòng đã có)."""
    import os
    import subprocess
    import threading
    import time

    ckpt = _ensure_ckpt()
    env = dict(os.environ)
    env.update({
        "BT_CKPT": ckpt,
        "BT_SRC": "/root/bt/data/synthetic/ja_indomain_bt.clean.ja",
        "BT_OUT_JA": "/persist/bt_vong3b.ja",
        "BT_OUT_VI": "/persist/bt_vong3b.vi",
        "BT_MAXTOK": str(maxtok),
        "BT_BATCH": str(batch),
        "BT_DEVICE": "cuda",
    })

    # nếu release đã có bản backup mới hơn Volume hiện tại (vd resume trên container mới),
    # tải về trước để không bắt đầu lại từ đầu
    if not os.path.exists("/persist/bt_vong3b.ja"):
        gh_env = f'GH_TOKEN={os.environ["GH_TOKEN"]}'
        subprocess.run(f"cd /persist && {gh_env} gh release download {BT_TAG} -R {REPO} "
                        f"--pattern 'bt_vong3b.*' --clobber", shell=True, capture_output=True)

    proc = subprocess.Popen("cd /root/bt && python3 cloud/backtranslate.py",
                             shell=True, env=env)

    stop = threading.Event()

    def guardian():
        last_up = 0
        while not stop.wait(300):
            try:
                vol.commit()
                print("[guardian] Volume committed", flush=True)
            except Exception as e:
                print("[guardian] commit lỗi:", e, flush=True)
            if time.time() - last_up > 900:
                last_up = time.time()
                _backup_release("guardian")

    g = threading.Thread(target=guardian, daemon=True)
    g.start()
    rc = proc.wait()
    stop.set()
    vol.commit()
    print(f"BT kết thúc rc={rc}. Volume đã commit.", flush=True)

    ok = False
    for attempt in range(1, 4):
        ok = _backup_release("final")
        if ok:
            break
        print(f"[final] thử lại lần {attempt}/3 thất bại, chờ 30s...", flush=True)
        time.sleep(30)
    if ok:
        print("[final] release đã có bản BT mới nhất — an toàn để dừng.", flush=True)
    else:
        print("[final] CẢNH BÁO: backup release thất bại — dữ liệu vẫn còn nguyên trong "
              "Volume, gọi lại ::run hoặc ::status để thử backup lại.", flush=True)


@app.function(image=image, volumes={"/persist": vol}, timeout=60)
def status():
    import os
    vol.reload()
    ja = "/persist/bt_vong3b.ja"
    if os.path.exists(ja):
        n = sum(1 for _ in open(ja, encoding="utf-8"))
        age = (__import__("time").time() - os.path.getmtime(ja)) / 60
        print(f"bt_vong3b.ja: {n:,} câu, cập nhật {age:.0f}' trước (mục tiêu 841,039)")
    else:
        print("(chưa có bt_vong3b.ja trong Volume — chưa chạy ::run hoặc chưa commit lần đầu)")


@app.local_entrypoint()
def main():
    print("Dùng: modal run cloud/modal_backtranslate.py::test  (đo tốc độ, ~2000 câu)")
    print("  -> modal run --detach cloud/modal_backtranslate.py::run  (chạy full, nền)")
    print("  -> modal run cloud/modal_backtranslate.py::status  (xem tiến độ)")
