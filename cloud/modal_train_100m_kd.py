"""Train 100M ja->vi FROM-SCRATCH trên KD full 10,5M câu (pilot chất lượng data).

Khác các bản v1-v4: KHÔNG resume checkpoint cũ, KHÔNG mix base — train mới hoàn
toàn trên 10.509.854 cặp ja->vi sạch (Gemini KD, rule-filter + NFKC). Data THUẦN
ja->vi: [BOS] >>vie<< <ja> [EOS] <vi> [EOS] — không >>jpn<<, không >>fix<<, không
>>thinking<<. Mục tiêu: xem 100M trên data KD sạch có thắng 292M cũ không.

Data: upload thẳng vào Modal Volume (KHÔNG qua GitHub release):
  modal volume create vija-100m-kd-vol
  modal volume put vija-100m-kd-vol data/bin bin      # -> /persist/bin/{train,dev}.*

Chạy:
  # 1) Calibration đo tok/s thật (300 step, ~vài xu) — 300 step NÀY được resume tiếp:
  python3 -m modal run cloud/modal_train_100m_kd.py::train --max-steps 300
  # 2) Xem log lấy tok/s -> tính $/epoch -> chốt max-steps vừa ngân sách:
  python3 -m modal run cloud/modal_train_100m_kd.py::status
  # 3) Chạy tiếp tới đích (resume từ 300):
  python3 -m modal run --detach cloud/modal_train_100m_kd.py::train --max-steps 15000

412M token, 1 epoch = ~3.145 step (max-tokens 8192 x grad-accum 16 = 131k tok/step).
"""
import modal

app = modal.App("vija-100m-kd")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy<2", "sentencepiece", "sacrebleu")
    .add_local_dir("src", "/root/bt/src")
    .add_local_dir("scripts", "/root/bt/scripts")
    .add_local_dir("tokenizer", "/root/bt/tokenizer")
)

vol = modal.Volume.from_name("vija-100m-kd-vol", create_if_missing=True)

# 100M: d768 / 12 layers / 12 heads / ff2048, vocab 32001 (khớp tokenizer, >>fix<< chết)
DIMS = "--d-model 768 --n-layers 12 --n-heads 12 --d-ff 2048 --vocab-size 32001"
# FROM-SCRATCH: lr 3e-4 đỉnh, warmup 1000, cosine tới min-lr; KHÔNG lr-anchor (bắt đầu step 0).
# max-seq 256 (data không có >>fix<< 320-token). pad-multiple 32 giữ torch.compile ổn.
# label-smoothing 0.1: chuẩn NMT, giúp beam search. LƯU Ý loss in ra sẽ cao hơn
# run cũ (0.757) khoảng +0,05-0,1 dù model KHÔNG tệ hơn — đừng so loss chéo.
BASE_ARGS = ("--max-tokens 8192 --grad-accum 16 --compile --max-seq 256 --pad-multiple 32 "
             "--lr 3e-4 --min-lr 3e-5 --warmup 1000 --label-smoothing 0.1 "
             "--save-every 500 --milestone-every 1000 --log-every 10")


@app.function(image=image, volumes={"/persist": vol}, gpu="L40S",
              cpu=8.0, memory=16384, timeout=24 * 3600)
def train(max_steps: int = 15000):
    """Train from-scratch tới max_steps. Resume tự động nếu /persist/checkpoints/last.pt có."""
    import os
    import subprocess
    import threading
    import time

    assert os.path.exists("/persist/bin/train.tokens.u16"), \
        "chưa có data! chạy: modal volume put vija-100m-kd-vol data/bin bin"
    os.makedirs("/persist/checkpoints", exist_ok=True)
    os.makedirs("/root/bt/data", exist_ok=True)
    for src, dst in [("/persist/bin", "/root/bt/data/bin"),
                     ("/persist/checkpoints", "/root/bt/checkpoints")]:
        if os.path.islink(dst) or os.path.exists(dst):
            subprocess.run(["rm", "-rf", dst])
        os.symlink(src, dst)

    env = dict(os.environ)
    env["BITNET_OPT"] = "off"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = (f"cd /root/bt && python3 scripts/train.py {DIMS} {BASE_ARGS} "
           f"--max-steps {max_steps}")
    print(f">> TRAIN 100M ja->vi from-scratch (max_steps={max_steps}):\n   {cmd}", flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)

    stop = threading.Event()

    def guardian():
        while not stop.wait(300):
            try:
                vol.commit()
                print("[guardian] Volume committed", flush=True)
            except Exception as e:
                print("[guardian] commit lỗi:", e, flush=True)

    threading.Thread(target=guardian, daemon=True).start()
    rc = proc.wait()
    stop.set()
    vol.commit()
    print(f"TRAIN kết thúc rc={rc}. Volume đã commit.", flush=True)

    log = "/persist/checkpoints/train.log"
    if os.path.exists(log):
        tail = subprocess.run(["tail", "-8", log], capture_output=True, text=True).stdout
        print("train.log (8 dòng cuối):\n" + tail, flush=True)


@app.function(image=image, volumes={"/persist": vol}, gpu="L40S",
              cpu=8.0, memory=16384, timeout=12 * 3600)
def train_v5(steps: int = 8000, lr: float = 8e-5, n_layers: int = 18,
             bin_dir: str = "bin_v5", ckpt_dir: str = "checkpoints_v5",
             compile: bool = True):
    """VÒNG 5 — DATA THẬT MIỀN MỚI (Quốc hội + CC-100), từ v4_avg5.

    Vòng 4 thêm 9% data mà 5,7% là tổng hợp -> +6 điểm bench thật nhưng p=0,155
    (chưa đủ ý nghĩa). Vòng 5 đảo tỷ lệ: phần thêm gần như TOÀN data THẬT chưa
    từng có trong corpus, chọn theo bốn lỗi ĐO ĐƯỢC:
      - câu dài Quốc hội   : bench câu dài 52%, probe không-ngắn 73%
      - câu giàu số        : bộ dò lech_so 94% chính xác, tăng 3,8%->16,1% theo độ dài
      - khẩu ngữ CC-100    : 17/25 câu ngắn sai là slang/idiom
      - quán ngữ sinh ×3   : 3.168/4.651 quán ngữ <50 lần, 2.098 mục = 0 lần
    Chỉ quán ngữ được oversample; data thật giữ nguyên tỷ lệ tự nhiên.

    steps 8000 = ngân sách $9 (~3,3 giờ ở $2,70/h) trừ ~17 phút compile.
    --fixed-shapes BẮT BUỘC: xem chú thích ở train_v4 (không có nó, compile treo
    65 phút vì 84 shape (B,T) vượt recompile_limit=64).

    Chuẩn bị: scripts/overnight_v5.sh làm hết (gộp -> binarize -> upload -> chạy).
    """
    import os
    import subprocess
    import threading

    assert os.path.exists(f"/persist/{bin_dir}/train.tokens.u16"), f"chưa có data {bin_dir}"
    assert os.path.exists(f"/persist/{ckpt_dir}/last.pt"), f"chưa có {ckpt_dir}/last.pt"

    os.makedirs("/root/bt/data", exist_ok=True)
    for src, dst in [(f"/persist/{bin_dir}", f"/root/bt/data/{bin_dir}"),
                     (f"/persist/{ckpt_dir}", "/root/bt/checkpoints")]:
        if os.path.islink(dst) or os.path.exists(dst):
            subprocess.run(["rm", "-rf", dst])
        os.symlink(src, dst)

    import torch
    ck = torch.load(f"/persist/{ckpt_dir}/last.pt", map_location="cpu")
    assert ck["cfg"]["n_layers"] == n_layers, f"checkpoint {ck['cfg']['n_layers']}L != {n_layers}"
    print(f">> VÒNG 5: {n_layers} layer, từ '{ck.get('from','?')}' step={ck['step']}",
          flush=True)

    env = dict(os.environ)
    env["BITNET_OPT"] = "off"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    args = (f"--d-model 768 --n-layers {n_layers} --n-heads 12 --d-ff 2048 "
            f"--vocab-size 32001 --max-tokens 8192 --grad-accum 16 --max-seq 256 "
            f"{'--compile --fixed-shapes ' if compile else ''}"
            f"--pad-multiple 32 --label-smoothing 0.1 --bin-dir data/{bin_dir} "
            f"--lr {lr} --min-lr {lr/10:.2e} --warmup 200 --lr-anchor 0 "
            # milestone 250 (vòng 4 dùng 500): v4_avg5 (5 mốc) hơn v4_avg (3 mốc)
            # trên CẢ hai thước đo, nên nhiều mốc để average là có lợi.
            f"--max-steps {steps} --save-every 250 --milestone-every 250 --log-every 10")
    cmd = f"cd /root/bt && python3 scripts/train.py {args}"
    print(f"   {cmd}", flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)

    stop = threading.Event()

    def guardian():
        while not stop.wait(300):
            try:
                vol.commit()
                print("[guardian] Volume committed", flush=True)
            except Exception as e:  # noqa: BLE001
                print("[guardian] commit lỗi:", e, flush=True)

    threading.Thread(target=guardian, daemon=True).start()
    rc = proc.wait()
    stop.set()
    vol.commit()
    print(f"VÒNG 5 kết thúc rc={rc}. Checkpoint ở /persist/{ckpt_dir}.", flush=True)
    log = f"/persist/{ckpt_dir}/train.log"
    if os.path.exists(log):
        print(subprocess.run(["tail", "-8", log], capture_output=True, text=True).stdout,
              flush=True)


@app.function(image=image, volumes={"/persist": vol}, gpu="L40S",
              cpu=8.0, memory=16384, timeout=12 * 3600)
def train_v6(steps: int = 16400, lr: float = 8e-5, n_layers: int = 18,
             bin_dir: str = "bin_v6", ckpt_dir: str = "checkpoints_v6",
             compile: bool = True):
    """VÒNG 6 — ĐÀO THEO NGUYÊN NHÂN LỖI, không theo chủ đề.

    Vòng 5 bơm 2,45M câu Quốc hội -> bench TED đứng yên (66% vs v4 64%, p=0,755)
    NHƯNG câu Quốc hội held-out nhảy 14%->95%. Bài học: bench đo sai miền.

    Vòng 6 chọn data theo PHÂN LOẠI NGUYÊN NHÂN 59 câu v5 hỏng mà Google dịch được
    (`eval/error_taxonomy_v5.json`) — thuật ngữ chuyên ngành chỉ chiếm 3% lỗi chính:
      cấu trúc câu 37% -> 400k câu nhiều mệnh đề MANG DẤU VĂN NÓI
      nghĩa từ thường 22% -> 512 từ đa nghĩa x 200 câu (mặt trận MỚI)
      ẩn chủ ngữ 10% -> 350k câu chủ ngữ TRUY HỒI ĐƯỢC trong câu
      slang 45% vs Google 90% -> 450k (corpus chỉ có 0,63%)
      phủ định khó -> 350k (corpus chỉ 0,43%)
      katakana + thuật ngữ -> 10.115 từ dưới ngưỡng

    Mốc hiệu chuẩn: keigo đạt 4,0% corpus -> 80% = NGANG Google. Đó là mật độ
    "đủ để học xong một kỹ năng".

    16.400 step = giữ ~2,75 epoch trên corpus 15,4M (+17,1% so với v5), ~$16,4.
    Loss v5 vẫn giảm -0,006/1000 step khi dừng ở 14.000 => CHƯA hội tụ, train thêm
    còn ăn. Backup checkpoint mỗi 5 phút về ổ D phòng hết credit giữa chừng.
    """
    import os
    import subprocess
    import threading

    assert os.path.exists(f"/persist/{bin_dir}/train.tokens.u16"), f"chưa có data {bin_dir}"
    assert os.path.exists(f"/persist/{ckpt_dir}/last.pt"), f"chưa có {ckpt_dir}/last.pt"

    os.makedirs("/root/bt/data", exist_ok=True)
    for src, dst in [(f"/persist/{bin_dir}", f"/root/bt/data/{bin_dir}"),
                     (f"/persist/{ckpt_dir}", "/root/bt/checkpoints")]:
        if os.path.islink(dst) or os.path.exists(dst):
            subprocess.run(["rm", "-rf", dst])
        os.symlink(src, dst)

    import torch
    ck = torch.load(f"/persist/{ckpt_dir}/last.pt", map_location="cpu")
    assert ck["cfg"]["n_layers"] == n_layers, f"checkpoint {ck['cfg']['n_layers']}L != {n_layers}"
    print(f">> VÒNG 6: {n_layers} layer, từ '{ck.get('from','?')}' step={ck['step']}",
          flush=True)

    env = dict(os.environ)
    env["BITNET_OPT"] = "off"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    args = (f"--d-model 768 --n-layers {n_layers} --n-heads 12 --d-ff 2048 "
            f"--vocab-size 32001 --max-tokens 8192 --grad-accum 16 --max-seq 256 "
            f"{'--compile --fixed-shapes ' if compile else ''}"
            f"--pad-multiple 32 --label-smoothing 0.1 --bin-dir data/{bin_dir} "
            f"--lr {lr} --min-lr {lr/10:.2e} --warmup 200 --lr-anchor 0 "
            # milestone 250 (vòng 4 dùng 500): v4_avg5 (5 mốc) hơn v4_avg (3 mốc)
            # trên CẢ hai thước đo, nên nhiều mốc để average là có lợi.
            f"--max-steps {steps} --save-every 250 --milestone-every 250 --log-every 10 "
            # dev-every 500: đo dev SẠCH trong train (eval_dev tất định, train.py 2026-07-28).
            f"--dev-every 500")
    cmd = f"cd /root/bt && python3 scripts/train.py {args}"
    print(f"   {cmd}", flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)

    stop = threading.Event()

    def guardian():
        while not stop.wait(300):
            try:
                vol.commit()
                print("[guardian] Volume committed", flush=True)
            except Exception as e:  # noqa: BLE001
                print("[guardian] commit lỗi:", e, flush=True)

    threading.Thread(target=guardian, daemon=True).start()
    rc = proc.wait()
    stop.set()
    vol.commit()
    print(f"VÒNG 6 kết thúc rc={rc}. Checkpoint ở /persist/{ckpt_dir}.", flush=True)
    log = f"/persist/{ckpt_dir}/train.log"
    if os.path.exists(log):
        print(subprocess.run(["tail", "-8", log], capture_output=True, text=True).stdout,
              flush=True)


@app.function(image=image, volumes={"/persist": vol}, gpu="L40S",
              cpu=8.0, memory=16384, timeout=12 * 3600)
def train_v4(steps: int = 6000, lr: float = 8e-5, n_layers: int = 18,
             bin_dir: str = "bin_v4", ckpt_dir: str = "checkpoints_v4",
             compile: bool = True):
    """VÒNG 4 — ĐỔI LOẠI DATA, GIỮ NGUYÊN KIẾN TRÚC (18L/152,1M từ v3_avg).

    Đây là THÍ NGHIỆM KIỂM CHỨNG một giả thuyết cụ thể, không phải "train thêm":

      Vòng 3 bơm 241k cặp tổng hợp sinh theo CHỦ ĐỀ TỰ NGHĨ RA -> hardbench +18 điểm
      nhưng bench câu THẬT đứng yên (v3 66% = v2 66%). Kết luận: data tổng hợp ngẫu
      nhiên chỉ củng cố cái model đã biết.

      Giả thuyết vòng 4: data phải đi từ DANH SÁCH ĐO ĐƯỢC. `measure_term_coverage.py`
      cho thấy 8.041/11.933 thuật ngữ glossary xuất hiện <= 5 lần trong corpus (2.343
      chưa từng xuất hiện). `gen_term_gap.py` sinh câu ĐÚNG CHO CHỖ THIẾU đó, bản
      dịch tiếng Việt lấy thẳng từ glossary chứ không để model thầy tự đoán.

    Ba thay đổi so với vòng 3, mỗi cái có lý do riêng:
      1. Thêm 934k cặp THẬT (OPUS đã KD) — giữ model bám phân bố tiếng Nhật thực tế.
      2. Hạ tỷ lệ tổng hợp (10,42% -> ~5%) — vòng 3 cho thấy tỷ lệ cao làm lệch giọng
         văn mà không đổi lấy được gì trên câu thật.
      3. Phần tổng hợp thêm mới là termgap + sciterm (theo danh sách phủ), không phải
         thêm nữa các mode phong cách vốn đã bão hòa.

    lr 8e-5 (thấp hơn 1e-4 của vòng 3): điểm xuất phát v3_avg đã hội tụ tốt hơn hẳn
    điểm xuất phát vòng 3 (vốn là model 12L vừa chèn block rỗng), nên cần ít xáo trộn
    hơn. Vẫn warmup 200 + lr-anchor 0 để LR chạy trọn một chu kỳ trên corpus mới.

    ĐO CÁI GÌ: bench câu THẬT (eval/bench_new.jsonl), KHÔNG phải hardbench. Hardbench
    là câu do chính pipeline này sinh ra nên nó luôn đẹp lên — đó chính là cái bẫy
    của vòng 3.

    Chuẩn bị:
      python scripts/init_round.py D:/Bit-Translate-data/checkpoints_v3/v3_avg.pt \
          -o D:/Bit-Translate-data/checkpoints_v4/last.pt
      modal volume put vija-100m-kd-vol D:/Bit-Translate-data/bin_v4 bin_v4
      modal volume put vija-100m-kd-vol D:/Bit-Translate-data/checkpoints_v4 checkpoints_v4
      bash scripts/backup_ckpt_loop.sh checkpoints_v4 600
    """
    import os
    import subprocess
    import threading

    assert os.path.exists(f"/persist/{bin_dir}/train.tokens.u16"), f"chưa có data {bin_dir}"
    assert os.path.exists(f"/persist/{ckpt_dir}/last.pt"), \
        f"chưa có {ckpt_dir}/last.pt — chạy init_round.py rồi modal volume put"

    os.makedirs("/root/bt/data", exist_ok=True)
    for src, dst in [(f"/persist/{bin_dir}", f"/root/bt/data/{bin_dir}"),
                     (f"/persist/{ckpt_dir}", "/root/bt/checkpoints")]:
        if os.path.islink(dst) or os.path.exists(dst):
            subprocess.run(["rm", "-rf", dst])
        os.symlink(src, dst)

    import torch
    ck = torch.load(f"/persist/{ckpt_dir}/last.pt", map_location="cpu")
    got_L = ck["cfg"]["n_layers"]
    assert got_L == n_layers, f"checkpoint có {got_L} layer nhưng --n-layers={n_layers}"
    print(f">> VÒNG 4: {n_layers} layer, từ '{ck.get('from','?')}' step={ck['step']}",
          flush=True)

    env = dict(os.environ)
    env["BITNET_OPT"] = "off"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # --fixed-shapes LÀ BẮT BUỘC ở vòng 4, không phải tinh chỉnh cho vui.
    # Lần chạy đầu (app ap-ZmiWOzNw...) đứng 65 phút chưa xong step 10 (vòng 3 cùng
    # cấu hình: 32 phút). Đo trong container: GPU 0%, CPU đúng 1 lõi ~97% — chữ ký
    # của TorchInductor codegen, tức KHÔNG treo mà compile lâu vô chừng. Nguyên nhân
    # đo được bằng cách chạy make_batches trên chính bin_v4: --pad-multiple 32 chỉ
    # chặn chiều T, chiều B vẫn trôi tự do -> 84 SHAPE (B,T) khác nhau, vượt
    # recompile_limit=64 nên dynamo recompile không dứt. Corpus v4 thêm 922k câu OPUS
    # rất ngắn nên B trải rộng hơn vòng 3 hẳn. Bật --fixed-shapes: còn 6 shape, giữ
    # 96,8% câu, padding thừa còn 22% (thấp hơn 27% của cách cũ).
    # Chạy eager (không compile) đo được 4,15 s/step = ~7 giờ cho 6000 step, so với
    # ~1,3 s/step khi compile chạy đúng.
    args = (f"--d-model 768 --n-layers {n_layers} --n-heads 12 --d-ff 2048 "
            f"--vocab-size 32001 --max-tokens 8192 --grad-accum 16 --max-seq 256 "
            f"{'--compile --fixed-shapes ' if compile else ''}"
            f"--pad-multiple 32 --label-smoothing 0.1 --bin-dir data/{bin_dir} "
            f"--lr {lr} --min-lr {lr/10:.2e} --warmup 200 --lr-anchor 0 "
            f"--max-steps {steps} --save-every 250 --milestone-every 500 --log-every 10")
    cmd = f"cd /root/bt && python3 scripts/train.py {args}"
    print(f"   {cmd}", flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)

    stop = threading.Event()

    def guardian():
        while not stop.wait(300):
            try:
                vol.commit()
                print("[guardian] Volume committed", flush=True)
            except Exception as e:  # noqa: BLE001
                print("[guardian] commit lỗi:", e, flush=True)

    threading.Thread(target=guardian, daemon=True).start()
    rc = proc.wait()
    stop.set()
    vol.commit()
    print(f"VÒNG 4 kết thúc rc={rc}. Checkpoint ở /persist/{ckpt_dir}.", flush=True)
    log = f"/persist/{ckpt_dir}/train.log"
    if os.path.exists(log):
        print(subprocess.run(["tail", "-8", log], capture_output=True, text=True).stdout,
              flush=True)


@app.function(image=image, volumes={"/persist": vol}, gpu="L40S",
              cpu=8.0, memory=16384, timeout=12 * 3600)
def train_v3(steps: int = 8000, lr: float = 1e-4, n_layers: int = 18,
             bin_dir: str = "bin_v2", ckpt_dir: str = "checkpoints_v3"):
    """VÒNG 3 — MỞ RỘNG ĐỘ SÂU 12L->18L (109,6M -> 152,1M), giữ d_model=768.

    Điểm khởi đầu KHÔNG ngẫu nhiên: `scripts/grow_depth.py` chèn 6 block mới với
    zero-init attn.wo + ffn.down nên chúng là IDENTITY — model 18L cho output y hệt
    model 12L (đã verify sai lệch logits = 0). Nghĩa là bắt đầu từ đúng chất lượng
    đã có (chrF 46,8 / 65% dùng được) chứ không phải từ 0.

    Vì sao SÂU chứ không RỘNG: giữ d_model=768 mới kế thừa được embedding + 12 layer
    đã train; đổi d_model là phải train lại từ đầu (~$21 thay vì ~$7). Ngoài ra fail
    còn lại nặng nhất là câu dài/cấu trúc lồng, vốn thiên về độ sâu.

    Đây cũng là THÍ NGHIỆM SẠCH cho câu hỏi "câu dài có phải giới hạn dung lượng":
    khởi đầu = model cũ chính xác, nên mọi thay đổi sau đó chỉ đến từ chỗ chứa mới.

    Chuẩn bị:
      python scripts/grow_depth.py D:/Bit-Translate-data/checkpoints_v2/v2_avg3.pt \
          --layers 18 -o D:/Bit-Translate-data/checkpoints_v3/last.pt --verify
      modal volume put vija-100m-kd-vol D:/Bit-Translate-data/checkpoints_v3 checkpoints_v3
      # backup song song (credit có thể cạn giữa chừng):
      bash scripts/backup_ckpt_loop.sh checkpoints_v3 600
    """
    import os
    import subprocess
    import threading

    assert os.path.exists(f"/persist/{bin_dir}/train.tokens.u16"), f"chưa có data {bin_dir}"
    assert os.path.exists(f"/persist/{ckpt_dir}/last.pt"), \
        f"chưa có {ckpt_dir}/last.pt — chạy grow_depth.py rồi modal volume put"

    os.makedirs("/root/bt/data", exist_ok=True)
    for src, dst in [(f"/persist/{bin_dir}", f"/root/bt/data/{bin_dir}"),
                     (f"/persist/{ckpt_dir}", "/root/bt/checkpoints")]:
        if os.path.islink(dst) or os.path.exists(dst):
            subprocess.run(["rm", "-rf", dst])
        os.symlink(src, dst)

    import torch
    ck = torch.load(f"/persist/{ckpt_dir}/last.pt", map_location="cpu")
    got_L = ck["cfg"]["n_layers"]
    assert got_L == n_layers, f"checkpoint có {got_L} layer nhưng --n-layers={n_layers}"
    print(f">> VÒNG 3: {n_layers} layer, từ '{ck.get('from','?')}'", flush=True)

    env = dict(os.environ)
    env["BITNET_OPT"] = "off"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    dims = (f"--d-model 768 --n-layers {n_layers} --n-heads 12 --d-ff 2048 "
            f"--vocab-size 32001")
    args = (f"{dims} --max-tokens 8192 --grad-accum 16 --compile --max-seq 256 "
            f"--pad-multiple 32 --label-smoothing 0.1 --bin-dir data/{bin_dir} "
            f"--lr {lr} --min-lr {lr/10:.2e} --warmup 200 --lr-anchor 0 "
            # save-every 250 + milestone-every 500: credit có thể cạn giữa chừng nên
            # cần nhiều điểm phục hồi. 16 milestone x 608MB = ~9,7GB, ổ D thừa chỗ.
            f"--max-steps {steps} --save-every 250 --milestone-every 500 --log-every 10")
    cmd = f"cd /root/bt && python3 scripts/train.py {args}"
    print(f"   {cmd}", flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)

    stop = threading.Event()

    def guardian():
        while not stop.wait(300):
            try:
                vol.commit()
                print("[guardian] Volume committed", flush=True)
            except Exception as e:  # noqa: BLE001
                print("[guardian] commit lỗi:", e, flush=True)

    threading.Thread(target=guardian, daemon=True).start()
    rc = proc.wait()
    stop.set()
    vol.commit()
    print(f"VÒNG 3 kết thúc rc={rc}. Checkpoint ở /persist/{ckpt_dir}.", flush=True)
    log = f"/persist/{ckpt_dir}/train.log"
    if os.path.exists(log):
        print(subprocess.run(["tail", "-8", log], capture_output=True, text=True).stdout,
              flush=True)


@app.function(image=image, volumes={"/persist": vol}, gpu="L40S",
              cpu=8.0, memory=16384, timeout=12 * 3600)
def train_v2(steps: int = 4000, lr: float = 1e-4, bin_dir: str = "bin_v2",
             ckpt_dir: str = "checkpoints_v2"):
    """VÒNG 2 — resume từ kd_avg5 trên corpus ĐÃ ENRICH (niche 8,39%, 451,5M token).

    Vì sao resume mà không from-scratch: user chọn tiết kiệm (~$3-5 thay vì $8-14).
    ⚠️ ĐÁNH ĐỔI: model đã học data cũ 5 epoch nên KHÔNG tách được sạch "data niche
    đóng góp bao nhiêu" — kết quả tốt lên có thể do train thêm chứ không hẳn do data
    mới. Muốn kết luận chắc về nút thắt data-vs-dung-lượng thì phải from-scratch.

    lr 1e-4 (thấp hơn 3e-4 của from-scratch) + warmup 200 + step=0/lr-anchor 0:
    LR đi từ mức vừa phải xuống min, đủ hấp thụ 964k cặp niche mà không phá bản đã
    hội tụ. Checkpoint ghi vào ckpt_dir RIÊNG -> run cũ (checkpoints/) còn nguyên.

    Cần upload trước:
      modal volume put vija-100m-kd-vol D:/Bit-Translate-data/bin_v2 bin_v2
      modal volume put vija-100m-kd-vol D:/Bit-Translate-data/checkpoints_v2 checkpoints_v2
    """
    import os
    import subprocess
    import threading

    assert os.path.exists(f"/persist/{bin_dir}/train.tokens.u16"), \
        f"chưa có data! modal volume put vija-100m-kd-vol D:/Bit-Translate-data/{bin_dir} {bin_dir}"
    assert os.path.exists(f"/persist/{ckpt_dir}/last.pt"), \
        f"chưa có điểm khởi đầu! modal volume put ... {ckpt_dir} {ckpt_dir}"

    os.makedirs("/root/bt/data", exist_ok=True)
    for src, dst in [(f"/persist/{bin_dir}", f"/root/bt/data/{bin_dir}"),
                     (f"/persist/{ckpt_dir}", "/root/bt/checkpoints")]:
        if os.path.islink(dst) or os.path.exists(dst):
            subprocess.run(["rm", "-rf", dst])
        os.symlink(src, dst)

    env = dict(os.environ)
    env["BITNET_OPT"] = "off"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    args = (f"{DIMS} --max-tokens 8192 --grad-accum 16 --compile --max-seq 256 "
            f"--pad-multiple 32 --label-smoothing 0.1 --bin-dir data/{bin_dir} "
            f"--lr {lr} --min-lr {lr/10:.2e} --warmup 200 --lr-anchor 0 "
            f"--max-steps {steps} --save-every 500 --milestone-every 500 --log-every 10")
    cmd = f"cd /root/bt && python3 scripts/train.py {args}"
    print(f">> VÒNG 2 resume từ avg5: {steps} step, lr {lr}\n   {cmd}", flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)

    stop = threading.Event()

    def guardian():
        while not stop.wait(300):
            try:
                vol.commit()
                print("[guardian] Volume committed", flush=True)
            except Exception as e:  # noqa: BLE001
                print("[guardian] commit lỗi:", e, flush=True)

    threading.Thread(target=guardian, daemon=True).start()
    rc = proc.wait()
    stop.set()
    vol.commit()
    print(f"VÒNG 2 kết thúc rc={rc}. Checkpoint ở /persist/{ckpt_dir}.", flush=True)
    log = f"/persist/{ckpt_dir}/train.log"
    if os.path.exists(log):
        print(subprocess.run(["tail", "-8", log], capture_output=True, text=True).stdout,
              flush=True)


@app.function(image=image, volumes={"/persist": vol}, gpu="L40S",
              cpu=8.0, memory=16384, timeout=6 * 3600)
def finetune_p2(steps: int = 1200, lr: float = 5e-5, bin_dir: str = "bin_p2"):
    """CURRICULUM PHA 2 — fine-tune nhẹ trên tập nhỏ CHẤT LƯỢNG CAO (niche +
    subset sạch nhất) sau khi pha 1 đã train xong trên toàn corpus.

    Vì sao 2 pha thay vì trộn phẳng: trộn phẳng thì niche (~1% corpus) bị pha
    loãng; pha 2 cho model "nhìn" data niche ở mật độ cao ngay trước khi dừng,
    thường nâng usable rate mạnh hơn. lr thấp (5e-5) + ít step để không quên
    pha 1 (catastrophic forgetting).

    Cần: modal volume put vija-100m-kd-vol data/bin_p2 bin_p2
    Chạy sau khi pha 1 xong (last.pt đã có trong /persist/checkpoints).
    """
    import os
    import shutil
    import subprocess
    import threading

    src_bin = f"/persist/{bin_dir}"
    assert os.path.exists(f"{src_bin}/train.tokens.u16"), \
        f"chưa có data pha 2! chạy: modal volume put vija-100m-kd-vol data/{bin_dir} {bin_dir}"
    last = "/persist/checkpoints/last.pt"
    assert os.path.exists(last), "chưa có last.pt của pha 1 — train pha 1 trước"

    # Giữ bản pha 1 lại để so sánh / quay lui (fine-tune sẽ ghi đè last.pt).
    p1 = "/persist/checkpoints/phase1_final.pt"
    if not os.path.exists(p1):
        shutil.copy2(last, p1)
        print(f">> đã lưu bản pha 1: {p1}", flush=True)

    os.makedirs("/root/bt/data", exist_ok=True)
    for src, dst in [(src_bin, f"/root/bt/data/{bin_dir}"),
                     ("/persist/checkpoints", "/root/bt/checkpoints")]:
        if os.path.islink(dst) or os.path.exists(dst):
            subprocess.run(["rm", "-rf", dst])
        os.symlink(src, dst)

    env = dict(os.environ)
    env["BITNET_OPT"] = "off"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Đọc step hiện tại để đặt max-steps = step + steps, và lr-anchor để LR
    # KHÔNG chạy lại từ đỉnh 3e-4 (sẽ phá model) mà đi từ lr thấp xuống min-lr.
    import torch
    cur = int(torch.load(last, map_location="cpu").get("step", 0))
    tgt = cur + steps
    args = (f"{DIMS} --max-tokens 8192 --grad-accum 16 --compile --max-seq 256 "
            f"--pad-multiple 32 --label-smoothing 0.1 --bin-dir data/{bin_dir} "
            f"--lr {lr} --min-lr {lr/10:.2e} --warmup 100 --lr-anchor {cur} "
            f"--max-steps {tgt} --save-every 200 --milestone-every 400 --log-every 10")
    cmd = f"cd /root/bt && python3 scripts/train.py {args}"
    print(f">> PHA 2 fine-tune: step {cur} -> {tgt}, lr {lr}\n   {cmd}", flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)

    stop = threading.Event()

    def guardian():
        while not stop.wait(300):
            try:
                vol.commit()
                print("[guardian] Volume committed", flush=True)
            except Exception as e:  # noqa: BLE001
                print("[guardian] commit lỗi:", e, flush=True)

    threading.Thread(target=guardian, daemon=True).start()
    rc = proc.wait()
    stop.set()
    vol.commit()
    print(f"PHA 2 kết thúc rc={rc}. Bản pha 1 giữ ở {p1}.", flush=True)


@app.function(image=image, volumes={"/persist": vol}, timeout=120)
def status():
    import os
    import subprocess
    vol.reload()
    log = "/persist/checkpoints/train.log"
    print("==== 100M ja->vi from-scratch (KD full) ====")
    if os.path.exists(log):
        out = subprocess.run(f"grep -E '^step ' {log} | tail -12", shell=True,
                             capture_output=True, text=True).stdout.strip()
        print(out or "(chưa có dòng step — đang compile warmup ~1-2')")
    else:
        print("(chưa có train.log)")
    if os.path.exists("/persist/checkpoints"):
        ms = sorted(f for f in os.listdir("/persist/checkpoints") if f.startswith("step"))
        print(f"milestone: {ms}")


@app.local_entrypoint()
def main():
    print("Upload data: modal volume put vija-100m-kd-vol data/bin bin")
    print("Calibrate  : modal run cloud/modal_train_100m_kd.py::train --max-steps 300")
    print("Full run   : modal run --detach cloud/modal_train_100m_kd.py::train --max-steps 15000")
