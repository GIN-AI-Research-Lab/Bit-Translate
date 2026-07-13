# Train tiếp trên GPU cloud (RTX A4000 16GB)

Mục tiêu: đưa training đang chạy ở nhà lên A4000 để chạy tiếp NHANH HƠN (không bị
power-cap, VRAM gấp đôi, Linux thật). Resume đúng step đang có, không mất tiến độ.

## 1. Ở NHÀ — tạo gói upload
Chạy trong `~/Train-model-translate` (checkpoint `last.pt` tự lưu mỗi 100 step nên luôn mới):

```bash
cd ~/Train-model-translate
tar -cf vija_cloud.tar src scripts tokenizer data/bin data/clean/flores checkpoints/last.pt cloud
```
Gói ~2.1GB. (Không nén vì phần lớn là nhị phân, nén cũng gần như không nhỏ hơn.)

## 2. Upload lên cloud
**Nếu có SSH/scp** (khuyên dùng — resume được nếu đứt mạng):
```bash
rsync -av --progress vija_cloud.tar USER@HOST:~/     # thay USER@HOST + cổng theo trang cấp
```
Hoặc dùng nút upload trên web của ckey nếu không có SSH.

## 3. Trên MÁY CLOUD — giải nén + cài + chạy
```bash
mkdir -p ~/Train-model-translate && tar -xf ~/vija_cloud.tar -C ~/Train-model-translate
cd ~/Train-model-translate
bash cloud/setup_cloud.sh          # cài deps (~1 phút)
nohup bash cloud/run_cloud.sh > checkpoints/cloud.log 2>&1 &   # chạy nền, tự resume
```
Xem tiến độ:
```bash
grep -E "^step " checkpoints/train.log | tail -5
```
Có ~1–3 phút warmup biên dịch (compile) trước khi log step đầu — bình thường.
Nếu log báo **OUT OF MEMORY**: sửa `cloud/run_cloud.sh` hạ `--max-tokens 6144` (hoặc 4096).

## 4. Lấy kết quả về
Khi log có `training loop exited` (hoặc bất cứ lúc nào muốn lấy):
```bash
# đóng gói model 1.58-bit (nhỏ ~40-50MB)
python3 scripts/export_packed.py --ckpt checkpoints/last.pt
# đo chất lượng chrF (GPU rảnh sau khi train xong)
python3 scripts/evaluate.py --ckpt checkpoints/last.pt --n 1000 --device cuda
```
Tải về nhà (một trong hai):
- `dist/model_1p58.npz` (~50MB) — model cuối để chạy CPU.
- `checkpoints/last.pt` (~1.3GB) — nếu muốn train tiếp ở nhà.

## Lưu ý QUAN TRỌNG
- **Tính tiền theo phút + tự xóa GPU khi hết số dư** → giữ đủ tiền, và **tải `last.pt` về định kỳ** (vài giờ/lần) phòng instance chết mất data. Community tier (96.4% uptime) có thể gián đoạn.
- Thuê tối đa 72h — dư sức (dự kiến train xong trong ~10–15h).
- Pause để nghỉ: `touch checkpoints/PAUSED` rồi kill python; chạy lại `bash cloud/run_cloud.sh`.
