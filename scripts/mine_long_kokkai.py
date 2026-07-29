# -*- coding: utf-8 -*-
"""
V7A nhánh MINE-LONG — đào câu dài từ kokkai_ja.txt (mỗi dòng = 1 câu, các dòng
liên tiếp = cùng bài phát biểu, không có marker ranh giới).

Loại A (long_single.txt):  câu đơn 120-250 ký tự.
Loại B (long_adjacent.txt): ghép 2-3 câu NGẮN (<120 ký tự) LIỀN KỀ, giữ nguyên
                            thứ tự + dấu câu gốc, mẫu 100-250 ký tự.

Dedup: blake2b(digest_size=8) trên câu strip, so với TOÀN BỘ clean_v6/train.ja
(15,3M dòng) + tự dedup. Reservoir sampling để lấy mẫu đều trên toàn file
(phủ nhiều khoá quốc hội) với seed cố định.
"""
import hashlib
import random
import re
import sys
import time

KOKKAI = "D:/Bit-Translate-data/raw/kokkai_ja.txt"
TRAIN = "D:/Bit-Translate-data/clean_v6/train.ja"
OUT_A = "D:/Bit-Translate-data/v7a/long_single.txt"
OUT_B = "D:/Bit-Translate-data/v7a/long_adjacent.txt"

CAP_A = 250_000   # mục tiêu 150-250k
CAP_B = 150_000   # mục tiêu 100-150k
SEED = 20260729

KANA_RE = re.compile(r"[぀-ヿ]")
KANJI_RE = re.compile(r"[一-鿿]")
URL_RE = re.compile(r"https?://|www\.", re.I)
# ~9,6% dòng bắt đầu bằng "○Tên+Chức vụ " = marker LƯỢT PHÁT BIỂU MỚI.
# Phải strip khỏi câu và dùng làm ranh giới run ghép liền kề (không ghép 2 người nói).
SPEAKER_RE = re.compile(r"^○[^ 　]{1,25}[ 　]+")
# Tên có dấu cách ("○階　猛委員　...") -> sau strip còn "猛委員　..." / "委員　...".
# Câu Nhật thật không có dấu cách sau các đuôi này -> strip thêm an toàn.
ROLE_RE = re.compile(
    r"^[^ 　。、]{0,14}(?:君|委員長|委員|大臣|議員|参考人|会長|長官|総裁|議長|"
    r"公述人|証人|分科員|主査|政務官|次官|補佐人)[ 　]+")


def h8(s: str) -> bytes:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


BAD_SYM_RE = re.compile(r"[〔〕◇◎△▲□■※★☆○●—―]")  # divider nghị trình, chú thích sân khấu
MANY_SPACE_RE = re.compile(r"(?:.*[ 　]){3,}")  # văn biểu chương/danh mục nghị trình


def valid_ja(s: str) -> bool:
    """kana + kanji, không URL, ASCII <= 30%, không divider/danh mục."""
    if URL_RE.search(s):
        return False
    if BAD_SYM_RE.search(s) or MANY_SPACE_RE.match(s):
        return False
    if not KANA_RE.search(s) or not KANJI_RE.search(s):
        return False
    n_ascii = len(s.encode("ascii", "ignore"))
    return n_ascii <= 0.30 * len(s)


class Reservoir:
    def __init__(self, cap: int, rng: random.Random):
        self.cap = cap
        self.rng = rng
        self.items = []
        self.n_seen = 0

    def add(self, item: str):
        self.n_seen += 1
        if len(self.items) < self.cap:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.n_seen)
            if j < self.cap:
                self.items[j] = item


def load_train_hashes() -> set:
    t0 = time.time()
    hs = set()
    with open(TRAIN, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            hs.add(h8(line.strip()))
            if (i + 1) % 5_000_000 == 0:
                print(f"  train hash {i+1:,} ({time.time()-t0:.0f}s)", file=sys.stderr)
    print(f"train hashes: {len(hs):,} unique / dòng đã đọc ({time.time()-t0:.0f}s)",
          file=sys.stderr)
    return hs


def main():
    train = load_train_hashes()
    rng_a = random.Random(SEED)
    rng_b = random.Random(SEED + 1)
    rng_t = random.Random(SEED + 2)  # ngưỡng độ dài ngẫu nhiên cho loại B

    res_a = Reservoir(CAP_A, rng_a)
    res_b = Reservoir(CAP_B, rng_b)
    seen_a = set()
    seen_b = set()
    dedup_rej_a = 0   # trúng train
    selfdup_a = 0
    dedup_rej_b = 0
    selfdup_b = 0

    # trạng thái ghép liền kề
    cur = []          # các câu ngắn liên tiếp đang chờ ghép
    cur_len = 0
    target = rng_t.randint(100, 200)  # emit khi tổng >= target (đa dạng độ dài)

    def reset_run():
        nonlocal cur, cur_len
        cur = []
        cur_len = 0

    t0 = time.time()
    with open(KOKKAI, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if (i + 1) % 1_000_000 == 0:
                print(f"  kokkai {i+1:,} A:{res_a.n_seen:,} B:{res_b.n_seen:,} "
                      f"({time.time()-t0:.0f}s)", file=sys.stderr)
            s = line.strip()
            if not s:
                reset_run()
                continue
            turn_start = False
            if s.startswith("○"):
                s2 = SPEAKER_RE.sub("", s)
                if s2 == s:  # ○ nhưng không khớp dạng marker -> dòng rác
                    reset_run()
                    continue
                s = s2
                turn_start = True  # người nói MỚI -> phá mạch liền kề
            # Marker mất ○ do tiền xử lý ("委員 ..." / "国務大臣 ...") hoặc tên
            # chứa dấu cách ("猛委員　...") -> strip thêm trên MỌI dòng.
            for _ in range(2):
                s2 = ROLE_RE.sub("", s)
                if s2 == s:
                    break
                s = s2
                turn_start = True
            if "○" in s:  # ○ giữa dòng (72/2M) -> loại
                reset_run()
                continue
            if turn_start:
                reset_run()
            L = len(s)

            # --- Loại A: câu đơn dài ---
            if 120 <= L <= 250 and valid_ja(s):
                hh = h8(s)
                if hh in train:
                    dedup_rej_a += 1
                elif hh in seen_a:
                    selfdup_a += 1
                else:
                    seen_a.add(hh)
                    res_a.add(s)

            # --- Loại B: run các câu ngắn liền kề ---
            if 15 <= L < 120 and valid_ja(s):
                cur.append(s)
                cur_len += L
                # bỏ đầu run nếu quá 3 câu hoặc quá 250 ký tự mà chưa emit được
                while len(cur) > 3 or cur_len > 250:
                    cur_len -= len(cur.pop(0))
                if cur_len >= target and len(cur) >= 2 and cur_len <= 250:
                    t = "".join(cur)  # giữ nguyên thứ tự + dấu câu gốc
                    hh = h8(t)
                    if hh in train:
                        dedup_rej_b += 1
                    elif hh in seen_b:
                        selfdup_b += 1
                    else:
                        seen_b.add(hh)
                        res_b.add(t)
                    reset_run()
                    target = rng_t.randint(100, 200)
            else:
                # dòng dài/không hợp lệ phá mạch liền kề
                reset_run()

    with open(OUT_A, "w", encoding="utf-8", newline="\n") as f:
        for s in res_a.items:
            f.write(s + "\n")
    with open(OUT_B, "w", encoding="utf-8", newline="\n") as f:
        for s in res_b.items:
            f.write(s + "\n")

    def stats(items):
        ls = sorted(len(x) for x in items)
        n = len(ls)
        if not n:
            return "rỗng"
        med = ls[n // 2]
        p10 = ls[n // 10]
        p90 = ls[(n * 9) // 10]
        return f"n={n:,} min={ls[0]} p10={p10} median={med} p90={p90} max={ls[-1]}"

    print("=== KẾT QUẢ ===")
    print(f"A candidates unique (sau dedup): {res_a.n_seen:,}; "
          f"loại vì trùng train: {dedup_rej_a:,}; tự-trùng: {selfdup_a:,}")
    print(f"A ghi {len(res_a.items):,} dòng -> {OUT_A}")
    print(f"A phân bố ký tự: {stats(res_a.items)}")
    print(f"B combos unique (sau dedup): {res_b.n_seen:,}; "
          f"loại vì trùng train: {dedup_rej_b:,}; tự-trùng: {selfdup_b:,}")
    print(f"B ghi {len(res_b.items):,} dòng -> {OUT_B}")
    print(f"B phân bố ký tự: {stats(res_b.items)}")


if __name__ == "__main__":
    main()
