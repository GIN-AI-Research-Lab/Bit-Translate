#!/usr/bin/env python3
import glob, os

def count_lines(filepath):
    try:
        with open(filepath, 'rb') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

opus_files = glob.glob('data/raw/opus/*/*.ja')
opus_cnt = sum(count_lines(f) for f in opus_files)

clean_cnt = count_lines('data/clean/train.ja')
syn_files = glob.glob('data/synthetic/*.ja') + glob.glob('data/synthetic/*/*.ja')
syn_cnt = sum(count_lines(f) for f in syn_files)

pool_cnt = count_lines('data/synthetic/ja_indomain_bt.clean.ja')

print("=== BÁO CÁO DỮ LIỆU THỰC TẾ ===")
print(f"1. Raw OPUS (9 corpus)    : {opus_cnt:>10,d} câu")
print(f"2. Clean train.ja         : {clean_cnt:>10,d} câu")
print(f"3. Synthetic (all vòng)   : {syn_cnt:>10,d} câu")
print(f"4. JA Mono Clean Pool     : {pool_cnt:>10,d} câu")
print(f"----------------------------------------")
print(f"TỔNG CÂU DỰ KIẾN KICKOFF KD : {clean_cnt + pool_cnt:>10,d} câu (~6.24M)")
