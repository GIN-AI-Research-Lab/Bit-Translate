"""Chuan hoa input truoc khi tokenize bang llama.cpp/bitnet.cpp — BAT BUOC.

Ly do (phat hien 2026-07-17): SentencePiece luc TRAIN ap NFKC (nmt_nfkc mac dinh)
nen model chi thay dau HALFWIDTH (? ! % ...). Nhung tokenizer runtime cua
llama.cpp KHONG chuan hoa — dau fullwidth tieng Nhat ？！％ bi tach thanh
byte-fallback (vd ？ -> [245,194,165] thay vi ? -> 274), model gan nhu chua
tung thay chuoi nay khi train. Moi cau hoi tieng Nhat that (ket thuc ？) deu dinh.

Dung o MOI diem vao model: eval scripts, app, server wrapper.
"""
import re
import unicodedata


def normalize_for_model(text: str) -> str:
    """NFKC (khop nmt_nfkc cua SPM o muc du dung cho VI/JA) + don khoang trang."""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t


_SENT_SPLIT = re.compile(r"(?<=[。！？!?])\s*")


def split_sentences(text: str) -> list[str]:
    """Tach input nhieu cau thanh tung cau (giu dau cuoi cau).

    Vi sao: data train (OpenSubtitles) chu yeu 1 cau/dong; input 2+ cau lam model
    110M bia dinh dang phu de "- ... - ...". Do 2026-07-17 (32 cau ja2vi nhieu-cau):
    tach cau giup ro cau formal/IT (id63: chrF 37->57), khong cuu khau ngu.
    """
    return [p for p in _SENT_SPLIT.split(text) if p.strip()]


_DASH_PREFIX = re.compile(r"^\s*[-–—]\s*")


def clean_output(text: str) -> str:
    """Don di chung phu de: gach dau dong "- " o dau output."""
    return _DASH_PREFIX.sub("", text.strip())


def restore_ja_fullwidth(text: str) -> str:
    """Hien thi output tieng Nhat tu nhien hon: ?!% halfwidth -> fullwidth.
    Chi dung cho output JA (vi2ja), KHONG dung cho input."""
    return text.replace("?", "？").replace("!", "！").replace("%", "％")
