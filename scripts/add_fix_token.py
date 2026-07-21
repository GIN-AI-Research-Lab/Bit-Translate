#!/usr/bin/env python3
"""Thêm token đặc biệt >>fix<< vào tokenizer SPM ĐÃ TRAIN mà KHÔNG train lại
(giữ nguyên toàn bộ ID cũ 0-31999 — chỉ APPEND thêm 1 piece ở cuối, vocab
32000 -> 32001). An toàn cho mọi binarized data cũ (ID không đổi chỗ).

  python scripts/add_fix_token.py

Sửa tại chỗ tokenizer/spm_vija_32k.model (giữ bản gốc ở .bak trước khi ghi).
"""
import shutil
import sys
from pathlib import Path

import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as pb2

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "tokenizer" / "spm_vija_32k.model"
NEW_PIECE = ">>fix<<"


def main():
    sp = spm.SentencePieceProcessor(model_file=str(MODEL_PATH))
    if sp.piece_to_id(NEW_PIECE) != sp.unk_id():
        print(f"'{NEW_PIECE}' đã có sẵn (id={sp.piece_to_id(NEW_PIECE)}), không cần thêm.")
        return
    old_vocab = sp.vocab_size()

    m = pb2.ModelProto()
    m.ParseFromString(MODEL_PATH.read_bytes())

    piece = m.pieces.add()
    piece.piece = NEW_PIECE
    piece.score = 0.0
    piece.type = pb2.ModelProto.SentencePiece.USER_DEFINED

    backup = MODEL_PATH.with_suffix(".model.bak")
    if not backup.exists():
        shutil.copy(MODEL_PATH, backup)
        print(f"backup gốc -> {backup}")
    MODEL_PATH.write_bytes(m.SerializeToString())

    sp2 = spm.SentencePieceProcessor(model_file=str(MODEL_PATH))
    new_id = sp2.piece_to_id(NEW_PIECE)
    assert new_id == old_vocab, f"kỳ vọng id={old_vocab}, thực tế={new_id}"
    print(f"OK: vocab {old_vocab} -> {sp2.vocab_size()}, '{NEW_PIECE}' = id {new_id}")
    # sanity: token cũ không đổi
    for p in [">>vie<<", ">>jpn<<", "<s>", "</s>"]:
        assert sp.piece_to_id(p) == sp2.piece_to_id(p), f"{p} bị đổi ID!"
    print("Xác nhận: toàn bộ ID cũ giữ nguyên.")


if __name__ == "__main__":
    sys.exit(main())
