# Nhật ký gate sức chứa — quyết định khi nào scale 110M → 200M

> Cách dùng: sau mỗi vòng train + eval, điền 1 dòng theo PLAN_BUOC5 §5.1.
> Tín hiệu trần: T1 = data nhắm không ăn (<+3 chrF dù ≥800 cặp sạch); T2 = domain cũ tụt >2 chrF;
> T3 = vi→ja FLORES < 27 sau Vòng 2; T4 = loss sàn (TB 500 step cuối chênh <0.02 so vòng trước).
> **≥2 tín hiệu cùng vòng HOẶC T1 lặp 2 vòng liên tiếp → scale 200M (from-scratch).**

| Vòng | Domain nhắm | Cặp mới sạch | Δprobe domain nhắm | Δdomain cũ xấu nhất | ΔFLORES vi→ja / ja→vi | TB loss 500 step cuối | Tín hiệu bật | Kết luận |
|---|---|---|---|---|---|---|---|---|
| 1 (14k→19k) | IT, câu khó, họp | 189.670 (BT+glossary+IT docs) | IT vi→ja **+14.6**; câu khó +10.5; họp +7.9 | hội thoại ja→vi −1.0 (trong nhiễu, <2) | +0.1 / +1.2 | ~1.39 (mốc gốc) | **0** | Data ăn đúng đích → tiếp Vòng 2, chưa scale |
| 2 (19k→23k) | hội thoại, idiom, phủ định kép, ẩm thực, keigo | 13.858 (11 mode) | _(điền sau eval)_ | _(điền)_ | _(điền — T3 nếu vi→ja <27)_ | _(điền — T4 nếu chênh <0.02)_ | _(điền)_ | _(điền)_ |
| 3 | số liệu, phủ định, câu phức | | | | | | | |
