# Nhật ký gate sức chứa — quyết định khi nào scale 110M → 200M

> Cách dùng: sau mỗi vòng train + eval, điền 1 dòng theo PLAN_BUOC5 §5.1.
> Tín hiệu trần: T1 = data nhắm không ăn (<+3 chrF dù ≥800 cặp sạch); T2 = domain cũ tụt >2 chrF;
> T3 = vi→ja FLORES < 27 sau Vòng 2; T4 = loss sàn (TB 500 step cuối chênh <0.02 so vòng trước).
> **≥2 tín hiệu cùng vòng HOẶC T1 lặp 2 vòng liên tiếp → scale 200M (from-scratch).**

| Vòng | Domain nhắm | Cặp mới sạch | Δprobe domain nhắm | Δdomain cũ xấu nhất | ΔFLORES vi→ja / ja→vi | TB loss 500 step cuối | Tín hiệu bật | Kết luận |
|---|---|---|---|---|---|---|---|---|
| 1 (14k→19k) | IT, câu khó, họp | 189.670 (BT+glossary+IT docs) | IT vi→ja **+14.6**; câu khó +10.5; họp +7.9 | hội thoại ja→vi −1.0 (trong nhiễu, <2) | +0.1 / +1.2 | ~1.39 (mốc gốc) | **0** | Data ăn đúng đích → tiếp Vòng 2, chưa scale |
| 2 (19k→23k) | hội thoại, idiom, phủ định kép, ẩm thực, keigo | 13.858 (11 mode) | hội thoại vi→ja **+0.4** (17.7→18.1) ⇒ **T1** *(họp vi→ja +8.0, câu khó vi→ja +6.5 vẫn ăn)* | IT vi→ja **−3.7**; câu khó ja→vi −5.9; họp ja→vi −4.4 (replay 70%) ⇒ **T2** | +0.04 (21.49) ⇒ **T3** / −0.61 (42.27) | **1.384** (vòng 1 ~1.39, chênh 0.006) ⇒ **T4** | **4 (T1+T2+T3+T4)** | ≥2 tín hiệu cùng vòng → **scale from-scratch** (§5.1); user chốt **~292M** (d1152/16L/ff3072 — PLAN §3.6) thay 200M để dư sức chứa vòng 3-5. Chống dương tính giả T1: đã eyeball 10 cặp ht/idh ngẫu nhiên — data sạch |
| 3 | số liệu, phủ định, câu phức | | | | | | | |
