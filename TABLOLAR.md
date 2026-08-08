# Tablolar (results/*.json'dan uretildi)

> bpc dusuk = iyi. sigma (o) = farkin standart hataya orani; 2o alti anlamsiz sayilir.


## T1 — Varyantlar ve kapsama (yogunluk 1/4, M=16)

| desen | tanim | kapsama |
|---|---|---|
| dense | yogun, gizli 4d | 1.000 |
| ring | poligon bandi | 0.498 |
| watts_strogatz | kucuk-dunya | 1.000 |
| random | rastgele seyrek (kontrol) | 1.000 |
| block | blok-kosegen | 0.250 |
| mozaik | satranc tahtasi | 0.250 |
| butterfly | kelebek (FFT) | 0.250 |

## T2 — Kalite: yogunluk taramasi (L=8, mixed init)

| desen | M | yogunluk | kapsama | val bpc | ± | dense'e fark | anlamli? |
|---|--:|--:|--:|--:|--:|--:|---|
| dense | 8 | 0.500 | 1.000 | 1.4466 | 0.0027 (n=9) | — |  |
| block | 8 | 0.500 | 0.500 | 1.4527 | 0.0030 (n=4) | +0.0061 | yogun iyi (3.5o) |
| mozaik | 8 | 0.500 | 0.500 | 1.4541 | 0.0029 (n=3) | +0.0075 | yogun iyi (3.9o) |
| ring | 8 | 0.500 | 0.998 | 1.4467 | 0.0022 (n=4) | +0.0001 | gurultu (0.1o) |
| watts_strogatz | 8 | 0.500 | 1.000 | 1.4485 | 0.0028 (n=4) | +0.0019 | gurultu (1.2o) |
| random | 8 | 0.500 | 1.000 | 1.4467 | 0.0039 (n=3) | +0.0001 | gurultu (0.0o) |
| block | 16 | 0.250 | 0.250 | 1.4923 | 0.0023 (n=3) | +0.0457 | yogun iyi (28.4o) |
| mozaik | 16 | 0.250 | 0.250 | 1.4925 | 0.0040 (n=3) | +0.0459 | yogun iyi (18.4o) |
| ring | 16 | 0.250 | 0.498 | 1.4752 | 0.0029 (n=6) | +0.0286 | yogun iyi (19.4o) |
| watts_strogatz | 16 | 0.250 | 1.000 | 1.4696 | 0.0036 (n=6) | +0.0230 | yogun iyi (13.3o) |
| random | 16 | 0.250 | 1.000 | 1.4682 | 0.0024 (n=6) | +0.0216 | yogun iyi (16.4o) |
| butterfly | 16 | 0.250 | 0.250 | 1.4765 | 0.0033 (n=3) | +0.0299 | yogun iyi (14.2o) |
| block | 32 | 0.125 | 0.125 | 1.5493 | 0.0000 (n=1) | +0.1028 | yogun iyi (116.2o) |
| ring | 32 | 0.125 | 0.248 | 1.5225 | 0.0000 (n=1) | +0.0759 | yogun iyi (85.8o) |
| watts_strogatz | 32 | 0.125 | 1.000 | 1.5063 | 0.0000 (n=1) | +0.0598 | yogun iyi (67.6o) |
| random | 32 | 0.125 | 1.000 | 1.5016 | 0.0000 (n=1) | +0.0550 | yogun iyi (62.2o) |
| watts_strogatz | 64 | 0.062 | 1.000 | 1.5378 | 0.0000 (n=1) | +0.0912 | yogun iyi (103.2o) |

## T3 — Derinlik: ring vs random farki (M=16, mixed, kilavuzsuz)

| L | dense | random | ring | butterfly | ring−random | anlamli? |
|--:|--:|--:|--:|--:|--:|---|
| 2 | 1.6338 | 1.5664 | 1.5736 | 1.5650 | +0.0073 | anlamli (2.8o) |
| 4 | 1.5285 | 1.5410 | 1.5428 | 1.5448 | +0.0017 | gurultu (0.1o) |
| 8 | 1.4466 | 1.4682 | 1.4752 | 1.4765 | +0.0070 | anlamli (4.6o) |
| 16 | 1.4024 | 1.4373 | 1.4521 | 1.4457 | +0.0149 | anlamli (20.3o) |

## T5 — Self-guided kilavuzun derinlige gore etkisi (M=16, mixed)

| L | desen | kilavuzsuz | kilavuzlu | kazanc | anlamli? |
|--:|---|--:|--:|--:|---|
| 2 | butterfly | 1.5650 | 1.5603 | +0.0047 | anlamli (3.3o) |
| 2 | random | 1.5664 | 1.5661 | +0.0002 | gurultu (0.1o) |
| 2 | ring | 1.5736 | 1.5673 | +0.0063 | gurultu (1.2o) |
| 4 | butterfly | 1.5448 | 1.5364 | +0.0084 | gurultu (0.9o) |
| 4 | random | 1.5410 | 1.5433 | -0.0023 | gurultu (0.3o) |
| 4 | ring | 1.5428 | 1.5444 | -0.0016 | gurultu (0.2o) |
| 8 | butterfly | 1.4765 | 1.4727 | +0.0038 | gurultu (1.8o) |
| 8 | random | 1.4682 | 1.4696 | -0.0014 | gurultu (0.9o) |
| 8 | ring | 1.4752 | 1.4724 | +0.0028 | gurultu (1.5o) |
| 16 | butterfly | 1.4457 | 1.4339 | +0.0118 | anlamli (13.5o) |
| 16 | random | 1.4373 | 1.4273 | +0.0099 | anlamli (4.9o) |
| 16 | ring | 1.4521 | 1.4363 | +0.0158 | anlamli (10.6o) |

## T6 — bmm uygulamasi: ayni kalite, farkli hiz (butterfly, mixed)

| L | M | uygulama | val bpc | k tok/s |
|--:|--:|---|--:|--:|
| 8 | 16 | masked | 1.4765 | 570 |
| 8 | 16 | bmm | 1.4758 | 309 |
| 16 | 16 | masked | 1.4457 | 297 |
| 16 | 16 | bmm | 1.4462 | 156 |
| 8 | 32 | bmm | 1.5198 | 185 |

## T4 — Dagitik iletisim (8×H200, d=8192, rank basina esit param)

| parcalama | iletisim MB/adim | muhatap | ileri geciş ms |
|---|--:|--:|--:|
| yogun tensor-paralel | 469.8 | 7 | 4.25 |
| halka | 134.2 | 2 | 2.83 |
| kelebek | 67.1 | 1 | 2.51 |
| blok | 0.0 | 0 | 1.90 |

## T7 — Olcekleme: 403M model (d=1024, L=32, enwik9 bayt, 2 tohum)

| desen | val bpc | ± | bellek GB | k tok/s |
|---|--:|--:|--:|--:|
| dense | 0.9096 | 0.0101 (n=2) | 38 | 771 |
| random | 0.9193 | 0.0011 (n=2) | 99 | 329 |
| butterfly | 0.9236 | 0.0003 (n=2) | 99 | 337 |
