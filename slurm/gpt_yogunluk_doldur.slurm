#!/bin/bash
# =====================================================================
# YOGUNLUK TARAMASINI DOLDURMA + tau'nun EN KESKIN TESTI
#
# NEDEN
# Tablo tab:kalite'de M=32 satirlarinin sigma sutunu "---". Cunku o
# yogunlukta her desenden TEK kosu var (n=1). M=64'te ise yalnizca
# kucuk-dunya var. 301 kosuluk, "coklu tohum + anlamlilik testi" iddiali
# bir makalede n=1 satirlari birakmak tutarsizdir. Ayrica KELEBEK bu iki
# yogunlukta hic kosulmamis - oysa dagitik bolumun bastigi desen odur.
#
# ONCEDEN ILAN EDILEN TAHMINLER (kosudan once, cumulative_coverage ile)
#
#   M=32:  random tau=1  ws tau=1  butterfly tau=4  ring tau=5  block tau=>L
#   M=64:  random tau=1  ws tau=1  butterfly tau=5  ring tau=>L block tau=>L
#
# 1) TEKRAR TESTI - kapsama ayni, tau farkli:
#    M=32'de kelebek ile blogun katman ici kapsamasi AYNI (0.125), ama
#    kelebek tau=4 ile tamamlar, blok asla tamamlayamaz. M=16'da bu fark
#    9.6 sigma cikmisti (kelebek 1.4766, blok 1.4923). Ayni ayrisma M=32'de
#    de gorulmeli. Bu, makalenin merkezi iddiasinin BAGIMSIZ tekrari.
#      ayrisirsa -> iddia tek bir yogunluga ozgu degil
#      ayrismazsa -> tau'nun gecerlilik alani M=16 ile sinirli, oyle yazilir
#
# 2) SIRALAMA CELISKISI - iki olcunun TERS hukum verdigi tek hucre:
#    M=64'te halkanin kapsamasi (0.123) kelebeginkinin (0.062) IKI KATI.
#    Kapsamaya bakarsan halka onde olmali. Ama halka 8 katmanda karismasini
#    BITIREMIYOR (C_eff=0.971, tau>L), kelebek bitiriyor (tau=5).
#      kelebek onde cikarsa  -> tau kapsamayi siralamada da yeniyor. Bu,
#                               yapabilecegimiz en guclu tek testtir.
#      halka onde cikarsa    -> tau yaniltiyor; kapsama geri gelir ve
#                               makalenin tezi ciddi bicimde zayiflar.
#    Iki sonuc da yayimlanabilir; onemli olan onceden soylemis olmak.
#
# NOT: mozaik bilerek disarida - blogun satir/sutun permutasyonu oldugu
# icin ayni model sinifini tanimlar, yeni bilgi tasimaz.
#
# sbatch slurm/gpt_yogunluk_doldur.slurm
# =====================================================================
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg35
#SBATCH -C H200
#SBATCH --reservation=yzup
#SBATCH -J gpt_dold
#SBATCH -o logs/doldur_%A_%a.out
#SBATCH -e logs/doldur_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --array=0-24%8

set -euo pipefail
mkdir -p logs results

module load apps/truba-ai/gpu-2024.0

# --- izgara: "ffn M tohum" -----------------------------------------
# Yalnizca EKSIK hucreler. Mevcut kosular (M=32 tohum 0, M=64 ws tohum 0)
# tekrar kosulmuyor; ayni tohum ozdes sonuc verip varyansi yapay kucultur.
GRID=(
  # M=32: dordu n=1'den n=3'e, kelebek sifirdan
  "random 32 1"          "random 32 2"
  "watts_strogatz 32 1"  "watts_strogatz 32 2"
  "ring 32 1"            "ring 32 2"
  "block 32 1"           "block 32 2"
  "butterfly 32 0"       "butterfly 32 1"       "butterfly 32 2"
  # M=64: neredeyse hepsi sifirdan
  "random 64 0"          "random 64 1"          "random 64 2"
  "watts_strogatz 64 1"  "watts_strogatz 64 2"
  "ring 64 0"            "ring 64 1"            "ring 64 2"
  "block 64 0"           "block 64 1"           "block 64 2"
  "butterfly 64 0"       "butterfly 64 1"       "butterfly 64 2"
)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#GRID[@]}" ]; then
  echo "gorev ${SLURM_ARRAY_TASK_ID} izgara disinda (izgara=${#GRID[@]}); atlaniyor."
  exit 0
fi

CFG=(${GRID[$SLURM_ARRAY_TASK_ID]})
FFN=${CFG[0]}; MULT=${CFG[1]}; SEED=${CFG[2]}

echo "== node=$(hostname) ffn=$FFN M=$MULT seed=$SEED (izgara ${#GRID[@]}) =="

python train_gpt.py \
  --data data \
  --ffn "$FFN" \
  --ffn-mult "$MULT" \
  --seed "$SEED" \
  --init mixed \
  --d-model 512 --n-layer 8 --n-head 8 \
  --block-size 512 --batch-size 32 \
  --iters 6000 \
  --out "results/gpt_${FFN}_m${MULT}_s${SEED}_mixed.json"
