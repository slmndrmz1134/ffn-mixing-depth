#!/bin/bash
# =====================================================================
# KESIF DENEYI: derinlik, zayif karisimi kurtariyor mu?
#
# SORU
# Bant (ring) topolojisi tek bir FFN icinde girdilerin ancak yarisina ulasir
# (katman ici kapsama 0.50). Ama katmanlar ust uste binince erisim buyumeli:
# her katman +-1 komsu ilerletiyorsa, L katman sonra +-L'ye ulasilir. Naif
# hesap tam karisim icin L >= 1/yogunluk diyor; yogunluk 1/4 icin L >= 4.
#
# TEORIYE TERS IPUCU (elimizde zaten var)
# L=8 ile kosulan M=16 deneyinde ring 1.4737, random 1.4671 idi. Eshik 4 iken
# derinlik 8'di, yani teoriye gore fark kapanmaliydi. KAPANMADI: +0.0066,
# 17 standart hata. Demek ki naif derinlik argumani eksik.
#
# BU DENEY
# Derinligi 2 / 4 / 8 / 16 tarayip ring ile random arasindaki farkin nasil
# degistigini olcer. Karsilastirma her zaman AYNI DERINLIKTEKI random ve
# dense'e karsi - derin model zaten daha iyi oldugu icin mutlak bpc'ler
# derinlikler arasi kiyaslanamaz, sadece FARK kiyaslanir.
#
# TAHMIN YOK - bu bir kesif. Uc olasi sonuc, ucu de bilgi:
#   fark derinlikle kapaniyor  -> derinlik karisimi geri getiriyor, naif
#                                 teori dogru, sadece esik 4'ten buyuk
#   fark sabit kaliyor         -> derinlik karisimi geri GETIRMIYOR; dikkat
#                                 katmani zaten yogun oldugu icin FFN'in
#                                 karistirma eksigi baska yoldan telafi
#                                 edilemiyor
#   fark aciliyor              -> derin agda zayif karisim birikiyor
#
# sbatch slurm/gpt_derinlik.slurm
# =====================================================================
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg35
#SBATCH -C H200
#SBATCH --reservation=yzup
#SBATCH -J gpt_derin
#SBATCH -o logs/derin_%A_%a.out
#SBATCH -e logs/derin_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --array=0-35%8

set -euo pipefail
mkdir -p logs results

module load apps/truba-ai/gpu-2024.0

# "ffn derinlik tohum" - hepsi M=16 (yogunluk 1/4), init mixed.
# Tohumlar 10-12: L=8 icin elimizde zaten 0-2 var. Ayni tohumu tekrarlamak
# ozdes sonuc uretip varyansi yapay kucultur; 10-12 BAGIMSIZ ornek ekler.
GRID=(
  "dense 2 10" "dense 2 11" "dense 2 12"
  "ring 2 10" "ring 2 11" "ring 2 12"
  "random 2 10" "random 2 11" "random 2 12"
  "dense 4 10" "dense 4 11" "dense 4 12"
  "ring 4 10" "ring 4 11" "ring 4 12"
  "random 4 10" "random 4 11" "random 4 12"
  "dense 8 10" "dense 8 11" "dense 8 12"
  "ring 8 10" "ring 8 11" "ring 8 12"
  "random 8 10" "random 8 11" "random 8 12"
  "dense 16 10" "dense 16 11" "dense 16 12"
  "ring 16 10" "ring 16 11" "ring 16 12"
  "random 16 10" "random 16 11" "random 16 12"
)

CFG=(${GRID[$SLURM_ARRAY_TASK_ID]})
FFN=${CFG[0]}; LAYER=${CFG[1]}; SEED=${CFG[2]}

# dense M'yi yok sayar (hep 4); seyrekler M=16 ile yogunluk 1/4 alir
MULT=16

echo "== node=$(hostname) ffn=$FFN L=$LAYER seed=$SEED =="

python train_gpt.py \
  --data data \
  --ffn "$FFN" \
  --ffn-mult "$MULT" \
  --seed "$SEED" \
  --init mixed \
  --d-model 512 --n-layer "$LAYER" --n-head 8 \
  --block-size 512 --batch-size 32 \
  --iters 6000 \
  --out "results/gpt_${FFN}_m${MULT}_L${LAYER}_s${SEED}_mixed.json"
