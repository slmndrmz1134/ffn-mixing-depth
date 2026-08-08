#!/bin/bash
# =====================================================================
# KONTROL DENEYI: kelebegin dezavantaji YAPISAL mi, OPTIMIZASYON mu?
#
# SORU
# Kelebek, random'a L=16'da 15.7 standart hata geride (1.4457 vs 1.4373).
# Bunu bir IFADE GUCU siniri diye yorumluyoruz. Ama literaturde (Wei ve ark.,
# NeurIPS 2024) yapilandirilmis FFN'lerin sifirdan egitilirken TAKILDIGI
# gosterilmis: baslangictaki simetri kisitlari optimizasyonu kotu bir yola
# sokuyor ve model yapabilecegini ogrenemiyor.
#
# YONTEM (Self-Guided Training)
# Egitimin ilk %30'unda FFN'e PARALEL bir yogun dal eklenir, katsayisi 1'den
# 0'a iner. alpha=0 olunca dal tamamen cikar - kodda dogrulandi: model o anda
# kilavuzsuz surumle BIREBIR ayni (fark 0.00e+00). Yani NIHAI model parametre
# olarak degismez; ek kapasite yalnizca egitim sirasinda, gecicidir.
# Bu, raporda acikca belirtilmesi gereken bir noktadir.
#
# TASARIM
# Kilavuz TUM seyrek varyantlara uygulanir (butterfly/random/ring), dense'e
# uygulanmaz - dense zaten kilavuzun kendisi. Boylece seyrekler arasi
# karsilastirma adil kalir.
# Tohumlar 10-12: kilavuzsuz kosular da ayni tohumlarla yapildi, yani
# karsilastirma ESLESTIRILMIS (paired) olur - gurultu azalir.
#
# UC OLASI SONUC, UCU DE BILGI
#   acik kapanir  -> dezavantaj optimizasyon kaynakliymis. "Tek muhatapla
#                    random kalitesi" denebilir; dagitik iddia cok guclenir.
#   acik kalir    -> sinir gercekten yapisal. Bilinen cozum denendi ve elendi.
#   acik buyur    -> kilavuz seyrek desene zarar veriyor; beklenmedik, incelenir.
#
# sbatch slurm/gpt_kilavuz.slurm
# =====================================================================
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg35
#SBATCH -C H200
#SBATCH --reservation=yzup
#SBATCH -J gpt_kilavuz
#SBATCH -o logs/kilavuz_%A_%a.out
#SBATCH -e logs/kilavuz_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --array=0-17%8

set -euo pipefail
mkdir -p logs results

module load apps/truba-ai/gpu-2024.0

# "ffn derinlik tohum" - hepsi M=16, init mixed, kilavuz acik
GRID=(
  "butterfly 8 10"  "butterfly 8 11"  "butterfly 8 12"
  "random 8 10"     "random 8 11"     "random 8 12"
  "ring 8 10"       "ring 8 11"       "ring 8 12"
  "butterfly 16 10" "butterfly 16 11" "butterfly 16 12"
  "random 16 10"    "random 16 11"    "random 16 12"
  "ring 16 10"      "ring 16 11"      "ring 16 12"
)

CFG=(${GRID[$SLURM_ARRAY_TASK_ID]})
FFN=${CFG[0]}; LAYER=${CFG[1]}; SEED=${CFG[2]}

echo "== node=$(hostname) ffn=$FFN L=$LAYER seed=$SEED kilavuz=0.3 =="

python train_gpt.py \
  --data data \
  --ffn "$FFN" \
  --ffn-mult 16 \
  --seed "$SEED" \
  --init mixed \
  --guide-frac 0.3 \
  --d-model 512 --n-layer "$LAYER" --n-head 8 \
  --block-size 512 --batch-size 32 \
  --iters 6000 \
  --out "results/gpt_${FFN}_m16_L${LAYER}_s${SEED}_mixed_kilavuz.json"
