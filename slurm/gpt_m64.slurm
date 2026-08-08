#!/bin/bash
# =====================================================================
# KELEBEGIN SON SINAVI: dilim sayisi buyudugunde fark aciliyor mu?
#
# NEDEN GEREKLI
# Tum onceki deneyler M=16'daydi, yani P=8 dilim. Orada halka 4 katmanda,
# kelebek 3 katmanda karisimi tamamliyor - neredeyse ayni. Ve gercekten de
# Self-Guided kontrolunden sonra ikisi ESITLENDI (1.5 std hata).
# Yani kelebegi EN AZ avantajli noktasinda test etmisiz.
#
# M=64 -> P=32 dilim. Olculen karisim mesafesi (0 = mukemmel):
#   katman:      1      5      8     16
#   kelebek:  0.938  0.000  0.000  0.000    <- 5. katmanda tamamliyor
#   halka:    0.879  0.728  0.669  0.564    <- 16 katmanda bile yarida
#
# TAHMIN (veriden ONCE yaziliyor)
#   M=64, L=8  -> kelebek halkayi BELIRGIN gecmeli
#   M=64, L=16 -> yine gecmeli (halka P=32'de hicbir derinlikte karisamiyor)
#   fark M=16'dakinden (0.0024, anlamsiz) COK daha buyuk olmali
#   random (kapsama 1.00, aninda karisir) seyrekler arasi tavan olmali
#     yani beklenen sira:  random <= kelebek < halka   (bpc'de dusuk=iyi)
#
# TUTMAZSA: karisim mesafesi olcusu degersiz demektir ve kelebegin KALITE
# iddiasi tamamen duser. Geriye yalnizca iletisim iddiasi kalir - ki o da
# olculmus ve saglam. Yani bu sinav kaybedilse bile calisma ayakta kalir;
# sadece iddia daralir.
#
# Kontrol: Self-Guided acik (kilavuz). Cunku bir onceki deneyde halkanin
# optimizasyon zorlugunun sonucu carpittigini bulduk; o confound elenmis
# halde olculmeli.
# Kelebek --fast (bmm) ile kosuyor: M=64'te maskeli surum 16 kat bosa is
# yapardi. Kalite karsilastirmasini etkilemez (ayni fonksiyon, dogrulandi)
# ama tok/s sutunu kelebek icin digerleriyle kiyaslanamaz.
#
# sbatch slurm/gpt_m64.slurm
# =====================================================================
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg35
#SBATCH -C H200
#SBATCH --reservation=yzup
#SBATCH -J gpt_m64
#SBATCH -o logs/m64_%A_%a.out
#SBATCH -e logs/m64_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-17%8

set -euo pipefail
mkdir -p logs results

module load apps/truba-ai/gpu-2024.0

# "ffn derinlik tohum" - hepsi M=64 (P=32 dilim), init mixed, kilavuz acik
GRID=(
  "butterfly 8 10"  "butterfly 8 11"  "butterfly 8 12"
  "ring 8 10"       "ring 8 11"       "ring 8 12"
  "random 8 10"     "random 8 11"     "random 8 12"
  "butterfly 16 10" "butterfly 16 11" "butterfly 16 12"
  "ring 16 10"      "ring 16 11"      "ring 16 12"
  "random 16 10"    "random 16 11"    "random 16 12"
)

CFG=(${GRID[$SLURM_ARRAY_TASK_ID]})
FFN=${CFG[0]}; LAYER=${CFG[1]}; SEED=${CFG[2]}

# kelebek bmm ile; digerleri maskeli (onlar icin bmm karsiligi yok)
FAST=""
if [ "$FFN" = "butterfly" ]; then FAST="--fast"; fi

echo "== node=$(hostname) ffn=$FFN M=64 L=$LAYER seed=$SEED kilavuz=0.3 $FAST =="

python train_gpt.py \
  --data data \
  --ffn "$FFN" \
  --ffn-mult 64 \
  --seed "$SEED" \
  --init mixed \
  --guide-frac 0.3 \
  $FAST \
  --d-model 512 --n-layer "$LAYER" --n-head 8 \
  --block-size 512 --batch-size 32 \
  --iters 6000 \
  --out "results/gpt_${FFN}_m64_L${LAYER}_s${SEED}_mixed_kilavuz.json"
