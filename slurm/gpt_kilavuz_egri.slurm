#!/bin/bash
# =====================================================================
# KILAVUZ KAZANCININ DERINLIK EGRISI - iki noktayi dorde cikarmak.
#
# Elimizde L=8 (etki yok, 0.8-1.8 std hata) ve L=16 (etki var, 5.0-13.4)
# var. Iki noktadan gecen sonsuz egri vardir; "derinlikle buyuyor" demek
# icin ara noktalar gerekir.
#
# Bu izgara L=2 ve L=4'te kilavuzu kosar. Kilavuzsuz karsiliklari zaten
# elimizde ve AYNI tohumlarla (10-12), yani karsilastirma eslestirilmis.
#
# Sonucta dort noktali bir egri olur:
#   L=2  ->  ?
#   L=4  ->  ?
#   L=8  ->  0.8-1.8 std hata (etki yok)     [olculdu]
#   L=16 ->  5.0-13.4 std hata (etki var)    [olculdu]
#
# DIKKAT - raporda dogru dil: kilavuzsuz derin modellerde plato/iraksama
# GOZLENMEDI. Uc desen de derinlestikce DUZELDI (or. butterfly 1.4765 ->
# 1.4457). Kilavuz bunu 0.010-0.016 daha iyilestirdi, yani ~%1. Bu bir
# "kurtarma" degil, olculebilir bir iyilesmedir. Mekanizma olculmedi.
#
# sbatch slurm/gpt_kilavuz_egri.slurm
# =====================================================================
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg35
#SBATCH -C H200
#SBATCH --reservation=yzup
#SBATCH -J gpt_kegri
#SBATCH -o logs/kegri_%A_%a.out
#SBATCH -e logs/kegri_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-17%8

set -euo pipefail
mkdir -p logs results

module load apps/truba-ai/gpu-2024.0

GRID=(
  "butterfly 2 10"  "butterfly 2 11"  "butterfly 2 12"
  "random 2 10"     "random 2 11"     "random 2 12"
  "ring 2 10"       "ring 2 11"       "ring 2 12"
  "butterfly 4 10"  "butterfly 4 11"  "butterfly 4 12"
  "random 4 10"     "random 4 11"     "random 4 12"
  "ring 4 10"       "ring 4 11"       "ring 4 12"
)

CFG=(${GRID[$SLURM_ARRAY_TASK_ID]})
FFN=${CFG[0]}; LAYER=${CFG[1]}; SEED=${CFG[2]}

echo "== node=$(hostname) ffn=$FFN M=16 L=$LAYER seed=$SEED kilavuz=0.3 =="

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
