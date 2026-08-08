#!/bin/bash
# =====================================================================
# KELEBEGIN bmm UYGULAMASI: sayilan FLOP'u OLCULEN hiza cevirmek.
#
# Simdiye kadar kelebegi MaskedLinear ile kosturduk: tam yogun agirlik +
# maske + yogun matmul. Sifirlar da hesaplanip atiliyordu, yani yogunluk
# 1/4'te 4 kat bosa is. Kalite karsilastirmasi bundan etkilenmiyordu (tum
# varyantlar ayni sekilde taklit ediliyordu) ama HIZ hakkinda tek kelime
# edemiyorduk.
#
# ButterflyLinear bunu duzeltir: cikti dilimi r yalnizca {r, r XOR bit}
# girdi dilimlerini okudugu icin P tane kucuk yogun matmul yeter (torch.bmm).
# Ayni fonksiyon oldugu kodda dogrulandi (maks fark 4.3e-06, nnz ozdes).
#
# BU KOSU IKI SEY OLCER
#   1) KALITE: bmm surumu maskeli surumle ayni bpc'yi vermeli. Vermezse
#      uygulamada hata var demektir - saglama noktasi.
#   2) HIZ: tok/s ne kadar artiyor? Iddiayi sayilan degil olculen hale getirir.
#
# Tohumlar 10-12 -> maskeli kosularla ESLESTIRILMIS karsilastirma.
#
# sbatch slurm/gpt_bmm.slurm
# =====================================================================
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg35
#SBATCH -C H200
#SBATCH --reservation=yzup
#SBATCH -J gpt_bmm
#SBATCH -o logs/bmm_%A_%a.out
#SBATCH -e logs/bmm_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-8%8

set -euo pipefail
mkdir -p logs results

module load apps/truba-ai/gpu-2024.0

# "M derinlik tohum" - hepsi butterfly, init mixed, bmm acik
GRID=(
  "16 8 10"  "16 8 11"  "16 8 12"
  "16 16 10" "16 16 11" "16 16 12"
  "32 8 10"  "32 8 11"  "32 8 12"
)

CFG=(${GRID[$SLURM_ARRAY_TASK_ID]})
MULT=${CFG[0]}; LAYER=${CFG[1]}; SEED=${CFG[2]}

echo "== node=$(hostname) ffn=butterfly M=$MULT L=$LAYER seed=$SEED bmm=acik =="

python train_gpt.py \
  --data data \
  --ffn butterfly \
  --ffn-mult "$MULT" \
  --seed "$SEED" \
  --init mixed \
  --fast \
  --d-model 512 --n-layer "$LAYER" --n-head 8 \
  --block-size 512 --batch-size 32 \
  --iters 6000 \
  --out "results/gpt_butterfly_m${MULT}_L${LAYER}_s${SEED}_mixed_bmm.json"
