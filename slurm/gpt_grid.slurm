#!/bin/bash
# =====================================================================
# GPT-mini topoloji izgarasi - 8x H200 rezervasyonunu paralel doldurur.
#
# Her array gorevi 1 GPU alir; %8 ile ayni anda 8 gorev koser.
# 8 gorev x 16 cekirdek = 128 cekirdek = rezervasyondaki 2 sunucu (kolyoz36-37).
#
# Kuyruga ver:   sbatch slurm/gpt_grid.slurm
# Izle:          squeue -u $USER
# =====================================================================
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg35
#SBATCH -C H200
#SBATCH --reservation=yzup
#SBATCH -J gpt_grid
#SBATCH -o logs/gpt_%A_%a.out
#SBATCH -e logs/gpt_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-22%8

set -euo pipefail
mkdir -p logs results

module load apps/truba-ai/gpu-2024.0

# --- deney izgarasi: "ffn mult seed" ---------------------------------
# 1) ANA KARSILASTIRMA: 5 varyant x 3 tohum, hepsi mult=16 (yogunluk 1/4)
#    Tek kosunun gurultusu yaniltir; 3 tohum sart.
# 2) YOGUNLUK TARAMASI: seyreklik arttikca kalite nerede kiriliyor?
GRID=(
  "dense 4 0"  "dense 4 1"  "dense 4 2"
  "ring 16 0"  "ring 16 1"  "ring 16 2"
  "watts_strogatz 16 0" "watts_strogatz 16 1" "watts_strogatz 16 2"
  "random 16 0" "random 16 1" "random 16 2"
  "block 16 0"  "block 16 1"  "block 16 2"
  "watts_strogatz 8 0"  "watts_strogatz 32 0" "watts_strogatz 64 0"
  "ring 8 0"            "ring 32 0"
  "block 8 0"           "block 32 0"
  "random 32 0"
)

CFG=(${GRID[$SLURM_ARRAY_TASK_ID]})
FFN=${CFG[0]}; MULT=${CFG[1]}; SEED=${CFG[2]}

echo "== node=$(hostname) gorev=$SLURM_ARRAY_TASK_ID ffn=$FFN mult=$MULT seed=$SEED =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train_gpt.py \
  --data data \
  --ffn "$FFN" \
  --ffn-mult "$MULT" \
  --seed "$SEED" \
  --d-model 512 --n-layer 8 --n-head 8 \
  --block-size 512 --batch-size 32 \
  --iters 6000 \
  --out "results/gpt_${FFN}_m${MULT}_s${SEED}.json"
