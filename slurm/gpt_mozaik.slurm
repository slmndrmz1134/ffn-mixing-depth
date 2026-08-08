#!/bin/bash
# =====================================================================
# SAGLAMA DENEYI: mozaik (satranc tahtasi) deseni.
#
# Mozaik, gozle bakildiginda blok-kosegenin TAM ZITTIdir: delikler her yere
# esit dagilmis, hicbir bolge bos degil. Ama satir/sutunlari (i mod s)
# sinifina gore siralayinca birebir blok-kosegene donusuyor (kodda dogrulandi)
# ve kapsamasi ayni: M=8'de 0.50, M=16'da 0.25.
#
# Sinir aginda gizli birimlerin sirasi keyfi oldugu icin ikisi AYNI model.
#
# TAHMIN (veri gelmeden yaziliyor):
#   mozaik M=8  ~ block M=8  = 1.4527
#   mozaik M=16 ~ block M=16 = 1.4923
#
# Tutarsa: kapsama olcusunun deseni gozle degil dogru yerden okudugu dogrulanir.
# Tutmazsa: kurulumda bir hata var demektir ve onu bulmamiz gerekir.
# Her iki durumda da bilgi kazaniyoruz - iyi bir sinav boyle olur.
#
# sbatch slurm/gpt_mozaik.slurm
# =====================================================================
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg35
#SBATCH -C H200
#SBATCH --reservation=yzup
#SBATCH -J gpt_moz
#SBATCH -o logs/moz_%A_%a.out
#SBATCH -e logs/moz_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-5%6

set -euo pipefail
mkdir -p logs results

module load apps/truba-ai/gpu-2024.0

GRID=(
  "mozaik 8 0 mixed"  "mozaik 8 1 mixed"  "mozaik 8 2 mixed"
  "mozaik 16 0 mixed" "mozaik 16 1 mixed" "mozaik 16 2 mixed"
)

CFG=(${GRID[$SLURM_ARRAY_TASK_ID]})
FFN=${CFG[0]}; MULT=${CFG[1]}; SEED=${CFG[2]}; INIT=${CFG[3]}

echo "== node=$(hostname) ffn=$FFN mult=$MULT seed=$SEED init=$INIT =="

python train_gpt.py \
  --data data \
  --ffn "$FFN" \
  --ffn-mult "$MULT" \
  --seed "$SEED" \
  --init "$INIT" \
  --d-model 512 --n-layer 8 --n-head 8 \
  --block-size 512 --batch-size 32 \
  --iters 6000 \
  --out "results/gpt_${FFN}_m${MULT}_s${SEED}_${INIT}v2.json"
