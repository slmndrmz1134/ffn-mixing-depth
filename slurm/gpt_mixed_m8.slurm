#!/bin/bash
# =====================================================================
# SON IZGARA: hipotezin EN GUCLU hali, 3 tohumla.
#
# Neden bu kombinasyon: iki eksende de seyreklige en elverisli nokta.
#   - yogunluk 1/2 (M=8): taramada seyrekligin dense'e en cok yaklastigi yer
#   - init 'mixed'      : seyrek varyantlar icin olculen en iyi init
#                         (M=16 ws: mixed 1.4686 < kaiming 1.5266 < gpt 1.5470)
#
# Elimizde bu noktada sadece n=1 var ve o tek kosu dense'i -0.0042 gecmisti.
# kaiming'de 3 tohumla bakinca fark gurultuye karisti (1.15 standart hata).
# Ayni sinavi mixed'de de yapmadan "topoloji kaybetti" demek, hipotezi en
# zayif halinden yargilamak olur.
#
# dense tabani da AYNI kod surumu ve AYNI init ile yeniden kosuluyor: eski
# izgaradaki 'mixed' kosulari cikis projeksiyonunu yeniden ornekliyordu, yeni
# kod ise carpiyor. Iki surumu karistirmamak icin taban burada tazeleniyor.
#
# Son uc gorev (ws M=16) KOPRU: yeni kodun 'mixed' modu eski izgaranin
# 1.4686 degerini yeniden uretiyor mu? Uretiyorsa iki izgara birlestirilebilir.
#
# sbatch slurm/gpt_mixed_m8.slurm
# =====================================================================
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg35
#SBATCH -C H200
#SBATCH --reservation=yzup
#SBATCH -J gpt_m8
#SBATCH -o logs/m8_%A_%a.out
#SBATCH -e logs/m8_%A_%a.err
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
  "dense 4 0 mixed"  "dense 4 1 mixed"  "dense 4 2 mixed"
  "ring 8 0 mixed"   "ring 8 1 mixed"   "ring 8 2 mixed"
  "watts_strogatz 8 0 mixed" "watts_strogatz 8 1 mixed" "watts_strogatz 8 2 mixed"
  "random 8 0 mixed" "random 8 1 mixed" "random 8 2 mixed"
  "block 8 0 mixed"  "block 8 1 mixed"  "block 8 2 mixed"
  "watts_strogatz 16 0 mixed" "watts_strogatz 16 1 mixed" "watts_strogatz 16 2 mixed"
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
