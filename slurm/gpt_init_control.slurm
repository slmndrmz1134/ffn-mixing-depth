#!/bin/bash
# =====================================================================
# KONTROL DENEYI: ana izgaradaki sonuc baslangic olceginin eseri mi?
#
# Ana izgarada seyrek katmanlar GELU'ya yogun katmandan 3.1x daha sicak
# sinyal sokuyordu (up cikti std: seyrek 1.41, yogun 0.45). Cunku yogun
# nn.Linear GPT gelenegine gore sabit std=0.02 aliyor, MaskedLinear ise
# kendi fan_in'ine gore olcekleniyordu. Bu, olculen +0.019 bpc farkini
# tek basina aciklayabilir.
#
# --init kaiming: her iki grupta da std = sqrt(2/fan_in). Olculdu, uc
# varyantta da up cikti std = 1.4132, yani sinyal olcegi ESIT.
#
# Sonuc iki init modunda da ayni yonde cikarsa bulgu saglamdir.
# Ters donerse, olctugumuz sey topoloji degil baslangic olcegiymis demektir.
#
# sbatch slurm/gpt_init_control.slurm
# =====================================================================
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg35
#SBATCH -C H200
#SBATCH --reservation=yzup
#SBATCH -J gpt_ctl
#SBATCH -o logs/init_%A_%a.out
#SBATCH -e logs/init_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-35%8

set -euo pipefail
mkdir -p logs results

module load apps/truba-ai/gpu-2024.0

# IKI init modu da BU kod surumuyle kosuluyor: karsilastirma kendi icinde
# kapali, eski izgaranin kod surumune bagimli degil. Tek degisen degisken init.
#   gpt     : her yerde sabit std=0.02  -> seyrek sinyal ZAYIF (0.23 vs 0.45)
#   kaiming : her yerde sqrt(2/fan_in)  -> sinyal olcegi ESIT  (1.41 = 1.41)
GRID=(
  "dense 4 0 gpt"  "dense 4 1 gpt"  "dense 4 2 gpt"
  "watts_strogatz 16 0 gpt" "watts_strogatz 16 1 gpt" "watts_strogatz 16 2 gpt"
  "ring 16 0 gpt"  "ring 16 1 gpt"  "ring 16 2 gpt"
  "block 16 0 gpt" "block 16 1 gpt" "block 16 2 gpt"
  "dense 4 0 kaiming"  "dense 4 1 kaiming"  "dense 4 2 kaiming"
  "watts_strogatz 16 0 kaiming" "watts_strogatz 16 1 kaiming" "watts_strogatz 16 2 kaiming"
  "ring 16 0 kaiming"  "ring 16 1 kaiming"  "ring 16 2 kaiming"
  "block 16 0 kaiming" "block 16 1 kaiming" "block 16 2 kaiming"
  # --- SINIR BOLGESI: yogunluk 1/2. Calismanin TEK pozitif sonucu burada
  # (ring/ws dense'i -0.004 bpc gecti) ama tek tohumla olculdu ve fark
  # tohum std'sinin ancak 1-2 kati. 3 tohumla sinaniyor. dense tabani
  # yukaridaki "dense 4 * kaiming" kosulari; dense M'den bagimsiz.
  "ring 8 0 kaiming" "ring 8 1 kaiming" "ring 8 2 kaiming"
  "watts_strogatz 8 0 kaiming" "watts_strogatz 8 1 kaiming" "watts_strogatz 8 2 kaiming"
  "random 8 0 kaiming" "random 8 1 kaiming" "random 8 2 kaiming"
  "block 8 0 kaiming" "block 8 1 kaiming" "block 8 2 kaiming"
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
  --out "results/gpt_${FFN}_m${MULT}_s${SEED}_${INIT}.json"
