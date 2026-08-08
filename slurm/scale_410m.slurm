#!/bin/bash
# =====================================================================
# OLCEKLEME DENEYI: 410M model, dense vs butterfly vs random, 8 GPU DDP.
#
# d=1024 L=32 head=16 M=16 -> ~403M parametre, butterfly P=8.
# Her varyant AYNI token butcesi (~2B), AYNI global batch. Tek degisen FFN.
#
# random = kalite TAVANI (kapsama 1.00, ama dagitilamaz - all-to-all).
# butterfly = aday (dagitilabilir, 1 muhatap). Soru: butterfly bu OLCEKTE
#   random'i yakaliyor mu? Kucuk olcekte 15o geride, L=16'da 3-4o'ya daralmisti.
# dense = ust referans.
#
# Her array gorevi 8 GPU'nun HEPSINI kullanir (veri-paralel), o yuzden %1:
# ayni anda tek gorev. 6 gorev (3 varyant x 2 tohum) sirayla koser.
#
# Kilavuz YOK (DDP'de kullanilmayan-parametre sorunu); seyrek sayilar
# muhafazakar. bmm YOK (butterfly de masked, random ile ayni hesap profili).
#
#   sbatch slurm/scale_410m.slurm
# =====================================================================
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg35
#SBATCH -C H200
#SBATCH --reservation=yzup
#SBATCH -J scale410
#SBATCH -o logs/scale_%A_%a.out
#SBATCH -e logs/scale_%A_%a.err
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --array=0-5%1

set -euo pipefail
mkdir -p logs results
module load apps/truba-ai/gpu-2024.0
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29531
export NCCL_DEBUG=WARN

GRID=("dense 0" "dense 1" "butterfly 0" "butterfly 1" "random 0" "random 1")
CFG=(${GRID[$SLURM_ARRAY_TASK_ID]})
FFN=${CFG[0]}; SEED=${CFG[1]}

echo "== sunucular: $SLURM_JOB_NODELIST | ffn=$FFN seed=$SEED =="
srun --kill-on-bad-exit=1 python train_ddp.py \
  --data data_enwik9 --ffn "$FFN" --ffn-mult 16 --seed "$SEED" \
  --d-model 1024 --n-layer 32 --n-head 16 --ctx 1024 --batch 24 \
  --steps 10000 --warmup 400 --eval-every 1000 \
  --out "results/scale_${FFN}_s${SEED}.json"
