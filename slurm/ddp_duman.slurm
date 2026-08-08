#!/bin/bash
# =====================================================================
# DUMAN TESTI: DDP + ozel katmanlar (masked/butterfly) 8 GPU'da calisiyor mu?
# Kucuk model, 60 adim, uc varyant. Amac SONUC degil, KILITLENMEDEN kosmasi.
# Asil 28 saatlik kosuyu baslatmadan once bunu gormek sart.
#   sbatch slurm/ddp_duman.slurm
# =====================================================================
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg35
#SBATCH -C H200
#SBATCH --reservation=yzup
#SBATCH -J ddp_duman
#SBATCH -o logs/ddpduman_%j.out
#SBATCH -e logs/ddpduman_%j.err
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00

set -euo pipefail
mkdir -p logs results
module load apps/truba-ai/gpu-2024.0
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29531
export NCCL_DEBUG=WARN
echo "== sunucular: $SLURM_JOB_NODELIST | master: $MASTER_ADDR =="

for FFN in dense butterfly random; do
  echo ""; echo "######## $FFN ########"
  srun --kill-on-bad-exit=1 python train_ddp.py \
    --data data_enwik9 --ffn "$FFN" --ffn-mult 16 \
    --d-model 256 --n-layer 4 --n-head 4 --ctx 256 --batch 8 \
    --steps 60 --warmup 10 --eval-every 30 \
    --out "results/ddpduman_${FFN}.json"
done
echo ""; echo "DUMAN TESTI BITTI - ucu de kilitlenmeden kostuysa asil kosuya gecebiliriz."
