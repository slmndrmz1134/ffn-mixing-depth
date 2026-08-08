#!/bin/bash
#SBATCH -p palamut-cuda
#SBATCH -A egitimg35
#SBATCH -J train_topo
#SBATCH -o logs/train_topo_%j.out
#SBATCH -e logs/train_topo_%j.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00

# kolyoz/H200 icin: -p kolyoz-cuda + -C H200 (ve varsa --reservation=yzup)

set -euo pipefail
mkdir -p logs results

module load apps/truba-ai/gpu-2024.0

echo "== node: $(hostname) =="
nvidia-smi --query-gpu=name --format=csv,noheader

# Iki gorevi de kosuyoruz: mixing yol uzunlugunu, teacher kapasiteyi olcer.
# Ayni deneyi 3 tohumla tekrarla - tek kosunun gurultusu yaniltici olur.
for TASK in mixing teacher; do
  for SEED in 0 1 2; do
    echo "=== task=${TASK} seed=${SEED} ==="
    python train_topology.py \
      --task "${TASK}" \
      --width 1024 \
      --depth 4 \
      --degree 16 \
      --steps 3000 \
      --seed "${SEED}" \
      --out "results/train_${TASK}_seed${SEED}_${SLURM_JOB_ID}.json"
  done
done
