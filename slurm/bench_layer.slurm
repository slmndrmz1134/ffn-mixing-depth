#!/bin/bash
#SBATCH -p palamut-cuda
#SBATCH -A egitimg35
#SBATCH -J bench_layer
#SBATCH -o logs/bench_layer_%j.out
#SBATCH -e logs/bench_layer_%j.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00

# NOT: kolyoz/H200'de kosmak icin ustteki iki satiri degistir:
#   #SBATCH -p kolyoz-cuda
#   #SBATCH -C H200
# ve (rezervasyona eklendiginde) #SBATCH --reservation=yzup ekle.
# GPU basina 16 cekirdek kurali her iki kuyrukta da gecerli.

set -euo pipefail
mkdir -p logs results

module load apps/truba-ai/gpu-2024.0

echo "== node: $(hostname) =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python bench_layer.py \
  --width 4096 \
  --batch 256 \
  --degrees 6 16 64 256 \
  --iters 50 \
  --out "results/bench_layer_${SLURM_JOB_ID}.json"
