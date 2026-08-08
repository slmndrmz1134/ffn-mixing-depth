#!/bin/bash
# =====================================================================
# 3. olcek: dagitik parcalama karsilastirmasi - 2 sunucu x 4 GPU = 8 H200.
#
# Kasitli olarak IKI sunucu isteniyor: tek sunucuda her sey NVLink uzerinden
# gider ve iletisim bedavaya yakin gorunur. Asil soru sunucular ARASINDAKI
# hatta ne oldugu; 2 sunucu istemek InfiniBand'i olcume sokar.
#
# sbatch slurm/distributed.slurm
# =====================================================================
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg35
#SBATCH -C H200
#SBATCH --reservation=yzup
#SBATCH -J dist_topo
#SBATCH -o logs/dist_%j.out
#SBATCH -e logs/dist_%j.err
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00

set -euo pipefail
mkdir -p logs results

module load apps/truba-ai/gpu-2024.0

# rank 0'in adresi: tahsis edilen ilk sunucu
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29517
export NCCL_DEBUG=WARN

echo "== sunucular: $SLURM_JOB_NODELIST | master: $MASTER_ADDR =="

# d_model'i buyutmek iletisim/hesap oranini gercekci tutar: kucuk modelde
# her sey onbelleğe sigar ve iletisim yapay olarak ucuz gorunur.
for D in 2048 4096 8192; do
  echo ""
  echo "############ d_model=${D} ############"
  srun --kill-on-bad-exit=1 python bench_distributed.py \
    --d-model "$D" \
    --hidden-mult 4 \
    --tokens 8192 \
    --iters 20 \
    --out "results/dist_d${D}_${SLURM_JOB_ID}.json"
done
