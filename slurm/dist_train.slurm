#!/bin/bash
# =====================================================================
# BOSLUK DENEYI: parcalanmis modeli GERCEKTEN dagitik egit.
#
# Simdiye kadar kalite tek GPU'da (maskelerle), iletisim 8 GPU'da (egitimsiz)
# olculdu; ikisini baglayan sey bir argumandi. Bu is ikisini AYNI NESNE
# uzerinde olcer: 8 H200'de, ileri VE geri gecisle, gercek egitim.
#
# Once kisa duman testi, sonra tam kosu:
#   STEPS=100  sbatch slurm/dist_train.slurm     # ~3 dk, hata var mi diye
#   sbatch slurm/dist_train.slurm                # tam kosu
#
# NOT: dikkat katmani YOK, yalnizca FFN yigini. Sebep train_dist_ffn.py
# basindaki aciklamada: dikkat tum vektore ihtiyac duydugu icin "tek muhatap"
# ozelligini yok eder ve kazanci 4'te 1'e indirir. Ayri bir arastirma sorusu.
# =====================================================================
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg35
#SBATCH -C H200
#SBATCH --reservation=yzup
#SBATCH -J dist_train
#SBATCH -o logs/disttrain_%j.out
#SBATCH -e logs/disttrain_%j.err
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00

set -euo pipefail
mkdir -p logs results

module load apps/truba-ai/gpu-2024.0

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29518
export NCCL_DEBUG=WARN

STEPS="${STEPS:-3000}"
SEEDS="${SEEDS:-0 1 2}"

echo "== sunucular: $SLURM_JOB_NODELIST | master: $MASTER_ADDR | adim: $STEPS =="

for FFN in dense_tp ring butterfly block; do
  for SEED in $SEEDS; do
    echo ""
    echo "######## ffn=$FFN seed=$SEED ########"
    srun --kill-on-bad-exit=1 python train_dist_ffn.py \
      --data data \
      --ffn "$FFN" \
      --d-model 512 --ctx 16 --layers 8 --ffn-mult 4 \
      --batch 512 --steps "$STEPS" --seed "$SEED" \
      --out "results/disttrain_${FFN}_s${SEED}.json"
  done
done

echo ""
echo "=== ozet ==="
python - << 'PY'
import json, glob, statistics as st
from collections import defaultdict
g = defaultdict(list)
for f in sorted(glob.glob("results/disttrain_*.json")):
    r = json.loads(open(f).read())
    g[r["args"]["ffn"]].append(r)
print(f"{'varyant':11s}{'val bpc':>10s}{'+-':>8s}{'muhatap':>9s}"
      f"{'iletisim MB/adim':>18s}{'sure dk':>9s}{'par/rank':>11s}{'n':>3s}")
print("-"*79)
for k in ("dense_tp", "ring", "butterfly", "block"):
    if k not in g: continue
    rs = g[k]; b = [r["val_bpc"] for r in rs]
    print(f"{k:11s}{st.mean(b):10.4f}"
          f"{(st.stdev(b) if len(b)>1 else 0):8.4f}{rs[0]['peers']:9d}"
          f"{rs[0]['comm_mb_fwd_bwd_per_step']:18.2f}"
          f"{st.mean(r['train_seconds'] for r in rs)/60:9.2f}"
          f"{rs[0]['params_per_rank']:11,d}{len(rs):3d}")
PY
