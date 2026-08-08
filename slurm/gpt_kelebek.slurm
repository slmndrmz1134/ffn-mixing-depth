#!/bin/bash
# =====================================================================
# YENI DESEN: kelebek (butterfly).
#
# NEDEN
# Halka her katmanda AYNI komsuyla konusur; erisim katman basina +-1 dilim
# buyur ve karisim hicbir zaman tamamlanmaz (L=8'de duzgun dagilimdan uzaklik
# hala 0.12). Kelebek her katmanda BASKA biriyle eslesir - once 1 uzaktakiyle,
# sonra 2, sonra 4 - ve log2(P) katmanda karisim TAM olur. Katman basina
# muhatap sayisi yine 1'dir; degisen tek sey kiminle konusuldugu.
#
# Olculen karisim mesafesi (0 = mukemmel), yogunluk 1/4, 8 dilim:
#   katman:      1      2      3      8
#   kelebek:  0.750  0.500  0.000  0.000
#   halka:    0.563  0.429  0.342  0.119
#   blok:     0.750  0.750  0.750  0.750
#   rastgele: 0.015  0.007  0.008  0.008
#
# TAHMIN (veri gelmeden yaziliyor)
#   L=2  -> karisim eksik (0.50), kelebek random'dan BELIRGIN kotu olmali
#   L>=4 -> karisim tam (0.00), kelebek random ile ESIT olmali
#   L=3'te keskin bir esik olmali.
#
# Tutarsa: katman basina TEK muhatapla tam karisim elde edilmis olur - dagitik
# tarafta aranan sey tam olarak budur. Calismanin ilk pozitif sonucu olur.
# Tutmazsa: karisim mesafesi de dogru ongorucu degil demektir ve olctugumuz
# seyin ne oldugunu yeniden dusunmemiz gerekir.
#
# Karsilastirma tabani: ayni derinlikteki ring/random/dense kosulari
# (gpt_derinlik.slurm, tohum 10-12) - ayni kod, ayni donanim, ayni tohumlar.
#
# sbatch slurm/gpt_kelebek.slurm
# =====================================================================
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg35
#SBATCH -C H200
#SBATCH --reservation=yzup
#SBATCH -J gpt_kelebek
#SBATCH -o logs/kelebek_%A_%a.out
#SBATCH -e logs/kelebek_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --array=0-11%8

set -euo pipefail
mkdir -p logs results

module load apps/truba-ai/gpu-2024.0

# "derinlik tohum" - hepsi butterfly, M=16, init mixed
GRID=(
  "2 10"  "2 11"  "2 12"
  "4 10"  "4 11"  "4 12"
  "8 10"  "8 11"  "8 12"
  "16 10" "16 11" "16 12"
)

CFG=(${GRID[$SLURM_ARRAY_TASK_ID]})
LAYER=${CFG[0]}; SEED=${CFG[1]}

echo "== node=$(hostname) ffn=butterfly L=$LAYER seed=$SEED =="

python train_gpt.py \
  --data data \
  --ffn butterfly \
  --ffn-mult 16 \
  --seed "$SEED" \
  --init mixed \
  --d-model 512 --n-layer "$LAYER" --n-head 8 \
  --block-size 512 --batch-size 32 \
  --iters 6000 \
  --out "results/gpt_butterfly_m16_L${LAYER}_s${SEED}_mixed.json"
