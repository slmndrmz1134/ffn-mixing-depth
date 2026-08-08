#!/bin/bash
# =====================================================================
# KONTROL DENEYI: "yapisiz rastgele en iyi" bulgusu, kontrol grubunun
# kendisi derinlikle degistigi icin mi boyle cikiyor?
#
# SORUN
# GPTMini her katmana farkli tohum veriyor (gpt.py, 'seed + 17*i'). Amac
# varyans azaltmaydi: tek bir sansli/sanssiz graf tum sonucu belirlemesin.
# YAN ETKISI fark edilmemis: tohumu TUKETEN desenler (random, watts_strogatz)
# boylece katmandan katmana YENIDEN cizilir. Kelebek de stage=layer_idx ile
# zaten oyle. Yani:
#
#   random         -> hem kapsamasi 1.0  HEM katmanlar arasi degisken
#   watts_strogatz -> hem kapsamasi 1.0  HEM katmanlar arasi degisken
#   ring/block/mozaik -> sabit
#
# Sonuc: "rastgele en iyi" (1. bulgu) ile "derinlikle degismek iyidir"
# (2. bulgu) AYNI kosulardan okunuyor; birbirinden ayrisamiyorlar.
#
# BU DENEY
# --fixed-mask bayragi topolojiyi TUM katmanlarda sabitler (tohum sabit,
# kelebek stage'i 0'a donuk). Tohumlar ARASINDA graf hala degisir, yani
# varyans azaltma korunur. Degisen tek sey derinlik eksenidir.
#
# ONCEDEN ILAN EDILEN TAHMINLER (kosudan once hesaplandi, cumulative_coverage
# ile; tau = tam karisma icin gereken katman sayisi):
#
#   desen       mevcut(degisken)    kontrol(sabit)     tau'nun tahmini
#   ---------------------------------------------------------------------
#   random      tau=1  C_eff=1.000  tau=1  C_eff=1.000  DEGISMEMELI
#   butterfly   tau=3  C_eff=1.000  tau=>L C_eff=0.250  COKMELI (~blok'a)
#   ring        tau=3  C_eff=1.000  tau=3  C_eff=1.000  BIREBIR AYNI
#
# Her hucre ayri bilgi tasiyor:
#
#  1) RANDOM - ana kontrol. tau kuramina gore sabit rastgele de kapsamayi
#     tek katmanda doldurur, o yuzden fark OLMAMALI.
#       fark yoksa  -> 1. bulgu ayakta: kalite kapsamadan geliyor, derinlik
#                      degiskenliginden degil. tau dogrulanir.
#       fark varsa  -> 1. bulgu 2. bulgunun icine coker: "rastgele iyi" demek
#                      aslinda "her katmanda yeniden cizilmek iyi" demekmis.
#                      Bu da makalenin ana iddiasini DEGISTIRIR - ve boyle
#                      raporlanmasi gerekir.
#
#  2) BUTTERFLY - en keskin test. Kelebek ile blok ayni katman-ici kapsamaya
#     sahip (0.250) ama olculen fark 6.7 sigma. tau bunu kelebegin stage'inin
#     donmesine bagliyor. Stage'i dondurursak kelebek blok'a inmeli:
#       ~1.492'ye inerse -> tau'nun mekanizmasi dogrulanir (cok guclu kanit)
#       ~1.477'de kalirsa -> fark stage'den degil baska bir seyden geliyor,
#                            tau'nun aciklamasi eksik
#
#  3) RING - BOS KONTROL. Halka zaten tohum kullanmiyor ve stage'i yok, yani
#     --fixed-mask onun icin hicbir sey degistirmemeli. Bu kosular mevcut
#     gpt_ring_m16_L8_s1{0,1,2}_mixed.json ile SAYISAL OLARAK AYNI cikmali
#     (1.4736 / 1.4802 / 1.4765). Cikmazsa bayrak istemeden baska bir seye
#     dokunuyordur ve digerlerinin sonucu da guvenilmezdir.
#
# Ayrica mevcut (degisken) kol 13-15 tohumlariyla genisletiliyor: karsilastirma
# 3'e 6 degil 6'ya 6 olsun, sigma'lar dar olsun diye.
#
# sbatch slurm/gpt_sabit_maske.slurm
# =====================================================================
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg35
#SBATCH -C H200
#SBATCH --reservation=yzup
#SBATCH -J gpt_sabit
#SBATCH -o logs/sabit_%A_%a.out
#SBATCH -e logs/sabit_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --array=0-56%8

set -euo pipefail
mkdir -p logs results

module load apps/truba-ai/gpu-2024.0

# --- izgara: "ffn L tohum mod" -------------------------------------
# mod=sabit    -> --fixed-mask (yeni kontrol kolu)
# mod=degisken -> bayraksiz (mevcut kolun tohum genislemesi)
GRID=()

# 1) KONTROL KOLU: sabit topoloji, 6 tohum
for L in 4 8 16; do
  for FFN in random butterfly; do
    for S in 10 11 12 13 14 15; do
      GRID+=("$FFN $L $S sabit")
    done
  done
done

# 2) MEVCUT KOLU GENISLET: 10-12 zaten var, 13-15 ekleniyor -> n=6
for L in 4 8 16; do
  for FFN in random butterfly; do
    for S in 13 14 15; do
      GRID+=("$FFN $L $S degisken")
    done
  done
done

# 3) BOS KONTROL: halka sabit == halka degisken olmali (bayrak inert mi?)
for S in 10 11 12; do
  GRID+=("ring 8 $S sabit")
done

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#GRID[@]}" ]; then
  echo "gorev ${SLURM_ARRAY_TASK_ID} izgara disinda (izgara=${#GRID[@]}); atlaniyor."
  exit 0
fi

CFG=(${GRID[$SLURM_ARRAY_TASK_ID]})
FFN=${CFG[0]}; LAYER=${CFG[1]}; SEED=${CFG[2]}; MOD=${CFG[3]}

MULT=16   # yogunluk 1/4, derinlik taramasiyla ayni

if [ "$MOD" = "sabit" ]; then
  FLAG="--fixed-mask"
  OUT="results/gpt_${FFN}_m${MULT}_L${LAYER}_s${SEED}_mixed_sabit.json"
else
  FLAG=""
  OUT="results/gpt_${FFN}_m${MULT}_L${LAYER}_s${SEED}_mixed.json"
fi

echo "== node=$(hostname) ffn=$FFN L=$LAYER seed=$SEED mod=$MOD (izgara ${#GRID[@]}) =="

python train_gpt.py \
  --data data \
  --ffn "$FFN" \
  --ffn-mult "$MULT" \
  --seed "$SEED" \
  --init mixed \
  $FLAG \
  --d-model 512 --n-layer "$LAYER" --n-head 8 \
  --block-size 512 --batch-size 32 \
  --iters 6000 \
  --out "$OUT"
