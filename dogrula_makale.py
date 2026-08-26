#!/usr/bin/env python
"""Makaledeki sayilari HAM VERIYLE karsilastirir.

NEDEN: tex_kontrol.py yapiyi (referans, ortam, sutun) dogrular ama SAYILARI
dogrulamaz; senkron.py iki dili karsilastirir ama ikisi ayni sekilde bayatsa
"temiz" der. Bu iki kor nokta yuzunden kollar genisletildiginde tablolardaki
degerler sessizce eskidi.

Bu betik tablolardaki bpc degerlerini results/'tan yeniden hesaplayip
diff'ler. Yeni bir kosu eklendiginde ONCE bunu calistir.

  python dogrula_makale.py            # her iki makale
  python dogrula_makale.py makale.tex
"""
import os
import re
import sys

import analiz_tau as A

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

RUNS = A.kosulari_yukle()
GRID = A.ana_izgara(RUNS)
TOLERANS = 0.00005          # 4 ondalikta yuvarlama payi


def bpc(ffn, M, L=8, sabit=False, kilavuz=False):
    sel = [r for r in RUNS if r["ffn"] == ffn and r["M"] == M and r["L"] == L
           and r["d_model"] == 512 and r["sabit"] == sabit
           and r["guide"] == kilavuz and r["init"] in ("None", "mixed")
           and not r["fast"] and r["ad"].startswith("gpt_")]
    return A.ozet(sel)


# (aciklama, ffn, M, L, sabit, kilavuz) -> makalede gecmesi gereken deger
BEKLENEN = []
for M in (8, 16, 32, 64):
    for f in ("random", "watts_strogatz", "ring", "butterfly", "block", "mozaik"):
        s = bpc(f, M)
        if s and s[2] >= 2:
            BEKLENEN.append((f"tab:kalite M={M} {f}", s[0]))
for L in (2, 4, 8, 16):
    for f in ("dense", "random", "ring", "butterfly"):
        s = bpc(f, 4 if f == "dense" else 16, L)
        if s and s[2] >= 2:
            BEKLENEN.append((f"tab:derinlik L={L} {f}", s[0]))
for L in (4, 8, 16):
    for f in ("random", "butterfly"):
        s = bpc(f, 16, L, sabit=True)
        if s:
            BEKLENEN.append((f"tab:sabit L={L} {f} (sabit)", s[0]))
for L in (8, 16):
    for f in ("butterfly", "random", "ring"):
        s = bpc(f, 16, L, kilavuz=True)
        if s:
            BEKLENEN.append((f"tab:kilavuz L={L} {f} (kilavuzlu)", s[0]))


def dogrula(dosya):
    s = open(dosya, encoding="utf-8").read()
    sayilar = set(re.findall(r"\b\d\.\d{4}\b", s))
    eksik = []
    for ad, deger in BEKLENEN:
        if not any(abs(float(x) - deger) < TOLERANS for x in sayilar):
            eksik.append((ad, deger))
    print(f"\n{dosya}: {len(BEKLENEN)} beklenen deger, "
          f"{len(BEKLENEN)-len(eksik)} eslesti")
    if eksik:
        print("  METINDE BULUNAMAYAN (bayat ya da hic yazilmamis):")
        for ad, d in eksik:
            print(f"    {ad:44s} -> {d:.4f}")
    else:
        print("  hepsi metinde mevcut")
    return len(eksik)


if __name__ == "__main__":
    hedef = sys.argv[1:] or ["makale.tex", "makale_en.tex"]
    toplam = sum(dogrula(f) for f in hedef)
    print(f"\n{'TEMIZ' if toplam == 0 else f'{toplam} uyusmazlik'}")
    sys.exit(1 if toplam else 0)
