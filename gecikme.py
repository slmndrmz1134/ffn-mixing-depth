#!/usr/bin/env python
"""GECIKME TABANI: seyreltmenin asla asamayacagi duvar.

Karar modeli (karar.py) yalnizca BANT GENISLIGINE bakar: T = B / beta.
Gercek kolektif maliyeti alfa-beta modelidir:

    T_iletisim  =  R * alpha   +   B / beta
                   ^^^^^^^^^       ^^^^^^^^
                   tur sayisi x    bayt /
                   gidis-donus     bant genisligi

KRITIK GOZLEM: matris seyreltmesi B'yi dusurur, R'yi DUSURMEZ.
Kelebek bayti ~7x azaltir ama tur sayisini yalnizca ~1.5x azaltir
(dikkat all-reduce'u her katmanda durmaya devam eder).

Dolayisiyla bant genisligi sonsuza giderse bile:

    lim T_iletisim = R * alpha        <-- SEYRELTMENIN ASAMADIGI TABAN
    beta -> sonsuz

Bu, makaledeki bilgi-teorik sinira paralel ikinci bir sinirdir: biri
KAPSAMAYI, digeri ZAMANI sinirlar.

  python gecikme.py
"""
import math


def turlar(L, P, kelebek):
    """Adim basina kolektif TUR sayisi (ileri + geri).

    dikkat  : her katmanda all-reduce -> ozyinelemeli yarilama ile ceil(log2 P) tur
    FFN     : yogun TP ayni sekilde ceil(log2 P); kelebek TEK muhatap -> 1 tur
    x2      : ileri + geri
    """
    lg = math.ceil(math.log2(P))
    ffn = 1 if kelebek else lg
    return L * 2 * (lg + ffn)


def rapor(ad, N_params, P, gpu_toplam, L=64, batch=1, seq=8192, tflops=500):
    flops = 6 * N_params * batch * seq
    t_hesap = flops / (gpu_toplam * tflops * 1e12)

    R_yogun = turlar(L, P, kelebek=False)
    R_kelebek = turlar(L, P, kelebek=True)

    print(f"\n{ad}")
    print(f"  {N_params/1e9:.0f}B parametre, {P} merkez, {gpu_toplam} GPU, L={L}")
    print(f"  adim basina hesap        : {t_hesap:6.2f} sn")
    print(f"  adim basina kolektif tur : yogun TP {R_yogun}, kelebek {R_kelebek} "
          f"({R_yogun/R_kelebek:.1f}x azalma)")
    print(f"\n  {'RTT':>8}  {'ortam':26s} {'kelebek gecikme tabani':>22} {'hesaba orani':>13}")
    print("  " + "-" * 76)
    for rtt_ms, ortam in [(0.002, "ayni node (NVLink)"),
                          (0.05, "ayni kume (InfiniBand)"),
                          (1.0, "ayni sehir / dedike fiber"),
                          (10.0, "sehirler arasi"),
                          (100.0, "kitalar arasi")]:
        taban = R_kelebek * rtt_ms / 1000
        oran = taban / t_hesap
        hkm = "ihmal edilir" if oran < 0.1 else ("tolere edilir" if oran < 1 else
                                                 "BASKIN" if oran < 10 else "OLUMCUL")
        print(f"  {rtt_ms:>6.3f}ms  {ortam:26s} {taban:>19.2f} sn {oran:>10.1f}x  {hkm}")

    print(f"\n  -> Bant genisligi SONSUZ olsa bile yukaridaki taban kalir; seyreltme")
    print(f"     bayti dusurur, TUR SAYISINI dusurmez.")


if __name__ == "__main__":
    print(__doc__)
    rapor("A) 500B model, 8 merkeze bolunmus", 500e9, P=8, gpu_toplam=64)
    print("\n" + "=" * 80)
    print("SONUC")
    print("=" * 80)
    print("""
  Sehirler arasi (10 ms) ve otesinde gecikme tabani hesabi asiyor; bant
  genisligini artirmak bunu DEGISTIRMEZ. Makaledeki '~400 Gbps koridoru'
  bulgusu bu yuzden 'ayni sehir / dedike fiber' ile ayni yere dusuyor:
  o rejim yalnizca bant genisliginin degil, GECIKMENIN de yettigi tek yer.

  Ve koridor seyreltmeyi iyilestirerek genisletilemez. Genisletmenin tek
  yolu TUR SAYISINI dusurmektir - ki bu, agirlik yapisini degil
  SENKRONIZASYON TAKVIMINI degistirmek demektir (DiLoCo / yerel-SGD).
""")
