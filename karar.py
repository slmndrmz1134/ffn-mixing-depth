#!/usr/bin/env python
"""KARAR MODELI: kelebek parcalamayla dagitik egitim benim durumumda rasyonel mi?

Bu betik bir URUN degil, calismanin sonucunu bir KARARA ceviren arac. Girdi:
senin altyapin (model boyutu, veri merkezi sayisi, GPU, VRAM, merkezler-arasi
bant genisligi). Cikti: hangi rejimdesin ve kelebek sana ne kazandirir.

Dayandigi OLCULEN gercekler (bu calismadan):
  - kelebek FFN iletisimi dense tensor-paralele gore 7x az, muhatap 7->1 (T4)
  - AMA dikkat katmani hala all-reduce ister; uctan uca kazanc ~2x (7x degil)
  - kelebegin kucuk kalici kalite bedeli var (403M'de dense'e +0.014 bpc)
  - bilgi duvari: katman-basi tam kapsama + tek-muhatap ayni katmanda imkansiz
"""

def karar(ad, N_params, P_dc, gpu_per_dc, vram_gb, gbps,
          d=8192, L=64, batch=1, seq=8192, tflops=500):
    B = batch * seq                                   # token/adim (1 mikro-yigin)
    Mfull = B * d * 2                                 # tam aktivasyon (bf16), bayt
    # --- iletisim (rank basina, adim basina, ileri+geri) ---
    # dikkat: her iki stratejide de all-reduce (FFN'i degil dikkati parcalamadik)
    attn = 2 * (P_dc - 1) / P_dc * Mfull
    ffn_dense = 2 * (P_dc - 1) / P_dc * Mfull         # FFN all-reduce
    ffn_bf = 2 * (Mfull / P_dc)                        # kelebek: 1 gonder+1 al, kendi dilimi
    comm_dense = L * 2 * (attn + ffn_dense)           # xL katman, x2 ileri+geri
    comm_bf = L * 2 * (attn + ffn_bf)
    Bps = gbps * 1e9 / 8                               # bant: bayt/sn
    # --- hesap ---
    flops = 6 * N_params * B                           # ileri+geri, adim basina
    total_gpu = P_dc * gpu_per_dc
    t_compute = flops / (total_gpu * tflops * 1e12)
    t_dense = comm_dense / Bps
    t_bf = comm_bf / Bps
    # --- bellek: model tek merkeze sigar mi? (egitim: ~16 bayt/param) ---
    node_vram = gpu_per_dc * vram_gb
    fits_one = N_params * 16 / 1e9 < node_vram

    print(f"\n{'='*66}\n{ad}")
    print(f"  model {N_params/1e9:.0f}B | {P_dc} merkez x {gpu_per_dc} GPU | "
          f"{vram_gb}GB/GPU | merkezler-arasi {gbps} Gbps")
    print(f"  tek merkeze sigar mi: {'EVET' if fits_one else 'HAYIR'} "
          f"(gerekli {N_params*16/1e9:.0f}GB vs merkez {node_vram:.0f}GB)")
    print(f"  hesap suresi/adim   : {t_compute*1000:8.1f} ms")
    print(f"  dense TP iletisim   : {t_dense*1000:8.1f} ms  "
          f"({'HESAP-bagimli, iyi' if t_dense<t_compute else 'HAT-bagimli, kotu'})")
    print(f"  kelebek iletisim    : {t_bf*1000:8.1f} ms  "
          f"({'HESAP-bagimli, iyi' if t_bf<t_compute else 'HAT-bagimli, kotu'})")
    print(f"  uctan uca iletisim azalmasi: {comm_dense/comm_bf:.1f}x")

    # --- KARAR AGACI ---
    print("  --- KARAR ---")
    if fits_one:
        print("  Model tek merkeze SIGIYOR. Modeli bolmene GEREK YOK.")
        print("  -> Veri-paralel + seyrek senkronizasyon (DiLoCo). Kelebek GEREKSIZ.")
        return
    print("  Model tek merkeze sigmiyor -> bolmek ZORUNDASIN.")
    if t_dense < t_compute:
        print("  Hat, dense TP icin bile yeterince hizli. Kelebek SART DEGIL")
        print(f"  ama iletisimi {comm_dense/comm_bf:.1f}x azaltir (bedava fayda).")
    elif t_bf < t_compute:
        print("  Dense TP hat-bagimli (kilitlenir) AMA kelebek hesabin altina iniyor.")
        print("  -> KELEBEGI KULLAN. Bu senaryoda ONU RASYONEL KILAN sey bu.")
        print(f"     Bedeli: ~%1 kalite (403M'de +0.014 bpc), ~%5-15 fazla token.")
    else:
        print("  Kelebek bile hesabin USTUNDE - hat cok yavas.")
        print("  -> Bu hatta model-paralel DENEME. Farkli yaklasim (sync sikligi) gerekir")
        print("     ya da daha hizli hat bul. Kalite tartismasi anlamsiz, is BITMEZ.")


if __name__ == "__main__":
    print("KELEBEK PARCALAMA KARAR MODELI (olculen sayilara dayali)")
    print("Capa: gercek olcumumuz InfiniBand'de kelebek 2.51ms < dense 4.25ms (d=8192).")
    print("Asagida o hesabi farkli bant genisliklerine tasiyoruz.")
    # A: kucuk model, tek merkeze sigar
    karar("A) 7B model, orta altyapi",
          N_params=7e9, P_dc=2, gpu_per_dc=8, vram_gb=140, gbps=10)
    # B: dev model, hizli ozel hat (ayni metro / dedike fiber)
    karar("B) 500B model, HIZLI ozel hat (400 Gbps, ayni sehir/dedike)",
          N_params=500e9, P_dc=8, gpu_per_dc=8, vram_gb=140, gbps=400)
    # D: dev model, orta hat
    karar("C) 500B model, orta hat (100 Gbps)",
          N_params=500e9, P_dc=8, gpu_per_dc=8, vram_gb=140, gbps=100)
    # C: dev model, tuketici hatti
    karar("D) 500B model, tuketici interneti (1 Gbps)",
          N_params=500e9, P_dc=8, gpu_per_dc=8, vram_gb=140, gbps=1)
    print(f"\n{'='*66}")
    print("OZET: kelebek yalnizca DAR bir koridorda rasyoneldir --")
    print("  (1) model tek merkeze SIGMAYACAK kadar buyuk (yoksa DiLoCo yeter),")
    print("  (2) hat dense TP'ye YETMEYECEK ama kelebege YETECEK kadar hizli")
    print("      (pratikte ~400 Gbps+, yani ayni sehir/dedike fiber; WAN degil).")
    print("  Koridor dar: senaryo B rasyonel, C ve D'de kelebek bile kurtarmaz.")
    print("  Buyuk resim: MERKEZLER-ARASI katman-katman parcalama neredeyse hep")
    print("  kotu fikir. Kelebegin gercek evi merkez-ICI genis kume (cok dugum,")
    print("  hizli fabric) - bizim InfiniBand olcumumuzun yasadigi yer.")
