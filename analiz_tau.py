#!/usr/bin/env python
"""Karisma derinligi (tau) analizi + sabit-maske kontrolunun hukmu.

tau = bir FFN topolojisinin TAM karismaya ulasmasi icin gereken katman sayisi:

    C_eff(L) = (1/d^2) * |{(i,j) : (R_L R_{L-1} ... R_1)_ij > 0}|,   R_l = D_l U_l
    tau      = min{ L : C_eff(L) = 1 }

Katman basina kapsama C, tek bir FFN'e bakar ve iki yonden de yaniltir:
  - ayni C, farkli kalite  (kelebek vs blok: ikisi de 0.250, 6.7 sigma fark)
  - farkli C, ayni kalite  (halka 0.498 vs kelebek 0.250: 0.6 sigma = gurultu)
tau ikisini de dogru yapar.

Kullanim:
    python analiz_tau.py              # tum tablolar + kontrol hukmu
    python analiz_tau.py --latex      # makaleye yapistirilacak LaTeX tablolari

Kontrol kosulari (slurm/gpt_sabit_maske.slurm) henuz yoksa o bolum atlanir.
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

from polytopo import topology

ROOT = os.path.dirname(os.path.abspath(__file__))
D_MODEL, REWIRE = 512, 0.1

# results/ icindeki duman testleri ve deneme kosulari gercek deneylerle AYNI
# semayi paylasiyor; isimle elenmezlerse her yeniden-analize sizarlar.
DUMAN = ("deneme", "ddpduman")


# --------------------------------------------------------------- veri yukleme
def kosulari_yukle():
    runs = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
        ad = os.path.basename(p)
        if ad.startswith(DUMAN) or not ad.startswith(("gpt_", "scale_")):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        a = d.get("args")
        if not a or "final_val_bpc" not in d:
            continue
        runs.append(dict(
            ad=ad, ffn=a.get("ffn"), M=a.get("ffn_mult"), L=a.get("n_layer"),
            d_model=a.get("d_model"), seed=a.get("seed"), init=str(a.get("init")),
            # Kilavuzlu mu? train_gpt.py'nin kendi olcutu: guide_frac > 0.
            # DIKKAT: eski kosular kilavuzsuzlugu None, yeniler 0.0 olarak
            # kaydeder (--guide-frac varsayilani 0.0); 'is not None' demek
            # butun yeni kosulari yanlislikla kilavuzlu sayar.
            guide=(a.get("guide_frac") or 0.0) > 0, fast=bool(a.get("fast")),
            sabit=bool(a.get("fixed_mask")), bpc=d["final_val_bpc"],
        ))
    return runs


def ana_izgara(runs):
    """Kalite tablolarinin kolu: kilavuzsuz, bmm'siz, d=512, varsayilan init."""
    return [r for r in runs if r["ffn"] and not r["guide"] and not r["fast"]
            and r["d_model"] == D_MODEL and r["init"] in ("None", "mixed")]


def ozet(sel):
    v = np.array([r["bpc"] for r in sel])
    if len(v) == 0:
        return None
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


def sigma(a, b):
    (m1, s1, n1), (m2, s2, n2) = a, b
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    return abs(m1 - m2) / se if se > 0 else float("inf")


# ------------------------------------------------------------------ tau hesabi
_tau_cache = {}


def tau_hesapla(kind, M, L, sabit=False):
    """(tau, C_katman, C_eff(L)) dondurur. tau bulunamazsa None."""
    key = (kind, M, L, sabit)
    if key in _tau_cache:
        return _tau_cache[key]
    h = M * D_MODEL
    k_up = max(1, round(4 * D_MODEL / M))
    k_down = max(1, min(h, 4 * D_MODEL))
    katmanlar = []
    for i in range(L):
        sd = 0 if sabit else 17 * i
        st = 0 if sabit else i
        kw = {"stage": st} if kind == "butterfly" else {"p": REWIRE}
        katmanlar.append((
            topology.build_rect(kind, h, D_MODEL, k_up, seed=sd, **kw),
            topology.build_rect(kind, D_MODEL, h, k_down, seed=sd + 1, **kw),
        ))
    c_kat = topology.ffn_coverage(*katmanlar[0])
    c_eff = topology.cumulative_coverage(katmanlar)
    tau = next((i + 1 for i, c in enumerate(c_eff) if c >= 0.999), None)
    _tau_cache[key] = (tau, c_kat, c_eff[-1])
    return _tau_cache[key]


def tau_str(t):
    return str(t) if t else ">L"


# =========================================================== 1) YOGUNLUK + tau
def tablo_yogunluk(grid):
    print("=" * 86)
    print("1) tau, katman basina kapsamanin aciklayamadigini acikliyor  (L=8)")
    print("=" * 86)
    print(f"{'M':>4} {'desen':16s} {'kat.arasi':10s} {'C_kat':>7s} {'C_eff':>7s} "
          f"{'tau':>4s} {'val bpc':>9s} {'std':>8s} {'n':>3s}")
    g = defaultdict(list)
    for r in grid:
        if r["L"] == 8 and not r["sabit"]:
            g[(r["M"], r["ffn"])].append(r)

    sira = ["dense", "random", "watts_strogatz", "ring", "butterfly", "block", "mozaik"]
    degisken = {"random", "watts_strogatz", "butterfly"}
    for M in sorted({k[0] for k in g}):
        print("-" * 86)
        for kind in sira:
            sel = g.get((M, kind))
            if not sel:
                continue
            m, s, n = ozet(sel)
            if kind == "dense":
                print(f"{M:>4} {kind:16s} {'-':10s} {1.0:7.3f} {1.0:7.3f} {'-':>4s} "
                      f"{m:9.4f} {s:8.4f} {n:>3}")
                continue
            t, cl, ce = tau_hesapla(kind, M, 8)
            print(f"{M:>4} {kind:16s} {'DEGISKEN' if kind in degisken else 'sabit':10s} "
                  f"{cl:7.3f} {ce:7.3f} {tau_str(t):>4s} {m:9.4f} {s:8.4f} {n:>3}")


def tablo_tau_gucu(grid):
    """tau'nun tahmin gucu iki ayri iddiaya ayrilir.

    KESKIN IDDIA  : tau <= L (karisma tamamlaniyor) vs tau > L (tamamlanmiyor).
                    Bu sinir HER zaman anlamli olmali - kalitedeki asil ucurum bu.
    DERECELI IDDIA: tamamlayanlar arasinda kalite tau ile birlikte kotulesir.
                    Etki buyuklugu seyreklikle olceklenir; dusuk seyreklikte
                    (M=8) tau=1 ile tau=2 arasindaki fark gurultunun altinda
                    kalabilir. Bunu gizlemek yerine oldugu gibi raporluyoruz.
    """
    print()
    print("=" * 86)
    print("2) tau'nun tahmin gucu: keskin sinir (tamamlaniyor mu?) + dereceli siralama")
    print("=" * 86)
    for M in (8, 16):
        bilgi = {}
        for r in grid:
            if r["L"] == 8 and r["M"] == M and not r["sabit"] and r["ffn"] != "dense":
                bilgi.setdefault(r["ffn"], []).append(r)
        bilgi = {k: (tau_hesapla(k, M, 8)[0] or 99, ozet(v))
                 for k, v in bilgi.items() if ozet(v) and ozet(v)[2] > 1}
        if len(bilgi) < 3:
            continue
        tamam = {k: v for k, v in bilgi.items() if v[0] <= 8}
        eksik = {k: v for k, v in bilgi.items() if v[0] > 8}
        print(f"\n  M={M}")
        print(f"    tamamlayan (tau<=L): " + ", ".join(
            f"{k}(tau={v[0]}, {v[1][0]:.4f})" for k, v in sorted(tamam.items(), key=lambda x: x[1][0])))
        print(f"    tamamlamayan (tau>L): " + ", ".join(
            f"{k}({v[1][0]:.4f})" for k, v in sorted(eksik.items())))

        # --- KESKIN SINIR: tamamlayan vs tamamlamayan, tum ciftler ---
        sinir = [(sigma(a[1], b[1]), ka, kb)
                 for ka, a in tamam.items() for kb, b in eksik.items()]
        if sinir:
            en_zayif = min(sinir)
            print(f"    [keskin] tamamlayan-vs-tamamlamayan en ZAYIF sigma = "
                  f"{en_zayif[0]:5.2f}  ({en_zayif[1]} vs {en_zayif[2]})"
                  f"  -> {'hepsi anlamli' if en_zayif[0] >= 2 else 'SINIR TUTMUYOR'}")

        # --- DERECELI: tamamlayanlar arasinda tau ile siralama ---
        sr = sorted(tamam.items(), key=lambda x: (x[1][0], x[1][1][0]))
        monoton = all(sr[i][1][1][0] <= sr[i + 1][1][1][0] + 1e-9 for i in range(len(sr) - 1))
        ic = [(sigma(a[1], b[1]), ka, kb, a[0], b[0])
              for i, (ka, a) in enumerate(sr) for kb, b in sr[i + 1:]]
        ayni = [x for x in ic if x[3] == x[4]]
        farkli = [x for x in ic if x[3] != x[4]]
        print(f"    [dereceli] tau ile siralama monoton mu: {'EVET' if monoton else 'HAYIR'}")
        if ayni:
            print(f"               tau AYNI olanlar   en buyuk sigma = {max(ayni)[0]:5.2f}"
                  f"  ({max(ayni)[1]} vs {max(ayni)[2]})"
                  f"  -> {'ayirt edilemez (beklenen)' if max(ayni)[0] < 2 else 'AYRISIYOR'}")
        if farkli:
            e = min(farkli)
            print(f"               tau FARKLI olanlar en kucuk sigma = {e[0]:5.2f}"
                  f"  ({e[1]} tau={e[3]} vs {e[2]} tau={e[4]})")
            if e[0] < 2:
                print(f"               -> bu M'de kucuk tau farki gurultunun altinda; "
                      f"tau ordinal, metrik degil.")


# ================================================================= 3) DERINLIK
def tablo_derinlik(grid):
    print()
    print("=" * 86)
    print("3) tau ile L'nin etkilesimi: derinlik egrisinin uc rejimi  (M=16)")
    print("=" * 86)
    print(f"{'L':>3} {'tau(halka)':>10} {'rejim':26s} {'halka-rastgele':>15} {'sigma':>7}")
    print("-" * 86)
    for L in (2, 4, 8, 16):
        r = ozet([x for x in grid if x["ffn"] == "ring" and x["M"] == 16
                  and x["L"] == L and not x["sabit"]])
        ra = ozet([x for x in grid if x["ffn"] == "random" and x["M"] == 16
                   and x["L"] == L and not x["sabit"]])
        if not r or not ra or r[2] < 2 or ra[2] < 2:
            continue
        t = tau_hesapla("ring", 16, L)[0]
        rej = ("L < tau: karisma tamamlanamiyor" if (t is None or L < t)
               else "L ~ tau: tam yetiyor" if L <= t + 1
               else "L > tau: sabit desen doyuyor")
        print(f"{L:>3} {tau_str(t):>10} {rej:26s} {r[0]-ra[0]:>+15.4f} {sigma(r, ra):>7.1f}")
    print("\n  L=4'te acigin kapanmasi tesaduf degil: halka tam orada karismasini")
    print("  bitiriyor (tau=3). Oncesinde eksik karisma, sonrasinda doygunluk cezasi.")


# ================================================================ 4) OLCEK
def tablo_olcek(runs):
    print()
    print("=" * 86)
    print("4) 403M olcek deneyi: sigma'lar (n=2, kendi olcutumuz sigma>=2)")
    print("=" * 86)
    sc = defaultdict(list)
    for r in runs:
        if r["ad"].startswith("scale_"):
            sc[r["ffn"]].append(r["bpc"])
    st = {k: (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0, len(v))
          for k, v in sc.items()}
    for k, (m, s, n) in sorted(st.items()):
        print(f"  {k:12s} {m:.4f} +- {s:.4f}  (n={n})  ham={[round(x,4) for x in sc[k]]}")
    print()
    for x, y in (("dense", "random"), ("dense", "butterfly"), ("random", "butterfly")):
        if x in st and y in st:
            s = sigma(st[x], st[y])
            print(f"  {x:10s} vs {y:10s} delta={st[y][0]-st[x][0]:+.4f} sigma={s:5.2f}"
                  f"   -> {'ANLAMLI' if s >= 2 else 'GURULTU (sigma<2)'}")


# ====================================================== 5) SABIT-MASKE KONTROLU
def kontrol_hukmu(grid):
    print()
    print("=" * 86)
    print("5) SABIT-MASKE KONTROLU  (slurm/gpt_sabit_maske.slurm)")
    print("=" * 86)
    sabitler = [r for r in grid if r["sabit"]]
    if not sabitler:
        print("  Kontrol kosulari henuz yok. TRUBA'da calistir:")
        print("      sbatch slurm/gpt_sabit_maske.slurm")
        print("  Bittiginde bu betigi tekrar calistir; hukum buraya basilir.")
        return

    def kol(kind, L, sabit):
        return ozet([r for r in grid if r["ffn"] == kind and r["M"] == 16
                     and r["L"] == L and r["sabit"] == sabit])

    # --- 5a) BOS KONTROL: bayrak halka icin inert olmali -------------------
    print("\n  [5a] BOS KONTROL - halka tohum/stage kullanmaz, bayrak etkisiz olmali")
    a, b = kol("ring", 8, True), kol("ring", 8, False)
    if a and b:
        s = sigma(a, b)
        print(f"       halka L=8  sabit={a[0]:.4f}+-{a[1]:.4f} (n={a[2]})  "
              f"degisken={b[0]:.4f}+-{b[1]:.4f} (n={b[2]})  sigma={s:.2f}")
        print(f"       -> {'GECTI: bayrak inert.' if s < 2 else 'KALDI! Bayrak istemeden baska bir seye dokunuyor; asagisi guvenilmez.'}")
    else:
        print("       halka kontrol kosulari eksik.")

    # --- 5b) ANA KONTROL: rastgele ----------------------------------------
    print("\n  [5b] ANA KONTROL - rastgele. tau tahmini: DEGISMEMELI (tau=1 -> tau=1)")
    for L in (4, 8, 16):
        a, b = kol("random", L, True), kol("random", L, False)
        if not a or not b:
            continue
        s = sigma(a, b)
        hkm = ("tahmin TUTTU: kapsama etkisi, derinlik degiskenligi degil"
               if s < 2 else
               "tahmin TUTMADI: 'rastgele iyi' kismen 'her katmanda yeniden cizilmek iyi'")
        print(f"       L={L:>2}  sabit={a[0]:.4f} (n={a[2]})  degisken={b[0]:.4f} (n={b[2]})  "
              f"delta={a[0]-b[0]:+.4f}  sigma={s:5.2f}")
        print(f"             -> {hkm}")

    # --- 5c) KESKIN TEST: kelebek -----------------------------------------
    print("\n  [5c] KESKIN TEST - kelebek. tau tahmini: COKMELI (tau=3 -> >L, C_eff 1.0 -> 0.25)")
    blok = kol("block", 8, False)
    for L in (4, 8, 16):
        a, b = kol("butterfly", L, True), kol("butterfly", L, False)
        if not a or not b:
            continue
        s = sigma(a, b)
        satir = (f"       L={L:>2}  sabit={a[0]:.4f} (n={a[2]})  degisken={b[0]:.4f} (n={b[2]})  "
                 f"delta={a[0]-b[0]:+.4f}  sigma={s:5.2f}")
        print(satir)
        # Yon ile anlamliligi AYIRIYORUZ: dogru yonde ama guclu olmayan bir etki
        # 'aciklama yanlis' demek degildir; o derinlikte sinyalin kendisi kucuktur.
        if a[0] > b[0] and s >= 2:
            print("             -> DOGRULANDI: stage donmeyince kelebek cokuyor")
        elif a[0] > b[0]:
            print("             -> yon dogru, bu derinlikte anlamli degil (guc yetersiz)")
        elif s >= 2:
            print("             -> TAHMINE TERS: sabit kelebek anlamli olarak DAHA IYI")
        else:
            print("             -> etki yok")
        if L == 8 and blok:
            print(f"             (blok, ayni C=0.250, tau=>L: {blok[0]:.4f} - "
                  f"sabit kelebek buraya inmeliydi, sigma={sigma(a, blok):.2f})")


# ==================================================================== LaTeX
def latex(grid):
    print()
    print("%" + "=" * 74)
    print("% Makaleye yapistir: tau tablosu (M=16, L=8)")
    print("%" + "=" * 74)
    print(r"\begin{table}[t]")
    print(r"\caption{Karışma derinliği $\tau$, katman başına kapsamanın "
          r"açıklayamadığı farkları açıklar ($M{=}16$, $L{=}8$). "
          r"$\tau$ aynı olan desenler istatistiksel olarak ayırt edilemez "
          r"($\sigma \le 0{,}81$); farklı olanlar her zaman anlamlıdır "
          r"($\sigma \ge 2{,}85$).}")
    print(r"\label{tab:tau}")
    print(r"\centering")
    print(r"\begin{tabular}{lccrc}")
    print(r"\toprule")
    print(r"Desen & Katmanlar arası & $\mathcal{C}_{\text{katman}}$ & $\tau$ & val bpc \\")
    print(r"\midrule")
    ad = {"random": "Rastgele", "watts_strogatz": "Küçük-dünya", "ring": "Halka",
          "butterfly": "Kelebek", "block": "Blok", "mozaik": "Mozaik"}
    degisken = {"random", "watts_strogatz", "butterfly"}
    satirlar = []
    for kind in ad:
        sel = [r for r in grid if r["ffn"] == kind and r["M"] == 16
               and r["L"] == 8 and not r["sabit"]]
        st = ozet(sel)
        if not st or st[2] < 2:
            continue
        t, cl, _ = tau_hesapla(kind, 16, 8)
        satirlar.append((t or 99, ad[kind],
                         "değişken" if kind in degisken else "sabit",
                         cl, tau_str(t), st))
    onceki = None
    for t, isim, va, cl, ts, st in sorted(satirlar):
        if onceki is not None and t != onceki:
            print(r"\midrule")
        onceki = t
        print(f"{isim} & {va} & {cl:.3f} & ${ts.replace('>L', '>L')}$ & "
              f"${st[0]:.4f} \\pm {st[1]:.4f}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="LaTeX tablolarini bas")
    args = ap.parse_args()

    runs = kosulari_yukle()
    grid = ana_izgara(runs)
    print(f"okunan kosu: {len(runs)}  (ana izgara: {len(grid)}, "
          f"sabit-maske: {sum(r['sabit'] for r in grid)})\n")

    tablo_yogunluk(grid)
    tablo_tau_gucu(grid)
    tablo_derinlik(grid)
    tablo_olcek(runs)
    kontrol_hukmu(grid)
    if args.latex:
        latex(grid)


if __name__ == "__main__":
    main()
