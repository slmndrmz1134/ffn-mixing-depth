#!/usr/bin/env python
"""Makale figurleri (PDF).

ONEMLI: Sayilar ARTIK GOMULU DEGIL. Onceki surumde bpc degerleri ve sigma'lar
elle yazilmisti; yeni kosular eklendikce sekiller sessizce veriden koptu.
Bu surum her seyi results/*.json'dan analiz_tau uzerinden okur, boylece
'sbatch -> gonder.sh sonuc -> make_figs.py' zinciri her zaman tutarlidir.

  python make_figs.py        -> TR figurleri (fig_kapsama/yogunluk/derinlik/tau)
"""
import sys

sys.path.insert(0, ".")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analiz_tau as A
from polytopo import topology as T

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.7})

RUNS = A.kosulari_yukle()
GRID = A.ana_izgara(RUNS)


def bpc(ffn, M, L=8, sabit=False):
    """(ortalama, std, n) ya da None"""
    return A.ozet([r for r in GRID if r["ffn"] == ffn and r["M"] == M
                   and r["L"] == L and r["sabit"] == sabit])


# ---------- FIG 1: kapsama kavrami (maske desenleri) ----------
n, k = 16, 4
desenler = [("Yogun", T.build_rect("ring", n, n, n)),
            ("Blok", T.build_rect("block", n, n, k)),
            ("Halka", T.build_rect("ring", n, n, k)),
            ("Kucuk-dunya", T.build_rect("watts_strogatz", n, n, k, p=0.25, seed=3))]
# bu oyuncak FFN'in kapsamasi (ayni maske hem yukari hem asagi izdusum),
# maskeden hesaplanir; Tablo 1'deki d=512 degeri DEGILDIR
desenler = [(f"{a}\n(kapsama {(np.asarray(m) @ np.asarray(m) > 0).mean():.2f})", m)
            for a, m in desenler]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0))
for ax, (ad, m) in zip(axes, desenler):
    ax.imshow(m, cmap="Purples", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(ad, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.5)
fig.tight_layout()
fig.savefig("fig_kapsama.pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("fig_kapsama.pdf yazildi")

# ---------- FIG 2: yogunluk taramasi (bpc vs yogunluk) ----------
DESEN = {"random": ("Rastgele", "o"), "watts_strogatz": ("Kucuk-dunya", "s"),
         "butterfly": ("Kelebek", "v"), "ring": ("Halka", "^"), "block": ("Blok", "D")}
MLER = [8, 16, 32, 64]

yogun = bpc("dense", 4)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.axhline(yogun[0], ls="--", color="0.4", lw=0.8, label="Yogun (referans)")
for f, (ad, mk) in DESEN.items():
    xs, ys, es = [], [], []
    for M in MLER:
        st = bpc(f, M)
        if st and st[2] >= 2:
            xs.append(4 / M); ys.append(st[0]); es.append(st[1] / np.sqrt(st[2]))
    if xs:
        ax.errorbar(xs, ys, yerr=es, marker=mk, ms=4, lw=1.0, capsize=2, label=ad)
ax.set_xscale("log", base=2)
ax.set_xlabel("Yogunluk (4/M)")
ax.set_ylabel("val bpc (dusuk = iyi)")
ax.set_xticks([0.0625, 0.125, 0.25, 0.5])
ax.set_xticklabels(["1/16", "1/8", "1/4", "1/2"])
ax.axvspan(0.45, 0.55, color="green", alpha=0.06)
ax.legend(fontsize=6.0, frameon=False, loc="upper right")
ax.grid(True, lw=0.3, alpha=0.5)
fig.tight_layout()
fig.savefig("fig_yogunluk.pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("fig_yogunluk.pdf yazildi")

# ---------- FIG 3: derinlik egrisi + tau rejimleri ----------
LLER = [2, 4, 8, 16]
fark, sig, tauL = [], [], []
for L in LLER:
    r, ra = bpc("ring", 16, L), bpc("random", 16, L)
    fark.append(r[0] - ra[0])
    sig.append(A.sigma(r, ra))
    tauL.append(A.tau_hesapla("ring", 16, L)[0])

fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(LLER, fark, marker="o", ms=5, lw=1.2, color="#7048c0", zorder=3)
for x, y, s, t in zip(LLER, fark, sig, tauL):
    ax.annotate(f"{s:.1f}$\\sigma$", (x, y), textcoords="offset points",
                xytext=(0, 7), fontsize=7, ha="center")
# tau esigini isaretle: halka icin tau=3, yani L=4 civari
ax.axvline(3, color="0.5", ls=":", lw=0.9, zorder=1)
ax.annotate(r"$L=\tau$", (3, max(fark)), textcoords="offset points",
            xytext=(4, -4), fontsize=7, color="0.35")
ax.set_xscale("log", base=2)
ax.set_xticks(LLER); ax.set_xticklabels(LLER)
ax.set_xlabel("Katman sayisi (L)")
ax.set_ylabel("Halka $-$ Rastgele bpc farki")
ax.axhline(0, color="0.6", lw=0.6)
ax.grid(True, lw=0.3, alpha=0.5)
fig.tight_layout()
fig.savefig("fig_derinlik.pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("fig_derinlik.pdf yazildi")

# ---------- FIG 4: tau, kapsamanin aciklayamadigini aciklar ----------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
renk = {1: "#2a7f4f", 3: "#7048c0", 4: "#7048c0", 5: "#c06048", 99: "#b03030"}

for ax, M in zip(axes, [16, 64]):
    pts = []
    for f, (nm, mk) in DESEN.items():
        st = bpc(f, M)
        if not st or st[2] < 2:
            continue
        t, C, _ = A.tau_hesapla(f, M, 8)
        pts.append((C, st[0], st[1] / np.sqrt(st[2]), nm, t, mk))

    ys = [p[1] for p in pts]
    span = max(ys) - min(ys)
    ax.set_ylim(min(ys) - 0.18 * span, max(ys) + 0.18 * span)
    lo, hi = ax.get_ylim()

    # Etiketler TEK satir; carpismayi onlemek icin y'ye gore siralayip
    # etiket yuksekligi kadar (yaklasik %9) minimum aralik zorluyoruz.
    order = sorted(range(len(pts)), key=lambda i: pts[i][1])
    bosluk = (hi - lo) * 0.105
    lab_y, prev = {}, -1e9
    for i in order:
        yy = max(pts[i][1], prev + bosluk)
        lab_y[i] = yy
        prev = yy

    for i, (C, y, err, nm, t, mk) in enumerate(pts):
        c = renk.get(t if t else 99, "0.4")
        ax.errorbar([C], [y], yerr=[err], marker=mk, ms=7, capsize=2, color=c,
                    lw=0, zorder=3)
        ly = lab_y[i]
        if abs(ly - y) > (hi - lo) * 0.015:
            ax.plot([C + 0.04, C + 0.11], [y, ly], lw=0.5, color=c, alpha=0.55,
                    zorder=2)
        # tau ifadesinin TAMAMI matematik modunda olmali; yoksa {>} suslu
        # parantezleri harfi harfine basiliyor.
        tau_str = rf"$\tau{{=}}{t}$" if t else r"$\tau{>}L$"
        ax.annotate(f"{nm} ({tau_str})",
                    (C + 0.13, ly), fontsize=6.4, color=c,
                    va="center", ha="left", zorder=4)

    ax.set_xlabel(r"Katman basina kapsama $\mathcal{C}$")
    ax.set_title(f"$M={M}$ (yogunluk {4/M:.3f})".replace("0.062", "1/16"), fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_xlim(-0.10, 1.95)
axes[0].set_ylabel("val bpc (dusuk = iyi)")
fig.text(0.5, -0.05,
         "Ayni kapsamada farkli kalite, farkli kapsamada ayni kalite: ayiran sey "
         r"$\mathcal{C}$ degil $\tau$'dur." "\n"
         r"$M{=}64$'te kelebek, kapsamasi halkanin YARISI oldugu halde onde --- "
         r"karismasini tamamlayabilen tek desen o.",
         ha="center", fontsize=7)
fig.tight_layout()
fig.savefig("fig_tau.pdf", bbox_inches="tight", dpi=300)
fig.savefig("fig_tau_onizleme.png", bbox_inches="tight", dpi=160)
plt.close(fig)
print("fig_tau.pdf + onizleme yazildi")

print("\nDORT FIGUR HAZIR (hepsi results/*.json'dan uretildi).")
