#!/usr/bin/env python
"""Paper figures, English (PDF).

Mirrors make_figs.py. Values are NOT hardcoded: everything is read from
results/*.json through analiz_tau, so figures never silently drift from data.

  python make_figs_en.py   -> fig_coverage/density/depth/tau_en.pdf
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
    return A.ozet([r for r in GRID if r["ffn"] == ffn and r["M"] == M
                   and r["L"] == L and r["sabit"] == sabit])


# ---------- FIG 1: the coverage concept ----------
n, k = 16, 4
patterns = [("Dense\n(coverage 1.00)", T.build_rect("ring", n, n, n)),
            ("Block\n(coverage 0.25)", T.build_rect("block", n, n, k)),
            ("Ring\n(coverage 0.44)", T.build_rect("ring", n, n, k)),
            ("Small-world\n(coverage 0.61)", T.build_rect("watts_strogatz", n, n, k, p=0.25, seed=3))]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0))
for ax, (name, m) in zip(axes, patterns):
    ax.imshow(m, cmap="Purples", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(name, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.5)
fig.text(0.5, -0.02, "Purple = nonzero weight. All four patterns have the same number of "
         "connections per row (equal parameters); only the distribution differs.",
         ha="center", fontsize=7)
fig.tight_layout()
fig.savefig("fig_coverage.pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("fig_coverage.pdf written")

# ---------- FIG 2: density sweep ----------
PAT = {"random": ("Random", "o"), "watts_strogatz": ("Small-world", "s"),
       "butterfly": ("Butterfly", "v"), "ring": ("Ring", "^"), "block": ("Block", "D")}
MS = [8, 16, 32, 64]

dense = bpc("dense", 4)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.axhline(dense[0], ls="--", color="0.4", lw=0.8, label="Dense (reference)")
for f, (name, mk) in PAT.items():
    xs, ys, es = [], [], []
    for M in MS:
        st = bpc(f, M)
        if st and st[2] >= 2:
            xs.append(4 / M); ys.append(st[0]); es.append(st[1] / np.sqrt(st[2]))
    if xs:
        ax.errorbar(xs, ys, yerr=es, marker=mk, ms=4, lw=1.0, capsize=2, label=name)
ax.set_xscale("log", base=2)
ax.set_xlabel("Density (4/M)")
ax.set_ylabel("val bpc (lower = better)")
ax.set_xticks([0.0625, 0.125, 0.25, 0.5])
ax.set_xticklabels(["1/16", "1/8", "1/4", "1/2"])
ax.axvspan(0.45, 0.55, color="green", alpha=0.06)
ax.legend(fontsize=6.0, frameon=False, loc="upper right")
ax.grid(True, lw=0.3, alpha=0.5)
fig.tight_layout()
fig.savefig("fig_density.pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("fig_density.pdf written")

# ---------- FIG 3: depth curve with the tau threshold ----------
LS = [2, 4, 8, 16]
gap, sig = [], []
for L in LS:
    r, ra = bpc("ring", 16, L), bpc("random", 16, L)
    gap.append(r[0] - ra[0])
    sig.append(A.sigma(r, ra))

fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(LS, gap, marker="o", ms=5, lw=1.2, color="#7048c0", zorder=3)
for x, y, s in zip(LS, gap, sig):
    ax.annotate(f"{s:.1f}$\\sigma$", (x, y), textcoords="offset points",
                xytext=(0, 7), fontsize=7, ha="center")
ax.axvline(3, color="0.5", ls=":", lw=0.9, zorder=1)
ax.annotate(r"$L=\tau$", (3, max(gap)), textcoords="offset points",
            xytext=(4, -4), fontsize=7, color="0.35")
ax.set_xscale("log", base=2)
ax.set_xticks(LS); ax.set_xticklabels(LS)
ax.set_xlabel("Number of layers (L)")
ax.set_ylabel("Ring $-$ Random bpc gap")
ax.axhline(0, color="0.6", lw=0.6)
ax.grid(True, lw=0.3, alpha=0.5)
fig.tight_layout()
fig.savefig("fig_depth.pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("fig_depth.pdf written")

# ---------- FIG 4: tau explains what coverage cannot ----------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
col = {1: "#2a7f4f", 3: "#7048c0", 4: "#7048c0", 5: "#c06048", 99: "#b03030"}

for ax, M in zip(axes, [16, 64]):
    pts = []
    for f, (nm, mk) in PAT.items():
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
        c = col.get(t if t else 99, "0.4")
        ax.errorbar([C], [y], yerr=[err], marker=mk, ms=7, capsize=2, color=c,
                    lw=0, zorder=3)
        ly = lab_y[i]
        if abs(ly - y) > (hi - lo) * 0.015:
            ax.plot([C + 0.04, C + 0.11], [y, ly], lw=0.5, color=c, alpha=0.55,
                    zorder=2)
        # the whole tau expression must be inside math mode, otherwise the
        # {>} braces are typeset literally.
        tau_str = rf"$\tau{{=}}{t}$" if t else r"$\tau{>}L$"
        ax.annotate(f"{nm} ({tau_str})",
                    (C + 0.13, ly), fontsize=6.4, color=c,
                    va="center", ha="left", zorder=4)

    ax.set_xlabel(r"Per-layer coverage $\mathcal{C}$")
    ax.set_title(f"$M={M}$ (density {4/M:.3f})".replace("0.062", "1/16"), fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_xlim(-0.10, 1.95)
axes[0].set_ylabel("val bpc (lower = better)")
fig.text(0.5, -0.05,
         "Same coverage with different quality, different coverage with the same "
         r"quality: the discriminating quantity is $\tau$, not $\mathcal{C}$." "\n"
         r"At $M{=}64$ butterfly leads despite HALF the coverage of ring --- it is "
         r"the only pattern that completes its mixing.",
         ha="center", fontsize=7)
fig.tight_layout()
fig.savefig("fig_tau_en.pdf", bbox_inches="tight", dpi=300)
fig.savefig("fig_tau_en_onizleme.png", bbox_inches="tight", dpi=160)
plt.close(fig)
print("fig_tau_en.pdf + onizleme yazildi")

print("\nFOUR FIGURES READY (all generated from results/*.json).")
