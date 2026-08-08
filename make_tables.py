#!/usr/bin/env python
"""results/*.json -> rapor tablolari (TABLOLAR.md). Butun sayilar veriden."""
import json, glob, statistics as st, io, sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
R = "results"


def load_gpt():
    """gpt_*.json -> kayitlar. Bayraklar dosya adindan, gerisi args'tan."""
    recs = []
    for f in glob.glob(f"{R}/gpt_*.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        a, e = d["args"], d["env"]
        name = Path(f).stem
        recs.append(dict(
            ffn=a["ffn"], mult=a["ffn_mult"], L=a["n_layer"],
            init=a.get("init", "mixed"),
            guide=("kilavuz" in name) or (a.get("guide_frac", 0) > 0),
            bmm=("bmm" in name) or a.get("fast", False),
            seed=a["seed"], bpc=d["final_val_bpc"],
            cov=e["ffn_coverage"], hid=e["ffn_hidden"],
            fp=e["params_ffn"], toks=d["tokens_per_sec"]))
    return recs


def agg(recs, **filt):
    """filtreye uyanlari topla -> (ort, std, n, ornek_kayit)."""
    xs = [r for r in recs
          if all(r.get(k) == v for k, v in filt.items())]
    if not xs:
        return None
    b = [r["bpc"] for r in xs]
    return (st.mean(b), st.stdev(b) if len(b) > 1 else 0.0, len(b), xs[0])


def sg(m1, s1, n1, m2, s2, n2):
    d = m1 - m2
    se = ((s1 / max(1, n1) ** 0.5) ** 2 + (s2 / max(1, n2) ** 0.5) ** 2) ** 0.5
    return d, (abs(d) / se if se > 0 else 0.0)


OUT = ["# Tablolar (results/*.json'dan uretildi)\n",
       "> bpc dusuk = iyi. sigma (o) = farkin standart hataya orani; "
       "2o alti anlamsiz sayilir.\n"]
recs = load_gpt()


# ---- T1: varyant tanimlari + kapsama ----
OUT.append("\n## T1 — Varyantlar ve kapsama (yogunluk 1/4, M=16)\n")
OUT.append("| desen | tanim | kapsama |")
OUT.append("|---|---|---|")
tanim = {"dense": "yogun, gizli 4d", "ring": "poligon bandi",
         "watts_strogatz": "kucuk-dunya", "random": "rastgele seyrek (kontrol)",
         "block": "blok-kosegen", "mozaik": "satranc tahtasi", "butterfly": "kelebek (FFT)"}
for v in ["dense", "ring", "watts_strogatz", "random", "block", "mozaik", "butterfly"]:
    a = agg(recs, ffn=v, mult=(4 if v == "dense" else 16), L=8, init="mixed",
            guide=False, bmm=False)
    cov = a[3]["cov"] if a else float("nan")
    OUT.append(f"| {v} | {tanim[v]} | {cov:.3f} |")


# ---- T2: ana kalite izgarasi (L=8, mixed, kilavuzsuz) ----
OUT.append("\n## T2 — Kalite: yogunluk taramasi (L=8, mixed init)\n")
OUT.append("| desen | M | yogunluk | kapsama | val bpc | ± | dense'e fark | anlamli? |")
OUT.append("|---|--:|--:|--:|--:|--:|--:|---|")
base = agg(recs, ffn="dense", mult=4, L=8, init="mixed", guide=False, bmm=False)
rows = []
for M in (8, 16, 32, 64):
    for v in ["dense", "block", "mozaik", "ring", "watts_strogatz", "random", "butterfly"]:
        if v == "dense" and M != 8:
            continue
        a = agg(recs, ffn=v, mult=(4 if v == "dense" else M), L=8, init="mixed",
                guide=False, bmm=False)
        if not a:
            continue
        m, s, n, ex = a
        if v == "dense":
            fark, verd = "—", ""
        else:
            d, o = sg(m, s, n, base[0], base[1], base[2])
            fark = f"{d:+.4f}"
            verd = "gurultu" if o < 2 else ("SEYREK IYI" if d < 0 else "yogun iyi")
            verd += f" ({o:.1f}o)"
        rows.append((M, v, ex["hid"], ex["cov"], m, s, n, fark, verd))
for M, v, hid, cov, m, s, n, fark, verd in rows:
    OUT.append(f"| {v} | {M} | {4/M:.3f} | {cov:.3f} | {m:.4f} | "
               f"{s:.4f} (n={n}) | {fark} | {verd} |")


# ---- T3: derinlik egrisi ----
OUT.append("\n## T3 — Derinlik: ring vs random farki (M=16, mixed, kilavuzsuz)\n")
OUT.append("| L | dense | random | ring | butterfly | ring−random | anlamli? |")
OUT.append("|--:|--:|--:|--:|--:|--:|---|")
for L in (2, 4, 8, 16):
    g = {v: agg(recs, ffn=v, mult=(4 if v == "dense" else 16), L=L, init="mixed",
                guide=False, bmm=False)
         for v in ("dense", "random", "ring", "butterfly")}
    if not all(g.values()):
        continue
    d, o = sg(g["ring"][0], g["ring"][1], g["ring"][2],
              g["random"][0], g["random"][1], g["random"][2])
    OUT.append(f"| {L} | {g['dense'][0]:.4f} | {g['random'][0]:.4f} | "
               f"{g['ring'][0]:.4f} | {g['butterfly'][0]:.4f} | "
               f"{d:+.4f} | {'gurultu' if o<2 else 'anlamli'} ({o:.1f}o) |")


# ---- T5: self-guided ----
OUT.append("\n## T5 — Self-guided kilavuzun derinlige gore etkisi (M=16, mixed)\n")
OUT.append("| L | desen | kilavuzsuz | kilavuzlu | kazanc | anlamli? |")
OUT.append("|--:|---|--:|--:|--:|---|")
for L in (2, 4, 8, 16):
    for v in ("butterfly", "random", "ring"):
        a = agg(recs, ffn=v, mult=16, L=L, init="mixed", guide=False, bmm=False)
        b = agg(recs, ffn=v, mult=16, L=L, init="mixed", guide=True, bmm=False)
        if not (a and b):
            continue
        d, o = sg(a[0], a[1], a[2], b[0], b[1], b[2])
        OUT.append(f"| {L} | {v} | {a[0]:.4f} | {b[0]:.4f} | {d:+.4f} | "
                   f"{'gurultu' if o<2 else 'anlamli'} ({o:.1f}o) |")


# ---- T6: bmm ----
OUT.append("\n## T6 — bmm uygulamasi: ayni kalite, farkli hiz (butterfly, mixed)\n")
OUT.append("| L | M | uygulama | val bpc | k tok/s |")
OUT.append("|--:|--:|---|--:|--:|")
for M, L in [(16, 8), (16, 16), (32, 8)]:
    for bmm in (False, True):
        a = agg(recs, ffn="butterfly", mult=M, L=L, init="mixed", guide=False, bmm=bmm)
        if not a:
            continue
        OUT.append(f"| {L} | {M} | {'bmm' if bmm else 'masked'} | "
                   f"{a[0]:.4f} | {a[3]['toks']/1e3:.0f} |")


# ---- T4: dagitik olcum ----
OUT.append("\n## T4 — Dagitik iletisim (8×H200, d=8192, rank basina esit param)\n")
OUT.append("| parcalama | iletisim MB/adim | muhatap | ileri geciş ms |")
OUT.append("|---|--:|--:|--:|")
dist = json.load(open(f"{R}/dist_d8192_1446294.json", encoding="utf-8"))
bf = [r for r in dist["rows"] if r["variant"].startswith("topo_butterfly")]
show = []
for r in dist["rows"]:
    if r["variant"].startswith("topo_butterfly"):
        continue
    show.append((r["variant"], r["comm_mb_fwd_bwd"], r["peers_per_rank"], r["fwd_ms"]))
if bf:
    show.insert(2, ("topo_butterfly", bf[0]["comm_mb_fwd_bwd"], 1,
                    st.mean(x["fwd_ms"] for x in bf)))
ad = {"tp_dense": "yogun tensor-paralel", "topo_ring": "halka",
      "topo_butterfly": "kelebek", "topo_block": "blok"}
for v, mb, pk, ms in show:
    OUT.append(f"| {ad.get(v, v)} | {mb:.1f} | {pk} | {ms:.2f} |")


# ---- T7: olcekleme 403M ----
OUT.append("\n## T7 — Olcekleme: 403M model (d=1024, L=32, enwik9 bayt, 2 tohum)\n")
OUT.append("| desen | val bpc | ± | bellek GB | k tok/s |")
OUT.append("|---|--:|--:|--:|--:|")
sc = defaultdict(list)
for f in glob.glob(f"{R}/scale_*.json"):
    d = json.load(open(f, encoding="utf-8"))
    sc[d["args"]["ffn"]].append(d)
for v in ("dense", "random", "butterfly"):
    xs = sc[v]
    b = [x["final_val_bpc"] for x in xs]
    OUT.append(f"| {v} | {st.mean(b):.4f} | "
               f"{(st.stdev(b) if len(b)>1 else 0):.4f} (n={len(b)}) | "
               f"{xs[0]['peak_mem_gb']:.0f} | {st.mean(x['tokens_per_sec'] for x in xs)/1e3:.0f} |")


txt = "\n".join(OUT) + "\n"
Path("TABLOLAR.md").write_text(txt, encoding="utf-8")
print(txt)