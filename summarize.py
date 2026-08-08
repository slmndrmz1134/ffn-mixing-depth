#!/usr/bin/env python
"""results/ altindaki GPT kosularini rapora hazir tek tabloya cevirir.

Tohumlar arasinda ortalama +- standart sapma verir. Tek kosu farklari
gurultudur; ancak tohumlar arasi fark std'den buyukse bir sey soyleyebiliriz.

Kullanim:  python summarize.py            (results/gpt_*.json okur)
           python summarize.py --csv ozet.csv
"""

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--glob", default="gpt_*.json")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    groups = defaultdict(list)
    for p in sorted(Path(args.dir).glob(args.glob)):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            print(f"atlandi (yarim dosya): {p}")
            continue
        a, e = r["args"], r["env"]
        # Eski kosularda --init bayragi yoktu. O surumde yogun katman std=0.02,
        # seyrek katman kaiming aliyordu - yani KARISIK ve esitsiz. 'gpt' diye
        # etiketlemek yanlis olur; ayri bir isim veriyoruz.
        # kilavuz (self-guided) ayri bir grup: ayni desenin kilavuzlu ve
        # kilavuzsuz halleri karistirilmamali
        kl = "kilavuz" if a.get("guide_frac", 0) > 0 else "-"
        # bmm ve maskeli kelebek AYNI modeldir ama farkli hizda kosar; ayni
        # gruba koyarsak tok/s ortalamasi anlamsizlasir. Ayri tutuluyor.
        if a.get("fast"):
            kl = (kl + "+bmm") if kl != "-" else "bmm"
        groups[(a["ffn"], a["ffn_mult"], a.get("init", "mixed"), a["n_layer"], kl)].append({
            "bpc": r["final_val_bpc"],
            "toks": r["tokens_per_sec"],
            "params": e["params_total_effective"],
            "ffn_params": e["params_ffn"],
            "hidden": e["ffn_hidden"],
            "cov": e["ffn_coverage"],
            "mem": r.get("peak_mem_gb", 0.0),
            "gpu": e.get("gpu", "?"),
        })

    if not groups:
        raise SystemExit(f"sonuc bulunamadi: {args.dir}/{args.glob}")

    # her init modunun KENDI dense tabani var - modlar arasi bpc kiyaslanmaz
    base = {}
    for (ffn, mult, init, L, kl), rows in groups.items():
        if ffn == "dense":
            b = [r["bpc"] for r in rows]
            sd = st.stdev(b) if len(b) > 1 else 0.0
            # taban derinlige de bagli: derin model zaten daha iyi, karsilastirma
            # ayni derinlikteki dense'e karsi yapilmali
            base[(init, L)] = (st.mean(b), sd / (len(b) ** 0.5))  # dense hep kilavuzsuz

    hdr = (f"{'varyant':16s}{'M':>4s}{'L':>3s}{'uygulama':>11s}{'init':>9s}{'gizli':>7s}{'yogunluk':>10s}"
           f"{'kapsama':>9s}{'FFN par.':>11s}{'val bpc':>10s}{'+-':>7s}"
           f"{'dense fark':>12s}{'anlamli?':>16s}{'k tok/s':>9s}{'kart':>6s}{'n':>3s}")
    print(hdr)
    print("-" * len(hdr))

    karma_uyari = []
    csv = ["varyant,mult,n_layer,kilavuz,init,gizli,yogunluk,kapsama,ffn_params,val_bpc,std,dense_farki,anlamli,tok_s,n"]
    for (ffn, mult, init, L, kl), rows in sorted(
            groups.items(),
            key=lambda kv: (kv[0][2], kv[0][3], kv[0][1], kv[0][0], kv[0][4])):
        bpcs = [r["bpc"] for r in rows]
        m = st.mean(bpcs)
        sd = st.stdev(bpcs) if len(bpcs) > 1 else 0.0
        r0 = rows[0]
        dens = 4.0 / mult
        # Fark gurultuden ayirt edilebiliyor mu? Iki ortalamanin standart
        # hatalarini birlestirip farki ona boluyoruz. Oran < 2 ise sonucu
        # "kazandi/kaybetti" diye sunmak yaniltici olur.
        bm = base.get((init, L))
        sem = sd / (len(bpcs) ** 0.5)
        if bm is None or ffn == "dense":
            delta, verdict = "-" if bm is None else "+0.0000", ""
        else:
            d = m - bm[0]
            csem = (sem ** 2 + bm[1] ** 2) ** 0.5
            ratio = abs(d) / csem if csem > 0 else float("inf")
            delta = f"{d:+.4f}"
            if ratio < 2:
                verdict = f"gurultu ({ratio:.1f}o)"
            else:
                verdict = ("SEYREK IYI" if d < 0 else "yogun iyi") + f" ({ratio:.1f}o)"
        # Ayni grupta birden fazla kart tipi varsa tok/s ortalamasi anlamsizdir:
        # A100 ile H200'u karistirmak "seyrek yavas mi" sorusunu cevaplanamaz kilar.
        kartlar = {r["gpu"].replace("NVIDIA ", "").split("-")[0] for r in rows}
        kart = "KARMA" if len(kartlar) > 1 else next(iter(kartlar))[:6]
        if len(kartlar) > 1:
            karma_uyari.append(f"{ffn} M={mult} L={L} {init}: {sorted(kartlar)}")
        print(f"{ffn:16s}{mult:4d}{L:3d}{kl:>11s}{init:>9s}{r0['hidden']:7d}{dens:10.3f}{r0['cov']:9.3f}"
              f"{r0['ffn_params']:11,d}{m:10.4f}{sd:7.4f}{delta:>12s}{verdict:>16s}"
              f"{st.mean(r['toks'] for r in rows)/1e3:9.1f}{kart:>6s}{len(rows):3d}")
        csv.append(f"{ffn},{mult},{L},{kl},{init},{r0['hidden']},{dens:.4f},{r0['cov']:.4f},"
                   f"{r0['ffn_params']},{m:.5f},{sd:.5f},{delta},{verdict},"
                   f"{st.mean(r['toks'] for r in rows):.1f},{len(rows)}")

    print("\nbpc dusuk = iyi. 'anlamli?' sutunu farkin kac standart hataya (o)")
    print("karsilik geldigini verir. 2o altindaki bir farki 'kazandi/kaybetti'")
    print("diye sunmak yaniltici olur - o yuzden 'gurultu' yaziyor.")
    print("Kapsama = FFN'in iki adimda karistirabildigi girdi orani (yapinin olcusu).")
    if karma_uyari:
        print("\nUYARI - su gruplar farkli kart tiplerinde kosmus, 'k tok/s' sutunu")
        print("bu satirlar icin ANLAMSIZ (kalite sutunlari etkilenmez):")
        for u in karma_uyari:
            print("  " + u)

    if args.csv:
        Path(args.csv).write_text("\n".join(csv), encoding="utf-8")
        print(f"\nyazildi: {args.csv}")


if __name__ == "__main__":
    main()
