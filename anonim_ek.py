#!/usr/bin/env python
"""TMLR cift-kor gonderimi icin ANONIM ek materyal paketi uretir.

Makalenin yeniden-uretilebilirlik paragrafi "provided as supplementary material
for review" diyor; bu betik o vaadi karsilayan paketi kurar.

ANONIMLESTIRME iki katmanli:
  1. DOSYA duzeyi  - kimlik tasiyan dosyalar hic alinmaz
                     (README.md, LICENSE, .zenodo.json, notes/, .tex, paket betikleri)
  2. ICERIK duzeyi - SLURM betiklerindeki kume/hesap adlari maskelenir
                     (hesap kodu, bolum, rezervasyon, modul adi kurumu ele verir)

Sonda dogrulama var: paketteki HER dosya kimlik terimlerine karsi taranir.

  python anonim_ek.py
"""
import io
import os
import re
import shutil
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

HEDEF = "supplementary_anon"
ZIP = "supplementary_material_anon.zip"

# --- alinacaklar ---
KLASORLER = ["polytopo", "results", "slurm"]
DOSYALAR = [
    "analiz_tau.py", "dogrula_makale.py", "gecikme.py", "karar.py",
    "make_figs.py", "make_figs_en.py", "make_tables.py", "summarize.py",
    "train_gpt.py", "train_topology.py", "train_ddp.py", "train_dist_ffn.py",
    "bench_layer.py", "bench_distributed.py",
    "prepare_text.py", "prepare_enwik9.py",
    "requirements.txt",
]

# --- SLURM icerik maskeleme: (desen, karsilik) ---
MASKE = [
    (r"#SBATCH -A egitimg35\b", "#SBATCH -A <account>"),
    (r"#SBATCH -p kolyoz-cuda\b", "#SBATCH -p <gpu-partition>"),
    (r"#SBATCH --reservation=yzup\b", "#SBATCH --reservation=<reservation>"),
    (r"module load apps/truba-ai/gpu-2024\.0", "module load <cuda+pytorch module>"),
    (r"kolyoz36\+37|kolyoz36-37", "two reserved nodes"),
    (r"kolyoz-cuda", "<gpu-partition>"),
    (r"\bkolyoz\b", "compute node"),
    (r"egitimg35[a-z0-9]*", "<account>"),
    (r"\byzup\b", "<reservation>"),
    (r"truba-ai/gpu-2024\.0", "<cuda+pytorch module>"),
]

OKUBENI = """Supplementary material (anonymized for double-blind review)
==========================================================

Code, raw experimental records, and analysis for the submitted paper.

WHAT IS HERE
------------
polytopo/            topology constructors (ffn_coverage, cumulative_coverage),
                     GPT-mini with a pluggable FFN, masked/bmm/butterfly layers
results/             raw record of every run, one JSON each
  _eski/               runs from an older code version and exact duplicates
  _duman/              smoke tests
slurm/               the experimental grids, one script per experiment family
analiz_tau.py        recomputes every coverage, tau and significance value in the paper
gecikme.py           the latency floor R*alpha
karar.py             the bandwidth decision model
make_figs*.py        regenerates the figures from results/
dogrula_makale.py    cross-checks the paper's numbers against the raw records
train_*.py           the training entry points
bench_*.py           layer and multi-GPU communication benchmarks

REPRODUCING THE PAPER'S NUMBERS
-------------------------------
    pip install -r requirements.txt
    python analiz_tau.py          # tau tables, depth regimes, fixed-mask verdict
    python gecikme.py             # latency floor
    python make_figs_en.py        # figures

No number in the tables or figures is hand-entered; all are recomputed from
results/ by these scripts.

PRE-REGISTERED PREDICTIONS
--------------------------
The paper states that two predictions were recorded before the corresponding
runs were queued. This can be verified directly: the predictions are written in
the headers of

    slurm/gpt_sabit_maske.slurm        (the fixed-mask control)
    slurm/gpt_yogunluk_doldur.slurm    (the M=32/64 density fill)

RECORDS THAT MUST NOT BE ANALYSED
---------------------------------
results/_eski/ holds nine layer-scale records whose recorded arguments are
identical to valid runs yet whose outcomes differ by a factor of 46: three came
from an older version of train_topology.py and three are byte-identical
duplicates. Nothing in the arguments distinguishes them, so pooling all nine
makes every difference dissolve into noise. results/_duman/ holds smoke tests.
The analysis scripts read only the top level of results/.

NOTES
-----
Source comments are in Turkish; identifiers and this file are in English.
Cluster-specific SLURM directives (account, partition, reservation, module) are
masked as <placeholders> for anonymity; replace them for your own site. A single
run needs none of them -- see the train_gpt.py invocation in any slurm header.
"""

# kimlik terimleri (dogrulama icin)
IZLER = ["slmndrmz", "Durmaz", "Selman", "Mustafa", "Beykent", "beykent",
         "TRUBA", "truba", "TUBITAK", "TÜBİTAK", "egitimg35", "kolyoz", "yzup",
         "zenodo", "github.com", "orcid", "ORCID", "0009-0006", "SELMAN"]


def main():
    if os.path.isdir(HEDEF):
        shutil.rmtree(HEDEF)
    if os.path.isfile(ZIP):
        os.remove(ZIP)
    os.makedirs(HEDEF)

    n_dosya = 0
    for k in KLASORLER:
        for kok, dirs, files in os.walk(k):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                src = os.path.join(kok, f)
                dst = os.path.join(HEDEF, src)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if f.endswith((".slurm", ".sh")):
                    s = io.open(src, encoding="utf-8").read()
                    for desen, yeni in MASKE:
                        s = re.sub(desen, yeni, s)
                    io.open(dst, "w", encoding="utf-8", newline="\n").write(s)
                else:
                    shutil.copy2(src, dst)
                n_dosya += 1

    for f in DOSYALAR:
        if os.path.exists(f):
            shutil.copy2(f, os.path.join(HEDEF, f))
            n_dosya += 1
        else:
            print(f"  ! yok, atlandi: {f}")

    io.open(os.path.join(HEDEF, "README.txt"), "w",
            encoding="utf-8", newline="\n").write(OKUBENI)
    n_dosya += 1

    # ---------------- DOGRULAMA ----------------
    print(f"paket: {n_dosya} dosya")
    print("\nkimlik taramasi (metin dosyalarinin TAMAMI):")
    bulgular = {}
    tarandi = 0
    for kok, _, files in os.walk(HEDEF):
        for f in files:
            p = os.path.join(kok, f)
            if not f.endswith((".py", ".slurm", ".sh", ".json", ".txt", ".md", ".csv")):
                continue
            tarandi += 1
            try:
                s = io.open(p, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            for iz in IZLER:
                if iz in s:
                    bulgular.setdefault(iz, []).append(os.path.relpath(p, HEDEF))
    print(f"  taranan dosya: {tarandi}")
    if bulgular:
        for iz, yerler in sorted(bulgular.items()):
            print(f"  !! {iz:14s} -> {yerler[:4]}{' ...' if len(yerler) > 4 else ''}")
    else:
        print("  KIMLIK IZI YOK - temiz")

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for kok, _, files in os.walk(HEDEF):
            for f in files:
                p = os.path.join(kok, f)
                z.write(p, os.path.relpath(p, HEDEF))
    mb = os.path.getsize(ZIP) / 1024 / 1024
    print(f"\n{ZIP}  ({mb:.1f} MB)   [TMLR siniri: 100 MB]")


if __name__ == "__main__":
    main()
