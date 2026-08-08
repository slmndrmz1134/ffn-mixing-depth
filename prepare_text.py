#!/usr/bin/env python
"""text8 veri setini indirip karakter-token ikili dosyalarina cevirir.

GIRIS NODE'UNDA calistirilir (hesaplama node'larinda internet YOK). Islem
neredeyse tamamen IO'dur; CPU kullanimi birkac saniyedir, 300 saniyelik giris
node'u limitine yaklasmaz.

Neden text8: 100 MB temiz Wikipedia metni, 27 karakterlik alfabe (a-z + bosluk).
Tokenizer indirmeye gerek yok, sonuc bits-per-character olarak literaturle
dogrudan karsilastirilabilir. Bolme literaturdeki standarttir: 90M/5M/5M.

Kullanim:
  python prepare_text.py --out data
  python prepare_text.py --out data --src /yol/kendi_metnim.txt   # indirmeden
"""

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URLS = [
    "http://mattmahoney.net/dc/text8.zip",
    "https://data.deepai.org/text8.zip",
    "http://web.archive.org/web/2020/http://mattmahoney.net/dc/text8.zip",
]


def fetch(dst: Path):
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f"zaten var, indirme atlandi: {dst} ({dst.stat().st_size/1e6:.1f} MB)")
        return
    last = None
    for url in URLS:
        try:
            print(f"deneniyor: {url}", flush=True)
            urllib.request.urlretrieve(url, dst)
            print(f"indi: {dst.stat().st_size/1e6:.1f} MB")
            return
        except Exception as e:  # aynadan aynaya gec
            last = e
            print(f"  basarisiz: {e}")
    raise SystemExit(
        f"hicbir ayna calismadi ({last}).\n"
        "Cozum: text8.zip'i kendi bilgisayarina indirip scp ile at, sonra\n"
        "  python prepare_text.py --out data --zip text8.zip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--zip", default=None, help="elde hazir text8.zip varsa yolu")
    ap.add_argument("--src", default=None, help="duz metin dosyasi (indirme yok)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.src:
        raw = Path(args.src).read_bytes()
    else:
        zp = Path(args.zip) if args.zip else out / "text8.zip"
        fetch(zp)
        with zipfile.ZipFile(zp) as z:
            raw = z.read(z.namelist()[0])
    print(f"ham metin: {len(raw)/1e6:.1f} M karakter")

    arr = np.frombuffer(raw, dtype=np.uint8)
    vocab = np.unique(arr)
    # 256'lik arama tablosu: 100M karakteri tek adimda esler (hizli, az CPU)
    lut = np.zeros(256, dtype=np.uint8)
    lut[vocab] = np.arange(len(vocab), dtype=np.uint8)
    ids = lut[arr]
    print(f"sozluk: {len(vocab)} karakter -> {''.join(chr(c) for c in vocab)!r}")

    n = len(ids)
    # standart text8 bolmesi: ilk 90M egitim, sonraki 5M dogrulama, son 5M test
    cuts = (int(n * 0.90), int(n * 0.95)) if n < 95_000_000 else (90_000_000, 95_000_000)
    for name, sl in [("train", slice(0, cuts[0])),
                     ("val", slice(cuts[0], cuts[1])),
                     ("test", slice(cuts[1], n))]:
        p = out / f"{name}.bin"
        ids[sl].tofile(p)
        print(f"{name:6s} {(sl.stop - sl.start)/1e6:7.2f} M token -> {p}")

    (out / "meta.json").write_text(json.dumps({
        "vocab_size": int(len(vocab)),
        "itos": [chr(int(c)) for c in vocab],
    }, ensure_ascii=False, indent=2))
    print(f"\nhazir: {out}/meta.json")


if __name__ == "__main__":
    main()
