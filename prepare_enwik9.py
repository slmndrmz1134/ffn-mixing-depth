#!/usr/bin/env python
"""enwik9'u (1 GB ham Wikipedia) bayt duzeyi token ikili dosyalarina cevirir.

Neden enwik9, neden bayt duzeyi:
  - text8 sadece 100M karakter; 400M parametreli bir model onu EZBERLER ve
    topoloji farki ezber gurultusunde kaybolur. enwik9 10 kat buyuk (1e9 bayt).
  - Bayt duzeyi (vocab 256) temizlik gerektirmez: her bayt bir token. bpc
    dogrudan Hutter Prize / sikistirma literaturuyle kiyaslanabilir olur.

GIRIS NODE'unda calistirilir (islem IO-agirlikli, birkac saniye CPU).

Kullanim:
  python prepare_enwik9.py --out data_enwik9
  python prepare_enwik9.py --out data_enwik9 --zip enwik9.zip   # elde varsa
"""

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URLS = [
    "http://mattmahoney.net/dc/enwik9.zip",
    "https://data.deepai.org/enwik9.zip",
    "http://web.archive.org/web/2020/http://mattmahoney.net/dc/enwik9.zip",
]


def fetch(dst: Path):
    if dst.exists() and dst.stat().st_size > 100_000_000:
        print(f"zaten var: {dst} ({dst.stat().st_size/1e6:.0f} MB)")
        return
    last = None
    for url in URLS:
        try:
            print(f"deneniyor: {url}  (~322 MB zip, 1 GB acilmis)", flush=True)
            urllib.request.urlretrieve(url, dst)
            print(f"indi: {dst.stat().st_size/1e6:.0f} MB")
            return
        except Exception as e:
            last = e
            print(f"  basarisiz: {e}")
    raise SystemExit(f"hicbir ayna calismadi ({last}). enwik9.zip'i elle indirip "
                     "--zip ile ver.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_enwik9")
    ap.add_argument("--zip", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="sadece ilk N bayti kullan (test icin; 0 = hepsi)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zp = Path(args.zip) if args.zip else out / "enwik9.zip"
    fetch(zp)

    with zipfile.ZipFile(zp) as z:
        raw = z.read(z.namelist()[0])
    if args.limit:
        raw = raw[:args.limit]
    arr = np.frombuffer(raw, dtype=np.uint8)
    print(f"ham metin: {len(arr)/1e6:.0f} M bayt")

    # BAYT DUZEYI: donusum yok, her bayt zaten 0-255. Model 256 sembol gorur.
    n = len(arr)
    c1, c2 = int(n * 0.90), int(n * 0.95)      # 90 / 5 / 5 bolme
    for name, sl in [("train", slice(0, c1)),
                     ("val", slice(c1, c2)),
                     ("test", slice(c2, n))]:
        p = out / f"{name}.bin"
        arr[sl].tofile(p)
        print(f"{name:6s} {(sl.stop - sl.start)/1e6:8.1f} M token -> {p}")

    # gorulen bayt degerleri (bazi degerler hic gecmeyebilir ama vocab=256 sabit)
    (out / "meta.json").write_text(json.dumps({
        "vocab_size": 256,
        "level": "byte",
        "note": "enwik9 bayt duzeyi; bpc Hutter Prize ile kiyaslanabilir",
    }, ensure_ascii=False, indent=2))
    print(f"\nhazir: {out}/meta.json  (vocab_size=256, bayt duzeyi)")


if __name__ == "__main__":
    main()
