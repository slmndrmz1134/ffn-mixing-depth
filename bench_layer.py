#!/usr/bin/env python
"""Katman duzeyi mikro-benchmark: teorik kazanim vs gercek duvar saati.

Ayni matematiksel islemi uc farkli sekilde olcer (dense / masked / sparse) ve
her biri icin ileri+geri gecis suresini, FLOP'u ve etkin parametre sayisini
raporlar. Amac su soruyu yanitlamak:

  "Seyrek poligon topolojisi parametreyi X kat dusuruyor - peki sure de
   dusuyor mu, yoksa kernel darbogazi yuzunden artiyor mu?"

Bu ayrimi yapmadan cikarilacak her hiz sonucu yaniltici olur.

Ornek:
  python bench_layer.py --width 4096 --batch 256 --degrees 6 16 64 --out results/bench.json
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from polytopo import layers, topology


def timed(fn, warmup=10, iters=50, device="cuda"):
    """Isinma sonrasi ortalama sure (ms). CUDA icin senkronizasyon sart."""
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def bench_one(impl, mask, batch, device, dtype, iters):
    layer = layers.make_layer(impl, mask).to(device=device, dtype=dtype)
    x = torch.randn(batch, mask.shape[1], device=device, dtype=dtype, requires_grad=True)

    def fwd():
        layer(x)

    def fwd_bwd():
        y = layer(x)
        y.sum().backward()
        layer.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    nnz = int(mask.sum()) if impl != "dense" else mask.size
    return {
        "impl": impl,
        "fwd_ms": timed(fwd, iters=iters, device=device),
        "fwd_bwd_ms": timed(fwd_bwd, iters=iters, device=device),
        "effective_params": layers.count_effective_params(layer),
        # matmul icin 2 islem (carpma + toplama) x nnz x batch
        "flops_per_sample": 2 * nnz,
        "nnz": nnz,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=4096, help="kare katman boyutu n")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--degrees", type=int, nargs="+", default=[6, 16, 64, 256],
                    help="poligon komsu sayilari (k)")
    ap.add_argument("--rewire", type=float, default=0.1, help="watts-strogatz p")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bench_layer.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)

    env = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "width": args.width,
        "batch": args.batch,
    }
    print(json.dumps(env, indent=2, ensure_ascii=False), flush=True)

    rows = []

    # --- referans: tam bagli katman ---
    dense_mask = topology.build("dense", args.width)
    base = bench_one("dense", dense_mask, args.batch, device, dtype, args.iters)
    base.update(topology=  "dense", k=args.width, avg_path_len=1.0, diameter=1)
    rows.append(base)
    print(f"[dense]  fwd={base['fwd_ms']:.3f}ms  "
          f"fwd_bwd={base['fwd_bwd_ms']:.3f}ms  params={base['effective_params']:,}",
          flush=True)

    # --- seyrek topolojiler ---
    for k in args.degrees:
        if k >= args.width:
            continue
        for kind in ("ring", "watts_strogatz", "random"):
            mask = topology.build(
                kind, args.width, k=k, p=args.rewire, seed=args.seed
            )
            st = topology.stats(mask, seed=args.seed)
            for impl in ("masked", "sparse"):
                try:
                    r = bench_one(impl, mask, args.batch, device, dtype, args.iters)
                except RuntimeError as e:  # bazi dtype/kernel kombinasyonlari desteklenmez
                    print(f"[{kind} k={k} {impl}] ATLANDI: {e}", flush=True)
                    continue
                r.update(
                    topology=kind, k=k,
                    density=st["density"],
                    avg_path_len=st["avg_path_len"],
                    diameter=st["diameter"],
                    param_reduction=base["effective_params"] / max(1, r["effective_params"]),
                    speedup_vs_dense=base["fwd_bwd_ms"] / r["fwd_bwd_ms"],
                )
                rows.append(r)
                print(f"[{kind:15s} k={k:4d} {impl:7s}] "
                      f"fwd_bwd={r['fwd_bwd_ms']:8.3f}ms  "
                      f"param x{r['param_reduction']:6.1f} dusuk  "
                      f"hiz x{r['speedup_vs_dense']:5.2f}  "
                      f"yol={st['avg_path_len']:.2f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"env": env, "args": vars(args), "rows": rows},
                              indent=2, ensure_ascii=False))
    print(f"\nyazildi: {out}", flush=True)

    # okuyucuya dogrudan uyari: parametre dususu hiz demek degildir
    best = max((r for r in rows if r.get("speedup_vs_dense")),
               key=lambda r: r["speedup_vs_dense"], default=None)
    if best and best["speedup_vs_dense"] < 1.0:
        print("\nDIKKAT: hicbir seyrek uygulama yogun matmul'u gecemedi. "
              "Bu, fikrin degil kernel'in sinirini gosterir - raporda "
              "FLOP/parametre kazanimi ile duvar saati ayri sunulmali.", flush=True)


if __name__ == "__main__":
    main()
