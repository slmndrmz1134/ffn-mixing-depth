#!/usr/bin/env python
"""Esit parametre butcesinde topoloji karsilastirmasi.

Sorulan soru: poligon/graf topolojisi, ayni parametre butcesindeki yogun bir
aga karsi ne kadar iyi ogrenir? Adil karsilastirma icin uc rakip vardir:

  1. dense_full    : ayni genislikte tam bagli (ust sinir, cok daha fazla parametre)
  2. dense_matched : DAHA DAR ama seyrek modelle ayni parametre sayisina sahip
                     tam bagli ag  <-- asil rakip budur
  3. topoloji      : ring / watts_strogatz / random, genis ama seyrek

3 numaranin 2 numarayi gecmesi, "genis ve seyrek olmak, dar ve yogun olmaktan
iyidir" demektir - fikrin gercek testi budur. 1 numarayi gecmesi beklenmez.

Ornek:
  python train_topology.py --task mixing --width 1024 --degree 16 --steps 3000
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from polytopo import data, layers, models, topology


def build_variant(kind, args, device):
    """(model, etiket, maske_istatistigi) dondurur."""
    if kind == "dense_full":
        mask = topology.build("dense", args.width)
        model = models.TopoMLP(args.in_dim, args.out_dim, args.width,
                               args.depth, mask, impl="dense")
        return model, {"topology": "dense_full", "width": args.width}

    if kind == "dense_matched":
        # seyrek modelin gizli katman nnz'sine esit parametreli dar yogun ag:
        # width_small^2 = width * (k+1)  =>  width_small = sqrt(width*(k+1))
        w_small = max(8, int(round((args.width * (args.degree + 1)) ** 0.5)))
        mask = topology.build("dense", w_small)
        model = models.TopoMLP(args.in_dim, args.out_dim, w_small,
                               args.depth, mask, impl="dense")
        return model, {"topology": "dense_matched", "width": w_small}

    mask = topology.build(kind, args.width, k=args.degree, p=args.rewire, seed=args.seed)
    st = topology.stats(mask, seed=args.seed)
    model = models.TopoMLP(args.in_dim, args.out_dim, args.width,
                           args.depth, mask, impl=args.impl)
    return model, {"topology": kind, "width": args.width, **st}


def evaluate(model, x, y, batch):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            xb, yb = x[i:i + batch], y[i:i + batch]
            total += nn.functional.mse_loss(model(xb), yb, reduction="sum").item()
            n += yb.numel()
    model.train()
    return total / n


def train_one(kind, args, device, xtr, ytr, xte, yte):
    torch.manual_seed(args.seed)
    model, meta = build_variant(kind, args, device)
    model = model.to(device)
    n_params = layers.count_effective_params(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    curve = []
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, xtr.shape[0], (args.batch,), device=device)
        loss = nn.functional.mse_loss(model(xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, args.steps // 20) == 0 or step == args.steps:
            curve.append({"step": step, "train_loss": loss.item()})

    if device == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0

    test_mse = evaluate(model, xte, yte, args.batch)
    res = {
        **meta,
        "variant": kind,
        "effective_params": n_params,
        "train_seconds": train_s,
        "sec_per_1k_steps": train_s / args.steps * 1000,
        "final_train_loss": curve[-1]["train_loss"],
        "test_mse": test_mse,
        "curve": curve,
    }
    print(f"[{kind:15s}] params={n_params:>10,}  "
          f"test_mse={test_mse:.5f}  sure={train_s:6.1f}s  "
          f"yol={meta.get('avg_path_len', 1.0):.2f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mixing", choices=["mixing", "teacher"])
    ap.add_argument("--width", type=int, default=1024, help="seyrek modelin gizli genisligi")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--degree", type=int, default=16, help="poligon komsu sayisi k")
    ap.add_argument("--rewire", type=float, default=0.1)
    ap.add_argument("--impl", default="masked", choices=["masked", "sparse"],
                    help="masked: dogruluk deneyleri icin. sparse: hiz olcumu icin.")
    ap.add_argument("--in-dim", type=int, default=128, dest="in_dim")
    ap.add_argument("--out-dim", type=int, default=16, dest="out_dim")
    ap.add_argument("--n-train", type=int, default=50000, dest="n_train")
    ap.add_argument("--n-test", type=int, default=5000, dest="n_test")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="+",
                    default=["dense_full", "dense_matched", "ring",
                             "watts_strogatz", "random"])
    ap.add_argument("--out", default="results/train_topology.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
    }
    print(json.dumps({**env, **vars(args)}, indent=2, ensure_ascii=False), flush=True)

    # AYNI gizli kural (func_seed=args.seed), FARKLI ornekler (sample_seed ayri).
    # Boylece egitim ve test ayni fonksiyonu paylasir; test verisi yalnizca
    # egitimde gorulmemis girdilerden olusur.
    task = data.make_task(args.task, args.in_dim, args.out_dim, func_seed=args.seed)
    xtr, ytr = task.sample(args.n_train, sample_seed=1000 + args.seed, device=device)
    xte, yte = task.sample(args.n_test, sample_seed=9000 + args.seed, device=device)

    rows = [train_one(k, args, device, xtr, ytr, xte, yte) for k in args.variants]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"env": env, "args": vars(args), "rows": rows},
                              indent=2, ensure_ascii=False))
    print(f"\nyazildi: {out}", flush=True)

    # asil karsilastirmayi acikca yaz
    matched = next((r for r in rows if r["variant"] == "dense_matched"), None)
    if matched:
        print("\n--- asil test: seyrek topoloji vs esit parametreli dar yogun ag ---")
        for r in rows:
            if r["variant"] in ("dense_full", "dense_matched"):
                continue
            delta = matched["test_mse"] - r["test_mse"]
            verdict = "KAZANDI" if delta > 0 else "kaybetti"
            print(f"  {r['variant']:15s} {verdict}  (fark={delta:+.5f})")


if __name__ == "__main__":
    main()
