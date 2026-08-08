#!/usr/bin/env python
"""GPT-mini: yogun FFN'e karsi topolojili FFN, ESIT parametre ve ESIT FLOP ile.

Bu, projenin asil benchmark'i. Katman ve oyuncak-MLP deneylerinin sonucunun
gercek bir dil modelinde ayakta kalip kalmadigini olcer.

Karsilastirilan varyantlar (hepsi ayni nnz):
  dense            gizli 4d, yogun            (referans)
  ring   --mult M  gizli Md, poligon bandi    (yerel, dusuk karistirma)
  watts_strogatz   gizli Md, kucuk-dunya      (yerel + birkac uzun kenar)
  random           gizli Md, rastgele seyrek  (KONTROL: yapi mi, seyreklik mi?)
  block            gizli Md, blok-kosegen     (endustri standardi + gercek hiz)

Adil olculer:
  - Her varyant AYNI token butcesini gorur (--iters sabit).
  - bits-per-character (bpc) raporlanir: text8 icin literaturun olcusu.
  - Duvar saati ayrica raporlanir; kaliteyle KARISTIRILMAZ.

Ornek:
  python train_gpt.py --ffn dense
  python train_gpt.py --ffn watts_strogatz --ffn-mult 16
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from polytopo.gpt import GPTMini


def get_batch(data, block_size, batch_size, device, gen):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, data, args, device, batches=100):
    """Sabit tohumlu degerlendirme: tum varyantlar AYNI dogrulama ornekleri uzerinde."""
    model.eval()
    gen = torch.Generator().manual_seed(1234)  # varyanttan bagimsiz, sabit
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(data, args.block_size, args.batch_size, device, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    nll = total / batches
    return nll, nll / math.log(2)  # (nats, bits-per-character)


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * step / max(1, args.warmup)
    t = (step - args.warmup) / max(1, args.iters - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ffn", default="dense",
                    choices=["dense", "ring", "watts_strogatz", "random",
                             "block", "mozaik", "butterfly"])
    ap.add_argument("--ffn-mult", type=int, default=16, dest="ffn_mult",
                    help="gizli katman = M*d. yogunluk = 4/M. dense icin yok sayilir.")
    ap.add_argument("--rewire", type=float, default=0.1)
    ap.add_argument("--fast", action="store_true",
                    help="kelebegi bmm ile hesapla (maskeli surumle ayni fonksiyon, "
                         "P/2 kat az carpma). Yalnizca --ffn butterfly icin.")
    ap.add_argument("--guide-frac", type=float, default=0.0, dest="guide_frac",
                    help="Self-Guided Training: egitimin ilk bu oranlik kisminda "
                         "paralel yogun bir dal calisir, katsayisi 1'den 0'a iner. "
                         "0 = kapali. dense varyantinda yok sayilir.")
    ap.add_argument("--init", default="gpt", choices=["gpt", "kaiming", "mixed"],
                    help="baslangic olcegi. 'gpt': sabit std=0.02 (gelenek). "
                         "'kaiming': fan_in'e gore - seyrek/yogun sinyal olcegini esitler.")
    ap.add_argument("--n-layer", type=int, default=8, dest="n_layer")
    ap.add_argument("--n-head", type=int, default=8, dest="n_head")
    ap.add_argument("--d-model", type=int, default=512, dest="d_model")
    ap.add_argument("--block-size", type=int, default=512, dest="block_size")
    ap.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5, dest="min_lr")
    ap.add_argument("--weight-decay", type=float, default=0.1, dest="weight_decay")
    ap.add_argument("--eval-every", type=int, default=500, dest="eval_every")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-mask", action="store_true", dest="fixed_mask",
                    help="topolojiyi TUM katmanlarda ayni tut. Varsayilan kurulum "
                         "her katmana farkli tohum verdigi icin random/watts_strogatz "
                         "(ve stage'iyle kelebek) derinlikle DEGISEN desenlerdir; bu "
                         "bayrak o ekseni kapatir. Tohumlar arasi graf hala degisir.")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default="results/gpt.json")
    args = ap.parse_args()

    if args.ffn == "dense":
        args.ffn_mult = 4  # dense tanim geregi 4d

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    d = Path(args.data)
    meta = json.loads((d / "meta.json").read_text())
    train = np.memmap(d / "train.bin", dtype=np.uint8, mode="r")
    val = np.memmap(d / "val.bin", dtype=np.uint8, mode="r")

    model = GPTMini(meta["vocab_size"], args.block_size, args.n_layer, args.n_head,
                    args.d_model, args.ffn, args.ffn_mult, args.rewire, args.seed,
                    init=args.init,
                    guide=(args.guide_frac > 0 and args.ffn != "dense"),
                    fast=args.fast, fixed_mask=args.fixed_mask).to(device)
    rep = model.param_report()
    env = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "vocab_size": meta["vocab_size"],
        "train_tokens": int(len(train)),
        **rep,
    }
    print(json.dumps({**env, **vars(args)}, indent=2, ensure_ascii=False), flush=True)
    kilavuz = (f" | kilavuz +{rep['params_guide']:,} gecici "
               f"(ilk %{args.guide_frac*100:.0f})") if rep.get("params_guide") else ""
    print(f"\n>>> {args.ffn} | gizli={rep['ffn_hidden']} | FFN param={rep['params_ffn']:,} "
          f"| kapsama={rep['ffn_coverage']:.3f}{kilavuz}\n", flush=True)

    if args.compile:
        model = torch.compile(model)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=(device == "cuda"))

    gen = torch.Generator().manual_seed(args.seed)
    curve = []
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(1, args.iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        if args.guide_frac > 0:
            # 1 -> 0 dogrusal inis; guide_frac oraninda tamamen kapanir
            model.set_guide_alpha(max(0.0, 1.0 - step / (args.guide_frac * args.iters)))
        x, y = get_batch(train, args.block_size, args.batch_size, device, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.iters:
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            vnll, vbpc = evaluate(model, val, args, device, batches=50)
            toks = step * args.batch_size * args.block_size
            curve.append({"step": step, "train_loss": loss.item(),
                          "val_bpc": vbpc, "seconds": elapsed, "tokens": toks})
            print(f"adim {step:6d}/{args.iters}  egitim={loss.item():.4f}  "
                  f"val_bpc={vbpc:.4f}  {toks/elapsed/1e3:7.1f}k tok/s  "
                  f"{elapsed/60:5.1f} dk", flush=True)

    if device == "cuda":
        torch.cuda.synchronize()
    total_s = time.perf_counter() - t0
    vnll, vbpc = evaluate(model, val, args, device, batches=200)  # son olcum daha hassas

    res = {
        "env": env, "args": vars(args), "curve": curve,
        "final_val_bpc": vbpc, "final_val_nll": vnll,
        "train_seconds": total_s,
        "tokens_per_sec": args.iters * args.batch_size * args.block_size / total_s,
        "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nSONUC  {args.ffn:15s} mult={args.ffn_mult:3d}  "
          f"param={rep['params_total_effective']:,}  val_bpc={vbpc:.4f}  "
          f"sure={total_s/60:.1f}dk  ->  {out}", flush=True)


if __name__ == "__main__":
    main()
