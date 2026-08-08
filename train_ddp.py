#!/usr/bin/env python
"""GPT-mini VERI-PARALEL egitim (8 GPU DDP) - olcekleme deneyi icin.

Tek-GPU train_gpt.py'i BOZMADAN ayri tutuldu; o dosya butun onceki sonuclari
uretti, dokunmuyoruz.

NEDEN veri-paralel: 400M model tek H200'de ~15-45 saat surer (masked seyrek en
kotusu). 8 GPU'ya veri-paralel yayinca ~8 kat hizlanir; her GPU tam model
kopyasi tutar, farkli veri gorur, adim sonunda gradyanlar otomatik ortalanir
(DDP). Kalite karsilastirmasi degismez - sadece daha hizli.

KULLANIM (SLURM, tek sunucu 8 GPU ya da 2x4):
  srun -N1 --ntasks-per-node=8 --gres=gpu:8 python train_ddp.py --ffn butterfly ...

ADALET: dense/butterfly/random hepsi AYNI token butcesi (--steps sabit), ayni
global batch, ayni veri. Tek degisen FFN topolojisi. Kilavuz (self-guided) YOK -
DDP'de kullanilmayan-parametre sorunu cikariyor; seyrek sayilar bu yuzden
muhafazakar (kilavuz onlari biraz daha iyilestirirdi, bkz. L=16 sonuclari).
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from polytopo.gpt import GPTMini


def setup():
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    local = int(os.environ.get("SLURM_LOCALID", 0))
    os.environ.setdefault("MASTER_PORT", "29531")
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    return rank, world, local


def get_batch(data, ctx, batch, device, gen):
    ix = torch.randint(len(data) - ctx - 1, (batch,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + ctx].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + ctx].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, data, ctx, batch, device, batches=100):
    model.eval()
    gen = torch.Generator().manual_seed(1234)      # sabit, varyanttan bagimsiz
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(data, ctx, batch, device, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    nll = total / batches
    return nll, nll / math.log(2)


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * step / max(1, args.warmup)
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_enwik9")
    ap.add_argument("--ffn", default="dense",
                    choices=["dense", "ring", "watts_strogatz", "random",
                             "block", "mozaik", "butterfly"])
    ap.add_argument("--ffn-mult", type=int, default=16, dest="ffn_mult")
    ap.add_argument("--fast", action="store_true", help="butterfly icin bmm")
    ap.add_argument("--init", default="mixed", choices=["gpt", "kaiming", "mixed"])
    ap.add_argument("--n-layer", type=int, default=32, dest="n_layer")
    ap.add_argument("--n-head", type=int, default=16, dest="n_head")
    ap.add_argument("--d-model", type=int, default=1024, dest="d_model")
    ap.add_argument("--ctx", type=int, default=1024, help="baglam uzunlugu")
    ap.add_argument("--batch", type=int, default=24, help="GPU BASINA batch")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5, dest="min_lr")
    ap.add_argument("--weight-decay", type=float, default=0.1, dest="weight_decay")
    ap.add_argument("--eval-every", type=int, default=1000, dest="eval_every")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/scale.json")
    args = ap.parse_args()
    if args.ffn == "dense":
        args.ffn_mult = 4

    rank, world, local = setup()
    dev = torch.device("cuda", local)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    d = Path(args.data)
    meta = json.loads((d / "meta.json").read_text())
    train = np.memmap(d / "train.bin", dtype=np.uint8, mode="r")
    val = np.memmap(d / "val.bin", dtype=np.uint8, mode="r")

    model = GPTMini(meta["vocab_size"], args.ctx, args.n_layer, args.n_head,
                    args.d_model, args.ffn, args.ffn_mult, seed=args.seed,
                    init=args.init, fast=args.fast).to(dev)
    rep = model.param_report()
    ddp = DDP(model, device_ids=[local])

    tok_per_step = world * args.batch * args.ctx
    if rank == 0:
        env = {"world_size": world, "gpu": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "vocab_size": meta["vocab_size"],
               "train_tokens": int(len(train)), "tokens_per_step": tok_per_step,
               "effective_batch": world * args.batch, **rep}
        print(json.dumps({**env, **vars(args)}, indent=2, ensure_ascii=False), flush=True)
        print(f"\n>>> {args.ffn} d={args.d_model} L={args.n_layer} | "
              f"toplam param={rep['params_total_effective']:,} | "
              f"gizli={rep['ffn_hidden']} | kapsama={rep['ffn_coverage']:.3f} | "
              f"{world} GPU x batch {args.batch} = {tok_per_step:,} tok/adim\n", flush=True)

    decay = [p for p in ddp.parameters() if p.dim() >= 2]
    nodecay = [p for p in ddp.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=True)

    # her rank FARKLI veri gorur (veri-paralelin ta kendisi): tohum + rank
    gen = torch.Generator().manual_seed(args.seed * 100003 + rank)
    curve = []
    dist.barrier(); torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        x, y = get_batch(train, args.ctx, args.batch, dev, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = ddp(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        opt.step()

        if not math.isfinite(loss.item()):
            if rank == 0:
                print(f"IRAKSAMA adim {step}: kayip={loss.item()}", flush=True)
            break

        if step % args.eval_every == 0 or step == args.steps:
            torch.cuda.synchronize()
            el = time.perf_counter() - t0
            if rank == 0:
                vnll, vbpc = evaluate(model, val, args.ctx, args.batch, dev, batches=40)
                toks = step * tok_per_step
                curve.append({"step": step, "train_loss": loss.item(),
                              "val_bpc": vbpc, "seconds": el, "tokens": toks})
                print(f"adim {step:6d}/{args.steps}  egitim={loss.item():.4f}  "
                      f"val_bpc={vbpc:.4f}  {toks/el/1e6:6.2f}M tok/s  "
                      f"{el/60:5.1f} dk", flush=True)
            dist.barrier()

    torch.cuda.synchronize()
    total_s = time.perf_counter() - t0

    if rank == 0:
        vnll, vbpc = evaluate(model, val, args.ctx, args.batch, dev, batches=200)
        res = {"env": env, "args": vars(args), "curve": curve,
               "final_val_bpc": vbpc, "final_val_nll": vnll,
               "train_seconds": total_s,
               "tokens_per_sec": args.steps * tok_per_step / total_s,
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
        outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\nSONUC {args.ffn:10s} d={args.d_model} L={args.n_layer}  "
              f"val_bpc={vbpc:.4f}  sure={total_s/60:.1f}dk  "
              f"param={rep['params_total_effective']:,}  -> {outp}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
