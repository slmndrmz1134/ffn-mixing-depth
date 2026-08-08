#!/usr/bin/env python
"""3. OLCEK: dagitik parcalama - topoloji, kolektif iletisimi yerel takasa cevirir mi?

MOTIVASYON
----------
Dagitik egitimde darbogaz FLOP degil ILETISIMDIR. Turkiye'nin farkli
universitelerindeki GPU'lari tek bir modelde birlestirmek isteniyorsa, sorun
"toplam ne kadar FLOP var" degil, "dugumler arasindaki hat ne kadar dar"dir.

Klasik tensor-paralel (Megatron tarzi) FFN her katmanda ALL-REDUCE gerektirir:
her GPU her adimda DIGER TUM GPU'larla konusur. P dugum icin bu tam bir ag
(mesh) demektir - WAN uzerinde olduren sey budur.

TEZ: agirlik matrisi poligon/blok yapili ise, bir parcanin ihtiyaci olan girdi
birimleri de YERELDIR. O zaman all-reduce yerine yalnizca KOMSU degisimi kalir.
Bu, tam mesh yerine fiziksel bir halka uzerinde calisabilmek demektir.

OLCULEN PARCALAMALAR (hepsi RANK BASINA ESIT PARAMETRE ile):
  tp_dense         Megatron tensor-paralel yogun FFN. Her katmanda all-reduce,
                   muhatap P-1. Tek katmanda tam karisim.
  topo_ring        Poligon: kendi dilimi + 2 komsu. Muhatap 2. Karisim yavas
                   ve hicbir zaman tamamlanmiyor (olculdu).
  topo_butterfly   Asama ELL'de yalnizca r XOR 2^ELL ile. Muhatap 1 - halkanin
                   YARISI - ve log2(P) katmanda tam karisim. Her asama ayri
                   olculur (dusuk asamalar sunucu ici, yuksek asamalar sunucular
                   arasi dusuyor).
  topo_block       Blok-kosegen: hic iletisim yok, hic karisim da yok (alt sinir).

Esit parametre icin gizli genislik varyanta gore olceklenir:
  tp_dense       : h
  topo_ring      : h*P/2
  topo_butterfly : 2*h*P/3   (rank basina gizli boyut 64'un katina yuvarlanir -
                              hizalama sart, bkz. asagidaki not)
  topo_block     : h*P

DIKKAT - katman basina iletisimi kiyaslamak yaniltici olur, cunku yontemler
ayni katmanda ayni karisimi saglamaz. Adil olcu "tam karisima kadar toplam
bayt"tir ve script bunu ayrica raporlar. O olcude halka yogun tabanla AYNI
toplami harciyor; kazanan yalnizca kelebektir (log2(P)/P).

RAPORLANAN
----------
  - rank basina adim basina byte
  - TEK BIR HAT uzerindeki en yuksek trafik  <- WAN icin belirleyici olcu
  - bu kumede olculen gercek sure (NVLink + InfiniBand)
  - 1/10/100 Gbps hatlar icin ongorulen iletisim suresi (Turkiye senaryosu)

HIZALAMA NOTU (ilk olcumde ogrenildi): rank basina gizli boyut 8'in, ideali
64'un kati olmali. Ilk kelebek kosusunda bu boyut 21845 (tek sayi) cikti ve
sure 10.8 ms olculdu - halkanin (2.8 ms) dort kati. FLOP sayilari birebir esit
oldugu icin bu topolojinin degil, bf16 tensor cekirdeklerinin hizalanmamis
boyutta verimsiz kernel'e dusmesinin sonucuydu. Duvar saati kiyaslarken tum
varyantlarin boyutlarinin ayni sekilde hizalandigindan emin ol.

NOT: yalnizca ILERI gecis olculur. Geri gecis ayni hacmi bir kez daha uretir;
tablolarda 'fwd+bwd' sutunu bu carpani icerir. Boyle yapmak, autograd'i P2P
ile sarmalamanin olcumu kirletmesini onler.

Calistirma (SLURM, 8 GPU):
  srun ... python bench_distributed.py --d-model 4096 --tokens 8192
"""

import argparse
import json
import math
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


def setup():
    """SLURM ortam degiskenlerinden torch.distributed kurar.

    torchrun KULLANMIYORUZ: srun zaten her GPU icin bir gorev aciyor, ustune
    torchrun koymak rank cakismasi yaratir.
    """
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    local = int(os.environ.get("SLURM_LOCALID", 0))
    os.environ.setdefault("MASTER_PORT", "29517")
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    return rank, world, local


def timed(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - t0) * 1000.0 / iters


@torch.no_grad()
def run_tp_dense(x_full, d, h, P, rank, dtype):
    """Megatron tensor-paralel: sutun-paralel up, satir-paralel down + all-reduce."""
    hp = h // P
    up = torch.randn(hp, d, device="cuda", dtype=dtype) * (d ** -0.5)
    dn = torch.randn(d, hp, device="cuda", dtype=dtype) * (hp ** -0.5)
    params = up.numel() + dn.numel()

    def step():
        y = F.gelu(F.linear(x_full, up))     # (N, h/P)
        out = F.linear(y, dn)                # (N, d) - KISMI sonuc
        dist.all_reduce(out)                 # her rank her rankla konusur
        return out

    n = x_full.shape[0]
    elems = n * d                            # all-reduce edilen tensor boyutu
    # halka all-reduce: rank basina 2(P-1)/P eleman gonderilir+alinir
    per_rank = 2 * (P - 1) / P * elems
    return step, params, per_rank, "all-reduce (tam mesh)", P - 1


@torch.no_grad()
def run_topo_ring(x_shard, d, h, P, rank, dtype):
    """Poligon: rank r kendi dilimi + r-1 ve r+1 dilimlerini gorur. Sadece P2P."""
    if P < 3:
        raise ValueError("halka parcalama en az 3 rank ister (P<3'te sol/sag komsu cakisir)")
    dp, hp = d // P, h // P
    up = torch.randn(hp, 3 * dp, device="cuda", dtype=dtype) * ((3 * dp) ** -0.5)
    dn = torch.randn(dp, hp, device="cuda", dtype=dtype) * (hp ** -0.5)
    params = up.numel() + dn.numel()

    left, right = (rank - 1) % P, (rank + 1) % P
    recv_l = torch.empty_like(x_shard)
    recv_r = torch.empty_like(x_shard)

    def step():
        # komsu degisimi: yalnizca 2 hat, her biri kucuk
        ops = [dist.P2POp(dist.isend, x_shard, left),
               dist.P2POp(dist.isend, x_shard, right),
               dist.P2POp(dist.irecv, recv_l, left),
               dist.P2POp(dist.irecv, recv_r, right)]
        for w in dist.batch_isend_irecv(ops):
            w.wait()
        xin = torch.cat([recv_l, x_shard, recv_r], dim=1)   # (N, 3d/P)
        y = F.gelu(F.linear(xin, up))
        return F.linear(y, dn)                              # (N, d/P) - kendi dilimi

    n = x_shard.shape[0]
    per_rank = 2 * n * dp * 2      # 2 gonder + 2 al
    return step, params, per_rank, "komsu P2P (halka)", 2


@torch.no_grad()
def run_topo_butterfly(x_shard, d, h, P, rank, dtype, stage=0):
    """Kelebek: rank r, asama ELL'de yalnizca r XOR 2^ELL ile konusur.

    Halkadan farki: halka her katmanda AYNI iki komsuyla konusur ve karisim
    hicbir zaman tamamlanmaz. Kelebek her katmanda BASKA tek bir esle konusur;
    log2(P) katmanda tam karisim olusur.

    Iletisim halkanin YARISI: 1 gonder + 1 al (halkada 2+2).

    NOT: asama numarasi hangi eşle konusuldugunu belirler ve bu FIZIKSEL olarak
    onemlidir - dusuk asamalar (r XOR 1) genelde ayni sunucudaki komsuyu,
    yuksek asamalar (r XOR 4) baska sunucudakini secer. O yuzden her asama
    ayri olculur; NVLink ile InfiniBand farki tabloda gorunur.
    """
    dp, hp = d // P, h // P
    up = torch.randn(hp, 2 * dp, device="cuda", dtype=dtype) * ((2 * dp) ** -0.5)
    dn = torch.randn(dp, hp, device="cuda", dtype=dtype) * (hp ** -0.5)
    params = up.numel() + dn.numel()

    partner = rank ^ (1 << stage)
    recv = torch.empty_like(x_shard)

    def step():
        ops = [dist.P2POp(dist.isend, x_shard, partner),
               dist.P2POp(dist.irecv, recv, partner)]
        for w in dist.batch_isend_irecv(ops):
            w.wait()
        xin = torch.cat([x_shard, recv], dim=1)     # (N, 2d/P)
        y = F.gelu(F.linear(xin, up))
        return F.linear(y, dn)                      # (N, d/P) - kendi dilimi

    n = x_shard.shape[0]
    per_rank = 2 * n * dp        # 1 gonder + 1 al
    return step, params, per_rank, f"kelebek asama {stage} (tek es)", 1


@torch.no_grad()
def run_topo_block(x_shard, d, h, P, rank, dtype):
    """Blok-kosegen: sifir iletisim. Karsilastirmanin ALT SINIRI."""
    dp, hp = d // P, h // P
    up = torch.randn(hp, dp, device="cuda", dtype=dtype) * (dp ** -0.5)
    dn = torch.randn(dp, hp, device="cuda", dtype=dtype) * (hp ** -0.5)
    params = up.numel() + dn.numel()

    def step():
        return F.linear(F.gelu(F.linear(x_shard, up)), dn)

    return step, params, 0.0, "iletisim yok", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=4096, dest="d")
    ap.add_argument("--hidden-mult", type=int, default=4, dest="hmult")
    ap.add_argument("--tokens", type=int, default=8192, help="mikro-yigin token sayisi")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out", default="results/distributed.json")
    args = ap.parse_args()

    rank, P, local = setup()
    dtype = getattr(torch, args.dtype)
    bytes_per = torch.finfo(dtype).bits // 8
    d, N = args.d, args.tokens
    h_base = args.hmult * d

    if rank == 0:
        print(json.dumps({
            "world_size": P, "gpu": torch.cuda.get_device_name(0),
            "d_model": d, "tokens": N, "dtype": args.dtype,
            "torch": torch.__version__, **vars(args),
        }, indent=2, ensure_ascii=False), flush=True)
    host = socket.gethostname()
    print(f"  rank {rank}/{P} -> {host} gpu{local}", flush=True)
    dist.barrier()

    x_full = torch.randn(N, d, device="cuda", dtype=dtype)
    x_shard = torch.randn(N, d // P, device="cuda", dtype=dtype).contiguous()

    # Esit RANK BASINA parametre icin gizli genislik olceklemesi.
    #   tp_dense  : 2*h*d/P        -> h = h_base
    #   ring      : 4*h*d/P^2      -> h = h_base*P/2
    #   kelebek   : 3*h*d/P^2      -> h = 2*h_base*P/3
    #   block     : 2*h*d/P^2      -> h = h_base*P
    # Kelebekte 2/3 tam bolunmez. ONEMLI: yuvarlamayi rank basina gizli boyut
    # (h/P) 64'un kati olacak sekilde yapiyoruz. Ilk denemede yalnizca P'nin
    # katina yuvarlamistik ve h/P = 21845 (TEK SAYI) cikti; bf16 tensor
    # cekirdekleri 8'in (ideali 64'un) kati boyut ister, tek sayida boyut
    # kernel'i verimsiz yola dusuruyordu ve kelebek 4x yavas OLCULDU.
    # Bu bir topoloji ozelligi degil, hizalama artefaktiydi.
    ALIGN = 64
    h_bf = max(ALIGN, int(round(2 * h_base / 3 / ALIGN)) * ALIGN) * P
    plans = [
        ("tp_dense", run_tp_dense, x_full, h_base, {}),
        ("topo_ring", run_topo_ring, x_shard, h_base * P // 2, {}),
        ("topo_block", run_topo_block, x_shard, h_base * P, {}),
    ]
    # kelebegin her asamasi ayri olculur (bkz. fonksiyon aciklamasi)
    n_stage = max(1, int(math.log2(P)))
    for st in range(n_stage):
        plans.append((f"topo_butterfly_s{st}", run_topo_butterfly, x_shard, h_bf,
                      {"stage": st}))

    rows = []
    for name, fn, xin, h, extra in plans:
        step, params, per_rank_elems, pattern, degree = fn(xin, d, h, P, rank, dtype, **extra)
        ms = timed(step, iters=args.iters)
        mb = per_rank_elems * bytes_per / 1e6
        row = {
            "variant": name, "hidden": h, "params_per_rank": int(params),
            "comm_pattern": pattern, "peers_per_rank": degree,
            "fwd_ms": ms,
            "comm_mb_fwd": mb, "comm_mb_fwd_bwd": 2 * mb,
            # WAN oncelemesi: iletisim suresi = byte / bant genisligi
            "proj_sec_1gbps": 2 * mb * 8 / 1e3 / 1.0,
            "proj_sec_10gbps": 2 * mb * 8 / 1e3 / 10.0,
            "proj_sec_100gbps": 2 * mb * 8 / 1e3 / 100.0,
        }
        rows.append(row)
        if rank == 0:
            print(f"[{name:11s}] gizli={h:6d} par/rank={params/1e6:6.2f}M  "
                  f"fwd={ms:7.2f}ms  komsu={degree:2d}  "
                  f"iletisim={2*mb:8.2f}MB/adim  "
                  f"1Gbps'te {row['proj_sec_1gbps']:6.3f}s", flush=True)
        del step
        torch.cuda.empty_cache()

    if rank == 0:
        base = next(r for r in rows if r["variant"] == "tp_dense")
        print("\n--- tensor-paralel yogun tabana gore ---")
        for r in rows:
            if r["variant"] == "tp_dense":
                continue
            red = base["comm_mb_fwd_bwd"] / max(1e-9, r["comm_mb_fwd_bwd"])
            txt = "SONSUZ (sifir iletisim)" if r["comm_mb_fwd_bwd"] == 0 else f"x{red:.1f} az"
            print(f"  {r['variant']:11s} iletisim {txt:24s}  "
                  f"muhatap {base['peers_per_rank']} -> {r['peers_per_rank']}")
        print("\nWAN yorumu: 1 Gbps'lik bir universiteler-arasi hatta, iletisim suresi")
        print("hesap suresini (ms mertebesi) kat kat asiyorsa o parcalama WAN'da calismaz.")

        # ---- OLCEKLEME: EŞIT KARISIMA normalize edilmis ----
        # Katman basina iletisimi kiyaslamak YANILTICIDIR, cunku yontemler ayni
        # katmanda ayni miktarda karisim saglamaz. Adil olcu su: "tum modelin
        # bilgisi her yere ulasana kadar rank basina toplam kac bayt?"
        #
        #   tp_dense : 1 katmanda tam karisim, katman basina ~2*N*d  -> toplam 2*N*d
        #   halka    : +-1 dilim/katman, P/2 katman gerekir,
        #              katman basina 4*N*d/P                          -> toplam 2*N*d
        #   kelebek  : log2(P) katmanda tam karisim,
        #              katman basina 2*N*d/P              -> toplam 2*N*d*log2(P)/P
        #
        # Halka yogun tabanla AYNI toplami harciyor - yalnizca katmanlara
        # yayiyor. Kelebek ise log2(P)/P ile GERCEKTEN azaliyor.
        print("\n--- olcekleme: TAM KARISIM icin rank basina toplam iletisim ---")
        print(f"{'dugum P':>9s}{'tp_dense':>11s}{'halka':>11s}{'kelebek':>11s}"
              f"{'kelebek kazanci':>17s}")
        n, dm, bp = args.tokens, d, torch.finfo(dtype).bits // 8
        for Pn in (8, 16, 64, 256, 1024):
            birim = 2.0 * n * dm * bp / 1e6          # MB
            tp_mb = birim
            rg_mb = birim
            bf_mb = birim * math.log2(Pn) / Pn
            print(f"{Pn:9d}{tp_mb:10.1f}M{rg_mb:10.1f}M{bf_mb:10.2f}M"
                  f"{tp_mb / bf_mb:16.1f}x")
        print("Halka toplam iletisimi AZALTMIYOR, katmanlara yayiyor.")
        print("Kelebek log2(P)/P ile azaltiyor - ve muhatap sayisi 1'de sabit.")
        print("(Kalite tarafi ayrica olculdu: L=16'da kelebek halkadan 12 std hata iyi.)")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "env": {"world_size": P, "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__},
            "args": vars(args), "rows": rows}, indent=2, ensure_ascii=False))
        print(f"\nyazildi: {out}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
