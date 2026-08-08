#!/usr/bin/env python
"""FFN yiginini 8 GPU'da GERCEKTEN dagitik egitir - ileri VE geri gecisle.

NEDEN BU DENEY
--------------
Simdiye kadar iki sey ayri olculdu:
  - KALITE : tek GPU'da, maskelerle taklit edilerek
  - ILETISIM: 8 GPU'da, ama egitim olmadan
Ikisini birbirine baglayan sey bir ARGUMANDI, olcum degil. Bu betik ikisini
ayni nesne uzerinde olcer: parcalanmis model gercekten egitilir.

NEDEN TAM TRANSFORMER DEGIL
---------------------------
Kelebegin kazanci her rank'in verinin yalnizca kendi dilimini tutmasindan
gelir. Dikkat katmani ise her adimda TUM vektore ihtiyac duyar; dilimleri
birlestirmek herkesle konusmak demektir ve "tek muhatap" ozelligi yok olur.
Dikkati de topolojik parcalamak ayri bir arastirma sorusudur (ustelik dikkat
zaten kafalar boyunca BLOK-KOSEGENdir ve blok en kotu karistiran desendir).
Bu yuzden burada dikkat yok: sadece FFN yigini. Iddianin konustugu nesne bu.

GOREV
-----
text8 uzerinde n-gram karakter tahmini: onceki C karakterden sonrakini tahmin
et. Dikkat gerektirmez ama gercek veridir ve bpc olarak raporlanir.

VERI YERLESIMI
--------------
  topo varyantlari : artik akis (residual stream) DILIMLENMIS durur.
                     Her rank d/P boyutu tutar. Kelebegin kazandigi yer burasi.
  dense_tp         : artik akis KOPYALI durur (Megatron'un yaptigi).
                     FFN sutun-paralel + satir-paralel, sonunda all-reduce.
Her yontem kendi dogal yerlesimiyle kosar; rank basina parametre esitlenir.

Calistirma (SLURM, 8 GPU):
  srun ... python train_dist_ffn.py --ffn butterfly --layers 8 --steps 3000
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
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# dagitik kurulum
# --------------------------------------------------------------------------
def setup():
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    local = int(os.environ.get("SLURM_LOCALID", 0))
    os.environ.setdefault("MASTER_PORT", "29518")
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    return rank, world, local


# --------------------------------------------------------------------------
# autograd-uyumlu iletisim ilkelleri
#
# Bu betigin teknik cekirdegi burasi. torch.distributed'in send/recv'i
# autograd bilmez; geri gecisin gradyani ters yone TASIYAN bir Function
# yazmak gerekir. Yoksa model egitilemez - gradyan rank sinirinda kaybolur.
# --------------------------------------------------------------------------
class _ExchangeMulti(torch.autograd.Function):
    """TUM eslerle TEK cagrida degis-tokus. Geri gecis simetrik.

    NEDEN TEK CAGRI - KILITLENME (deadlock) TUZAGI:
    Her es icin ayri bir degis-tokus yapilirsa halka varyanti kilitlenir.
    Rank r once soluyla (r-1) konusmak ister; ama r-1 de once KENDI soluyla
    (r-2) konusmak ister ve r'nin gonderdigini almaya hazir degildir. Bekleme
    zinciri halka boyunca dolasip r'ye geri doner -> herkes bekler, hicbir sey
    ilerlemez. Butun gonder/al islemlerini tek batch_isend_irecv'e koymak bunu
    onler: NCCL hepsini birlikte eslestirir.

    Ileri : recv_i = es_i'nin x'i
    Geri  : d(kayip)/d(recv_i) es_i'nin x'ine aittir -> ona gonderilir;
            karsiliginda es_i'nin BIZIM x'imiz icin hesapladigi gradyan alinir.
            Butun esler ayni x'i kullandigi icin gelen gradyanlar TOPLANIR.
    """

    @staticmethod
    def forward(ctx, x, *partners):
        ctx.partners = partners
        xc = x.contiguous()
        recvs = [torch.empty_like(x) for _ in partners]
        ops = []
        for pr, rv in zip(partners, recvs):
            ops.append(dist.P2POp(dist.isend, xc, pr))
            ops.append(dist.P2POp(dist.irecv, rv, pr))
        for w in dist.batch_isend_irecv(ops):
            w.wait()
        return tuple(recvs)

    @staticmethod
    def backward(ctx, *grads):
        gbacks = [torch.empty_like(g) for g in grads]
        ops = []
        for pr, g, gb in zip(ctx.partners, grads, gbacks):
            ops.append(dist.P2POp(dist.isend, g.contiguous(), pr))
            ops.append(dist.P2POp(dist.irecv, gb, pr))
        for w in dist.batch_isend_irecv(ops):
            w.wait()
        total = gbacks[0]
        for gb in gbacks[1:]:
            total = total + gb
        return (total,) + (None,) * len(ctx.partners)


def exchange(x, partners):
    return _ExchangeMulti.apply(x, *partners)


class _AllReduce(torch.autograd.Function):
    """Ileri gecisde toplar, geri gecisde aynen gecirir (Megatron 'g' operatoru)."""

    @staticmethod
    def forward(ctx, x):
        y = x.clone()
        dist.all_reduce(y)
        return y

    @staticmethod
    def backward(ctx, g):
        return g


def all_reduce_diff(x):
    return _AllReduce.apply(x)


class _AllGather(torch.autograd.Function):
    """Dilimleri birlestirip tam vektor uretir. Geri gecis kendi dilimini alir.

    HATA NOTU: ilk surumde duz dist.all_gather kullaniyordum. O fonksiyon
    autograd bilmez - dense_tp'nin gomme katmanina gradyan AKMIYORDU, yani
    taban model eksik egitiliyordu. Bir tabanin sakat olmasi butun
    karsilastirmayi gecersiz kilar.
    Ileri : (B, d/P) -> (B, d)
    Geri  : (B, d) -> kendi diliminin gradyani (gather'in tersi scatter'dir)
    """

    @staticmethod
    def forward(ctx, x, P, rank):
        ctx.rank, ctx.dp = rank, x.shape[-1]
        out = [torch.empty_like(x) for _ in range(P)]
        dist.all_gather(out, x.contiguous())
        return torch.cat(out, dim=-1)

    @staticmethod
    def backward(ctx, g):
        d = ctx.dp
        return g[..., ctx.rank * d:(ctx.rank + 1) * d].contiguous(), None, None


def all_gather_diff(x, P, rank):
    return _AllGather.apply(x, P, rank)


# --------------------------------------------------------------------------
# dilimlenmis LayerNorm
# --------------------------------------------------------------------------
class ShardedLayerNorm(nn.Module):
    """Artik akis dilimliyken normalizasyon.

    Ortalama ve varyans TUM d boyutu uzerinden gerekir, ama her rank yalnizca
    d/P tutar. Cozum: yerel toplam ve kare toplamini all-reduce etmek - token
    basina yalnizca 2 sayi. Aktivasyonun kendisini tasimaya kiyasla ihmal
    edilebilir (d=512, P=8 icin 2 sayiya karsi 64 sayi).
    """

    def __init__(self, d_shard, d_full, eps=1e-5):
        super().__init__()
        self.d_full, self.eps = d_full, eps
        self.weight = nn.Parameter(torch.ones(d_shard))
        self.bias = nn.Parameter(torch.zeros(d_shard))

    def forward(self, x):
        # IKI GECISLI hesap. Tek gecisli E[x^2]-E[x]^2 formulu kolayca
        # negatif varyans uretir (buyuk sayilarda basamak kaybi) ve rsqrt
        # NaN dondurur. Ekstra kolektif token basina 1 sayi - ihmal edilebilir.
        mean = all_reduce_diff(x.sum(-1, keepdim=True)) / self.d_full
        xc = x - mean
        var = all_reduce_diff((xc * xc).sum(-1, keepdim=True)) / self.d_full
        return xc * torch.rsqrt(var + self.eps) * self.weight + self.bias


# --------------------------------------------------------------------------
# FFN varyantlari - hepsi RANK BASINA esit parametre
# --------------------------------------------------------------------------
class FFNDenseTP(nn.Module):
    """Megatron: sutun-paralel up, satir-paralel down, sonunda all-reduce.
    Artik akis KOPYALI. Muhatap P-1."""

    def __init__(self, d, h, P, rank):
        super().__init__()
        self.up = nn.Linear(d, h // P, bias=False)
        self.down = nn.Linear(h // P, d, bias=False)
        self.peers, self.elems = P - 1, None

    def forward(self, x):
        return all_reduce_diff(self.down(F.gelu(self.up(x))))


class FFNShardedTopo(nn.Module):
    """Artik akis DILIMLENMIS. Her rank kendi dilimini + es(ler)inkini okur.

    kind='ring'      : sol ve sag komsu (2 muhatap), her katmanda AYNI
    kind='butterfly' : r XOR 2^asama (1 muhatap), her katmanda FARKLI
    kind='block'     : hic es yok (0 muhatap) - alt sinir
    """

    def __init__(self, d_shard, h_shard, P, rank, kind, layer_idx):
        super().__init__()
        self.kind, self.P, self.rank = kind, P, rank
        if kind == "ring":
            if P < 3:
                raise ValueError("halka en az 3 rank ister (P<3'te sol=sag, "
                                 "ayni esle iki gonderim eslesmesi belirsizlesir)")
            self.partners = [(rank - 1) % P, (rank + 1) % P]
        elif kind == "butterfly":
            self.partners = [rank ^ (1 << (layer_idx % max(1, int(math.log2(P)))))]
        elif kind == "block":
            self.partners = []
        else:
            raise ValueError(kind)
        fan = d_shard * (len(self.partners) + 1)
        self.up = nn.Linear(fan, h_shard, bias=False)
        self.down = nn.Linear(h_shard, d_shard, bias=False)
        self.peers = len(self.partners)

    def forward(self, x):
        if self.partners:
            xin = torch.cat([x] + list(exchange(x, self.partners)), dim=-1)
        else:
            xin = x
        return self.down(F.gelu(self.up(xin)))


class Model(nn.Module):
    """Konum-basina gomme -> L adet FFN blogu -> dilimlenmis cikis basi.

    Gomme: her rank baglamin C/P konumunu sahiplenir, kendi d/P dilimini
    iletisimsiz uretir. Cikis basi sutun-parcali; adim basina TEK all-reduce
    (katman basina degil) ve bu tum varyantlarda ayni, yani karsilastirmayi
    bozmaz.
    """

    def __init__(self, vocab, ctx, d, layers, ffn_mult, kind, P, rank):
        super().__init__()
        assert d % P == 0 and ctx % P == 0
        self.P, self.rank, self.kind = P, rank, kind
        self.d, self.dp = d, d // P
        self.replicated = (kind == "dense_tp")
        self.my_pos = slice(rank * (ctx // P), (rank + 1) * (ctx // P))
        e = self.dp // (ctx // P)
        self.emb = nn.Parameter(torch.randn(ctx // P, vocab, e) * 0.02)

        h = ffn_mult * d
        blocks, norms = [], []
        for i in range(layers):
            if self.replicated:
                blocks.append(FFNDenseTP(d, h, P, rank))
                norms.append(nn.LayerNorm(d))
            else:
                # RANK BASINA PARAMETRE ESITLIGI
                #   dense_tp : up (h/P, d) + down (d, h/P)      = 2*h*d/P
                #   topo     : up (h_s, (es+1)*d/P) + down (d/P, h_s)
                #              = h_s * (d/P) * (es+2)
                #   esitle -> h_s = 2*h / (es + 2)
                # 32'nin katina hizalanir (bf16 tensor cekirdegi icin sart;
                # bench_distributed'da hizalamayi atlayinca 4x yavaslik olculmustu).
                es = 2 if kind == "ring" else 1 if kind == "butterfly" else 0
                h_s = max(32, int(round(2 * h / (es + 2) / 32)) * 32)
                blocks.append(FFNShardedTopo(self.dp, h_s, P, rank, kind, i))
                norms.append(ShardedLayerNorm(self.dp, d))
        self.blocks = nn.ModuleList(blocks)
        self.norms = nn.ModuleList(norms)
        # Cikis basi TUM varyantlarda ayni: sutun-parcali (girdi d/P) + tek
        # all-reduce. Boylece bas parametreleri ve son iletisim esitlenir,
        # karsilastirma yalnizca FFN'e kalir.
        self.ln_f = (nn.LayerNorm(d) if self.replicated
                     else ShardedLayerNorm(self.dp, d))
        self.head = nn.Linear(self.dp, vocab, bias=False)
        nn.init.normal_(self.head.weight, std=0.02)
        for b in self.blocks:
            nn.init.normal_(b.up.weight, std=math.sqrt(2.0 / b.up.in_features))
            nn.init.normal_(b.down.weight, std=0.02 / math.sqrt(2 * layers))
        self.peers = self.blocks[0].peers

    def forward(self, idx, target):
        # (B, C) -> her rank kendi konumlarindan kendi d/P dilimini uretir
        loc = idx[:, self.my_pos]                                   # (B, C/P)
        parts = [self.emb[j][loc[:, j]] for j in range(loc.shape[1])]
        x = torch.cat(parts, dim=-1)                                # (B, d/P)
        if self.replicated:
            # dense_tp kopyali calisir: dilimleri birlestir (adim basina 1 kez)
            x = all_gather_diff(x, self.P, self.rank)               # (B, d)
        for blk, nrm in zip(self.blocks, self.norms):
            x = x + blk(nrm(x))
        x = self.ln_f(x)
        if self.replicated:
            x = x[:, self.rank * self.dp:(self.rank + 1) * self.dp]
        logits = all_reduce_diff(self.head(x))                      # adim basina 1
        return F.cross_entropy(logits, target)


def get_batch(data, ctx, batch, device, gen):
    ix = torch.randint(len(data) - ctx - 1, (batch,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + ctx].astype(np.int64)) for i in ix])
    y = torch.from_numpy(np.array([data[i + ctx] for i in ix], dtype=np.int64))
    return x.to(device), y.to(device)


def comm_bytes(kind, batch, d, P, layers, bp=4):
    """Adim basina rank basina ILERI gecis baytlari (geri gecis ayni hacmi tekrar eder)."""
    dp = d // P
    if kind == "dense_tp":
        return layers * 2 * (P - 1) / P * batch * d * bp        # katman basina all-reduce
    per = {"ring": 4, "butterfly": 2, "block": 0}[kind]
    ln = 2 * (P - 1) / P * batch * 2 * bp                        # LayerNorm istatistigi
    return layers * (per * batch * dp * bp + ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ffn", default="butterfly",
                    choices=["dense_tp", "ring", "butterfly", "block"])
    ap.add_argument("--d-model", type=int, default=512, dest="d")
    ap.add_argument("--ctx", type=int, default=16)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ffn-mult", type=int, default=4, dest="ffn_mult")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--warmup-steps", type=int, default=10, dest="warmup_steps",
                    help="sureye dahil edilmeyen isinma adimi (NCCL kurulumu icin)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/dist_train.json")
    args = ap.parse_args()

    rank, P, local = setup()
    torch.manual_seed(args.seed)
    dev = "cuda"

    d = Path(args.data)
    meta = json.loads((d / "meta.json").read_text())
    train = np.memmap(d / "train.bin", dtype=np.uint8, mode="r")
    val = np.memmap(d / "val.bin", dtype=np.uint8, mode="r")

    model = Model(meta["vocab_size"], args.ctx, args.d, args.layers,
                  args.ffn_mult, args.ffn, P, rank).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    tot = torch.tensor([n_par], device=dev)
    dist.all_reduce(tot)

    if rank == 0:
        print(json.dumps({"world": P, "gpu": torch.cuda.get_device_name(0),
                          "par_rank": n_par, "par_toplam": int(tot.item()),
                          "muhatap": model.peers, **vars(args)},
                         indent=2, ensure_ascii=False), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    gen = torch.Generator().manual_seed(args.seed)

    # ISINMA - sureye DAHIL DEGIL.
    # NCCL iletisimcisini ilk kolektifte tembel kurar ve bu maliyet ilk
    # adimlara duser. all_reduce kurulumu P2P'den pahali oldugu icin
    # dense_tp orantisiz cezalanir. Duman testinde dense_tp 0.71 dk, digerleri
    # 0.08-0.14 dk cikmisti; fark iletisim hacmiyle aciklanamayacak kadar
    # buyuktu. Isinma adimlari bu yanliligi kaldirir.
    for _ in range(args.warmup_steps):
        x, y = get_batch(train, args.ctx, args.batch, dev, gen)
        loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    opt.zero_grad(set_to_none=True)

    curve = []
    dist.barrier(); torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / max(1, args.warmup))
        lr *= 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train, args.ctx, args.batch, dev, gen)
        loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not math.isfinite(loss.item()) or loss.item() > 20.0:
            if rank == 0:
                print(f"IRAKSAMA adim {step}: kayip={loss.item()}. "
                      f"lr dusur (--lr) veya adim sayisini kontrol et.", flush=True)
            break
        if step % max(1, args.steps // 15) == 0 or step == args.steps:
            torch.cuda.synchronize()
            el = time.perf_counter() - t0
            if rank == 0:
                print(f"adim {step:5d}/{args.steps}  kayip={loss.item():.4f}  "
                      f"bpc={loss.item()/math.log(2):.4f}  {el/60:5.2f} dk", flush=True)
            curve.append({"step": step, "loss": loss.item(), "sec": el})

    torch.cuda.synchronize()
    total_s = time.perf_counter() - t0

    # dogrulama - tum ranklar ayni ornekleri gorur (sabit tohum)
    model.eval()
    vg = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        tot_loss = 0.0
        for _ in range(50):
            x, y = get_batch(val, args.ctx, args.batch, dev, vg)
            tot_loss += model(x, y).item()
    vbpc = tot_loss / 50 / math.log(2)

    if rank == 0:
        mb = comm_bytes(args.ffn, args.batch, args.d, P, args.layers) / 1e6
        res = {"env": {"world": P, "gpu": torch.cuda.get_device_name(0)},
               "args": vars(args), "curve": curve,
               "val_bpc": vbpc, "train_seconds": total_s,
               "peers": model.peers, "params_per_rank": n_par,
               "params_total": int(tot.item()),
               "comm_mb_fwd_per_step": mb, "comm_mb_fwd_bwd_per_step": 2 * mb}
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\nSONUC {args.ffn:10s} L={args.layers:2d}  val_bpc={vbpc:.4f}  "
              f"sure={total_s/60:.2f}dk  muhatap={model.peers}  "
              f"iletisim={2*mb:.1f}MB/adim  par/rank={n_par:,}  -> {out}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
