"""Ayni matematigi farkli sekilde hesaplayan katman uygulamalari.

Kritik ayrim: bir maske uygulamak PARAMETRE sayisini dusurur, ama HIZ
kazandirmasi kullanilan kernel'e baglidir. Ayni matematigi iki farkli
uygulamayla olcmek, "yavas cikti" sonucunun fikirden mi yoksa kernel
seciminden mi geldigini ayirt etmemizi saglar.

- MaskedLinear : yogun matmul + eleman bazli maske. Parametre sayisi dusuk
                 sayilir ama FLOP ve sure yogun katmanla ayni (hatta biraz
                 daha yavas). Dogruluk/kapasite deneyleri icin dogru arac.
- SparseLinear  : gercek seyrek kernel (torch.sparse). FLOP gercekten dusuk
                 ama seyrek kernel'ler yogun matmul kadar optimize degildir.
                 Hiz iddialarini bununla olcmek gerekir.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _kaiming_std(fan_in):
    return math.sqrt(2.0 / max(1, fan_in))


class MaskedLinear(nn.Module):
    """Yogun agirlik matrisi, sabit ikili maske ile carpilir."""

    def __init__(self, mask, bias=True):
        super().__init__()
        mask = torch.as_tensor(mask, dtype=torch.float32)
        out_features, in_features = mask.shape
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("mask", mask)

        # fan_in'i maskeye gore olcekle: seyrek katmanda her noron daha az
        # girdi aldigi icin standart init varyansi bozar.
        fan_in = mask.sum(dim=1).mean().item()
        w = torch.randn(out_features, in_features) * _kaiming_std(fan_in)
        self.weight = nn.Parameter(w * mask)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @property
    def nnz(self):
        return int(self.mask.sum().item())

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class SparseLinear(nn.Module):
    """Yalnizca sifir olmayan degerleri parametre olarak tutar, seyrek mm kullanir.

    torch.sparse_coo_tensor uzerinden her ileri gecişte yeniden insa edilir;
    autograd degerlere (values) dogru sekilde akar.
    """

    def __init__(self, mask, bias=True):
        super().__init__()
        mask = torch.as_tensor(mask, dtype=torch.float32)
        out_features, in_features = mask.shape
        self.in_features, self.out_features = in_features, out_features

        idx = mask.nonzero(as_tuple=False).t().contiguous()  # (2, nnz)
        self.register_buffer("indices", idx)
        self._nnz = idx.shape[1]

        fan_in = mask.sum(dim=1).mean().item()
        self.values = nn.Parameter(torch.randn(self._nnz) * _kaiming_std(fan_in))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @property
    def nnz(self):
        return self._nnz

    def forward(self, x):
        w = torch.sparse_coo_tensor(
            self.indices,
            self.values,
            (self.out_features, self.in_features),
            device=x.device,
            dtype=x.dtype,
        )
        # sparse.mm(W: (out,in), xT: (in,B)) -> (out,B)
        flat = x.reshape(-1, self.in_features)
        y = torch.sparse.mm(w, flat.t()).t()
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], self.out_features)


IMPLS = {"masked": MaskedLinear, "sparse": SparseLinear}


def make_layer(impl, mask, bias=True):
    if impl == "dense":
        # gercek nn.Linear: maskesiz referans
        out_features, in_features = mask.shape
        layer = nn.Linear(in_features, out_features, bias=bias)
        nn.init.normal_(layer.weight, std=_kaiming_std(in_features))
        if bias:
            nn.init.zeros_(layer.bias)
        return layer
    if impl not in IMPLS:
        raise ValueError(f"bilinmeyen impl: {impl}. secenekler: dense, masked, sparse")
    return IMPLS[impl](mask, bias=bias)


def count_effective_params(module):
    """Maskelenen sifirlari saymayan gercek parametre sayisi."""
    total = 0
    for m in module.modules():
        if isinstance(m, MaskedLinear):
            total += m.nnz + (m.out_features if m.bias is not None else 0)
        elif isinstance(m, (SparseLinear, BlockDiagLinear, ButterflyLinear)):
            total += m.nnz + (m.out_features if m.bias is not None else 0)
        elif isinstance(m, nn.Linear):
            total += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
    return total


class BlockDiagLinear(nn.Module):
    """Blok-kosegen katman - maske YOK, gercekten kucuk matrisler.

    Neden ayri bir sinif: MaskedLinear seyrekligi taklit eder (yogun matmul +
    maske), yani FLOP'u dusurmez. Blok-kosegen ise g adet bagimsiz kucuk matmul
    demektir ve torch.bmm'ye birebir eslenir - bu yuzden TEK gercek duvar saati
    kazanci veren topolojidir. Kalite karsilastirmasinda 'block' maskesiyle
    matematiksel olarak ozdestir, sadece hizli hesaplanir.
    """

    def __init__(self, out_features, in_features, groups, bias=True):
        super().__init__()
        if in_features % groups or out_features % groups:
            raise ValueError(f"grup sayisi bolmuyor: {in_features}/{out_features} % {groups}")
        self.in_features, self.out_features, self.groups = in_features, out_features, groups
        self.gi, self.go = in_features // groups, out_features // groups
        w = torch.randn(groups, self.go, self.gi) * _kaiming_std(self.gi)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @property
    def nnz(self):
        return self.groups * self.go * self.gi

    def forward(self, x):
        lead = x.shape[:-1]
        # (N, g, gi) -> (g, N, gi) -> bmm -> (g, N, go) -> (N, g, go)
        xg = x.reshape(-1, self.groups, self.gi).transpose(0, 1)
        y = torch.bmm(xg, self.weight.transpose(1, 2)).transpose(0, 1)
        y = y.reshape(*lead, self.out_features)
        return y + self.bias if self.bias is not None else y


class ButterflyLinear(nn.Module):
    """Kelebek deseni - maske YOK, gercek kucuk matmul'lar (bmm).

    NEDEN GEREKLI
    MaskedLinear seyrekligi TAKLIT eder: tam yogun bir agirlik tutar, maskeyle
    carpar, yogun matmul calistirir. Sifirlar da hesaplanip atilir - yogunluk
    1/4'te 4 kat bosa is. Kalite deneyleri icin sorun degildi (adil karsilastirma
    korunuyor), ama hiz iddiasi kurulamiyor ve olceklenirken butcenin cogunu
    sifirlar yiyor.

    KELEBEK BUNA ELVERISLI
    Cikti dilimi r yalnizca {r, r XOR bit} girdi dilimlerini okur - yani tam 2
    blok. Dolayisiyla P tane (out/P, 2*in/P) boyutunda kucuk yogun matmul olarak
    yazilabilir ve torch.bmm ile tek cagrida calisir.

      MaskedLinear : N * out * in       carpma
      ButterflyLinear: N * out * (2*in/P) carpma   ->  P/2 kat az

    Dilim sayisi k'den turetilir (satir basina baglanti): 2*(in/P) = k => P = 2*in/k.
    """

    def __init__(self, out_features, in_features, k, stage=0, bias=False):
        super().__init__()
        half = max(1, int(k) // 2)
        P = in_features // half
        if P < 2 or P & (P - 1):
            raise ValueError(f"kelebek icin dilim sayisi ikinin kuvveti olmali, P={P}")
        if out_features % P or in_features % P:
            raise ValueError(f"{out_features}/{in_features} dilim sayisina ({P}) bolunmuyor")
        self.in_features, self.out_features, self.P = in_features, out_features, P
        self.gi, self.go = in_features // P, out_features // P
        n_stage = max(1, int(math.log2(P)))
        bit = 1 << (int(stage) % n_stage)
        self.register_buffer("perm", torch.tensor([r ^ bit for r in range(P)],
                                                  dtype=torch.long))
        w = torch.randn(P, self.go, 2 * self.gi) * _kaiming_std(2 * self.gi)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @property
    def nnz(self):
        return self.P * self.go * 2 * self.gi

    def forward(self, x):
        lead = x.shape[:-1]
        xr = x.reshape(-1, self.P, self.gi)                     # (N, P, in/P)
        # her dilim: kendisi + XOR esi
        xin = torch.cat([xr, xr[:, self.perm]], dim=-1)         # (N, P, 2*in/P)
        y = torch.bmm(xin.transpose(0, 1),
                      self.weight.transpose(1, 2))              # (P, N, out/P)
        y = y.transpose(0, 1).reshape(*lead, self.out_features)
        return y + self.bias if self.bias is not None else y
