"""Topoloji uygulanan gizli katmanlardan olusan MLP.

Giris ve cikis projeksiyonlari kasitli olarak yogundur (seyrek ag
literaturunde standart pratik): topolojinin etkisini yalnizca gizli
katmanlarda izole etmek istiyoruz.
"""

import torch.nn as nn

from .layers import make_layer


class TopoMLP(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, mask, impl="masked"):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.hidden = nn.ModuleList(
            [make_layer(impl, mask, bias=True) for _ in range(depth)]
        )
        self.out = nn.Linear(width, out_dim)
        self.act = nn.GELU()
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.width, self.depth, self.impl = width, depth, impl

    def forward(self, x):
        h = self.act(self.inp(x))
        for layer, norm in zip(self.hidden, self.norms):
            # artik (residual) baglanti: derin seyrek aglarda sinyalin
            # kaybolmamasi icin gerekli, ayrica topolojiler arasi
            # karsilastirmayi optimizasyon zorlugundan arindirir
            h = h + self.act(layer(norm(h)))
        return self.out(h)
