"""Sentetik gorevler - veri seti indirmeye ihtiyac duymaz.

Hesaplama dugumlerinin internet erisimi olmadigi icin (HPC'de tipik durum)
katman duzeyi deneylerinde sentetik gorev kullaniyoruz.

KRITIK TASARIM NOKTASI - iki ayri tohum (seed):
  - func_seed  : ogrenilecek GIZLI KURALI belirler (ogretmen agirliklari /
                 carpim indeksleri). Egitim ve test AYNI func_seed'i kullanmali,
                 yoksa modeli A sinavina calistirip B sinaviyla test etmis
                 olursunuz - test hatasi anlamsiz cikar.
  - sample_seed: sadece GIRDI ORNEKLERINI degistirir. Egitim ve test farkli
                 sample_seed kullanir ki test verisi egitimde gorulmemis olsun.

Normalizasyon istatistikleri (mean/std) buyuk bir referans orneginden BIR KEZ
hesaplanir ve hem egitime hem teste ayni sekilde uygulanir. Boylece "hicbir
sey ogrenmeyen" modelin MSE'si ~1.0 olur ve sonuclar yorumlanabilir kalir.

Iki gorev kasitli olarak farkli seyleri olcer:
  - teacher : rastgele yogun bir ogretmen agini taklit etme. KAPASITE olcer.
  - mixing  : cikti, birbirinden uzak girdi koordinatlarinin carpimina baglidir.
              Bilgi KARISTIRMA kapasitesi olcer - halka topolojisinin uzun yol
              uzunlugu burada cezalandirilir.
"""

import torch
import torch.nn as nn


class Task:
    """Sabit bir gizli fonksiyon + o fonksiyondan ornek uretme.

    Fonksiyon func_seed ile bir kez kurulur; egitim ve test ayni Task
    nesnesinden ornek alarak ayni kurali paylasir.
    """

    def __init__(self, kind, in_dim, out_dim, func_seed=0,
                 n_terms=8, hidden=256, norm_ref=20000):
        self.kind = kind
        self.in_dim = in_dim
        self.out_dim = out_dim
        g = torch.Generator(device="cpu").manual_seed(func_seed)

        if kind == "teacher":
            self._teacher = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, out_dim),
            )
            for p in self._teacher.parameters():
                p.data.copy_(torch.randn(p.shape, generator=g)
                             * (1.0 / max(1, p.shape[-1]) ** 0.5))
            for p in self._teacher.parameters():
                p.requires_grad_(False)
        elif kind == "mixing":
            half = in_dim // 2
            # uzak indeks ciftleri: a alt yaridan, b ust yaridan
            self._a = torch.randint(0, half, (n_terms, out_dim), generator=g)
            self._b = torch.randint(half, in_dim, (n_terms, out_dim), generator=g)
            self._n_terms = n_terms
        else:
            raise ValueError(f"bilinmeyen gorev: {kind}. secenekler: teacher, mixing")

        # normalizasyon istatistiklerini sabit bir referans orneginden hesapla
        xr = torch.randn(norm_ref, in_dim, generator=g)
        yr = self._raw(xr)
        self._mean = yr.mean(0, keepdim=True)
        self._std = yr.std(0, keepdim=True) + 1e-6

    @torch.no_grad()
    def _raw(self, x):
        if self.kind == "teacher":
            return self._teacher(x)
        y = torch.zeros(x.shape[0], self.out_dim)
        for j in range(self._n_terms):
            y += x[:, self._a[j]] * x[:, self._b[j]]
        return y

    @torch.no_grad()
    def sample(self, n_samples, sample_seed, device="cpu"):
        """Ayni gizli kuraldan, verilen tohumla yeni girdi ornekleri uretir."""
        g = torch.Generator(device="cpu").manual_seed(sample_seed)
        x = torch.randn(n_samples, self.in_dim, generator=g)
        y = (self._raw(x) - self._mean) / self._std
        return x.to(device), y.to(device)


def make_task(kind, in_dim, out_dim, func_seed=0):
    return Task(kind, in_dim, out_dim, func_seed=func_seed)
