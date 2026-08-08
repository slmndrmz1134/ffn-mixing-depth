"""Topoloji takilabilir FFN'e sahip kucuk GPT (nanoGPT tarzi, sadelestirilmis).

ARASTIRMA SORUSU
----------------
Bir transformer'in parametresinin ~2/3'u FFN'dedir ve FFN iki YOGUN DIKDORTGEN
matristir: d -> 4d ve 4d -> d. "Yogun kare/dikdortgen matris varsayimini
birakmak ise yarar mi?" sorusunun gercek modeldeki testi budur.

ADIL KARSILASTIRMA
------------------
Tek dugme: --ffn-mult M (gizli katman genisligi = M * d).

    dense  : gizli katman 4d,  yogunluk 1      -> nnz = 8d^2
    topo   : gizli katman M*d, yogunluk 4/M    -> nnz = 8d^2   (AYNI)

Yani her varyantta parametre sayisi ve FLOP sayisi ESITTIR; degisen tek sey
agirlik matrisinin SEKLI (dar+yogun mu, genis+seyrek mi) ve seyrekligin
YAPISI (poligon / kucuk-dunya / rastgele / blok).

M = 4  -> yogunluk 1, yani dense ile ozdes (saglama noktasi)
M = 8  -> yogunluk 1/2,  gizli katman 2x genis
M = 16 -> yogunluk 1/4,  gizli katman 4x genis
M = 32 -> yogunluk 1/8,  gizli katman 8x genis

Dikkat (durustluk notu): 'masked' uygulama seyrekligi TAKLIT eder, yogun matmul
calistirir. Yani M buyudukce KALITE karsilastirmasi adil kalir ama DUVAR SAATI
artar. Bu bir kusur degil, katman duzeyi deneyinde bulunan gercegin ta kendisi:
seyreklik teorik FLOP'u dusurur, mevcut GPU kernel'leri bunu paraya cevirmez.
Tek istisna 'block'tur; o bmm'ye eslenir ve gercekten hizlanir.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import topology
from .layers import BlockDiagLinear, ButterflyLinear, MaskedLinear


class CausalSelfAttention(nn.Module):
    """Standart cok kafali dikkat. TUM varyantlarda AYNI tutulur.

    Boylece olculen kalite farki yalnizca FFN topolojisinden gelir; dikkat
    katmani bir karistirici degisken olmaz.
    """

    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.d_head = n_head, d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.d_head).transpose(1, 2)
                   for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class FFN(nn.Module):
    """Tek dugmeli FFN: kind + mult.

    kind='dense' ise klasik d -> 4d -> d.
    Aksi halde d -> M*d -> d, satir basina k baglanti ile (nnz esit).
    """

    def __init__(self, d_model, kind="dense", mult=4, rewire=0.1, seed=0, layer_idx=0,
                 guide=False, fast=False):
        super().__init__()
        # layer_idx yalnizca kelebek icin anlamli: hangi asamada (stage)
        # oldugumuzu belirler.
        # DIKKAT: "diger desenler katmandan katmana ayni kalir" DOGRU DEGIL.
        # GPTMini her katmana farkli 'seed' verir; tohumu tuketen desenler
        # (random, watts_strogatz) her katmanda YENIDEN cizilir. Katman ici
        # sabit topoloji icin GPTMini(fixed_mask=True) kullanin.
        self.layer_idx = layer_idx
        self.kind, self.mult = kind, mult
        h = mult * d_model
        self.d_model, self.hidden = d_model, h
        self.act = nn.GELU()

        if kind == "dense":
            if mult != 4:
                raise ValueError("dense varyanti mult=4 ile kullanilmali")
            self.up = nn.Linear(d_model, h, bias=False)
            self.down = nn.Linear(h, d_model, bias=False)
            self.coverage = 1.0
            return

        if mult < 4:
            raise ValueError("mult >= 4 olmali (yoksa nnz esitlenemez)")
        # nnz esitligi:  h*k_up = 4d^2  ve  d*k_down = 4d^2
        k_up = max(1, round(4 * d_model / mult))     # up:   (h, d) satir basina
        k_down = max(1, min(h, 4 * d_model))         # down: (d, h) satir basina
        self.k_up, self.k_down = k_up, k_down

        if kind == "block":
            if guide:
                self._kur_kilavuz(d_model)
            # maskeye gerek yok: gercek bmm, gercek hiz kazanci
            g = max(1, d_model // k_up)
            self.up = BlockDiagLinear(h, d_model, g, bias=False)
            self.down = BlockDiagLinear(d_model, h, g, bias=False)
            self.coverage = 1.0 / g
            return

        kw = {"stage": layer_idx} if kind == "butterfly" else {"p": rewire}
        if guide:
            self._kur_kilavuz(d_model)

        if kind == "butterfly" and fast:
            # bmm uygulamasi: ayni fonksiyon, P/2 kat az carpma.
            # Kapsama istatistigi icin maske bir kez kurulup atilir.
            um = topology.build_rect(kind, h, d_model, k_up, stage=layer_idx)
            dm = topology.build_rect(kind, d_model, h, k_down, stage=layer_idx)
            self.coverage = topology.ffn_coverage(um, dm)
            del um, dm
            self.up = ButterflyLinear(h, d_model, k_up, stage=layer_idx, bias=False)
            self.down = ButterflyLinear(d_model, h, k_down, stage=layer_idx, bias=False)
            return

        up_mask = topology.build_rect(kind, h, d_model, k_up, seed=seed, **kw)
        dn_mask = topology.build_rect(kind, d_model, h, k_down, seed=seed + 1, **kw)
        self.coverage = topology.ffn_coverage(up_mask, dn_mask)
        self.up = MaskedLinear(up_mask, bias=False)
        self.down = MaskedLinear(dn_mask, bias=False)

    def _kur_kilavuz(self, d_model):
        """SELF-GUIDED TRAINING (Wei ve ark., NeurIPS 2024).

        Yapilandirilmis FFN'ler sifirdan egitilirken takilir: baslangictaki
        simetri kisitlari optimizasyonu kotu bir yola sokar ve model
        yapabilecegini ogrenemez. Cozum, egitimin basinda PARALEL bir yogun
        dal koyup katsayisini yavasca sifira indirmek - bisiklette yan teker.

        alpha=1 iken yogun dal yolu gosterir, alpha=0 olunca tamamen cikar ve
        geriye YALNIZCA yapilandirilmis matrisler kalir. Yani NIHAI model
        parametre olarak degismez; ek kapasite sadece egitim sirasinda,
        gecici olarak kullanilir. Raporda bu acikca belirtilmeli.
        """
        self.guide_up = nn.Linear(d_model, 4 * d_model, bias=False)
        self.guide_down = nn.Linear(4 * d_model, d_model, bias=False)
        self.alpha = 1.0

    def forward(self, x):
        y = self.down(self.act(self.up(x)))
        # alpha=0 oldugunda dal HIC hesaplanmaz - egitimin cogunda bedava
        if getattr(self, "guide_up", None) is not None and self.alpha > 0.0:
            y = y + self.alpha * self.guide_down(self.act(self.guide_up(x)))
        return y


class Block(nn.Module):
    def __init__(self, d_model, n_head, ffn_kind, ffn_mult, rewire, seed, dropout,
                 layer_idx=0, guide=False, fast=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, ffn_kind, ffn_mult, rewire, seed,
                       layer_idx=layer_idx, guide=guide, fast=fast)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPTMini(nn.Module):
    def __init__(self, vocab_size, block_size=512, n_layer=8, n_head=8,
                 d_model=512, ffn_kind="dense", ffn_mult=4, rewire=0.1,
                 seed=0, dropout=0.0, init="gpt", guide=False, fast=False,
                 fixed_mask=False):
        super().__init__()
        self.block_size = block_size
        self.init_mode = init
        self.fixed_mask = fixed_mask
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        # Varsayilan (fixed_mask=False): her katmana farkli tohum. Amac varyans
        # azaltmaydi - tek bir sansli/sanssiz graf tum sonucu belirlemesin diye.
        # YAN ETKISI: tohumu tuketen desenler (random, watts_strogatz) boylece
        # KATMANDAN KATMANA DEGISIR; kelebek de stage=layer_idx ile zaten oyle.
        # Yani "yapisiz rastgele" kontrolu aslinda derinlikle-degisen bir desen
        # ve 'rastgele en iyi' bulgusu 'derinlikle degismek iyidir' bulgusuyla
        # ayrisamaz hale gelir.
        #
        # fixed_mask=True: topoloji TUM katmanlarda ayni (tohum sabit, kelebek
        # stage'i 0'a donuk). Tohumlar ARASINDA hala farkli graf cizilir, yani
        # varyans azaltma korunur; degisen tek sey derinlik eksenidir.
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, ffn_kind, ffn_mult, rewire,
                  seed if fixed_mask else seed + 17 * i, dropout,
                  layer_idx=0 if fixed_mask else i, guide=guide, fast=fast)
            for i in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # agirlik paylasimi

        self._init_all(init)
        # derin artik yiginlarda cikis varyansinin patlamamasi icin cikis
        # projeksiyonlarini kucult. CARPARAK yapiyoruz, yeniden ornekleyerek
        # degil: boylece maskelenmis sifirlar sifir kalir ve secilen init
        # modunun olcegi korunur.
        with torch.no_grad():
            for n, p in self.named_parameters():
                if n.endswith("proj.weight") or n.endswith("down.weight"):
                    p.mul_(1.0 / math.sqrt(2 * n_layer))

    def set_guide_alpha(self, a):
        """Kilavuz dalin katsayisini ayarlar (egitim dongusunden cagrilir)."""
        for b in self.blocks:
            if getattr(b.ffn, "guide_up", None) is not None:
                b.ffn.alpha = float(a)

    @torch.no_grad()
    def _init_all(self, mode):
        """Baslangic agirliklari - ADALET ACISINDAN KRITIK.

        'gpt'     : GPT gelenegi, her katmana sabit std=0.02.
        'kaiming' : varyans koruyan, gercek fan_in'e gore std=sqrt(2/fan_in).
        'mixed'   : yogun katman GPT gelenegi (0.02), seyrek katman kendi
                    fan_in'ine gore kaiming. Her katman tipine kendi
                    literaturundeki init'i verir. Ilk izgarada kazara boyleydi
                    ve seyrek varyantlar icin OLCULEN EN IYI mod oldugu icin
                    artik acik bir secenek: hipotezin en guclu halini test
                    etmek, onu en zayif halinden daha durust bir sinavdir.

        Neden iki mod: seyrek katmanin fan_in'i yogun katmandan cok kucuktur
        (or. 128'e karsi 512). Sabit std kullanilirsa iki grup GELU'ya farkli
        olcekte sinyal sokar; fan_in'e gore olceklenirse olcek esitlenir ama
        bu sefer yogun taban gelenekten sapar. Hicbir tek secim notr degildir,
        bu yuzden sonuc IKI modda da raporlanir. Sonuc modlar arasinda
        degisiyorsa, olculen sey topoloji degil baslangic olceğidir.
        """
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, MaskedLinear):
                fan_in = m.mask.sum(dim=1).mean().item()
                # 'mixed' seyrek katmani kaiming ile baslatir (gpt'den ayrilir)
                std = 0.02 if mode == "gpt" else math.sqrt(2.0 / max(1.0, fan_in))
                m.weight.normal_(0.0, std).mul_(m.mask)
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, ButterflyLinear):
                std = 0.02 if mode == "gpt" else math.sqrt(2.0 / (2 * m.gi))
                m.weight.normal_(0.0, std)
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, BlockDiagLinear):
                std = 0.02 if mode == "gpt" else math.sqrt(2.0 / m.gi)
                m.weight.normal_(0.0, std)
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, nn.Linear):
                # 'mixed' yogun katmani GPT gelenegiyle birakir (kaiming'den ayrilir)
                std = 0.02 if mode in ("gpt", "mixed") else math.sqrt(2.0 / m.in_features)
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        if targets is None:
            return self.head(x[:, [-1], :]), None
        logits = self.head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def param_report(self):
        """Etkin (maskelenmis sifirlari saymayan) parametre dokumu."""
        from .layers import count_effective_params
        ffn = sum(count_effective_params(b.ffn) for b in self.blocks)
        # Kilavuz dal GECICIDIR (alpha 0'a inince cikar). Nihai model onu
        # icermez, o yuzden parametre esitligi raporlanirken DUSULUR;
        # ayrica ayri bir alanda gosterilir ki egitim sirasindaki ek
        # kapasite gizlenmis olmasin.
        guide = sum(m.weight.numel() for b in self.blocks
                    for m in (getattr(b.ffn, "guide_up", None),
                              getattr(b.ffn, "guide_down", None)) if m is not None)
        ffn -= guide
        attn = sum(count_effective_params(b.attn) for b in self.blocks)
        # head.weight, tok_emb.weight ile paylasimli - iki kez sayma
        total = count_effective_params(self) + self.pos_emb.weight.numel() - guide
        return {
            "params_total_effective": int(total),
            "params_ffn": int(ffn),
            "params_guide": int(guide),
            "params_attn": int(attn),
            "ffn_coverage": float(self.blocks[0].ffn.coverage),
            "ffn_hidden": int(self.blocks[0].ffn.hidden),
        }
