"""Poligon / graf topolojilerinden ikili (binary) baglanti maskeleri uretir.

Her maske n x n boyutunda simetrik bir komsuluk matrisidir; mask[i, j] = 1
demek "j'inci dugumden i'inci dugume baglanti var" anlamina gelir. Kosegen
(self-loop) her zaman 1'dir, boylece dugum kendi degerini tasiyabilir.
"""

import math
from collections import deque

import numpy as np


def dense(n, **_):
    """Tam bagli (standart kare agirlik matrisi) - referans."""
    return np.ones((n, n), dtype=np.float32)


def ring(n, k=6, **_):
    """Poligon: n kose, her kose en yakin k komsusuna bagli (circulant).

    k komsu sayisi cift olmalidir; i dugumu i±1 ... i±k/2 (mod n) ile baglanir.
    Bu, duzgun bir n-genin kenarlari (k=2) ve kisa kosegenleri (k>2) demektir.
    """
    if k % 2 != 0:
        raise ValueError("k cift olmali")
    m = np.zeros((n, n), dtype=np.float32)
    idx = np.arange(n)
    for d in range(1, k // 2 + 1):
        m[idx, (idx + d) % n] = 1.0
        m[idx, (idx - d) % n] = 1.0
    np.fill_diagonal(m, 1.0)
    return m


def watts_strogatz(n, k=6, p=0.1, seed=0, **_):
    """Kucuk-dunya: poligon halkasi + p olasilikla yeniden baglanan kenarlar.

    Halka topolojisinin zayif noktasi uzun yol uzunlugudur (bilgi karsi tarafa
    ancak n/k adimda ulasir). Kenarlarin kucuk bir kismini rastgele yeniden
    baglamak, yogunlugu neredeyse hic artirmadan capi dramatik olarak dusurur.
    """
    rng = np.random.default_rng(seed)
    m = ring(n, k)
    np.fill_diagonal(m, 0.0)
    # ust ucgendeki kenarlari dolas, p olasilikla hedefi degistir
    src, dst = np.nonzero(np.triu(m))
    for s, d in zip(src, dst):
        if rng.random() >= p:
            continue
        new_d = int(rng.integers(n))
        if new_d == s or m[s, new_d] > 0:
            continue
        m[s, d] = m[d, s] = 0.0
        m[s, new_d] = m[new_d, s] = 1.0
    np.fill_diagonal(m, 1.0)
    return m


def random_sparse(n, density=None, k=6, seed=0, **_):
    """Ayni yogunlukta rastgele seyrek graf - KONTROL grubu.

    Bu en kritik karsilastirma: poligon yapisi gercekten mi ise yariyor,
    yoksa sadece "seyrek olmak" mi ise yariyor? Rastgele graf ayni parametre
    butcesine sahip oldugu icin farki yapinin kendisine atfedebiliriz.
    """
    rng = np.random.default_rng(seed)
    if density is None:
        density = (k + 1) / n
    target = int(round(density * n * n))
    m = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(m, 1.0)
    remaining = max(0, (target - n) // 2)
    tries = 0
    while remaining > 0 and tries < 50 * target:
        i, j = rng.integers(n), rng.integers(n)
        tries += 1
        if i == j or m[i, j] > 0:
            continue
        m[i, j] = m[j, i] = 1.0
        remaining -= 1
    return m


BUILDERS = {
    "dense": dense,
    "ring": ring,
    "watts_strogatz": watts_strogatz,
    "random": random_sparse,
}


def build(kind, n, **kw):
    if kind not in BUILDERS:
        raise ValueError(f"bilinmeyen topoloji: {kind}. secenekler: {list(BUILDERS)}")
    return BUILDERS[kind](n, **kw)


def _bfs_lengths(adj_list, start):
    n = len(adj_list)
    dist = np.full(n, -1, dtype=np.int32)
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj_list[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def stats(mask, sample=128, seed=0):
    """Maskenin yapisal olculeri.

    avg_path_len ve diameter, topolojinin bilgi karistirma kapasitesinin
    dogrudan gostergesidir - dusuk olan daha iyidir. Buyuk n icin rastgele
    ornekleme ile tahmin edilir.
    """
    n = mask.shape[0]
    a = mask.copy()
    np.fill_diagonal(a, 0.0)
    nnz = int(mask.sum())
    adj_list = [np.nonzero(a[i])[0] for i in range(n)]

    rng = np.random.default_rng(seed)
    sources = range(n) if n <= sample else rng.choice(n, sample, replace=False)
    total, count, diameter, disconnected = 0, 0, 0, False
    for s in sources:
        d = _bfs_lengths(adj_list, int(s))
        reach = d[d > 0]
        if (d < 0).any():
            disconnected = True
        if reach.size:
            total += int(reach.sum())
            count += reach.size
            diameter = max(diameter, int(reach.max()))

    return {
        "n": n,
        "nnz": nnz,
        "density": nnz / (n * n),
        "avg_degree": float(a.sum(axis=1).mean()),
        "avg_path_len": (total / count) if count else float("inf"),
        "diameter": diameter,
        "disconnected": disconnected,
    }


# ---------------------------------------------------------------------------
# DIKDORTGEN maskeler (GPT FFN icin)
#
# Yukaridaki maskeler kare (n x n). Ama bir transformer'in FFN'i dikdortgendir:
# d -> h ve h -> d. Asil arastirma sorusu ("yogun dikdortgen matris sart mi?")
# icin poligon fikrini dikdortgene tasimamiz gerekir.
#
# Tasarim: her CIKTI birimi, girdi ekseninde kendi konumuna karsilik gelen
# noktanin etrafindaki k girdiye baglanir (circulant bant). Boylece satir basina
# tam olarak k baglanti olur ve nnz = out * k ile parametre butcesi net kontrol
# edilir. Tum varyantlar ayni k'yi kullanir => esit parametre, esit FLOP.
# ---------------------------------------------------------------------------


def rect_ring(out_features, in_features, k, **_):
    """Poligon/bant: cikti i, girdi ekseninde kendi konumu etrafindaki k girdiyi gorur.

    Kare haldeki 'ring' ile ayni fikir: yerel, duzenli, kaydirmali (circulant).
    Zayifligi ayni: uzak girdiler ancak cok adimda karisir.
    """
    k = int(min(k, in_features))
    m = np.zeros((out_features, in_features), dtype=np.float32)
    center = (np.arange(out_features) * in_features // max(1, out_features))
    offs = np.arange(-(k // 2), k - k // 2)
    cols = (center[:, None] + offs[None, :]) % in_features
    rows = np.repeat(np.arange(out_features)[:, None], k, axis=1)
    m[rows, cols] = 1.0
    return m


def rect_watts_strogatz(out_features, in_features, k, p=0.1, seed=0, **_):
    """Bant + p olasilikla yeniden baglama. Satir basina k sabit kalir."""
    k = int(min(k, in_features))
    m = rect_ring(out_features, in_features, k)
    if p <= 0:
        return m
    rng = np.random.default_rng(seed)
    for i in range(out_features):
        cols = np.nonzero(m[i])[0]
        for c in cols[rng.random(cols.size) < p]:
            for _ in range(20):
                nc = int(rng.integers(in_features))
                if m[i, nc] == 0.0:
                    m[i, c] = 0.0
                    m[i, nc] = 1.0
                    break
    return m


def rect_random(out_features, in_features, k, seed=0, **_):
    """Satir basina k rastgele girdi - KONTROL grubu (ayni parametre, yapi yok)."""
    k = int(min(k, in_features))
    rng = np.random.default_rng(seed)
    m = np.zeros((out_features, in_features), dtype=np.float32)
    pick = rng.random((out_features, in_features)).argpartition(k - 1, axis=1)[:, :k]
    m[np.arange(out_features)[:, None], pick] = 1.0
    return m


def rect_block(out_features, in_features, k, **_):
    """Blok-kosegen: girdiler g gruba bolunur, her cikti yalnizca kendi grubunu gorur.

    Endustride fiilen kullanilan seyreklik budur (grouped conv, MoE, GQA).
    Ayrica bmm'ye birebir esledigi icin GERCEK hiz kazanci veren tek varyanttir
    (bkz. layers.BlockDiagLinear). Diger topolojiler icin karsilastirma tabani.
    """
    k = int(min(k, in_features))
    g = max(1, in_features // k)
    m = np.zeros((out_features, in_features), dtype=np.float32)
    for gi in range(g):
        r0, r1 = gi * out_features // g, (gi + 1) * out_features // g
        c0, c1 = gi * in_features // g, (gi + 1) * in_features // g
        m[r0:r1, c0:c1] = 1.0
    return m


def rect_mozaik(out_features, in_features, k, **_):
    """Mozaik / satranc tahtasi: satir i, s adim atlayarak (i mod s) ofsetinden baslar.

    Gozle bakildiginda en "adil" desendir - delikler her yere esit dagilmis,
    hicbir bolge bos degil. Sezgi bunun en iyi karistiran desen olmasi
    gerektigini soyler.

    Sezgi YANLISTIR: satir ve sutunlari (i mod s) sinifina gore yeniden
    siralarsaniz bu matris tam olarak blok-kosegene donusur. Kapsamasi da
    ayni: 1/s. Bir sinir aginda gizli birimlerin sirasi keyfi oldugu icin
    (ve artik akisinin tabani baslangicta ayricalikli olmadigi icin) mozaik
    ile blok AYNI modeldir - sadece farkli etiketlenmis.

    Bu yuzden bu desen bir kesif degil, bir SAGLAMA noktasidir: olctugumuz
    seyin gercekten kapsama oldugunu, deseni gozle degerlendirmedigimizi
    dogrular. mozaik ile blok farkli cikarsa kurulumumuzda hata var demektir.
    """
    k = int(min(k, in_features))
    s = max(1, in_features // k)
    m = np.zeros((out_features, in_features), dtype=np.float32)
    for i in range(out_features):
        m[i, (i % s)::s] = 1.0
    return m


def rect_butterfly(out_features, in_features, k, stage=0, **_):
    """Kelebek (butterfly): katman ELL'de dilim r, r XOR 2^ELL ile eslesir.

    HALKANIN HATASI: her katmanda AYNI komsuyla konusur. Erisim katman basina
    sadece +-1 dilim buyur, tam karisim icin P/2 katman gerekir.

    KELEBEGIN FARKI: her katmanda BASKA biriyle eslesir - once 1 uzaktaki,
    sonra 2, sonra 4... log2(P) katmanda herkes herkese ulasir. Katman basina
    muhatap sayisi yine 1'dir; degisen tek sey KIMINLE konusuldugu.

      katman 1: r <-> r XOR 1   -> 2 dilim erisilir
      katman 2: r <-> r XOR 2   -> 4 dilim
      katman 3: r <-> r XOR 4   -> 8 dilim (P=8 icin hepsi)

    FFT, hiperkup all-reduce ve Monarch matrislerinin altindaki yapi budur.

    Dilim sayisi k'den turetilir: her cikti 2 dilim okur, dolayisiyla
    dilim boyu = k/2 ve P = in_features / (k/2). P ikinin kuvveti olmali.
    """
    k = int(min(k, in_features))
    half = max(1, k // 2)
    P = max(2, in_features // half)
    if P & (P - 1):
        raise ValueError(f"kelebek icin dilim sayisi ikinin kuvveti olmali, P={P}")
    n_stage = max(1, int(math.log2(P)))
    bit = 1 << (int(stage) % n_stage)

    m = np.zeros((out_features, in_features), dtype=np.float32)
    osz = out_features // P
    for r in range(P):
        r0, r1 = r * osz, (r + 1) * osz
        for c in (r, r ^ bit):
            m[r0:r1, c * half:(c + 1) * half] = 1.0
    return m


def cumulative_coverage(layer_masks):
    """Katmanlar BOYUNCA birikmis kapsama.

    ffn_coverage tek bir FFN'e bakar. Ama desen katmandan katmana degisiyorsa
    (kelebek gibi) asil onemli olan L katman sonunda toplam ne kadar girdiye
    ulasildigidir. Boolean matris carpimiyla hesaplanir.

    layer_masks: her katman icin (up_mask, down_mask) ciftleri.
    Donen: her katman sonundaki birikmis kapsama listesi.
    """
    out = []
    acc = None
    for up, dn in layer_masks:
        r = (dn @ up) > 0                      # bu katmanin erisim matrisi (d x d)
        acc = r if acc is None else ((acc.astype(np.float32) @ r.astype(np.float32)) > 0)
        out.append(float(acc.mean()))
    return out


RECT_BUILDERS = {
    "ring": rect_ring,
    "butterfly": rect_butterfly,
    "mozaik": rect_mozaik,
    "watts_strogatz": rect_watts_strogatz,
    "random": rect_random,
    "block": rect_block,
}


def build_rect(kind, out_features, in_features, k, **kw):
    if kind not in RECT_BUILDERS:
        raise ValueError(f"bilinmeyen dikdortgen topoloji: {kind}. "
                         f"secenekler: {list(RECT_BUILDERS)}")
    return RECT_BUILDERS[kind](out_features, in_features, k, **kw)


def ffn_coverage(up_mask, down_mask, chunk=1024):
    """FFN'in iki adimda ne kadar girdiyi karistirabildigi (0-1 arasi).

    up: (h, d), down: (d, h). (down @ up) > 0 olan hucreler, bir cikti biriminin
    gizli katman uzerinden ULASABILDIGI girdi birimleridir. Bu, kare graftaki
    'ortalama yol uzunlugu'nun dikdortgen karsiligidir:

      1.0  -> her cikti tum girdileri gorebiliyor (iyi karistirma)
      dusuk-> bilgi lokal kaliyor (bant/blok topolojilerin zayif noktasi)

    Bu olcu, kalite farklarini topolojinin YAPISINA baglamamizi saglar; yoksa
    elimizde yalnizca "su varyant daha iyi cikti" gibi aciklamasiz bir sonuc olur.
    """
    d, h = down_mask.shape
    total = 0.0
    for i in range(0, d, chunk):
        reach = down_mask[i:i + chunk] @ up_mask  # (chunk, d)
        total += float((reach > 0).sum())
    return total / (d * up_mask.shape[1])
