# Deney Envanteri

`results/` altındaki **310 kaydın tamamı** (324 eğitim + 10 iletişim/gecikme ölçümü; katman ölçeğindeki her kayıt 5 eğitim içerir), aile aile: ne ölçüldü, neden, ne çıktı,
makalenin neresinde. Makalede karşılığı olmayanlar **⚠ MAKALEDE YOK** ile işaretli.

Üretim: `python analiz_tau.py` (kalite), `python gecikme.py` (gecikme tabanı),
`python karar.py` (bant genişliği koridoru).

---

## Deney 1 — Sentetik katman görevleri (30 eğitim = 6 geçerli kayıt × 5 topoloji, A100)

**Betik:** `train_topology.py` · **Dosyalar:** `train_mixing_*.json`, `train_teacher_*.json`
**Kurulum:** genişlik 1024, derinlik 4, derece 16, 3 tohum × 3 iş

**Amaç.** Dil modelinden önce, topolojinin *kendi başına* bir şey ifade edip etmediğini
en yalın ortamda görmek. İki görev: `mixing` (girdileri karıştırmayı gerektirir) ve
`teacher` (rastgele bir öğretmen ağını taklit).

> ### ⚠ ÖNEMLİ: 9 dosyanın 3'ü ESKİ KODDAN, 3'ü de KOPYA
> Dokuz dosyanın `args`'ı **birebir aynı** (aynı tohum, aynı yapılandırma) ama sonuçlar
> 46 kat ayrışıyor:
>
> | İş | halka test_mse | Durum |
> |---|---|---|
> | `1431389` | ~1.95 | **eski/hatalı kod sürümü — KULLANMA** |
> | `1431458` | ~0.042 | geçerli |
> | `1431474` | ~0.042 | `1431458`'in **birebir kopyası** — gereksiz |
>
> Args'lar ayırt etmediği için naif bir analiz üçünü de havuza atar ve varyansı
> ortalamadan büyük gösterir (std/ort = 1.41), her fark "gürültü" çıkar. Aşağıdaki
> tablo **yalnızca `1431458`** iledir.

**Sonuç (temizlenmiş, n=3).**

| Varyant | params | mixing | teacher |
|---|---|---|---|
| dense_full | 4.35M | 0.1247 ± 0.0073 | 0.1343 ± 0.0039 |
| dense_matched (dar) | 89K | 0.3182 ± 0.0239 | **0.1030 ± 0.0022** |
| **ring** | 222K | **0.0402 ± 0.0038** | 0.1321 ± 0.0036 |
| watts_strogatz | 222K | 0.0418 ± 0.0033 | 0.1339 ± 0.0032 |
| random | 222K | 0.0471 ± 0.0047 | 0.1360 ± 0.0031 |

**Yorum — görev bağımlı tersinme, ve güçlü.** `mixing` görevi
`y = Σ x[a]·x[b]` (a alt yarıdan, b üst yarıdan) — tanımı gereği uzak girdilerin
birleştirilmesini zorunlu kılıyor. Orada geniş-seyrek her iki yoğun kontrolü de eziyor:
halka vs dar-yoğun **19.9σ**, vs geniş-yoğun **17.7σ**. `teacher` görevinde (sabit rastgele
MLP taklidi, karıştırma zorunluluğu yok) sıralama **tersine dönüyor**: dar-yoğun kazanıyor
(**12.0σ**).

Yani seyreklikle satın alınan genişlik, **yalnızca görev karıştırma talep ettiğinde**
işe yarıyor. Bu, makalenin tamamının ölçtüğü olgunun en küçük ve en temiz hâli.

✅ **MAKALEYE EKLENDİ** (bu denetimden sonra): Tablo `tab:katman` / `tab:layer`,
Bölüm "Katman ölçeği". Öncesinde özet "üç ölçek" diyordu ama katman ölçeğinin
tablosu yoktu.

---

## Deney 2 — Katman hız kıyaslaması (1 kayıt, A100)

**Betik:** `bench_layer.py` · **Dosya:** `bench_layer_1431381.json`
**Kurulum:** genişlik 4096, yığın 256, dereceler {6, 16, 64, 256}, float32

**Amaç.** Üç uygulamanın (`dense`, `masked`, `sparse`) gerçek hızını ölçmek — "seyrek =
hızlı" varsayımını sınamak.

**Sonuç (ileri geçiş, ms).**

| nnz | dense | masked | sparse |
|---|---|---|---|
| 28.7K | 0.731 | 0.838 | **0.397** |
| 69.6K | 0.731 | 0.838 | 0.780–1.016 |
| 266K | 0.731 | 0.838 | 0.643–1.959 |
| 1.05M | 0.731 | 0.838–0.924 | 1.729–3.183 |

**Yorum.** `masked` her zaman yoğundan yavaş (beklenen: yoğun matmul + maske).
Ama `sparse` (gerçek `torch.sparse`) **en seyrek uçta yoğunu geçiyor** (0.397 vs 0.731,
1.8× hızlı), yoğunluk arttıkça hızla kötüleşiyor.

⚠ **MAKALEDE YOK.** Makale "bmm 1.9× yavaş" diyor ve hız iddiasını doğru şekilde
daraltıyor, ama *gerçek seyrek çekirdeğin bir rejimde yoğunu geçtiği* ölçümü hiç
geçmiyor. Bu, "yapısal seyreklik hızlandırmaz" hükmünü vermemek için ek bir gerekçe.

---

## Deney 3 — Mimari taraması: yoğunluk (≈120 koşu, H200)

**Betik:** `train_gpt.py` · **Izgara:** `slurm/gpt_grid.slurm`, `slurm/gpt_yogunluk_doldur.slurm`
**Kurulum:** GPT-mini d=512, L=8, text8, 6000 adım, M ∈ {4,8,16,32,64}

**Amaç.** Eşit parametre/FLOP altında yedi deseni karşılaştırmak; kaliteyi neyin
belirlediğini bulmak.

**Sonuç.** Yoğunluk 1/2'ye kadar seyreklik bedelsiz (σ≤1.2). 1/4'te yoğun net kazanıyor
(+0.022 bpc, 16σ). **Aynı kapsamada farklı kalite** karşıörneği üç yoğunlukta birden:
M=16'da kelebek–blok 9.6σ, M=32'de 7.1σ, M=64'te 9.0σ.

✅ Makalede: Tablo `tab:kalite`, Şekil `fig:yogunluk`.

---

## Deney 4 — Karışma derinliği τ (yeni koşu yok, mevcut veriden)

**Betik:** `analiz_tau.py` (`polytopo/topology.py::cumulative_coverage`)

**Amaç.** Katman başına kapsamanın açıklayamadığını açıklamak. τ = her çıktının her
girdiye ulaşması için gereken katman sayısı; maskelerden, eğitim yapmadan hesaplanır.

**Sonuç.** 4 yoğunlukta 41 ikili karşılaştırma: **35 doğru, 9 ayırt edilemez, 1 ihlal.**
Sıralama hiçbir yerde ters dönmüyor. Tek kusur ayırt edememe (M=64, random vs
küçük-dünya, aynı τ ama 8.2σ).

✅ Makalede: Tablo `tab:tau`, Şekil `fig:tau`, Bölüm "Kapsama nerede yetmiyor".

---

## Deney 5 — Derinlik taraması (36 koşu, H200)

**Izgara:** `slurm/gpt_derinlik.slurm` · **Kurulum:** M=16, L ∈ {2,4,8,16}, tohum 10–12

**Amaç.** Sabit desenin derinlikten faydalanıp faydalanmadığını ölçmek.

**Sonuç.** Halka–rastgele farkı tek yönlü büyümüyor: 2.8σ → **0.3σ** → 4.3σ → 23.3σ.
L=4'te açık kapanıyor çünkü halka tam orada karışmasını bitiriyor (τ=3).

✅ Makalede: Tablo `tab:derinlik`, Şekil `fig:derinlik`, üç rejim anlatımı.

---

## Deney 6 — Sabit-maske kontrolü (57 koşu, H200)

**Izgara:** `slurm/gpt_sabit_maske.slurm` · **Bayrak:** `train_gpt.py --fixed-mask`

**Amaç.** "Rastgele en iyi" bulgusunu "derinlikle değişmek iyidir" bulgusundan ayırmak.
Kendi kontrol grubumuz (rastgele) her katmanda yeniden çiziliyordu.

**Öngörüler koşulardan ÖNCE betiğin başlığına yazıldı.**

**Sonuç.** Boş kontrol geçti (halka 0.49σ). Rastgele değişmedi (0.88 / 0.48 / 1.26σ) →
1. bulgu kapsama etkisi. Kelebek çöktü ve **bloğun 0.14σ yakınına indi** → τ'nun nokta
tahmini tuttu.

✅ Makalede: Tablo `tab:sabit`, Bölüm "Sabit-maske kontrolü".

---

## Deney 7 — Başlangıç ölçeği kontrolü (36 koşu, H200)

**Izgara:** `slurm/gpt_init_control.slurm` · **Rejimler:** `gpt`, `kaiming`, `mixed`

**Amaç.** Seyrek katmanların GELU'ya farklı ölçekte sinyal sokması (çıkış std 1.41 vs
0.45) sonuçları bozuyor mu?

**Sonuç.** Kapsama sıralaması üç rejimde de korundu → sonuçlar başlangıç seçiminden
bağımsız.

✅ Makalede: "Kontrol deneyleri" alt bölümü (tablosuz, bir paragraf).

---

## Deney 8 — Öz-rehberli eğitim (18 koşu, H200)

**Izgara:** `slurm/gpt_kilavuz.slurm`, `gpt_kilavuz_egri.slurm` · **Bayrak:** `--guide-frac 0.3`

**Amaç.** Yapılandırılmış katmanların sıfırdan eğitim zorluğunu izole etmek (Wei ve ark.).

**Sonuç.** Faz geçişi: L≤8 etkisiz, L=16'da üç desende birden anlamlı. **Kritik:**
kılavuz altında kelebeğin halkaya üstünlüğü kayboluyor (12.0σ → 1.5σ) — mimari sanılan
farkın büyük kısmı optimizasyon zorluğu.

✅ Makalede: Tablo `tab:kilavuz`. **Özette de üçüncü bulgu olarak var.**

---

## Deney 9 — bmm uygulaması (6 koşu, H200)

**Izgara:** `slurm/gpt_bmm.slurm` · **Bayrak:** `--fast`

**Amaç.** Kelebeğin gerçek toplu matris çarpımı uygulamasının maskeliyle aynı kaliteyi
verip vermediği + hız.

**Sonuç.** Kalite aynı (fark ≤0.0007 bpc), ama bmm bu ölçekte **1.9× yavaş** —
XOR eşleşmesinin gerektirdiği toplama/birleştirme adımları bellek-bağımlı.

✅ Makalede: "Uygulama ve ölçekleme" alt bölümü.

---

## Deney 10 — Dağıtık iletişim ölçümü (9 kayıt, 8×H200)

**Betik:** `bench_distributed.py` · **Izgara:** `slurm/distributed.slurm`
**Kurulum:** d ∈ {2048, 4096, 8192}, world=8, bf16

**Amaç.** Parçalama stratejilerinin gerçek muhatap sayısı, bayt ve ileri-geçiş süresi.

**Sonuç (ileri geçiş, ms).**

| d | tp_dense (7 muhatap) | ring (2) | butterfly (1) | block (0) |
|---|---|---|---|---|
| 2048 | 0.950 | 0.552 | 0.359 | 0.259 |
| 4096 | 1.878 | 1.097 | 0.814 | 0.610 |
| 8192 | 4.247 | 2.829 | 2.333 | 1.903 |

Kelebeğin yoğun TP'ye hızlanma oranı: **2.6× → 2.3× → 1.8×** (d büyüdükçe azalıyor).

⚠ **KISMEN MAKALEDE.** Makale yalnızca **d=8192** satırını veriyor (`tab:dagitik`).
Üç genişlikteki **ölçekleme eğilimi yok** — oysa kazancın d ile azalması anlamlı bir
bulgu. Ayrıca kayıtlarda `proj_sec_1gbps/10gbps/100gbps` alanları var, karar modelinin
girdisi bunlar; makalede bu bağ açıkça kurulmuyor.

---

## Deney 11 — Dağıtık EĞİTİM (12 koşu, 8×H200)

**Betik:** `train_dist_ffn.py` · **Izgara:** `slurm/dist_train.slurm`
**Kurulum:** dikkatsiz FFN yığını, d=512, L=8, 3000 adım, rank başına eşit parametre

**Amaç.** Parçalanmış FFN'in gerçekten uçtan uca eğitilebildiğini ve kalitesini göstermek.

**Sonuç.**

| Varyant | val bpc | Muhatap | İletişim (MB/adım) | Süre (sn) |
|---|---|---|---|---|
| dense_tp | 2.7347 ± 0.0572 | 7 | 29.36 | 96 |
| block | 2.6532 ± 0.0096 | 0 | 0.11 | 121 |
| **butterfly** | **2.0416 ± 0.0014** | **1** | **4.31** | 144 |
| **ring** | **2.0155 ± 0.0016** | 2 | 8.50 | 152 |

**Yorum — bu makalenin en pozitif sonucu.** Kelebek, yoğun TP'ye göre **6.8× az
iletişim, 7× az muhatap ve 0.69 bpc daha İYİ kalite** veriyor (≈21σ). Ayrıca blok
(karışmayan, τ>L) 2.65'te kalırken karışan iki desen 2.02–2.04'e iniyor: **τ bulgusu
tamamen farklı bir ortamda da tutuyor.**

⚠ **MAKALEDE TEK CÜMLE, TABLO YOK.** Şu an yalnızca "ring 2.02, butterfly 2.04 …
beats narrow-dense (2.73)" diye geçiyor. İletişim ve muhatap sütunları hiç görünmüyor,
τ bağlantısı hiç kurulmuyor.

---

## Deney 12 — Ölçekleme (6 koşu, 8×H200)

**Izgara:** `slurm/scale_410m.slurm` · **Kurulum:** ~403M (d=1024, L=32), enwik9, ~2B token

**Amaç.** Küçük ölçekteki sıralamanın büyük ölçekte de geçerli olup olmadığı.

**Sonuç.** Yoğun 0.9096±0.0101, rastgele 0.9193±0.0011, kelebek 0.9236±0.0003 (n=2).
Kendi σ≥2 ölçütümüzle: **yoğun–rastgele anlamlı DEĞİL** (1.34σ), yoğun–kelebek sınırda
(1.95σ). Anlamlı kalan tek karşılaştırma rastgele–kelebek (5.36σ) ve açık ölçekle
yarılanıyor.

✅ Makalede: Tablo `tab:olcek`, dürüst σ yorumuyla.

---

## Deney 13 — Duman testleri (4 kayıt) — DENEY DEĞİL

`ddpduman_*.json` (d=256, DDP boru hattı doğrulaması) ve `deneme.json`.
Gerçek deneylerle **aynı JSON şemasını** paylaşıyorlar, dolayısıyla naif bir yeniden
analiz bunları içeri alır. `analiz_tau.py` isimle eliyor.

⚠ `results/_duman/` altına taşınmaları daha güvenli olur.

---

# Denetimin sonuçları

## ✅ Yapıldı — makaleye eklendi

**1. Dağıtık eğitim tablosu (Deney 11).** 12 koşuluk, 21σ'lık, makalenin **tek büyük
pozitif sonucu** tek bir cümleye sıkışmıştı. Artık `tab:disttrain` olarak tablo:
kelebek yoğun TP'ye karşı hem 6.8× az iletişim hem 0.69 bpc daha iyi kalite. Ayrıca
blok–kelebek farkının **109.6σ** olması τ'nun bağımsız bir doğrulaması olarak yazıldı.

**2. Katman ölçeği tablosu (Deney 1).** Özet "üç ölçek" diyordu, ikisinin tablosu vardı.
Artık `tab:katman` var — ve veriyi temizleyince sonuç beklediğimden çok daha güçlü çıktı
(19.9σ / 12.0σ ters yönlü). Bu arada 3 eski + 3 kopya dosya tespit edildi.

**3. Gecikme tabanı (yeni analiz, `gecikme.py`).** Karar modeli yalnızca bant genişliğine
bakıyordu. α–β modeliyle: seyreltme baytı düşürür ama **tur sayısını düşürmez**, dolayısıyla
`lim(β→∞) T = R·α` diye bir taban var. Şehirler arası (10 ms) bu taban hesabı 6.7×
aşıyor. Bu, koridorun neden "aynı şehir"de kapandığını bağımsız olarak açıklıyor ve
koridorun topoloji iyileştirerek genişletilemeyeceğini gösteriyor.
Eklendi: `sec:gecikme`, Tablo `tab:gecikme`.

## ⏳ Yapılabilir — senin kararın

### 3. İletişim ölçekleme eğilimi (Deney 10)
`tab:dagitik`'e d=2048 ve d=4096 satırları eklemek: kelebeğin avantajı 2.6×'ten
1.8×'e iniyor. Bir hakem "bu kazanç ölçekle ne oluyor?" diye soracak; cevabı elde var.

### 4. Gerçek seyrek çekirdek ölçümü (Deney 2)
`torch.sparse` en seyrek uçta yoğunu 1.8× geçiyor. "Hızlanma yok" iddiasını
daraltmak için ikinci bir dayanak — üstelik ölçülmüş.

### 5. Duman dosyalarını `results/_duman/` altına taşımak
Yeniden üretilebilirlik hijyeni.
