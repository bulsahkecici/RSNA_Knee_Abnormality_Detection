# Yeniden inceleme ve çıkış yolu — 15 Ekim 2026

**Sonuç:** Aynı DINO/KneeNet tarifinin seed, LR ve ensemble komşuluğunda belirgin doygunluk var. Buna karşılık public reader'ın veri temsilini kendi eğitimimizde henüz gerçekten yeniden üretmedik. En somut araştırma yönü, **KneeNet + ortak anatomik geometri + tam seri hacimleri + değişken seri/pencere örnekleme**. Bu bir başarı sonucu değil, kaynak kod ve kayıtlarla desteklenen deney hipotezi. Birincilik garantisi veya mevcut skorlarla birinciliğin yakın olduğu iddiası yok.

Bu inceleme deney günlüğü, P0/public araştırmalar, son Colab review/metrics/prediction dosyaları, submission kayıtları, fold kuralları, cache/DICOM kodu, KneeNet notebook üreticisi ve kullanılan reader'ın kaynak kodunu kapsar. Her eski commit satır satır denetlenmedi. Ham raporlar okunmadı; sayısal eğitim hedefleri ve seri metadata'sı incelendi. Yeni GPU işi, upload veya submission başlatılmadı. Geçmiş nihai seçimler değiştirilmedi.

## Gerçekte neredeyiz?

15 Ekim salt okunur Kaggle API kontrolü:

| Gönderim | Public AUC | Kaynak |
|---|---:|---|
| 57143034 | 0.862 | Kendi D3ALLLR2 modelimiz |
| 57178710 | **0.929** | Public ConvNeXt reader, üç checkpoint |
| 57184259 | 0.927 | Reader; Effusion kendi DINO modelimizden |
| 57184253 | 0.926 | Reader %80 / kendi DINO %20 rank blend |
| Public lider | **0.967** | Güncel leaderboard ilk sayfası |

En iyi gönderim ile public lider arasındaki fark 0.038. Kendi en iyi gönderimimizin farkı 0.105. Private sonuç ve güncel kendi sıramız bu incelemede ölçülmedi. Public liderlik final birinciliğini belirlemez. [Leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard).

Takvim yeniden doğrulandı: takım birleşimi/katılım **16 Ekim 02:59 Türkiye**, final gönderim **23 Ekim 02:59 Türkiye**. Bugün tamamlayıcı modeli olan bir takım arkadaşıyla birleşme seçeneğinin son günü; bu teknik deneylerden ayrı bir strateji kararı. Kimseyle iletişim kurulmadı. [Resmî takvim](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline).

## Hangi yollar zaten denendi?

`docs/EXPERIMENTS.md` sonuçlarına göre:

- Yerel hard etiket → public soft etiket ve DINO son dört blok uyarlaması gerçek, birden fazla fold/seed'de tekrarlanan ilerleme sağladı.
- Daha çok eğitim verisi ve backbone LR 2e-5 public skoru 0.835 → 0.845 → 0.862 taşıdı.
- DINO tam açma, mevcut tarifte CoAtNet-0 ve kısmen açılan ConvNeXt-T, daha fazla seed/fold ensemble, LR 4e-5 ve aynı DINO öğretmenlerinden pseudo-label ilerlemeyi doğrulamadı.
- DINO'da v5/5-slot/252 px deneyi desteklenmedi. Bu, KneeNet'te geometri veya çözünürlük değişiminin işe yaramayacağı anlamına gelmez.
- KneeNet + farklı ön-eğitim/soft etiket/augmentasyon/EMA paketi kendi gold AUC'mizi 0.848'den yaklaşık 0.892'ye taşıdı. Bu paket deneyinden tek bir bileşenin katkısı çıkarılamaz.
- Üç KneeNet seed'i birlikte gold 0.902; reader gold 0.911. Sabit reader/KneeNet blend gold'da reader'ı geçmedi. Daha fazla aynı seed ilk öncelik olmamalı.
- DINO'nun gold Effusion üstünlüğünü public'te kullanmak başarısız oldu. Sınıf bazında gold'a bakarak routing/weight aramayı büyütmek iyi bir çıkış değil.

Önceki C1 sonucu, “ConvNeXt kötü” hükmü değildir: C1 kısmen açılmış ImageNet-1k backbone + DINO çalışma düzeniydi; K5 tamamen eğitilen in12k→in1k backbone + KneeNet düzeni. Aynı şekilde CoAtNet-0/224 sonucu, büyük CoAtNet-RMLP-2/384/MIL tarifini test etmez.

## Yeni bulgular

### 1. Reader ile kendi KneeNet'imiz aynı görüntü temsilini kullanmıyor

İndirilen `train.py`, `preprocess.py`, `infer.py` SHA256'ları mevcut `reader-files.json` ile birebir eşleşti. Dolayısıyla karşılaştırılan kaynak, gold değerlendirmesinde kullanılmış sürüm. [Reader kaynağı](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader).

| Bileşen | Public reader | Kendi K5/v5 |
|---|---|---|
| Düzlem içi yön | IOP ile transpose/flip, ortak LPS eksenleri | IOP/IPP ile dilim sırası; ortak düzlem içi yön dönüşümü yok |
| Dilim sırasının yönü | Düzlem başına sabit anatomik yön | IOP çapraz çarpım normalinin işaretine bağlı |
| Küçük FOV | Fiziksel ölçü korunur, eksik alan pad edilir | Crop doğrudan kareye resize edilir; FOV küçükse fiziksel ölçek değişebilir |
| Eğitimde seri seçimi | Her slotun adayları arasında rastgele | Cache hazırlanırken tek seri seçilir |
| Eğitimde pencere | Hacimde yeni merkezler | 12 sabit pencere arasından 8 örnek |
| Seri slotları | 6 | 5; axial non-fat-suppressed yok |
| Konum girdisi | Gerçek merkez / (n−1) | İdeal `linspace(.05,.95,12)` |
| Hacim/piksel deposu | 384 px, tüm seri / en çok 64 dilim | 256 px sabit tripletler |
| Seri yoğunluk window'u | Hacimden, [1,99.5] | Seçilen dilimlerin crop'larından, [0.5,99.5] |

Bu farkların varlığı doğrulandı; gold/public açığının hangi kısmına yol açtıkları **ölçülmedi**. Özellikle in-plane yön çeşitliliği varsa, küçük rotasyon augmentasyonu 90°/180°/ayna farklarını ortadan kaldırmaz. Konum embedding'ine ters yönlü seriler verilmesi de olası sorun. Gerçek veride yön dağılımı için DICOM header pilotu gerekir; seri CSV'sinden bu dağılım çıkarılamaz.

Sayısal seri metadata'sı: 4407 çalışmada 24371 seri; **1936 çalışmada aynı düzlem/fat-suppression slotunda birden fazla seri**, **857 çalışmada axial non-fat-suppressed seri** var. Bunlar modelin kaç çalışmada kazanç sağlayacağını göstermez; kaybedilen çeşitliliğin kapsamını gösterir.

64 çalışmalık mevcut v5 pilot manifestinde 3444 mevcut pencerenin ideal konumuyla gerçek yuvarlanmış merkez konumu arasındaki ortalama mutlak fark 0.00895, maksimum 0.03052. Küçük bir fark; büyük performans kazancı beklenen ana kaldıraç olarak sunulmuyor.

### 2. Soft etiketler mevcutken eski hard-label filtresi 188 çalışmayı atıyor

`make_kneenet_notebook.py` eğitim UIDs'i `make_colab_notebook.train_uids()` üzerinden alıyor. Bu fonksiyon eski yerel numeric etiketlerde en az bir observed target arıyor. Oysa K5 başka bir soft etiket tablosunu kullanıyor.

- Gold ve gold grubuna ait diğer üyeleri çıkardıktan sonra eğitime uygun: **4348**.
- Bunların tamamında public-mean4 soft hedef var.
- K5ALL eğitiminde kullanılan: **4160**.
- Dışarıda kalmış ama **12/12 soft hedefi bulunan: 188** (~%4.3).
- K5ALL train/gold UID kesişimi: **0**.

Bunun düzeltilmesi, etiket değişiminden sonra eğitim kapsamının kullanılan hedef kaynağına göre belirlenmesini sağlar. Gold/grup exclusions aynen uygulanmalı; bu 188 örneğin eklenmesinin AUC kazancı henüz bilinmiyor. Mevcut v5 full manifestindeki cache erişilebilirlikleri yeniden doğrulanmadan 4348'in hepsinin hazır olduğu varsayılmamalı.

### 3. Hassasiyet farkı açığın açıklaması değil

Mevcut `gold_fp32` olasılıkları üzerinden 58 gold için AUC'leri yeniden hesapladım:

| Model | bf16 değerlendirme AUC | fp32 AUC |
|---|---:|---:|
| K5ALLS2 | 0.894996 | 0.895225 |
| K5ALLS3 | 0.897813 | 0.897917 |

Olasılık farklarının maksimumu 0.065/0.062; parity açısından önemli, ama macro AUC farkı sadece 0.000229/0.000104. fp32'ye geçmek bir sıçrama yaratmıyor. K5ALL'ın fp32 kaydı yok; üç-seed E3 için fp32 sonuç çıkarılmadı. Reader kendi fp16 inference'ıyla ölçülmüş referanstır. Ayrıntı: `STRATEGY_AUDIT_20261015.json`.

Dört soft etiket kaynağının sayısal sütunlarında negatif veya >1 hedef bulunmadı. Bu girdilerde `-1`'in yanlışlıkla BCE hedefi yapılmış olması açıklaması desteklenmiyor. Klinik doğruluk bundan çıkarılamaz.

## Ben olsam kalan haftayı nasıl kullanırdım?

**Önce bir güçlü veri-temsili deneyi, sonra tek tamamlayıcı model.** 18 koşuluk yeni bir tarama önermiyorum. Mevcut reader-only yedek aday olarak korunur.

1. **K0: KneeNet için ölçüm ve kapsam kontrolü.** K5F0 sonucu mevcutsa toplanır; gelmediyse tamamlanmış varsayılmaz. Fold-0 kontrolünün fp32 tahminleri ve sabit sayısal hedefleri alınır. Yeni soft-source train UID listesiyle 188 örneğin fold-0 dışındaki kısmı eklenir; aynı cache/model/schedule ile bir kontrol eğitimi yapılır. Eski deneylerin kimlik/hash'ları değiştirilmez.
2. **K1: Gerçek reader veri temsili.** Önce 64–128 gerçek çalışmanın header/decode pilotu: ortak in-plane yön ve slice-order yönü, fiziksel ölçek/padding, 6 slot, tüm aday seriler, tam hacim ve gerçek merkez konumu. Eksik/bozuk seriler açık maskelenir; sıfır hacim başarı sayılmaz. Model 256 px, label/schedule/seed K0 ile aynı kalır. Önce bu temsil paketi tek koşuda karşılaştırılır; kazanç varsa bileşenleri ayırmak gerekir. İlk seferde yüksek model çözünürlüğü eklenmez. Ham MRI Mac'e indirilmez.
3. **K2: K1 başarılıysa model çözünürlüğü 320/384.** Hacim cache'inden sadece forward çözünürlüğü değiştirilir. Aynı 256 px cache'i büyütmek detay geri getirmez. K1 başarısızsa bu kola GPU bütçesi ayrılmaz.
4. **Tek farklı aile:** K1 üzerinde tam açılmış ConvNeXt-Small veya kaynak/hash incelemesi yapılmış CoAtNet-RMLP-2 + aynı study head. İkisinden yalnız biri. Gold nokta tahmininden çok sabit dış fold, sınıf gerilemeleri ve ek inference maliyetiyle karar verilir. Kendi önceki C1/C2 negatifleri bu eşleşmiş deneyi dışlamaz.

K0/K1 için karar kuralı **koşulardan önce** yazılır: sabit weak hedeflerde macro AUC delta en az +0.004 ve eşleştirilmiş bootstrap alt sınırı >0; gold delta/CI ve sınıf gerilemeleri ayrıca raporlanır. En iyi tarif ikinci dış fold'da aynı yönde doğrulanır. Gold, çok kez seçimde kullanıldı; yeni bağımsız test diye adlandırılmaz. K1 paket deneyi artış gösterirse nedenini tek bir bileşene atfetmeyiz. Tahmini süre yerine yeni GPU'da kısa gerçek throughput ölçümü yapılır; önceden sınırlandırılmış 2–3 tam pilot sonrası başarısız dal kapanır.

**Pseudo-label ikinci sıra:** Önce gerçekten daha güçlü/geometrisi farklı öğretmen yaratılmalı. Mevcut public reader'ın ordinary train-study tahminlerini dış fold öğrenciye verip weak sonucu bağımsız sayamayız. Dış-fold'u hiç görmeyen öğretmenler yeniden eğitilir veya tam eğitim provenance'ı doğrulanmış cross-fit tahminler kullanılır. Mevcut başarısız DINO self-distillation bütün pseudo-label yöntemlerini çürütmez; sadece o öğretmen/karışım tarifini çürütür.

**Etiket kolu dar tutulur:** Kaynak uyuşmazlığı ve model/etiket uyuşmazlığı yüksek hücrelerden yerel geliştirme örneklemi seçilir; özellikle Synovitis gibi tekrar eden zayıf sınıflar için host ontolojisiyle kaynak kanıtı kontrol edilir. Yerel LM Studio tek concurrency. Ham rapor buluta gönderilmez. Public label AUC'sini klinik doğruluk yerine kullanarak sınıf bazında kaynak seçilmez. Güçlü yeni etiket/öğretmen yokken ortalamaya yeni bir kaynak katmak otomatik iyileşme sayılmaz.

## Birincilik hedefi hakkında dürüst değerlendirme

Bu plan mevcut kör noktaları kapatır; 0.929 → 0.967 sıçraması için kanıt yok. Geometri pilotu negatif çıkarsa aynı tarifi yeniden seed'lemek yerine durmak gerekir. Kalan haftada tamamlayıcı güçlü bir takım modeli ve uygunluğu doğrulanmış güçlü öğretmen, ince LR ayarlarından daha yüksek tavan sunabilir; kazanma ihtimali sayısallaştırılmadı.

Dış veri tartışması bugün yeniden okundu: 748480'de host cevabı görünmüyor; 741819'deki OAI izni erişim şartlarına bağlı. Public notebookta kullanılmış olması uygunluk kararı yerine geçmez. Mevcut no-OAI politikasını bu rapor değiştirmez. [Yeni soru](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748480), [host koşullu açıklaması](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819).

Alternatif amaç olarak verimlilik ödülü ayrıca araştırılabilir; reader'ın 0.929 skoru veya kısa süre tek başına bu ödülün rekabetçiliğini kanıtlamaz. Ana birincilik hedefinin yerine geçirilmedi.

Kanıt dosyaları: `STRATEGY_AUDIT_20261015.json`, `docs/EXPERIMENTS.md`, `artifacts/colab/p1-K5ALL*/output/`, `artifacts/kaggle/rsna-reader-gold-eval-20261014/output/reader-files.json`. Yeni AUC'ler yalnız mevcut gerçek tahminlerin yeniden hesaplanmasıdır; yeni eğitim sonucu değildir. Rapor mevcut submit/promotion kurallarını değiştirmez.
