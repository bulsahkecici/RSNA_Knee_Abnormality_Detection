# RSNA Knee: yüksek skor için araştırma ve deney planı

**Tarih:** 9 Ekim 2026, Türkiye saati. **Amaç:** daha güçlü ve güvenilir bir yarışma sistemi kurmak. Birincilik veya 0.9+ skor garantisi yoktur. Bu belge araştırma ve uygulama planıdır; aşağıdaki yeni deneylerin sonuçları henüz ölçülmedi.

## 1. Şu an tam olarak neredeyiz?

8. kurtarma notebooku tamamlandı. Attention eğitimi 8 epoch, 3.368 optimizer denemesi, 3.361 başarılı adım ve 7 AMP atlamasıyla bitti; değerlendirme ve arşiv doğrulandı. Bu hata yeniden çözülmesi gereken açık bir iş değildir.

| Model | Gold-29 AUC | Zayıf etiketli 824 çalışma AUC | Ölçülmüş public LB |
|---|---:|---:|---:|
| Altı merkez, tek blok | 0.781277 | 0.796132 | 0.767 |
| Altı merkez, iki blok, ortalama | **0.795320** | 0.808382 | Gönderilmedi |
| Altı merkez, iki blok, attention | 0.783277 | 0.810280 | Gönderilmedi |
| Son iki modelin eşit karışımı | 0.790402 | **0.811031** | Gönderilmedi |

İki bloklu ortalama modelimiz mevcut yerel referanstır. Gold-29 çok küçük ve tekrar tekrar seçimde kullanıldı; attention'ın bütün koşullarda kötü olduğu sonucu çıkmaz. Weak AUC rapor etiketleriyle uyumu ölçer; görüntüden uzman tanısının aynısı değildir. Yerel AUC ile public LB birbirine doğrudan çevrilemez.

9 Ekim canlı Kaggle CLI sorgusunda liderler **0.964**, ilk 20 içindeki skorlar **0.960–0.964**. Ölçülmüş public referansımızla lider arasındaki fark 0.197. Liderlerin özel yöntemlerinin tamamı kamuya açık değildir. [Resmî liderlik tablosu](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard).

**Kaggle gönderimi kullanıcının talebiyle kapalıdır.** `state/submission-policy.json` bu kararı yayınlayıcıda uygular; araştırma veya model sayısını artırma izni bunu kaldırmaz.

## 2. Araştırmanın ana sonucu

En güçlü çalışma hipotezi: büyük fark, yalnızca backbone büyüklüğünden değil, modelin gördüğü anatomi, eğitim hedeflerinin niteliği ve MRI'ya uyarlamanın birlikte sınırlı kalmasından doğuyor. Öncelik, daha iyi tek model ailesi oluşturmak; çeşitlilik sağlandıktan sonra ensemble büyütmek.

| Kaynak | Doğrulanan / bildirilen bulgu | Bizim için karar |
|---|---|---|
| [Kaggle ayrıntılı CoAtNet tarifi](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792) | Yazar 0.949 bildiriyor; çok seri, yoğun dilim grupları ve bulguya özel attention anlatılıyor. Önceki kaynak incelemesinde eğitim kodu görüldü; skor burada yeniden üretilmedi. | Görüntü kapsamı ve pencere örneklemeyi öne al. OAI içeren ağırlıkları otomatik benimseme. |
| [Tek-model tartışması](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304) | DINOv2-S, ConvNeXt ve küçük CNN'lerle yüksek skor beyanları; pseudo-label ve veri kalitesi vurgusu. Skorlar bağımsız doğrulanmış deneylerimiz değil. | Aynı zayıf modelin çok seed'i yerine ayrı öğrenme yaklaşımları dene. |
| [Açık challenge_rsna deposu](https://github.com/mathischmp/challenge_rsna) | README daha geniş görüntü kapsamına geçişte 0.853→0.920 public değişimi bildiriyor; mimari de değişmiş, nedensel ablation değil. Tam eğitim akışı ve olumsuz deneyler açıklanıyor. | Kamuya açık tarifi başlangıç noktası yap; kopyalanan bütün tercihler için kendi kontrolünü kur. |
| [Etiket benchmark'ı](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747540) | Yazar, public etiket kaynakları arasında fark ve yüksek belirsizlik bildiriyor; en iyi tek kaynak her zaman konsensüsten kötü değil. | Kaynakların ortalamasını otomatik doğru kabul etme; maske, ontoloji ve anlaşmazlık analizi yap. |
| [NTejas açık deposu](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection) | Çoklu model aileleri ve etiket düzeltmesinin her zaman iyileştirmediğine ilişkin deney kayıtları içeriyor. | Model çeşitliliğini ve etiket değişimini ayrı deney olarak ölç. |
| [DINOv2 resmî kod/ağırlıklar](https://github.com/facebookresearch/dinov2), [ConvNeXt resmî kod](https://github.com/facebookresearch/ConvNeXt) | Kullanılabilir farklı omurgalar ve ön-eğitim kaynakları mevcut. | Kendi fold modellerimizi denetlenebilir başlangıç ağırlıklarından eğit. |
| [OrthoFoundation yazar deposu](https://github.com/ytrsk/OrthoFoundation) | Diz MR/röntgeniyle ön-eğitilmiş modelin checkpoint dosyası listeleniyor; indirme/hash doğrulaması yapılmadı. OAI/fastMRI ve özel klinik kaynaklar içeriyor. | Teknik olarak güçlü öncelikli aday; önce kullanım/çakışma incelemesi, sonra dondurulmuş özellik deneyi. Yarışmaya uygunluğu henüz doğrulanmadı. |
| [OrthoDiffusion yazar ağırlıkları](https://huggingface.co/lanstat0123/orthodiffusion) | Üç düzleme ait 3D checkpoint listeleri mevcut. | Daha ağır entegrasyon nedeniyle ikinci sıra; bu takvimde tam üretici model eğitimi yerine varsa özellik çıkarımı pilotu. |
| [RadImageNet resmî depo](https://github.com/BMEII-AI/RadImageNet) | Tıbbi görüntülerle ön-eğitilmiş CNN ve ağırlık bağlantıları mevcut. | Küçük bir tamamlayıcılık deneyi; hazır 12 sınıflı diz tanı modeli olarak değerlendirme. |

Kaggle başlığı, oy sayısı, bir GitHub README tablosu veya score sıralaması, tek başına doğrulanmış güncel LB sonucu değildir. Kaynak ayrıntıları [Kaggle araştırma notunda](KAGGLE_STRATEGY_20261009.md) ve [açık kaynak araştırma notunda](OPEN_SOURCE_STRATEGY_20261009.md) tutulur.

## 3. Darboğazlar ve önerilen değişiklikler

### Görüntü kapsamı

Mevcut model **3 seri × 6 merkez = 18 üç-dilimli pencere** görüyor. Bu, 18 ham dilim demek değildir: 54 dilim kanalı işlenir ve komşuluk nedeniyle tekrar olabilir. Esas fark, kapsanan seri çeşitliliği, fiziksel konumlar ve her epoch'ta aynı noktaların görülmesidir.

Öneri: sagittal/coronal/axial fluid-sensitive serilere sagittal ve coronal yapısal seriler ekle; daha yoğun ham dilim deposundan eğitimde rastgele, çıkarımda sabit pencereler seç. Üç kanallı pencere aynı serinin fiziksel olarak komşu dilimlerinden gelmeli; seri sınırları aşılmamalı. Eksik seri maske olarak kalmalı. Önce 3-seri/yoğun ve 5-seri/yoğun farkını ayır.

Depoda başlayan `cache_v4_source` sıkıştırılmış 12-merkezli taslak **tamamlanmış veya gerçek veride doğrulanmış sayılmıyor**. Bir sonraki mühendislik adımı bunu denetlemek ve 64 çalışmalık gerçek pilotla kapsam, bozuk dosya, disk ve decode süresini ölçmek. 6-merkez depomuz 11.43 GB; aynı triplet düzeninde 5 seri × 12 merkez yaklaşık **38.1 GB ham** eder. Sıkıştırma oranı bilinmiyor. Tercih: ham dilimleri bir kez saklayıp tripletleri okurken kurmak; gerekirse birden fazla veri parçası. 570 GB arşiv Mac'e indirilmeyecek.

### MRI'ya uyarlama ve model aileleri

DINOv2-S için son 2, son 4 ve bütün blokları aşamalı açmayı aynı veri üzerinde karşılaştır. Daha yüksek çözünürlükte doğrudan 384 seçmek yerine patch boyutuna uygun 336/392 kullan; maliyeti gerçek GPU'da ölç. ConvNeXt-Tiny ve CoAtNet-0/1 için 256/384 adayları hazırlanabilir. CoAtNet eğitim tarifi başlangıç hipotezidir; veri/etiket/optimizasyon farklarını tek seferde değiştirerek nedensel kazanç iddia etme.

Attention, yoğun pencere grubunda tekrar değerlendirilmeli. Mevcut negatif gold sonucu yalnız seyrek altı-merkez koşuluna aittir. CLS+patch özeti, seriye/konuma ait gömme ve bulguya özel ağırlıklandırma ikinci öncelikli seçeneklerdir. Bir görüşe ait 256 patch'in tümü üzerinde pahalı global attention kurmadan önce ucuz özet özellikleri dene.

### Etiketler ve öğretmen modeller

Qwen etiketleri başlangıç kaynağı olarak kalır. Public sayısal etiket kaynaklarıyla UID, kolon anlamı, eksik/negatif ayrımı ve sınıf bazında anlaşmazlık tablosu çıkar. Raporlar yalnız yerelde işlenir. `-1`, belirsiz ve bahsedilmeyen hücreler mevcut rapor hedeflerinde maske kalır. Öğretmen olasılığı ayrı kaynak, ağırlık ve sürümle saklanır; sessizce gerçek etiket yapılmaz.

Güncel iki-blok modelde MCL gold AUC 0.4872 (yalnız 3 pozitif), weak 0.8013. Bu tek başına ters kolon ya da etiket hatası kanıtı değildir. Lateral meniscus weak AUC 0.7447, Synovitis weak AUC 0.7796 ve yalnız 12 gözlenen negatif: bunlar kalite incelemesi için adaylardır, gold üstünde sınıfa özel ensemble ağırlığı seçmek için yeterli değildir.

Öğretmen–öğrenci ilk deneyi sabit bir dış doğrulama kümesiyle yapılır. **Öğretmenler de öğrencinin dış doğrulama görüntülerini görmemeli.** Dış eğitim havuzunda 3 iç fold öğretmen kurup dış-eğitim örneklerine çapraz tahmin üret; öğrenciyi bununla eğit. Tüm veri üzerinde üretilmiş sıradan OOF tahminleri, öğretmenler öğrencinin dış fold'unu gördüyse öğrencinin bağımsız değerlendirmesini sağlamaz.

### Doğrulama düzeni

Yeni büyük seri için sabit, tekrar/benzer görüntü gruplarını ayıran fold manifesti üret. StudyInstanceUID hasta kimliği değildir. Varsa güvenilir kurum/cihaz bilgisiyle ayrıca domain dayanıklılığını raporla. Rapor tekrar grupları yerelde hash olarak çıkarılabilir; metin buluta gitmez.

Yeni fold modelleri resmî genel ön-eğitimden başlar. Eski supervised frozen checkpoint, başka fold'un doğrulama örneklerini zaten gördüğünden bağımsız OOF başlangıcı olamaz. Daha güçlü yeni değerlendirme serisinde mümkünse 58 gold'un tamamını yeni model eğitiminden dışla; geçmişte seçimde kullanıldıklarından bunları 'hiç görülmemiş test' diye adlandırma. Gold-29 tarihsel karşılaştırma ayrıca korunur.

En iyi iki aday ikinci dış fold'da doğrulanır; sonra 3 fold'a genişletilir. Karar: sabit masked weak hedeflerde fold tutarlılığı, gold belirsizlik aralığı, sınıf/domain gerilemeleri ve süre birlikte. 29 gold üstündeki her küçük eksiye otomatik veto koymak yeni araştırma seçim kuralı olmamalı; fakat bu öneri mevcut otomatik yayın politikasını değiştirmez.

## 4. Uygulanacak deney sırası

Aşağıdaki sayılar sonuç değil, azami ilk plan kapsamıdır. İlk tarama tam eğitimden önce kısa arıza/VRAM kontrolü içerir; kısa eğitimin sıralaması nihai performans yerine geçmez.

| Aşama | Deney | Eğitim sayısı | Karar |
|---|---|---:|---|
| P0 | Doğrulama/fold manifesti, sayısal etiket denetimi, 64 çalışma cache pilotu; bir public modelin köken incelemesi | 0 | Veri ve ölçüm altyapısı hazır mı? |
| P1 | DINO: seyrek yeniden başlangıç, 3-seri yoğun, 5-seri yoğun, 4-blok, tam açma, yoğun attention | en çok 6 | Görüntü ve uyarlama iyileşiyor mu? |
| P2 | ConvNeXt-Tiny; CoAtNet-0/1, aynı pencere/etiket politikası | 2 | Daha güçlü veya tamamlayıcı aile var mı? |
| P3 | En iyi iki aileyi toplam 3 dış fold'a tamamlama | en çok 4 ek | Kazanç tek fold tesadüfü mü? |
| P4 | En iyi tarifte alternatif sayısal etiket; dış-train içinde 3 öğretmen + 1 öğrenci | en çok 5 | Etiket ve görüntü öğretmeni katkısı var mı? |
| P5 | İhtiyaç varsa RadImageNet/CLS+patch gibi tek alternatif | en çok 1 | Ensemble'a farklı hata profili katıyor mu? |

**İlk üst kapsam: yaklaşık 18 tam eğitim; ayrıca en fazla iki kısa dondurulmuş-özellik sınıflandırıcı deneyi.** Kazanç vermeyen dallar büyütülmez. Bu sayı zorunlu hedef veya hemen başlatılmış işler değildir. Öğrenciyi üç dış fold'a doğrulamak gerekirse diğer iki dış fold için ayrı iç öğretmenler gerekir: en fazla 8 ek eğitim; yalnız P4 başarılı ve zaman/bütçe uygunsa. Birden çok seed en sona bırakılır.

Kamuya açık güçlü ağırlıklarla ikinci kol: lisans, tam sürüm, eğitim veri kökeni ve görüntü çakışması incelenir. Uygun bir model önce **REFERENCE** olarak çalıştırılır. Bilinmeyen gold çakışması varsa onun gold AUC'si kendi modelimizin bağımsız sonucu sayılmaz. Bu kol erken pratik bir kıyaslama sağlayabilir; paylaşılan notebookun yüksek skorunu bize ait skor olarak raporlamayacağız.

## 5. Kaynak, süre ve son tarih

İki tamamlanmış Colab handshake'i **NVIDIA A100-SXM4-80GB** doğruluyor; yeni oturumda aynı GPU garanti değil. Mac M5 Max/36 GB: hazırlık, analiz, yerel etiketleme ve küçük işler. Tam eğitim Colab GPU; Kaggle ise ham MRI ön işleme ve ileride izin verilirse offline doğrulama için. TPU'ya geçiş, mevcut CUDA akışını yeniden kurma maliyeti nedeniyle şu takvimde ilk tercih değil.

Canlı `kaggle quota`: GPU **19.13 saat kalan**, 10.87 kullanılan; TPU 20 saat. API yenileme değeri `2026-10-10T00:00:00`, saat dilimi belirtilmedi. Colab kredi bakiyesi ve kredi/saat oranı bilinmiyor; compute unit, GPU saati değildir.

Tahmin yöntemi: her yeni adayda gerçek 100–200 çalışma ile saniye/çalışma, tepe VRAM ve veri okuma süresi ölçülür. `t_epoch = n_train × t_study / 3600`; tam iş buna doğrulama, kaydetme ve başlangıç sürelerini ekler. A100 üstünde 18 farklı koşulun tamamının aynı sürede biteceği varsayılmaz.

| Planlama senaryosu | Ayrılabilecek Colab GPU saati | Kapsam |
|---|---:|---|
| Dar | 24 | 3–4 güçlü aday ve seçilmiş ikinci-fold doğrulaması; öğretmen kolu ertelenebilir |
| Orta | 72 | Yaklaşık 8 aday taraması, en iyi iki ailenin fold genişlemesi; ölçülen süreye bağlı |
| Geniş | 144 | 18 eğitimlik plan ve başarılıysa öğretmen kolu; yine 13 günlük takvime bağlı |

Bunlar **ölçülmüş maliyet veya harcama taahhüdü değildir**; pilot sürelerine göre koşu sayısı ayarlanır. Kaggle kotasının yaklaşık 9 saati ilerideki offline tam koşu için tutulabilir; bu, modelin 9 saat sürmesi gerektiği anlamına gelmez.

Önerilen takvim: 9–10 Ekim P0; 10–13 Ekim P1/P2; 14–17 Ekim P3/P4; 18–20 Ekim ensemble ve süre; 21–22 Ekim paket dondurma ve yalnız kullanıcı gönderimi yeniden açarsa resmî skor doğrulaması. Son tarih **22 Ekim 23:59 UTC = 23 Ekim 02:59 Türkiye**; katılım/takım birleşimi son tarihi 15 Ekim 23:59 UTC. [Resmî takvim](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline).

## 6. Ensemble ve seçim kuralları

Başlangıçta model aileleri arasında eşit olasılık ortalaması. OOF üzerinde hata benzerliği ve sınıf/domain tutarlılığı ölç; yalnız düşük korelasyon olması yeterli değil, birleşimin ölçülen katkısı olmalı. Öğrenilmiş ağırlık gerekiyorsa az parametreli, ayrı dış-fold doğrulamalı olsun; 29 gold'a göre 12 ayrı sınıf ağırlığı öğrenme.

EMA/SWA ve anatomik olarak doğru TTA son aşamada süre başına kazançla değerlendirilir. Her pencere/seri için geometri doğrulanmadan genel yatay flip kullanılmaz. Çok büyük ensemble sığmazsa OOF öğretmenlerden küçük öğrenciye distillation araştırılır.

Offline son paket için 30 gerçek çalışma ve uzun/çok serili kuyruk örnekleriyle ölçüm, ardından tam doğrulama gerekir. Toplam bütçe: kurulum + ağırlık yükleme + DICOM decode + tüm modeller/TTA + CSV. İç hedef 6 saat civarı tahmin ve kalan sürede belirsizlik payı; yarışmanın gerçek şartı notebook başına en fazla 9 saat ve internet kapalı olmasıdır. [Code Requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements).

## 7. Açık kaynakların kullanım kararı

Öncelik: resmî DINOv2/ConvNeXt ve erişilebilir CoAtNet ön-eğitimi. Paralel ucuz özellik taramasında RadImageNet ve, erişim/kullanım incelemesi uygunsa, OrthoFoundation. OrthoFoundation diz alanına yakın olduğu için önemli bir yeni aday; özel klinik/OAI kaynakları ve upstream DINOv3 koşulları nedeniyle henüz yarışma için onaylı kabul edilmiyor. BioMedCLIP/DINOv3, hazır erişim ve hızlı pilot katkısı varsa yedek aday. 3D MRI ön-eğitimi, segmentasyonla ROI ve MRNet/fastMRI+ bu takvimde koşullu ikinci sıra: görev eşlemesi, veri erişimi ve ek hazırlık gerekir. X-ray başarısı doğrudan MRI başarısı değildir.

[KneeCoT için açık host yasağı](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109) var. [OAI ağırlıkları sorusunda](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747455) incelenen yanıtlarda host açıklığı yok; 0.949 etiketi otomatik kullanım izni sağlamaz. Public label/weight kaynağının açıklığı, veri kökeni ve lisansı ayrı ayrı doğrulanır.

## Karar

Bir sonraki somut iş **P0 veri/doğrulama hazırlığı, yoğun MRI pilotu ve OrthoFoundation/RadImageNet kullanım incelemesi**, ardından **daha geniş görüntü üzerinde güçlü tek-model karşılaştırmaları**. Sonuç veren iki aile fold sayısıyla büyütülür; öğretmen/öğrenci ve ensemble daha sonra gelir. Bu araştırma sırasında yeni eğitim veya Kaggle gönderimi başlatılmadı.
