# Lateral OA ve sinovit: sayısal hata incelemesi

Durum: iki hedefin incelemesi tamamlandı. Lateral OA protokolü uygulandı ve Kaggle v1 RUNNING; sinovit için değerlendirme grubu sınırlaması devam ediyor. Yeni competition submission yapılmadı. Bu dosya deney protokolü ve bulgu raporudur.

Kaynaklar: tamamlanan 58-gold referans koşusunun dört bileşen tahmini, mevcut K1/K1F1 dış fold tahminleri, sayısal hedefler ve hash-doğrulanmış cache manifestleri. Ham rapor veya görüntü okunmadı. Etiketler değiştirilmedi. UID içeren vaka listesi yalnızca ignored artifacts dizininde tutulur.

## Bulgular

| Hedef | Gold pozitif / negatif | Referans AUC | R6 AUC | Tüm dört bileşende ortak yanlış sıralanan çift |
|---|---:|---:|---:|---:|
| Lateral OA | 11 / 47 | 0.84720 | 0.86460 | 52 |
| Sinovit | 27 / 31 | 0.83632 | 0.85902 | 53 |

Referans lateral OA'da 517 pozitif-negatif çiftin 79'unu, sinovitte 837 çiftin 137'sini yanlış sıralıyor. R6 lateral OA'da referansın 9 hatalı çiftini düzeltiyor ve bu örneklemde referansın doğru çiftlerini bozmuyor. Sinovitte 57 çifti düzeltirken 38 doğru çifti bozuyor. Bunlar yeniden kullanılan gold verisindeki tanımlayıcı karşılaştırmalar; R6 ağırlığını artırmak için bağımsız kanıt değil. Yeni bir ağırlık taraması yapılmadı.

Gold AUC bootstrap %95 aralıkları lateral OA [0.68373, 0.97500], sinovit [0.72378, 0.92770]. 2000 duplicate-group örnekleme; geniş aralıklar küçük örneklemi yansıtıyor. CoAtNet gold görüntülerini eğitimde gördü. Bu aralıklar eğitim örtüşmesini veya geliştirme yanlılığını ortadan kaldırmaz.

58 örneğin tamamında kendi canonical cache'imizde sagittal, coronal ve axial seri var; manifestlerde seri hatası yok. Bu bilgi public modellerin DICOM çözümlemesini doğrulamaz, görüntü kalitesini veya anatomik odağı kanıtlamaz.

## Dış fold kontrolü

| Hedef / alt grup | K1 fold0 AUC (pozitif/negatif) | K1F1 fold1 AUC (pozitif/negatif) |
|---|---:|---:|
| Lateral OA, tüm etiketli | 0.84728 (178/276) | 0.88275 (190/289) |
| Lateral OA, medial OA pozitif | 0.75602 (124/67) | 0.79744 (130/60) |
| Lateral OA, medial OA negatif | 0.82479 (23/201) | 0.83440 (21/224) |
| Sinovit, tüm etiketli | 0.76610 (118/10) | 0.93165 (109/20) |
| Sinovit, efüzyon pozitif | 0.71847 (111/4) | 0.77083 (108/4) |
| Sinovit, efüzyon negatif | 0.70833 (4/6) | 1.00000 (1/16) |

Maskeli / belirsiz etiketler negatif yapılmadı. Alt grup etiketleri eksik olabildiği için alt grup sayıları tüm etiketli sayısına eşit olmayabilir. Mevcut checkpoint eğitim grupları dış fold ve gold gruplarından ayrıdır. Bunlar zayıf rapor etiketleriyle uyum ölçümleridir; görüntü gerçeği değildir. K1 ve referans farklı modeller olduğu için bu alt grup sonuçları referansın neden hata yaptığını tek başına göstermez.

Lateral OA'daki medial-OA-pozitif grup düşüşü iki dış fold'da görülüyor. Kompartman ayrımını test etmek için bir hipotez sağlar; anatomik karışıklık tanısı değildir. Sinovitte efüzyon-pozitif grup ölçümü yalnızca dört negatif/fold'a dayanıyor. Fold1 efüzyon-negatif grubunun 1.0 AUC'si de yalnızca bir pozitif içeriyor; başarı kanıtı olarak kullanılmamalı.

## Deney OA-COND-01 — kompartman ayrımını test et

Hipotez: lateral OA öğreniminde medial OA ile ortak görülmeye aşırı bağımlılığı azaltmak, özellikle medial OA pozitif olguların içindeki lateral OA ayrımını iyileştirebilir.

Kontrol ve aday aynı fold-matched K1/K1F1 frozen MRI encoder, aynı altı seri slotu, aynı görüntü örneklemesi, aynı iki hedefli medial/lateral OA başlığı, aynı sabit 15 epoch ve aynı update sayısını kullanacak. Tek fark örneklemedir: kontrol doğal örnekleme; aday yalnızca eğitim etiketleriyle bilinen medial/lateral OA dört grubunda frekansın karekök tersiyle ağırlıklandırılmış örnekleme. Ağırlıklar fold eğitim verisinde hesaplanıp ortalaması 1 olacak şekilde normalize edilir ve 3 ile sınırlandırılır. Eksik companion etiketi olan örnekler kaybolmaz; ağırlıkları 1'dir. Belirsiz etiketler loss maskesi olarak kalır; gold ve holdout grupları eğitimden çıkarılır. Etiketler inference girdisi olmayacak. Anatomik ROI veya lokalizasyon başarısı iddia edilmeyecek.

Önceden sabitlenmiş birincil ölçüm lateral OA dış-fold AUC farkıdır: iki fold'da pozitif fark ve duplicate-group paired-bootstrap ortalama fark CI alt sınırı >0. İkincil ölçüm medial OA pozitif grupta lateral OA farkıdır; iki fold'da da negatif olmamalı. Medial OA değişimi ayrıca raporlanır. Gold epoch, ağırlık veya örnekleme katsayısı seçmek için kullanılmaz. Bu kapı geçilse bile mevcut 0.958 ensemble katkısı ayrı değerlendirilir. GPU builder scripts/prepare_oa_conditional.py ile uygulandı; kontrol/aday eşit update ve leakage kontrolleri doğrulandı.

## Deney SYN-DISCORD-01 — önce efüzyon/sinovit ayrımının ölçümünü güçlendir

Hipotez: efüzyon varlığını sinovit göstergesi olarak kullanma eğilimi olabilir. Mevcut sayısal sonuçlar bunu kanıtlamıyor.

Önce mevcut gold'u büyütme veya aynı 58 örnekte ağırlık ayarlama yerine, yeni model seçiminden önce ayrı bir değerlendirme protokolü oluşturulmalı. Özellikle efüzyon pozitif/sinovit negatif ve efüzyon negatif/sinovit pozitif olgular uzman görüntü değerlendirmesiyle doğrulanmalı; yalnızca LLM rapor çıkarımı görüntü doğrulaması sayılmaz. Eğitimden ayrılan ve duplicate-group sızıntısı olmayan sabit bir grup gerekir. Mevcut modellerin gördüğü eğitim görüntüleri, onların bağımsız holdout'u diye sunulamaz. Gerekirse yeni split ve yeniden eğitim gerekir.

Efüzyon durumuna göre iki alt grubun her birinde en az 20 sinovit pozitif ve 20 sinovit negatif gözlem, çalıştırma öncesi asgari uygulanabilirlik şartı olarak önerilir; güç analizi veya yeterli güven garantisi değildir. Mevcut dış fold'lar bu şartı geçmiyor. Bu yüzden şu an bu hipotez için eğitim başlatmak yerine değerlendirme grubu tasarımı önceliklidir.

Ölçüm şartı karşılandığında aynı altı slotlu MRI encoder ve masked sinovit/efüzyon iki-hedefli başlıkla kontrol/aday karşılaştırılır. Tek değişken eğitimde bilinen iki etiketin dört grubuna göre sınırlı dengeli örneklemedir; ayrı gold ağırlık ayarı yapılmaz. İki dış fold'da sinovit farkı pozitif, paired CI alt sınırı >0 ve efüzyon-pozitif alt grupta negatif olmayan fark şartı aranır. Test etiketleri inference girdisi olmaz. Sonuç yalnızca daha sonraki ensemble katkısı testine geçiş sağlayabilir.

## Tekrar üretim ve dosyalar

`python scripts/diagnose_oa_synovitis.py`

- `artifacts/research/oa-synovitis-diagnostic-20261020/diagnostic.json`: sayılar, bileşen karşılaştırmaları, alt gruplar, kaynak hash'leri.
- `artifacts/research/oa-synovitis-diagnostic-20261020/case-review-local.csv`: yerel vaka inceleme sırası; tıbbi tanı veya kesin yanlış etiket listesi değildir.
- `tests/test_ranking_diagnostic.py`: eşit puanlar, sıralama değişmezliği, maskeler, sonlu skorlar ve UID hizalama kontrolleri.

Karar: aynı head üzerinde ağırlık taraması yapılmaz. Lateral OA kontrollü eğitimi Kaggle’da çalışıyor; sinovit eğitim deneyi mevcut değerlendirme sayılarıyla desteklenmiyor. Hiçbir aday terfi ettirilmedi veya submit edilmedi; kayıtlı public referans 0.958.

## Uygulama güncellemesi

OA-COND-01 artık notebook olarak uygulanıp Kaggle v1'e gönderildi: `bulsahkecici/rsna-oa-conditional-twofold-20261020`. Her iki kolda da replacement sampling kullanılarak tek fark olasılıklar olarak tutuldu. Eğitim hedefleri mevcut numeric-training paketindeki gözlenen ikili OA etiketleridir; eksikler maskelenir. 2301/2292 eğitim örneği, 15 epoch, fold başına her kolda1080 güncelleme. Başlatma kaydı artifacts/kaggle/rsna-oa-conditional-twofold-20261020/push-receipt.json içinde. Sinovit için değerlendirme sınırlaması devam ediyor; ayrı eğitim başlatılmadı.


## Sonuç güncellemesi — 21 Ekim 2026

OA-COND-01 Kaggle v1 COMPLETE; sayısal çıktılar indirildi ve reviewer doğrulamaları geçti. Lateral OA AUC doğal/koşullu: fold0 0.827105/0.804226; fold1 0.859284/0.830007. Koşullu eksi doğal farkın grup-bootstrap %95 aralığı [-0.041320, -0.008533]. Medial-OA-pozitif alt grupta da iki fold geriliyor. Karar REJECT_CONDITIONAL_SAMPLING; yeni karışım veya submission yapılmadı. Önceki RUNNING ifadeleri tarihsel durumu anlatır.

Provenans düzeltmesi: yukarıdaki CoAtNet gold eğitim örtüşmesi cümlesi kesin kanıt olarak kullanılmamalı. Güncel yazar açıklaması 746792, gold58 eğitimden hariç, seçimde yeniden kullanıldı diyor; bu araştırma exact checkpoint eğitim UID listesini bağımsız doğrulamadı. Bağımsız gold değerlendirme iddiası her durumda yok. Ayrıntılar docs/research/STRATEGY_20261021.md.
