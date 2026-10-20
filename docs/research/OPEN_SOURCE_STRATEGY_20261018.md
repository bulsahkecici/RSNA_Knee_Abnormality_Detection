# Açık kaynak araştırması ve son dört günün stratejisi — 18 Ekim 2026

Karar: eski 0.929 reader çevresinde yeni backbone denemelerine devam etmek yerine, açık CoAtNet + R6 tabanını sabitleyip yeniden üretmek birinci öncelik. Açık taban yaklaşık 0.958; güncel public lider 0.969. Birincilik sağlayacak ek kazanç henüz gösterilmiş değil. Public sıralama private sonuç değildir.

## Kanıt ve kapsam

Resmî Kaggle sayfaları, host 733343, ilk 20 leaderboard satırı, 100 public notebook kaydı ve dönen 20 tartışma kaydı alındı. Beş notebook kaynak olarak indirildi ve statik okundu; çalıştırılmadı. Üç GitHub README ve repo metadata dosyası indirildi. aakashkavuru101/RSNA arama sonucunda görünse de doğrudan API indirmesi 404 verdi: yürütülebilir aday olarak doğrulanmadı.

Ham araştırma: `artifacts/research/open-source-20261018/`; dosya parmak izleri `sha256.json`. MRI veya yeni model ağırlıkları bu araştırma için Mac'e indirilmedi. Eğitim ya da submission başlatılmadı.

## Mevcut deneylerimiz

- K1: public 0.928, receipt 57232441. Eski public reader: 0.929, receipt 57178710.
- K2R320 weak doğrulamada K1'i geçmedi.
- K3EV2: weak AUC 0.847756 vs K1 0.878159; eşleştirilmiş fark -0.030402, %95 CI [-0.044436, -0.017641]. Gold 0.874717 vs 0.903113. Sabit reader karışımı da kazandırmadı. Bu sonuç tüm EfficientNet ailesini reddetmez; mevcut pakete yeni bütçe ayırmayı desteklemez.

## Kaynakların değerlendirilmesi

| Kaynak | Bulgular / kanıt düzeyi | Kullanım kararı |
|---|---|---|
| [goodpjw CoAtNet + R6](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) | Kaggle indekslenmiş skor alanı 0.958; v27 kaynak indirildi. R6 ConvNeXt dört fold + nartaa CoAtNet384/224 + düşük eski-reader katkısı. | İlk yeniden üretim adayı. Henüz bizim hesabımızda ölçülmedi. |
| [Heliosli yeni paket](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586) | Başlık 0.9586 diyor; kaynak indirildi. Kesin skor alanı bağımsız doğrulanmadı. Aynı temel aileler, farklı sınıf ağırlıkları. | Bağımsız yeni model gibi ensemble'a eklenmemeli. |
| [nartaa CoAtNet açıklaması](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792) | Yazar tek CoAtNet için 0.949 bildiriyor. Native crop, anatomik mirror ve 2399 OAI dizinden kısmi hedeflerle eğitim; birlikte değişen reçeteler, izole ablasyon değil. | Güçlü tamamlayıcı modelin kaynağı. |
| [nartaa S6 açıklaması](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359) | Yazar S6 all için 0.950 bildiriyor; farklı mimari, anatomi çapalı 100 mm crop ve bulgu bazlı attention. | R6 ile benzerlik ve ek katkı ölçülmeden eklenmez. |
| [mathischmp](https://github.com/mathischmp/challenge_rsna) | README kendi CoAtNet sonucunu 0.920 olarak bildiriyor; geometri ve MIL uygulaması yararlı. | Skor sıçraması için öncelikli ağırlık kaynağı değil. |
| [NTejas-1](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection) | README 0.940 tabana iki kol ekleyerek 0.941 bildiriyor; etiket düzeltme deneyi gerilemiş. Eski açık skor tavanı açıklaması artık güncel değil. | Kör label repair ve benzer model eğitmenin maliyetine karşı örnek. |
| [anhquan1111](https://github.com/anhquan1111/rsna-knee-abnormality-detection) | README son public 0.771; küçük gold kümesinde tekrar seçim, ortak attention dilimleri ve duplicate sızıntısı tartışılıyor. | Kazanan checkpoint değil, değerlendirme hatalarını denetleme kaynağı. |

GitHub sonuçları yazar beyanıdır; submission receipt'leri tarafımızca doğrulanmadı. Notebook isimleri tek başına skor kanıtı kabul edilmedi. Kaiwalya kaynağı da incelendi: aynı public model bileşenlerini kullanıyor; yeni bağımsız mimari kanıtı değil.

## İlk iş: güçlü paketi aynen yeniden üret

1. goodpjw v27 notebook ve veri sürümlerini sabitle. Kaynak SHA256: `a906df9328b6a53d7e0fe4e18962c655867c8617c54e6745a8af647033b2399c`.
2. R6 kaynağı mutasyona uğramış: dondurulmuş `rostigor/rsna-knee-r6-fe055406` veya yazarın byte-identical dediği `heliosli/rsna-knee-r6-frozen-fe055406-w1` için gerçek manifesti kontrol et. Beklenen R6 manifest SHA256: `6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648`. Etiketin R6 olması yeterli değil.
3. Ağırlık lisansları, eğitim kodu/provenansı, checkpoint dosyaları ve gold UID dışlamasını doğrula. Notebook lisansı tüm veri/ağırlık lisanslarının yerine geçmez. R6 gold dışlaması bu araştırmada doğrulanmadı.
4. Offline bağımlılıklar, sütun sırası, UID kapsamı, finite tahminler ve çalışma bütçesini kontrol et. Görünen üç test örneğindeki süre gizli yaklaşık 1300 çalışmanın süresini kanıtlamaz.
5. İlk submission değişmemiş referans olmalı. Sonucu receipt ile kaydet; 0.958 bizim skorumuş gibi raporlanmamalı.

## Birincilik için ek kazanç hipotezi

En makul hipotez, daha iyi anatomik yerelleştirme ile farklı ve daha doğru hedef bilgisini birleştirmek. Yeni encoder adı tek başına gerekçe değil. nartaa'nın OAI ile kısmen denetlenen reçetesi bunu araştırmaya değer kılıyor; nedensel +AUC garantisi sağlamıyor. Kalan sürede yeni dış veri erişim/cache hattını sıfırdan kurmak yerine yayımlanmış ağırlıklar başlangıç olmalı.

Yeni tabanın sınıf bazında hataları ölçüldükten sonra yalnız bir özgün uzman seç: örneğin PF/lateral OA veya synovitis için anatomik bölgeye odaklanan, güvenilir etiketli model. Bu sınıfların bizde en zayıf olduğu henüz ölçülmedi. Önce ölç, sonra seç. Bilinmeyen/raporda geçmeyen/-1 hedefler maskelenir; public reçetelerdeki prior doldurma otomatik aktarılmaz. Gerekli rapor incelemesi yalnız yerel `rsna labels`/LM Studio ile yapılır.

Doğrulama tuzağı: tüm train üzerinde eğitilmiş public modelin bizim weak holdout'ta skoru bağımsız OOF değildir. Eğitim kapsamı ve fold eşlemesi bilinmiyorsa bunu açıkça kontamine tanısal skor olarak ayır. Kendi yeni adayında duplicate grupları ve gold dışlaması korunmuş en az iki dış fold kullan; weak AUC rapor etiketleriyle uyumu ölçer, görüntü gerçeğini tek başına kanıtlamaz. Tekrar kullanılmış 58 gold ile onlarca sınıf ağırlığı seçme. Sabit bir birleşimin eşleştirilmiş kazancını ve sınıf gerilemelerini değerlendir.

R6/nartaa mirror işlemleri checkpoint'in eğitim geometrisine bağlıdır. Rastgele tüm planlarda yatay çevirme veya genel medial/lateral değişimi ekleme. Mevcut public kaynakta eski reader'ın büyük ağırlığı zarar vermiş; K1/K3 de yalnız çeşitlilik var diye eklenmemeli.

## Zaman ve durdurma kuralları

- 18–19 Ekim: sabit public tabanın lisans/provenans, offline ve ilk submission kontrolü.
- 19–21 Ekim: taban doğrulanırsa tek özgün aday ve sabit birleşim; iki dış fold yetişmiyorsa doğrulanmış kazanç iddiası yok.
- 22 Ekim: yeni mimari başlatma; tam çalışma provası, zaman payı, iki final adayının receipt ve seçimini kontrol et.
- Final: **23 Ekim 2026 02:59 Türkiye** (22 Ekim 23:59 UTC). Takım birleştirme süresi geçti. Günde 5 submission, iki final seçim; araştırma anında bugün 0/5 kullanılmıştı. GPU kotası bu taramada yenilenmedi.

İlerleme ölçütü önce kendi hesabımızda yeni tabanı tekrar üretmek, sonra ona gösterilebilir ek katkı sağlamak. Açık 0.958 paketi kopyalamak güncel 0.969 lideri geçme kanıtı değildir.

Resmî kaynaklar: [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard), [timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline), [code requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements), [rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules), [host ölçütleri](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343).
