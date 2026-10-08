# PC kapandıktan sonra kampanyayı sürdürme

Her tamamlanan etiket `data/labels/label_cache.sqlite` dosyasına ayrı SQLite
işlemiyle kaydedilir. Devam anahtarı çalışma kimliği, rapor hash'i, model ve
revizyon, prompt hash'i, ontoloji ve kanıt politikasını içerir. Aynı girdilerle
`--resume` geçerli kayıtları yeniden LLM'ye göndermez. Karantinadaki kayıtlar
tekrar denenir. Kapanma anında henüz tamamlanmamış tek istek yeniden işlenebilir.

PC açıldığında işlem kendiliğinden başlamaz. Önce LM Studio'da aynı
`qwen/qwen3.8-27b` modelini yükleyip yerel sunucuyu `127.0.0.1:1234` üzerinde açın.

Proje dizinindeki birinci terminalde:

```bash
.venv/bin/rsna labels run --resume --live --development-only --run-id labels-full-20261003-v2 --progress-file state/campaigns/campaign-20261003-v2/local-full-label-progress.json
```

İkinci terminalde:

```bash
caffeinate -i .venv/bin/rsna campaign --run-id campaign-20261003-v2 --owner bulsahkecici --pilot-run-id labels-dev-20261003-v2-100 --labels-run-id labels-full-20261003-v2
```

Aynı işler hâlâ çalışıyorsa ikinci kopyalarını başlatmayın. Denetleyici kendi
kilidini tutar. Girdi kimliği değişirse mevcut run_id ile devam etmeyi reddeder;
resume sırasında model/prompt değiştirerek eski etiket sürümüyle karıştırmayın.

Etiketleme daha önce tamamlandıysa birinci terminal komutunu tekrar çalıştırmak
gerekmez; denetleyici final registry kaydını ve dosya hash'ini doğrular.
Kaggle'da zaten başlatılmış bir iş, Mac kapansa da Kaggle üzerinde devam edebilir;
denetleyici tekrar açıldığında kayıtlı işin durumunu sorgular. Belirsiz bir push
sonucunu otomatik tekrar göndermez.

2026-10-03 kontrolü: SQLite `quick_check=ok`, synchronous=FULL.
Tutarlı yedek `data/labels/backups/label_cache-20261003-225735.sqlite` oluşturuldu;
yedek de `quick_check=ok` sonucunu verdi. Ham rapor/kanıt içerebilen bu SQLite
dosyaları yalnızca yerelde tutulur.

Kamuya açık etiket kaynakları (dosya listeleri Kaggle CLI ile doğrulandı):

- [Steven Lee Hans — RSNA Knee LLM Report Labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels): `llm_labels_full.csv`, `llm_labels_v2.csv`, `llm_labels_v4_blend.csv`.
- [Yunus Gümüşsoy — RSNA Knee LLM Report Labels](https://www.kaggle.com/datasets/yunusgmsoy/rsna-knee-llm-report-labels): `yunus_llm_labels.csv`.

Bu kaynaklar henüz bu kampanyanın eğitim girdisi olarak kullanılmadı. Kamuya
açık olmak klinik doğruluk veya bağımsız değerlendirme garantisi değildir.
Sürüm/lisans, UID kapsamı, hedef tanımları, belirsiz/eksik maskeleri ve gold-eval
dışlama kontrolü yapılmadan mevcut etiket sürümünün yerine geçirilmemelidir.
