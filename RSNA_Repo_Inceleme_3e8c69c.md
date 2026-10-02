# RSNA repo üçüncü inceleme

Tarih: 3 Ekim 2026, Türkiye saati.

Repo: https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection

İncelenen commit: `3e8c69cf119abcc771a1543764f61b97d76960e5`.
Karşılaştırma: `833da87e3e7f1e5581d3fc67d67f0e7771989705`.

**Karar: Önceki eğitim döngüsü hatalarının önemli bir bölümü kapanmış. Worker artık gerçek dosya yükleme, trainer ve kayıtlı checkpoint'ten değerlendirme fonksiyonlarını çağırıyor. Ancak etiket üreticisi ile eğitim loader'ının veri sözleşmesi uyuşmuyor; belgelenen CLI/Colab akışı tamamlanmıyor. Uzun GPU kampanyasına hazır değil. Öncelik yeni özellik değil, aynı run_id ile çalışan küçük bir pilotun bağlantılarını tamamlamak.**

31 dosyada 2174 ekleme / 245 silme incelendi; bunlara önceki inceleme raporunun repoya eklenmesi de dahil. Ayrı checkout kullanıldı. Repo kaynakları değiştirilmedi; GitHub'a yazılmadı, gerçek Kaggle gönderimi veya ücretli GPU eğitimi yapılmadı. LM Studio, kullanıcının Mac'i ve Colab oturumu kullanılmadı. Kanıtlar örnek verilerden üretildi; gerçek MRI pilotu değildir.

## Kapanan ve kısmen kapanan bulgular

| Önceki bulgu | Güncel durum |
| --- | --- |
| Test gerçek folds.csv üzerine `REAL,0` yazıyordu | Yazma kaldırılmış; test roots override ve dosya hash kontrolü eklenmiş. Dokümanda yerel fold'un yeniden üretildiği yazıyor; kullanıcının yerel dosyasını bu incelemede görmedim. |
| Worker trainer çağırmıyordu | Artık çağırıyor. Ham items + eski checkpoint başarılı eğitim sayılmıyor; production TinyEncoder reddediliyor. |
| Son kısmi accumulation uygulanmıyordu | Düzelmiş; regression testi geçti. |
| Microbatch input'u bölmüyordu | Gerçek slicing eklenmiş; regression testi geçti. Bölme effective_batch sınırını aşabildiği için genel durum tamamlanmamış. |
| Partial-unfreeze resume optimizer grup hatası | Doğrudan trainer testinde düzelmiş; küçük microbatch ile devam testi geçti. Worker'da resume henüz açılmamış. |
| Cache çözünürlük değişimini görmüyordu | Boyut/centers/slots/preprocessing ve shard hash kontrolü eklenmiş. Ham DICOM kimliği hâlâ izlenmiyor. |
| Manifest var ama shard yokken CACHE_READY | Eksik shard artık engelleniyor. |
| Gerçek değerlendirme yalnız BLOCKED dönüyordu | Kayıtlı head dahil model reload, tahmin dosyası ve gold/weak metrik fonksiyonları eklenmiş. Etiket/gold merge bağlantısı eksik. |
| PASS string'i tek başına submit açıyordu | Run/kernel/version/checkpoint/package hash alanları isteniyor. Audit üreticisi bu belgeyi henüz üretemiyor. |
| Belirsiz receipt yeniden gönderilebiliyordu | SUBMITTING/UNKNOWN kalıcı kaydı ve unresolved attempt engeli eklenmiş. Gerçek reconciliation yok. |
| Kaggle kaynak/checkpoint mount sözleşmesi yoktu | `asset/src`, `asset/assets/checkpoint.pt` ve bootstrap eklenmiş. Offline dependencies ve cache notebook bağlantısı tamamlanmamış. |
| Pause talebi sürekli tekrar durduruyordu | Dosya talebinin consume/ack/resume davranışı testi geçti. |

## Doğrulama

Python 3.12.14, CPU Torch 2.14.1+cpu; CPU thread sayıları 1.

| Kontrol | Sonuç |
| --- | --- |
| Normal pytest | 60 geçti, 3 hata. Üç hata bu çalışma ortamında süreç ölçümünün `psutil.NoSuchProcess(pid=2)` vermesinden kaynaklandı. |
| Yalnız `runtime.preflight.snapshot` açık mock ile pytest | **63 geçti, 0 atlandı.** Üç uyarı fixture DICOM UID formatlarıyla ilgili. |
| Ruff: src/tests/scripts | Tüm kontroller geçti. |
| git diff --check | Başarılı. |
| Yeni adversarial kontroller | Aşağıdaki etiket, weak overlap, effective batch, job taşıma, audit, DICOM reuse ve checkpoint doğrulama sorunları yeniden üretildi. |
| Gerçek MRI / resmi ağırlık parity / CUDA OOM / canlı Colab / Kaggle offline runtime | Çalıştırılmadı. |

## P0 — Etiket komutunun çıktısı eğitim loader'ına uygun değil

Kaynak: `labels/extract.py:235–247`, `pipeline.py:260–284`, `runtime/train_payload.py:58–100,130–137,178–183`.

Gerçek extractor şu biçimde dönüyor; pipeline bunu JSONL'a aynen yazıyor:

```json
{"status":"ok","payload":{"StudyInstanceUID":"1.2.10","targets":{"ACL":{"state":"criterion_positive","evidence_span":"tear","criterion_mapping":"positive"}}},"resume_key":"..."}
```

`load_label_table` ise UID'yi üst seviyede `StudyInstanceUID` veya `uid`, etiketleri de üst seviyede sayısal `targets` veya `y` olarak bekliyor. Gerçek çıktıda UID payload içinde olduğundan kayıt atlanıyor. State sözlüğü sayısal hedefe de çevrilmiyor. Mevcut `states_to_training_arrays` ve `gold_to_training` yardımcıları bu üretim yolunda kullanılmıyor.

**Doğrudan kanıt:** canonical extractor biçimindeki dosya → loader entry sayısı **0**. Aynı dosyayla iki training study yükleniyor; toplam label mask **0.0**. Allowlist'teki test ağırlığıyla worker **exit_reason=ok, kind=real, steps=2** dönüyor. Gold AUC **null**, tanımlı sınıf sayısı **0**. Bu deney random fixture ağırlığını resmi pretrained olarak sunmaz; amacı etiket kaybının başarı kontrolünden geçtiğini göstermek.

Trainer sıfır denetimli ağırlık toplamında bile optimizer/scheduler step sayıyor. AdamW weight decay checkpoint'i değiştirebilir; yeni hash ve step>0, anlamlı supervised training kanıtı değildir.

Gold etiketler de train.csv'den loader'a bağlanmamış. Job yalnız folds/labels/cache/weights okuyor; folds CSV gold hedef değerlerini içermiyor. Dolayısıyla `gold_split=eval` seçmek gerçek gold target kullanıldığını garanti etmiyor. Örnek test, hazır sayısal hedefleri JSONL'a kendisi koyduğu için bu bağlantı açığını yakalamıyor.

Çözüm: extraction envelope → doğrulanmış canonical training table; numeric y/mask/weight, source ve ontology kimliği açık olsun. Gold hedefler metadata'dan gelsin ve LLM yerine öncelikli olsun; gold eval görüntüleri eğitimden dışlansın. Etiketsiz/mask toplamı sıfır gruplar optimizer update olarak sayılmasın. Başarı raporu gerçek denetimli örnek ve hedef sayısını içersin. Entegrasyon testi gerçek `stage_labels` çıktısını worker'a vermeli.

## P0 — Belgelenen CLI → Colab → evaluate zinciri hâlâ kapanmıyor

Kaynak: `cli.py:76–87,181–212,259–261`; `pipeline.py:224–228,336–358,419–478,640–669`; `runtime/worker.py:88–103,201–204`; `runtime/jobs.py:83–103`.

- `rsna labels pilot --live` ayrı bir run yaratıyor. Ardından `rsna pipeline run --profile pilot` yeni run yaratıp `stage_labels` fonksiyonunu `live=False` ile çağırıyor. Önceden üretilen label artifact'ini bağlamıyor; `labels_require_live_lmstudio` ile duruyor. **Stage çağrısı doğrudan NEEDS_RUNTIME üretti.**
- Runtime aşaması train/job hazırlığından önce. Varsayılan Colab handoff NEEDS_RUNTIME olunca sequential runner duruyor; dokümandaki worker için beklenen training job o akışta henüz üretilmiyor.
- `rsna experiment run --config` config'i doğrulasa da kaynak URI'leri olmayan bir job yazıyor. `stage_train` ise profile/experiment ayarlarını yüklemek yerine epochs=1, effective_batch=1, microbatch=1, seed=0 sabit değerlerini veriyor. Campaign/pilot ayarları eğitim davranışına uygulanmıyor.
- Worker CLI transport'a sonuç yazıyor; aynı run'ın registry checkpoint/metrics alanlarını güncellemiyor. `runtime/ingest.py` yardımcı fonksiyonu aktif CLI/pipeline yolunda çağrılmıyor. Bu nedenle remote worker'ın başarılı çıktısı `rsna evaluate --run-id` için otomatik checkpoint kaydı olmuyor.
- `stage_train`, kaynaklar bulunduğunda worker'ı mevcut makinede doğrudan çağırıyor. GPU gereksinimi/CPU pilot bütçesi bu çağrıda uygulanmıyor.

**Taşıma kanıtı:** job klasörü yeni yere kopyalanıp eski klasör kaldırıldı, `RSNA_INPUT_ROOT` verildi. `load_job_bundle` eski absolute `bundle_path` değerini bıraktı; validation **bundle_missing** döndü. Kaynaklar için relative lookup var; manifest ve `checkpoint_out` için aynı rebasing yok.

Trainer'da resume düzelmiş olsa da worker `resume=True` ile çağırmıyor ve job'dan resume checkpoint'i almıyor. Kesilen remote training'in iş düzeyinde devam ettiği kanıtlanmış değil.

Çözüm: tek run_id; reusable/explicit label artifact; job hazırlığını runtime handoff'tan önce yap; taşınabilir bundle/input/output sözleşmesi; gerçek config uygulama; fenced result ingest → registry checkpoint/metrics; worker resume; gerektiğinde sınırlı CPU pilotu ve campaign için açık GPU gate. Runbook komutlarını bu akışın kendisiyle test et.

## P1 — Audit üreticisi yeni kimlik sözleşmesini karşılamıyor

Kaynak: `pipeline.py:557–635`, `agents/auditor.py`, `gates.py:33–42`, `submission/package.py:70–82,165–171`, `cli.py:249–261`.

`stage_audit` production'da ontology alanını None, leakage/cache/resources/external/offline alanlarını BLOCKED ve identity_errors alanını daima `not_live_verified` veriyor. Bu nedenle gerekli kaynaklar mevcut olsa bile bu fonksiyonun gerçek evidence ile PASS üretme yolu yok. Auditor sonucu yalnız overall/gates/targets/read_from alanlarını döndürüyor; run_id, checkpoint/package hash, kernel/version değerlerini sonuç belgesinde korumuyor.

**Doğrudan kanıt:** real metrics ve train hash bulunan state → audit FAIL; çıktı belgesinde run_id ve package hash yok.

Stage sırası audit → package; package gate henüz üretilmemiş paketin hash'ini audit'te istiyor. Package her çağrıda klasörü silip yeniden tar.gz üretiyor; aynı içerikle iki çağrıda tar hash'i farklı çıktı. Bu nedenle önceki paket için alınan onay da otomatik korunmuyor.

Ek olarak `rsna package --run-id` registry'deki checkpoint'i okumadan `package_run(run_id)` çağırıyor. Bu komut checkpoint'siz paket oluşturur; aynı run paket klasörü varsa yeniden yaratırken eski içeriği siler. CLI audit de yeni audit çalıştırmak yerine yalnız var olan dosyayı okur.

Çözüm: candidate package → package verification/offline smoke → evidence audit → exact frozen package/kernel version için READY. Audit kimlikleri çıktı belgesinde korunsun. Paket onaydan sonra yeniden üretilmesin veya deterministik içerik kimliği kullanılsın. CLI package registry'den doğru checkpoint/config almalı.

## P1 — Eğitim doğruluğunda kalan iki açık

1. `training/loop.py:258–292`: microbatch take değeri accumulation grubunda kalan kapasiteyi dikkate almıyor. **6 study, effective_batch=4, microbatch=3 → 1 actual step, 2 planned update.** Model tüm 6 study'yi tek grupta işliyor; microbatch değişimi optimizer group ve scheduler davranışını değiştirebilir. `take` kalan effective batch kapasitesiyle sınırlandırılmalı; >effective batch ve tam bölünmeyen değerler de test edilmeli.
2. `runtime/train_payload.py:167–189`: job `train_uids` override'ı yalnız gold-eval exclusion üzerinden denetleniyor; fold-0 weak holdout'tan çıkarılmıyor. **train_uids=['1.2.12'] örneğinde loader kabul etti ve aynı UID weak_uids içinde kaldı.** Train/weak/gold kesişimleri ve duplicate group sınırları çalışma başlamadan doğrulanmalı.

## P1 — Cache kaynak kimliği ham DICOM'u kapsamıyor

Kaynak: `data/cache_build.py:130–163,212–225,286`.

Reuse kontrolü artık shard hash ve preprocessing'i doğruluyor; fakat source_hashes yalnız series CSV hash'i. DICOM dosyaları veya dicom_root değişince kaynak kimliği değişmiyor; reuse ham dosyaları okumadan gerçekleşiyor.

**Doğrudan kanıt:** gerçek formatta fixture DICOM PixelData değiştirildi, dosya hash'i değişti; aynı CSV ve parametrelerle resume **resumed=True**, önceki shard hash'i aynen kaldı.

Çözüm: immutable veri sürümü + seçilmiş SOP UID/file identity envanteri veya güvenilir kaynak digest'i; kaynak değişikliği yeni shard üretmeli. READY/worker kontrolleri dtype, preprocessing ve requested UID kapsamını da beklenen sözleşmeyle karşılaştırmalı.

## P1 — Campaign RAM ve job bütünlüğü

`prepare_training` bütün train shard'larını `np.load` ile açıp float32'ye çeviriyor ve batches listesinde tutuyor. Lazy/memmap dataset yolu yok. Varsayılan 3 slot × 3 center × 3 kanal × 224² × float32 yalnız görüntüler için **1000 study'de yaklaşık 5.05 GiB** tutar. Yaklaşık 3500 study, geçici kopyalar/model/optimizer hariç yaklaşık 17.7 GiB. Bu CPU örnek pilotunda sorun değildir; campaign ve 36GB Mac/Colab host RAM için bütçeli streaming gerekir.

Job hash'i input_manifest'i kapsıyor, fakat worker job'daki train/resources/train_uids/encoder/checkpoint_out değerlerini manifest ile karşılaştırmıyor. **Job train alanı epochs=999 olarak değiştirildi, validate_job errors=[] kaldı.** `job_sha256` doğrulanmıyor; disk job JSON'u bu alan eklenmeden önce yazılıyor. Config dosyası kaynak olarak job'da bulunsa bile prepare_training required listesinde değil ve içeriği yüklenmiyor. Resume input identity pretrained weights ve training ayarlarını tam kapsamıyor.

Çözüm: immutable hashed job specification; doğrulanmış config'in gerçek trainer argümanlarına dönüşmesi; weights/config/fold/cache/label kimliği; lazy shard yükleme ve RAM sınırı.

## P1/P2 — Paket ve süre kanıtı kısmi

- Inference bootstrap ve device seçimi eklenmiş. Offline wheels henüz vendored değil; verify_package bunu açıkça engelliyor. Ancak yalnız wheels klasörünün varlığını kontrol etmek yeterli değil: wheel manifest/pin/hash ve offline install/import testi gerekli. Mevcut subprocess smoke aynı venv'in kurulu bağımlılıklarını kullanıyor; bağımsız temiz ortam kanıtı değil.
- `_checkpoint_errors` model'in dict olmasıyla yetiniyor. **encoder_name doğru, step=1, model={} → errors=[]** çıktı. Gerçek model instantiate + strict state load + fixture forward gerekir; pretrained/source provenance checkpoint'e taşınmalı.
- Cache kernel entry hâlâ doğrudan rsna_knee import ediyor, dataset_sources boş; inference asset bootstrap düzeltmesi cache kernel'e uygulanmamış. `prepare_cache_kernel` varsayılan limit boşsa full cache dener; küçük pilot sınırı açık olmalı.
- Inference stopwatch, build_cache ve model yüklemesinden sonra başlıyor. DICOM decode/cache/model startup sürelerini hariç tutan projeksiyon toplam 9 saat bütçesi kanıtı olamaz. `guaranteed_under_9h` yerine toplam ölçüm + marj + doğrulanma durumu kullanılmalı.
- Submit command dict'e submission_file eklenebilir, ancak execute yolunda provider.submit_kernel'e taşınmıyor. Kota kontrolü hâlâ çağrılmıyor. UNKNOWN tekrarını engellemek düzelmiş; reconciliation provider'ında list_submissions yok. Varsayımsal history içindeki ilk UUID'yi almak kernel/version/attempt eşleştirmesi değildir.
- Controller stage resume giriş hash'lerini kontrol ediyor; complete stage'in output dosyalarının varlığını/hash'ini doğrulamıyor. Değişen config train dependency listesine dahil değil. Silinmiş output ve config değişikliği için entegrasyon testleri gerekli.
- Resmi pretrained SHA listesi boş; yerel allowlist destekleniyor ve pretrained=False ayrımı doğru. Resmi ağırlıklarla yükleme/bağımsız forward parity canlı doğrulanmış sayılmamalı.

## Cursor'a verilecek sonraki prompt

```text
AGENTS.md ve RSNA_Repo_Inceleme_3e8c69c.md dosyalarını oku.
Bu inceleme 3e8c69c commit'ine aittir. Güncel kodda kapanan bulguları tekrar yapma.
Mevcut microbatch slicing, partial flush, unfreeze restore, roots isolation,
held-out checkpoint reload ve UNKNOWN submit engelini koru.

Bu tur hedefi aynı run_id ile çalışan küçük pilotun bütün bağlantılarıdır.
Yeni bir framework, agent katmanı veya yalnız guard ekleyerek tamamlandı deme.

1. Gerçek extractor envelope çıktısını canonical training labels'a dönüştür.
   payload UID ve state kayıtlarını doğrula; states_to_training_arrays kullan.
   Gold target'ları train metadata'dan oku; gold önceliği ve -1 mask korunmalı.
   Gold eval görüntüleri image training'e girmesin. Mask/weight toplamı sıfır
   olan gruplar optimizer step ve başarılı eğitim sayılmasın.
   Gerçek stage_labels formatını worker'a veren entegrasyon testi yaz.
2. Tek run_id ile labels/cache/config/weights seçimini ve artifact reuse'u kur.
   Job, Colab handoff'tan önce üretilebilsin. experiment run gerçek kaynakları
   ve config ayarlarını kullansın. Profile/epochs/seed/effective batch sabit
   değerlerle geçersiz kılınmasın. Runbook'taki komut sırasını gerçekten test et.
3. Bundle manifest/input/output yollarını taşıma sonrası çöz. RSNA_INPUT_ROOT
   tek başına yeterliyse bunu relocation testiyle göster. Mac absolute output
   yolu Colab'da kullanılmasın. Fenced worker result ingest, checkpoint ve
   metrics'i controller registry'ye bağlasın; CLI evaluate aynı run'ı okuyabilsin.
   Worker resume checkpoint/identity/phase desteğini trainer'a iletsin.
4. Take boyutunu effective_batch grubunda kalan kapasiteyle sınırla.
   6 study/effective=4/micro=3 için 2 update ve microbatch bağımsızlığı test et.
   Train/weak/gold UID ve duplicate group kesişimlerini açıkça reddet.
5. Cache kaynak identity'sine ham DICOM veya immutable dataset revision ekle.
   PixelData veya kaynak kökü değişince eski shard reuse edilmesin.
   Train loader lazy/memmap olsun; tüm dataset float32 olarak RAM'e yüklenmesin.
   Pilot/campaign RAM ve GPU gereksinimleri gerçek execution yolunda uygulansın.
6. Hash ile korunan tek job spec kur; worker resources/config/train/train_uids
   alanlarını doğrulanan spec'ten alsın. Train ayarı değişikliği validation'da
   görülsün. Resume identity ağırlıklar ve training config'i kapsasın.
7. Candidate package -> verification/offline smoke -> evidence audit ->
   frozen exact package/kernel version sırasını kur. Audit identity alanlarını
   sonuçta koru. CLI package kayıtlı checkpoint'i alsın; CLI audit evidence'i
   yeniden doğrulasın. Package onaydan sonra sessizce yeniden üretilmesin.
   Model={} checkpoint'i strict load + forward doğrulamasında reddedilsin.
8. Offline dependency manifest ve kurulum, cache kernel bootstrap, toplam
   decode+model+inference timing ve stage output doğrulamasını tamamla.
   Receipt UNKNOWN engelini koru; reconciliation yoksa açık BLOCKED kalsın.

Önce bu rapordaki doğrudan yeniden üretimleri regression test yap.
63 mevcut testin geçmesi tek bitiş ölçütü değildir. CLI tabanlı örnek pilot:
extraction-format -> normalized labels+gold -> job -> relocated worker ->
supervised optimizer update -> interrupted resume -> result ingest ->
saved checkpoint evaluation -> verified offline candidate package.
Bu koşuda eğitim ve değerlendirme kümeleri ayrık, supervised count pozitif,
aynı run_id ve hashes bütün adımlarda tutarlı olmalı. Test fixture ağırlığı
resmi pretrained veya gerçek MRI pilotu diye raporlanmasın.

Ruff ve Torch CPU testlerini çalıştır. Canlı LM Studio/official weights/CUDA/
MRI/Kaggle kontrolü yapılmadıysa açık yaz. Credential eksikliği bağımsız
bağlantıları yarım bırakma gerekçesi olmasın. Ücretli campaign veya gerçek
submission başlatma; gerçek küçük MRI pilotu için çalışır komutları hazırla.
```

İlk kabul ölçütü leaderboard AUC değildir: canonical gerçek etiket formatı,
pozitif supervision sayısı, eğitimden ayrı gold hedefler, taşınabilir job,
resume edilen worker sonucu ve aynı run'a kaydedilen doğrulanmış checkpoint.
