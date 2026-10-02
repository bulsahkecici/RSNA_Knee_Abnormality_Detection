# RSNA Cursor repo incelemesi

İnceleme tarihi: 2 Ekim 2026. Repo: https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection

İncelenen `master` commit: `2911e7c17148ee81087dd8e77ea27af78f6f6b67`.

**Karar: Kullanılabilir bir başlangıç altyapısı kurulmuş; gerçek MRI eğitiminden Kaggle submission'a uzanan pipeline henüz tamamlanmamış. Önce aşağıdaki engeller çözülmeli, sonra küçük bir gerçek veri pilotu yapılmalı.**

GitHub'a commit, issue, PR veya submission gönderilmedi. Kod incelemesi ve testler ayrı bir yerel kopyada yapıldı. Repodaki “Mac'te live verified” ifadeleri Cursor'ın beyanıdır; bu incelemede kullanıcının Mac'i, LM Studio sunucusu, Kaggle oturumu veya Colab GPU'su doğrulanmadı.

## Kurulmuş parçalar

| Parça | İncelemede görülen durum |
| --- | --- |
| Python paket / Typer CLI / config | Kurulabiliyor; birçok komut mevcut. |
| SQLite registry / controller lock | Gerçek kod mevcut; tüm akışa bağlanması eksik. |
| Cursor agent, skill ve rule tanımları | İstenen klasörlerde mevcut. Bunlar çalışan kampanya motoru yerine geçmiyor. |
| Etiket ontolojisi / belirsiz etiket maskeleri | Temel ayrımlar ve testler var. |
| LM Studio istemcisi | Local URL kontrolü, model listesi, JSON istekleri, UID ve hash tabanlı extraction cache'i var. Canlı performans doğrulanmadı. |
| DICOM yardımcıları | IOP/IPP sıralaması, MONOCHROME1, spacing ile crop, uint8 cache yardımcıları mevcut. Study/series üretim akışına bağlanmamış. |
| AUC / CSV kontrolleri | Gerçek dizilerden AUC hesabı, tanımsız sınıflar ve sütun kontrolleri var. Eğitim checkpoint'inin değerlendirmesine bağlanmamış. |
| GPU politikası | Sanal saat ve scripted provider testleri var. Gerçek Colab otomatik tahsisi yok. |
| Kaggle adaptörü | CLI çağrıları mevcut; job üretimi, notebook paketleme ve sonuç uzlaştırma tamamlanmamış. |

## Doğrulama sonuçları

Python 3.12.14 ile izole ortam kuruldu; dev, dicom, langgraph ve kaggle extras kuruldu. Torch/CUDA kurulmadı ve gerçek GPU eğitimi yapılmadı.

| Kontrol | Sonuç / sınır |
| --- | --- |
| Test keşfi | 22 test. |
| İlk normal test koşusu | 20 geçti, 1 hata, 1 atlandı. Smoke testi bu çalışma ortamında `psutil.NoSuchProcess(pid=2)` ile durdu; Mac'te aynı hatanın oluştuğuna kanıt değildir. Torch testi, Torch bulunmadığı için atlandı. |
| Doctor ve metadata fetch yerine açık test mock'larıyla koşu | 21 geçti, 1 Torch testi atlandı. Bu sonuç canlı entegrasyon başarısı değildir. |
| Ruff | 19 uyarı/hata; çoğu import sırası / kullanılmayan değişken / modern type annotation. Eğitim engellerinden daha düşük öncelikte. |
| compileall | `src` ve `scripts` başarılı. |
| `experiment run --config EXP-002-frozen.yaml` | `trained:false` yazdı; arka planda sentetik tiny eğitim artifact'i üretiyor. Gerçek deney çalışmadı. |

Eksik Torch kontrolünü “backward/resume testi geçti” diye sunmak doğru olmaz: testin NumPy kısmı çalıştı, Torch bölümü atlandı.

## P0 — gerçek çalışmadan önce düzelt

### 1. Production akışına sentetik veri sızabiliyor

Kaynak: [pipeline.py](https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection/blob/2911e7c17148ee81087dd8e77ea27af78f6f6b67/src/rsna_knee/pipeline.py), `stage_labels` satır 117, `stage_train` satır 288, `stage_evaluate` satır 305, `stage_package` satır 351.

- `stage_labels` varsayılan `live=False` nedeniyle `state.synthetic=False` olsa da dört fixture raporu kullanıyor. `run_pipeline` live bayrağı vermiyor.
- `stage_train` içindeki `state.synthetic or exp_path is None`, gerçek pipeline'da exp_path verilmediği için tiny sentetik eğitim çalıştırabilir.
- `stage_evaluate` her durumda yeni TinyEncoder/random head kullanıyor. Üretilmiş eğitim checkpoint'ini yüklemiyor.
- `stage_package` her durumda `syn-*` UID'leri için örnek tahmin üretiyor.

**Doğrudan yeniden üretildi:** `synthetic=False` state için etiketler `syn-000…syn-003` oldu; `LABELS_READY` yazıldı. Train artifact'i `tiny_numpy`, `metrics_kind=synthetic` oldu. Sonraki CSV'de `syn-000` vardı.

Düzeltme: sentetik akış yalnız açık `--synthetic` ile, ayrı kök altında çalışmalı. Production'da eksik artifact/erişim hata veya BLOCKED olmalı; örnek veri fallback'i olmamalı. Etiket, cache, model ve metrik artifact'lerinin kökeni kontrol edilmeli.

### 2. Audit FAIL, READY_TO_SUBMIT durumunu engellemiyor

Kaynak: aynı dosyada `stage_audit`, `stage_package`, `sequential_run`; ayrıca [submission/kaggle.py](https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection/blob/2911e7c17148ee81087dd8e77ea27af78f6f6b67/src/rsna_knee/submission/kaggle.py).

`stage_audit` FAIL/BLOCKED sonucunu kaydetse bile state'i `AUDITING` yapıyor. Graph devam ediyor; paketleme yalnız kendi ürettiği fixture CSV'nin formatına bakıp `READY_TO_SUBMIT` yazıyor.

**Doğrudan yeniden üretildi:** audit `FAIL` → sonraki `stage_package` → `READY_TO_SUBMIT`.

`submit_run` da run varlığı, production flag, audit PASS, paket hash'i, kernel/version eşleşmesi ve önceki gönderim durumunu kontrol etmeden CLI çağrısına geçebiliyor. CLI exit=0 sonucunu kullanıyor; UUID çıkmayınca yerel UUID üretmesini sunucu submission kimliğiyle karıştırmamak gerekiyor. Kota cevabı alınıyor fakat uygulanmıyor; aynı kernel/version için tekrarlı gönderim koruması yok.

Düzeltme: audit'i son doğrulanmış paket, checkpoint, test UID listesi ve offline çalışma kanıtına bağla. PASS dışındaki durum submission'ı engellesin. Local attempt ID ile Kaggle receipt/server ID ayrı olsun. Belirsiz ağ sonucu tekrar gönderimden önce uzlaştırılsın.

### 3. Gerçek DINOv2 model ve trainer yok

Kaynak: [models/encoder.py](https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection/blob/2911e7c17148ee81087dd8e77ea27af78f6f6b67/src/rsna_knee/models/encoder.py), `TorchDinoStub` satır 41; [training/train.py](https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection/blob/2911e7c17148ee81087dd8e77ea27af78f6f6b67/src/rsna_knee/training/train.py); [cli.py](https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection/blob/2911e7c17148ee81087dd8e77ea27af78f6f6b67/src/rsna_knee/cli.py), `experiment_run` satır 154.

`TorchDinoStub` bir Conv2d + pooling + Linear ağı. Gerçek DINOv2 ViT değil. `strict=False` ile alakasız state dict bile missing/unexpected keys kontrol edilmeden yüklenmiş sayılabilir; bu durumda `is_stub=False` yazılması ayrıca yanlış güven üretir. Bu risk statik kod incelemesinden çıkarıldı; Torch kurulmadığı için burada çalıştırılmadı.

Mevcut trainer NumPy tiny head güncelliyor. Gerçek PyTorch study classifier, frozen/partial unfreeze loop, AMP, optimizer/scheduler, validation ve OOM uyarlaması bağlanmamış. Effective batch yardımcı fonksiyonları var, trainer kullanmıyor. `experiment run` state'i daima synthetic oluşturuyor.

Düzeltme: gerçek ViT-S/14 mimarisi, doğrulanmış pretrained weights ve hash; normalizasyon; [B,slots,centers,3,H,W] 2.5D tensor sözleşmesi; PyTorch masked weighted BCE; remote GPU training; epoch/step ve kaynak hash'li checkpoint. Model/weights uyumsuzluğunda açık hata.

### 4. Cache ve inference notebook'ları esas işi yapmıyor

Kaynak: [01_kaggle_build_cache.ipynb](https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection/blob/2911e7c17148ee81087dd8e77ea27af78f6f6b67/notebooks/01_kaggle_build_cache.ipynb), [04_kaggle_inference.ipynb](https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection/blob/2911e7c17148ee81087dd8e77ea27af78f6f6b67/notebooks/04_kaggle_inference.ipynb), `cli.cache_build`.

Cache notebook'u input yolunu bulup “read_dicom_dir kullanın” yazdırıyor. CLI handoff metni dönüyor. Seri seçimi, plane/sequence slotları, çoklu slice center seçimi, study loop, hata karantinası ve cache çıktısı yok. `arrays_to_study` tek serinin tek center triplet'ini yazıyor.

Inference notebook'u input yolunu bulup “checkpoint yükleyin ve submission.csv yazın” yazdırıyor. Checkpoint yükleme, gerçek test study loop ve CSV üretimi yok. Paketleyici varsayılan olarak eğitim checkpoint'i veya offline wheels eklemiyor; directory içindeki tüm dosyaları manifest hash listesine de koymuyor. Notebook bootstrap/package erişimi eksik.

Düzeltme: gerçek Kaggle cache üretimi ve internet kapalı inference. Eğitim ve inference aynı preprocessing kodunu kullansın. Paket gerekli weights/code/codec/dependencies içerip eksikse fail etsin. 3 örnek test üzerinden süre garantisi verilmesin.

## P1 — dayanıklılık ve veri doğruluğu

### 5. Worker/controller bağlantısı tamamlanmamış

Kaynak: [runtime/worker.py](https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection/blob/2911e7c17148ee81087dd8e77ea27af78f6f6b67/src/rsna_knee/runtime/worker.py), [02_colab_worker.ipynb](https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection/blob/2911e7c17148ee81087dd8e77ea27af78f6f6b67/notebooks/02_colab_worker.ipynb), `scripts/collect_results.py`.

Notebook sadece handshake'i ekrana yazıyor. Controller bunu okuyup job dispatch etmiyor. CLI worker `run_job`'a items vermiyor; worker bundle'daki input_uris/config'ten veri yüklemiyor. `accept_job` ayrı helper, run_job girişinde kullanılmıyor. Schema ve kaynak hash doğrulaması yok; sadece bir başlangıç heartbeat'i var.

**Doğrudan yeniden üretildi:** non-synthetic job, items yok → `metrics.kind=unavailable`, boş checkpoints, `exit_reason=ok`.

Collector, registry'deki beklenen attempt/token ile doğrulamadan gelen token'ı kaydediyor. Aktif heartbeat yoksa bitmiş sonucu da reddedebiliyor. Transport yerel klasör; konfigüre edilmiş Drive/API transferine bağlanmamış.

Düzeltme: job builder → bundle transfer → worker validation → gerçek dataset/trainer → periyodik heartbeat/checkpoint → immutable result → controller token/hash doğrulama. Kullanıcı Colab'a bağlandıktan sonra işin gerçekten devam ettiği gösterilmeli.

### 6. Resume ve pause mevcut sözleşmeyi karşılamıyor

Kaynak: `workflow/graph.py: sequential_run`, `training/train.py: train_tiny`, `cli.py: pipeline_pause`.

`--resume` eski state'i yüklese de bütün stages baştan çalışıyor; tamamlanmış stage/hash kontrolü yok. LangGraph build helper'ı tanımlı ama CLI pipeline bunu çağırmıyor. Config hash'i yalnız profile ve synthetic alanlarından geliyor.

NumPy trainer sadece weight/bias/step yüklüyor; epoch/sampler/RNG pozisyonunu geri almıyor. **Doğrudan deney:** kesintisiz 2 epoch = 8 adım; 2 adım sonrası resume = 10 adım ve farklı ağırlıklar. Test yalnız step artışı aradığı için bunu yakalamıyor.

`pipeline pause` başka CLI process'inde process-local bayrak set ediyor; çalışan controller bunu görmüyor. Registry state payload'ını yalnız run_id/stage ile değiştirip mevcut artifact/hash/profile bilgisini silebiliyor.

Düzeltme: kalıcı pause talebi, stage completion hash'leri, dependency invalidation, exact checkpoint continuation. Sonradan tekrar çalıştırma ile kesintiden devam etme ayrıştırılmalı.

### 7. Sentetik ve gerçek artifact alanları ortak

Sentetik folds aynı `state/folds.csv` üstüne yazılıyor. Smoke aynı `data/cache` ve `index.json` alanına fixture'lar koyuyor. Gerçek `stage_cache`, herhangi bir `.npy` bulunmasını CACHE_READY sayabiliyor; UID/köken/shape/version/hash kontrol etmiyor. Smoke'ta index'e 32px shard hash'i kaydedilip aynı dosya sonra 16px ile yeniden yazılıyor.

`StudyDataset` bulunmayan study cache'i için sessiz sıfır görüntü döndürüyor. Eksik bir seri için açık maskeli boş slot kabul edilebilir; bütün study verisinin bulunmamasını geçerli training sample gibi geçirmek kabul edilemez.

Düzeltme: ayrı synthetic kökü; gerçek cache manifest ve UID kapsamı doğrulaması; study eksikliğinde quarantine/error; son yazılan shard'a göre hash kontrolü.

### 8. Etiket extraction kimliği ve ayarları

Kaynak: `labels/extract.py`, `labels/lmstudio.py`, `pipeline.py: stage_labels`.

LLM'nin döndürdüğü StudyInstanceUID, istenen UID ile karşılaştırılmıyor. **Doğrudan mock testi:** RIGHT-UID isteği → WRONG-UID cevabı → `status=ok`.

JSON schema probe tanımlı ama stage onu çağırmıyor; fallback otomatik uygulanmıyor. Context/output/thinking/timeout/retries ayarlarının önemli kısmı config'te var fakat extraction çağrısına aktarılmıyor. Model revision için `owned_by` kullanılıyor; bu weights revision/quantization kimliği değil. `labels run --no-resume` dikkate alınmıyor. Gold-train pilot quality/calibration ve weak/gold label merge eğitim akışına bağlanmamış.

Düzeltme: UID eşleşmesi zorunlu, schema probe ve bounded fallback, gerçek model revision/config identity, uzun rapor token kontrolü, etiket kapsamı manifest'i. Pilot sayısının başarılı yeni örnekler mi yoksa incelenen örnekler mi olduğunu açıklaştır.

### 9. Leakage/audit henüz kanıt okumuyor

Gold eval split ve leakage helper var ama trainer bunu çağırmıyor. Duplicate report helper fold üretimine bağlanmamış. Folds gold-vs-weak stratification yapıyor; 12 hedefin dağılımını dengelemiyor. Az gold nedeniyle her hedefte pozitif/negatif sayıları raporlanmalı.

`stage_audit` leakage ve external data için sabit PASS gönderiyor. Auditor da bazı kontrolleri varsayılan PASS yapıyor. `rsna audit` production=False ve sentetik metrics ile çalışıyor; gerçek artifact okumuyor. `rsna evaluate` sadece registry kaydı yazdırıyor.

Düzeltme: actual train/eval UID'leri, duplicate groups, source/model overlap, preprocessing parity, checkpoint identity ve gerçek OOF tahmin dosyasından doğrulama. Eksik kanıt BLOCKED olsun.

## İstenen 5 dakika GPU fallback'inin durumu

`A100 → L4 → T4`, 300 saniye/aday, toplam 900 saniye config ve scripted testlerde bulunuyor. **Gerçek Colab'da çalışmıyor:** provider ilk A100 isteğinde manuel handoff döndürüyor ve controller duruyor. Bu sınırın dürüstçe yazılması doğru; fakat “bağlanınca otomatik sürdürür” kısmı henüz bağlanmamış.

Kaggle provider `request_gpu` kernel push yapmıyor; kernel kimliği üretmeden `pending_kernel` dönüyor. Bu haliyle alternatifi gerçekten başlatmıyor.

GPU politikası için ek doğruluk açıkları:

- Poll sırasında gelen auth/balance hatası durdurulmuyor. **Doğrudan test:** pending A100 → poll auth_failure → L4 allocated. Başlangıç auth testi bunu kapsamıyor.
- Cancel unsupported sonucunu kontrol etmeden sonraki adaya geçebilir; önceki isteğin hâlâ açık olma riski var.
- Testler Retry-After, gerçek GPU kimliği, cancel unsupported ve ayrı controller process'leri arasında duplicate request konularını kapsamıyor.

Sıra: önce dürüst ve çalışan manuel bağlantı sonrası worker akışı kurulsun. Desteklenen tahsis arayüzü doğrulanırsa otomatik 300 saniye geçişi eklenir. Headless API uydurulmamalı.

## Uygulama sırası

1. Production/synthetic ayrımı, audit ve submit kapıları, ayrı artifact kökleri.
2. Gerçek metadata → fold → label manifest → Kaggle study cache pilotu.
3. Gerçek DINOv2 model + PyTorch training/evaluation + checkpoint resume.
4. Colab manuel bağlantı sonrası job dispatch/worker/result akışı; Kaggle fallback.
5. Offline Kaggle inference, eksiksiz paket, runtime benchmark ve kontrollü submission.

Agent/skill dosyalarını yeniden çoğaltmak yerine bu beş teslimi bitirmek gerekiyor. Pilot için 10–20 gerçek study cache ve birkaç gerçek training batch yeterli başlangıç doğrulamasıdır; bu küçük veriyle yarışma AUC iddiası yapılmamalı.

## Cursor'a verilecek düzeltme prompt'u

Aşağıdaki bloğu repo kökünde Cursor Agent'a ver. Önceki master prompt'u da referans olarak tut.

```text
Bu repoyu gerçek RSNA MRI pipeline'ına tamamla. Önce AGENTS.md,
CURSOR_RSNA_MASTER_PROMPT.md ve RSNA_Cursor_Repo_Inceleme.md dosyalarını oku.
İnceleme 2911e7c commit'ine aittir; yeni kod varsa bulguların hâlâ geçerli olup
olmadığını kontrol et. Çalışan kodu koru, çözülen problemi yeniden yazma.

Hedef yalnız dosya yapısı ve mock başarı değil: metadata/folds/weak labels →
gerçek DICOM cache → gerçek DINOv2 training → checkpoint evaluation → offline
Kaggle inference akışını kur. Credentials veya remote GPU yoksa bağımsız kodu
tamamla, eksik canlı doğrulamayı açık BLOCKED olarak raporla. Bana her aşamada
devam izni sorma; yetkilendirilmiş yerel geliştirme ve testleri tamamla.

Önce P0 doğruluk kapıları:
- synthetic=False state hiçbir fixture, TinyEncoder veya syn-* UID kullanamaz.
  Eksik girdide NEEDS_AUTH/NEEDS_RUNTIME/BLOCKED veya açık hata üret.
- Synthetic smoke ayrı artifact/state/cache kökü kullansın; gerçek folds.csv,
  label cache ve cache index'e dokunmasın. Smoke offline ve credentials'sız olsun.
- FAIL/BLOCKED audit READY_TO_SUBMIT ve submit'i engellesin. Artifact/hash/run
  identity doğrulaması zorunlu olsun. Csv format kontrolü tek başına yetmez.
- CLI submit doğrudan kapıyı atlayamasın. Server submission receipt ile local
  attempt ID ayrı olsun; aynı kernel/version tekrar gönderimi engellensin.
- Yeni olumsuz regression testleri, eski kodda hatayı gösterecek nitelikte olsun.

Sonra gerçek veri yolu:
- Kaggle cache notebook ve cache build CLI yalnız talimat yazdırmasın.
  Study/series metadata'dan plane/sequence seç, IOP/IPP geometry ile sıralı
  DICOM decode yap, çoklu slice center ve adjacent 3-channel triplets üret.
  Aynı preprocessing'i training/inference kullansın. Codec eksikliği ve kötü
  study quarantine olsun. Mac'e tüm 570GB arşivi indirme.
- Cache manifest UID/shape/dtype/version/source hashes içersin. Missing study
  sessiz sıfır görüntü olmasın. Eksik series maskeleri ayrı ele alınsın.
- LM Studio actual /v1/models model id, UID eşleşmesi, actual revision/config
  identity, JSON capability probe+bounded fallback ve config ayarlarını uygula.
  Context'te uzun rapor truncation olmasın. Local LLM concurrency=1.
- Gold eval study'leri zayıf etiketli hâlleri dahil bütün image training'den
  çıkar. Actual train UID listesinde leakage assert çalışsın. Duplicate groups
  fold üretimine bağlansın; sınıf başına gold pos/neg sayıları raporlansın.

Gerçek model/trainer:
- TorchDinoStub'ı production encoder olarak kullanma. Gerçek DINOv2 ViT-S/14
  mimarisi ve doğru pretrained checkpoint'i yükle. Missing/unexpected keys ve
  architecture uyuşmazlığında fail et. is_stub=False bayrağı kanıt yerine geçmez.
- Torch nn.Module study classifier, frozen encoder + masked pooling + 12 logits,
  ardından partial unfreeze; masked weighted BCE, AMP, optimizer, scheduler,
  actual effective batch ve bounded OOM microbatch reduction kur.
- Training input [B,slots,centers,3,H,W] sözleşmesini açık ve testli yap.
- Evaluate actual saved checkpoint'i held-out data üstünde çalıştırsın; yeni
  random model yaratmasın. Gold/weak metrik ve OOF provenance ayrı tutulsun.
- Resume model/optimizer/scheduler/scaler/RNG/sampler/epoch/step/hash restore
  etsin. Interrupted+resumed ile uninterrupted sonuçlarını karşılaştıran test
  kur. Sadece step arttı testi yeterli değil.

Controller/worker:
- Job builder, actual bundle+input manifest transfer, worker job validation,
  heartbeat/checkpoint ve result ingest'i CLI pipeline'a bağla.
- Worker no-items/no-checkpoint durumunu ok saymasın. Expected schema/hash/
  attempt/fencing token job başlamadan ve sonuç kabulünde kontrol edilsin.
- Collector beklenen token'ı gelen token ile değiştirerek kabul etmesin.
  Bitmiş immutable result'i yalnız eski heartbeat nedeniyle reddetmesin.
- --resume yalnız valid tamamlanmış aşamaları atlasın; input hash değişince
  bağlı aşamaları invalidate etsin. Pause başka CLI process'inden kalıcı kontrol
  kaydıyla çalışan controller'a ulaşsın; mevcut state payload'ını silmesin.
- Colab headless allocation API uydurma. Mevcut resmi eklentiyle manuel bağlantı
  sonrası handshake → job çalışması → sonuç alma akışı gerçekten çalışsın.
- A100/L4/T4 300s policy yalnız desteklenen allocation arayüzünde otomatik olsun.
  Poll auth/balance failure durdursun; cancel unsupported ise önceki isteği
  açık bırakıp yeni allocation başlatmasın. Retry-After/deadline/token testleri
  ekle. Kaggle fallback gerçek kernels push/status/output akışını kullansın.

Submission:
- 04_kaggle_inference gerçek weights yükleyip hidden test metadata'daki bütün
  study'leri decode/preprocess/predict etsin, /kaggle/working/submission.csv yazsın.
- Bundle checkpoint, source, gerekli codec/dependencies ve tüm dosyaların hash
  manifest'ini içersin. Kernel metadata ve attached assets hazırlansın.
- Offline audit paketin son hâlini doğrulasın; varsayılan PASS kaldır.
- Gerçek GPU/runtime benchmark mevcutsa hidden test ölçeğine süre projeksiyonu
  yap; 3 visible study'den 9 saat garantisi verme.
- Bu düzeltme çalışmasında ücretli campaign, remote eğitim dispatch'i ve gerçek
  Kaggle submission başlatma. Bunları yapılabilir ve incelenebilir hazırla.

Test ve teslim:
1. Önce correctness gate'lerini düzelt, regression testleri çalıştır.
2. Ardından bir gerçek veri pilotunu çalıştırabilecek cache/model/trainer/worker
   akışını tamamla; test-only model açık ayrı isim ve synthetic namespace'te olsun.
3. pytest, gerekli Torch CPU forward/backward/resume testi ve Ruff çalıştır.
4. Synthetic offline E2E test ile fixture sonuçlarının production'a giremediğini
   göster. Canlı GPU eğitiminin yapılmadığı durumda bunu geçti diye yazma.
5. IMPLEMENTATION_STATUS, USER_RUNBOOK_TR ve sağlayıcı capability tablosunu
   actual implemented / tested / live verified / blocked ayrımıyla güncelle.
6. Son cevapta değişen davranış, test sonuçları, canlı doğrulama eksikleri ve
   benim uygulayacağım kısa ilk gerçek pilot komutlarını ver.

Plan veya placeholder dosyalarla bitirme; canlı erişim gerektirmeyen bütün
yetkilendirilmiş geliştirmeyi ve doğrulamayı tamamla.
```

Prompt'u kullanırken bu inceleme dosyasını repo köküne koy. Yerel geliştirme/test
tamamlandıktan sonra ilk hedef tüm kampanya değil, gerçek 10–20 study cache ve
checkpoint üretip yeniden yükleyebilen bir pilot olmalı.
