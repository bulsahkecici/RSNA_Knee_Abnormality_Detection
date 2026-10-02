# RSNA repo ikinci inceleme

Tarih: 3 Ekim 2026, Türkiye saati.

Repo: https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection

İncelenen commit: `833da87e3e7f1e5581d3fc67d67f0e7771989705`.
Önceki inceleme: `2911e7c17148ee81087dd8e77ea27af78f6f6b67`.

**Karar: Önceki doğruluk açıklarının bir kısmı kapanmış ve gerçek model/cache yardımcıları eklenmiş. Ancak gerçek worker eğitimi, held-out evaluation ve Kaggle asset bağlantısı tamamlanmamış. Uzun GPU kampanyası için hazır değil. Sıradaki teslim, aşağıdaki hataları kapatan küçük bir gerçek pilot olmalı.**

60 dosyada 3387 ekleme / 446 silme incelendi. Testler ve ek kontroller ayrı yerel checkout'ta yapıldı. GitHub'a commit, PR, issue veya Kaggle gönderimi yapılmadı. Kullanıcının Mac/LM Studio/Colab oturumu bu incelemede kullanılmadı.

## Önceki bulguların durumu

| Önceki bulgu | Yeni durum |
| --- | --- |
| Production stage'lerde fixture/tiny fallback | Ana pipeline'da kaldırılmış; eksik girişte bloklanıyor. Worker'ın ayrı production doğrulaması hâlâ eksik. |
| Smoke gerçek folds/cache'e yazıyor | Smoke ayrı `state/synthetic` kullanıyor. Fakat yeni testin kendisi gerçek folds dosyasını değiştiriyor. |
| Audit FAIL → READY_TO_SUBMIT | FAIL/BLOCKED artık engelliyor. PASS belgesinin paket/checkpoint kimliğine bağlanması hâlâ eksik. |
| DINOv2 aslında CNN stub | Stub production kullanımından çıkarılmış, 12 bloklu ViT-S/14 ve StudyModel eklenmiş. Resmi pretrained ağırlıklarla parity doğrulanmamış. |
| Cache notebook yalnız print | Gerçek DICOM decode ve study cache builder eklenmiş. Resume/hash doğrulaması eksik. |
| Inference notebook yalnız print | Gerçek inference fonksiyonu çağrılıyor. Kaggle code/weights bootstrap ve bağlı asset'ler eksik. |
| Worker no-items → ok | `no_items` artık BLOCKED. Worker yine gerçek trainer'ı çağırmıyor. |
| Yanlış extraction UID kabulü | Expected UID kontrolü eklenmiş. |
| Poll auth failure sonrası yeni GPU isteği | Durdurma ve cancel-unsupported testleri eklenmiş. |
| Resume tüm stage'leri tekrar çalıştırıyor | Stage kayıtları eklenmiş. Dosya değişikliği algılama, pause talebinin tüketilmesi ve unfreeze resume eksik. |
| Ruff uyarıları | Temiz. |

## Test kanıtı

Python 3.12.14, CPU Torch `2.14.1+cpu`; CPU thread sayıları 1 ile sınırlı tutuldu.

| Kontrol | Sonuç |
| --- | --- |
| Normal pytest | 47 geçti, 3 hata. Üç hata da bu çalışma ortamında `psutil.NoSuchProcess(pid=2)` veren süreç ölçümünden kaynaklandı. |
| Yalnız preflight resource snapshot açık test mock'uyla pytest | **50 geçti, 0 atlandı.** 3 uyarı, fixture DICOM UID'lerinin geçerli UID formatında olmamasından kaynaklandı. |
| Ruff | Tüm kontroller geçti. |
| compileall | `src` ve `scripts` başarılı. |
| Gerçek GPU/official pretrained weights/Kaggle execution | Yapılmadı. CPU testleri canlı GPU kampanyası kanıtı değildir. |

Bu kez Torch forward/backward ve mevcut interrupted-resume testi çalıştırıldı. Ancak mevcut test çoğunlukla `microbatch=1`, `effective_batch=1`, frozen encoder ve 28px fixture verisini kapsıyor. Aşağıdaki ayrı kontroller bunun dışındaki hataları gösterdi.

## P0: Testin kendisi gerçek folds.csv üzerine yazıyor

Kaynak: [tests/test_smoke_namespace.py](https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection/blob/833da87e3e7f1e5581d3fc67d67f0e7771989705/tests/test_smoke_namespace.py), satır 9–17.

Test, `tmp_path` verilmesine rağmen `STATE_DIR / 'folds.csv'` üzerine şu içeriği yazıyor:

```csv
StudyInstanceUID,fold
REAL,0
```

Önceki dosyayı geri yüklemiyor. Aynı şekilde gerçek labels dizininde fixture dosyası oluşturuyor. İnceleme checkout'unda testlerden sonra folds.csv'nin gerçekten bu iki satır olduğu görüldü.

**Kullanıcının proje dizininde pytest çalıştırıldıysa folds.csv kontrol edilmeli.** Yalnız `REAL,0` kaldıysa gerçek metadata üzerinden yeniden üretilmeli; mevcut dosya eğitimde kullanılmamalı. Bu durum “kullanıcının Mac'inde dosya kesin bozuldu” iddiası değildir; testin açıkça aynı hedefe yazdığı doğrulandı.

Çözüm: bütün testlerde roots/registry/cache/control alanlarını `tmp_path` altında yönlendir. Gerçek mevcut dosyalara sentinel yazma. Var olan üretim dosyalarının hash'lerinin bütün test koşusu boyunca değişmediğini ayrıca doğrula.

## P0: Gerçek eğitim/evaluation yolu hâlâ bağlanmamış

Kaynaklar:

- [pipeline.py](https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection/blob/833da87e3e7f1e5581d3fc67d67f0e7771989705/src/rsna_knee/pipeline.py): `stage_train` satır 400–436, `stage_evaluate` satır 442–451.
- [runtime/worker.py](https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection/blob/833da87e3e7f1e5581d3fc67d67f0e7771989705/src/rsna_knee/runtime/worker.py): satır 76–98.
- `runtime/colab_session.py: run_after_handshake`, `cli.py: worker_cmd`.

Production `stage_train` bütün dallarda NEEDS_RUNTIME dönüyor. CUDA görünen dal bile trainer'ı çağırmıyor. `train_study_model` yalnız testlerden çağrılıyor; gerçek pipeline/worker çağrısı yok.

Colab `run_after_handshake` ve CLI worker, `items=None` veriyor. Job builder çoğunlukla kaynak hash metinlerini bir JSON'a yazıyor; worker'ın okuyabileceği cache/folds/labels/config URI ve gerçek training payload'ı oluşturmuyor. Yerel absolute `bundle_path` Colab'a taşınınca otomatik çözülmüyor. Uçtan uca transfer/dispatch/ingest mevcut pipeline'a bağlanmamış.

Worker'a items verilirse production dalı eğitim yapmıyor; sadece job'daki checkpoint dosyasının varlığına bakıyor.

**Doğrudan yeniden üretildi:** gerçek olmayan düz metin `dummy.pt` ve anlamsız items listesi → worker `exit_reason=ok`, `metrics.kind=real`. Dosya model olarak yüklenmedi, training step çalışmadı. Ayrıca `synthetic=False, encoder=tiny_test_encoder` ayrı worker yolunda açıkça reddedilmiyor.

Production evaluation mevcut checkpoint olsa bile `held_out_eval_not_run` ile BLOCKED. CLI evaluate checkpoint yolunu yazdırıyor; held-out study tahmini/AUC üretmiyor.

Çözüm: gerçek dataset/label merge/batch hazırlama → doğrulanmış pretrained encoder → Torch trainer → output checkpoint → held-out tahmin dosyası ve sınıf metrikleri zincirini worker'a bağla. Başarı, var olan bir dosyanın tespiti değil yeni ve doğrulanmış çalışma çıktısı olsun. Gold exclusions gerçek loader UID'lerinde denetlensin.

## P1: Gradient accumulation/microbatch ve unfreeze resume hataları

Kaynak: [training/loop.py](https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection/blob/833da87e3e7f1e5581d3fc67d67f0e7771989705/src/rsna_knee/training/loop.py), satır 68–69, 77–93, 110–165.

### A. Son kısmi accumulation grubu uygulanmıyor

**CPU'da doğrudan deney:** bir batch, bir epoch, effective_batch=4, microbatch=1 → `step=0`, head ağırlıkları tamamen aynı kaldı. Checkpoint yine yazıldı. Epoch sonunda eksik accumulation için optimizer step yok. Birkaç study'lik pilot bu nedenle hiç öğrenmeden bitmiş görünebilir.

### B. Microbatch parametresi görüntüleri bölmüyor

**Doğrudan forward hook:** input batch 3 study; microbatch=1 → model yine 3 study'yi tek forward'da gördü. Parametre yalnız accumulation sayısını değiştiriyor. `oob_once` gerçek CUDA OOM yakalamıyor; sadece sayacı azaltıyor. Bu haliyle GPU bellek kullanımını azaltmaz. Scheduler'ın toplam update hesabı da `batches` listesinin elemanlarını microbatch'e tekrar bölerek tutarsız olabiliyor.

### C. Partial-unfreeze sonrası resume kırılıyor

**Doğrudan deney:** ilk optimizer adımı sonrası last-block unfreeze ve checkpoint → yeni frozen model ile resume:

```text
ValueError: loaded state dict has a different number of parameter groups
```

Unfreeze sırasında optimizer'a eklenen ikinci parametre grubu checkpoint yüklemeden önce yeniden oluşturulmuyor. `requires_grad`/unfreeze phase state_dict'te saklanmıyor. CUDA RNG, input/config hash ve accumulation progress de bu loop checkpoint'ine bağlı değil.

Çözüm: batch'in study boyutunda gerçek slicing; gerçek OOM yakalama/retry; örnek sayısına göre loss ve accumulation hesabı; kısmi son grubun doğru uygulanması; unfreeze phase/optimizer groups/RNG/sampler/input hashes restore. Farklı effective batch, son kısa batch ve unfreeze+resume testleri ekle.

## P1: Cache resume eski çözünürlük/veriyi sessizce kullanıyor

Kaynak: [data/cache_build.py](https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection/blob/833da87e3e7f1e5581d3fc67d67f0e7771989705/src/rsna_knee/data/cache_build.py), satır 154–157; `pipeline.py: stage_cache`.

Prior shard dosyası varsa shape/preprocessing/source/content hash karşılaştırılmadan reuse yapılıyor.

**Doğrudan deney:** size=28,n_centers=3 cache → size=42,n_centers=1,resume=True → study shape yine `[3,3,3,28,28]`; yeni manifest global preprocessing hash'i ile study preprocessing hash'i farklı.

**İkinci deney:** yalnız manifest içinde gerçek görünümlü bir UID/shape/hash, disk üzerinde hiçbir shard yok → `stage_cache` CACHE_READY yazdı.

Çözüm: beklenen preprocessing/slot/center/source identity, actual file hash/shape/dtype ve requested UID kapsamı doğrulanmadan cache reuse veya READY olmasın. Eksik/kötü study açık quarantine olsun; 20 study pilot manifest'i tam dataset cache'i gibi kullanılmasın.

## P1: Submission kapısı paket kimliğine bağlı değil

Kaynak: `pipeline.py: stage_package`, [submission/kaggle.py](https://github.com/bulsahkecici/RSNA_Knee_Abnormality_Detection/blob/833da87e3e7f1e5581d3fc67d67f0e7771989705/src/rsna_knee/submission/kaggle.py), `gates.py`, `submission/package.py`.

FAIL audit'in engellenmesi düzelmiş. Fakat PASS hâlâ sadece `overall` alanı üzerinden kabul ediliyor.

**Doğrudan deney:** PASS JSON'u, checkpoint olmayan production state → `stage_package` READY_TO_SUBMIT. Aynı paketin `verify_package` sonucu `['checkpoint_missing']`; bu sonuç state geçişinde kullanılmıyor. `stage_package`, state checkpoint'ini package_run'a da vermiyor.

**Mock provider deneyi:** registry'de olmayan run_id + `audit={'overall':'PASS'}` → submit çağrısı yapıldı ve SUBMITTED kaydı oluşturuldu. Bu denemede gerçek Kaggle çağrısı yapılmadı. CLI injection kullanmasa da okunan audit belgesinin run/paket/kernel/version kimliği yeniden doğrulanmıyor.

Sunucu mesajında UUID bulunmazsa BLOCKED dönüp belirsiz attempt'i kaydetmiyor. Başarılı fakat farklı biçimdeki receipt veya ağ kopması sonrasında tekrar çağrı aynı gönderimi yine yapabilir. Kurulu resmi Kaggle CLI 2.2.4 kodunda code submit çıktısı server `message` alanından geliyor; bu alanın her zaman UUID içerdiği kanıtlanmamış. Kota kontrolü kullanılmıyor. `-f submission.csv` çıktı dosyasını adapter sözleşmesinde açık taşımak da gerekli.

Çözüm: audit üretimini gerçek evidence ve son paket hash'ine bağla; submit'te aynı run/config/checkpoint/package/kernel/version yeniden doğrula. SUBMITTING/UNKNOWN attempt gönderimden önce kalıcı kaydedilsin; belirsiz sonuç server geçmişi ile uzlaştırılmadan yeniden gönderilmesin. Testler yalnız PASS string'iyle başarılı olmamalı.

## P1: Kaggle paketi çalıştırılabilir bağlı asset oluşturmuyor

Cache kernel script'i ve inference notebook'u `rsna_knee` import ediyor. Paket local `src` kopyalasa da `dataset_sources`/`kernel_sources` boş. Notebook'ta package bootstrap yok. Inference sabit `/kaggle/input/bundle/checkpoint.pt` okuyor; paket checkpoint'i `checkpoint/<dosya-adı>` altına koyuyor, bunu Kaggle dataset/model asset'i olarak attach etmiyor.

Kurulu resmi Kaggle CLI `kernels_push` uygulaması metadata'daki code_file'ı request.text olarak gönderiyor ve belirtilen data/model sources'u bağlıyor; local klasördeki tüm src/checkpoint dosyalarını otomatik dataset olarak yüklemiyor.

**Doğrudan paket deneyi:** düz metin checkpoint dosyasıyla paket üretildi ve `identity_errors=[]` döndü. Model yüklenebilirliği/weights provenance kontrol edilmiyor. Offline wheels/codec bootstrap ve gerçek Kaggle runtime testi yok.

Inference model ve tensorları CPU'da bırakıyor; GPU açık metadata bu kodu CUDA'ya taşımıyor. Süre ölçümü kullanıcıdan verilen seconds_per_study sayısına bağlı; gerçek stopwatch ölçümü ve güvenlik payı yok.

Çözüm: versiyonlu Kaggle asset içine code, weights ve gerekli offline dependencies; notebook'ta gerçek mount yolundan bootstrap; exact checkpoint path; device seçimi; actual inference timing. Fresh offline ortamda paket import+load+fixture inference testi yap. Canlı remote koşu yoksa açık blocked/live-not-verified yaz.

## P1/P2: Controller resume ve pause kısmen uygulanmış

- `new_state` config hash'i hâlâ yalnız profile+synthetic'ten oluşuyor; gerçek YAML/experiment/prompt/model identity hash'i değil.
- Stage resume kayıtları stored hash metinlerini karşılaştırıyor; mevcut dosyaları yeniden okuyup content hash kontrolü yapmıyor. Silinmiş/değişmiş artifact'i tamamlanmış sayabilir.
- Pause control kaydı tüketilmiyor. **Doğrudan deney:** aynı run için iki resume çağrısı da PAUSED kaldı. Resume komutu talebi temizlemiyor.
- Train stage kayıtları, worker job ve result registry/ingest arasında state ilerleme bağlantısı yok. LangGraph helper hâlâ aktif CLI runner değil; bunu dokümanda net belirtmek yeterli, ikinci motor zorunlu değil.

Çözüm: gerçek dependency hash'leri, output hash/existence kontrolü, request acknowledgment ve resume/release semantics. Remote input URI çözümlemesi ve expected attempt/token registry'den yapılmalı.

## İkincil etiket/model doğruluk notları

- Tek extraction'da yanlış UID kontrolü düzeldi. Çok parçalı raporda `_merge_payloads` chunk UID'lerini nihai UID ile değiştiriyor; ilk not_mentioned dışı label'ı seçiyor. Çelişen pozitif/negatif parçalar reviewer/quarantine olmadan birleştirilebilir.
- Context kontrolü tokenizer yerine yaklaşık dört karakter/token kullanıyor ve system/schema overhead'i tam hesaplamıyor. “No silent truncation” kanıtı olarak yetersiz.
- JSON schema HTTP isteği hata verirse exception iç mode loop dışına çıkıyor; json_object fallback yerine aynı schema mode retry yapılabiliyor.
- Extraction cache key output/thinking/config kimliğini içermiyor. Model revision API'de yoksa boş; değişen yerel weights için tekrar kullanım riski sürüyor.
- `load_dinov2_vits14`, mimariye uyan bütün ağırlıkları `pretrained=True` sayıyor. **Doğrudan deney:** random-init kendi ViT state_dict'i load edilince pretrained=True. Official weights revision/hash allowlist ve independent output parity gerekiyor. Bir state_dict roundtrip testi, pretrained doğrulaması değildir.

## GPU fallback durumu

Poll auth/balance hatası ve cancel unsupported sırasında ikinci allocation açmama davranışı/testleri iyileşmiş. A100 → L4 → T4 300s policy mevcut. Gerçek Colab allocation API bulunmadığı için hâlâ manuel bağlantı var. Bu sınır doğru yazılmış; GPU seçimi otomatik olmuş sayılmamalı.

Sıradaki gerekli iş otomatik tahsis API'si aramak değil, manuel bağlanmış bir Colab runtime'ında job'ın gerçek veriyi yükleyip trainer çalıştırması ve controller'ın sonucu kabul etmesi.

## Cursor'a sonraki görev

Bu dosyayı repo köküne koyup aşağıdaki prompt'u ver:

```text
AGENTS.md, CURSOR_RSNA_MASTER_PROMPT.md ve
RSNA_Repo_Tekrar_Inceleme_833da87.md dosyalarını oku.
İnceleme 833da87 commit'ine ait; güncel kodda çözülmüş bulguları yeniden yazma.

Hedef bu kez yeni guard/placeholder eklemek değil: gerçek veri yükleyen worker
ile DINOv2 training → held-out evaluation → yeniden yüklenebilir checkpoint
pilotunu tamamlamak. Mevcut çalışan düzeltmeleri koru.

1. Önce test isolation: hiçbir test gerçek state/folds.csv, labels, cache,
   registry veya control dosyasına yazmasın. tmp_path/roots injection kullan.
   Mevcut folds.csv REAL,0 ile ezilmişse bunu açık raporla ve metadata varsa
   gerçek fold'u yeniden oluştur; real artifact'leri fixture ile doldurma.
2. Job bundle gerçek config/folds/labels/cache/pretrained weights URI ve hashes
   içersin. Worker bunları yüklesin; actual train UIDs'de leakage assert çalışsın.
   TRAINING job var olan checkpoint'in bulunmasını training success saymasın.
   Production'da TinyEncoder ve random weights pretrained iddiası reddedilsin.
3. train_study_model'a actual microbatch slicing, son kısmi accumulation update,
   doğru scheduler update hesabı ve gerçek OOM retry ekle. Unfreeze phase ve
   optimizer param groups checkpoint'ten yükleme öncesi kurulsun. CUDA RNG ve
   input/config identities saklansın. Unfreeze+resume ve batch-size değişiminde
   kesintisiz koşuyla eşdeğerlik test et.
4. Pipeline/worker trainer'ı gerçekten çağırsın; no_items'a bilerek None verme.
   Actual saved model ile gold/weak ayrılmış held-out prediction/AUC üret.
   stage_evaluate ve CLI evaluate yalnız checkpoint yolu yazdırmakla kalmasın.
5. Cache resume preprocessing/shape/centers/slots/source/content hash doğrulasın.
   Eksik shard bulunan manifest CACHE_READY olmasın. Pilot UID kapsamı açık olsun.
6. stage_package checkpoint'i gerçekten kopyalasın, verify_package sonucunu
   uygulasın. PASS audit run+checkpoint+package+kernel/version kimliğine bağlansın.
   CLI submit aynı doğrulamayı yeniden yapsın; stale PASS yeterli olmasın.
   SUBMITTING/UNKNOWN attempt kalıcı ve idempotent olsun; receipt parse/ağ
   belirsizliği sonrası server reconciliation olmadan tekrar gönderim yapma.
7. Kaggle code/weights/offline dependencies için gerçek versioned asset ve
   notebook bootstrap hazırla. Local klasörün kernels push ile komple runtime'a
   taşındığını varsayma. Asset mount ve checkpoint yolu sözleşmesi tutarlı olsun.
   Inference device seçimi ve gerçek stopwatch benchmark ekle.
8. Resume actual dosya/hash değişimini algılasın; pause request consume/ack ve
   resume davranışı testli olsun. Colab headless allocation API uydurma; manuel
   bağlantı sonrası çalışabilen bir worker kur.

Yukarıdaki doğrudan yeniden üretim örneklerini regression testlere dönüştür.
Mevcut 50 testin geçmesini tek tamamlanma ölçütü sayma. Torch CPU testlerini,
Ruff'u ve fresh offline bundle smoke'u çalıştır. Test sonuçlarını değiştirmeden
doğru raporla; GPU/official weights/LM Studio canlı kontrol yapılmadıysa belirt.

Credential veya remote GPU eksikliği bağımsız kodu yarım bırakma gerekçesi olmasın.
Ücretli training campaign veya gerçek submission başlatma; tamamlanmış kod ve
küçük gerçek pilot için uygulanabilir komutları hazırla. Örnek veri testi ile
gerçek MRI pilotunu ayrı isimle raporla. Plan/placeholder ile bitirme.
```

İlk başarı ölçütü: doğru kaynak hashes ve gerçek UID'ler ile yüklenen küçük cache,
gerçek pretrained modelden en az bir optimizer update, checkpoint restore,
held-out tahmin dosyası ve doğrulanmış worker result. Küçük pilot AUC'si yarışma
genelleme performansının kanıtı olarak sunulmamalı.
