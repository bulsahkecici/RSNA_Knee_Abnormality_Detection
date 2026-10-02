# Kullanım kılavuzu (Türkçe)

## Kurulum (macOS)

```bash
cd /Users/bulsahkecici/Projects/RSNA_Knee_Abnormality_Detection
./scripts/bootstrap_macos.sh
source .venv/bin/activate
rsna doctor
```

## En kısa akış

1. `rsna doctor` — Python, RAM, MPS/CUDA, LM Studio, Kaggle, Colab yetenekleri.
2. `rsna smoke --synthetic` — ayrı `state/synthetic` kökü. Gerçek `folds.csv` ve etiket/cache dizinine yazmaz. Aşama `BLOCKED` kalır; `READY_TO_SUBMIT` olmaz.
3. LM Studio’da `qwen/qwen3.8-27b` yüklüyse: `rsna labels pilot --limit 5 --live`
4. Colab: `rsna runtime acquire --gpu-order A100,L4,T4 --wait-seconds 300` çıktısındaki **handoff** dosyasını izleyin. Eklentiyle kernel seçin. Bu adım otomatik GPU tahsisi değildir. Allocation API yokken 300 saniye beklenmez.
5. `rsna pipeline run --profile pilot` — eksik cache, CUDA veya audit varsa `NEEDS_RUNTIME` / `BLOCKED` durur. TinyEncoder çalıştırmaz.

Örnek veri testleri (`tests/test_review_3e8c69c.py`) gerçek MRI pilotu değildir. Onlar 28px tensör ve geçici allowlist kullanır; `weights_official` false kalır.

Küçük gerçek pilot, tam arşiv indirmeden. Aynı `RUN` bütün adımlarda kullanılır. `pipeline run` Colab handoff'tan önce `job.json` yazar. Uzak makinede Mac mutlak yolu değil `RSNA_INPUT_ROOT` geçerlidir. Bu komutlar submission başlatmaz.

```bash
# Resmi ağırlık bir kez indirilir; SHA uydurulmaz.
# https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
mkdir -p artifacts/weights
# dosyayı artifacts/weights/dinov2_vits14_pretrain.pth olarak koyun
shasum -a 256 artifacts/weights/dinov2_vits14_pretrain.pth
# configs/dinov2_vits14_allowlist.json -> {"sha256": ["<çıkan hash>"]}

export RUN=pilot-$(date +%Y%m%d)
rsna doctor
rsna folds create
rsna labels pilot --limit 20 --live --run-id "$RUN"
rsna cache build \
  --series-csv data/metadata/train_series.csv \
  --dicom-root /path/to/a-few-studies \
  --dest data/cache \
  --limit 20
rsna pipeline run --profile pilot --run-id "$RUN"
# Handoff BLOCKED/NEEDS_RUNTIME ise job hazırdır. GPU makinesinde:
# export RSNA_INPUT_ROOT=/path/to/copied/bundle
rsna worker --job-bundle artifacts/runs/$RUN/job/job.json
rsna evaluate --run-id "$RUN"
rsna package --run-id "$RUN"
rsna audit --run-id "$RUN"
```

`--dicom-root` yoksa cache komutu kernel klasörünü hazırlar ve `kaggle kernels push` çağırmaz. `kernels push` yerel `src` ve checkpoint'i runtime'a taşımaz; paket `asset/` altında dataset metadata yazar, `executed` false kalır. Allowlist'te olmayan ağırlıkla worker `unverified_weights` döner. Colab'da otomatik GPU API yoktur; eklentiyle kernel bağlandıktan sonra aynı `rsna worker --job-bundle` çalışır. `RSNA_INPUT_ROOT` uzak makinede bundle kökünü gösterir.

## Sık komutlar

- Metadata (CSV, 570GB değil): `rsna metadata fetch` / `rsna metadata audit`
- Fold: `rsna folds create`
- Cache pilot (sentetik): `rsna cache pilot --limit 20`
- Asıl DICOM cache: `rsna cache build --series-csv ... --dicom-root ... --dest data/cache --limit 20` veya mount edilmiş yarışma girdisinde `notebooks/01_kaggle_build_cache.ipynb`
- Durum: `rsna status --watch`
- Başka süreçten duraklatma: `rsna pipeline pause --run-id RUN` (control kaydı; state payload silinmez)
- Paket: `rsna package --run-id RUN` — hash manifest’i tek başına submit açmaz
- Submit, audit `PASS` ve sunucu fişi olmadan `SUBMITTED` yazmaz: `rsna submit --run-id RUN --kernel USER/nb --version 1 --submit`

## Cursor / VS Code

Cursor agent’ları klasör açılınca kendiliğinden eğitim başlatmaz. VS Code görevleri `.vscode/tasks.json` içindedir. Colab eklentisi: `Google.colab`.

## Yapmayın

- Ham raporları sohbete yapıştırmayın.
- Simülatör AUC’sini liderboard sanmayın.
- CPU’da saatlerce tam eğitim başlatmayın.
- Cookie kopyalayarak Colab GPU “hack”lemeyin.
