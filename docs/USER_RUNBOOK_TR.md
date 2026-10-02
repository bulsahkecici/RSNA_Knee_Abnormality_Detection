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
2. `rsna smoke --synthetic` — sahte AUC üretmez; `kind=synthetic` yazar.
3. LM Studio’da `qwen/qwen3.8-27b` yüklüyse: `rsna labels pilot --limit 5 --live`
4. Colab: `rsna runtime acquire --gpu-order A100,L4,T4 --wait-seconds 300` çıktısındaki **handoff** dosyasını izleyin. Eklentiyle kernel seçin. Bu adım otomatik GPU tahsisi değildir.
5. `rsna pipeline run --profile pilot`

## Sık komutlar

- Metadata (CSV, 570GB değil): `rsna metadata fetch` / `rsna metadata audit`
- Fold: `rsna folds create`
- Cache pilot (sentetik): `rsna cache pilot --limit 20`
- Asıl DICOM cache: Kaggle notebook `notebooks/01_kaggle_build_cache.ipynb`
- Durum: `rsna status --watch`
- Paket: `rsna package --run-id RUN`
- Submit **yapmaz** ta ki `--submit` ve gerçek kernel versiyonu: `rsna submit --run-id RUN --kernel USER/nb --version 1 --submit`

## Cursor / VS Code

Cursor agent’ları klasör açılınca kendiliğinden eğitim başlatmaz. VS Code görevleri `.vscode/tasks.json` içindedir. Colab eklentisi: `Google.colab`.

## Yapmayın

- Ham raporları sohbete yapıştırmayın.
- Simülatör AUC’sini liderboard sanmayın.
- CPU’da saatlerce tam eğitim başlatmayın.
- Cookie kopyalayarak Colab GPU “hack”lemeyin.
