# Otonom deney kampanyası

Kullanıcı 2026-10-09 tarihinde agent/skill destekli deney döngüsünü ve proje için tam yetkiyi açıkça verdi. Birincilik hedefidir, garanti değildir. Referans gönderim 56973673: public 0.740, yerel gold29 AUC0.75068253.

Kalıcı çalışma scripts/autonomous_campaign.py üzerinden yürür; agentların açık kalması eğitimi çalıştırmaz. En fazla iki build-time agent ve bir GPU işi aynı anda çalışır. İlk tur en fazla üç iş, 24 saat duvar saati ve ölçülen12 GPU saatiyle sınırlıdır; kota okunamazsa yeni iş başlatılmaz. Bu sınırlar kaynak korumasıdır, yeniden izin isteme şartı değildir.

Döngü: hazırlanmış işi yükle → gerçek makbuzu kaydet → 60s aralıkla takip et → gerçek metrik/tahmin/checkpoint indir → aynı UID ve gold ayrımıyla doğrula → gold ve weak değişimi + paired bootstrap belirsizliği hesapla → aday belirle. Belirsiz yükleme otomatik tekrarlanmaz; hatalı eğitim başarılı sayılmaz. İlerleme state/campaigns/autonomous-20261009 altında atomik kayıtlarla korunur.

Adayı seçmek gönderim değildir. Modelin kendi ön işlemesiyle internet kapalı benchmark, CSV ve <=9h süre kontrolü geçmelidir; sonra yetkilendirilmiş gönderim gerçek makbuzla kaydedilir. Mevcut 0.740 modeli korunur. İlk tur eğitim süresi, öğrenme oranı ve düzlemleri ayrı koruyan model başlığı denemelerini kapsar.

Ham radyoloji raporları cloud agentlara aktarılmaz, belirsiz/unmentioned etiketler maskelenir, gold-eval görüntüleri hiçbir eğitim listesine girmez. Public leaderboard üzerinde hedefe özel ağırlık ayarı yapılmaz.

Mac açık olmalıdır. caffeinate yalnızca normal uyumayı önler; yeniden başlatmada otomatik OS servisi henüz kurulmamıştır. Bu chat üzerinden zamanlanmış mesaj aracı yoktur; yerel Mac bildirimleri kullanılır.

Aday kapıları ve çevrimdışı doğrulama geçerse candidate_publish.py model paketini oluşturur, internet kapalı test işi çalıştırır ve izin verilmiş gönderimi bir kez yapar. Gönderim makbuzu belirsizse yeniden göndermez. Kota dönemi değişimi görülürse bütçe hesabı yeniden incelenene kadar yeni iş başlatılmaz.
