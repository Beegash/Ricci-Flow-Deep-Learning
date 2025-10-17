# Ricci-NN Kurulum ve Çalıştırma Kılavuzu

## Gereksinimler
- Python 3.8 veya üzeri
- MacOS (mevcut sistem)

## Adım 1: Sanal Ortam (venv) Oluşturma

Terminal'i açın ve proje klasörüne gidin:
```bash
cd /Users/ifozmen/Downloads/Ricci-NN-main
```

Python sanal ortamı oluşturun:
```bash
python3 -m venv venv
```

## Adım 2: Sanal Ortamı Aktifleştirme

```bash
source venv/bin/activate
```

Sanal ortam aktifleştiğinde terminal prompt'unuzun başında `(venv)` göreceksiniz.

## Adım 3: MacOS SSL Sertifika Sorunu Çözümü

MacOS'ta Python SSL sertifika hatası almamak için:

```bash
# Python sertifikalarını yükle
/Applications/Python\ 3.11/Install\ Certificates.command
```

**Not:** Python sürümünüz farklıysa (3.9, 3.10 vb.), yukarıdaki yolu buna göre düzenleyin.

Eğer bu komut çalışmazsa veya dosya bulunamazsa endişelenmeyin, kod içinde SSL bypass ekledik.

## Adım 4: Bağımlılıkları Yükleme

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Bu işlem birkaç dakika sürebilir.

## Adım 5: Fashion MNIST Veri Setini İndirme

```bash
python download_dataset.py
```

Bu script:
- Fashion MNIST veri setini otomatik olarak indirir
- `fashion-mnist_train.csv` (60,000 örnek) oluşturur
- `fashion-mnist_test.csv` (10,000 örnek) oluşturur

## Adım 6: Veri Setinden Label 5 ve 9'u Çıkarma

```bash
python fmnist_extraction.py
```

Bu script `data_fmnist` klasöründe şu dosyaları oluşturur:
- `test5.csv` - Sandal test verileri
- `test9.csv` - Bot test verileri
- `train5.csv` - Sandal eğitim verileri
- `train9.csv` - Bot eğitim verileri

## Adım 7: Model Eğitimi

```bash
python training.py
```

Bu script:
- 3 farklı DNN modeli eğitir (her biri 5 katmanlı, 50 nöronlu)
- Her epoch'ta ilerlemeyi gösterir
- Şu dosyaları oluşturur:
  - `activation_model0.h5`, `activation_model1.h5`, `activation_model2.h5`
  - `model_predict.npy`
  - `accuracy.npy`
  - `x_test.csv`
  - `y_test.csv`

**Not:** Bu işlem yaklaşık 10-20 dakika sürebilir.

## Adım 8: Ricci Eğriliği Analizi

```bash
python knn.py
```

Bu script:
- k-NN grafikleri oluşturur
- Forman-Ricci eğriliklerini hesaplar
- Korelasyon analizlerini yapar
- Sonuçları terminal'de gösterir

## Sanal Ortamdan Çıkış

İşiniz bittiğinde:
```bash
deactivate
```

## Sorun Giderme

### SSL Certificate Hatası (CERTIFICATE_VERIFY_FAILED)
Bu sorunu çözmek için:

**Yöntem 1:** Python sertifikalarını yükleyin
```bash
/Applications/Python\ 3.11/Install\ Certificates.command
```

**Yöntem 2:** Manuel olarak certifi yükleyin
```bash
pip install --upgrade certifi
```

**Not:** `download_dataset.py` dosyası zaten SSL bypass içeriyor, bu yüzden çoğu durumda sorun yaşamazsınız.

### TensorFlow kurulum hatası
Eğer TensorFlow yükleme hatası alırsanız:
```bash
pip install tensorflow-macos  # Apple Silicon için
# veya
pip install tensorflow  # Intel Mac için
```

### Keras hatası
```bash
pip uninstall keras
pip install keras==2.13.1
```

### Memory hatası
Eğer bellek hatası alırsanız, `training.py` dosyasında `b = 3` satırını `b = 1` olarak değiştirin.

## Çıktı Dosyaları

Başarılı bir çalıştırma sonunda şu dosyalar oluşur:
```
Ricci-NN-main/
├── fashion-mnist_train.csv
├── fashion-mnist_test.csv
├── activation_model0.h5
├── activation_model1.h5
├── activation_model2.h5
├── model_predict.npy
├── accuracy.npy
├── x_test.csv
├── y_test.csv
└── data_fmnist/
    ├── test5.csv
    ├── test9.csv
    ├── train5.csv
    └── train9.csv
```

## Hızlı Başlangıç (Tüm Adımlar)

Tüm adımları tek seferde çalıştırmak için:
```bash
cd /Users/ifozmen/Downloads/Ricci-NN-main
python3 -m venv venv
source venv/bin/activate

# SSL sertifika sorununu çöz (opsiyonel)
/Applications/Python\ 3.11/Install\ Certificates.command 2>/dev/null || true

pip install --upgrade pip
pip install -r requirements.txt
python download_dataset.py
python fmnist_extraction.py
python training.py
python knn.py
```

**VEYA** master script kullanarak:
```bash
cd /Users/ifozmen/Downloads/Ricci-NN-main
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python run_all.py
```

