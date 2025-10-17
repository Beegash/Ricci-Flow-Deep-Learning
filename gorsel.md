# Ricci-NN Görselleştirme Kılavuzu

Bu kılavuz, Ricci-NN projesindeki analizleri görselleştirmek için kullanabileceğiniz araçları açıklar.

## 📊 Mevcut Görselleştirmeler

### 1. **Ricci Akış Analizi** (`knn.py`)

Bu script zaten Ricci eğrilik analizini yapıyor ve grafik oluşturuyor:

```bash
python knn.py
```

**Oluşturulan Grafik:** `ricci_flow_analysis.png`

İçerik:
- **Panel 1:** Katmanlara göre geodezik mesafe değişimi (boxplot)
- **Panel 2:** Katmanlara göre Forman-Ricci eğriliği (boxplot)
- **Panel 3:** Katmanlar arası korelasyon (scatter plot + regresyon)

### 2. **Detaylı Görselleştirmeler** (`visualize_results.py`)

Daha kapsamlı analizler için:

```bash
python visualize_results.py
```

**Oluşturulan Grafikler:**

#### `model_comparison.png`
- Farklı modellerin test accuracy karşılaştırması
- Renkli bar chart

#### `layer_activations.png`
- Her katmandaki aktivasyon değerlerinin dağılımı
- Histogram + istatistikler (ortalama, std)

#### `activation_heatmap.png`
- Katmanlar arası aktivasyon korelasyon ısı haritası
- Her katmanın ortalama aktivasyon profili

#### `pca_variance.png`
- Her katmanda PCA ile açıklanan varyans analizi
- Kümülatif varyans eğrileri

#### `tsne_layers.png`
- t-SNE ile katman temsillerinin 2D projeksiyonu
- Her katman için ayrı panel
- Sınıflara göre renklendirme

## 🎨 Grafik Özellikleri

Tüm grafikler:
- **Yüksek çözünürlük:** 300 DPI
- **Profesyonel stil:** Seaborn + Matplotlib
- **Makale kalitesi:** Publication-ready
- **Bilgilendirici:** İstatistikler ve açıklamalar dahil

## 📝 Kullanım Senaryoları

### Senaryo 1: Hızlı Bakış
Sadece Ricci analizi sonuçlarını görmek için:
```bash
python knn.py
```

### Senaryo 2: Tam Analiz
Tüm görselleştirmeleri oluşturmak için:
```bash
python knn.py
python visualize_results.py
```

### Senaryo 3: Otomatik Pipeline
Tüm süreci baştan sona çalıştırmak için:
```bash
python run_all.py
python visualize_results.py
```

## 🔧 Özelleştirme

### DPI Değişimi
Daha yüksek çözünürlük için `visualize_results.py` dosyasında:
```python
plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 300 yerine 600
```

### Renk Şeması
Renk paleti değiştirmek için:
```python
sns.set_palette("husl")  # veya "Set2", "pastel", vb.
```

### Grafik Boyutu
Figür boyutunu ayarlamak için:
```python
plt.figure(figsize=(16, 12))  # Genişlik x Yükseklik (inç)
```

## 📋 Grafik Açıklamaları

### Geodesic Distance (Geodezik Mesafe)
- Her katman arasındaki geometrik mesafe değişimi
- Pozitif değer: Genişleme
- Negatif değer: Daralma

### Forman-Ricci Curvature
- Ağın geometrik eğriliği
- Negatif eğrilik: Hiperbolik yapı (genişleme)
- Pozitif eğrilik: Küresel yapı (daralma)

### t-SNE Projection
- Yüksek boyutlu verilerin 2D görselleştirmesi
- Benzer örnekler yakın kümelenir
- Sınıf ayrımı netliği gösterir

### PCA Variance
- Veri boyut azaltma etkinliği
- %95 varyansı açıklamak için gereken bileşen sayısı
- Bilgi kaybını gösterir

## 💡 İpuçları

1. **Büyük veri setleri için:** t-SNE hesaplaması uzun sürebilir. Perplexity değerini ayarlayın.

2. **Bellek optimizasyonu:** Eğer RAM sorunu yaşıyorsanız, `training.py`'de `b=1` yapın (daha az model).

3. **Grafik formatı:** PNG yerine PDF istiyorsanız:
   ```python
   plt.savefig('grafik.pdf', format='pdf', bbox_inches='tight')
   ```

4. **Interaktif grafikler:** Matplotlib backend'ini değiştirin:
   ```python
   import matplotlib
   matplotlib.use('TkAgg')  # veya 'Qt5Agg'
   ```

## 🎯 Makale İçin Öneriler

Makalede kullanım için:
- **Ana sonuç:** `ricci_flow_analysis.png` (3 panel)
- **Destekleyici:** `tsne_layers.png` (görsel ayrım)
- **Ek materyal:** `pca_variance.png`, `activation_heatmap.png`

## 🐛 Sorun Giderme

### "No display name" hatası
Sunucuda çalışıyorsanız:
```python
import matplotlib
matplotlib.use('Agg')  # Headless mode
```

### "Memory error"
Daha az örnek kullanın:
```python
sample_indices = np.random.choice(len(y_test), 1000, replace=False)
```

### Grafikler görünmüyor
`plt.show()` ekleyin veya kaldırın:
```python
plt.show()  # İnteraktif gösterim için
# veya
plt.savefig(...)  # Sadece kaydetmek için
```

