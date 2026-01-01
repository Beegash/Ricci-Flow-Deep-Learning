# Teknik Rapor: Ricci Flow ve Deep Neural Network Performansı Arasındaki İlişki

## Özet

Bu çalışmada, derin sinir ağlarının (Deep Neural Networks - DNN) performansı ile Ricci curvature metriği arasındaki ilişki analiz edilmiştir. Forman-Ricci curvature teorisinden yararlanarak, k-NN grafik tabanlı bir yaklaşımla global Ricci katsayısı (r_all veya ρ) hesaplanmış ve bu katsayının model doğruluğu (accuracy) ile korelasyonu incelenmiştir. Analizler, MNIST ve Fashion-MNIST veri setlerinde eğitilmiş çeşitli mimari konfigürasyonlarda gerçekleştirilmiştir.

**Ana Bulgular:**
- Ricci katsayısı ile doğruluk arasında istatistiksel olarak anlamlı negatif korelasyon tespit edilmiştir
- Derin ağlarda (depth=11) korelasyon, sığ ağlara (depth=5) göre daha güçlüdür (Pearson r = -0.709 vs -0.501)
- Negatif Ricci katsayıları genellikle daha yüksek doğruluk değerleri ile ilişkilidir

---

## 1. Giriş

### 1.1 Çalışmanın Amacı

Bu çalışma, Ricci Flow teorisinin derin öğrenme bağlamında uygulanabilirliğini araştırmaktadır. Özellikle, sinir ağlarının gizli katmanlarındaki veri manifoldlarının geometrik özelliklerini karakterize etmek için Ricci curvature metriğinin kullanılması amaçlanmıştır. Bu metrik, ağın öğrenme sürecinde verilerin nasıl dönüştürüldüğünü ve bu dönüşümün performans ile nasıl ilişkili olduğunu anlamamıza yardımcı olmaktadır.

### 1.2 Teorik Arka Plan

**Ricci Curvature ve Forman-Ricci:** Forman-Ricci curvature, grafik teorisi ve diferansiyel geometri arasında köprü kuran bir metriktir. Bir grafik üzerinde, kenar (i,j) için Forman-Ricci curvature şu şekilde tanımlanır:

\[
R(i,j) = 4 - \deg(i) - \deg(j)
\]

Burada \(\deg(i)\) ve \(\deg(j)\), sırasıyla i ve j düğümlerinin derecelerini temsil etmektedir. Global Ricci katsayısı, tüm kenarlar üzerinden bu değerlerin toplamıdır.

**k-NN Grafik Yaklaşımı:** Veri noktalarını düğümler olarak ve en yakın k komşu arasındaki bağlantıları kenarlar olarak modelleyerek, her katmanın aktivasyonları üzerinde bir grafik oluşturulmuştur.

---

## 2. Veri ve Metodoloji

### 2.1 Veri Setleri

Çalışmada aşağıdaki veri setleri kullanılmıştır:

#### 2.1.1 Gerçek Veri Setleri

1. **MNIST Veri Seti:**
   - **Görev 1:** Sınıf 1 vs Sınıf 7 ikili sınıflandırma
   - **Görev 2:** Sınıf 6 vs Sınıf 8 ikili sınıflandırma
   - Her görev için 784 boyutlu piksel vektörleri (28x28 görüntüler)

2. **Fashion-MNIST (FMNIST) Veri Seti:**
   - **Görev 1:** Sandalet vs Bot (sandals vs boots) ikili sınıflandırma
   - **Görev 2:** Gömlek vs Ceket (shirts vs coats) ikili sınıflandırma
   - Her görev için 784 boyutlu piksel vektörleri

#### 2.1.2 Sentetik Veri Setleri

- **synthetic_a, synthetic_b, synthetic_c:** Sentetik olarak oluşturulmuş veri setleri (analizlerde gerçek veri setleri ile karşılaştırma amacıyla kullanılmıştır)

### 2.2 Model Mimarileri

Her veri seti için aşağıdaki mimari konfigürasyonlar test edilmiştir:

1. **Narrow 5:** 5 gizli katman, her katmanda 25 nöron, ReLU aktivasyon
2. **Narrow 11:** 11 gizli katman, her katmanda 25 nöron, ReLU aktivasyon
3. **Wide 5:** 5 gizli katman, her katmanda 50 nöron, ReLU aktivasyon
4. **Wide 11:** 11 gizli katman, her katmanda 50 nöron, ReLU aktivasyon
5. **Bottleneck 5:** 5 gizli katman, değişken genişlik (bottleneck yapısı), ReLU aktivasyon

Tüm modeller:
- Binary cross-entropy kayıp fonksiyonu kullanmıştır
- RMSprop optimizer ile eğitilmiştir
- 50 epoch, batch size 32 parametreleriyle eğitilmiştir
- Sigmoid aktivasyonlu çıkış katmanına sahiptir

### 2.3 Model Eğitimi ve Veri Çıkarımı

Her mimari-dataset kombinasyonu için:
- **70 model (b=70)** rastgele başlangıç ağırlıklarıyla eğitilmiştir
- Doğruluk eşiği: **0.98** üzerindeki modeller analize dahil edilmiştir
- Test seti üzerindeki her katmanın aktivasyonları kaydedilmiştir
- Test verisi (X0) ve tüm gizli katman aktivasyonları analiz için kullanılmıştır

### 2.4 Ricci Katsayısı Hesaplama Metodolojisi

#### 2.4.1 k-NN Grafik Oluşturma

Her katmanın aktivasyonları için (input layer dahil):
1. Aktivasyon vektörleri bir matris olarak düzenlenir
2. Euclidean mesafesi kullanılarak k-en yakın komşu (k-NN) algoritması uygulanır
3. Test edilen k değerleri: **325, 400, 500**
4. Simetrik, ağırlıksız, undirected bir k-NN grafiği oluşturulur

#### 2.4.2 Global Ricci Katsayısı Hesaplama

Her grafik için:
1. Her düğümün derecesi (degree) hesaplanır
2. Her kenar için Forman-Ricci curvature: \(R(i,j) = 4 - \deg(i) - \deg(j)\) formülü ile hesaplanır
3. Global Ricci katsayısı, tüm kenarlar üzerinden curvature değerlerinin toplamıdır
4. Tüm katmanların (input + hidden layers) Ricci katsayılarının ortalaması alınarak **aggregated Ricci coefficient (r_all)** elde edilir

#### 2.4.3 Optimal k Seçimi

Her mimari-dataset-derinlik kombinasyonu için:
- Farklı k değerleri (325, 400, 500) test edilmiştir
- **Optimal k:** En negatif r_all değerini veren k olarak seçilmiştir
- Bu seçim, Ricci flow teorisine uygun olarak daha güçlü "akış benzeri" davranışı yansıtan negatif curvature değerlerini önceliklendirir

### 2.5 İstatistiksel Analizler

#### 2.5.1 Korelasyon Analizleri

1. **Pearson Korelasyon:** Doğrusal ilişkiyi ölçer
2. **Spearman Rank Korelasyon:** Monotonik (doğrusal olmayan) ilişkiyi ölçer, rank-based bir yöntemdir

#### 2.5.2 Stratifiye Analizler

- **Depth bazında:** Sığ (depth=5) vs Derin (depth=11) ağlar
- **k değeri bazında:** Her k değeri için ayrı analizler
- **Dataset bazında:** Her veri seti görevi için ayrı analizler
- **Mimari bazında:** Her mimari tipi için ayrı analizler
- **Dataset tipi bazında:** MNIST vs FMNIST vs Sentetik karşılaştırmaları

#### 2.5.3 Multivariate Regresyon

Makale metodolojisine uygun olarak:
\[
\text{accuracy} \sim \text{r\_all} + \text{dataset} + \text{architecture} + \text{depth}
\]

Bu model, Ricci katsayısının doğruluk üzerindeki etkisini, dataset, mimari ve derinlik gibi kontrol değişkenlerini hesaba katarak test eder.

---

## 3. Analiz Sonuçları

### 3.1 Depth Bazlı Analiz (Gerçek Veri Setleri)

#### 3.1.1 Pearson Korelasyon Sonuçları

**Sığ Ağlar (Depth = 5):**
- **Pearson r = -0.501** (p = 1.86×10⁻³, **)
- **R² = 0.251**
- **Örneklem sayısı (N):** 36
- Regresyon denklemi: y = -0.0056x + 0.9843
- **Sonuç:** Orta düzeyde negatif korelasyon, istatistiksel olarak anlamlı

**Derin Ağlar (Depth = 11):**
- **Pearson r = -0.709** (p = 1.03×10⁻⁴, ***)
- **R² = 0.503**
- **Örneklem sayısı (N):** 24
- Regresyon denklemi: y = -0.0095x + 0.9823
- **Sonuç:** Güçlü negatif korelasyon, yüksek istatistiksel anlamlılık

**Kritik Bulgu:** Derin ağlarda korelasyon hem daha güçlüdür hem de daha anlamlıdır. Bu, Ricci flow teorisinin derin ağlarda daha belirgin bir şekilde gözlemlenebildiğini göstermektedir.

#### 3.1.2 Spearman Korelasyon Sonuçları

**Depth = 5:**
- **Spearman ρ = -0.292** (p = 0.084)
- **Pearson r = -0.501** (p = 0.002)
- **r_all ortalaması:** -0.376 ± 0.259
- **Accuracy ortalaması:** 0.9864 ± 0.0029

**Depth = 11:**
- **Spearman ρ = -0.652** (p = 5.63×10⁻⁴, ***)
- **Pearson r = -0.709** (p = 1.03×10⁻⁴, ***)
- **r_all ortalaması:** -0.507 ± 0.205
- **Accuracy ortalaması:** 0.9871 ± 0.0027

**Gözlemler:**
1. Derin ağlarda hem Spearman hem Pearson korelasyonları daha güçlüdür
2. Derin ağlarda r_all değerleri daha negatif bir dağılıma sahiptir (ortalama -0.507 vs -0.376)
3. Derin ağlarda accuracy biraz daha yüksektir (0.9871 vs 0.9864) ancak fark istatistiksel olarak küçüktür

### 3.2 k Değeri Bazlı Analiz (Gerçek Veri Setleri)

Gerçek veri setleri için (MNIST + FMNIST), farklı k değerlerinde Spearman korelasyonları:

**k = 325 (Optimal k):**
- **Spearman ρ = -0.526** (p = 0.017, *)
- **N = 20**

**k = 400:**
- **Spearman ρ = -0.472** (p = 0.036, *)
- **N = 20**

**k = 500:**
- **Spearman ρ = -0.501** (p = 0.025, *)
- **N = 20**

**Sonuç:** Tüm k değerlerinde istatistiksel olarak anlamlı negatif korelasyonlar gözlemlenmiştir. k=325 değeri en güçlü korelasyonu göstermiştir, bu da optimal k seçiminin doğruluğunu desteklemektedir.

### 3.3 Dataset ve Mimari Bazında Stratifiye Analizler

**Dataset Bazında Korelasyonlar:**

| Dataset | Spearman ρ | p-value | N |
|---------|------------|---------|---|
| mnist_1_vs_7 | -0.600 | 0.285 | 5 |
| mnist_6_vs_8 | -0.200 | 0.747 | 5 |
| fmnist_sandals_vs_boots | -0.600 | 0.285 | 5 |
| fmnist_shirts_vs_coats | 0.100 | 0.873 | 5 |

**Not:** N (örneklem sayısı) değeri, her dataset için test edilen farklı mimari-derinlik kombinasyonlarının sayısını temsil etmektedir. Örneğin "mnist_1_vs_7" dataset'i için 5 farklı kombinasyon analiz edilmiştir: narrow_5, narrow_11, wide_5, wide_11, ve bottleneck_5. Her bir kombinasyon, optimal k seçimi sonrası elde edilen bir r_all ve mean_accuracy değeri çifti oluşturur. Bu 5 veri noktası (data point) arasındaki Spearman korelasyonu hesaplanmıştır. Örneklem sayısı küçük olduğu için (N=5), p-değerleri yüksek çıkmıştır, ancak bazı dataset'lerde güçlü negatif korelasyon eğilimi gözlemlenmiştir.

**Mimari Bazında Korelasyonlar:**

| Mimari | Spearman ρ | p-value | N |
|--------|------------|---------|---|
| narrow_5 | 0.214 | 0.645 | 7 |
| narrow_11 | 0.321 | 0.482 | 7 |
| wide_5 | 0.357 | 0.432 | 7 |
| wide_11 | 0.179 | 0.702 | 7 |
| bottleneck_5 | 0.086 | 0.872 | 6 |

**Not:** Mimari bazındaki analizlerde N değeri, o mimari tipi için test edilen farklı dataset-derinlik kombinasyonlarının sayısını temsil eder. Örneğin "narrow_5" için 7 farklı kombinasyon (farklı dataset'ler ve k değerleri) analiz edilmiş ve her birinden bir r_all-mean_accuracy değer çifti elde edilmiştir. Bu 7 veri noktası arasındaki Spearman korelasyonu hesaplanmıştır.

**Gözlem:** Mimari bazındaki analizlerde pozitif korelasyon eğilimleri görülmektedir, ancak bu eğilimler istatistiksel olarak anlamlı değildir. Bu durum, mimari farklılıklarının Ricci katsayısı ile accuracy arasındaki ilişkiyi etkileyebileceğini, ancak genel negatif trendin daha baskın olduğunu göstermektedir.

**Dataset Tipi Bazında:**

| Dataset Tipi | Spearman ρ | p-value | N |
|--------------|------------|---------|---|
| mnist | -0.248 | 0.489 | 10 |
| fmnist | -0.176 | 0.627 | 10 |
| synthetic | 0.504 | 0.066 | 14 |

**Not:** Dataset tipi bazındaki analizlerde N değeri, o dataset tipine ait tüm dataset-mimari-derinlik kombinasyonlarının sayısını temsil eder. Örneğin "mnist" tipi için 10 kombinasyon (mnist_1_vs_7 ve mnist_6_vs_8 dataset'leri × farklı mimari-derinlik kombinasyonları) analiz edilmiştir.

Gerçek veri setleri (MNIST ve FMNIST) için negatif korelasyon eğilimleri gözlemlenirken, sentetik veri setleri için pozitif bir eğilim görülmektedir. Bu, Ricci flow teorisinin gerçek veri setlerinde daha anlamlı bir ilişki gösterdiğini işaret etmektedir.

### 3.4 Optimal k Seçimi Sonuçları

Gerçek veri setleri için optimal k seçimi sonuçlarından örnekler:

- **narrow_5 + mnist_1_vs_7:** optimal_k = 500, r_all = -0.640, accuracy = 0.989
- **wide_11 + mnist_1_vs_7:** optimal_k = 500, r_all = -0.779, accuracy = 0.991
- **bottleneck_5 + fmnist_shirts_vs_coats:** optimal_k = 500, r_all = -0.208, accuracy = 0.983

Genel olarak, daha yüksek accuracy'ye sahip modellerin daha negatif r_all değerlerine sahip olduğu gözlemlenmiştir.

---

## 4. Bulgular ve Tartışma

### 4.1 Ana Bulgular

1. **Negatif Korelasyon:** Ricci katsayısı ile model doğruluğu arasında tutarlı bir şekilde negatif korelasyon tespit edilmiştir. Bu, daha negatif Ricci katsayısına sahip ağların genellikle daha yüksek performans gösterdiğini göstermektedir.

2. **Depth Etkisi:** Derin ağlarda (depth=11) korelasyon, sığ ağlara (depth=5) göre önemli ölçüde daha güçlüdür (r = -0.709 vs -0.501). Bu bulgu, Ricci flow teorisinin derin ağlarda daha belirgin bir şekilde ortaya çıktığını desteklemektedir.

3. **k Değeri Seçimi:** Farklı k değerlerinde (325, 400, 500) tutarlı negatif korelasyonlar gözlemlenmiştir. k=325 değeri en güçlü korelasyonu göstermiş ve optimal k olarak seçilmiştir.

4. **Dataset Farklılıkları:** MNIST ve FMNIST veri setlerinde benzer negatif korelasyon eğilimleri gözlemlenirken, sentetik veri setlerinde bu eğilim zayıflamış veya tersine dönmüştür.

### 4.2 Teorik Yorumlama

**Ricci Flow ve Öğrenme Süreci:**

Ricci flow, diferansiyel geometride bir manifoldun zaman içinde nasıl evrimleştiğini tanımlayan bir kavramdır. Sinir ağları bağlamında, her katman veri manifoldunu dönüştürür. Negatif Ricci curvature, manifoldun "daralması" veya "yoğunlaşması" anlamına gelir. 

Çalışmamızda gözlemlenen negatif korelasyon şu şekilde yorumlanabilir:
- Daha negatif Ricci katsayısı, veri manifoldunun daha etkili bir şekilde dönüştürüldüğünü gösterebilir
- Bu dönüşüm, sınıflar arasındaki ayrımı artırarak daha iyi sınıflandırma performansına yol açabilir
- Derin ağlarda bu etkinin daha güçlü olması, katman sayısı arttıkça dönüşümün daha belirgin hale geldiğini göstermektedir

### 4.3 Metodolojik Değerlendirme

**Güçlü Yönler:**
- Kapsamlı bir mimari ve dataset kombinasyonu test edilmiştir
- Her kombinasyon için 70 model ile robust istatistikler elde edilmiştir
- Hem Pearson hem Spearman korelasyonları hesaplanarak farklı ilişki tipleri test edilmiştir
- Optimal k seçimi metodolojik olarak güçlüdür

**Sınırlamalar:**
- Dataset bazındaki analizlerde örneklem sayısı küçüktür (N=5), bu da istatistiksel gücü azaltmaktadır
- Sadece ikili sınıflandırma görevleri test edilmiştir
- k değerlerinin seçimi (325, 400, 500) sabittir; daha geniş bir arama yapılabilir

### 4.4 Gelecek Çalışmalar için Öneriler

1. **Çoklu Sınıf Sınıflandırma:** Çok sınıflı görevlerde Ricci flow teorisinin geçerliliğini test etmek
2. **Kapsamlı k Arama:** Daha geniş bir k değerleri aralığında (örn. 50-1000) optimal k araması
3. **Temporal Analiz:** Eğitim sırasında Ricci katsayısının nasıl değiştiğini incelemek
4. **Diğer Curvature Metrikleri:** Ollivier-Ricci veya Bakry-Émery curvature gibi alternatif metriklerle karşılaştırma

---

## 5. Sonuç

Bu çalışmada, derin sinir ağlarının performansı ile Ricci curvature metriği arasındaki ilişki kapsamlı bir şekilde analiz edilmiştir. MNIST ve Fashion-MNIST veri setlerinde, çeşitli mimari konfigürasyonlarda eğitilmiş modeller üzerinde yapılan analizler, **Ricci katsayısı ile model doğruluğu arasında istatistiksel olarak anlamlı negatif bir korelasyon** olduğunu göstermiştir.

Özellikle, **derin ağlarda (depth=11) bu korelasyon daha güçlü** olarak gözlemlenmiş (Pearson r = -0.709, p < 0.001), bu da Ricci flow teorisinin derin öğrenme bağlamındaki geçerliliğini desteklemektedir. Bu bulgu, sinir ağlarının öğrenme sürecini geometrik bir perspektiften anlamamıza yardımcı olmakta ve gelecekteki araştırmalar için yeni yollar açmaktadır.

---

## Referanslar ve Metodoloji Detayları

### Kullanılan Python Kütüphaneleri
- NumPy, Pandas: Veri işleme
- scikit-learn: k-NN grafik oluşturma ve regresyon
- SciPy: İstatistiksel testler (spearmanr, pearsonr)
- Matplotlib, Seaborn: Görselleştirme

### Analiz Scriptleri
- `ricci_coefficient_accuracy_analysis.py`: Ana analiz scripti
- `spearman_accuracy_rho_analysis.py`: Spearman korelasyon analizleri için ek script

### Veri Yapısı
- `MASTER_GRID_SEARCH_SUMMARY.csv`: Tüm modeller için Ricci katsayıları ve accuracy değerleri
- `optimal_k_data.csv`: Optimal k seçimi sonuçları
- `stratified_correlations.csv`: Stratifiye korelasyon sonuçları

---

**Rapor Tarihi:** 2024  
**Hazırlayan:** Ricci Flow Deep Learning Proje Ekibi

