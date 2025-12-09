# Accuracy ve Rho (Ricci Curvature) Değerleri Hakkında

## 1. ACCURACY (Doğruluk) Nedir?

**Accuracy**, bir derin öğrenme modelinin test verisi üzerindeki performansını ölçen temel metriklerden biridir.

### Ne İfade Eder?
- **Tanım**: Modelin doğru tahmin yaptığı örneklerin toplam örnek sayısına oranı
- **Formül**: `Accuracy = (Doğru Tahminler) / (Toplam Örnekler)`
- **Değer Aralığı**: 0.0 (hiç doğru tahmin yok) ile 1.0 (tüm tahminler doğru) arasında
- **Bu Projede**: Binary classification (ikili sınıflandırma) için hesaplanıyor

### Örnek:
- 1000 test örneği var
- Model 980 tanesini doğru tahmin etti
- **Accuracy = 980/1000 = 0.98 (veya %98)**

### Kodda Nasıl Hesaplanıyor?
```python
# training.py veya grid_search.py'de:
acc = model.evaluate(x_test, y_test, verbose=0)[1]  # [1] = accuracy metriği
```

---

## 2. RHO (Ricci Curvature) Nedir?

**Rho (ρ)**, bu projede **Global Forman-Ricci Curvature** değerini temsil eder. Bu, "Deep Learning as Ricci Flow" makalesindeki temel geometrik metriktir.

### Ne İfade Eder?
- **Tanım**: Model aktivasyonlarının geometrik yapısını ölçen bir curvature (eğrilik) metriği
- **Hesaplama**: Her layer için kNN grafiği oluşturulur, bu grafikteki edge'lerin Forman-Ricci curvature'leri toplanır
- **Formül**: 
  - Bir edge (i,j) için: `R(i,j) = 4 - deg(i) - deg(j)`
  - Global Rho: `ρ = Σ R(i,j)` (tüm edge'ler üzerinden toplam)

### Geometrik Anlamı:
- **Pozitif Rho**: Daha "dışbükey" (convex) bir geometri, daha iyi ayrıştırma
- **Negatif Rho**: Daha "içbükey" (concave) bir geometri, daha zor ayrıştırma
- **Bu Projede**: Genellikle negatif değerler görülür (çünkü `4 - deg(i) - deg(j)` formülü kNN grafiklerinde genelde negatif olur)

### Kodda Nasıl Hesaplanıyor?
```python
# knn_fixed.py veya grid_search.py'de:
def global_forman_ricci(A: csr_matrix) -> float:
    deg = np.asarray(A.sum(axis=1)).ravel()  # Her node'un derecesi
    A_ut = sp_triu(A, k=1).tocoo()  # Upper triangle (her edge'i bir kez say)
    curv = 4.0 - deg[A_ut.row] - deg[A_ut.col]  # Her edge için curvature
    return float(curv.sum())  # Toplam = Rho
```

### Neden Önemli?
Makaleye göre, **daha yüksek accuracy'ye sahip modeller, genellikle daha yüksek (daha az negatif) Rho değerlerine sahiptir**. Bu, Ricci Flow teorisinin derin öğrenmedeki uygulamasını gösterir.

---

## 3. ACCURACY ve RHO DIŞINDA KULLANILABİLECEK DEĞERLER

Evet, başka metrikler de kullanılabilir! İşte bazı alternatifler:

### A. Model Performans Metrikleri (Accuracy yerine):
1. **F1-Score**: Precision ve Recall'ın harmonik ortalaması (dengesiz veri setleri için daha iyi)
2. **AUC-ROC**: ROC eğrisi altındaki alan (binary classification için)
3. **Precision/Recall**: Sınıf bazlı performans metrikleri
4. **Loss Value**: Training veya validation loss (düşük loss = iyi performans)

### B. Geometrik Metrikler (Rho yerine):
1. **Geodesic Mass (g_l)**: Tüm node çiftleri arasındaki en kısa yol uzunluklarının toplamı
   ```python
   g_l = Σ_{i<j} shortest_path_distance(i, j)
   ```
2. **Δg_l (Geodesic Mass Değişimi)**: Ardışık layer'lar arasındaki geodesic mass farkı
   ```python
   Δg_l = g_l - g_{l-1}
   ```
3. **Average Node Degree**: kNN grafiğindeki ortalama node derecesi
4. **Graph Diameter**: Grafikteki en uzun en kısa yol
5. **Clustering Coefficient**: Yerel kümeleme katsayısı

### C. Hibrit Metrikler:
1. **Layer-wise Correlation**: Her layer için ayrı ayrı hesaplanan korelasyonlar
2. **Architecture Complexity**: Model derinliği, genişliği gibi yapısal özellikler
3. **Training Dynamics**: Epoch sayısı, learning rate, convergence hızı

### Örnek Kullanım Senaryoları:

#### Senaryo 1: F1-Score vs. Rho
```python
# Dengesiz veri setleri için daha uygun
spearmanr(f1_scores, rho_values)
```

#### Senaryo 2: Loss vs. Geodesic Mass
```python
# Training loss ile geometrik metrik arasındaki ilişki
spearmanr(training_loss, geodesic_mass)
```

#### Senaryo 3: Multi-Metric Analysis
```python
# Birden fazla metriği birlikte analiz et
metrics = ['accuracy', 'f1_score', 'auc_roc']
geometric_metrics = ['rho', 'geodesic_mass', 'avg_degree']
# Her kombinasyon için korelasyon hesapla
```

---

## 4. SPEARMAN KORELASYON NEDEN KULLANILIYOR?

**Spearman Rank Correlation**, değerlerin **sıralamasına** dayalı bir korelasyon ölçüsüdür.

### Neden Spearman?
1. **Non-parametric**: Verinin normal dağılım göstermesini gerektirmez
2. **Outlier-resistant**: Aykırı değerlere karşı daha dayanıklı
3. **Monotonic Relationships**: Doğrusal olmayan ama monoton ilişkileri yakalayabilir
4. **Rank-based**: Değerlerin mutlak büyüklüğünden ziyade sıralamasına odaklanır

### Accuracy ve Rho için Neden Uygun?
- Accuracy genellikle 0.95-1.0 arasında sıkışmış olabilir (küçük varyans)
- Rho değerleri çok büyük negatif sayılar olabilir (örn: -10^8)
- Bu durumda **Pearson korelasyonu** yanıltıcı olabilir, ama **Spearman** sıralamaya dayandığı için daha güvenilirdir

### Örnek:
```
Model A: Accuracy=0.98, Rho=-1000000
Model B: Accuracy=0.99, Rho=-500000
Model C: Accuracy=0.97, Rho=-2000000
```

Spearman korelasyonu: "En yüksek accuracy'ye sahip model (B) en yüksek Rho'ya sahip mi?" sorusunu cevaplar.

---

## 5. SONUÇ

- **Accuracy**: Model performansını ölçer (0-1 arası)
- **Rho**: Geometrik yapıyı ölçer (genellikle negatif, büyük mutlak değerler)
- **Spearman Correlation**: Bu ikisi arasındaki **sıralama ilişkisini** ölçer
- **Alternatif Metrikler**: F1-score, loss, geodesic mass, vb. de kullanılabilir

**Beklenti**: Daha yüksek accuracy → Daha yüksek (daha az negatif) Rho
**Test**: Spearman ρ değeri 1'e yakınsa, bu beklenti doğrulanmış olur.

