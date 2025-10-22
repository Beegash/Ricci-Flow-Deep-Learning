import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from keras.models import load_model
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import squareform, pdist
import os
import glob

# path = "/home/anthbapt/Documents/fMNIST_DNN_training/wk"
# os.chdir(path)


# Dinamik yol - training_outputs klasöründen veri okuyoruz
data_path = os.path.join(os.getcwd(), 'training_outputs')
os.chdir(data_path)


# Load file and model
model = np.load("model_predict.npy", allow_pickle=True)
accuracy = np.load("accuracy.npy")

# hatalı satır ilk sefer aldığımız hatanın kaynağı bu kısım
# burası bizim shape kaynaklı aldığımız hatanın olduğu yer
# x_test = np.array(pd.read_csv("x_test.csv", header = None))[1::]

# remove number of features
x_test = np.array(pd.read_csv("x_test.csv", header=None))#bu şekilde düzeltildi

k = 350
a = 0.98


def corr_analysis(scalar_curvs1, accuracy, x_test):
    # Generate FR curvature data frame
    accuracy = np.asarray(accuracy)
    acc = accuracy[::2]
    
    # ============================================================================
    # ESKİ KOD (HATALI) - yorum satırına alındı:
    # ============================================================================
    # s = scalar_curvs1[np.where(acc > a)[0][0]]
    # sr = s[0]
    # for i in range(1, len(s)):
    #     sr = np.column_stack((sr, s[i]))
    # for i in np.where(acc > a)[1:]:    # ← HATA BURADA!
    #     s = scalar_curvs1[i]
    #     sr2 = s[0]
    #     for j in range(1, len(s)):
    #         sr2 = np.column_stack((sr2, s[j]))
    #     sr = np.column_stack((sr, sr2))
    #
    # HATA SEBEBİ: 
    # np.where(acc > a) bir tuple döndürür: (array([0, 2]),)
    # [1:] ile tuple'ı kesince boş tuple elde edilir: ()
    # Sonuç: Döngü hiç çalışmaz, sadece ilk model işlenir
    # Bu yüzden sr matrisi (2000, 6) şeklinde kalır, (2000, 12) olması gerekirken
    # ============================================================================
    
    # YENİ KOD (DÜZELTİLMİŞ):
    # Get indices of models with accuracy > threshold
    good_model_indices = np.where(acc > a)[0]
    
    # Start with first good model
    s = scalar_curvs1[good_model_indices[0]]
    sr = s[0]
    for i in range(1, len(s)):
        sr = np.column_stack((sr, s[i]))
    
    # Add remaining good models
    for model_idx in good_model_indices[1:]:
        s = scalar_curvs1[model_idx]
        sr2 = s[0]
        for j in range(1, len(s)):
            sr2 = np.column_stack((sr2, s[j]))
        sr = np.column_stack((sr, sr2))

    
    # ============================================================================
    # ESKİ KOD (HATALI VE GEREKSIZ) - yorum satırına alındı:
    # ============================================================================
    # column_names = np.repeat(np.arange(1, len(scalar_curvs1[0]) + 1), len(np.where(acc > a)[0]))
    # pd.DataFrame(sr, columns=column_names)
    #
    # HATA SEBEBİ:
    # 1) column_names hesaplaması yanlış yapılıyor:
    #    - 6 katman × 2 model = [1,1,2,2,3,3,4,4,5,5,6,6] → 12 sütun adı
    #    - Ama eski kodda sr (2000, 6) oluşuyordu → Boyut uyumsuzluğu!
    # 2) DataFrame oluşturuluyor ama hiçbir değişkene atanmıyor (boşa işlem)
    # 3) Bu DataFrame zaten sonraki adımlarda kullanılmıyor
    # Sonuç: Bu kod bloğu tamamen gereksiz, kaldırıldı
    # ============================================================================
    
    # YENİ KOD: Bu blok tamamen kaldırıldı
    # Debug: print shapes to understand the structure
    print(f"sr shape: {sr.shape}")
    print(f"len(scalar_curvs1[0]): {len(scalar_curvs1[0])}")
    print(f"len(good_model_indices): {len(good_model_indices)}")

    
    # ============================================================================
    # ESKİ KOD (HATALI) - yorum satırına alındı:
    # ============================================================================
    # ssr = sr[:, 0].copy()
    # for i in range(1, sr.shape[1]):
    #     ssr = np.column_stack((ssr, sr[:, i]))
    #
    # HATA SEBEBİ:
    # column_stack ne yapar? İki array'i YAN YANA koyar → 2 BOYUTLU array oluşturur!
    # 
    # Adım adım:
    # 1) İlk durum: ssr = sr[:, 0] → şekil (2000,) (1D array)
    # 2) İkinci sütun: column_stack → şekil (2000, 2) (2D array!)
    # 3) Üçüncü sütun: column_stack → şekil (2000, 3) (2D array!)
    # ...
    # 12) Son durum: ssr → şekil (2000, 12) (2D array!)
    #
    # Ama DataFrame için 1D array gerekiyor: (24000,)
    # Çünkü:
    # - 2000 örnek × 6 katman × 2 model = 24000 satır
    # - Her satır: bir örnek + bir katman + bir model kombinasyonu
    #
    # Sonuç: ssr (2000, 12) ile layer_all (24000,) birleştirilemez → HATA!
    # ============================================================================
    
    # YENİ KOD (DÜZELTİLMİŞ):
    # Extract to data frame summary
    # sr has shape (n_samples, n_layers * n_models)
    # We need to flatten it into long format: each row = one sample in one layer for one model
    
    n_samples = x_test.shape[0]
    n_layers = len(scalar_curvs1[0])
    n_models = len(good_model_indices)
    
    # Debug: print dimensions
    print(f"n_samples: {n_samples}")
    print(f"n_layers: {n_layers}")
    print(f"n_models: {n_models}")
    print(f"Expected total rows: {n_samples * n_layers * n_models}")
    
    # Flatten sr column by column to create ssr (scalar curvature values)
    # sr columns are organized as: [layer1_model1, layer2_model1, ..., layer6_model1, layer1_model2, layer2_model2, ...]
    # After flattening: [all samples layer1_model1, all samples layer2_model1, ..., all samples layer6_model2]
    ssr = sr.flatten(order='F')  # Flatten column-wise (Fortran order) → (24000,) 1D array
    
    # ============================================================================
    # ESKİ KOD (HATALI) - yorum satırına alındı:
    # ============================================================================
    # layer = np.repeat(1, x_test.shape[0])
    # for i in range(2, len(scalar_curvs1[0])):
    #     layer = np.column_stack((layer, np.repeat(i, x_test.shape[0])))
    # layer_all = np.repeat(layer, len(np.where(acc > a)[0]))
    #
    # HATA SEBEBİ - 3 AYRI HATA VAR:
    # 
    # HATA #1: range(2, 6) = [2, 3, 4, 5] → Katman 6 hiç eklenmiyor!
    # - Olması gereken: range(1, 7) veya range(1, n_layers+1)
    # - Sonuç: Sadece 5 katman var, 6. katman eksik
    #
    # HATA #2: column_stack yine 2D array oluşturuyor!
    # - İlk: layer = [1,1,...,1] → (2000,) 1D
    # - i=2: column_stack → (2000, 2) 2D
    # - i=3: column_stack → (2000, 3) 2D
    # - i=5: column_stack → (2000, 5) 2D (son durum)
    #
    # HATA #3: Boyut uyumsuzluğu!
    # - 2D array'i repeat edince: (20000,) oluyor
    # - Olması gereken: (24000,) 
    # - Eksik: 4000 satır! (çünkü 6. katman yok)
    # ============================================================================
    
    # YENİ KOD (DÜZELTİLMİŞ):
    # Create layer indicators to match the flattening order
    # Pattern: [1]*n_samples, [2]*n_samples, ..., [6]*n_samples, [1]*n_samples, ..., [6]*n_samples
    layer_all = np.tile(np.repeat(np.arange(1, n_layers + 1), n_samples), n_models)
    
    # ============================================================================
    # ESKİ KOD (HATALI) - yorum satırına alındı:
    # ============================================================================
    # mod = np.repeat(1, len(scalar_curvs1[0]) * x_test.shape[0])
    # for i in range(2, len(np.where(acc > a)[0])):
    #     mod = np.column_stack((mod, np.repeat(i, len(scalar_curvs1[0]) * x_test.shape[0])))
    #
    # HATA SEBEBİ - 3 AYRI HATA VAR:
    # 
    # HATA #1: range(2, 2) = BOŞ RANGE!
    # - len(np.where(acc > a)[0]) = 2 (2 model var)
    # - range(2, 2) = [] → Döngü hiç çalışmıyor!
    # - Sadece ilk satır çalışıyor: mod = [1,1,...,1] (12000 eleman)
    # - Model 2 hiç eklenmiyor!
    #
    # HATA #2: Yine column_stack sorunu!
    # - Eğer 3+ model olsaydı bile:
    # - İlk: mod = [1,1,...,1] → (12000,) 1D
    # - Döngüde: column_stack → (12000, 2) 2D
    # - Yine 2D array problemi!
    #
    # HATA #3: Boyut uyumsuzluğu!
    # - Sadece model 1 var: mod → (12000,)
    # - Olması gereken: (24000,) (2 model için)
    # - Eksik: 12000 satır! (Model 2 eksik)
    # - ssr ve layer_all ile birleştirilemez!
    # ============================================================================
    
    # YENİ KOD (DÜZELTİLMİŞ):
    # Create model indicators to match the flattening order
    # Pattern: [1]*(n_layers*n_samples), [2]*(n_layers*n_samples), ...
    mod = np.repeat(np.arange(1, n_models + 1), n_layers * n_samples)
    
    # Debug: print final shapes
    print(f"ssr shape: {ssr.shape}")
    print(f"layer_all shape: {layer_all.shape}")
    print(f"mod shape: {mod.shape}")
    
    # ============================================================================
    # ESKİ KOD (ÇALIŞIR AMA KARMAŞIK) - yorum satırına alındı:
    # ============================================================================
    # data = pd.DataFrame(np.column_stack((ssr, layer_all, mod)), columns=["ssr", "layer", "mod"])
    #
    # SORUN:
    # Eski kodda (Değişiklik 3-4-5 yapılmadan önce):
    # - ssr → (2000, 12) 2D
    # - layer_all → (20000,) 1D
    # - mod → (12000,) 1D
    # - Boyutlar uyuşmuyor → ValueError!
    #
    # Şimdi (Değişiklik 3-4-5'ten sonra):
    # - ssr → (24000,) 1D ✓
    # - layer_all → (24000,) 1D ✓
    # - mod → (24000,) 1D ✓
    # - Boyutlar eşleşiyor, çalışır!
    #
    # Ama daha iyi bir yöntem var (yeni kod):
    # - column_stack ara adımı gereksiz
    # - Dictionary yöntemi daha Pythonic
    # - Daha okunabilir ve güvenli
    # ============================================================================
    
    # YENİ KOD (DÜZELTİLMİŞ ve DAHA İYİ):
    data = pd.DataFrame({'ssr': ssr, 'layer': layer_all, 'mod': mod})
    
    return data


def filter_(fr_data, sc_data):
    # Deal with infinite geodesics by removing models with these
    d = sc_data.groupby(['layer', 'mod'])['ssr'].agg(
        ['mean', 'std']).reset_index()
    # Replace Inf values with NaN for further processing
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Remove rows with NaN in the mean column
    d.dropna(subset=['mean'], inplace=True)
    # Find models with Inf mean values
    rmod = d.loc[(d['mean'] == np.inf) | (
        d['mean'] == -np.inf), 'mod'].tolist()
    # Filter sc_data and fr_data based on rmod
    if rmod:
        m = sc_data['mod'].isin(rmod)
        rm2 = np.where(~m)[0]
        sc_data2 = sc_data.iloc[rm2].copy()
    else:
        sc_data2 = sc_data.copy()

    if rmod:
        m = fr_data['mod'].isin(rmod)
        rm2 = np.where(~m)[0]
        fr_data2 = fr_data.iloc[rm2].copy()
    else:
        fr_data2 = fr_data.copy()

    # Aggregate the data frames and sum over data points
    d = sc_data2.groupby(['layer', 'mod'])['ssr'].agg(
        ['mean', 'std']).reset_index()
    d.columns = ['layer', 'mod', 'sd', 'mean']

    # Print maximum value excluding Inf
    max_val = d.loc[d['mean'].replace(
        [np.inf, -np.inf], np.nan).idxmax(skipna=True)]
    print(max_val)

    # Aggregate the data frames and sum over data points
    msc = sc_data2.groupby(['layer', 'mod'])['ssr'].sum().reset_index()
    mfr = fr_data2.groupby(['layer', 'mod'])['ssr'].sum().reset_index()

    return mfr, msc


def create_figure(msc, mfr):
    plt.figure(figsize=(8, 8))

    # Geodesics over layer
    plt.subplot(2, 2, 1)
    msc.boxplot(column='ssr', by='layer', grid=False)
    plt.xlabel('Layer')
    plt.ylabel('Total geodesic change from prior layer')

    # FR curvature over layer
    plt.subplot(2, 2, 2)
    mfr.boxplot(column='ssr', by='layer', grid=False)
    plt.xlabel('Layer')
    plt.ylabel('Total FR Curvature')

    # Skip layer FR curvature vs geodesic
    plt.subplot(2, 2, 3)
    plt.scatter(msc.loc[msc['layer'] != 1, 'ssr'], mfr.loc[mfr['layer'] != mfr['layer'].max(), 'ssr'],
                c=mfr.loc[mfr['layer'] != mfr['layer'].max(), 'layer'], marker='o', cmap='viridis')
    plt.xlabel('Total geodesic change from prior layer from l-1->l')
    plt.ylabel('Total FR Curvature of l-1')
    plt.title('Layer Skip')
    plt.plot(np.unique(msc.loc[msc['layer'] != 1, 'ssr']),
             np.poly1d(np.polyfit(msc.loc[msc['layer'] != 1, 'ssr'],
                       mfr.loc[mfr['layer'] != mfr['layer'].max(), 'ssr'], 1))
             (np.unique(msc.loc[msc['layer'] != 1, 'ssr'])), color='red')

# bu satırı ekledik çünkü oluşturulan figürleri kaydetmek istiyoruz.
    plt.tight_layout()
    plt.savefig('ricci_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.close()


# Set intermediate lists to loop through the models
scalar_curvs2 = np.empty(len(model), dtype=object)
FR1 = np.empty(len(model), dtype=object)

# Loop through models
for j in range(len(model)):
    activations = model[j]
    # Compute kNN graphs
    gs1 = np.empty(len(activations), dtype=object)
    for i in range(len(activations)):
        av = activations[i]
        neigh = NearestNeighbors(metric="euclidean", n_neighbors=k)
        neigh.fit(av)
        gs1[i] = neigh.kneighbors_graph(av)
        print(gs1[i])

    neigh.fit(x_test)
    g0 = neigh.kneighbors_graph(x_test)

    # Compute FR curvatures for each act # Assuming activations is a 0-based list in Python # Assuming activations is a 0-based list in Pythonivation
    F_n_list1 = []
    for i in range(len(gs1)):
        graph_i = nx.from_scipy_sparse_array(gs1[i])
        D = np.array([graph_i.degree(i) for i in graph_i.nodes])
        F_mat = 4 - np.outer(D, D)
        a_mat = nx.to_numpy_array(graph_i)
        F_mat[a_mat == 0] = 0
        F_n1 = np.sum(F_mat, axis=1)
        F_n_list1.append(F_n1)

    Ric1 = np.empty(len(gs1), dtype=object)
    Ric1[0] = np.subtract(squareform(pdist(gs1[0].todense())),
                          squareform(pdist(g0.todense())))

    # Compute Ric1 for each i
    for i in range(1, len(gs1)):
        Ric1[i] = np.subtract(squareform(
            pdist(gs1[i].todense())), squareform(pdist(gs1[i-1].todense())))

    # Compute sum of Ric1 over data points
    sc2 = [np.apply_along_axis(lambda x: np.sum(
        x, axis=0, keepdims=True), 1, ric) for ric in Ric1]
    # Output
    scalar_curvs2[j] = sc2
    FR1[j] = F_n_list1


fr_data = corr_analysis(FR1, accuracy, x_test)
sc_data = corr_analysis(scalar_curvs2, accuracy, x_test)
mfr, msc = filter_(fr_data, sc_data)
create_figure(msc, mfr)


# Calculate correlations
aa = pearsonr(msc['ssr'], mfr['ssr'])
aa2 = pearsonr(msc.loc[msc['layer'] != 1, 'ssr'],
               mfr.loc[mfr['layer'] != mfr['layer'].max(), 'ssr'])

print("gdists: ", msc)
print("FR: ", mfr)
print("correlation: ", [aa[0], aa[1]])
print("correlation_shift: ", [aa2[0], aa2[1]])
# print("n_mods: ", sum(acc > a))
