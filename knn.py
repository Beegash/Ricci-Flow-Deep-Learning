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

# Mevcut çalışma dizinini kullan
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load file and model
model = np.load("model_predict.npy",allow_pickle = True)
accuracy = np.load("accuracy.npy")
# Load x_test
x_test = np.array(pd.read_csv("x_test.csv", header = None))

k = 30
a = 0.98


def corr_analysis(scalar_curvs1, accuracy, x_test):
    # Generate FR curvature data frame
    accuracy = np.asarray(accuracy)
    acc = accuracy  # Tüm accuracy değerlerini kullan
    
    # Accuracy > a olan modelleri bul
    good_models = np.where(acc > a)[0]
    
    # Tüm verileri birleştir
    all_data = []
    
    for model_idx in good_models:
        model_data = scalar_curvs1[model_idx]  # Bu modelin tüm layerları
        for layer_idx, layer_data in enumerate(model_data):
            # Her layer için bir DataFrame satırı oluştur
            for point_idx in range(len(layer_data)):
                all_data.append({
                    'ssr': float(layer_data[point_idx]),  # float tipine çevir
                    'layer': int(layer_idx + 1),  # Layer numarası 1'den başlar
                    'mod': int(model_idx + 1)  # Model numarası 1'den başlar
                })
    
    data = pd.DataFrame(all_data)
    # Veri tiplerini açıkça belirt
    data['ssr'] = data['ssr'].astype(float)
    data['layer'] = data['layer'].astype(int)
    data['mod'] = data['mod'].astype(int)
    return data
    

def filter_(fr_data, sc_data):
    # Deal with infinite geodesics by removing models with these
    d = sc_data.groupby(['layer', 'mod'])['ssr'].agg(['mean', 'std']).reset_index()
    # Replace Inf values with NaN for further processing
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Remove rows with NaN in the mean column
    d.dropna(subset=['mean'], inplace=True)
    # Find models with Inf mean values
    rmod = d.loc[(d['mean'] == np.inf) | (d['mean'] == -np.inf), 'mod'].tolist()
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
    d = sc_data2.groupby(['layer', 'mod'])['ssr'].agg(['mean', 'std']).reset_index()
    d.columns = ['layer', 'mod', 'sd', 'mean']

    # Print maximum value excluding Inf
    max_val = d.loc[d['mean'].replace([np.inf, -np.inf], np.nan).idxmax(skipna=True)]
    print(max_val)
    
    # Aggregate the data frames and sum over data points
    msc = sc_data2.groupby(['layer', 'mod'])['ssr'].sum().reset_index()
    mfr = fr_data2.groupby(['layer', 'mod'])['ssr'].sum().reset_index()
    
    # Veri tiplerini kontrol et ve düzelt
    msc['ssr'] = pd.to_numeric(msc['ssr'], errors='coerce')
    mfr['ssr'] = pd.to_numeric(mfr['ssr'], errors='coerce')
    
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
             np.poly1d(np.polyfit(msc.loc[msc['layer'] != 1, 'ssr'], mfr.loc[mfr['layer'] != mfr['layer'].max(), 'ssr'], 1))
             (np.unique(msc.loc[msc['layer'] != 1, 'ssr'])), color='red')

    plt.tight_layout()
    plt.savefig('ricci_flow_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Grafik kaydedildi: ricci_flow_analysis.png")
    plt.show()  # Grafiği göster


# Set intermediate lists to loop through the models
scalar_curvs2 = np.empty(len(model), dtype = object)
FR1 = np.empty(len(model), dtype = object)

# Loop through models
for j in range(len(model)):
    activations = model[j]
    # Compute kNN graphs
    gs1 = np.empty(len(activations), dtype = object)
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

    Ric1 = np.empty(len(gs1), dtype = object)
    Ric1[0] = np.subtract(squareform(pdist(gs1[0].todense())), squareform(pdist(g0.todense())))
    
    # Compute Ric1 for each i
    for i in range(1, len(gs1)):
        Ric1[i] = np.subtract(squareform(pdist(gs1[i].todense())), squareform(pdist(gs1[i-1].todense())))
    
    # Compute sum of Ric1 over data points
    sc2 = [np.apply_along_axis(lambda x: np.sum(x, axis=0, keepdims=True), 1, ric) for ric in Ric1]
    # Output
    scalar_curvs2[j] = sc2
    FR1[j] = F_n_list1
    
    
fr_data = corr_analysis(FR1, accuracy, x_test)
sc_data = corr_analysis(scalar_curvs2, accuracy, x_test)
mfr, msc = filter_(fr_data, sc_data)
create_figure(msc, mfr)


# Calculate correlations
aa = pearsonr(msc['ssr'], mfr['ssr'])
aa2 = pearsonr(msc.loc[msc['layer'] != 1, 'ssr'], mfr.loc[mfr['layer'] != mfr['layer'].max(), 'ssr'])

print("gdists: ", msc)
print("FR: ", mfr)
print("correlation: ", [aa[0], aa[1]])
print("correlation_shift: ", [aa2[0], aa2[1]])
#print("n_mods: ", sum(acc > a))
