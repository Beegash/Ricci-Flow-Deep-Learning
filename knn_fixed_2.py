#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ricci-flow style analysis (HIDDEN LAYERS ONLY - NO INPUT LAYER)
---------------------------------------------------------------
Bu script:
1. 'outputs/model_data' klasörünü tarar (Akıllı yol tespiti ile).
2. Input Layer'ı (activation_input.npy) KESİNLİKLE YÜKLEMEZ.
3. Analizi 'activation_layer_0.npy' (Hidden Layer 1) üzerinden başlatır.
   - X0 (Baseline) = Hidden Layer 1
   - Sequence = Hidden Layer 2, 3, ... 11
4. Sadece npy dosyalarını kullanır (Model eğitmez).
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, triu as sp_triu
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings

# ----------------------------
# Helpers: graphs & metrics
# ----------------------------

def build_knn_graph(X: np.ndarray, k: int) -> csr_matrix:
    """Return an undirected, unweighted kNN adjacency in CSR."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.dtype != np.float32 and X.dtype != np.float64:
        X = X.astype(np.float32, copy=False)

    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(X)
    A = knn.kneighbors_graph(X, mode="connectivity")
    A = A.maximum(A.T) # Symmetrization
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr()


def sum_shortest_paths(A: csr_matrix) -> float:
    """Compute g = sum of all-pairs shortest-path distances."""
    dist = shortest_path(A, directed=False, unweighted=True)
    iu = np.triu_indices_from(dist, k=1)
    vals = dist[iu]
    finite = np.isfinite(vals)
    if not np.all(finite):
        vals = vals[finite]
    return float(vals.sum())


def global_forman_ricci(A: csr_matrix) -> float:
    """Global Ricci coefficient Ric_l."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    A_ut = sp_triu(A, k=1).tocoo()
    curv = 4.0 - deg[A_ut.row] - deg[A_ut.col]
    return float(curv.sum())


# ----------------------------
# Analysis pipeline
# ----------------------------

def analyze_model_layers(activations: List[np.ndarray], X0: np.ndarray, k: int) -> Dict[str, np.ndarray]:
    """For one model: build graphs, compute (g_l, Ric_l)."""
    # X0 burada Hidden Layer 1 olacak (Input Layer değil)
    A0 = build_knn_graph(X0, k)
    g0 = sum_shortest_paths(A0)
    Ric0 = global_forman_ricci(A0)

    g_list = [g0]
    ric_list = [Ric0]

    for l, Xl in enumerate(activations, start=1):
        A = build_knn_graph(np.asarray(Xl), k)
        g_list.append(sum_shortest_paths(A))
        ric_list.append(global_forman_ricci(A))

    return {"g": np.array(g_list, dtype=float), "Ric": np.array(ric_list, dtype=float)}


def collect_across_models(models: List[List[np.ndarray]], X0: np.ndarray, k: int,
                          acc: np.ndarray, acc_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run analysis over models."""
    keep = np.arange(len(models)) # Debug modunda hepsini al

    rows_fr = []
    rows_sc = []
    for m in keep:
        acts = models[m]
        res = analyze_model_layers(acts, X0, k)
        g = res["g"]
        Ric = res["Ric"]
        L = len(acts)
        
        dgs = g[1:] - g[:-1]
        
        for l in range(1, L+1):
            rows_sc.append({"layer": l, "mod": int(m), "ssr": float(dgs[l-1])})
            rows_fr.append({"layer": l-1, "mod": int(m), "ssr": float(Ric[l-1])})

    msc = pd.DataFrame(rows_sc, columns=["layer", "mod", "ssr"])
    mfr = pd.DataFrame(rows_fr, columns=["layer", "mod", "ssr"])
    return mfr, msc


def correlation_report(mfr: pd.DataFrame, msc: pd.DataFrame) -> Dict[str, float]:
    """Compute overall Pearson correlations."""
    from scipy.stats import pearsonr
    
    mfr_shifted = mfr.copy()
    mfr_shifted['layer'] = mfr_shifted['layer'] + 1
    
    merged = msc.merge(mfr_shifted, on=["mod", "layer"], how="inner", suffixes=("_dg", "_fr"))
    
    if len(merged) < 2:
        return {"r_all": 0.0, "p_all": 1.0, "r_skip": 0.0, "p_skip": 1.0}

    r_all = pearsonr(merged["ssr_dg"].values, merged["ssr_fr"].values)
    
    merged_skip = merged[merged["layer"] != 1]
    if len(merged_skip) < 2:
        r_skip = (0.0, 1.0)
    else:
        r_skip = pearsonr(merged_skip["ssr_dg"].values, merged_skip["ssr_fr"].values)
        
    return {
        "r_all": float(r_all[0]), "p_all": float(r_all[1]),
        "r_skip": float(r_skip[0]), "p_skip": float(r_skip[1]),
    }


def plot_summary(msc: pd.DataFrame, mfr: pd.DataFrame, out_png: str | None = None) -> None:
    """Recreate the 2x2 style overview used by the authors."""
    fig = plt.figure(figsize=(10, 10))

    # Plot 1: Delta g
    ax1 = plt.subplot(2, 2, 1)
    if not msc.empty:
        msc.boxplot(column='ssr', by='layer', grid=False, ax=ax1)
    ax1.set_xlabel('Layer Transition (Hidden Only)')
    ax1.set_ylabel('Δg')
    ax1.set_title('Change in Geodesic Mass')

    # Plot 2: Ricci
    ax2 = plt.subplot(2, 2, 2)
    if not mfr.empty:
        mfr.boxplot(column='ssr', by='layer', grid=False, ax=ax2)
    ax2.set_xlabel('Layer index')
    ax2.set_ylabel('Ricci Curvature')
    ax2.set_title('Ricci Curvature')

    # Plot 3: Correlation
    ax3 = plt.subplot(2, 2, 3)
    mfr_shifted = mfr.copy()
    mfr_shifted['layer'] = mfr_shifted['layer'] + 1
    merged = msc.merge(mfr_shifted, on=['mod', 'layer'], suffixes=('_dg', '_fr'))
    
    if not merged.empty:
        sc = ax3.scatter(merged['ssr_dg'], merged['ssr_fr'], c=merged['layer'], marker='o', cmap='viridis')
        plt.colorbar(sc, ax=ax3, label='Layer')
        ax3.set_xlabel('Δg')
        ax3.set_ylabel('Ric (Previous Layer)')
        ax3.set_title('Correlation (Hidden Layers Only)')

        if len(merged) >= 2:
            xs = merged['ssr_dg'].values
            ys = merged['ssr_fr'].values
            z = np.polyfit(xs, ys, 1)
            p = np.poly1d(z)
            xsu = np.unique(xs)
            ax3.plot(xsu, p(xsu), "r--", alpha=0.7)

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved figure to {out_png}")
    else:
        plt.show()
    plt.close(fig)


# ----------------------------
# MAIN ADAPTER FOR HIDDEN LAYERS ONLY
# ----------------------------
if __name__ == "__main__":
    print("="*70)
    print("RUNNING ANALYSIS ON HIDDEN LAYERS ONLY (SKIPPING INPUT)")
    print("="*70)

    # 1. Path Detection (FIXED)
    current_dir = os.getcwd()
    
    # Olası veri yollarını sırayla kontrol et
    possible_paths = [
        os.path.join(current_dir, 'outputs'), # clean/outputs
        '/Users/cihan/Documents/GitHub/ricci_flow_laboratory/outputs', # Hardcoded absolute path
        os.path.join(os.path.dirname(current_dir), 'ricci_flow_laboratory', 'outputs') # Sibling folder
    ]

    OUTPUT_DIR = None
    MODELS_DIR = None

    for path in possible_paths:
        candidate = os.path.join(path, 'model_data')
        if os.path.exists(candidate):
            OUTPUT_DIR = path
            MODELS_DIR = candidate
            print(f"--> [FOUND] Data directory found at: {MODELS_DIR}")
            break
    
    if MODELS_DIR is None:
        print(f"[ERROR] Could not find 'outputs/model_data' in any likely location.")
        print(f"Searched in: {possible_paths}")
        raise FileNotFoundError("Please check where your 'model_data' folder is located.")

    # 2. LOAD DATA (LOGIC DEĞİŞTİ)
    # Input layer (activation_input.npy) YÜKLENMİYOR.
    # Onun yerine Hidden Layer 1 (activation_layer_0.npy), bizim yeni "X0"ımız oluyor.
    
    try:
        # Layer 0 (İlk Hidden Layer) -> Bu artık bizim BASELINE'ımız
        x0_path = os.path.join(MODELS_DIR, 'activation_layer_0.npy')
        if not os.path.exists(x0_path):
            raise FileNotFoundError(f"Hidden Layer 1 (activation_layer_0.npy) bulunamadı! Yol: {x0_path}")
            
        X0 = np.load(x0_path)
        print(f"--> Loaded X0 (Hidden Layer 1): {X0.shape}")
        
    except Exception as e:
        print(f"[ERROR] Could not load start layer: {e}")
        exit()

    # 3. Load Remaining Hidden Layers (Layer 1 to N)
    activations = []
    layer_idx = 1 # Layer 0'ı X0 yaptık, o yüzden 1'den başlıyoruz
    while True:
        file_path = os.path.join(MODELS_DIR, f'activation_layer_{layer_idx}.npy')
        if os.path.exists(file_path):
            act = np.load(file_path)
            activations.append(act)
            # print(f"    Loaded Hidden Layer {layer_idx+1}")
            layer_idx += 1
        else:
            break
    
    print(f"--> Loaded {len(activations)} additional hidden layers sequence.")
    print(f"--> Total analysis depth: {len(activations) + 1} layers.")

    # Model list formatı
    model_list = [activations] 
    
    # Try to load real accuracy if available
    accuracy_path = os.path.join(MODELS_DIR, 'accuracy.npy')
    if os.path.exists(accuracy_path):
        accuracy = np.load(accuracy_path)
        print(f"--> Model Accuracy: {accuracy[0]:.4f} (loaded from file)")
    else:
        accuracy = np.array([0.99])  
        print(f"--> Model Accuracy: Not available (using placeholder)") 

    # 4. Run Parameters
    K = 350
    ACC_THRESHOLD = 0.50 

    print(f"--> Starting Analysis Pipeline (k={K})...")

    # 5. Execute Analysis
    mfr, msc = collect_across_models(models=model_list, X0=X0, k=K, acc=accuracy, acc_threshold=ACC_THRESHOLD)
    
    # 6. Report
    stats = correlation_report(mfr, msc)
    print("\n" + "="*70)
    print("RESULTS (HIDDEN LAYERS ONLY)")
    print("="*70)
    print("[STATS] Pearson r (all):   r=%.4f, p=%.3e" % (stats['r_all'], stats['p_all']))
    
    # 7. Plot
    plot_summary(msc, mfr, out_png=os.path.join(OUTPUT_DIR, "hidden_only_plot.png"))
    
    # 8. Save CSVs
    mfr.to_csv(os.path.join(OUTPUT_DIR, "mfr_hidden.csv"), index=False)
    msc.to_csv(os.path.join(OUTPUT_DIR, "msc_hidden.csv"), index=False)
    print(f"[INFO] Saved CSVs and Plot to {OUTPUT_DIR}")