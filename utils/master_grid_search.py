#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASTER GRID SEARCH - Deep Learning as Ricci Flow
Comprehensive hyperparameter search matching the paper's methodology

This script performs a complete grid search over:
- 6 DNN architectures (narrow/wide/bottleneck × shallow/deep)
- 7 datasets (MNIST 1v7, 6v8, fMNIST sandals/boots, shirts/coats, Synthetic A/B/C)
- k-NN neighbor values (paper's ranges)

Strategy: Train b=70 models ONCE per (architecture, dataset) pair,
         then REUSE for all k values to save computation time.

Based on: training.py and knn_fixed.py
Paper: "Deep Learning as Ricci Flow" (Baptista et al., 2024)
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs
from scipy.sparse import csr_matrix, triu as sp_triu
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import List, Dict, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - Paper's Parameters
# ============================================================================

# Fixed parameters
B_VALUE = 70  # Number of models to train (fixed, train once)
ACC_THRESHOLD = 0.98  # Accuracy threshold for analysis
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Architecture configurations
ARCHITECTURES = {
    'narrow_5': {'width': 25, 'depth': 5, 'bottleneck': False},
    'narrow_11': {'width': 25, 'depth': 11, 'bottleneck': False},
    'wide_5': {'width': 50, 'depth': 5, 'bottleneck': False},
    'wide_11': {'width': 50, 'depth': 11, 'bottleneck': False},
    'bottleneck_5': {'width': 50, 'depth': 5, 'bottleneck': True},  # 50→25
    'bottleneck_11': {'width': 50, 'depth': 11, 'bottleneck': True},  # 50→25
}

# Dataset configurations
DATASETS = {
    'mnist_1_vs_7': {'type': 'mnist', 'classes': [1, 7]},
    'mnist_6_vs_8': {'type': 'mnist', 'classes': [6, 8]},
    'fmnist_sandals_vs_boots': {'type': 'fmnist', 'classes': [5, 9]},
    'fmnist_shirts_vs_coats': {'type': 'fmnist', 'classes': [6, 8]},
    'synthetic_a': {'type': 'synthetic', 'variant': 'A'},
    'synthetic_b': {'type': 'synthetic', 'variant': 'B'},
    'synthetic_c': {'type': 'synthetic', 'variant': 'C'},
}

# k-NN values (paper's ranges)
K_VALUES_SYNTHETIC = [6, 7, 9, 10, 15, 18, 20, 30, 50, 90, 100]
K_VALUES_REAL = [325, 400, 500]

# Output directory
BASE_OUTPUT_DIR = os.path.join(os.getcwd(), 'output_layers')
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_mnist_data(classes: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST data for binary classification."""
    data_path = os.path.join(os.getcwd(), 'extracted_datasets', 'extracted_data_mnist')
    
    # Load extracted data
    train_class1 = pd.read_csv(os.path.join(data_path, f"train_{classes[0]}.csv"))
    train_class2 = pd.read_csv(os.path.join(data_path, f"train_{classes[1]}.csv"))
    test_class1 = pd.read_csv(os.path.join(data_path, f"test_{classes[0]}.csv"))
    test_class2 = pd.read_csv(os.path.join(data_path, f"test_{classes[1]}.csv"))
    
    # Combine
    x_train = pd.concat([train_class1, train_class2], ignore_index=True)
    x_test = pd.concat([test_class1, test_class2], ignore_index=True)
    
    # Extract labels and features
    y_train = x_train['label'].values
    y_test = x_test['label'].values
    x_train = x_train.iloc[:, 1:].values
    x_test = x_test.iloc[:, 1:].values
    
    # Convert labels to binary (0, 1)
    y_train = (y_train == classes[1]).astype(int)
    y_test = (y_test == classes[1]).astype(int)
    
    return x_train, y_train, x_test, y_test


def load_fmnist_data(classes: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Fashion-MNIST data for binary classification."""
    data_path = os.path.join(os.getcwd(), 'extracted_datasets', 'extracted_data_fmnist')
    
    # Load extracted data
    train_class1 = pd.read_csv(os.path.join(data_path, f"train{classes[0]}.csv"))
    train_class2 = pd.read_csv(os.path.join(data_path, f"train{classes[1]}.csv"))
    test_class1 = pd.read_csv(os.path.join(data_path, f"test{classes[0]}.csv"))
    test_class2 = pd.read_csv(os.path.join(data_path, f"test{classes[1]}.csv"))
    
    # Combine
    x_train = pd.concat([train_class1, train_class2], ignore_index=True)
    x_test = pd.concat([test_class1, test_class2], ignore_index=True)
    
    # Extract labels and features
    y_train = x_train['label'].values
    y_test = x_test['label'].values
    x_train = x_train.iloc[:, 1:].values
    x_test = x_test.iloc[:, 1:].values
    
    # Convert labels to binary (0, 1)
    y_train = (y_train == classes[1]).astype(int)
    y_test = (y_test == classes[1]).astype(int)
    
    return x_train, y_train, x_test, y_test


def generate_synthetic_data(variant: str, n_train: int = 1000, n_test: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic datasets A, B, or C from the paper."""
    np.random.seed(42)
    
    if variant == 'A':
        # Entangled spirals (previously investigated - Data set A)
        theta_train = np.sqrt(np.random.rand(n_train)) * 2 * np.pi
        theta_test = np.sqrt(np.random.rand(n_test)) * 2 * np.pi
        
        r_a = 2 * theta_train + np.pi
        x1_train = r_a * np.cos(theta_train)
        y1_train = r_a * np.sin(theta_train)
        
        r_b = -2 * theta_train - np.pi
        x2_train = r_b * np.cos(theta_train)
        y2_train = r_b * np.sin(theta_train)
        
        r_a_test = 2 * theta_test + np.pi
        x1_test = r_a_test * np.cos(theta_test)
        y1_test = r_a_test * np.sin(theta_test)
        
        r_b_test = -2 * theta_test - np.pi
        x2_test = r_b_test * np.cos(theta_test)
        y2_test = r_b_test * np.sin(theta_test)
        
        x_train = np.vstack([np.column_stack([x1_train, y1_train]),
                             np.column_stack([x2_train, y2_train])])
        y_train = np.hstack([np.zeros(n_train), np.ones(n_train)])
        
        x_test = np.vstack([np.column_stack([x1_test, y1_test]),
                            np.column_stack([x2_test, y2_test])])
        y_test = np.hstack([np.zeros(n_test), np.ones(n_test)])
        
    elif variant == 'B':
        # Intersecting manifolds that cannot become linearly separable
        X1, _ = make_blobs(n_samples=n_train, centers=[[2, 2]], cluster_std=0.8, random_state=42)
        X2, _ = make_blobs(n_samples=n_train, centers=[[2, 2]], cluster_std=1.5, random_state=43)
        
        x_train = np.vstack([X1, X2])
        y_train = np.hstack([np.zeros(n_train), np.ones(n_train)])
        
        X1_test, _ = make_blobs(n_samples=n_test, centers=[[2, 2]], cluster_std=0.8, random_state=44)
        X2_test, _ = make_blobs(n_samples=n_test, centers=[[2, 2]], cluster_std=1.5, random_state=45)
        
        x_test = np.vstack([X1_test, X2_test])
        y_test = np.hstack([np.zeros(n_test), np.ones(n_test)])
        
    elif variant == 'C':
        # Intersecting linear manifolds
        theta = np.linspace(0, np.pi, n_train)
        X1_train = np.column_stack([theta, np.sin(theta)])
        X2_train = np.column_stack([theta, -np.sin(theta)])
        
        theta_test = np.linspace(0, np.pi, n_test)
        X1_test = np.column_stack([theta_test, np.sin(theta_test)])
        X2_test = np.column_stack([theta_test, -np.sin(theta_test)])
        
        x_train = np.vstack([X1_train, X2_train]) + np.random.randn(2*n_train, 2) * 0.1
        y_train = np.hstack([np.zeros(n_train), np.ones(n_train)])
        
        x_test = np.vstack([X1_test, X2_test]) + np.random.randn(2*n_test, 2) * 0.1
        y_test = np.hstack([np.zeros(n_test), np.ones(n_test)])
    
    # Shuffle
    train_idx = np.random.permutation(len(x_train))
    test_idx = np.random.permutation(len(x_test))
    
    return x_train[train_idx], y_train[train_idx], x_test[test_idx], y_test[test_idx]


def load_dataset(dataset_name: str, dataset_config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset based on configuration."""
    dataset_type = dataset_config['type']
    
    if dataset_type == 'mnist':
        return load_mnist_data(dataset_config['classes'])
    elif dataset_type == 'fmnist':
        return load_fmnist_data(dataset_config['classes'])
    elif dataset_type == 'synthetic':
        return generate_synthetic_data(dataset_config['variant'])
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


# ============================================================================
# MODEL TRAINING FUNCTIONS (from training.py)
# ============================================================================

def build_model(architecture_config: Dict, input_dim: int) -> Sequential:
    """Build DNN model based on architecture configuration."""
    model = Sequential()
    
    width = architecture_config['width']
    depth = architecture_config['depth']
    is_bottleneck = architecture_config['bottleneck']
    
    # First layer
    if is_bottleneck:
        model.add(Dense(units=50, activation='relu', input_shape=(input_dim,)))
    else:
        model.add(Dense(units=width, activation='relu', input_shape=(input_dim,)))
    
    # Hidden layers
    for _ in range(depth - 1):
        if is_bottleneck:
            model.add(Dense(units=25, activation='relu'))
        else:
            model.add(Dense(units=width, activation='relu'))
    
    # Output layer
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Compile
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])
    
    return model


def train_models(architecture_config: Dict, x_train: np.ndarray, y_train: np.ndarray,
                 x_test: np.ndarray, y_test: np.ndarray, b: int,
                 output_dir: str) -> Dict:
    """Train b models and save outputs."""
    
    print(f"  Training {b} models...")
    accuracy_list = []
    model_predict = np.empty(b, dtype=object)
    
    input_dim = x_train.shape[1]
    
    with tqdm(total=b, desc="    Progress", leave=False,
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for j in range(b):
            # Build model
            model = build_model(architecture_config, input_dim)
            
            # Train
            model.fit(x_train, y_train,
                     epochs=EPOCHS,
                     batch_size=BATCH_SIZE,
                     validation_split=VALIDATION_SPLIT,
                     verbose=0)
            
            # Evaluate
            acc = model.evaluate(x_test, y_test, verbose=0)[1]
            accuracy_list.append(acc)
            
            # Extract activations (all hidden layers)
            activations = []
            current_input = x_test
            for layer in model.layers[:-1]:  # Exclude output layer
                current_output = layer(current_input)
                activations.append(current_output.numpy())
                current_input = current_output
            
            model_predict[j] = activations
            pbar.update(1)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "model_predict.npy"), model_predict)
    np.save(os.path.join(output_dir, "accuracy.npy"), np.array(accuracy_list))
    pd.DataFrame(x_test).to_csv(os.path.join(output_dir, "x_test.csv"), index=False, header=None)
    pd.DataFrame(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index=False, header=None)
    
    mean_acc = np.mean(accuracy_list)
    n_passing = np.sum(np.array(accuracy_list) > ACC_THRESHOLD)
    
    print(f"  ✓ Training complete: mean_acc={mean_acc:.4f}, passing={n_passing}/{b}")
    
    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': np.std(accuracy_list),
        'n_models_passing': n_passing,
        'accuracy_list': accuracy_list
    }


# ============================================================================
# RICCI FLOW ANALYSIS FUNCTIONS (from knn_fixed.py)
# ============================================================================

def build_knn_graph(X: np.ndarray, k: int) -> csr_matrix:
    """Return an undirected, unweighted kNN adjacency in CSR."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.dtype != np.float32 and X.dtype != np.float64:
        X = X.astype(np.float32, copy=False)
    
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(X)
    A = knn.kneighbors_graph(X, mode="connectivity")
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr()


def sum_shortest_paths(A: csr_matrix) -> float:
    """Compute g = sum of all-pairs shortest-path distances, i<j."""
    dist = shortest_path(A, directed=False, unweighted=True)
    iu = np.triu_indices_from(dist, k=1)
    vals = dist[iu]
    finite = np.isfinite(vals)
    if not np.all(finite):
        vals = vals[finite]
    return float(vals.sum())


def global_forman_ricci(A: csr_matrix) -> float:
    """Global Ricci coefficient Ric_l = sum over edges of Forman-Ricci curvatures."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    A_ut = sp_triu(A, k=1).tocoo()
    curv = 4.0 - deg[A_ut.row] - deg[A_ut.col]
    return float(curv.sum())


def analyze_model_layers(activations: List[np.ndarray], X0: np.ndarray, k: int) -> Dict[str, np.ndarray]:
    """For one model: build graphs, compute (g_l, Ric_l) for l=0..L."""
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
    """Run analysis over models passing the accuracy threshold."""
    keep = np.where(acc > acc_threshold)[0]
    if keep.size == 0:
        print(f"    WARNING: No models exceed threshold {acc_threshold}. Using all models.")
        keep = np.arange(len(models))
    
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


def correlation_report(mfr: pd.DataFrame, msc: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute overall Pearson correlations and per-layer correlations."""
    mfr_shifted = mfr.copy()
    mfr_shifted['layer'] = mfr_shifted['layer'] + 1
    merged = msc.merge(mfr_shifted, on=["mod", "layer"], how="inner", suffixes=("_dg", "_fr"))
    
    if len(merged) < 2:
        per_layer_df = pd.DataFrame(columns=['layer', 'correlation', 'p_value', 'n_samples'])
        return {"r_all": np.nan, "p_all": np.nan, "r_skip": np.nan, "p_skip": np.nan}, per_layer_df
    
    r_all = pearsonr(merged["ssr_dg"].values, merged["ssr_fr"].values)
    
    merged_skip = merged[merged["layer"] != 1]
    if len(merged_skip) < 2:
        r_skip = (np.nan, np.nan)
    else:
        r_skip = pearsonr(merged_skip["ssr_dg"].values, merged_skip["ssr_fr"].values)
    
    # Compute per-layer correlations
    per_layer_correlations = []
    layers = sorted(merged['layer'].unique())
    
    for layer in layers:
        layer_data = merged[merged['layer'] == layer]
        if len(layer_data) >= 2:
            r_layer, p_layer = pearsonr(layer_data['ssr_dg'].values, layer_data['ssr_fr'].values)
            per_layer_correlations.append({
                'layer': int(layer),
                'correlation': float(r_layer),
                'p_value': float(p_layer),
                'n_samples': len(layer_data)
            })
        else:
            per_layer_correlations.append({
                'layer': int(layer),
                'correlation': np.nan,
                'p_value': np.nan,
                'n_samples': len(layer_data)
            })
    
    per_layer_df = pd.DataFrame(per_layer_correlations)
    
    return {
        "r_all": float(r_all[0]), "p_all": float(r_all[1]),
        "r_skip": float(r_skip[0]), "p_skip": float(r_skip[1]),
    }, per_layer_df


def plot_summary(msc: pd.DataFrame, mfr: pd.DataFrame, per_layer_corr: pd.DataFrame, out_png: str) -> None:
    """Generate summary plots including per-layer correlation visualization."""
    fig = plt.figure(figsize=(12, 10))
    
    ax1 = plt.subplot(2, 3, 1)
    msc.boxplot(column='ssr', by='layer', grid=False, ax=ax1)
    ax1.set_xlabel('Layer l')
    ax1.set_ylabel('Δg_l = g_l - g_{l-1}')
    ax1.set_title('Geodesic Change per Layer')
    
    ax2 = plt.subplot(2, 3, 2)
    mfr.boxplot(column='ssr', by='layer', grid=False, ax=ax2)
    ax2.set_xlabel('Layer index (for Ric_{l-1})')
    ax2.set_ylabel('Global Forman-Ricci (Ric_{l-1})')
    ax2.set_title('Ricci Curvature per Layer')
    
    ax3 = plt.subplot(2, 3, 3)
    mfr_shifted = mfr.copy()
    mfr_shifted['layer'] = mfr_shifted['layer'] + 1
    merged = msc.merge(mfr_shifted, on=['mod', 'layer'], suffixes=('_dg', '_fr'))
    
    if len(merged) > 0:
        sc = ax3.scatter(merged['ssr_dg'], merged['ssr_fr'], c=merged['layer'], 
                        marker='o', cmap='viridis', alpha=0.6)
        ax3.set_xlabel('Δg_l')
        ax3.set_ylabel('Ric_{l-1}')
        ax3.set_title('Layer-skip correlation (l vs l-1)')
        plt.colorbar(sc, ax=ax3, label='Layer')
        
        if len(merged) >= 2:
            xs = merged['ssr_dg'].values
            ys = merged['ssr_fr'].values
            z = np.polyfit(xs, ys, 1)
            p = np.poly1d(z)
            xsu = np.unique(xs)
            ax3.plot(xsu, p(xsu), 'r-', linewidth=2, alpha=0.8)
    
    # New: Per-layer correlation plot
    ax4 = plt.subplot(2, 3, 4)
    if len(per_layer_corr) > 0 and not per_layer_corr['correlation'].isna().all():
        valid_data = per_layer_corr.dropna(subset=['correlation'])
        if len(valid_data) > 0:
            ax4.plot(valid_data['layer'], valid_data['correlation'], 'o-', linewidth=2, 
                    markersize=8, color='steelblue', label='Correlation')
            ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero correlation')
            ax4.set_xlabel('Layer')
            ax4.set_ylabel('Pearson Correlation (r)')
            ax4.set_title('Per-Layer Correlation: Δg_l vs Ric_{l-1}')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            ax4.set_ylim([-1.1, 1.1])
    
    # New: Per-layer p-value plot (significance)
    ax5 = plt.subplot(2, 3, 5)
    if len(per_layer_corr) > 0 and not per_layer_corr['p_value'].isna().all():
        valid_data = per_layer_corr.dropna(subset=['p_value'])
        if len(valid_data) > 0:
            # Plot -log10(p-value) for better visualization
            log_p = -np.log10(valid_data['p_value'].clip(lower=1e-10))
            ax5.plot(valid_data['layer'], log_p, 's-', linewidth=2, 
                    markersize=8, color='coral', label='-log10(p-value)')
            ax5.axhline(y=-np.log10(0.05), color='r', linestyle='--', alpha=0.5, 
                       label='p=0.05 threshold')
            ax5.axhline(y=-np.log10(0.01), color='orange', linestyle='--', alpha=0.5, 
                       label='p=0.01 threshold')
            ax5.set_xlabel('Layer')
            ax5.set_ylabel('-log10(p-value)')
            ax5.set_title('Statistical Significance per Layer')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
    
    # New: Combined correlation and sample size
    ax6 = plt.subplot(2, 3, 6)
    if len(per_layer_corr) > 0:
        valid_data = per_layer_corr.dropna(subset=['correlation'])
        if len(valid_data) > 0:
            # Create a scatter plot with size proportional to sample size
            sizes = valid_data['n_samples'] * 10  # Scale for visibility
            scatter = ax6.scatter(valid_data['layer'], valid_data['correlation'], 
                                 s=sizes, c=valid_data['correlation'], 
                                 cmap='coolwarm', alpha=0.6, 
                                 vmin=-1, vmax=1, edgecolors='black', linewidth=1)
            ax6.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax6.set_xlabel('Layer')
            ax6.set_ylabel('Pearson Correlation (r)')
            ax6.set_title('Correlation by Layer\n(size = sample count)')
            ax6.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax6, label='Correlation')
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_analysis(k: int, model_dir: str, output_dir: str) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Run kNN analysis with specific k value on pre-trained models."""
    # Load artifacts
    model = np.load(os.path.join(model_dir, "model_predict.npy"), allow_pickle=True)
    accuracy = np.asarray(np.load(os.path.join(model_dir, "accuracy.npy")))
    X0 = np.array(pd.read_csv(os.path.join(model_dir, "x_test.csv"), header=None))
    
    # Run analysis
    mfr, msc = collect_across_models(models=model, X0=X0, k=k, acc=accuracy, acc_threshold=ACC_THRESHOLD)
    stats, per_layer_corr = correlation_report(mfr, msc)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    mfr.to_csv(os.path.join(output_dir, "mfr.csv"), index=False)
    msc.to_csv(os.path.join(output_dir, "msc.csv"), index=False)
    per_layer_corr.to_csv(os.path.join(output_dir, "per_layer_correlations.csv"), index=False)
    plot_summary(msc, mfr, per_layer_corr, out_png=os.path.join(output_dir, "analysis_plot.png"))
    
    return stats, per_layer_corr


# ============================================================================
# MASTER GRID SEARCH
# ============================================================================

def main():
    print("=" * 100)
    print("MASTER GRID SEARCH - Deep Learning as Ricci Flow")
    print("=" * 100)
    print(f"Architectures: {len(ARCHITECTURES)} ({', '.join(ARCHITECTURES.keys())})")
    print(f"Datasets: {len(DATASETS)} ({', '.join(DATASETS.keys())})")
    print(f"k-values: {len(K_VALUES_SYNTHETIC)} (synthetic), {len(K_VALUES_REAL)} (real)")
    print(f"Models per combo: b={B_VALUE}")
    print(f"Total training sessions: {len(ARCHITECTURES) * len(DATASETS)}")
    print(f"Total analysis runs: ~{len(ARCHITECTURES) * 4 * len(K_VALUES_REAL) + len(ARCHITECTURES) * 3 * len(K_VALUES_SYNTHETIC)}")
    print("=" * 100)
    
    # Master results storage
    all_results = []
    
    # Total combinations
    total_combos = len(ARCHITECTURES) * len(DATASETS)
    combo_idx = 0
    
    # Grid search
    for arch_name, arch_config in ARCHITECTURES.items():
        for dataset_name, dataset_config in DATASETS.items():
            combo_idx += 1
            
            print(f"\n{'='*100}")
            print(f"[{combo_idx}/{total_combos}] {arch_name} + {dataset_name}")
            print(f"{'='*100}")
            
            # Create output directory
            combo_dir = os.path.join(BASE_OUTPUT_DIR, f"{arch_name}_{dataset_name}")
            model_dir = os.path.join(combo_dir, f"models_b{B_VALUE}")
            
            # Step 1: Load data
            print(f"[1/3] Loading dataset: {dataset_name}...")
            try:
                x_train, y_train, x_test, y_test = load_dataset(dataset_name, dataset_config)
                print(f"  Train: {x_train.shape}, Test: {x_test.shape}")
            except Exception as e:
                print(f"  ERROR loading dataset: {e}")
                print(f"  Skipping this combination...")
                continue
            
            # Step 2: Train models (ONCE)
            print(f"[2/3] Training models...")
            start_train = time.time()
            
            if os.path.exists(os.path.join(model_dir, "model_predict.npy")):
                print(f"  Models already trained. Loading from {model_dir}...")
                accuracy = np.load(os.path.join(model_dir, "accuracy.npy"))
                train_stats = {
                    'mean_accuracy': np.mean(accuracy),
                    'std_accuracy': np.std(accuracy),
                    'n_models_passing': np.sum(accuracy > ACC_THRESHOLD),
                }
                train_time = 0
            else:
                train_stats = train_models(arch_config, x_train, y_train, x_test, y_test, 
                                          B_VALUE, model_dir)
                train_time = time.time() - start_train
            
            # Step 3: Run analysis for all k values
            print(f"[3/3] Running Ricci flow analysis for all k values...")
            
            # Determine k values based on dataset type
            if dataset_config['type'] == 'synthetic':
                k_values = K_VALUES_SYNTHETIC
            else:
                k_values = K_VALUES_REAL
            
            # Storage for this specific combination
            specific_results = []
            
            with tqdm(total=len(k_values), desc="  k-value sweep", 
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                for k in k_values:
                    analysis_dir = os.path.join(combo_dir, f"analysis_k{k}")
                    
                    start_analysis = time.time()
                    stats, per_layer_corr = run_analysis(k, model_dir, analysis_dir)
                    analysis_time = time.time() - start_analysis
                    
                    # Store result
                    result = {
                        'architecture': arch_name,
                        'depth': arch_config['depth'],
                        'width': arch_config['width'],
                        'bottleneck': arch_config['bottleneck'],
                        'dataset': dataset_name,
                        'dataset_type': dataset_config['type'],
                        'k': k,
                        'b': B_VALUE,
                        'r_all': stats['r_all'],
                        'p_all': stats['p_all'],
                        'r_skip': stats['r_skip'],
                        'p_skip': stats['p_skip'],
                        'mean_accuracy': train_stats['mean_accuracy'],
                        'std_accuracy': train_stats['std_accuracy'],
                        'n_models_passing': train_stats['n_models_passing'],
                        'train_time_s': train_time,
                        'analysis_time_s': analysis_time,
                    }
                    
                    all_results.append(result)
                    specific_results.append({
                        'k': k,
                        'r_all': stats['r_all'],
                        'p_all': stats['p_all'],
                        'r_skip': stats['r_skip'],
                        'p_skip': stats['p_skip'],
                        'analysis_time_s': analysis_time,
                    })
                    
                    pbar.update(1)
            
            # Save specific summary for this combination
            specific_df = pd.DataFrame(specific_results)
            specific_summary_path = os.path.join(combo_dir, f"{arch_name}_{dataset_name}_summary.csv")
            specific_df.to_csv(specific_summary_path, index=False)
            print(f"  ✓ Saved specific summary: {specific_summary_path}")
            
            # Save intermediate master summary
            master_df = pd.DataFrame(all_results)
            master_path = os.path.join(BASE_OUTPUT_DIR, "MASTER_GRID_SEARCH_SUMMARY.csv")
            master_df.to_csv(master_path, index=False)
    
    # ========================================================================
    # FINAL SUMMARY GENERATION
    # ========================================================================
    
    print(f"\n{'='*100}")
    print("GENERATING FINAL SUMMARIES")
    print(f"{'='*100}")
    
    master_df = pd.DataFrame(all_results)
    
    # 1. Master summary
    master_path = os.path.join(BASE_OUTPUT_DIR, "MASTER_GRID_SEARCH_SUMMARY.csv")
    master_df.to_csv(master_path, index=False)
    print(f"✓ Master summary saved: {master_path}")
    
    # 2. Summary by architecture
    by_arch = master_df.groupby('architecture').agg({
        'r_all': ['mean', 'std', 'min', 'max'],
        'r_skip': ['mean', 'std', 'min', 'max'],
        'mean_accuracy': ['mean', 'std'],
    }).round(4)
    by_arch_path = os.path.join(BASE_OUTPUT_DIR, "by_architecture_summary.csv")
    by_arch.to_csv(by_arch_path)
    print(f"✓ By-architecture summary saved: {by_arch_path}")
    
    # 3. Summary by dataset
    by_dataset = master_df.groupby('dataset').agg({
        'r_all': ['mean', 'std', 'min', 'max'],
        'r_skip': ['mean', 'std', 'min', 'max'],
        'mean_accuracy': ['mean', 'std'],
    }).round(4)
    by_dataset_path = os.path.join(BASE_OUTPUT_DIR, "by_dataset_summary.csv")
    by_dataset.to_csv(by_dataset_path)
    print(f"✓ By-dataset summary saved: {by_dataset_path}")
    
    # 4. Summary by k value
    by_k = master_df.groupby('k').agg({
        'r_all': ['mean', 'std', 'min', 'max'],
        'r_skip': ['mean', 'std', 'min', 'max'],
        'mean_accuracy': ['mean', 'std'],
    }).round(4)
    by_k_path = os.path.join(BASE_OUTPUT_DIR, "by_k_value_summary.csv")
    by_k.to_csv(by_k_path)
    print(f"✓ By-k-value summary saved: {by_k_path}")
    
    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    
    print(f"\n{'='*100}")
    print("MASTER GRID SEARCH COMPLETE!")
    print(f"{'='*100}")
    print(f"\nTotal experiments completed: {len(master_df)}")
    print(f"Output directory: {BASE_OUTPUT_DIR}")
    
    print("\n📊 TOP RESULTS (by most negative r_skip):")
    top_results = master_df.nlargest(10, 'r_skip', keep='first')[
        ['architecture', 'dataset', 'k', 'r_all', 'r_skip', 'p_skip', 'mean_accuracy']
    ]
    print(top_results.to_string(index=False))
    
    print("\n📉 STRONGEST RICCI FLOW (by most negative r_all):")
    strongest = master_df.nsmallest(10, 'r_all', keep='first')[
        ['architecture', 'dataset', 'k', 'r_all', 'p_all', 'mean_accuracy']
    ]
    print(strongest.to_string(index=False))
    
    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()

