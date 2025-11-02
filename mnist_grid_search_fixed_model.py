#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST Grid Search with Fixed Model Architecture

This script:
1. Trains ONE set of models (b models) ONCE
2. Reuses the trained models for different k-NN configurations
3. Uses MNIST dataset (digits 1 vs 7)

Key difference from grid_search.py:
- Training happens ONCE, not per k-value
- Only k parameter varies across experiments
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, triu as sp_triu
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import List, Dict, Tuple
import time

# ============================================================================
# GRID SEARCH PARAMETERS
# ============================================================================
B_VALUE = 70  # Number of models to train (ONCE)
K_VALUES = [350, 450, 500]  # k-NN neighbor values to test
ACC_THRESHOLD = 0.98  # Accuracy threshold for analysis
TEST_SUBSET_SIZE = 2000  # Use full test data (set to 2000 for all samples)

# ============================================================================
# HELPER FUNCTIONS FROM knn_fixed.py
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
        missing = (~finite).sum()
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
        raise ValueError(f"No models exceed accuracy threshold {acc_threshold}. Max acc={acc.max():.4f}")

    rows_fr = []
    rows_sc = []
    
    # Progress bar for analyzing individual models
    with tqdm(total=len(keep), desc="     Analyzing models", position=1, leave=False, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar_analyze:
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
            pbar_analyze.update(1)

    msc = pd.DataFrame(rows_sc, columns=["layer", "mod", "ssr"])
    mfr = pd.DataFrame(rows_fr, columns=["layer", "mod", "ssr"])
    return mfr, msc


def correlation_report(mfr: pd.DataFrame, msc: pd.DataFrame) -> Dict[str, float]:
    """Compute overall Pearson correlations."""
    mfr_shifted = mfr.copy()
    mfr_shifted['layer'] = mfr_shifted['layer'] + 1
    merged = msc.merge(mfr_shifted, on=["mod", "layer"], how="inner", suffixes=("_dg", "_fr"))
    r_all = pearsonr(merged["ssr_dg"].values, merged["ssr_fr"].values)
    merged_skip = merged[merged["layer"] != 1]
    r_skip = pearsonr(merged_skip["ssr_dg"].values, merged_skip["ssr_fr"].values)
    return {
        "r_all": float(r_all[0]), "p_all": float(r_all[1]),
        "r_skip": float(r_skip[0]), "p_skip": float(r_skip[1]),
    }


def plot_summary(msc: pd.DataFrame, mfr: pd.DataFrame, out_png: str) -> None:
    """Generate summary plots."""
    fig = plt.figure(figsize=(8, 8))

    ax1 = plt.subplot(2, 2, 1)
    msc.boxplot(column='ssr', by='layer', grid=False, ax=ax1)
    ax1.set_xlabel('Layer l')
    ax1.set_ylabel('Δg_l = g_l - g_{l-1}')

    ax2 = plt.subplot(2, 2, 2)
    mfr.boxplot(column='ssr', by='layer', grid=False, ax=ax2)
    ax2.set_xlabel('Layer index (for Ric_{l-1})')
    ax2.set_ylabel('Global Forman-Ricci (Ric_{l-1})')

    ax3 = plt.subplot(2, 2, 3)
    mfr_shifted = mfr.copy()
    mfr_shifted['layer'] = mfr_shifted['layer'] + 1
    merged = msc.merge(mfr_shifted, on=['mod', 'layer'], suffixes=('_dg', '_fr'))
    sc = ax3.scatter(merged['ssr_dg'], merged['ssr_fr'], c=merged['layer'], marker='o')
    ax3.set_xlabel('Δg_l')
    ax3.set_ylabel('Ric_{l-1}')
    ax3.set_title('Layer-skip correlation (l vs l-1)')

    if len(merged) >= 2:
        xs = merged['ssr_dg'].values
        ys = merged['ssr_fr'].values
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        xsu = np.unique(xs)
        ax3.plot(xsu, p(xsu))

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# TRAINING FUNCTION (MODIFIED - TRAIN ONCE)
# ============================================================================

def train_models_once(b: int, output_dir: str, x_train: np.ndarray, y_train: np.ndarray, 
                      x_test: np.ndarray, y_test: np.ndarray) -> None:
    """Train b models ONCE and save outputs to output_dir."""
    print(f"\n{'='*80}")
    print(f"TRAINING {b} MODELS (ONE TIME ONLY)")
    print(f"{'='*80}")
    
    accuracy = list()
    model_predict = np.empty(b, dtype=object)

    # Progress bar for training individual models
    with tqdm(total=b, desc="Training models", position=0, 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar_train:
        for j in range(b):
            # Define DNN architecture - NARROW 5 (25 neurons, 5 layers)
            model = Sequential()
            model.add(Dense(units=25, activation='relu', input_shape=(x_test.shape[1],)))
            model.add(Dense(units=25, activation='relu'))
            model.add(Dense(units=25, activation='relu'))
            model.add(Dense(units=25, activation='relu'))
            model.add(Dense(units=25, activation='relu'))
            model.add(Dense(units=1, activation='sigmoid'))

            # Binary cross-entropy loss function
            model.compile(loss='binary_crossentropy',
                          optimizer=RMSprop(),
                          metrics=['accuracy'])

            # Train model on training data
            dnn_history = model.fit(x_train, y_train,
                                    epochs=50, batch_size=32,
                                    validation_split=0.2,
                                    verbose=0)  # Suppress training output

            # Check accuracy on test data
            acc = model.evaluate(x_test, y_test, verbose=0)[1]
            accuracy.append(acc)

            # Get activations
            activations = []
            current_input = x_test
            for layer in model.layers[:-1]:  # Exclude last layer
                current_output = layer(current_input)
                activations.append(current_output.numpy())
                current_input = current_output

            model_predict[j] = activations
            pbar_train.update(1)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "model_predict.npy"), model_predict)
    np.save(os.path.join(output_dir, "accuracy.npy"), accuracy)
    pd.DataFrame(x_test).to_csv(os.path.join(output_dir, "x_test.csv"), index=False, header=None)
    pd.DataFrame(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index=False, header=None)
    
    print(f"\n✓ Training complete!")
    print(f"  Models saved to: {output_dir}")
    print(f"  Mean accuracy: {np.mean(accuracy):.4f} (±{np.std(accuracy):.4f})")
    print(f"  Models above threshold ({ACC_THRESHOLD}): {np.sum(np.array(accuracy) > ACC_THRESHOLD)}/{b}")


# ============================================================================
# ANALYSIS FUNCTION (REUSE TRAINED MODELS)
# ============================================================================

def run_analysis_with_k(k: int, model_dir: str, output_dir: str) -> Dict[str, float]:
    """Run kNN analysis with specific k value on pre-trained models."""
    # Load artifacts from model_dir
    model = np.load(os.path.join(model_dir, "model_predict.npy"), allow_pickle=True)
    accuracy = np.asarray(np.load(os.path.join(model_dir, "accuracy.npy")))
    X0 = np.array(pd.read_csv(os.path.join(model_dir, "x_test.csv"), header=None))

    # Run analysis with this k value
    mfr, msc = collect_across_models(models=model, X0=X0, k=k, acc=accuracy, acc_threshold=ACC_THRESHOLD)
    stats = correlation_report(mfr, msc)

    # Save results to output_dir
    os.makedirs(output_dir, exist_ok=True)
    mfr.to_csv(os.path.join(output_dir, "mfr.csv"), index=False)
    msc.to_csv(os.path.join(output_dir, "msc.csv"), index=False)
    plot_summary(msc, mfr, out_png=os.path.join(output_dir, "analysis_plot.png"))

    return stats


# ============================================================================
# MAIN GRID SEARCH
# ============================================================================

def main():
    print("=" * 80)
    print("MNIST GRID SEARCH WITH FIXED MODEL ARCHITECTURE")
    print("=" * 80)
    print(f"b value (models): {B_VALUE}")
    print(f"k values (kNN): {K_VALUES}")
    print(f"Total k configurations: {len(K_VALUES)}")
    print("=" * 80)
    print("\nStrategy: Train models ONCE, then test with different k values")
    print("=" * 80)

    # Get base path
    base_path = os.path.abspath(os.getcwd())
    
    # Load MNIST data (digits 1 vs 7)
    print("\n[1/4] Loading MNIST data (digits 1 vs 7)...")
    # Data is in parent directory (same as extraction script output)
    data_path = os.path.join(os.path.dirname(base_path), 'extracted_data_mnist')
    
    print(f"   Base directory: {base_path}")
    print(f"   Data directory: {data_path}")
    
    # Load extracted MNIST data
    try:
        train_1 = pd.read_csv(os.path.join(data_path, "train_1.csv"))
        train_7 = pd.read_csv(os.path.join(data_path, "train_7.csv"))
        test_1 = pd.read_csv(os.path.join(data_path, "test_1.csv"))
        test_7 = pd.read_csv(os.path.join(data_path, "test_7.csv"))
        
        # Combine train and test
        x_train = pd.concat([train_1, train_7], ignore_index=True)
        x_test = pd.concat([test_1, test_7], ignore_index=True)
        
        # Extract labels and features
        y_train = x_train['label'].values
        y_test = x_test['label'].values
        x_train = x_train.iloc[:, 1:].values  # Skip label column
        x_test = x_test.iloc[:, 1:].values
        
        # Convert labels: 1->0, 7->1
        y_train[y_train == 1] = 0
        y_train[y_train == 7] = 1
        y_test[y_test == 1] = 0
        y_test[y_test == 7] = 1
        
        print(f"   Training samples: {x_train.shape[0]}")
        print(f"   Test samples: {x_test.shape[0]}")
        print(f"   Feature dimensions: {x_train.shape[1]}")
        
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find extracted MNIST data!")
        print(f"Please run mnist_extraction_1_vs_7.py first to extract the data.")
        print(f"Expected files in {data_path}:")
        print("  - train_1.csv")
        print("  - train_7.csv")
        print("  - test_1.csv")
        print("  - test_7.csv")
        return

    # Use subset of test data if specified
    if TEST_SUBSET_SIZE < x_test.shape[0]:
        print(f"   Using subset of test data: {TEST_SUBSET_SIZE} samples (out of {x_test.shape[0]})")
        indices = np.random.choice(x_test.shape[0], TEST_SUBSET_SIZE, replace=False)
        x_test = x_test[indices]
        y_test = y_test[indices]

    # Train models ONCE
    print(f"\n[2/4] Training {B_VALUE} models (ONE TIME)...")
    model_dir = os.path.join(base_path, f"mnist_1_vs_7_models_b{B_VALUE}")
    start_time = time.time()
    train_models_once(B_VALUE, model_dir, x_train, y_train, x_test, y_test)
    train_time = time.time() - start_time
    print(f"⏱️  Total training time: {train_time:.1f}s ({train_time/B_VALUE:.1f}s per model)")

    # Run analysis for each k value
    print(f"\n[3/4] Running kNN analysis for different k values...")
    print("=" * 80)
    results = []
    
    with tqdm(total=len(K_VALUES), desc="Overall Progress", position=0,
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar_main:
        for idx, k in enumerate(K_VALUES, 1):
            # Create output directory for this k value
            output_dir = os.path.join(base_path, f"mnist_1_vs_7_analysis_b{B_VALUE}_k{k}")
            
            # Update progress
            pbar_main.set_description(f"k-value {idx}/{len(K_VALUES)} [k={k}]")
            
            # Run analysis
            tqdm.write(f"\n{'='*80}")
            tqdm.write(f"Analysis {idx}/{len(K_VALUES)}: k={k}")
            tqdm.write(f"{'='*80}")
            tqdm.write(f"  Output folder: {os.path.basename(output_dir)}")
            tqdm.write(f"  Running kNN analysis with k={k}...")
            
            start_time = time.time()
            stats = run_analysis_with_k(k, model_dir, output_dir)
            analysis_time = time.time() - start_time
            
            tqdm.write(f"  ✓ Analysis complete ({analysis_time:.1f}s)")
            
            # Store results
            results.append({
                'b': B_VALUE,
                'k': k,
                'r_all': stats['r_all'],
                'p_all': stats['p_all'],
                'r_skip': stats['r_skip'],
                'p_skip': stats['p_skip'],
                'analysis_time_s': analysis_time,
                'output_dir': output_dir
            })
            
            tqdm.write(f"  📊 Results: r_all={stats['r_all']:.4f} (p={stats['p_all']:.3e}), r_skip={stats['r_skip']:.4f} (p={stats['p_skip']:.3e})")
            
            pbar_main.update(1)

    # Save summary
    print(f"\n[4/4] Saving summary results...")
    results_df = pd.DataFrame(results)
    summary_path = os.path.join(base_path, "mnist_grid_search_summary.csv")
    results_df.to_csv(summary_path, index=False)
    
    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE!")
    print("=" * 80)
    print(f"\nSummary saved to: {summary_path}")
    print(f"\nResults for all k values (b={B_VALUE}):")
    print(results_df[['k', 'r_all', 'p_all', 'r_skip', 'p_skip', 'analysis_time_s']])
    print("\nBest k by r_skip (layer-skip correlation):")
    best_skip = results_df.loc[results_df['r_skip'].idxmax()]
    print(f"  k={int(best_skip['k'])}: r_skip={best_skip['r_skip']:.4f} (p={best_skip['p_skip']:.3e})")
    print("\nBest k by r_all (overall correlation):")
    best_all = results_df.loc[results_df['r_all'].idxmax()]
    print(f"  k={int(best_all['k'])}: r_all={best_all['r_all']:.4f} (p={best_all['p_all']:.3e})")
    print("\n" + "=" * 80)
    print(f"\n⏱️  Total time saved by training once: ~{train_time * (len(K_VALUES) - 1):.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()

