#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid Search for Hyperparameter Tuning: b (number of models) and k (kNN neighbors)

This script runs a comprehensive grid search over:
- b values: number of models to train
- k values: k-NN neighbor parameter

For each combination, it:
1. Trains b models (from training.py logic)
2. Runs kNN Ricci-flow analysis (from knn_fixed.py logic)
3. Saves all outputs to a unique folder
4. Collects correlation statistics

Final output: summary CSV comparing all (b, k) combinations
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
B_VALUES = [70]  # Number of models to train
K_VALUES = [400, 450, 500, 600]  # k-NN neighbor values
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
# TRAINING FUNCTION (from training.py)
# ============================================================================

def train_models(b: int, output_dir: str, x_train: np.ndarray, y_train: np.ndarray, 
                 x_test: np.ndarray, y_test: np.ndarray) -> None:
    """Train b models and save outputs to output_dir."""
    accuracy = list()
    model_predict = np.empty(b, dtype=object)

    # Progress bar for training individual models
    with tqdm(total=b, desc="     Training models", position=1, leave=False, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar_train:
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
    np.save(os.path.join(output_dir, "model_predict.npy"), model_predict)
    np.save(os.path.join(output_dir, "accuracy.npy"), accuracy)
    pd.DataFrame(x_test).to_csv(os.path.join(output_dir, "x_test.csv"), index=False, header=None)
    pd.DataFrame(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index=False, header=None)


# ============================================================================
# ANALYSIS FUNCTION
# ============================================================================

def run_analysis(k: int, output_dir: str) -> Dict[str, float]:
    """Run kNN analysis on trained models in output_dir."""
    # Load artifacts
    model = np.load(os.path.join(output_dir, "model_predict.npy"), allow_pickle=True)
    accuracy = np.asarray(np.load(os.path.join(output_dir, "accuracy.npy")))
    X0 = np.array(pd.read_csv(os.path.join(output_dir, "x_test.csv"), header=None))

    # Run analysis
    mfr, msc = collect_across_models(models=model, X0=X0, k=k, acc=accuracy, acc_threshold=ACC_THRESHOLD)
    stats = correlation_report(mfr, msc)

    # Save results
    mfr.to_csv(os.path.join(output_dir, "mfr.csv"), index=False)
    msc.to_csv(os.path.join(output_dir, "msc.csv"), index=False)
    plot_summary(msc, mfr, out_png=os.path.join(output_dir, "analysis_plot.png"))

    return stats


# ============================================================================
# MAIN GRID SEARCH
# ============================================================================

def main():
    print("=" * 80)
    print("GRID SEARCH FOR HYPERPARAMETER TUNING (b, k)")
    print("=" * 80)
    print(f"b values: {B_VALUES}")
    print(f"k values: {K_VALUES}")
    print(f"Total combinations: {len(B_VALUES) * len(K_VALUES)}")
    print("=" * 80)

    # Load data once
    print("\n[1/3] Loading Fashion-MNIST data...")
    # Get absolute path to ensure consistency
    base_path = os.path.abspath(os.getcwd())
    data_path = os.path.join(base_path, 'our_data_fmnist')
    
    print(f"   Base directory: {base_path}")
    print(f"   Data directory: {data_path}")
    
    x_test = pd.read_csv(os.path.join(data_path, "fashion-mnist_test.csv"))
    y_test = x_test['label']
    x_test = x_test.iloc[:, 1:]

    x_train = pd.read_csv(os.path.join(data_path, "fashion-mnist_train.csv"))
    y_train = x_train['label']
    x_train = x_train.iloc[:, 1:]

    # Restrict to labels 5 and 9
    labels_1_7 = [5, 9]
    train_1_7 = np.concatenate([np.where(y_train == label)[0] for label in labels_1_7])
    test_1_7 = np.concatenate([np.where(y_test == label)[0] for label in labels_1_7])

    y_train = y_train.iloc[train_1_7].values
    y_test = y_test.iloc[test_1_7].values

    y_test[y_test == 5] = 0
    y_test[y_test == 9] = 1
    y_train[y_train == 5] = 0
    y_train[y_train == 9] = 1

    x_train = np.array(x_train.iloc[train_1_7, :])
    x_test = np.array(x_test.iloc[test_1_7, :])
    
    # Use subset of test data for faster analysis
    if TEST_SUBSET_SIZE < x_test.shape[0]:
        print(f"   Using subset of test data: {TEST_SUBSET_SIZE} samples (out of {x_test.shape[0]})")
        indices = np.random.choice(x_test.shape[0], TEST_SUBSET_SIZE, replace=False)
        x_test = x_test[indices]
        y_test = y_test[indices]

    print(f"   Training samples: {x_train.shape[0]}")
    print(f"   Test samples: {x_test.shape[0]}")

    # Grid search
    print("\n[2/3] Running grid search...")
    print("=" * 80)
    results = []
    total_combinations = len(B_VALUES) * len(K_VALUES)
    
    with tqdm(total=total_combinations, desc="Overall Progress", position=0, 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar_main:
        for idx, (b, k) in enumerate([(b, k) for b in B_VALUES for k in K_VALUES], 1):
            # Create output directory with absolute path
            output_dir = os.path.abspath(os.path.join(
                base_path, 
                f"training_outputs_narrow_5_ankleboot_vs_sandal_b{b}_k{k}"
            ))
            
            # Ensure directory exists
            try:
                os.makedirs(output_dir, exist_ok=True)
                # Verify it was created
                if not os.path.exists(output_dir):
                    raise Exception(f"Failed to create directory: {output_dir}")
            except Exception as e:
                tqdm.write(f"ERROR: Could not create output directory: {e}")
                continue
            
            # Update progress
            pbar_main.set_description(f"Combination {idx}/{total_combinations} [b={b}, k={k}]")
            
            # Train models
            tqdm.write(f"\n{'='*80}")
            tqdm.write(f"Combination {idx}/{total_combinations}: b={b}, k={k}")
            tqdm.write(f"{'='*80}")
            tqdm.write(f"  Output folder: {os.path.basename(output_dir)}")
            tqdm.write(f"  [STEP 1/2] Training {b} models...")
            start_time = time.time()
            train_models(b, output_dir, x_train, y_train, x_test, y_test)
            train_time = time.time() - start_time
            tqdm.write(f"  ✓ Training complete ({train_time:.1f}s, {train_time/b:.1f}s per model)")
            
            # Run analysis
            tqdm.write(f"  [STEP 2/2] Running kNN analysis (k={k})...")
            start_time = time.time()
            stats = run_analysis(k, output_dir)
            analysis_time = time.time() - start_time
            tqdm.write(f"  ✓ Analysis complete ({analysis_time:.1f}s)")
            
            # Store results
            results.append({
                'b': b,
                'k': k,
                'r_all': stats['r_all'],
                'p_all': stats['p_all'],
                'r_skip': stats['r_skip'],
                'p_skip': stats['p_skip'],
                'train_time_s': train_time,
                'analysis_time_s': analysis_time,
                'total_time_s': train_time + analysis_time,
                'output_dir': output_dir
            })
            
            tqdm.write(f"  📊 Results: r_all={stats['r_all']:.4f} (p={stats['p_all']:.3e}), r_skip={stats['r_skip']:.4f} (p={stats['p_skip']:.3e})")
            tqdm.write(f"  ⏱️  Total time: {train_time + analysis_time:.1f}s")
            
            pbar_main.update(1)

    # Save summary
    print("\n[3/3] Saving summary results...")
    results_df = pd.DataFrame(results)
    summary_path = os.path.join(base_path, "grid_search_summary.csv")
    results_df.to_csv(summary_path, index=False)
    
    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE!")
    print("=" * 80)
    print(f"\nSummary saved to: {summary_path}")
    print("\nTop 5 combinations by r_skip (layer-skip correlation):")
    print(results_df.sort_values('r_skip', ascending=False).head()[['b', 'k', 'r_all', 'r_skip', 'p_skip']])
    print("\nTop 5 combinations by r_all (overall correlation):")
    print(results_df.sort_values('r_all', ascending=False).head()[['b', 'k', 'r_all', 'r_skip', 'p_all']])
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

