#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ricci-flow style analysis on DNN layers (fixed to match paper definitions)
- kNN graphs on test activations per layer
- Geodesic distance = shortest-path distance on the kNN graph
- Forman–Ricci curvature on an edge (i,j): 4 - deg(i) - deg(j)  (unit weights)
- Global Ricci coefficient at layer l: Ric_l = sum_{(i,j) in E_l} R_l(i,j)
- Geodesic mass at layer l: g_l = sum_{i<j} gamma_l(i,j)
- We correlate Δg_l := g_l - g_{l-1} with Ric_{l-1}

Inputs written by training script:
  - model_predict.npy : object array, each entry = list of layer activations (n_test, d_l)
  - accuracy.npy      : list/array, test accuracy per model
  - x_test.csv        : (n_test, d0) raw test features (headerless)

Usage (parameters at bottom):
  - Adjust K, ACC_THRESHOLD as desired and run.

Author: fixed version for strict paper compliance.
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

# ----------------------------
# Helpers: graphs & metrics
# ----------------------------

def build_knn_graph(X: np.ndarray, k: int) -> csr_matrix:
    """Return an undirected, unweighted kNN adjacency in CSR.
    Symmetrize kNN (mutualization by max) and set diagonal to 0.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.dtype != np.float32 and X.dtype != np.float64:
        X = X.astype(np.float32, copy=False)

    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(X)
    A = knn.kneighbors_graph(X, mode="connectivity")  # (n,n) sparse {0,1}
    # symmetrize (undirected)
    A = A.maximum(A.T)
    # zero-out diagonal (sklearn may set self-links when k>=1)
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr()


def sum_shortest_paths(A: csr_matrix) -> float:
    """Compute g = sum of all-pairs shortest-path distances, i<j.
    If graph is disconnected, infinite distances are discarded (paper assumes k chosen so connected).
    """
    dist = shortest_path(A, directed=False, unweighted=True)
    # take upper triangle (i<j)
    iu = np.triu_indices_from(dist, k=1)
    vals = dist[iu]
    finite = np.isfinite(vals)
    if not np.all(finite):
        # Warn once; user should increase k (paper needs connected graph)
        missing = (~finite).sum()
        print(f"[WARN] Disconnected graph: {missing} pair distances are inf; they will be ignored.")
        vals = vals[finite]
    return float(vals.sum())


def global_forman_ricci(A: csr_matrix) -> float:
    """Global Ricci coefficient Ric_l = sum over edges of Forman–Ricci curvatures with unit weights.
    For undirected graph: R(i,j) = 4 - deg(i) - deg(j). Count each edge once.
    """
    deg = np.asarray(A.sum(axis=1)).ravel()  # degree per node
    # use upper triangle to count undirected edges once
    A_ut = sp_triu(A, k=1).tocoo()
    # curvature per stored edge
    curv = 4.0 - deg[A_ut.row] - deg[A_ut.col]
    return float(curv.sum())


# ----------------------------
# Analysis pipeline
# ----------------------------

def analyze_model_layers(activations: List[np.ndarray], X0: np.ndarray, k: int) -> Dict[str, np.ndarray]:
    """For one model: build graphs, compute (g_l, Ric_l) for l=0..L.
    l=0 refers to baseline on the raw test input X0.
    Returns dict with arrays: g (L+1,), Ric (L+1,) where Ric_0 is defined on G^0 (may be useful for checks).
    """
    # baseline graph on input space
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
    """Run analysis over models passing the accuracy threshold.
    Returns two long DataFrames:
      - mfr(layer, mod, ssr): global Ricci coefficient at layer l (we will use l-1 for correlation)
      - msc(layer, mod, ssr): Δg_l = g_l - g_{l-1}
    Layers use 1-based indexing for hidden layers, consistent with the paper’s notation.
    """
    keep = np.where(acc > acc_threshold)[0]
    if keep.size == 0:
        raise ValueError(f"No models exceed accuracy threshold {acc_threshold}. Max acc={acc.max():.4f}")

    rows_fr = []
    rows_sc = []
    for m in keep:
        acts = models[m]  # list of arrays per layer
        res = analyze_model_layers(acts, X0, k)
        g = res["g"]       # length L+1 (including baseline 0)
        Ric = res["Ric"]   # length L+1
        L = len(acts)
        # Δg_l for l=1..L
        dgs = g[1:] - g[:-1]
        # FR for layers 0..L-1 (we will correlate Ric_{l-1} with Δg_l)
        for l in range(1, L+1):
            rows_sc.append({"layer": l, "mod": int(m), "ssr": float(dgs[l-1])})
            rows_fr.append({"layer": l-1, "mod": int(m), "ssr": float(Ric[l-1])})

    msc = pd.DataFrame(rows_sc, columns=["layer", "mod", "ssr"])  # Δg_l
    mfr = pd.DataFrame(rows_fr, columns=["layer", "mod", "ssr"])  # Ric_{l-1}
    return mfr, msc


def correlation_report(mfr: pd.DataFrame, msc: pd.DataFrame) -> Dict[str, float]:
    """Compute overall Pearson correlations:
      - corr_all between msc.ssr and mfr shifted to align (layer l vs layer l-1)
      - corr_skip excludes l=1 (since it matches Ric_0)
    """
    from scipy.stats import pearsonr
    # Align by (mod, layer) pairing: use msc.layer=l with mfr.layer=l-1
    # FIXED: Create a shifted version of mfr where we add 1 to layer numbers
    # so that mfr[layer=0] becomes mfr[layer=1], matching the paper's requirement
    # that Δg_l should correlate with Ric_{l-1}
    mfr_shifted = mfr.copy()
    mfr_shifted['layer'] = mfr_shifted['layer'] + 1
    merged = msc.merge(mfr_shifted, on=["mod", "layer"], how="inner", suffixes=("_dg", "_fr"))
    # Now merged contains: msc[layer=l] paired with original mfr[layer=l-1]
    r_all = pearsonr(merged["ssr_dg"].values, merged["ssr_fr"].values)
    # skip the first layer pairs (l==1) as in the paper's layer-skip plot
    merged_skip = merged[merged["layer"] != 1]
    r_skip = pearsonr(merged_skip["ssr_dg"].values, merged_skip["ssr_fr"].values)
    return {
        "r_all": float(r_all[0]), "p_all": float(r_all[1]),
        "r_skip": float(r_skip[0]), "p_skip": float(r_skip[1]),
    }


def plot_summary(msc: pd.DataFrame, mfr: pd.DataFrame, out_png: str | None = None) -> None:
    """Recreate the 2x2 style overview used by the authors (boxplots + scatter)."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 8))

    ax1 = plt.subplot(2, 2, 1)
    msc.boxplot(column='ssr', by='layer', grid=False, ax=ax1)
    ax1.set_xlabel('Layer l')
    ax1.set_ylabel('Δg_l = g_l - g_{l-1}')

    ax2 = plt.subplot(2, 2, 2)
    # For FR, display Ric_{l-1} against its layer index (so starts at 0)
    mfr.boxplot(column='ssr', by='layer', grid=False, ax=ax2)
    ax2.set_xlabel('Layer index (for Ric_{l-1})')
    ax2.set_ylabel('Global Forman–Ricci (Ric_{l-1})')

    ax3 = plt.subplot(2, 2, 3)
    # FIXED: Shift mfr layers to properly align Δg_l with Ric_{l-1}
    mfr_shifted = mfr.copy()
    mfr_shifted['layer'] = mfr_shifted['layer'] + 1
    merged = msc.merge(mfr_shifted, on=['mod', 'layer'], suffixes=('_dg', '_fr'))
    sc = ax3.scatter(merged['ssr_dg'], merged['ssr_fr'], c=merged['layer'], marker='o')
    ax3.set_xlabel('Δg_l')
    ax3.set_ylabel('Ric_{l-1}')
    ax3.set_title('Layer-skip correlation (l vs l-1)')

    # add simple least-squares fit
    if len(merged) >= 2:
        xs = merged['ssr_dg'].values
        ys = merged['ssr_fr'].values
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        xsu = np.unique(xs)
        ax3.plot(xsu, p(xsu))

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved figure to {out_png}")
    else:
        # Show for interactive runs
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


# ----------------------------
# Main entry
# ----------------------------
if __name__ == "__main__":
    # Parameters
    # Data I/O
    # Use dynamic path - training_outputs folder created by training.py
    DATA_DIR = os.path.join(os.getcwd(), 'training_outputs')
    K = 350  # ~17.5% of 2000 test samples (paper recommends 10-25%)
    ACC_THRESHOLD = 0.98

    # Load artifacts
    model = np.load(os.path.join(DATA_DIR, "model_predict.npy"), allow_pickle=True)
    accuracy = np.asarray(np.load(os.path.join(DATA_DIR, "accuracy.npy")))
    X0 = np.array(pd.read_csv(os.path.join(DATA_DIR, "x_test.csv"), header=None))  # skip header row in saved file

    print(f"Loaded {len(model)} models; acc in [{accuracy.min():.4f}, {accuracy.max():.4f}]  |  k={K}")

    # Run analysis and stats
    mfr, msc = collect_across_models(models=model, X0=X0, k=K, acc=accuracy, acc_threshold=ACC_THRESHOLD)
    stats = correlation_report(mfr, msc)

    print("[STATS] Pearson r (all):   r=%.4f, p=%.3e" % (stats['r_all'], stats['p_all']))
    print("[STATS] Pearson r (skip):  r=%.4f, p=%.3e" % (stats['r_skip'], stats['p_skip']))

    # Plot (set a file name to save instead of interactive display)
    plot_summary(msc, mfr, out_png=None)

    # Dump CSVs for downstream analysis
    mfr.to_csv("mfr.csv", index=False)
    msc.to_csv("msc.csv", index=False)
    print("[INFO] Wrote mfr.csv and msc.csv")
