#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Ricci-like behaviour from saved layer activations.

Input: a layer_outputs .npz produced by main.py, which includes
  hidden_1, hidden_2, ..., logit, X_test, y_test

We will:
  1) Build k-NN graphs G_k(X^l) for each layer l (on test activations)
  2) For each layer l, compute:
       - total curvature Ric_l = sum_{(i,j) in E_k} (4 - deg(i) - deg(j))  [Forman-Ricci for unweighted graphs]
       - total geodesic sum g_l = sum_{i<j} d_G(i,j)  [shortest path distance]
  3) Compute η_l = g_{l+1} - g_l
  4) Compute Pearson correlation ρ_k between {η_l} and {Ric_l} across layers

We also report Fisher's z: z = arctanh(ρ) / sqrt(L-4), where L is #layers considered.

Notes:
- We ignore the final single-logit layer for distance geometry; we consider only hidden_[1..H] layers.
- For speed, we allow optional random subsampling of node pairs when computing g_l (sum of all-pairs distances).
- If the graph is disconnected for a given k/l, we skip that k unless --allow-disconnected is set; if allowed,
  disconnected pairs contribute a large penalty (set by --disconnected-penalty).

Usage:
  python ricci_analysis.py --npz run_spirals/layer_outputs_test.npz --k-list 10 20 30 50 90 120 150 \
      --sample-pairs 200000 --out run_spirals/ricci_report.json

This script prints a small table and writes a JSON with per-k metrics.
"""

import argparse
import json
import math
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path as cs_shortest_path

# ------------------------
# Graph + metrics
# ------------------------


def build_knn_graph(X: np.ndarray, k: int) -> nx.Graph:
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    # indices include self at position 0; skip self
    G = nx.Graph()
    n = X.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in indices[i, 1:]:
            G.add_edge(i, int(j))
    return G


def forman_ricci_total(G: nx.Graph) -> float:
    # For unweighted graphs with vertex/edge weights=1: R(i,j) = 4 - deg(i) - deg(j)
    deg = dict(G.degree())
    total = 0.0
    for u, v in G.edges():
        total += 4.0 - deg[u] - deg[v]
    return float(total)


def total_geodesic_sum(
    G: nx.Graph,
    sample_pairs: int = 0,
    disconnected_penalty: float = 1e6,
    allow_disconnected: bool = False,
) -> float:
    n = G.number_of_nodes()
    if sample_pairs and sample_pairs < n*(n-1)//2:
        # sample unordered pairs
        rng = np.random.default_rng(42)
        total = 0.0
        for _ in range(sample_pairs):
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n-1))
            if j >= i:
                j += 1
            # shortest path length
            try:
                d = nx.shortest_path_length(G, i, j)
            except nx.NetworkXNoPath:
                if not allow_disconnected:
                    raise
                d = disconnected_penalty
            total += float(d)
        # scale to approximate full sum
        factor = (n*(n-1)//2) / sample_pairs
        return total * factor
    else:
        # full all-pairs (could be heavy for n>2000)
        total = 0.0
        for src, lengths in nx.all_pairs_shortest_path_length(G):
            # lengths is dict {node: dist}
            if len(lengths) < n and not allow_disconnected:
                raise nx.NetworkXNoPath(
                    "Graph disconnected; use --allow-disconnected or increase k")
            if allow_disconnected:
                # fill missing with penalty
                miss = n - len(lengths)
                total += sum(float(d) for d in lengths.values()) + \
                    miss * disconnected_penalty
            else:
                total += sum(float(d) for d in lengths.values())
        # each pair counted twice in all_pairs; divide by 2
        return total / 2.0


def total_geodesic_sum_fast(G: nx.Graph,
                            allow_disconnected: bool = False,
                            disconnected_penalty: float = 1e6) -> float:
    """Compute full all-pairs shortest-path sum using SciPy's csgraph (fast).
    For unweighted undirected graphs this uses BFS internally.
    If the graph is disconnected and allow_disconnected=False, raises; otherwise
    replaces inf distances with `disconnected_penalty`.
    """
    n = G.number_of_nodes()
    # Build CSR adjacency
    rows = []
    cols = []
    for u, v in G.edges():
        rows.append(u)
        cols.append(v)
        rows.append(v)
        cols.append(u)
    data = np.ones(len(rows), dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    # distances: shape (n, n); unweighted=True -> BFS distances
    D = cs_shortest_path(A, directed=False, unweighted=True)

    if not allow_disconnected and np.isinf(D).any():
        raise nx.NetworkXNoPath(
            "Graph disconnected; use --allow-disconnected or increase k")

    if allow_disconnected:
        D = np.where(np.isinf(D), disconnected_penalty, D)

    # sum of upper triangle (i<j)
    iu = np.triu_indices(n, k=1)
    return float(D[iu].sum())

# ------------------------
# Ricci computation per k
# ------------------------


def ricci_for_k(
    layers: List[np.ndarray],
    k: int,
    sample_pairs: int,
    allow_disconnected: bool,
    disconnected_penalty: float,
    prefer_exact: bool = True,
    progress: bool = False,
) -> Dict[str, float]:
    Ric_list: List[float] = []
    g_list: List[float] = []

    # Build graphs & compute metrics per layer
    layer_iter = enumerate(layers)
    if progress:
        layer_iter = enumerate(tqdm(layers, desc=f"layers | k={k}", leave=False))
    for _, X in layer_iter:
        G = build_knn_graph(X, k)
        Ric = forman_ricci_total(G)
        g = None
        if prefer_exact:
            try:
                g = total_geodesic_sum_fast(
                    G,
                    allow_disconnected=allow_disconnected,
                    disconnected_penalty=disconnected_penalty,
                )
            except Exception:
                g = None
        if g is None:
            g = total_geodesic_sum(
                G,
                sample_pairs=sample_pairs,
                allow_disconnected=allow_disconnected,
                disconnected_penalty=disconnected_penalty,
            )
        Ric_list.append(Ric)
        g_list.append(g)

    # compute eta_l = g_{l+1} - g_l  for l=1..L-1
    eta = np.diff(np.array(g_list))
    Ric_arr = np.array(Ric_list[:-1])

    # Pearson correlation between eta and Ric
    if eta.size < 2:
        return {"rho": float("nan"), "z": float("nan")}
    eta_c = eta - eta.mean()
    Ric_c = Ric_arr - Ric_arr.mean()
    denom = np.linalg.norm(eta_c) * np.linalg.norm(Ric_c)
    rho = float(eta_c.dot(Ric_c) / denom) if denom > 0 else float("nan")

    # Fisher's z with L = number of layers considered
    L = len(layers)
    try:
        z = float(np.arctanh(rho) / math.sqrt(max(L - 4, 1)))
    except FloatingPointError:
        z = float("nan")

    return {"rho": rho, "z": z,
            "Ric": Ric_list, "g": g_list}


# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compute Ricci-like metrics from layer activations")
    ap.add_argument("--npz", required=True,
                    help="Path to layer_outputs_test.npz")
    ap.add_argument("--k-list", nargs="+", type=int,
                    default=[6, 7, 9, 10, 15, 18, 20, 30, 50, 90, 100, 120, 150])
    ap.add_argument("--sample-pairs", type=int, default=200000,
                    help="0 means full all-pairs (slow)")
    ap.add_argument("--allow-disconnected", action="store_true")
    ap.add_argument("--disconnected-penalty", type=float, default=1e6)
    ap.add_argument("--skip-exact", action="store_true",
                    help="Skip exact shortest paths and rely on sampling-based estimates")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    data = np.load(args.npz)
    # Collect hidden layers in order
    layer_keys = sorted([k for k in data.keys() if k.startswith(
        "hidden_")], key=lambda s: int(s.split("_")[1]))
    layers = [data[k].astype(np.float32) for k in layer_keys]
    print(
        f"[ricci] layers={len(layers)} | n_test={layers[0].shape[0]} | dims={[x.shape[1] for x in layers]}")

    results: Dict[int, Dict[str, float]] = {}
    best_k = None
    best_rho = +1.0

    for k in tqdm(args.k_list, desc=f"k sweep | {args.npz}"):
        try:
            res = ricci_for_k(
                layers,
                k,
                args.sample_pairs,
                args.allow_disconnected,
                args.disconnected_penalty,
                prefer_exact=not args.skip_exact,
            )
        except Exception as e:
            res = {"error": str(e)}
        results[k] = res
        if isinstance(res, dict) and "rho" in res and not np.isnan(res["rho"]):
            if res["rho"] < best_rho:
                best_rho = res["rho"]
                best_k = k

    summary = {
        "npz": args.npz,
        "layers": layer_keys,
        "k_list": args.k_list,
        "skip_exact": bool(args.skip_exact),
        "best_k": best_k,
        "best_rho": float(best_rho) if best_k is not None else None,
        "results": results,
    }

    print("\nRicci coefficient sweep (lower is more Ricci-like):")
    for k in args.k_list:
        r = results[k]
        if isinstance(r, dict) and "rho" in r:
            print(f"  k={k:4d} | rho={r['rho']!r}  z={r.get('z')!r}")
        else:
            print(f"  k={k:4d} | ERROR: {r}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved report to {args.out}")


if __name__ == "__main__":
    main()
