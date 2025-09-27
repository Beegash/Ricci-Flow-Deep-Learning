#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topologically-focused Ricci analysis with methods that emphasize connectivity structure.

This version focuses on graph construction methods that capture topological 
relationships rather than geometric distances, which is more appropriate for 
Ricci flow analysis.

Methods included:
1. Standard k-NN (baseline)
2. Mutual k-NN (reduces noise, more stable)
3. Gabriel graph (geometric topology)
4. Relative neighborhood graph (proximity topology)
5. Epsilon-neighborhood (density-adaptive)
6. Adaptive k-NN (density-adaptive topology)

Usage:
  python ricci_analysis_topological.py --npz run_moons/layer_outputs_test.npz \
      --method mutual_knn --k-list 6 7 9 10 15 18 20 30 50 90 \
      --sample-pairs 20000 --out run_moons/ricci_topological.json
"""

import argparse
import json
import math
from typing import Dict, List, Tuple, Optional
from enum import Enum

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path as cs_shortest_path
from tqdm.auto import tqdm


class TopologicalMethod(Enum):
    KNN = "knn"
    MUTUAL_KNN = "mutual_knn"
    GABRIEL = "gabriel"
    RELATIVE_NEIGHBORHOOD = "relative_neighborhood"
    EPSILON = "epsilon"
    ADAPTIVE_KNN = "adaptive_knn"


def build_knn_graph(X: np.ndarray, k: int) -> nx.Graph:
    """Standard k-NN graph construction."""
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    G = nx.Graph()
    n = X.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in indices[i, 1:]:  # Skip self
            G.add_edge(i, int(j))
    return G


def build_mutual_knn_graph(X: np.ndarray, k: int) -> nx.Graph:
    """Mutual k-NN: edge exists only if both points are in each other's k-NN."""
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    G = nx.Graph()
    n = X.shape[0]
    G.add_nodes_from(range(n))
    
    # Create adjacency matrix for mutual k-NN
    adj_matrix = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in indices[i, 1:]:  # Skip self
            adj_matrix[i, j] = True
    
    # Only keep mutual edges
    mutual_adj = adj_matrix & adj_matrix.T
    
    for i in range(n):
        for j in range(i + 1, n):
            if mutual_adj[i, j]:
                G.add_edge(i, j)
    
    return G


def build_gabriel_graph(X: np.ndarray) -> nx.Graph:
    """Gabriel graph: no point inside the circle with diameter (i,j)."""
    n = X.shape[0]
    distances = euclidean_distances(X)
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for i in range(n):
        for j in range(i + 1, n):
            # Gabriel condition: no point k inside the circle with diameter (i,j)
            is_gabriel = True
            for k in range(n):
                if k == i or k == j:
                    continue
                # Check if k is inside the circle with diameter (i,j)
                if (distances[i, k]**2 + distances[j, k]**2 < distances[i, j]**2):
                    is_gabriel = False
                    break
            
            if is_gabriel:
                G.add_edge(i, j)
    
    return G


def build_relative_neighborhood_graph(X: np.ndarray) -> nx.Graph:
    """Relative neighborhood graph: edge if no closer point to both endpoints."""
    n = X.shape[0]
    distances = euclidean_distances(X)
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for i in range(n):
        for j in range(i + 1, n):
            # Relative neighborhood condition: max(d(i,k), d(j,k)) >= d(i,j)
            is_relative = True
            for k in range(n):
                if k == i or k == j:
                    continue
                if max(distances[i, k], distances[j, k]) < distances[i, j]:
                    is_relative = False
                    break
            
            if is_relative:
                G.add_edge(i, j)
    
    return G


def build_epsilon_graph(X: np.ndarray, epsilon: float) -> nx.Graph:
    """Epsilon-neighborhood graph: connect points within distance epsilon."""
    distances = euclidean_distances(X)
    n = X.shape[0]
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] <= epsilon:
                G.add_edge(i, j)
    
    return G


def build_adaptive_knn_graph(X: np.ndarray, k_min: int = 3, k_max: int = 20) -> nx.Graph:
    """Adaptive k-NN based on local density."""
    n = X.shape[0]
    
    # Compute local density for each point
    nbrs = NearestNeighbors(n_neighbors=k_max+1, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # Use k-th nearest neighbor distance as density measure
    local_densities = distances[:, k_min]  # k_min-th nearest neighbor distance
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for i in range(n):
        # Adaptive k based on local density
        adaptive_k = min(k_max, max(k_min, int(k_min * (local_densities.mean() / local_densities[i]))))
        
        # Get adaptive_k nearest neighbors
        nbrs_adaptive = NearestNeighbors(n_neighbors=adaptive_k+1, algorithm='auto').fit(X)
        _, indices = nbrs_adaptive.kneighbors(X[i:i+1])
        
        for j in indices[0, 1:]:  # Skip self
            G.add_edge(i, int(j))
    
    return G


def build_topological_graph(X: np.ndarray, method: TopologicalMethod, k: int = 10, epsilon: float = 0.1) -> nx.Graph:
    """Build graph using specified topological method."""
    if method == TopologicalMethod.KNN:
        return build_knn_graph(X, k)
    elif method == TopologicalMethod.MUTUAL_KNN:
        return build_mutual_knn_graph(X, k)
    elif method == TopologicalMethod.GABRIEL:
        return build_gabriel_graph(X)
    elif method == TopologicalMethod.RELATIVE_NEIGHBORHOOD:
        return build_relative_neighborhood_graph(X)
    elif method == TopologicalMethod.EPSILON:
        return build_epsilon_graph(X, epsilon)
    elif method == TopologicalMethod.ADAPTIVE_KNN:
        return build_adaptive_knn_graph(X, k_min=3, k_max=k)
    else:
        raise ValueError(f"Unknown topological method: {method}")


def forman_ricci_total(G: nx.Graph) -> float:
    """Compute total Forman-Ricci curvature for unweighted graphs."""
    deg = dict(G.degree())
    total = 0.0
    for u, v in G.edges():
        total += 4.0 - deg[u] - deg[v]
    return float(total)


def total_geodesic_sum_fast(G: nx.Graph,
                            allow_disconnected: bool = False,
                            disconnected_penalty: float = 1e6) -> float:
    """Compute full all-pairs shortest-path sum using SciPy's csgraph."""
    n = G.number_of_nodes()
    
    # Build CSR adjacency matrix
    rows = []
    cols = []
    for u, v in G.edges():
        rows.append(u); cols.append(v)
        rows.append(v); cols.append(u)
    data = np.ones(len(rows), dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # Compute shortest paths
    D = cs_shortest_path(A, directed=False, unweighted=True)
    
    if not allow_disconnected and np.isinf(D).any():
        raise nx.NetworkXNoPath("Graph disconnected; use --allow-disconnected or increase k")
    
    if allow_disconnected:
        D = np.where(np.isinf(D), disconnected_penalty, D)
    
    # Sum of upper triangle (i<j)
    iu = np.triu_indices(n, k=1)
    return float(D[iu].sum())


def ricci_for_k(
    layers: List[np.ndarray],
    k: int,
    method: TopologicalMethod,
    epsilon: float,
    sample_pairs: int,
    allow_disconnected: bool,
    disconnected_penalty: float,
) -> Dict[str, float]:
    """Compute Ricci metrics for a given k and topological method."""
    Ric_list: List[float] = []
    g_list: List[float] = []
    
    # Build graphs & compute metrics per layer
    for X in layers:
        G = build_topological_graph(X, method, k, epsilon)
        
        # Compute Ricci curvature
        Ric = forman_ricci_total(G)
        
        # Compute geodesic sum
        try:
            g = total_geodesic_sum_fast(G,
                                        allow_disconnected=allow_disconnected,
                                        disconnected_penalty=disconnected_penalty)
        except Exception as e:
            print(f"Warning: Fast geodesic computation failed: {e}")
            # Fallback to sampling method
            g = 0.0
            n = G.number_of_nodes()
            sample_size = min(sample_pairs, n * (n - 1) // 2)
            rng = np.random.default_rng(42)
            
            for _ in range(sample_size):
                i = int(rng.integers(0, n))
                j = int(rng.integers(0, n-1))
                if j >= i:
                    j += 1
                try:
                    d = nx.shortest_path_length(G, i, j)
                    g += float(d)
                except nx.NetworkXNoPath:
                    if allow_disconnected:
                        g += disconnected_penalty
                    else:
                        raise
            
            if sample_size > 0:
                factor = (n * (n - 1) // 2) / sample_size
                g *= factor
        
        Ric_list.append(Ric)
        g_list.append(g)
    
    # Compute eta_l = g_{l+1} - g_l for l=1..L-1
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
    
    return {"rho": rho, "z": z, "Ric": Ric_list, "g": g_list}


def main():
    ap = argparse.ArgumentParser(description="Topologically-focused Ricci analysis")
    ap.add_argument("--npz", required=True, help="Path to layer_outputs_test.npz")
    ap.add_argument("--method", type=str, default="knn", 
                   choices=[m.value for m in TopologicalMethod],
                   help="Topological graph construction method")
    ap.add_argument("--k-list", nargs="+", type=int, default=[6,7,9,10,15,18,20,30,50,90,100,120,150])
    ap.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for epsilon-neighborhood graphs")
    ap.add_argument("--sample-pairs", type=int, default=200000, help="0 means full all-pairs (slow)")
    ap.add_argument("--allow-disconnected", action="store_true")
    ap.add_argument("--disconnected-penalty", type=float, default=1e6)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    
    method = TopologicalMethod(args.method)
    
    data = np.load(args.npz)
    # Collect hidden layers in order
    layer_keys = sorted([k for k in data.keys() if k.startswith("hidden_")], key=lambda s: int(s.split("_")[1]))
    layers = [data[k].astype(np.float32) for k in layer_keys]
    print(f"[ricci_topological] method={method.value} | layers={len(layers)} | n_test={layers[0].shape[0]} | dims={[x.shape[1] for x in layers]}")
    
    results: Dict[int, Dict[str, float]] = {}
    best_k = None
    best_rho = +1.0
    
    for k in tqdm(args.k_list, desc=f"k sweep | {args.npz} | {method.value}"):
        try:
            res = ricci_for_k(layers, k, method, args.epsilon, args.sample_pairs, 
                            args.allow_disconnected, args.disconnected_penalty)
        except Exception as e:
            res = {"error": str(e)}
        results[k] = res
        if isinstance(res, dict) and "rho" in res and not np.isnan(res["rho"]):
            if res["rho"] < best_rho:
                best_rho = res["rho"]
                best_k = k
    
    summary = {
        "method": method.value,
        "npz": args.npz,
        "layers": layer_keys,
        "k_list": args.k_list,
        "epsilon": args.epsilon,
        "best_k": best_k,
        "best_rho": float(best_rho) if best_k is not None else None,
        "results": results,
    }
    
    print(f"\nRicci coefficient sweep (lower is more Ricci-like) - {method.value}:")
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
