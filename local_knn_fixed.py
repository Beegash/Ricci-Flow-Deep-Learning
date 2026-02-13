#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local kNN-based Ricci Flow Analysis
====================================

Computes per-layer Ricci coefficients using per-data-point scalar curvature
and local expansion, following the approach from Baptista et al.

Supported curvature types:
- Forman-Ricci
- Augmented-Forman-Ricci
- Approx-Ollivier-Ricci (Tian et al. 2023)
- Ollivier-Ricci (Earth Mover's Distance)

Key difference from knn_fixed_2_1.py:
- This computes per-NODE curvature and expansion
- Correlates across data points (not layers) to get per-layer coefficient
- L1, L2, ... values are Pearson r in range [-1, 1]
"""

import os
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix, triu as sp_triu, lil_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Optional
import warnings
import multiprocessing as mp

try:
    import ot
    HAS_OT = True
except ImportError:
    HAS_OT = False
    warnings.warn("'ot' (POT) package not installed. Ollivier-Ricci curvature will not be available.")


# =============================================================================
# RICCI CURVATURE COMPUTATION (Edge-wise, not summed)
# =============================================================================

def compute_forman_ricci_matrix(A: csr_matrix) -> csr_matrix:
    """
    Compute Forman-Ricci curvature for each edge (returns matrix, not scalar).
    
    Ric(e_ij) = 4 - deg(i) - deg(j)
    
    Returns:
        csr_matrix: Symmetric matrix with Ricci curvature for each edge
    """
    n = A.shape[0]
    degrees = np.asarray(A.sum(axis=1)).ravel()
    Ric = lil_matrix(A.shape, dtype=np.float32)
    
    A_upper = sp_triu(A, k=1)
    rows, cols = A_upper.nonzero()
    
    for i, j in zip(rows, cols):
        curv = 4.0 - degrees[i] - degrees[j]
        Ric[i, j] = curv
        Ric[j, i] = curv
    
    return Ric.tocsr()


def compute_augmented_forman_ricci_matrix(A: csr_matrix) -> csr_matrix:
    """
    Compute Augmented Forman-Ricci curvature for each edge.
    
    Ric(e_ij) = 4 - deg(i) - deg(j) + 3 * triangles(i,j)
    
    Returns:
        csr_matrix: Symmetric matrix with Augmented Forman-Ricci curvature for each edge
    """
    degrees = np.asarray(A.sum(axis=1)).ravel()
    A2 = A @ A  # A^2[i,j] counts common neighbors (triangles through edge i-j)
    Ric = lil_matrix(A.shape, dtype=np.float32)
    
    A_upper = sp_triu(A, k=1)
    rows, cols = A_upper.nonzero()
    
    for i, j in zip(rows, cols):
        triangles = A2[i, j]
        curv = 4.0 - degrees[i] - degrees[j] + 3.0 * triangles
        Ric[i, j] = curv
        Ric[j, i] = curv
    
    return Ric.tocsr()


def compute_approx_ollivier_ricci_matrix(A: csr_matrix) -> csr_matrix:
    """
    Compute the Approximation of Ollivier-Ricci curvature proposed by
    Tian et al. in "Curvature-based Clustering on Graphs" (2023).
    
    Returns:
        csr_matrix: Symmetric matrix with approximate Ollivier-Ricci curvature for each edge
    """
    degrees = np.asarray(A.sum(axis=1)).ravel()
    A2 = A @ A
    Ric = lil_matrix(A.shape, dtype=np.float32)
    
    A_upper = sp_triu(A, k=1)
    rows, cols = A_upper.nonzero()
    
    for i, j in zip(rows, cols):
        triangles = A2[i, j]
        di, dj = degrees[i], degrees[j]
        curv = (0.5 * (triangles / max(di, dj))
                - 0.5 * (max(0, 1 - 1/di - 1/dj - triangles / min(di, dj))
                         + max(0, 1 - 1/di - 1/dj - triangles / max(di, dj))
                         - triangles / max(di, dj)))
        Ric[i, j] = curv
        Ric[j, i] = curv
    
    return Ric.tocsr()


def _orc_edge(args):
    """Helper for multiprocessing Ollivier-Ricci: compute curvature for one edge."""
    edge, apsp, A_csr = args
    i, j = edge
    S_1x = A_csr[i].indices
    S_1y = A_csr[j].indices
    d_x = len(S_1x)
    d_y = len(S_1y)
    cost_matrix = np.array([[apsp[k, l] for l in S_1y] for k in S_1x])
    W = ot.emd2([1/d_x]*d_x, [1/d_y]*d_y, cost_matrix)
    return (i, j, 1 - W)


def compute_ollivier_ricci_matrix(A: csr_matrix, apsp: np.ndarray) -> csr_matrix:
    """
    Compute Ollivier-Ricci curvature for each edge using optimal transport.
    Uses multiprocessing for speed.
    
    Args:
        A: Adjacency matrix (symmetric, unweighted)
        apsp: All-pairs shortest path distance matrix
    
    Returns:
        csr_matrix: Symmetric matrix with Ollivier-Ricci curvature for each edge
    """
    if not HAS_OT:
        raise ImportError("'ot' (POT) package is required for Ollivier-Ricci curvature. "
                          "Install with: pip install POT")
    
    Ric = lil_matrix(A.shape, dtype=np.float32)
    A_upper = sp_triu(A, k=1)
    edges = np.vstack(A_upper.nonzero()).T
    
    args = [(tuple(e), apsp, A) for e in edges]
    n_workers = max(1, mp.cpu_count())
    chunksize, extra = divmod(len(edges), n_workers * 4)
    if extra:
        chunksize += 1
    chunksize = max(1, chunksize)
    
    with mp.get_context("fork").Pool(n_workers) as pool:
        results = pool.imap_unordered(_orc_edge, args, chunksize=chunksize)
        for i, j, curv in results:
            Ric[i, j] = curv
            Ric[j, i] = curv
    
    return Ric.tocsr()


# Curvature type constants
CURVATURE_TYPES = ['Forman-Ricci', 'Augmented-Forman-Ricci', 'Approx-Ollivier-Ricci', 'Ollivier-Ricci']
CURVATURE_PREFIXES = {
    'Forman-Ricci': 'FR',
    'Augmented-Forman-Ricci': 'AFR',
    'Approx-Ollivier-Ricci': 'AOR',
    'Ollivier-Ricci': 'OR',
}


def compute_curvature_matrix(A: csr_matrix, curv: str, apsp: np.ndarray = None) -> csr_matrix:
    """
    Dispatcher: compute curvature matrix based on curvature type string.
    
    Args:
        A: Adjacency matrix
        curv: One of 'Forman-Ricci', 'Augmented-Forman-Ricci', 'Approx-Ollivier-Ricci', 'Ollivier-Ricci'
        apsp: All-pairs shortest path matrix (required for Ollivier-Ricci)
    
    Returns:
        csr_matrix: Symmetric curvature matrix
    """
    if curv == 'Forman-Ricci':
        return compute_forman_ricci_matrix(A)
    elif curv == 'Augmented-Forman-Ricci':
        return compute_augmented_forman_ricci_matrix(A)
    elif curv == 'Approx-Ollivier-Ricci':
        return compute_approx_ollivier_ricci_matrix(A)
    elif curv == 'Ollivier-Ricci':
        if apsp is None:
            raise ValueError("apsp (all-pairs shortest paths) is required for Ollivier-Ricci curvature.")
        return compute_ollivier_ricci_matrix(A, apsp)
    else:
        raise ValueError(f"Curvature type '{curv}' not supported. Use one of: {CURVATURE_TYPES}")


def compute_scalar_curvature(A: csr_matrix, Ric: csr_matrix) -> np.ndarray:
    """
    Compute scalar curvature at each node.
    
    scalar_curv(x) = sum(Ric[x, neighbors]) / degree(x)
    
    Returns:
        np.ndarray: Scalar curvature for each of N nodes
    """
    n = A.shape[0]
    scalar_curvs = np.zeros(n, dtype=np.float32)
    
    for x in range(n):
        neighbors = A[x].indices
        if len(neighbors) > 0:
            curv_sum = Ric[x, neighbors].sum()
            scalar_curvs[x] = curv_sum / len(neighbors)
    
    return scalar_curvs


# =============================================================================
# kNN GRAPH CONSTRUCTION
# =============================================================================

def build_knn_graph(X: np.ndarray, k: int) -> csr_matrix:
    """Build undirected, unweighted k-NN graph."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.dtype != np.float32 and X.dtype != np.float64:
        X = X.astype(np.float32, copy=False)
    
    # Directed k-NN graph
    A_directed = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    # Make undirected (symmetric)
    A = A_directed.maximum(A_directed.T)
    return A.tocsr()


# =============================================================================
# LAYER RICCI COEFFICIENT
# =============================================================================

def compute_layer_ricci_coefficients(
    activations: List[np.ndarray], 
    k: int,
    curv: str = 'Forman-Ricci'
) -> np.ndarray:
    """
    Compute per-layer Ricci coefficients.
    
    For each layer l:
    1. Compute scalar curvature at each data point
    2. Compute local expansion/contraction at each data point
    3. Correlate across data points → layer Ricci coefficient
    
    Args:
        activations: List of L activation arrays, each shape (N, d_l)
        k: Number of neighbors for kNN graph
        
    Returns:
        np.ndarray: Array of L-1 layer Ricci coefficients
    """
    L = len(activations)
    N = len(activations[0])  # Number of data points
    
    if L < 2:
        return np.array([])
    
    # Build kNN graphs for all layers
    knn_graphs = []
    for act in activations:
        A = build_knn_graph(act, k)
        knn_graphs.append(A)
    
    # Compute all-pairs shortest paths for all layers
    apsps = []
    for A in knn_graphs:
        apsp = dijkstra(csgraph=A, directed=False, unweighted=True, return_predecessors=False)
        apsps.append(apsp)
    
    # Compute curvature matrices for all layers (except last)
    curvatures = []
    for i in range(L - 1):
        Ric = compute_curvature_matrix(knn_graphs[i], curv, apsp=apsps[i])
        curvatures.append(Ric)
    
    # Compute layer Ricci coefficients
    layer_ricci_coefficients = np.empty(L - 1, dtype=np.float32)
    
    for i in range(L - 1):
        scalar_curvs = []
        eta = []
        
        for x in range(N):
            # Get neighbors of x in layer i
            neighbors = knn_graphs[i][x].indices
            
            if len(neighbors) == 0:
                continue
            
            # Check if one-hop neighborhoods are connected in next layer
            one_hop_connected = True
            expansion = 0.0
            
            for y in neighbors:
                if np.isinf(apsps[i + 1][x, y]):
                    warnings.warn(f'One-hop neighbors not connected in layer {i+1}')
                    one_hop_connected = False
                    break
                expansion += apsps[i + 1][x, y] - apsps[i][x, y]
            
            if one_hop_connected:
                # Scalar curvature at point x
                curv_sum = curvatures[i][x, neighbors].sum()
                scalar_curvs.append(curv_sum / len(neighbors))
                
                # Local expansion at point x
                eta.append(expansion / len(neighbors))
        
        # Compute Pearson correlation across data points
        if len(scalar_curvs) >= 2:
            r, _ = pearsonr(scalar_curvs, eta)
            layer_ricci_coefficients[i] = r
        else:
            layer_ricci_coefficients[i] = np.nan
    
    return layer_ricci_coefficients


def compute_global_ricci_coefficient(
    activations: List[np.ndarray],
    k: int
) -> Tuple[float, float]:
    """
    Compute the global Ricci coefficient (r_all, r_skip).
    
    This is the original approach: correlate Δg with Ric across layers.
    
    Returns:
        Tuple[r_all, r_skip]: Global correlation coefficients
    """
    L = len(activations)
    
    if L < 2:
        return np.nan, np.nan
    
    # Build graphs and compute global metrics per layer
    g_list = []
    ric_list = []
    
    for act in activations:
        A = build_knn_graph(act, k)
        
        # Geodesic mass (sum of all shortest paths)
        apsp = dijkstra(csgraph=A, directed=False, unweighted=True, return_predecessors=False)
        g = apsp[np.triu_indices_from(apsp, k=1)].sum()
        g_list.append(g)
        
        # Global Forman-Ricci (sum)
        Ric_matrix = compute_forman_ricci_matrix(A)
        ric = Ric_matrix.sum() / 2  # Divide by 2 since symmetric
        ric_list.append(ric)
    
    g = np.array(g_list)
    Ric = np.array(ric_list)
    
    # Δg_l = g_{l+1} - g_l
    dg = g[1:] - g[:-1]
    
    # Correlate Δg_l with Ric_l (for l = 0 to L-2)
    if len(dg) >= 2:
        r_all, _ = pearsonr(Ric[:-1], dg)
    else:
        r_all = np.nan
    
    # Skip first layer
    if len(dg) >= 3:
        r_skip, _ = pearsonr(Ric[1:-1], dg[1:])
    else:
        r_skip = np.nan
    
    return r_all, r_skip


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_network(
    activations: List[np.ndarray],
    k: int = 350
) -> Dict:
    """
    Complete analysis of a network's layer activations.
    
    Args:
        activations: List of activation arrays per layer
        k: Number of neighbors for kNN graph
        
    Returns:
        Dict with:
            - 'layer_coefficients': Array of per-layer Ricci coefficients (L1, L2, ...)
            - 'r_all': Global Ricci coefficient (all layers)
            - 'r_skip': Global Ricci coefficient (skip first layer)
    """
    # Per-layer Ricci coefficients
    layer_coefs = compute_layer_ricci_coefficients(activations, k)
    
    # Global Ricci coefficients
    r_all, r_skip = compute_global_ricci_coefficient(activations, k)
    
    return {
        'layer_coefficients': layer_coefs,
        'r_all': r_all,
        'r_skip': r_skip
    }


def format_layer_dict(layer_coefs: np.ndarray, max_layers: int = 12, prefix: str = '') -> Dict:
    """
    Format layer coefficients as L1, L2, ..., L12 dict with optional prefix.
    
    Args:
        layer_coefs: Array of layer coefficients
        max_layers: Maximum number of layers to include
        prefix: Optional prefix for column names (e.g., 'FR' -> 'FR_L1', 'FR_L2', ...)
    
    Returns:
        Dict with keys like 'L1' or 'FR_L1' to 'L12' or 'FR_L12' (NaN for unused layers)
    """
    result = {}
    key_prefix = f'{prefix}_' if prefix else ''
    for i in range(1, max_layers + 1):
        if i <= len(layer_coefs):
            result[f'{key_prefix}L{i}'] = float(layer_coefs[i - 1])
        else:
            result[f'{key_prefix}L{i}'] = np.nan
    return result


# =============================================================================
# TEST/DEMO
# =============================================================================

if __name__ == "__main__":
    print("Local kNN Ricci Coefficient Analysis")
    print("=" * 50)
    
    # Create synthetic test data
    np.random.seed(42)
    N = 100  # Data points
    
    # Simulate 5-layer network activations
    activations = [
        np.random.randn(N, 64),   # Layer 1
        np.random.randn(N, 64),   # Layer 2
        np.random.randn(N, 64),   # Layer 3
        np.random.randn(N, 64),   # Layer 4
        np.random.randn(N, 64),   # Layer 5
    ]
    
    print(f"Analyzing {len(activations)} layers with {N} data points...")
    print(f"Using k=50 for test")
    
    result = analyze_network(activations, k=50)
    
    print("\nResults:")
    print(f"  Layer coefficients: {result['layer_coefficients']}")
    print(f"  r_all: {result['r_all']:.4f}")
    print(f"  r_skip: {result['r_skip']:.4f}")
    
    layer_dict = format_layer_dict(result['layer_coefficients'])
    print(f"\n  L1-L5: {[layer_dict[f'L{i}'] for i in range(1, 6)]}")
