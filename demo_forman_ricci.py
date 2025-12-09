#!/usr/bin/env python3
"""
Demonstration of global_forman_ricci function with step-by-step output.
"""

import numpy as np
from scipy.sparse import csr_matrix, triu as sp_triu

# Create a sample undirected graph with 5 nodes
# Graph structure:
#   1 -- 2
#   |    |
#   4 -- 3 -- 5
#   |    |
#   5 -- 3 (already connected)
#
# Edges: (1,2), (1,4), (2,3), (3,4), (3,5), (4,5)

# Create adjacency matrix (5x5)
adjacency_data = np.array([
    [0, 1, 0, 1, 0],  # Node 1: connected to 2, 4
    [1, 0, 1, 0, 0],  # Node 2: connected to 1, 3
    [0, 1, 0, 1, 1],  # Node 3: connected to 2, 4, 5
    [1, 0, 1, 0, 1],  # Node 4: connected to 1, 3, 5
    [0, 0, 1, 1, 0],  # Node 5: connected to 3, 4
])

print("=" * 70)
print("STEP-BY-STEP DEMONSTRATION OF global_forman_ricci FUNCTION")
print("=" * 70)

print("\n1. INPUT: Adjacency Matrix A (5x5)")
print("-" * 70)
print("   Nodes: 1, 2, 3, 4, 5")
print("   Edges: (1,2), (1,4), (2,3), (3,4), (3,5), (4,5)")
print("\n   Adjacency Matrix:")
print(adjacency_data)
print("\n   Matrix interpretation:")
print("   - Row i = node i")
print("   - Column j = node j")
print("   - A[i,j] = 1 means edge exists between node i and node j")

# Convert to CSR format (as the function expects)
A = csr_matrix(adjacency_data)
print(f"\n   Converted to CSR sparse matrix: {A.shape}")

print("\n" + "=" * 70)
print("2. CALCULATE DEGREES: deg = A.sum(axis=1)")
print("-" * 70)
print("   Summing each row (axis=1) to get degree of each node:")
print("   - Row 0 (Node 1): sum =", adjacency_data[0].sum())
print("   - Row 1 (Node 2): sum =", adjacency_data[1].sum())
print("   - Row 2 (Node 3): sum =", adjacency_data[2].sum())
print("   - Row 3 (Node 4): sum =", adjacency_data[3].sum())
print("   - Row 4 (Node 5): sum =", adjacency_data[4].sum())

deg = np.asarray(A.sum(axis=1)).ravel()
print("\n   Result: deg =", deg)
print("   Interpretation:")
for i, d in enumerate(deg):
    print(f"   - Node {i+1} has degree {d} (connected to {d} neighbors)")

print("\n" + "=" * 70)
print("3. EXTRACT UPPER TRIANGLE: A_ut = sp_triu(A, k=1).tocoo()")
print("-" * 70)
print("   Why upper triangle? To count each undirected edge only once.")
print("   k=1 means we exclude the diagonal (no self-loops).")
print("\n   Full matrix (symmetric):")
print(adjacency_data)
print("\n   Upper triangle (k=1, excluding diagonal):")
A_ut = sp_triu(A, k=1).tocoo()
upper_triangle = np.zeros_like(adjacency_data)
for i, j, val in zip(A_ut.row, A_ut.col, A_ut.data):
    upper_triangle[i, j] = val
print(upper_triangle)
print("\n   Edges in upper triangle (i < j):")
edges = []
for i, j, val in zip(A_ut.row, A_ut.col, A_ut.data):
    if val > 0:
        edges.append((i+1, j+1))  # +1 for 1-based node numbering
        print(f"   - Edge ({i+1}, {j+1})")
print(f"\n   Total edges to process: {len(edges)}")

print("\n" + "=" * 70)
print("4. CALCULATE CURVATURE PER EDGE: curv = 4.0 - deg[i] - deg[j]")
print("-" * 70)
print("   Formula: R(i,j) = 4 - deg(i) - deg(j)")
print("   For each edge in the upper triangle:\n")
curv = 4.0 - deg[A_ut.row] - deg[A_ut.col]
curvatures = []
for idx, (i, j) in enumerate(edges):
    deg_i = deg[i-1]  # Convert to 0-based
    deg_j = deg[j-1]
    curvature = curv[idx]
    curvatures.append(curvature)
    print(f"   Edge ({i}, {j}):")
    print(f"     - deg({i}) = {deg_i}")
    print(f"     - deg({j}) = {deg_j}")
    print(f"     - R({i},{j}) = 4 - {deg_i} - {deg_j} = {curvature:.1f}")
    print()

print("\n   All curvatures:", curv)
print("   Curvature array shape:", curv.shape)

print("\n" + "=" * 70)
print("5. SUM ALL CURVATURES: return float(curv.sum())")
print("-" * 70)
total_ricci = float(curv.sum())
print(f"   Sum of all edge curvatures: {total_ricci:.1f}")
print("\n   Breakdown:")
for idx, (i, j) in enumerate(edges):
    print(f"   + R({i},{j}) = {curvatures[idx]:.1f}")
print(f"   ─────────────────────────────")
print(f"   Total Ric = {total_ricci:.1f}")

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)
print(f"   Global Forman-Ricci coefficient: {total_ricci:.1f}")
print("\n   Interpretation:")
print("   - This is the sum of Forman-Ricci curvatures over all edges")
print("   - Negative values indicate expansion-like behavior")
print("   - Positive values indicate contraction-like behavior")
print("   - In this example, the graph has overall negative curvature,")
print("     suggesting an expansion-like geometry")

print("\n" + "=" * 70)
print("VERIFICATION: Manual calculation")
print("=" * 70)
print("   Let's verify by calculating manually:")
manual_sum = 0
for i, j in edges:
    deg_i = deg[i-1]
    deg_j = deg[j-1]
    r = 4.0 - deg_i - deg_j
    manual_sum += r
    print(f"   R({i},{j}) = 4 - {deg_i} - {deg_j} = {r:.1f}")
print(f"\n   Manual sum: {manual_sum:.1f}")
print(f"   Function result: {total_ricci:.1f}")
print(f"   Match: {'✓ YES' if abs(manual_sum - total_ricci) < 0.001 else '✗ NO'}")




