#!/usr/bin/env python3
"""
Demonstration of sparse matrix formats and COO (Coordinate) format.
Shows what sp_triu().tocoo() returns and how .row and .col work.
"""

import numpy as np
from scipy.sparse import csr_matrix, triu as sp_triu

# Create the same sample adjacency matrix
adjacency_data = np.array([
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1],
    [0, 0, 1, 1, 0],
])

print("=" * 70)
print("UNDERSTANDING sp_triu(A, k=1).tocoo() RETURN TYPE")
print("=" * 70)

# Start with CSR format (as in the actual function)
A = csr_matrix(adjacency_data)
print("\n1. INPUT: A (CSR format)")
print("-" * 70)
print(f"   Type: {type(A)}")
print(f"   Format: {A.format}")
print(f"   Shape: {A.shape}")
print("\n   Full matrix representation:")
print(adjacency_data)

print("\n" + "=" * 70)
print("2. sp_triu(A, k=1) - Extract Upper Triangle")
print("-" * 70)
A_ut_sparse = sp_triu(A, k=1)
print(f"   Type after sp_triu: {type(A_ut_sparse)}")
print(f"   Format: {A_ut_sparse.format}")
print(f"   Shape: {A_ut_sparse.shape}")
print("\n   Upper triangle matrix (only i < j):")
print(A_ut_sparse.toarray())

print("\n" + "=" * 70)
print("3. .tocoo() - Convert to COO (Coordinate) Format")
print("-" * 70)
A_ut = A_ut_sparse.tocoo()
print(f"   Type after .tocoo(): {type(A_ut)}")
print(f"   Format: {A_ut.format}")
print(f"   Shape: {A_ut.shape}")

print("\n   COO format stores sparse matrices as three arrays:")
print(f"   - .data:  {A_ut.data}")
print(f"   - .row:   {A_ut.row}")
print(f"   - .col:   {A_ut.col}")

print("\n   What do these mean?")
print("   COO = Coordinate format - stores only non-zero entries")
print("   Each non-zero entry is represented by:")
print("   - row index (which row it's in)")
print("   - column index (which column it's in)")
print("   - value (the actual number)")

print("\n" + "=" * 70)
print("4. INTERPRETING .row and .col ARRAYS")
print("-" * 70)
print("   The COO format stores edges as:")
print("   (row[i], col[i]) = data[i]")
print("\n   Let's see each non-zero entry:")
for i in range(len(A_ut.data)):
    row_idx = A_ut.row[i]
    col_idx = A_ut.col[i]
    value = A_ut.data[i]
    print(f"   Entry {i}: A_ut[{row_idx}, {col_idx}] = {value}")
    print(f"            → Edge from node {row_idx+1} to node {col_idx+1}")

print("\n" + "=" * 70)
print("5. WHY USE COO FORMAT?")
print("-" * 70)
print("   COO format is perfect for this use case because:")
print("   1. Easy iteration: We can loop through all edges")
print("   2. Direct access: .row and .col give us node indices")
print("   3. Efficient: Only stores non-zero entries")
print("\n   Example usage in the function:")
print("   ```python")
print("   A_ut = sp_triu(A, k=1).tocoo()")
print("   curv = 4.0 - deg[A_ut.row] - deg[A_ut.col]")
print("   ```")
print("\n   This works because:")
print("   - A_ut.row[i] gives us the row index (node i)")
print("   - A_ut.col[i] gives us the column index (node j)")
print("   - deg[A_ut.row[i]] gets the degree of node i")
print("   - deg[A_ut.col[i]] gets the degree of node j")

print("\n" + "=" * 70)
print("6. DEMONSTRATION: How deg[A_ut.row] and deg[A_ut.col] Work")
print("-" * 70)
deg = np.asarray(A.sum(axis=1)).ravel()
print(f"   Degrees: deg = {deg}")
print(f"   (Node 0 has degree {deg[0]}, Node 1 has degree {deg[1]}, etc.)")
print("\n   For each edge in A_ut:")
print("   " + "-" * 60)
for i in range(len(A_ut.data)):
    row_idx = A_ut.row[i]
    col_idx = A_ut.col[i]
    deg_i = deg[row_idx]
    deg_j = deg[col_idx]
    print(f"   Edge {i}: ({row_idx+1}, {col_idx+1})")
    print(f"     - A_ut.row[{i}] = {row_idx} → deg[{row_idx}] = {deg_i}")
    print(f"     - A_ut.col[{i}] = {col_idx} → deg[{col_idx}] = {deg_j}")
    print(f"     - Curvature = 4 - {deg_i} - {deg_j} = {4 - deg_i - deg_j:.1f}")
    print()

print("\n" + "=" * 70)
print("7. VECTORIZED OPERATION EXPLANATION")
print("-" * 70)
print("   The line: curv = 4.0 - deg[A_ut.row] - deg[A_ut.col]")
print("   uses NumPy array indexing to compute all curvatures at once:")
print("\n   Step-by-step:")
print(f"   1. A_ut.row = {A_ut.row}")
print(f"   2. deg[A_ut.row] = {deg[A_ut.row]}  (degrees of all 'i' nodes)")
print(f"   3. A_ut.col = {A_ut.col}")
print(f"   4. deg[A_ut.col] = {deg[A_ut.col]}  (degrees of all 'j' nodes)")
print(f"   5. curv = 4.0 - {deg[A_ut.row]} - {deg[A_ut.col]}")
curv = 4.0 - deg[A_ut.row] - deg[A_ut.col]
print(f"      = {curv}")
print("\n   This is much faster than looping!")

print("\n" + "=" * 70)
print("8. COMPARING SPARSE MATRIX FORMATS")
print("-" * 70)
print("   Different formats store the same data differently:")
print("\n   CSR (Compressed Sparse Row):")
print("   - Good for: row operations, matrix-vector products")
print("   - Storage: indptr, indices, data arrays")
print("   - No direct .row/.col access")
print("\n   COO (Coordinate):")
print("   - Good for: construction, iteration over non-zeros")
print("   - Storage: row, col, data arrays")
print("   - Direct .row/.col access ← Why we use it here!")
print("\n   CSC (Compressed Sparse Column):")
print("   - Good for: column operations")
print("   - Storage: similar to CSR but column-wise")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("   Return type of sp_triu(A, k=1).tocoo():")
print(f"   → {type(A_ut)} (scipy.sparse.coo.coo_matrix)")
print("\n   A_ut has .row and .col because:")
print("   → COO format stores sparse matrices as (row, col, data) tuples")
print("   → .row[i] = row index of i-th non-zero entry")
print("   → .col[i] = column index of i-th non-zero entry")
print("   → This allows easy iteration and indexing!")




