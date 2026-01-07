"""Test xpr visualization of the full batched_matmul vmap transform."""

import nabla
from nabla import reduce_sum, reshape
from nabla.transforms.vmap import vmap
from nabla.utils.debug import capture_trace

# =============================================================================
# Building matmul from primitives (from test_vmap_matmul.py)
# =============================================================================

def dot(a, b):
    """Dot product: element-wise multiply + sum."""
    return reduce_sum(a * b, axis=0)

def mv_prod(matrix, vector):
    """Matrix-vector product: vmap(dot) over rows."""
    return vmap(dot, in_axes=(0, None))(matrix, vector)

def mm_prod(a, b):
    """Matrix-matrix product: vmap(mv_prod) over columns."""
    return vmap(mv_prod, in_axes=(None, 1), out_axes=1)(a, b)

def batched_matmul(a, b):
    """Batched matmul: vmap(mm_prod) over batches."""
    return vmap(mm_prod, in_axes=(0, None))(a, b)

# =============================================================================
# Trace and print
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DEBUG TRACE: Full Batched Matmul via Nested vmap")
    print("=" * 70)
    print()
    
    # (batch, m, k) @ (k, n) -> (batch, m, n)
    batch, m, k, n = 2, 3, 4, 5
    
    a = reshape(nabla.arange(0, batch * m * k), (batch, m, k))  # 2x3x4
    b = reshape(nabla.arange(0, k * n), (k, n))                  # 4x5
    
    print(f"Input a shape: {tuple(a.shape)}")
    print(f"Input b shape: {tuple(b.shape)}")
    print()
    
    print("-" * 70)
    print("TRACE: batched_matmul(a, b)")
    print("-" * 70)
    trace = capture_trace(batched_matmul, a, b)
    print(trace)
