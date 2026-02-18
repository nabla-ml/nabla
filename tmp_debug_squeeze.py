"""Minimal probe to isolate squeeze_physical fwd_fwd failure."""
import os
os.environ["NABLA_DEBUG_NESTED_JVP"] = "1"

import nabla as nb
import numpy as np

# x shape (2, 3, 4), reduce_sum_physical on axis=1 with keepdims=False
x_np = np.random.default_rng(311).standard_normal((2, 3, 4)).astype(np.float32)
x = nb.Tensor.from_dlpack(x_np)

def f(x):
    y = nb.reduce_sum_physical(x * x, axis=1, keepdims=False)
    return nb.reduce_sum(y * y)

# Test single jacfwd first
print("=== Single jacfwd ===")
try:
    j1 = nb.jacfwd(f)(x)
    print(f"  jacfwd OK, shape={j1.shape}")
except Exception as e:
    print(f"  jacfwd FAIL: {e}")

# Test fwd_fwd (nested jacfwd)
print("\n=== Nested jacfwd (fwd_fwd) ===")
try:
    h = nb.jacfwd(nb.jacfwd(f))(x)
    print(f"  fwd_fwd OK, shape={h.shape}")
except Exception as e:
    import traceback
    print(f"  fwd_fwd FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
