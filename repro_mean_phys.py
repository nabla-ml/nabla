import os
import sys

# Add the project root to sys.path
sys.path.append("/Users/tillife/Documents/CodingProjects/nabla")

import nabla as nb
import numpy as np
from tests.unit.common import cleanup_caches

def test_hessian_mean_physical():
    cleanup_caches()
    print("--- Testing Hessian mean_physical ---")
    x = nb.Tensor.ones((2, 3))
    
    def f(x):
        y = nb.mean_physical(x, axis=1) # Reduce '3' dim
        return nb.reduce_sum(y * y)

    os.environ["NABLA_DEBUG_OP_CALL"] = "1"
    os.environ["NABLA_DEBUG_PHYS"] = "1"
    
    print("\nCalling jacfwd(jacfwd(f))(x)...")
    try:
        hess_fn = nb.jacfwd(nb.jacfwd(f))
        res = hess_fn(x)
        print("\nResult realize...")
        res.realize()
        print("\nSuccess!")
    except Exception as e:
        print(f"\nCaught Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hessian_mean_physical()
