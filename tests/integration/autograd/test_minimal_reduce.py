"""Minimal test for reduction op gradient."""

import numpy as np
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace


def test_minimal_reduce_sum():
    """Test gradient of z = sum(x)."""
    x = nb.Tensor.from_dlpack(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    
    def fn(a):
        return a.sum(axis=0)
    
    traced = trace(fn, x)
    # result shape is (2,)
    cotangent = nb.Tensor.from_dlpack(np.array([10.0, 20.0], dtype=np.float32))
    grads = backward_on_trace(traced, cotangent)
    
    # z = sum(x, axis=0)
    # dz/dx = 1 (broadcasted)
    # grad_x = [[10, 20], [10, 20]]
    
    grad_x_data = np.from_dlpack(grads[x])
    expected = np.array([[10.0, 20.0], [10.0, 20.0]], dtype=np.float32)
    np.testing.assert_allclose(grad_x_data, expected, rtol=1e-5)
    
    print("✓ Minimal reduce_sum gradient test passed!")


if __name__ == "__main__":
    test_minimal_reduce_sum()
