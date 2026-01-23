"""Minimal test for MulOp gradient."""

import numpy as np
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace


def test_minimal_mul():
    """Test gradient of z = x * y."""
    x = nb.Tensor.from_dlpack(np.array([2.0, 3.0], dtype=np.float32))
    y = nb.Tensor.from_dlpack(np.array([4.0, 5.0], dtype=np.float32))
    
    def fn(a, b):
        return a * b
    
    traced = trace(fn, x, y)
    cotangent = nb.Tensor.from_dlpack(np.ones(2, dtype=np.float32))
    grads = backward_on_trace(traced, cotangent)
    
    # z = x * y
    # dz/dx = y
    # dz/dy = x
    # grad_x = cot * y = [4, 5]
    # grad_y = cot * x = [2, 3]
    
    grad_x_data = np.from_dlpack(grads[x])
    grad_y_data = np.from_dlpack(grads[y])
    
    np.testing.assert_allclose(grad_x_data, np.array([4.0, 5.0], dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(grad_y_data, np.array([2.0, 3.0], dtype=np.float32), rtol=1e-5)
    
    print("✓ Minimal mul gradient test passed!")


if __name__ == "__main__":
    test_minimal_mul()
