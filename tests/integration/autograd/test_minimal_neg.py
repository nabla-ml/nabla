"""Minimal test for unary op gradient."""

import numpy as np
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace


def test_minimal_neg():
    """Test gradient of z = -x."""
    x = nb.Tensor.from_dlpack(np.array([2.0, -3.0], dtype=np.float32))
    
    def fn(a):
        return -a
    
    traced = trace(fn, x)
    cotangent = nb.Tensor.from_dlpack(np.ones(2, dtype=np.float32))
    grads = backward_on_trace(traced, cotangent)
    
    # z = -x
    # dz/dx = -1
    # grad_x = cot * (-1) = [-1, -1]
    
    grad_x_data = np.from_dlpack(grads[x])
    np.testing.assert_allclose(grad_x_data, np.array([-1.0, -1.0], dtype=np.float32), rtol=1e-5)
    
    print("✓ Minimal neg gradient test passed!")


if __name__ == "__main__":
    test_minimal_neg()
