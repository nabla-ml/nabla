"""Minimal test for autograd - just test AddOp gradient."""

import numpy as np
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace


def test_minimal_add():
    """Simplest possible gradient test: z = x + y."""
    # Create inputs
    x = nb.Tensor.from_dlpack(np.array([2.0, 3.0], dtype=np.float32))
    y = nb.Tensor.from_dlpack(np.array([4.0, 5.0], dtype=np.float32))
    
    # Define function
    def fn(a, b):
        return a + b
    
    # Trace it
    traced = trace(fn, x, y)
    print(f"Trace: {repr(traced)}")
    print(f"Trace nodes: {traced.nodes}")
    
    # Create cotangent (gradient from next layer - all ones for simplicity)
    cotangent = nb.Tensor.from_dlpack(np.ones(2, dtype=np.float32))
    
    # Compute gradients
    grads = backward_on_trace(traced, cotangent)
    
    print(f"Gradients: {grads}")
    
    # For z = x + y, we expect:
    # dz/dx = 1
    # dz/dy = 1
    # So grad_x = cotangent * 1 = [1, 1]
    # And grad_y = cotangent * 1 = [1, 1]
    
    assert x in grads, "x should have a gradient"
    assert y in grads, "y should have a gradient"
    
    grad_x_data = np.from_dlpack(grads[x])
    grad_y_data = np.from_dlpack(grads[y])
    
    expected = np.ones(2, dtype=np.float32)
    np.testing.assert_allclose(grad_x_data, expected, rtol=1e-5)
    np.testing.assert_allclose(grad_y_data, expected, rtol=1e-5)
    
    print("✓ Minimal add gradient test passed!")


if __name__ == "__main__":
    test_minimal_add()
