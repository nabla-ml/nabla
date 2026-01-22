#!/usr/bin/env python3
"""Simple test for Trace-based backpropagation."""

import numpy as np
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace


def test_simple_add_mul():
    """Test: y = x1 * x2 + x1"""
    print("=" * 70)
    print("Test: y = x1 * x2 + x1")
    print("=" * 70)
    
    # Create inputs
    x1 = nb.Tensor.from_dlpack(np.array([2.0, 3.0], dtype=np.float32))
    x2 = nb.Tensor.from_dlpack(np.array([4.0, 5.0], dtype=np.float32))
    
    # Define traced computation
    def compute(x1, x2):
        prod = nb.mul(x1, x2)
        result = nb.add(prod, x1)
        return result
    
    # Trace the computation
    traced = trace(compute, x1, x2)
    
    print("\nForward Trace:")
    print(traced)
    
    # Get the output
    y = compute(x1, x2)
    print(f"\nOutput: {y}")
    
    # Create cotangent (gradient of loss w.r.t. output)
    # For simplicity, use ones (as if loss = sum(y))
    cotangent = nb.Tensor.from_dlpack(np.ones_like(y.to_numpy()))
    
    # Run backprop
    print("\nRunning backward_on_trace...")
    gradients = backward_on_trace(traced, cotangent)
    
    print(f"\nNumber of gradients computed: {len(gradients)}")
    
    # Extract gradients
    grad_x1 = None
    grad_x2 = None
    
    for inp in [x1, x2]:
        inp_id = id(inp._impl)
        if inp_id in gradients:
            grad_impl = gradients[inp_id]
            grad_tensor = nb.Tensor(impl=grad_impl)
            if inp is x1:
                grad_x1 = grad_tensor
            else:
                grad_x2 = grad_tensor
    
    print(f"\nGradient w.r.t. x1: {grad_x1}")
    print(f"Gradient w.r.t. x2: {grad_x2}")
    
    # Expected gradients:
    # dy/dx1 = x2 + 1 = [4+1, 5+1] = [5, 6]
    # dy/dx2 = x1 = [2, 3]
    print(f"\nExpected grad_x1: [5.0, 6.0]")
    print(f"Expected grad_x2: [2.0, 3.0]")


def test_matmul():
    """Test: y = matmul(x1, x2)"""
    print("\n" + "=" * 70)
    print("Test: y = matmul(x1, x2)")
    print("=" * 70)
    
    # Create inputs
    x1 = nb.Tensor.from_dlpack(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))  # 2x2
    x2 = nb.Tensor.from_dlpack(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))  # 2x2
    
    # Define traced computation
    def compute(x1, x2):
        return nb.matmul(x1, x2)
    
    # Trace the computation
    traced = trace(compute, x1, x2)
    
    print("\nForward Trace:")
    print(traced)
    
    # Get the output
    y = compute(x1, x2)
    print(f"\nOutput:\n{y}")
    
    # Create cotangent
    cotangent = nb.Tensor.from_dlpack(np.ones_like(y.to_numpy()))
    
    # Run backprop
    print("\nRunning backward_on_trace...")
    gradients = backward_on_trace(traced, cotangent)
    
    # Extract gradients
    grad_x1 = None
    grad_x2 = None
    
    for inp in [x1, x2]:
        inp_id = id(inp._impl)
        if inp_id in gradients:
            grad_impl = gradients[inp_id]
            grad_tensor = nb.Tensor(impl=grad_impl)
            if inp is x1:
                grad_x1 = grad_tensor
            else:
                grad_x2 = grad_tensor
    
    print(f"\nGradient w.r.t. x1:\n{grad_x1}")
    print(f"\nGradient w.r.t. x2:\n{grad_x2}")
    
    # For matmul: dy/dx1 = cotangent @ x2^T, dy/dx2 = x1^T @ cotangent
    print(f"\nExpected grad_x1 (ones @ x2^T):")
    print(f"Expected grad_x2 (x1^T @ ones):")


if __name__ == "__main__":
    test_simple_add_mul()
    test_matmul()
