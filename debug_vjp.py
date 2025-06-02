#!/usr/bin/env python3
"""Debug VJP functionality to see what's happening with gradients."""

import numpy as np
import nabla as nb


def test_simple_vjp():
    """Test basic VJP functionality."""
    print("=== Testing Simple VJP ===")
    
    def simple_func(x):
        return x * x  # f(x) = x²
    
    x = nb.array([2.0])
    print(f"Input: {x}")
    
    outputs, vjp_fn = nb.vjp(simple_func, x)
    print(f"Output: {outputs} (expected: 4.0)")
    
    cotangent = nb.array([1.0])
    gradients = vjp_fn(cotangent)
    print(f"Gradient type: {type(gradients)}")
    print(f"Gradient: {gradients}")
    
    # Extract gradient value
    if isinstance(gradients, tuple):
        grad_value = gradients[0].to_numpy()
    else:
        grad_value = gradients.to_numpy()
    
    print(f"Gradient value: {grad_value} (expected: 4.0)")
    
    expected = 4.0  # d/dx(x²) = 2x, at x=2 -> 2*2 = 4
    is_correct = np.isclose(grad_value, expected, rtol=1e-6)
    print(f"Is gradient correct: {is_correct}")
    
    return is_correct


def test_mlp_forward_vjp():
    """Test VJP on the MLP forward function."""
    print("\n=== Testing MLP Forward VJP ===")
    
    # Simple 2-layer MLP: input -> hidden(2) -> output(1)
    def simple_mlp(params):
        w1, b1, w2, b2 = params
        x = nb.array([[1.0]])  # Single input
        
        # First layer
        h = nb.matmul(x, w1) + b1
        h = nb.relu(h)
        
        # Second layer
        out = nb.matmul(h, w2) + b2
        return out
    
    # Initialize simple weights
    w1 = nb.array([[0.5, 0.3]])  # 1x2
    b1 = nb.array([[0.1, 0.2]])  # 1x2
    w2 = nb.array([[0.4], [0.6]])  # 2x1
    b2 = nb.array([[0.05]])  # 1x1
    
    params = [w1, b1, w2, b2]
    
    print("Testing forward pass...")
    outputs, vjp_fn = nb.vjp(simple_mlp, params)
    print(f"Forward output: {outputs}")
    
    print("Testing backward pass...")
    cotangent = nb.array([[1.0]])
    gradients = vjp_fn(cotangent)
    
    print(f"Gradient type: {type(gradients)}")
    print(f"Number of gradients: {len(gradients) if hasattr(gradients, '__len__') else 'N/A'}")
    
    if isinstance(gradients, tuple):
        print(f"Gradients tuple length: {len(gradients)}")
        # The gradients should be a tuple containing the gradient list
        if len(gradients) == 1 and isinstance(gradients[0], list):
            param_grads = gradients[0]
            print(f"Parameter gradients list length: {len(param_grads)}")
            for i, grad in enumerate(param_grads):
                print(f"Gradient {i} shape: {grad.shape}, values: {grad.to_numpy()}")
                # Check if gradient is all zeros (indicating no gradient flow)
                is_zero = np.allclose(grad.to_numpy(), 0.0)
                print(f"Gradient {i} is zero: {is_zero}")
        else:
            for i, grad in enumerate(gradients):
                if hasattr(grad, 'shape'):
                    print(f"Gradient {i} shape: {grad.shape}, values: {grad.to_numpy()}")
                    is_zero = np.allclose(grad.to_numpy(), 0.0)
                    print(f"Gradient {i} is zero: {is_zero}")
                else:
                    print(f"Gradient {i} type: {type(grad)}, value: {grad}")
    
    return gradients


def test_list_input_vjp():
    """Test VJP with list inputs like in the MLP training."""
    print("\n=== Testing List Input VJP ===")
    
    def list_func(inputs):
        x, params = inputs[0], inputs[1:]
        w, b = params[0], params[1]
        return [nb.matmul(x, w) + b]
    
    x = nb.array([[1.0]])
    w = nb.array([[0.5]])
    b = nb.array([[0.1]])
    
    inputs = [x, w, b]
    
    print("Testing forward pass...")
    outputs, vjp_fn = nb.vjp(list_func, inputs)
    print(f"Forward output: {outputs}")
    
    print("Testing backward pass...")
    cotangent = [nb.array([[1.0]])]
    gradients = vjp_fn(cotangent)
    
    print(f"Gradient type: {type(gradients)}")
    if isinstance(gradients, tuple):
        for i, grad in enumerate(gradients):
            print(f"Gradient {i}: {grad.to_numpy()}")
    
    return gradients


if __name__ == "__main__":
    # Test basic VJP
    test1_passed = test_simple_vjp()
    
    # Test MLP forward VJP
    test2_gradients = test_mlp_forward_vjp()
    
    # Test list input VJP (like in training)
    test3_gradients = test_list_input_vjp()
    
    print("\n=== Summary ===")
    print(f"Simple VJP test passed: {test1_passed}")
    print("If any of the above tests show all-zero gradients, that indicates the VJP refactoring broke gradient computation.")
