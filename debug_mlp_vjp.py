#!/usr/bin/env python3
"""Test the specific VJP pattern used in MLP training."""

import numpy as np
import nabla as nb


def simple_loss_func(inputs):
    """Mimics mlp_forward_and_loss_leaky structure."""
    x, targets = inputs[0], inputs[1]
    params = inputs[2:]  # w1, b1, w2, b2
    
    # Simple forward pass
    w1, b1, w2, b2 = params[0], params[1], params[2], params[3]
    h = nb.matmul(x, w1) + b1
    h = nb.relu(h)
    out = nb.matmul(h, w2) + b2
    
    # Simple MSE loss
    diff = out - targets
    loss = nb.sum(diff * diff) / nb.array([np.float32(out.shape[0])])
    return [loss]


def test_mlp_vjp_pattern():
    """Test the exact VJP pattern used in MLP training."""
    print("=== Testing MLP VJP Pattern ===")
    
    # Create inputs like in MLP training
    x = nb.array([[1.0]])
    targets = nb.array([[0.5]])
    
    # Simple parameters
    w1 = nb.array([[0.5, 0.3]])
    b1 = nb.array([[0.1, 0.2]])
    w2 = nb.array([[0.4], [0.6]])
    b2 = nb.array([[0.05]])
    
    params = [w1, b1, w2, b2]
    all_inputs = [x, targets] + params
    
    print(f"Number of inputs: {len(all_inputs)}")
    print(f"Input types: {[type(inp) for inp in all_inputs]}")
    
    # Test VJP like in training
    loss_values, vjp_fn = nb.vjp(simple_loss_func, all_inputs)
    print(f"Loss: {loss_values}")
    
    # Backward pass
    cotangent = [nb.array([np.float32(1.0)])]
    gradients = vjp_fn(cotangent)
    
    print(f"Gradients type: {type(gradients)}")
    print(f"Gradients length: {len(gradients) if hasattr(gradients, '__len__') else 'N/A'}")
    
    if isinstance(gradients, tuple):
        print(f"Gradients is tuple with {len(gradients)} elements")
        first_element = gradients[0]
        print(f"First element type: {type(first_element)}")
        print(f"First element length: {len(first_element) if hasattr(first_element, '__len__') else 'N/A'}")
        
        if isinstance(first_element, list):
            print("Gradients structure: tuple containing list of gradients")
            print(f"Total gradients in list: {len(first_element)}")
            
            # Test OLD way (would fail)
            try:
                param_gradients_old = gradients[2:]  # This is what the original code does
                print("OLD WAY would extract gradients 2 onwards from tuple:")
                print(f"param_gradients_old length: {len(param_gradients_old)}")
                print("This will be WRONG - it's slicing the tuple, not the gradient list!")
            except Exception as e:
                print(f"OLD WAY failed: {e}")
            
            # Test NEW way (correct)
            all_gradients = first_element  # Extract the list from the tuple
            param_gradients_new = all_gradients[2:]  # Skip x and targets gradients
            print("NEW WAY extracts gradients correctly:")
            print(f"param_gradients_new length: {len(param_gradients_new)}")
            print("Parameter gradients:")
            for i, grad in enumerate(param_gradients_new):
                print(f"  Param {i}: shape {grad.shape}, mean {grad.to_numpy().mean():.6f}")
                is_zero = np.allclose(grad.to_numpy(), 0.0)
                print(f"  Is zero: {is_zero}")
    
    return gradients


if __name__ == "__main__":
    test_mlp_vjp_pattern()
