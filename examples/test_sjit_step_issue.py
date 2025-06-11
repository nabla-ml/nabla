#!/usr/bin/env python3
"""Test to demonstrate the sjit issue with changing step parameter."""

import numpy as np
import nabla as nb

def adam_update_with_step(param, grad, m, v, step):
    """Adam update that uses step parameter - this will break with sjit."""
    beta1, beta2, lr, eps = 0.9, 0.999, 0.001, 1e-8
    
    new_m = beta1 * m + (1.0 - beta1) * grad
    new_v = beta2 * v + (1.0 - beta2) * (grad * grad)
    
    # This line uses step in computation - problematic for sjit
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    
    new_param = param - lr * (new_m / bias_correction1) / ((new_v / bias_correction2)**0.5 + eps)
    
    return new_param, new_m, new_v

@nb.jit
def adam_update_jit(param, grad, m, v, step):
    """Regular JIT version."""
    return adam_update_with_step(param, grad, m, v, step)

@nb.sjit  
def adam_update_sjit(param, grad, m, v, step):
    """Static JIT version - this will have issues with changing step."""
    return adam_update_with_step(param, grad, m, v, step)

def test_step_issue():
    """Test the step parameter issue."""
    print("Testing step parameter issue with jit vs sjit...")
    
    # Initialize some test data
    param_np = np.array([[1.0, 2.0]], dtype=np.float32)
    grad_np = np.array([[0.1, 0.2]], dtype=np.float32)
    m_np = np.zeros_like(param_np)
    v_np = np.zeros_like(param_np)
    
    param = nb.Array.from_numpy(param_np)
    grad = nb.Array.from_numpy(grad_np)
    m = nb.Array.from_numpy(m_np)
    v = nb.Array.from_numpy(v_np)
    
    print("Initial param:", param.to_numpy())
    
    # Test regular JIT for 3 steps
    print("\n=== Regular JIT ===")
    param_jit, m_jit, v_jit = param, m, v
    for step in range(1, 4):
        param_jit, m_jit, v_jit = adam_update_jit(param_jit, grad, m_jit, v_jit, step)
        print(f"Step {step} - param: {param_jit.to_numpy()}")
    
    # Test static JIT for 3 steps  
    print("\n=== Static JIT ===")
    param_sjit, m_sjit, v_sjit = param, m, v
    for step in range(1, 4):
        try:
            param_sjit, m_sjit, v_sjit = adam_update_sjit(param_sjit, grad, m_sjit, v_sjit, step)
            print(f"Step {step} - param: {param_sjit.to_numpy()}")
        except Exception as e:
            print(f"Step {step} - ERROR: {e}")
            break

if __name__ == "__main__":
    test_step_issue()
