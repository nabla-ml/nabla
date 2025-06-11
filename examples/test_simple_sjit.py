#!/usr/bin/env python3
"""Debug the sjit signature issue."""

import numpy as np
import nabla as nb

def simple_function(param, step):
    """Simple function that uses a non-Array parameter."""
    return param * step

@nb.sjit
def simple_sjit(param, step):
    """Static JIT version."""
    return simple_function(param, step)

def test_simple_sjit():
    """Test simple sjit with changing non-Array parameter."""
    print("Testing simple sjit with changing step...")
    
    param = nb.Array.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    
    print("Initial param:", param.to_numpy())
    
    for step in range(1, 4):
        try:
            result = simple_sjit(param, step)
            print(f"Step {step} - result: {result.to_numpy()}")
        except Exception as e:
            print(f"Step {step} - ERROR: {e}")
            break

if __name__ == "__main__":
    test_simple_sjit()
