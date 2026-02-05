import nabla as nb
import nabla.ops as ops
from nabla.core import Trace
import numpy as np

def test_backward_scalar():
    """Test standard scalar loss backward pass."""
    print("Testing backward() on scalar loss...")
    
    # 1. Create a leaf tensor
    x = nb.constant([1.0, 2.0, 3.0])
    x.requires_grad = True
    print(f"x.requires_grad: {x.requires_grad}")
    
    # Computation
    y = x ** 2
    z = y.sum()
    
    # Backward pass
    print("Calling z.backward()...")
    z.backward()
    
    # Check gradient: d/dx (sum(x^2)) = 2x = [2, 4, 6]
    if x.grad is not None:
        print(f"x.grad values: {x.grad.realize()}")
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(x.grad.numpy(), expected)
        print("Success: x.grad matches expected [2, 4, 6]!")
    else:
        print("FAIL: x.grad is NONE")
    
    print("Test finished!\n")

def test_backward_intermediate():
    """Test that requires_grad correctly populates intermediate gradients."""
    print("Testing backward() with intermediate requires_grad...")
    
    a = nb.constant([1.0, 2.0])
    a.requires_grad = True
    
    # b is an intermediate that also requires grad
    b = a * 2.0
    b.requires_grad = True 
    
    c = b ** 2
    d = c.sum()
    
    d.backward()
    
    if a.grad is not None:
        print(f"a.grad values: {a.grad.realize()}")
        expected_a = np.array([8.0, 16.0])
        np.testing.assert_allclose(a.grad.numpy(), expected_a)
    else:
        print("FAIL: a.grad is NONE")
        
    if b.grad is not None:
        print(f"b.grad values: {b.grad.realize()}")
        expected_b = np.array([4.0, 8.0])
        np.testing.assert_allclose(b.grad.numpy(), expected_b)
    else:
        print("FAIL: b.grad is NONE")
    
    print("Test finished!\n")

def test_backward_complex_graph():
    """Test a more complex DAG with shared nodes."""
    print("Testing backward() with complex DAG (shared nodes)...")
    
    x = nb.constant([2.0])
    x.requires_grad = True
    
    # y = x * 3
    # z = x + y = x + 3x = 4x
    # out = z * y = 4x * 3x = 12x^2
    # d(out)/dx = 24x = 24 * 2 = 48
    y = x * 3.0
    z = x + y
    out = z * y
    
    print(f"out value: {out.realize().item()}")
    out.backward()
    
    if x.grad is not None:
        print(f"x.grad value: {x.grad.realize().item()}")
        expected = 48.0
        np.testing.assert_allclose(x.grad.numpy(), expected)
        print(f"Success: x.grad matches expected {expected}!")
    else:
        print("FAIL: x.grad is NONE")
    
    print("Test finished!\n")

if __name__ == "__main__":
    try:
        test_backward_scalar()
        test_backward_intermediate()
        test_backward_complex_graph()
        print("All backward tests passed successfully!")
    except Exception as e:
        print(f"Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
