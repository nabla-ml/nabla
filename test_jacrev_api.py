import nabla as nb
import numpy as np


def test_basic_jacrev():
    """Test basic jacrev functionality"""
    print("=== Test Basic jacrev ===")
    
    def func(x, y):
        return x * y
    
    a = nb.arange((2, 3))
    b = nb.arange((2, 3))
    
    # Test default behavior (argnums=0)
    jac_0 = nb.jacrev(func)(a, b)
    print("jacrev with argnums=0 (default):")
    print(jac_0)
    print()
    
    # Test argnums=1
    jac_1 = nb.jacrev(func, argnums=1)(a, b)
    print("jacrev with argnums=1:")
    print(jac_1)
    print()
    
    # Test argnums=(0, 1)
    jac_both = nb.jacrev(func, argnums=(0, 1))(a, b)
    print("jacrev with argnums=(0, 1):")
    for i, grad in enumerate(jac_both):
        print(f"Gradient w.r.t. arg {i}:")
        print(grad)
    print()


def test_has_aux():
    """Test has_aux functionality"""
    print("=== Test has_aux ===")
    
    def func_with_aux(x, y):
        result = x * y
        aux_data = {"sum": x + y, "shapes": [x.shape, y.shape]}
        return result, aux_data
    
    a = nb.arange((2, 2))
    b = nb.arange((2, 2))
    
    # Test with has_aux=True
    jac, aux = nb.jacrev(func_with_aux, has_aux=True)(a, b)
    print("jacrev with has_aux=True:")
    print("Jacobian:")
    print(jac)
    print("Auxiliary data:")
    print(aux)
    print()


def test_single_arg_function():
    """Test with single argument function"""
    print("=== Test Single Argument Function ===")
    
    def single_func(x):
        return x ** 2
    
    a = nb.arange((3,))
    
    jac = nb.jacrev(single_func)(a)
    print("jacrev of x^2:")
    print(jac)
    print()


def test_multiple_outputs():
    """Test with function that has multiple outputs"""
    print("=== Test Multiple Outputs ===")
    
    def multi_output_func(x, y):
        return [x * y, x + y]
    
    a = nb.arange((2,))
    b = nb.arange((2,))
    
    jac = nb.jacrev(multi_output_func)(a, b)
    print("jacrev with multiple outputs:")
    print(jac)
    print()


def test_error_cases():
    """Test error handling"""
    print("=== Test Error Cases ===")
    
    def simple_func(x, y):
        return x * y
    
    a = nb.arange((2,))
    b = nb.arange((2,))
    
    try:
        # Test invalid argnums
        nb.jacrev(simple_func, argnums=5)(a, b)
    except ValueError as e:
        print(f"Expected error for invalid argnums: {e}")
    
    try:
        # Test has_aux with function that doesn't return tuple
        nb.jacrev(simple_func, has_aux=True)(a, b)
    except ValueError as e:
        print(f"Expected error for has_aux mismatch: {e}")
    
    print()


if __name__ == "__main__":
    test_basic_jacrev()
    test_has_aux()
    test_single_arg_function()
    test_multiple_outputs()
    test_error_cases()
    print("All tests completed!")
