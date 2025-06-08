#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jax import vmap

def test_jax_vmap_with_dict_input():
    """Test JAX vmap with dictionary input and simple axis specification."""
    
    # Create test data
    x1 = jnp.array([1.0, 2.0, 3.0])  # shape (3,)
    y1 = jnp.array([4.0, 5.0, 6.0])  # shape (3,)
    
    # Create batched dictionary input
    dict_input = {"x": x1, "y": y1}
    
    def dict_func(inputs):
        return inputs["x"] + inputs["y"]
    
    print("Testing JAX vmap with dict input and simple axis spec...")
    print(f"x shape: {dict_input['x'].shape}")
    print(f"y shape: {dict_input['y'].shape}")
    
    try:
        # This should work: vmap along axis 0 for all elements in the dict
        vmapped_func = vmap(dict_func, in_axes=0)
        result = vmapped_func(dict_input)
        print(f"SUCCESS: Result shape: {result.shape}")
        print(f"Result values: {result}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_jax_vmap_with_tuple_input():
    """Test JAX vmap with tuple input and simple axis specification."""
    
    # Create test data
    x1 = jnp.array([1.0, 2.0, 3.0])  # shape (3,)
    y1 = jnp.array([4.0, 5.0, 6.0])  # shape (3,)
    
    # Create tuple input
    tuple_input = (x1, y1)
    
    def tuple_func(inputs):
        return inputs[0] + inputs[1]
    
    print("\nTesting JAX vmap with tuple input and simple axis spec...")
    print(f"x shape: {tuple_input[0].shape}")
    print(f"y shape: {tuple_input[1].shape}")
    
    try:
        # This should work: vmap along axis 0 for all elements in the tuple
        vmapped_func = vmap(tuple_func, in_axes=0)
        result = vmapped_func(tuple_input)
        print(f"SUCCESS: Result shape: {result.shape}")
        print(f"Result values: {result}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_jax_vmap_with_multiple_dict_args():
    """Test JAX vmap with multiple dictionary arguments and simple axis specification."""
    
    # Create test data
    x1 = jnp.array([1.0, 2.0, 3.0])  # shape (3,)
    y1 = jnp.array([4.0, 5.0, 6.0])  # shape (3,)
    
    x2 = jnp.array([7.0, 8.0, 9.0])  # shape (3,)
    y2 = jnp.array([10.0, 11.0, 12.0])  # shape (3,)
    
    dict1 = {"x": x1, "y": y1}
    dict2 = {"x": x2, "y": y2}
    
    def multi_dict_func(d1, d2):
        return d1["x"] + d1["y"] + d2["x"] + d2["y"]
    
    print("\nTesting JAX vmap with multiple dict args and simple axis spec...")
    
    try:
        # This should work: vmap along axis 0 for all elements in both dicts
        vmapped_func = vmap(multi_dict_func, in_axes=0)
        result = vmapped_func(dict1, dict2)
        print(f"SUCCESS: Result shape: {result.shape}")
        print(f"Result values: {result}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_jax_complex_nested_broadcast():
    """Test JAX vmap with complex nested structures and simple axis broadcasting."""
    
    print("\n" + "=" * 60)
    print("Testing JAX complex nested structures with axis broadcasting")
    print("=" * 60)
    
    # Create complex nested inputs
    # Input 1: List of arrays
    list_input = [
        jnp.array([1.0, 2.0, 3.0]),   # shape (3,)
        jnp.array([4.0, 5.0, 6.0]),   # shape (3,)
        jnp.array([7.0, 8.0, 9.0])    # shape (3,)
    ]
    
    # Input 2: Dictionary of arrays
    dict_input = {
        "a": jnp.array([10.0, 11.0, 12.0]),   # shape (3,)
        "b": jnp.array([13.0, 14.0, 15.0]),   # shape (3,)
        "c": jnp.array([16.0, 17.0, 18.0])    # shape (3,)
    }
    
    # Input 3: Tuple of tuples of arrays
    nested_tuple_input = (
        (jnp.array([19.0, 20.0, 21.0]), jnp.array([22.0, 23.0, 24.0])),  # shape (3,), (3,)
        (jnp.array([25.0, 26.0, 27.0]), jnp.array([28.0, 29.0, 30.0]))   # shape (3,), (3,)
    )
    
    def complex_func(list_arg, dict_arg, tuple_arg):
        """Function that operates on complex nested structures."""
        # Sum all elements from list
        list_sum = list_arg[0] + list_arg[1] + list_arg[2]
        
        # Sum all elements from dict
        dict_sum = dict_arg["a"] + dict_arg["b"] + dict_arg["c"]
        
        # Sum all elements from nested tuple
        tuple_sum = tuple_arg[0][0] + tuple_arg[0][1] + tuple_arg[1][0] + tuple_arg[1][1]
        
        return list_sum + dict_sum + tuple_sum
    
    print(f"List input structure: {[arr.shape for arr in list_input]}")
    print(f"Dict input structure: {[(k, v.shape) for k, v in dict_input.items()]}")
    print(f"Tuple input structure: nested tuples with shapes")
    
    try:
        print("\nTest 1: Simple axis broadcasting (in_axes=0)")
        # This should broadcast axis=0 to ALL arrays in ALL nested structures
        vmapped_func = vmap(complex_func, in_axes=0)
        result = vmapped_func(list_input, dict_input, nested_tuple_input)
        print(f"SUCCESS: Result shape: {result.shape}")
        print(f"Result values: {result}")
        return True
        
    except Exception as e:
        print(f"ERROR in Test 1: {e}")
        return False

def test_jax_mixed_axis_specifications():
    """Test JAX vmap with mixed axis specifications - some explicit, some broadcast."""
    
    print("\n" + "=" * 60)
    print("Testing JAX mixed axis specifications")
    print("=" * 60)
    
    # Create inputs with different structures
    # Input 1: Simple array
    simple_input = jnp.array([1.0, 2.0, 3.0])  # shape (3,)
    
    # Input 2: Dictionary with explicit axis specification
    dict_input = {
        "x": jnp.array([4.0, 5.0, 6.0]),   # shape (3,)
        "y": jnp.array([7.0, 8.0, 9.0])    # shape (3,)
    }
    
    # Input 3: Nested list
    list_input = [
        jnp.array([10.0, 11.0, 12.0]),
        jnp.array([13.0, 14.0, 15.0])
    ]
    
    def mixed_func(simple_arg, dict_arg, list_arg):
        """Function operating on mixed structures."""
        simple_part = simple_arg * 2
        dict_part = dict_arg["x"] + dict_arg["y"]  
        list_part = list_arg[0] + list_arg[1]
        
        return simple_part + dict_part + list_part
    
    try:
        print("\nTest 2a: Mixed specification - explicit dict, broadcast others")
        # Specify explicit structure for dict, let others broadcast
        in_axes = (
            0,  # simple_arg: axis 0
            {"x": 0, "y": 0},  # dict_arg: explicit specification
            0   # list_arg: broadcast to all elements
        )
        
        vmapped_func = vmap(mixed_func, in_axes=in_axes)
        result = vmapped_func(simple_input, dict_input, list_input)
        print(f"SUCCESS: Result shape: {result.shape}")
        print(f"Result values: {result}")
        return True
        
    except Exception as e:
        print(f"ERROR in Test 2a: {e}")
        return False

def test_jax_deeply_nested_structures():
    """Test JAX with very deeply nested structures."""
    
    print("\n" + "=" * 60)
    print("Testing JAX deeply nested structures")
    print("=" * 60)
    
    # Create deeply nested structure
    deep_nested = {
        "level1": {
            "level2": [
                (jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0])),
                (jnp.array([7.0, 8.0, 9.0]), jnp.array([10.0, 11.0, 12.0]))
            ],
            "simple": jnp.array([13.0, 14.0, 15.0])
        },
        "other": jnp.array([16.0, 17.0, 18.0])
    }
    
    def deep_func(nested_arg):
        """Function operating on deeply nested structure."""
        level2_sum = (nested_arg["level1"]["level2"][0][0] + 
                     nested_arg["level1"]["level2"][0][1] +
                     nested_arg["level1"]["level2"][1][0] + 
                     nested_arg["level1"]["level2"][1][1])
        
        simple_part = nested_arg["level1"]["simple"]
        other_part = nested_arg["other"]
        
        return level2_sum + simple_part + other_part
    
    try:
        print("\nTest 3: Deeply nested with simple axis broadcast")
        # Should broadcast axis=0 to ALL arrays in the deep structure
        vmapped_func = vmap(deep_func, in_axes=0)
        result = vmapped_func(deep_nested)
        print(f"SUCCESS: Result shape: {result.shape}")
        print(f"Result values: {result}")
        return True
        
    except Exception as e:
        print(f"ERROR in Test 3: {e}")
        return False

if __name__ == "__main__":
    print("Testing JAX vmap behavior with nested structures")
    print("=" * 60)
    
    test1 = test_jax_vmap_with_dict_input()
    test2 = test_jax_vmap_with_tuple_input()
    test3 = test_jax_vmap_with_multiple_dict_args()
    test4 = test_jax_complex_nested_broadcast()
    test5 = test_jax_mixed_axis_specifications() 
    test6 = test_jax_deeply_nested_structures()
    
    print("\n" + "=" * 60)
    print("JAX FINAL RESULTS")
    print("=" * 60)
    
    if all([test1, test2, test3, test4, test5, test6]):
        print("✅ ALL JAX TESTS PASSED! JAX vmap works correctly with complex nested structures!")
    else:
        print("❌ Some JAX tests failed!")
        print(f"Test 1 (Dict input): {'✅' if test1 else '❌'}")
        print(f"Test 2 (Tuple input): {'✅' if test2 else '❌'}")
        print(f"Test 3 (Multiple dict args): {'✅' if test3 else '❌'}")
        print(f"Test 4 (Complex nested broadcast): {'✅' if test4 else '❌'}")
        print(f"Test 5 (Mixed specifications): {'✅' if test5 else '❌'}")
        print(f"Test 6 (Deeply nested structures): {'✅' if test6 else '❌'}")
