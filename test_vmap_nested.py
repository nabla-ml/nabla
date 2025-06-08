#!/usr/bin/env python3

import sys
sys.path.append('/Users/tillife/Documents/CodingProjects/nabla')

import nabla as nb
from nabla.core.trafos import vmap

def test_vmap_with_dict_input():
    """Test vmap with dictionary input and simple axis specification."""
    
    # Create test data
    x1 = nb.array([1.0, 2.0, 3.0])  # shape (3,)
    y1 = nb.array([4.0, 5.0, 6.0])  # shape (3,)
    
    # Create batched dictionary input
    dict_input = {"x": x1, "y": y1}
    
    def dict_func(inputs):
        return inputs["x"] + inputs["y"]
    
    print("Testing vmap with dict input and simple axis spec...")
    print(f"x shape: {dict_input['x'].shape}")
    print(f"y shape: {dict_input['y'].shape}")
    
    try:
        # This should work: vmap along axis 0 for all elements in the dict
        vmapped_func = vmap(dict_func, in_axes=0)
        result = vmapped_func(dict_input)
        print(f"SUCCESS: Result shape: {result.shape}")
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    return True

def test_vmap_with_tuple_input():
    """Test vmap with tuple input and simple axis specification."""
    
    # Create test data
    x1 = nb.array([1.0, 2.0, 3.0])  # shape (3,)
    y1 = nb.array([4.0, 5.0, 6.0])  # shape (3,)
    
    # Create tuple input
    tuple_input = (x1, y1)
    
    def tuple_func(inputs):
        return inputs[0] + inputs[1]
    
    print("\nTesting vmap with tuple input and simple axis spec...")
    print(f"x shape: {tuple_input[0].shape}")
    print(f"y shape: {tuple_input[1].shape}")
    
    try:
        # This should work: vmap along axis 0 for all elements in the tuple
        vmapped_func = vmap(tuple_func, in_axes=0)
        result = vmapped_func(tuple_input)
        print(f"SUCCESS: Result shape: {result.shape}")
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    return True

def test_vmap_with_multiple_dict_args():
    """Test vmap with multiple dictionary arguments and simple axis specification."""
    
    # Create test data
    x1 = nb.array([1.0, 2.0, 3.0])  # shape (3,)
    y1 = nb.array([4.0, 5.0, 6.0])  # shape (3,)
    
    x2 = nb.array([7.0, 8.0, 9.0])  # shape (3,)
    y2 = nb.array([10.0, 11.0, 12.0])  # shape (3,)
    
    dict1 = {"x": x1, "y": y1}
    dict2 = {"x": x2, "y": y2}
    
    def multi_dict_func(d1, d2):
        return d1["x"] + d1["y"] + d2["x"] + d2["y"]
    
    print("\nTesting vmap with multiple dict args and simple axis spec...")
    
    try:
        # This should work: vmap along axis 0 for all elements in both dicts
        vmapped_func = vmap(multi_dict_func, in_axes=0)
        result = vmapped_func(dict1, dict2)
        print(f"SUCCESS: Result shape: {result.shape}")
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    return True



def test_complex_nested_broadcast():
    """Test vmap with complex nested structures and simple axis broadcasting."""
    
    print("=" * 60)
    print("Testing complex nested structures with axis broadcasting")
    print("=" * 60)
    
    # Create complex nested inputs
    # Input 1: List of arrays
    list_input = [
        nb.array([1.0, 2.0, 3.0]),   # shape (3,)
        nb.array([4.0, 5.0, 6.0]),   # shape (3,)
        nb.array([7.0, 8.0, 9.0])    # shape (3,)
    ]
    
    # Input 2: Dictionary of arrays
    dict_input = {
        "a": nb.array([10.0, 11.0, 12.0]),   # shape (3,)
        "b": nb.array([13.0, 14.0, 15.0]),   # shape (3,)
        "c": nb.array([16.0, 17.0, 18.0])    # shape (3,)
    }
    
    # Input 3: Tuple of tuples of arrays
    nested_tuple_input = (
        (nb.array([19.0, 20.0, 21.0]), nb.array([22.0, 23.0, 24.0])),  # shape (3,), (3,)
        (nb.array([25.0, 26.0, 27.0]), nb.array([28.0, 29.0, 30.0]))   # shape (3,), (3,)
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
        print(f"Result values: {result.to_numpy()}")
        
    except Exception as e:
        print(f"ERROR in Test 1: {e}")
        return False
    
    return True

def test_mixed_axis_specifications():
    """Test vmap with mixed axis specifications - some explicit, some broadcast."""
    
    print("\n" + "=" * 60)
    print("Testing mixed axis specifications")
    print("=" * 60)
    
    # Create inputs with different structures
    # Input 1: Simple array
    simple_input = nb.array([1.0, 2.0, 3.0])  # shape (3,)
    
    # Input 2: Dictionary with explicit axis specification
    dict_input = {
        "x": nb.array([4.0, 5.0, 6.0]),   # shape (3,)
        "y": nb.array([7.0, 8.0, 9.0])    # shape (3,)
    }
    
    # Input 3: Nested list
    list_input = [
        nb.array([10.0, 11.0, 12.0]),
        nb.array([13.0, 14.0, 15.0])
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
        print(f"Result values: {result.to_numpy()}")
        
    except Exception as e:
        print(f"ERROR in Test 2a: {e}")
        return False
        
    try:
        print("\nTest 2b: Mixed specification with None (broadcasting)")
        # Test with some None values for broadcasting
        in_axes = (
            0,    # simple_arg: axis 0
            None, # dict_arg: broadcast (no vectorization)
            0     # list_arg: vectorize along axis 0
        )
        
        # Need different data for this test - make dict not batched
        dict_scalar = {
            "x": nb.array([100.0]),  # shape (1,) - will be broadcast
            "y": nb.array([200.0])   # shape (1,) - will be broadcast  
        }
        
        vmapped_func = vmap(mixed_func, in_axes=in_axes)
        result = vmapped_func(simple_input, dict_scalar, list_input)
        print(f"SUCCESS: Result shape: {result.shape}")
        print(f"Result values: {result.to_numpy()}")
        
    except Exception as e:
        print(f"ERROR in Test 2b: {e}")
        return False
    
    return True

def test_deeply_nested_structures():
    """Test with very deeply nested structures."""
    
    print("\n" + "=" * 60)
    print("Testing deeply nested structures")
    print("=" * 60)
    
    # Create deeply nested structure
    deep_nested = {
        "level1": {
            "level2": [
                (nb.array([1.0, 2.0, 3.0]), nb.array([4.0, 5.0, 6.0])),
                (nb.array([7.0, 8.0, 9.0]), nb.array([10.0, 11.0, 12.0]))
            ],
            "simple": nb.array([13.0, 14.0, 15.0])
        },
        "other": nb.array([16.0, 17.0, 18.0])
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
        print(f"Result values: {result.to_numpy()}")
        
    except Exception as e:
        print(f"ERROR in Test 3: {e}")
        return False
    
    return True

def test_axis_validation():
    """Test that axis validation works correctly."""
    
    print("\n" + "=" * 60)
    print("Testing axis validation")
    print("=" * 60)
    
    # Create test data
    test_dict = {
        "a": nb.array([1.0, 2.0, 3.0]),
        "b": nb.array([4.0, 5.0, 6.0])
    }
    
    def simple_func(d):
        return d["a"] + d["b"]
    
    try:
        print("\nTest 4a: Valid axis specification")
        vmapped_func = vmap(simple_func, in_axes=0)
        result = vmapped_func(test_dict)
        print(f"SUCCESS: Valid specification worked")
        
    except Exception as e:
        print(f"ERROR in Test 4a: {e}")
        return False
    
    try:
        print("\nTest 4b: Invalid axis - out of bounds")
        # This should fail - axis 5 doesn't exist for shape (3,)
        vmapped_func = vmap(simple_func, in_axes=5)
        result = vmapped_func(test_dict)
        print(f"ERROR: Should have failed with out of bounds axis")
        return False
        
    except Exception as e:
        print(f"SUCCESS: Correctly caught invalid axis: {e}")
    
    return True

if __name__ == "__main__":
    test1 = test_vmap_with_dict_input()
    test2 = test_vmap_with_tuple_input()
    test3 = test_vmap_with_multiple_dict_args()
    
    test4 = test_complex_nested_broadcast()
    test5 = test_mixed_axis_specifications() 
    test6 = test_deeply_nested_structures()
    test7 = test_axis_validation()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    if all([test1, test2, test3, test4, test5, test6, test7]):
        print("✅ ALL TESTS PASSED! Vmap works correctly with complex nested structures!")
    else:
        print("❌ Some tests failed!")
        print(f"Test 1 (Complex nested broadcast): {'✅' if test1 else '❌'}")
        print(f"Test 2 (Mixed specifications): {'✅' if test2 else '❌'}")
        print(f"Test 3 (Deeply nested): {'✅' if test3 else '❌'}")
        print(f"Test 4 (Axis validation): {'✅' if test4 else '❌'}")
        print(f"Test 5 (Complex nested broadcast): {'✅' if test5 else '❌'}")
        print(f"Test 6 (Mixed axis specifications): {'✅' if test6 else '❌'}")
        print(f"Test 7 (Deeply nested structures): {'✅' if test7 else '❌'}")