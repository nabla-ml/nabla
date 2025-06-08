#!/usr/bin/env python3

import sys

sys.path.append("/Users/tillife/Documents/CodingProjects/nabla")

import nabla as nb
from nabla.core.trafos import vmap


def debug_shape_issue():
    """Debug the specific shape issue in Test 2a."""

    # Create inputs exactly like Test 2a
    simple_input = nb.array([1.0, 2.0, 3.0])  # shape (3,)

    dict_input = {
        "x": nb.array([4.0, 5.0, 6.0]),  # shape (3,)
        "y": nb.array([7.0, 8.0, 9.0]),  # shape (3,)
    }

    list_input = [nb.array([10.0, 11.0, 12.0]), nb.array([13.0, 14.0, 15.0])]

    def mixed_func(simple_arg, dict_arg, list_arg):
        simple_part = simple_arg * 2
        dict_part = dict_arg["x"] + dict_arg["y"]
        list_part = list_arg[0] + list_arg[1]

        return simple_part + dict_part + list_part

    print("=== Debugging shape issue ===")

    # Test 1: Simple broadcast (should work fine)
    print("\n1. Simple broadcast (in_axes=0):")
    vmapped_func1 = vmap(mixed_func, in_axes=0)
    result1 = vmapped_func1(simple_input, dict_input, list_input)
    print(f"   Result shape: {result1.shape}")
    print(f"   Result values: {result1.to_numpy()}")

    # Test 2: Mixed specification (the problematic one)
    print('\n2. Mixed specification (in_axes=(0, {"x": 0, "y": 0}, 0)):')
    in_axes = (0, {"x": 0, "y": 0}, 0)
    vmapped_func2 = vmap(mixed_func, in_axes=in_axes)
    result2 = vmapped_func2(simple_input, dict_input, list_input)
    print(f"   Result shape: {result2.shape}")
    print(f"   Result values: {result2.to_numpy()}")

    # Test 3: Let's try different combinations
    print("\n3. All explicit specification:")
    in_axes3 = (0, {"x": 0, "y": 0}, [0, 0])
    vmapped_func3 = vmap(mixed_func, in_axes=in_axes3)
    result3 = vmapped_func3(simple_input, dict_input, list_input)
    print(f"   Result shape: {result3.shape}")
    print(f"   Result values: {result3.to_numpy()}")


if __name__ == "__main__":
    debug_shape_issue()
