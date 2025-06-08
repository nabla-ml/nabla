#!/usr/bin/env python3

import sys

sys.path.append("/Users/tillife/Documents/CodingProjects/nabla")

import nabla as nb
from nabla.core.trafos import vmap


def analyze_mixed_func():
    """Analyze what's happening in the mixed function specifically."""

    # Exact inputs from the problematic test
    simple_input = nb.array([1.0, 2.0, 3.0])  # shape (3,)

    dict_input = {
        "x": nb.array([4.0, 5.0, 6.0]),  # shape (3,)
        "y": nb.array([7.0, 8.0, 9.0]),  # shape (3,)
    }

    list_input = [nb.array([10.0, 11.0, 12.0]), nb.array([13.0, 14.0, 15.0])]

    def mixed_func_debug(simple_arg, dict_arg, list_arg):
        print(f"   simple_arg shape: {simple_arg.shape}")
        print(f"   dict_arg['x'] shape: {dict_arg['x'].shape}")
        print(f"   dict_arg['y'] shape: {dict_arg['y'].shape}")
        print(f"   list_arg[0] shape: {list_arg[0].shape}")
        print(f"   list_arg[1] shape: {list_arg[1].shape}")

        simple_part = simple_arg * 2
        print(f"   simple_part shape: {simple_part.shape}")

        dict_part = dict_arg["x"] + dict_arg["y"]
        print(f"   dict_part shape: {dict_part.shape}")

        list_part = list_arg[0] + list_arg[1]
        print(f"   list_part shape: {list_part.shape}")

        result = simple_part + dict_part + list_part
        print(f"   final result shape: {result.shape}")
        return result

    print("=== Analyzing mixed function ===")

    # Test with simple broadcasting
    print("\n1. Direct call:")
    direct_result = mixed_func_debug(simple_input, dict_input, list_input)
    print(f"Direct result shape: {direct_result.shape}")

    print("\n2. Vmap with simple broadcast:")
    vmapped_func = vmap(mixed_func_debug, in_axes=0)
    vmap_result = vmapped_func(simple_input, dict_input, list_input)
    print(f"Vmap result shape: {vmap_result.shape}")


if __name__ == "__main__":
    analyze_mixed_func()
