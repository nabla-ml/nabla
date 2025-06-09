#!/usr/bin/env python3
"""Test script to verify tree_map behavior with mixed types."""

import sys
from pathlib import Path

# Add the nabla package to the path
sys.path.insert(0, str(Path(__file__).parent / "."))

from nabla.core.trafos import make_traced_pytree, tree_flatten
from nabla.ops.creation import array


def test_tree_flatten_behavior():
    """Test that tree_flatten only extracts Arrays."""
    print("=== Testing tree_flatten behavior ===")

    # Create a mixed pytree
    arr1 = array([1, 2, 3])
    arr2 = array([4, 5, 6])

    mixed_tree = {
        "arrays": [arr1, arr2],
        "scalars": [1, 2.5, "hello"],
        "nested": {"more_arrays": arr1, "more_scalars": 42},
    }

    print("Original tree structure:")
    print(f"  arrays: {type(mixed_tree['arrays'])}")
    print(f"  scalars: {type(mixed_tree['scalars'])}")
    print(f"  nested: {type(mixed_tree['nested'])}")

    leaves, structure = tree_flatten(mixed_tree)

    print(f"\nExtracted leaves (should only be Arrays): {len(leaves)}")
    for i, leaf in enumerate(leaves):
        print(f"  Leaf {i}: {type(leaf)} - {leaf}")

    print("\nStructure (should preserve non-Arrays):")
    print(f"  Type: {type(structure)}")
    print(f"  Content: {structure}")


def test_tree_map_with_tracing():
    """Test that tree_map only applies function to Arrays."""
    print("\n=== Testing tree_map with tracing ===")

    # Create a mixed pytree
    arr1 = array([1, 2, 3])
    arr2 = array([4, 5, 6])

    mixed_tree = {
        "array": arr1,
        "scalar": 42,
        "float": 3.14,
        "string": "hello",
        "list_mixed": [arr2, 99, "world"],
    }

    print("Before make_traced_pytree:")
    print(f"  arr1.traced (if exists): {getattr(arr1, 'traced', 'NO ATTR')}")
    print(f"  arr2.traced (if exists): {getattr(arr2, 'traced', 'NO ATTR')}")
    print(f"  Scalar: {mixed_tree['scalar']}")
    print(f"  Float: {mixed_tree['float']}")

    # Apply tracing
    try:
        traced_tree = make_traced_pytree(mixed_tree)
        print("\nAfter make_traced_pytree:")
        print("  Success! No errors with non-Array objects")

        # Check that Arrays got traced
        traced_arrays, _ = tree_flatten(traced_tree)
        print(f"  Number of traced arrays: {len(traced_arrays)}")
        for i, arr in enumerate(traced_arrays):
            print(f"    Array {i}.traced: {getattr(arr, 'traced', 'NO ATTR')}")

        # Check that non-Arrays are preserved
        print(f"  Scalar preserved: {traced_tree['scalar']}")
        print(f"  Float preserved: {traced_tree['float']}")
        print(f"  String preserved: {traced_tree['string']}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_tree_flatten_behavior()
    test_tree_map_with_tracing()
