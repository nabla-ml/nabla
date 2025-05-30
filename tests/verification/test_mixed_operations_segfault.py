#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Test to reproduce the segfault with mixed operations."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from nabla import array
from nabla.core.execution_context import global_execution_context


def test_mixed_operations_in_single_graph():
    """Test mixed add and mul operations in a single computation graph."""
    print("Testing mixed operations in single computation graph...")

    # Clear cache
    global_execution_context.clear()

    try:
        # Create tensors
        a = array([[1.0, 2.0]])
        b = array([[3.0, 4.0]])
        c = array([[5.0, 6.0]])

        print("Created input arrays")

        # Mix add and mul in single computation
        result = (a + b) * c  # This should trigger the segfault pattern
        print("Created computation graph with mixed operations")

        # Realize the computation
        result.realize()
        print("Computation realized successfully")

        # Get result
        output = result.to_numpy()
        print(f"Result: {output}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_sequential_mixed_operations():
    """Test sequential mixed operations on different graphs."""
    print("\nTesting sequential mixed operations...")

    global_execution_context.clear()

    try:
        # First: add operation
        a1 = array([[1.0, 2.0]])
        b1 = array([[3.0, 4.0]])
        result1 = a1 + b1
        result1.realize()
        print("✓ First add operation succeeded")

        # Second: mul operation
        a2 = array([[1.0, 2.0]])
        b2 = array([[3.0, 4.0]])
        result2 = a2 * b2
        result2.realize()
        print("✓ Second mul operation succeeded")

        # Third: mixed operation
        a3 = array([[1.0, 2.0]])
        b3 = array([[3.0, 4.0]])
        c3 = array([[5.0, 6.0]])
        result3 = (a3 + b3) * c3
        result3.realize()
        print("✓ Mixed operation succeeded")

        return True

    except Exception as e:
        print(f"Error in sequential test: {e}")
        return False


def test_multiple_mixed_graphs():
    """Test multiple mixed operation graphs to stress test the cache."""
    print("\nTesting multiple mixed operation graphs...")

    global_execution_context.clear()

    try:
        for i in range(5):
            print(f"  Iteration {i + 1}...")

            # Create new arrays each time
            a = array([[1.0 + i, 2.0 + i]])
            b = array([[3.0 + i, 4.0 + i]])
            c = array([[5.0 + i, 6.0 + i]])

            # Mixed operation
            result = (a + b) * c
            result.realize()
            _ = result.to_numpy()

            print(f"    ✓ Iteration {i + 1} completed")
            print(f"    Cache size: {global_execution_context.size()}")

        return True

    except Exception as e:
        print(f"Error in multiple graphs test: {e}")
        return False


def test_complex_mixed_computation():
    """Test a more complex mixed computation."""
    print("\nTesting complex mixed computation...")

    global_execution_context.clear()

    try:
        # Create inputs
        a = array([[1.0, 2.0]])
        b = array([[3.0, 4.0]])
        c = array([[5.0, 6.0]])
        d = array([[7.0, 8.0]])

        # Complex mixed computation: ((a + b) * c) + (a * d)
        intermediate1 = a + b
        intermediate2 = intermediate1 * c
        intermediate3 = a * d
        result = intermediate2 + intermediate3

        result.realize()
        output = result.to_numpy()
        print(f"Complex computation result: {output}")

        return True

    except Exception as e:
        print(f"Error in complex computation: {e}")
        return False


if __name__ == "__main__":
    print("=== Mixed Operations Segfault Test ===")

    tests = [
        test_mixed_operations_in_single_graph,
        test_sequential_mixed_operations,
        test_multiple_mixed_graphs,
        test_complex_mixed_computation,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print(f"=== Results: {passed}/{len(tests)} tests passed ===")

    if passed == len(tests):
        print("All tests passed - no segfault reproduced")
    else:
        print("Some tests failed - segfault may have been reproduced")
