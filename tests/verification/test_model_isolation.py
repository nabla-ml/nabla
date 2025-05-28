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

"""Test to check if there's a model sharing or isolation issue."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from nabla import array
from nabla.core.execution_context import global_execution_context


def test_sequential_operations():
    """Test operations sequentially to see if there's cross-contamination."""
    print("Testing sequential operations...")

    # Clear the cache first
    global_execution_context.clear()
    print(f"Cache cleared. Size: {global_execution_context.size()}")

    # Test 1: Only mul operations
    print("\n1. Testing only mul operations...")
    try:
        x1 = array([[1.0, 2.0]])
        y1 = array([[3.0, 4.0]])
        result1 = x1 * y1
        result1.realize()
        _ = result1.get_numpy()
        print("✓ Mul-only operation succeeded")
        print(f"Cache size after mul: {global_execution_context.size()}")
    except Exception as e:
        print(f"✗ Mul-only operation failed: {e}")
        return

    # Test 2: Only add operations (new tensors)
    print("\n2. Testing only add operations...")
    try:
        x2 = array([[5.0, 6.0]])
        y2 = array([[7.0, 8.0]])
        result2 = x2 + y2
        result2.realize()
        _ = result2.get_numpy()
        print("✓ Add-only operation succeeded")
        print(f"Cache size after add: {global_execution_context.size()}")
    except Exception as e:
        print(f"✗ Add-only operation failed: {e}")
        return

    # Test 3: Mixed operations (this should trigger the segfault)
    print("\n3. Testing mixed operations...")
    try:
        x3 = array([[1.0, 2.0]])
        y3 = array([[3.0, 4.0]])
        z3 = array([[5.0, 6.0]])

        # Mix add and mul in the same computation
        result3 = (x3 + y3) * z3
        result3.realize()
        _ = result3.get_numpy()
        print("✓ Mixed operation succeeded")
        print(f"Cache size after mixed: {global_execution_context.size()}")
    except Exception as e:
        print(f"✗ Mixed operation failed: {e}")
        print(f"Cache size after failed mixed: {global_execution_context.size()}")


def test_cache_keys():
    """Test if cache keys are computed correctly for different operation types."""
    print("\nTesting cache key computation...")

    from nabla.core.graph_execution import GraphTracer

    # Clear cache
    global_execution_context.clear()

    # Create test operations
    x = array([[1.0, 2.0]])
    y = array([[3.0, 4.0]])
    z = array([[5.0, 6.0]])

    # Operation 1: mul only
    mul_result = x * y
    inputs1, trace1, key1 = GraphTracer.get_trace([mul_result])
    print(f"Mul-only cache key: {key1}")
    print(f"Mul-only trace length: {len(trace1)}")

    # Operation 2: add only
    add_result = x + y
    inputs2, trace2, key2 = GraphTracer.get_trace([add_result])
    print(f"Add-only cache key: {key2}")
    print(f"Add-only trace length: {len(trace2)}")

    # Operation 3: mixed
    mixed_result = (x + y) * z
    inputs3, trace3, key3 = GraphTracer.get_trace([mixed_result])
    print(f"Mixed cache key: {key3}")
    print(f"Mixed trace length: {len(trace3)}")

    # Check if keys are unique
    keys = [key1, key2, key3]
    unique_keys = set(keys)
    print(f"All keys unique: {len(keys) == len(unique_keys)}")

    return keys


def test_model_reuse():
    """Test if cached models can be safely reused."""
    print("\nTesting model reuse...")

    global_execution_context.clear()

    # Create same computation multiple times
    for i in range(3):
        print(f"\nIteration {i + 1}:")
        try:
            x = array([[1.0, 2.0]])
            y = array([[3.0, 4.0]])

            # Same computation should reuse cached model
            result = x * y
            result.realize()
            _ = result.get_numpy()
            print(f"✓ Iteration {i + 1} succeeded")
            print(f"Cache size: {global_execution_context.size()}")
        except Exception as e:
            print(f"✗ Iteration {i + 1} failed: {e}")
            break


if __name__ == "__main__":
    print("=== Model Isolation Test ===")

    # Test cache keys first
    keys = test_cache_keys()

    # Test model reuse
    test_model_reuse()

    # Test sequential operations
    test_sequential_operations()

    print("\n=== Test Complete ===")
