#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or beautiful, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Test if the issue is with random seed or randn operation itself."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from nabla import add, arange, array, mul, randn
from nabla.core.execution_context import global_execution_context


def test_randn_without_mixed_ops():
    """Test randn without mixing operations in same graph."""
    print("Testing randn without mixing operations...")

    try:
        # Test 1: Only randn + add
        global_execution_context.clear()
        x = randn((2, 2), seed=42)
        y = array([[1.0, 1.0], [1.0, 1.0]])
        z = add(x, y)
        z.realize()
        print("✓ randn + regular array with add works")

        # Test 2: Only randn + mul (fresh context)
        global_execution_context.clear()
        x = randn((2, 2), seed=42)
        y = array([[1.0, 1.0], [1.0, 1.0]])
        z = mul(x, y)
        z.realize()
        print("✓ randn + regular array with mul works")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_mixed_arange_randn():
    """Test the exact combination that fails: arange + randn with mixed ops."""
    print("Testing arange + randn with mixed operations...")

    global_execution_context.clear()

    try:
        # Reproduce the exact failing test conditions
        x = arange((2, 2))  # This creates deterministic values [0,1,2,3]
        y = randn((2, 2), seed=42)  # This creates random values

        print("  Arrays created")

        # First operation: add
        z1 = add(x, y)
        z1.realize()
        print("  Add operation completed")

        # Second operation: mul (this is where it crashes)
        z2 = mul(x, y)
        z2.realize()
        print("  Mul operation completed")

        print("✓ arange + randn with mixed operations works")
        return True

    except Exception as e:
        print(f"✗ arange + randn with mixed operations failed: {e}")
        return False


def test_cache_key_hypothesis():
    """Test if the issue is with cache key computation for randn operations."""
    print("Testing cache key computation for randn operations...")

    try:
        from nabla.core.graph_execution import GraphTracer

        # Test cache keys for different combinations
        global_execution_context.clear()

        # Combination 1: arange + arange
        x1 = arange((2, 2))
        y1 = arange((2, 2))
        add_result1 = add(x1, y1)
        mul_result1 = mul(x1, y1)

        _, _, key1_add = GraphTracer.get_trace([add_result1])
        _, _, key1_mul = GraphTracer.get_trace([mul_result1])
        print(f"  arange+arange: add key={key1_add}, mul key={key1_mul}")

        # Combination 2: arange + randn (the failing case)
        x2 = arange((2, 2))
        y2 = randn((2, 2), seed=42)
        add_result2 = add(x2, y2)
        mul_result2 = mul(x2, y2)

        _, _, key2_add = GraphTracer.get_trace([add_result2])
        _, _, key2_mul = GraphTracer.get_trace([mul_result2])
        print(f"  arange+randn: add key={key2_add}, mul key={key2_mul}")

        # Check if keys are unique
        all_keys = [key1_add, key1_mul, key2_add, key2_mul]
        unique_keys = set(all_keys)
        print(f"  All keys unique: {len(all_keys) == len(unique_keys)}")

        return True

    except Exception as e:
        print(f"✗ Cache key test failed: {e}")
        return False


if __name__ == "__main__":
    print("=== RANDN ISSUE INVESTIGATION ===")

    # Test 1: randn without mixing operations
    test_randn_without_mixed_ops()
    print()

    # Test 2: cache key computation
    test_cache_key_hypothesis()
    print()

    # Test 3: the exact failing combination (run last)
    test_mixed_arange_randn()
    print()

    print("=== Investigation Complete ===")
