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

"""Test to reproduce the exact caching scenario that causes segfaults."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_caching_behavior():
    """Test the exact caching scenario that might cause segfaults."""
    import nabla
    from nabla.core.execution_context import global_execution_context
    from nabla.core.graph_execution import GraphTracer

    print("Testing caching behavior...")
    print(f"Initial cache size: {global_execution_context.size()}")

    # Clear cache to start fresh
    global_execution_context.clear()
    print(f"Cache cleared, size: {global_execution_context.size()}")

    # Create arrays exactly like in the failing test
    print("\n1. Creating arrays...")
    x = nabla.arange((2, 2))
    y = nabla.randn((2, 2), seed=42)

    # Create operations exactly like in the failing test
    print("\n2. Creating operations...")
    z1 = nabla.add(x, y)  # This works
    z2 = nabla.mul(x, y)  # This causes segfault
    z3 = x * y  # Also multiplication
    z4 = x * y  # Same multiplication again

    # Check cache keys
    print("\n3. Checking cache keys...")
    _, _, key1 = GraphTracer.get_trace([z1])
    _, _, key2 = GraphTracer.get_trace([z2])
    _, _, key3 = GraphTracer.get_trace([z3])
    _, _, key4 = GraphTracer.get_trace([z4])

    print(f"z1 (add) cache key: {key1}")
    print(f"z2 (mul) cache key: {key2}")
    print(f"z3 (x*y) cache key: {key3}")
    print(f"z4 (x*y) cache key: {key4}")

    # Check if z3 and z4 have the same cache key (they should)
    if key3 == key4:
        print("✅ z3 and z4 have same cache key (expected)")
    else:
        print("❌ z3 and z4 have different cache keys (unexpected)")

    # Now realize them one by one and check cache status
    print(f"\n4. Before realization, cache size: {global_execution_context.size()}")

    print("   Realizing z1 (add)...")
    z1.realize()
    print(f"   After z1, cache size: {global_execution_context.size()}")
    print(f"   Cache contains key {key1}: {global_execution_context.contains(key1)}")

    print("   Realizing z2 (mul)...")
    try:
        z2.realize()  # This might segfault
        print("   ✅ z2 realized successfully")
        print(f"   After z2, cache size: {global_execution_context.size()}")
        print(
            f"   Cache contains key {key2}: {global_execution_context.contains(key2)}"
        )
    except Exception as e:
        print(f"   ❌ z2 failed: {e}")
        return False

    print("   Realizing z3 (x*y)...")
    try:
        z3.realize()  # This should use cached model if key3 == key2
        print("   ✅ z3 realized successfully")
        print(f"   After z3, cache size: {global_execution_context.size()}")
    except Exception as e:
        print(f"   ❌ z3 failed: {e}")
        return False

    print("   Realizing z4 (x*y again)...")
    try:
        z4.realize()  # This should definitely use cached model
        print("   ✅ z4 realized successfully")
        print(f"   After z4, cache size: {global_execution_context.size()}")
    except Exception as e:
        print(f"   ❌ z4 failed: {e}")
        return False

    print("\n✅ All operations completed successfully!")
    return True


def test_multiple_identical_operations():
    """Test creating and realizing multiple identical operations."""
    import nabla
    from nabla.core.execution_context import global_execution_context

    print("\n" + "=" * 50)
    print("Testing multiple identical operations...")

    global_execution_context.clear()

    x = nabla.arange((2, 2))
    y = nabla.randn((2, 2), seed=42)

    # Create many identical multiplication operations
    muls = []
    for i in range(5):
        mul_op = nabla.mul(x, y)
        muls.append(mul_op)
        print(f"Created mul operation {i + 1}")

    # Realize them all
    print(f"\nBefore realization, cache size: {global_execution_context.size()}")

    for i, mul_op in enumerate(muls):
        print(f"Realizing mul operation {i + 1}...")
        try:
            mul_op.realize()
            print(f"  ✅ Success, cache size: {global_execution_context.size()}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            return False

    print("✅ All identical operations completed!")
    return True


if __name__ == "__main__":
    success1 = test_caching_behavior()
    success2 = test_multiple_identical_operations()

    if success1 and success2:
        print("\n🎉 All caching tests passed!")
    else:
        print("\n❌ Some caching tests failed!")
