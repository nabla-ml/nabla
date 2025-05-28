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

"""Test to reproduce the randn segfault."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_randn_crash():
    """Run the exact code that causes the segfault with randn."""
    print("Testing with randn (should crash)...")

    import nabla

    print("   Creating arrays...")
    x = nabla.arange((2, 2))
    y = nabla.randn((2, 2), seed=42)  # This is the problematic one

    print("   Testing function calls...")
    # Test function calls
    z1 = nabla.add(x, y)
    z2 = nabla.mul(x, y)

    print("   Testing operator overloading...")
    # Test operator overloading
    z3 = x + y
    z4 = x * y

    print("   Realizing results...")
    # Realize results
    print("     Realizing z1 (add)...")
    z1.realize()
    print("     Realizing z2 (mul)...")
    z2.realize()  # This should crash with randn
    print("     Realizing z3 (+ operator)...")
    z3.realize()
    print("     Realizing z4 (* operator)...")
    z4.realize()

    print(f"   Add result shape: {z1.shape}")
    print(f"   Mul result shape: {z2.shape}")
    print(f"   Operator + works: {z3.shape}")
    print(f"   Operator * works: {z4.shape}")

    print("✅ All operations completed successfully!")


def test_randn_only():
    """Test randn operation by itself."""
    print("\nTesting randn operation alone...")

    import nabla
    from nabla.core.execution_context import global_execution_context

    global_execution_context.clear()

    print("   Creating randn array...")
    y = nabla.randn((2, 2), seed=42)

    print("   Realizing randn array...")
    y.realize()

    print(f"   Randn result shape: {y.shape}")
    print("✅ Randn alone works!")


def test_randn_mixed_minimal():
    """Minimal test of randn mixed with another operation."""
    print("\nTesting minimal randn + operation mix...")

    import nabla
    from nabla.core.execution_context import global_execution_context

    global_execution_context.clear()

    print("   Creating arrays...")
    x = nabla.arange((2, 2))
    y = nabla.randn((2, 2), seed=42)

    print("   Creating mixed operation...")
    result = x + y  # Simple mixed operation

    print("   Realizing mixed operation...")
    result.realize()  # This should crash

    print(f"   Mixed result shape: {result.shape}")
    print("✅ Mixed operation works!")


if __name__ == "__main__":
    print("=== RANDN SEGFAULT INVESTIGATION ===")

    # Test 1: randn alone (should work)
    try:
        test_randn_only()
    except Exception as e:
        print(f"❌ Randn alone failed: {e}")

    # Test 2: minimal mixed operation (should crash)
    try:
        test_randn_mixed_minimal()
    except Exception as e:
        print(f"❌ Minimal mixed failed: {e}")

    # Test 3: full reproduction (should crash)
    try:
        test_randn_crash()
    except Exception as e:
        print(f"❌ Full test failed: {e}")
