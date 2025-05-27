#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or beautiful, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Minimal reproduction of the segfault to isolate the exact cause."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from nabla import add, arange, mul, randn
from nabla.core.execution_context import global_execution_context


def test_minimal_reproduction():
    """Reproduce the exact conditions from the failing test."""
    print("=== Minimal Segfault Reproduction ===")

    global_execution_context.clear()
    print("Cache cleared")

    try:
        print("1. Creating arrays with arange and randn...")
        x = arange((2, 2))
        y = randn((2, 2), seed=42)
        print("   Arrays created successfully")

        print("2. Testing add operation...")
        z1 = add(x, y)
        z1.realize()
        print("   Add operation successful")

        print("3. Testing mul operation (this should trigger segfault)...")
        z2 = mul(x, y)
        z2.realize()
        print("   Mul operation successful")

        return True

    except Exception as e:
        print(f"   Error: {e}")
        return False


def test_with_array_creation():
    """Test with array() function instead of arange/randn."""
    print("\n=== Test with array() function ===")

    global_execution_context.clear()

    try:
        from nabla import array

        print("1. Creating arrays with array() function...")
        x = array([[0.0, 1.0], [2.0, 3.0]])
        y = array([[1.0, 1.0], [1.0, 1.0]])
        print("   Arrays created successfully")

        print("2. Testing add operation...")
        z1 = add(x, y)
        z1.realize()
        print("   Add operation successful")

        print("3. Testing mul operation...")
        z2 = mul(x, y)
        z2.realize()
        print("   Mul operation successful")

        return True

    except Exception as e:
        print(f"   Error: {e}")
        return False


def test_only_arange():
    """Test with only arange arrays."""
    print("\n=== Test with only arange arrays ===")

    global_execution_context.clear()

    try:
        print("1. Creating arrays with arange only...")
        x = arange((2, 2))
        y = arange((2, 2))
        print("   Arrays created successfully")

        print("2. Testing add operation...")
        z1 = add(x, y)
        z1.realize()
        print("   Add operation successful")

        print("3. Testing mul operation...")
        z2 = mul(x, y)
        z2.realize()
        print("   Mul operation successful")

        return True

    except Exception as e:
        print(f"   Error: {e}")
        return False


def test_only_randn():
    """Test with only randn arrays."""
    print("\n=== Test with only randn arrays ===")

    global_execution_context.clear()

    try:
        print("1. Creating arrays with randn only...")
        x = randn((2, 2), seed=42)
        y = randn((2, 2), seed=43)
        print("   Arrays created successfully")

        print("2. Testing add operation...")
        z1 = add(x, y)
        z1.realize()
        print("   Add operation successful")

        print("3. Testing mul operation...")
        z2 = mul(x, y)
        z2.realize()
        print("   Mul operation successful")

        return True

    except Exception as e:
        print(f"   Error: {e}")
        return False


def test_operations_separately():
    """Test operations in separate executions."""
    print("\n=== Test operations separately ===")

    try:
        # Test 1: Only add
        print("1. Testing only add...")
        global_execution_context.clear()
        x1 = arange((2, 2))
        y1 = randn((2, 2), seed=42)
        z1 = add(x1, y1)
        z1.realize()
        print("   Add-only successful")

        # Test 2: Only mul
        print("2. Testing only mul...")
        global_execution_context.clear()
        x2 = arange((2, 2))
        y2 = randn((2, 2), seed=42)
        z2 = mul(x2, y2)
        z2.realize()
        print("   Mul-only successful")

        return True

    except Exception as e:
        print(f"   Error: {e}")
        return False


if __name__ == "__main__":
    tests = [
        test_with_array_creation,
        test_only_arange,
        test_only_randn,
        test_operations_separately,
        test_minimal_reproduction,  # Run this last as it may segfault
    ]

    for test in tests:
        try:
            result = test()
            if not result:
                print(f"   ❌ {test.__name__} failed")
            else:
                print(f"   ✅ {test.__name__} passed")
        except Exception as e:
            print(f"   ❌ {test.__name__} crashed: {e}")

    print("\n=== Test Complete ===")
