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

"""Test the specific operations that cause segfault."""


def test_lazy_execution_operations():
    """Test specific operations in lazy execution mode."""
    print("🔍 Testing specific operations from clean architecture test...")

    import nabla

    print("✅ Import successful")

    # Test what the clean architecture test does
    print("Testing array creation...")
    x = nabla.arange((2, 2))
    y = nabla.randn((2, 2), seed=42)
    print(f"✅ x.shape={x.shape}, y.shape={y.shape}")

    # Test function calls (this might be causing the issue)
    print("Testing nabla.add function...")
    z1 = nabla.add(x, y)
    print(f"✅ nabla.add result: {z1.shape}")

    print("Testing nabla.mul function...")
    z2 = nabla.mul(x, y)
    print(f"✅ nabla.mul result: {z2.shape}")

    # Test operator overloading
    print("Testing operators...")
    z3 = x + y
    z4 = x * y
    print(f"✅ Operators work: z3.shape={z3.shape}, z4.shape={z4.shape}")

    # Test realize (this is often where segfaults happen)
    print("Testing realize...")
    print("Realizing z1...")
    z1.realize()
    print("✅ z1 realized")

    print("Realizing z2...")
    z2.realize()
    print("✅ z2 realized")

    print("Realizing z3...")
    z3.realize()
    print("✅ z3 realized")

    print("Realizing z4...")
    z4.realize()
    print("✅ z4 realized")

    print("🎉 All operations completed successfully!")
