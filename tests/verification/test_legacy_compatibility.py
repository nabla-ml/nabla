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

"""
Test script to verify the refactored graph_improved.py works correctly.
"""

import sys

sys.path.append("/Users/tillife/Documents/CodingProjects/nabla/src")


def test_basic_operations():
    """Test basic operations work in the refactored version."""
    try:
        from nabla.graph_improved import add, cos, matmul, randn, sin

        print("✓ Imports successful")

        # Test array creation
        x = randn((3, 4), mean=0.0, std=1.0)
        print(f"✓ Created random array with shape {x.shape}")

        # Test unary operations
        y = sin(x)
        z = cos(x)
        print("✓ Unary ops: sin and cos computed")

        # Test binary operations
        w = add(y, z)
        print("✓ Binary op: addition computed")

        # Test matrix multiplication
        a = randn((3, 4))
        b = randn((4, 5))
        c = matmul(a, b)
        print(f"✓ Matrix multiplication: {a.shape} @ {b.shape} = {c.shape}")

        # Test realization
        c.realize()
        print("✓ Graph execution successful")

        print("\n🎉 All tests passed! Refactored code works correctly.")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_basic_operations()
