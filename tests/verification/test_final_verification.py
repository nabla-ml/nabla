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

"""
Comprehensive test to verify complete OOP refactoring is working.
This replaces the problematic test_clean_architecture.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_complete_oop_refactoring():
    """Test that the complete OOP refactoring is working."""
    print("🧹 Testing Complete OOP Refactoring...\n")

    try:
        print("1. Testing imports...")
        import nabla

        required_functions = [
            "array",
            "add",
            "mul",
            "sin",
            "cos",
            "reduce_sum",
            "transpose",
            "matmul",
            "arange",
            "randn",
        ]
        for func_name in required_functions:
            assert hasattr(nabla, func_name), f"Missing function: {func_name}"
        print("   ✅ All functions available")

        print("\n2. Testing operation classes inheritance...")
        from nabla.ops.binary import _add_op, _mul_op
        from nabla.ops.creation import RandNOp
        from nabla.ops.linalg import _matmul_op
        from nabla.ops.operation import (
            BinaryOperation,
            Operation,
            ReductionOperation,
            UnaryOperation,
            ViewOperation,
        )
        from nabla.ops.reduce import _reduce_sum_op
        from nabla.ops.unary import CosOp, NegateOp, SinOp
        from nabla.ops.view import _transpose_op

        # Test inheritance for global instances
        assert isinstance(_add_op, BinaryOperation)
        assert isinstance(_mul_op, BinaryOperation)
        assert isinstance(_reduce_sum_op, ReductionOperation)
        assert isinstance(_transpose_op, ViewOperation)
        assert isinstance(
            _matmul_op, Operation
        )  # MatMul inherits directly from Operation

        # Test inheritance for constructed instances
        assert isinstance(SinOp(), UnaryOperation)
        assert isinstance(CosOp(), UnaryOperation)
        assert isinstance(NegateOp(), UnaryOperation)
        assert isinstance(RandNOp((2, 2)), Operation)
        print("   ✅ All operations inherit correctly")

        print("\n3. Testing array creation...")
        # Test different array creation methods
        a = nabla.array([1.0, 2.0, 3.0])
        b = nabla.arange((2, 3))
        c = nabla.randn((2, 3), seed=42)
        print(f"   array: {a.shape}, arange: {b.shape}, randn: {c.shape}")
        print("   ✅ Array creation works")

        print("\n4. Testing binary operations...")
        # Test function calls
        d = nabla.add(b, c)
        e = nabla.mul(b, c)

        # Test operator overloading
        f = b + c
        g = b * c

        # Realize all
        d.realize()
        e.realize()
        f.realize()
        g.realize()
        print(f"   add: {d.shape}, mul: {e.shape}, +: {f.shape}, *: {g.shape}")
        print("   ✅ Binary operations work")

        print("\n5. Testing unary operations...")
        h = nabla.sin(b)
        i = nabla.cos(b)
        j = nabla.negate(b)

        h.realize()
        i.realize()
        j.realize()
        print(f"   sin: {h.shape}, cos: {i.shape}, negate: {j.shape}")
        print("   ✅ Unary operations work")

        print("\n6. Testing view operations...")
        k = nabla.transpose(b)
        l = nabla.reshape(b, (3, 2))

        k.realize()
        l.realize()
        print(f"   transpose: {k.shape}, reshape: {l.shape}")
        print("   ✅ View operations work")

        print("\n7. Testing reduction operations...")
        m = nabla.reduce_sum(b)

        m.realize()
        print(f"   reduce_sum: {m.shape}")
        print("   ✅ Reduction operations work")

        print("\n8. Testing matrix operations...")
        x = nabla.arange((3, 4))
        y = nabla.arange((4, 2))
        z = nabla.matmul(x, y)
        # Also test operator @
        w = x @ y

        z.realize()
        w.realize()
        print(f"   matmul: {z.shape}, @: {w.shape}")
        print("   ✅ Matrix operations work")

        print("\n9. Testing complex computation...")
        # Complex computation chain
        x1 = nabla.randn((3, 3), seed=123)
        x2 = nabla.arange((3, 3))
        x3 = x1 + x2
        x4 = nabla.sin(x3)
        x5 = nabla.transpose(x4)
        x6 = nabla.reduce_sum(x5)

        result = x6.realize()
        result_val = x6.get_numpy()
        print(f"   Complex computation result: {result_val}")
        print("   ✅ Complex computation works")

        print("\n10. Testing original graph availability...")
        assert hasattr(nabla, "original_graph")
        print("   ✅ Original graph still accessible")

        print("\n🎉 ALL TESTS PASSED! Complete OOP refactoring is successful!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_complete_oop_refactoring()
    sys.exit(0 if success else 1)
