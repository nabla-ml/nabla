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

"""Test script for OOP-based operations."""


def test_oop_operations():
    """Test OOP-based operations."""
    print("🧪 Testing OOP Operation Classes...")

    # Test basic imports
    print("✅ Testing imports...")
    from nabla.ops.binary import AddOp, MulOp
    from nabla.ops.linalg import MatMulOp
    from nabla.ops.operation import (
        BinaryOperation,
        Operation,
        ReductionOperation,
        UnaryOperation,
        ViewOperation,
    )
    from nabla.ops.reduce import ReduceSumOp
    from nabla.ops.unary import NegateOp, SinOp
    from nabla.ops.view import TransposeOp

    print("✅ All operation classes imported successfully!")

    # Test instantiation
    print("✅ Testing operation instantiation...")
    add_op = AddOp()
    mul_op = MulOp()
    negate_op = NegateOp()
    sin_op = SinOp()
    reduce_sum_op = ReduceSumOp((2, 3))  # ReduceSumOp requires arg_shape parameter
    transpose_op = TransposeOp()
    matmul_op = MatMulOp()
    print("✅ All operations instantiated successfully!")

    # Test that they inherit from correct base classes
    print("✅ Testing inheritance...")
    assert isinstance(add_op, BinaryOperation)
    assert isinstance(mul_op, BinaryOperation)
    assert isinstance(negate_op, UnaryOperation)
    assert isinstance(sin_op, UnaryOperation)
    assert isinstance(reduce_sum_op, ReductionOperation)
    assert isinstance(transpose_op, ViewOperation)
    assert isinstance(matmul_op, BinaryOperation)
    print("✅ All inheritance relationships correct!")

    # Test that all operations inherit from Operation
    print("✅ Testing base Operation inheritance...")
    assert isinstance(add_op, Operation)
    assert isinstance(mul_op, Operation)
    assert isinstance(negate_op, Operation)
    assert isinstance(sin_op, Operation)
    assert isinstance(reduce_sum_op, Operation)
    assert isinstance(transpose_op, Operation)
    assert isinstance(matmul_op, Operation)
    print("✅ All operations inherit from Operation base class!")

    print("🎉 All OOP operation tests passed!")

    # Test passes - return nothing (pytest expects None)
    return None
