#!/usr/bin/env python3
"""Test script for OOP-based operations."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("🧪 Testing OOP Operation Classes...")

try:
    # Test basic imports
    print("✅ Testing imports...")
    from nabla.ops.operation import (
        Operation,
        UnaryOperation,
        BinaryOperation,
        ReductionOperation,
        ViewOperation,
    )
    from nabla.ops.unary import NegateOp, SinOp, CosOp, CastOp
    from nabla.ops.binary import AddOp, MulOp
    from nabla.ops.reduce import SumOp
    from nabla.ops.view import TransposeOp, ReshapeOp, BroadcastToOp
    from nabla.ops.linalg import MatMulOp
    from nabla.ops.creation import RandNOp

    print("✅ All operation classes imported successfully!")

    # Test instantiation
    print("✅ Testing operation instantiation...")
    add_op = AddOp()
    mul_op = MulOp()
    negate_op = NegateOp()
    sin_op = SinOp()
    sum_op = SumOp()
    transpose_op = TransposeOp()
    matmul_op = MatMulOp()
    print("✅ All operations instantiated successfully!")

    # Test that they inherit from correct base classes
    print("✅ Testing inheritance...")
    assert isinstance(add_op, BinaryOperation)
    assert isinstance(mul_op, BinaryOperation)
    assert isinstance(negate_op, UnaryOperation)
    assert isinstance(sin_op, UnaryOperation)
    assert isinstance(sum_op, ReductionOperation)
    assert isinstance(transpose_op, ViewOperation)
    assert isinstance(matmul_op, BinaryOperation)
    print("✅ All inheritance relationships correct!")

    # Test that all operations inherit from Operation
    print("✅ Testing base Operation inheritance...")
    assert isinstance(add_op, Operation)
    assert isinstance(mul_op, Operation)
    assert isinstance(negate_op, Operation)
    assert isinstance(sin_op, Operation)
    assert isinstance(sum_op, Operation)
    assert isinstance(transpose_op, Operation)
    assert isinstance(matmul_op, Operation)
    print("✅ All operations inherit from Operation base class!")

    print("🎉 All OOP operation tests passed!")

except Exception as e:
    print(f"❌ Error testing OOP operations: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
