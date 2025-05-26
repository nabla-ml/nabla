#!/usr/bin/env python3
"""Test script for OOP-based operations with arrays."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("🧪 Testing OOP Operations with Arrays...")

try:
    # Import the clean entry point
    import nabla

    print("✅ Imported nabla successfully!")

    # Test array creation
    print("✅ Testing array creation...")
    x = nabla.arange((2, 3))
    y = nabla.arange((2, 3))
    print(f"Created arrays: x.shape={x.shape}, y.shape={y.shape}")

    # Test binary operations (using OOP)
    print("✅ Testing binary operations...")
    z = x + y  # Should use AddOp
    w = x * y  # Should use MulOp
    print(f"Addition result shape: {z.shape}")
    print(f"Multiplication result shape: {w.shape}")

    # Test unary operations (using OOP)
    print("✅ Testing unary operations...")
    neg_x = nabla.negate(x)  # Should use NegateOp
    sin_x = nabla.sin(x)  # Should use SinOp
    cos_x = nabla.cos(x)  # Should use CosOp
    print(f"Negate result shape: {neg_x.shape}")
    print(f"Sin result shape: {sin_x.shape}")
    print(f"Cos result shape: {cos_x.shape}")

    # Test reduction operations (using OOP)
    print("✅ Testing reduction operations...")
    sum_x = nabla.sum(x)  # Should use SumOp
    sum_axis = nabla.sum(x, axes=0)  # Should use SumOp
    print(f"Sum all shape: {sum_x.shape}")
    print(f"Sum axis 0 shape: {sum_axis.shape}")

    # Test view operations (using OOP)
    print("✅ Testing view operations...")
    t_x = nabla.transpose(x)  # Should use TransposeOp
    r_x = nabla.reshape(x, (3, 2))  # Should use ReshapeOp
    b_x = nabla.broadcast_to(x, (4, 2, 3))  # Should use BroadcastToOp
    print(f"Transpose shape: {t_x.shape}")
    print(f"Reshape shape: {r_x.shape}")
    print(f"Broadcast shape: {b_x.shape}")

    # Test linalg operations (using OOP)
    print("✅ Testing linalg operations...")
    a = nabla.arange((2, 3))
    b = nabla.arange((3, 2))
    c = a @ b  # Should use MatMulOp
    print(f"Matmul result shape: {c.shape}")

    print("🎉 All OOP operation functionality tests passed!")

except Exception as e:
    print(f"❌ Error testing OOP operations with arrays: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
