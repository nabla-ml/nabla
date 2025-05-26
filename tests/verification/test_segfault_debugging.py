#!/usr/bin/env python3
"""Test the specific operations that cause segfault."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("🔍 Testing specific operations from clean architecture test...")

try:
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

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
