#!/usr/bin/env python3
"""Standalone script to reproduce the segfault."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    """Run the exact code that causes the segfault."""
    print("Running the exact crashing code...")

    import nabla

    print("   Creating arrays...")
    x = nabla.arange((2, 2))
    y = nabla.arange((2, 2))

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
    z2.realize()  # This might be where it crashes
    print("     Realizing z3 (+ operator)...")
    z3.realize()
    print("     Realizing z4 (* operator)...")
    z4.realize()  # Or this might be where it crashes

    print(f"   Add result shape: {z1.shape}")
    print(f"   Mul result shape: {z2.shape}")
    print(f"   Operator + works: {z3.shape}")
    print(f"   Operator * works: {z4.shape}")

    print("✅ All operations completed successfully!")


if __name__ == "__main__":
    main()
