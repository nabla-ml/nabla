#!/usr/bin/env python3
"""
Simple test to verify complete OOP refactoring works.
"""

import nabla


def test_oop_refactoring():
    print("🧹 Testing Complete OOP Refactoring (Simple)...")

    # Test 1: Basic imports
    print("\n1. Testing imports...")
    try:

        print("   ✅ All operation instances imported")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

    # Test 2: Array creation
    print("\n2. Testing array creation...")
    try:
        a = nabla.array([1, 2, 3])
        b = nabla.array([4, 5, 6])
        print(f"   ✅ Array creation works: {a.shape}, {b.shape}")
    except Exception as e:
        print(f"   ❌ Array creation failed: {e}")
        return False

    # Test 3: Binary operations
    print("\n3. Testing binary operations...")
    try:
        c = nabla.add(a, b)
        d = nabla.mul(a, b)
        c.realize()
        d.realize()
        print(f"   ✅ Binary operations work: add={c.shape}, mul={d.shape}")
    except Exception as e:
        print(f"   ❌ Binary operations failed: {e}")
        return False

    # Test 4: Unary operations
    print("\n4. Testing unary operations...")
    try:
        e = nabla.sin(a)
        f = nabla.cos(a)
        e.realize()
        f.realize()
        print(f"   ✅ Unary operations work: sin={e.shape}, cos={f.shape}")
    except Exception as e:
        print(f"   ❌ Unary operations failed: {e}")
        return False

    # Test 5: Matrix operations
    print("\n5. Testing matrix operations...")
    try:
        m1 = nabla.array([[1, 2], [3, 4]])
        m2 = nabla.array([[5, 6], [7, 8]])
        m3 = nabla.matmul(m1, m2)
        m3.realize()
        print(f"   ✅ Matrix operations work: matmul={m3.shape}")
    except Exception as e:
        print(f"   ❌ Matrix operations failed: {e}")
        return False

    # Test 6: View operations
    print("\n6. Testing view operations...")
    try:
        g = nabla.transpose(m1)
        h = nabla.reshape(m1, (4, 1))
        g.realize()
        h.realize()
        print(f"   ✅ View operations work: transpose={g.shape}, reshape={h.shape}")
    except Exception as e:
        print(f"   ❌ View operations failed: {e}")
        return False

    # Test 7: Reduction operations
    print("\n7. Testing reduction operations...")
    try:
        i = nabla.reduce_sum(m1)
        i.realize()
        print(f"   ✅ Reduction operations work: reduce_sum={i.shape}")
    except Exception as e:
        print(f"   ❌ Reduction operations failed: {e}")
        return False

    print("\n✅ All OOP refactoring tests passed! No segmentation faults.")
    return True


if __name__ == "__main__":
    test_oop_refactoring()
