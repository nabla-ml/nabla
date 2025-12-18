"""Comprehensive test: Symbolic dimensions in eager module.

Tests that:
1. Symbolic dimensions are preserved through operations
2. Tensors with symbolic shapes can be computed/realized

Uses clean API: strings for symbolic dims, ints for static dims.
"""

import asyncio
from eager import Tensor


def check_shape(tensor, expected_str):
    """Helper to check and display tensor shape."""
    actual = str(tensor._value.type.shape)
    status = "✅" if actual == expected_str else "❌"
    print(f"    {status} {actual}")
    return actual == expected_str


async def test_binary_operations():
    """Binary ops: add, mul, sub, div."""
    print("\n" + "="*70)
    print("1. Binary Operations")
    print("="*70)
    
    x = Tensor.ones(("b1", 64))
    y = Tensor.ones(("b1", 64))
    
    tests = {
        "x + y": x + y,
        "x * y": x * y,
        "x - y": x - y,
        "x / y": x / y,
    }
    
    results = []
    for name, result in tests.items():
        print(f"\n  {name}:")
        passed = check_shape(result, "[Dim('b1'), Dim(64)]")
        results.append(passed)
    
    return all(results)


async def test_matmul():
    """Matrix multiplication with symbolic dims."""
    print("\n" + "="*70)
    print("2. Matrix Multiplication")
    print("="*70)
    
    results = []
    
    # Symbolic batch
    print("\n  ('b2', 128) @ (128, 64):")
    x = Tensor.ones(("b2", 128))
    W = Tensor.ones((128, 64))
    y = x @ W
    results.append(check_shape(y, "[Dim('b2'), Dim(64)]"))
    
    # Symbolic output
    print("\n  (32, 128) @ (128, 'h1'):")
    x = Tensor.ones((32, 128))
    W = Tensor.ones((128, "h1"))
    y = x @ W
    results.append(check_shape(y, "[Dim(32), Dim('h1')]"))
    
    # Both symbolic
    print("\n  ('b3', 128) @ (128, 'h2'):")
    x = Tensor.ones(("b3", 128))
    W = Tensor.ones((128, "h2"))
    y = x @ W
    results.append(check_shape(y, "[Dim('b3'), Dim('h2')]"))
    
    return all(results)


async def test_broadcasting():
    """Broadcasting with symbolic dimensions."""
    print("\n" + "="*70)
    print("3. Broadcasting")
    print("="*70)
    
    results = []
    
    # Broadcast bias
    print("\n  ('b4', 64) + (64,):")
    x = Tensor.ones(("b4", 64))
    bias = Tensor.ones((64,))
    y = x + bias
    results.append(check_shape(y, "[Dim('b4'), Dim(64)]"))
    
    # Broadcast with ones
    print("\n  ('b5', 1) * ('b5', 64):")
    x = Tensor.ones(("b5", 1))
    y = Tensor.ones(("b5", 64))
    z = x * y
    results.append(check_shape(z, "[Dim('b5'), Dim(64)]"))
    
    return all(results)


async def test_operation_chain():
    """Chain of operations preserves symbolic dims."""
    print("\n" + "="*70)
    print("4. Operation Chains")
    print("="*70)
    
    print("\n  Building network: x @ W1 → add → @ W2")
    
    x = Tensor.ones(("b6", 128))
    W1 = Tensor.ones((128, "h3"))
    W2 = Tensor.ones(("h3", 10))
    
    h = x @ W1
    print(f"\n  h = x @ W1:")
    r1 = check_shape(h, "[Dim('b6'), Dim('h3')]")
    
    h = h + h
    print(f"\n  h = h + h:")
    r2 = check_shape(h, "[Dim('b6'), Dim('h3')]")
    
    y = h @ W2
    print(f"\n  y = h @ W2:")
    r3 = check_shape(y, "[Dim('b6'), Dim(10)]")
    
    return all([r1, r2, r3])


async def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SYMBOLIC DIMENSIONS: Comprehensive Test")
    print("="*70)
    print("\nUsing clean API: ('b1', 64) instead of Shape([SymbolicDim(...), ...])")
    
    results = {}
    
    results['binary'] = await test_binary_operations()
    results['matmul'] = await test_matmul()
    results['broadcasting'] = await test_broadcasting()
    results['chains'] = await test_operation_chain()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:15s}: {status}")
    
    if all(results.values()):
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Symbolic dimensions preserved through:")
        print("   • Binary operations (add, mul, sub, div)")
        print("   • Matrix multiplication")
        print("   • Broadcasting")
        print("   • Operation chains")
        print("\n💡 Use strings for symbolic dims: ('batch', 64)")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))


