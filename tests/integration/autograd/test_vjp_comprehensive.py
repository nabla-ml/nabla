#!/usr/bin/env python3
"""Comprehensive tests for VJP and JVP rules."""

import numpy as np
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace


def test_binary_ops_vjp():
    """Test VJP for all binary operations."""
    print("=" * 70)
    print("Test: VJP for binary operations")
    print("=" * 70)

    x1 = nb.Tensor.from_dlpack(np.array([2.0, 3.0], dtype=np.float32))
    x2 = nb.Tensor.from_dlpack(np.array([4.0, 5.0], dtype=np.float32))

    # Test Add
    print("\n--- Testing Add VJP ---")

    def add_fn(a, b):
        return nb.add(a, b)

    traced = trace(add_fn, x1, x2)
    cotangent = nb.Tensor.from_dlpack(np.ones(2, dtype=np.float32))
    grads = backward_on_trace(traced, cotangent)
    print(f"✓ Add VJP: {len(grads)} gradients computed")

    # Test Mul
    print("\n--- Testing Mul VJP ---")

    def mul_fn(a, b):
        return nb.mul(a, b)

    traced = trace(mul_fn, x1, x2)
    grads = backward_on_trace(traced, cotangent)
    print(f"✓ Mul VJP: {len(grads)} gradients computed")

    # Test Sub
    print("\n--- Testing Sub VJP ---")

    def sub_fn(a, b):
        return nb.sub(a, b)

    traced = trace(sub_fn, x1, x2)
    grads = backward_on_trace(traced, cotangent)
    print(f"✓ Sub VJP: {len(grads)} gradients computed")

    # Test Div
    print("\n--- Testing Div VJP ---")

    def div_fn(a, b):
        return nb.div(a, b)

    traced = trace(div_fn, x1, x2)
    grads = backward_on_trace(traced, cotangent)
    print(f"✓ Div VJP: {len(grads)} gradients computed")

    print("\n✓ All binary operations support VJP!")


def test_unary_ops_vjp():
    """Test VJP for unary operations."""
    print("\n" + "=" * 70)
    print("Test: VJP for unary operations")
    print("=" * 70)

    x = nb.Tensor.from_dlpack(np.array([1.0, -2.0, 3.0], dtype=np.float32))

    # Test ReLU
    print("\n--- Testing ReLU VJP ---")

    def relu_fn(a):
        return nb.ops.relu(a)

    traced = trace(relu_fn, x)
    cotangent = nb.Tensor.from_dlpack(np.ones(3, dtype=np.float32))
    grads = backward_on_trace(traced, cotangent)
    print(f"✓ ReLU VJP: {len(grads)} gradients computed")

    # Test Exp
    print("\n--- Testing Exp VJP ---")

    def exp_fn(a):
        return nb.ops.exp(a)

    traced = trace(exp_fn, x)
    grads = backward_on_trace(traced, cotangent)
    print(f"✓ Exp VJP: {len(grads)} gradients computed")

    # Test Neg
    print("\n--- Testing Neg VJP ---")

    def neg_fn(a):
        return nb.ops.neg(a)

    traced = trace(neg_fn, x)
    grads = backward_on_trace(traced, cotangent)
    print(f"✓ Neg VJP: {len(grads)} gradients computed")

    print("\n✓ All unary operations support VJP!")


def test_composed_operations():
    """Test VJP on composed operations."""
    print("\n" + "=" * 70)
    print("Test: VJP for composed operations")
    print("=" * 70)

    x = nb.Tensor.from_dlpack(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    # Test: y = exp(-x) (sigmoid component)
    print("\n--- Testing exp(-x) ---")

    def composed_fn(a):
        neg_a = nb.ops.neg(a)
        return nb.ops.exp(neg_a)

    traced = trace(composed_fn, x)
    cotangent = nb.Tensor.from_dlpack(np.ones(3, dtype=np.float32))
    grads = backward_on_trace(traced, cotangent)
    print(f"✓ Composed exp(-x) VJP: {len(grads)} gradients computed")

    # Test: y = relu(x) * 2
    print("\n--- Testing relu(x) * 2 ---")
    two = nb.Tensor.from_dlpack(np.array([2.0, 2.0, 2.0], dtype=np.float32))

    def composed_fn2(a):
        r = nb.ops.relu(a)
        return nb.mul(r, two)

    traced = trace(composed_fn2, x)
    grads = backward_on_trace(traced, cotangent)
    print(f"✓ Composed relu(x)*2 VJP: {len(grads)} gradients computed")

    print("\n✓ Composed operations support VJP!")


def test_chain_rule():
    """Test chain rule: y = relu(x1 * x2 + x1)."""
    print("\n" + "=" * 70)
    print("Test: Chain rule with multiple operations")
    print("=" * 70)

    x1 = nb.Tensor.from_dlpack(np.array([2.0, -1.0, 3.0], dtype=np.float32))
    x2 = nb.Tensor.from_dlpack(np.array([1.0, 2.0, -1.0], dtype=np.float32))

    def chain_fn(a, b):
        prod = nb.mul(a, b)
        sum_val = nb.add(prod, a)
        return nb.ops.relu(sum_val)

    traced = trace(chain_fn, x1, x2)
    print(f"\nTrace:\n{traced}")

    cotangent = nb.Tensor.from_dlpack(np.ones(3, dtype=np.float32))
    grads = backward_on_trace(traced, cotangent)

    print(f"\n✓ Chain rule: Computed {len(grads)} gradients")
    for inp in [x1, x2]:
        if inp in grads:
            grad = grads[inp]
            print(f"  Gradient shape: {grad.shape}")


def test_matmul_chain():
    """Test matmul in a computation chain."""
    print("\n" + "=" * 70)
    print("Test: Matmul in computation chain")
    print("=" * 70)

    W = nb.Tensor.from_dlpack(np.random.randn(4, 3).astype(np.float32))
    x = nb.Tensor.from_dlpack(np.random.randn(3, 2).astype(np.float32))

    def matmul_chain(weight, input_x):
        out = nb.matmul(weight, input_x)
        return nb.ops.relu(out)

    traced = trace(matmul_chain, W, x)
    print(f"\nTrace:\n{traced}")

    cotangent = nb.Tensor.from_dlpack(np.ones((4, 2), dtype=np.float32))
    grads = backward_on_trace(traced, cotangent)

    print(f"\n✓ Matmul chain: Computed {len(grads)} gradients")


if __name__ == "__main__":
    all_pass = True

    try:
        all_pass &= test_binary_ops_vjp()
    except Exception as e:
        print(f"\n✗ Binary ops test failed: {e}")
        all_pass = False

    try:
        all_pass &= test_unary_ops_vjp()
    except Exception as e:
        print(f"\n✗ Unary ops test failed: {e}")
        all_pass = False

    try:
        all_pass &= test_composed_operations()
    except Exception as e:
        print(f"\n✗ Composed ops test failed: {e}")
        all_pass = False

    try:
        all_pass &= test_chain_rule()
    except Exception as e:
        print(f"\n✗ Chain rule test failed: {e}")
        all_pass = False

    try:
        all_pass &= test_matmul_chain()
    except Exception as e:
        print(f"\n✗ Matmul chain test failed: {e}")
        all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("✅ ALL VJP TESTS PASSED!")
    else:
        print("❌ SOME VJP TESTS FAILED")
    print("=" * 70)

    exit(0 if all_pass else 1)
