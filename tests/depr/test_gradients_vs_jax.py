#!/usr/bin/env python3
"""
Integration tests for automatic differentiation comparing Nabla against JAX.

Tests real-world workloads including:
- Multi-layer networks
- Composed operations
- Matrix computations
- Activation functions
"""

# Import testing utilities
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nabla as nb
from nabla.core.autograd import backward_on_trace
from nabla.core.graph.tracing import trace

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "unit"))


def to_numpy(t):
    """Convert Nabla Tensor to numpy array, ensuring it's realized."""
    from nabla.core.graph.engine import GRAPH

    # Ensure tensor is evaluated
    if not t._impl.is_realized:
        GRAPH.evaluate(t)
    return np.asarray(t.to_numpy())


def test_simple_linear_layer():
    """Test gradients for a simple linear layer: y = W @ x + b."""
    print("\n" + "=" * 70)
    print("Test: Simple Linear Layer - y = W @ x + b")
    print("=" * 70)

    # Setup
    np.random.seed(42)
    W_np = np.random.randn(4, 3).astype(np.float32) * 0.1
    x_np = np.random.randn(3, 2).astype(np.float32)
    b_np = np.random.randn(4, 1).astype(np.float32) * 0.1

    # Nabla forward + backward
    W_nb = nb.Tensor.from_dlpack(W_np.copy())
    x_nb = nb.Tensor.from_dlpack(x_np.copy())
    b_nb = nb.Tensor.from_dlpack(b_np.copy())

    def linear_layer(W, x, b):
        return nb.add(nb.matmul(W, x), b)

    traced = trace(linear_layer, W_nb, x_nb, b_nb)
    # Don't call the function again - traced already has the computation

    # Backward
    cotangent = nb.Tensor.from_dlpack(np.ones((4, 2), dtype=np.float32))
    grads_nb = backward_on_trace(traced, cotangent)

    # JAX gradients
    def linear_layer_jax(W, x, b):
        return jnp.add(jnp.matmul(W, x), b)

    grad_fn = jax.grad(
        lambda W, x, b: jnp.sum(linear_layer_jax(W, x, b)), argnums=(0, 1, 2)
    )
    grads_jax = grad_fn(W_np, x_np, b_np)

    # Compare - grads_nb now contains Tensor objects directly
    grad_W_nb = to_numpy(grads_nb[W_nb])
    grad_x_nb = to_numpy(grads_nb[x_nb])
    grad_b_nb = to_numpy(grads_nb[b_nb])

    print("\nGradient shapes:")
    print(f"  ∇W: Nabla {grad_W_nb.shape}, JAX {grads_jax[0].shape}")
    print(f"  ∇x: Nabla {grad_x_nb.shape}, JAX {grads_jax[1].shape}")
    print(f"  ∇b: Nabla {grad_b_nb.shape}, JAX {grads_jax[2].shape}")

    np.testing.assert_allclose(grad_W_nb, grads_jax[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grad_x_nb, grads_jax[1], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grad_b_nb, grads_jax[2], rtol=1e-5, atol=1e-6)

    print("✓ All gradients match JAX!")


def test_two_layer_network():
    """Test gradients for a two-layer network with ReLU."""
    print("\n" + "=" * 70)
    print("Test: Two-Layer Network - y = W2 @ relu(W1 @ x)")
    print("=" * 70)

    # Setup
    np.random.seed(123)
    W1_np = np.random.randn(5, 3).astype(np.float32) * 0.1
    W2_np = np.random.randn(4, 5).astype(np.float32) * 0.1
    x_np = np.random.randn(3, 2).astype(np.float32)

    # Nabla
    W1_nb = nb.Tensor.from_dlpack(W1_np.copy())
    W2_nb = nb.Tensor.from_dlpack(W2_np.copy())
    x_nb = nb.Tensor.from_dlpack(x_np.copy())

    def two_layer(W1, W2, x):
        h1 = nb.matmul(W1, x)
        h1_act = nb.ops.relu(h1)
        return nb.matmul(W2, h1_act)

    traced = trace(two_layer, W1_nb, W2_nb, x_nb)

    # Backward
    cotangent = nb.Tensor.from_dlpack(np.ones((4, 2), dtype=np.float32))
    grads_nb = backward_on_trace(traced, cotangent)

    # JAX
    def two_layer_jax(W1, W2, x):
        h1 = jnp.matmul(W1, x)
        h1_act = jax.nn.relu(h1)
        return jnp.matmul(W2, h1_act)

    grad_fn = jax.grad(
        lambda W1, W2, x: jnp.sum(two_layer_jax(W1, W2, x)), argnums=(0, 1, 2)
    )
    grads_jax = grad_fn(W1_np, W2_np, x_np)

    # Compare
    grad_W1_nb = to_numpy(grads_nb[W1_nb])
    grad_W2_nb = to_numpy(grads_nb[W2_nb])
    grad_x_nb = to_numpy(grads_nb[x_nb])

    print("\nGradient shapes:")
    print(f"  ∇W1: Nabla {grad_W1_nb.shape}, JAX {grads_jax[0].shape}")
    print(f"  ∇W2: Nabla {grad_W2_nb.shape}, JAX {grads_jax[1].shape}")
    print(f"  ∇x: Nabla {grad_x_nb.shape}, JAX {grads_jax[2].shape}")

    np.testing.assert_allclose(
        grad_W1_nb,
        grads_jax[0],
        rtol=1e-5,
        atol=1e-6,
        err_msg="Gradient w.r.t. W1 mismatch",
    )
    np.testing.assert_allclose(
        grad_W2_nb,
        grads_jax[1],
        rtol=1e-5,
        atol=1e-6,
        err_msg="Gradient w.r.t. W2 mismatch",
    )
    np.testing.assert_allclose(
        grad_x_nb,
        grads_jax[2],
        rtol=1e-5,
        atol=1e-6,
        err_msg="Gradient w.r.t. x mismatch",
    )

    print("✓ All gradients match JAX!")


def test_elementwise_operations():
    """Test gradients for composed elementwise operations."""
    print("\n" + "=" * 70)
    print("Test: Elementwise Operations - y = exp(-relu(x1 * x2))")
    print("=" * 70)

    # Setup
    np.random.seed(456)
    x1_np = np.random.randn(3, 4).astype(np.float32)
    x2_np = np.random.randn(3, 4).astype(np.float32)

    # Nabla
    x1_nb = nb.Tensor.from_dlpack(x1_np.copy())
    x2_nb = nb.Tensor.from_dlpack(x2_np.copy())

    def elementwise_fn(x1, x2):
        prod = nb.mul(x1, x2)
        relu_prod = nb.ops.relu(prod)
        neg_relu = nb.ops.neg(relu_prod)
        return nb.ops.exp(neg_relu)

    traced = trace(elementwise_fn, x1_nb, x2_nb)

    # Backward
    cotangent = nb.Tensor.from_dlpack(np.ones((3, 4), dtype=np.float32))
    grads_nb = backward_on_trace(traced, cotangent)

    # JAX
    def elementwise_fn_jax(x1, x2):
        prod = jnp.multiply(x1, x2)
        relu_prod = jax.nn.relu(prod)
        neg_relu = -relu_prod
        return jnp.exp(neg_relu)

    grad_fn = jax.grad(
        lambda x1, x2: jnp.sum(elementwise_fn_jax(x1, x2)), argnums=(0, 1)
    )
    grads_jax = grad_fn(x1_np, x2_np)

    # Compare
    grad_x1_nb = to_numpy(grads_nb[x1_nb])
    grad_x2_nb = to_numpy(grads_nb[x2_nb])

    print("\nGradient shapes:")
    print(f"  ∇x1: Nabla {grad_x1_nb.shape}, JAX {grads_jax[0].shape}")
    print(f"  ∇x2: Nabla {grad_x2_nb.shape}, JAX {grads_jax[1].shape}")

    np.testing.assert_allclose(
        grad_x1_nb,
        grads_jax[0],
        rtol=1e-5,
        atol=1e-6,
        err_msg="Gradient w.r.t. x1 mismatch",
    )
    np.testing.assert_allclose(
        grad_x2_nb,
        grads_jax[1],
        rtol=1e-5,
        atol=1e-6,
        err_msg="Gradient w.r.t. x2 mismatch",
    )

    print("✓ All gradients match JAX!")


def test_mixed_operations():
    """Test gradients for a mix of linear and elementwise operations."""
    print("\n" + "=" * 70)
    print("Test: Mixed Operations - y = (W @ x) * relu(bias)")
    print("=" * 70)

    # Setup
    np.random.seed(789)
    W_np = np.random.randn(4, 3).astype(np.float32) * 0.1
    x_np = np.random.randn(3, 2).astype(np.float32)
    bias_np = np.random.randn(4, 2).astype(np.float32)

    # Nabla
    W_nb = nb.Tensor.from_dlpack(W_np.copy())
    x_nb = nb.Tensor.from_dlpack(x_np.copy())
    bias_nb = nb.Tensor.from_dlpack(bias_np.copy())

    def mixed_fn(W, x, bias):
        linear = nb.matmul(W, x)
        bias_act = nb.ops.relu(bias)
        return nb.mul(linear, bias_act)

    traced = trace(mixed_fn, W_nb, x_nb, bias_nb)

    # Backward
    cotangent = nb.Tensor.from_dlpack(np.ones((4, 2), dtype=np.float32))
    grads_nb = backward_on_trace(traced, cotangent)

    # JAX
    def mixed_fn_jax(W, x, bias):
        linear = jnp.matmul(W, x)
        bias_act = jax.nn.relu(bias)
        return jnp.multiply(linear, bias_act)

    grad_fn = jax.grad(
        lambda W, x, bias: jnp.sum(mixed_fn_jax(W, x, bias)), argnums=(0, 1, 2)
    )
    grads_jax = grad_fn(W_np, x_np, bias_np)

    # Compare
    grad_W_nb = to_numpy(grads_nb[W_nb])
    grad_x_nb = to_numpy(grads_nb[x_nb])
    grad_bias_nb = to_numpy(grads_nb[bias_nb])

    print("\nGradient shapes:")
    print(f"  ∇W: Nabla {grad_W_nb.shape}, JAX {grads_jax[0].shape}")
    print(f"  ∇x: Nabla {grad_x_nb.shape}, JAX {grads_jax[1].shape}")
    print(f"  ∇bias: Nabla {grad_bias_nb.shape}, JAX {grads_jax[2].shape}")

    np.testing.assert_allclose(
        grad_W_nb,
        grads_jax[0],
        rtol=1e-5,
        atol=1e-6,
        err_msg="Gradient w.r.t. W mismatch",
    )
    np.testing.assert_allclose(
        grad_x_nb,
        grads_jax[1],
        rtol=1e-5,
        atol=1e-6,
        err_msg="Gradient w.r.t. x mismatch",
    )
    np.testing.assert_allclose(
        grad_bias_nb,
        grads_jax[2],
        rtol=1e-5,
        atol=1e-6,
        err_msg="Gradient w.r.t. bias mismatch",
    )

    print("✓ All gradients match JAX!")


def test_deep_chain():
    """Test gradients through a deeper computation chain."""
    print("\n" + "=" * 70)
    print("Test: Deep Chain - Multi-step computation")
    print("=" * 70)

    # Setup
    np.random.seed(111)
    x_np = np.random.randn(5, 3).astype(np.float32)
    y_np = np.random.randn(5, 3).astype(np.float32)

    # Nabla
    x_nb = nb.Tensor.from_dlpack(x_np.copy())
    y_nb = nb.Tensor.from_dlpack(y_np.copy())

    def deep_chain(x, y):
        # Step 1: Add
        z1 = nb.add(x, y)
        # Step 2: ReLU
        z2 = nb.ops.relu(z1)
        # Step 3: Multiply by x
        z3 = nb.mul(z2, x)
        # Step 4: Exp
        z4 = nb.ops.exp(z3)
        # Step 5: Subtract y
        z5 = nb.sub(z4, y)
        return z5

    traced = trace(deep_chain, x_nb, y_nb)

    # Backward
    cotangent = nb.Tensor.from_dlpack(np.ones((5, 3), dtype=np.float32))
    grads_nb = backward_on_trace(traced, cotangent)

    # JAX
    def deep_chain_jax(x, y):
        z1 = jnp.add(x, y)
        z2 = jax.nn.relu(z1)
        z3 = jnp.multiply(z2, x)
        z4 = jnp.exp(z3)
        z5 = jnp.subtract(z4, y)
        return z5

    grad_fn = jax.grad(lambda x, y: jnp.sum(deep_chain_jax(x, y)), argnums=(0, 1))
    grads_jax = grad_fn(x_np, y_np)

    # Compare
    grad_x_nb = to_numpy(grads_nb[x_nb])
    grad_y_nb = to_numpy(grads_nb[y_nb])

    print("\nGradient shapes:")
    print(f"  ∇x: Nabla {grad_x_nb.shape}, JAX {grads_jax[0].shape}")
    print(f"  ∇y: Nabla {grad_y_nb.shape}, JAX {grads_jax[1].shape}")

    np.testing.assert_allclose(
        grad_x_nb,
        grads_jax[0],
        rtol=1e-5,
        atol=1e-6,
        err_msg="Gradient w.r.t. x mismatch",
    )
    np.testing.assert_allclose(
        grad_y_nb,
        grads_jax[1],
        rtol=1e-5,
        atol=1e-6,
        err_msg="Gradient w.r.t. y mismatch",
    )

    print("✓ All gradients match JAX!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Nabla vs JAX Gradient Comparison Tests")
    print("=" * 70)

    all_pass = True

    tests = [
        ("Simple Linear Layer", test_simple_linear_layer),
        ("Two-Layer Network", test_two_layer_network),
        ("Elementwise Operations", test_elementwise_operations),
        ("Mixed Operations", test_mixed_operations),
        ("Deep Computation Chain", test_deep_chain),
    ]

    for test_name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"\n✗ {test_name} FAILED:")
            print(f"  {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("✅ ALL TESTS PASSED - Nabla gradients match JAX!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70)

    exit(0 if all_pass else 1)
