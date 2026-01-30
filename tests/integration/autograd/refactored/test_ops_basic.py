import numpy as np
import pytest
import nabla as nb
from .utils import check_vjp, HAS_JAX

np.random.seed(42)


def test_unary_ops_vjp():
    ops = [
        ("relu", nb.ops.relu, pytest.importorskip("jax.nn").relu if HAS_JAX else None),
        ("exp", nb.ops.exp, np.exp if HAS_JAX else None),
        ("tanh", nb.ops.tanh, np.tanh if HAS_JAX else None),
        (
            "sigmoid",
            nb.ops.sigmoid,
            pytest.importorskip("jax.nn").sigmoid if HAS_JAX else None,
        ),
        ("neg", nb.ops.neg, np.negative if HAS_JAX else None),
        ("abs", nb.ops.abs, np.abs if HAS_JAX else None),
        ("log", nb.ops.log, np.log if HAS_JAX else None),
        ("sqrt", nb.ops.sqrt, np.sqrt if HAS_JAX else None),
    ]

    x_default = np.random.randn(8, 8).astype(np.float32)
    x_pos = np.abs(x_default) + 0.1

    print("\n=== Testing Unary Ops VJP ===")
    for name, n_fn, j_fn in ops:
        args = (x_pos,) if name in ["log", "sqrt"] else (x_default,)
        check_vjp(name, n_fn, j_fn, args)


def test_binary_ops_vjp():
    if HAS_JAX:
        import jax.numpy as jnp
    else:
        jnp = None

    ops = [
        ("add", nb.add, jnp.add if HAS_JAX else None),
        ("sub", nb.sub, jnp.subtract if HAS_JAX else None),
        ("mul", nb.mul, jnp.multiply if HAS_JAX else None),
        ("div", nb.div, jnp.divide if HAS_JAX else None),
    ]

    x1 = np.random.randn(8, 8).astype(np.float32)
    x2 = np.random.randn(8, 8).astype(np.float32)

    print("\n=== Testing Binary Ops VJP ===")
    for name, n_fn, j_fn in ops:
        check_vjp(name, n_fn, j_fn, (x1, x2))


def test_reduction_ops_vjp():
    if HAS_JAX:
        import jax.numpy as jnp
    else:
        jnp = None

    x = np.random.randn(4, 4, 4).astype(np.float32)

    cases = [
        (
            "sum_axis0",
            lambda x: nb.reduce_sum(x, axis=0),
            lambda x: jnp.sum(x, axis=0) if HAS_JAX else None,
        ),
        (
            "sum_axis1_keepdims",
            lambda x: nb.reduce_sum(x, axis=1, keepdims=True),
            lambda x: jnp.sum(x, axis=1, keepdims=True) if HAS_JAX else None,
        ),
        (
            "mean_axis0",
            lambda x: nb.mean(x, axis=0),
            lambda x: jnp.mean(x, axis=0) if HAS_JAX else None,
        ),
        (
            "max_axis0",
            lambda x: nb.reduce_max(x, axis=0),
            lambda x: jnp.max(x, axis=0) if HAS_JAX else None,
        ),
    ]

    print("\n=== Testing Reduction Ops VJP ===")
    for name, n_fn, j_fn in cases:
        check_vjp(name, n_fn, j_fn, (x,))


def test_view_ops_vjp():
    if HAS_JAX:
        import jax.numpy as jnp
    else:
        jnp = None

    x = np.random.randn(4, 6).astype(np.float32)

    cases = [
        (
            "reshape",
            lambda x: nb.ops.reshape(x, (2, 12)),
            lambda x: jnp.reshape(x, (2, 12)) if HAS_JAX else None,
        ),
        (
            "swap_axes",
            lambda x: nb.ops.swap_axes(x, axis1=1, axis2=0),
            lambda x: jnp.swapaxes(x, 1, 0) if HAS_JAX else None,
        ),
        (
            "unsqueeze",
            lambda x: nb.ops.unsqueeze(x, axis=1),
            lambda x: jnp.expand_dims(x, axis=1) if HAS_JAX else None,
        ),
        (
            "squeeze",
            lambda x: nb.ops.squeeze(x, axis=1),
            lambda x: jnp.squeeze(x, axis=1) if HAS_JAX else None,
            np.random.randn(4, 1, 6).astype(np.float32),
        ),
    ]

    print("\n=== Testing View Ops VJP ===")
    for case in cases:
        if len(case) == 4:
            name, n_fn, j_fn, custom_x = case
            check_vjp(name, n_fn, j_fn, (custom_x,))
        else:
            name, n_fn, j_fn = case
            check_vjp(name, n_fn, j_fn, (x,))


def test_matmul_vjp():
    if HAS_JAX:
        import jax.numpy as jnp
    else:
        jnp = None

    x = np.random.randn(4, 8).astype(np.float32)
    y = np.random.randn(8, 4).astype(np.float32)

    print("\n=== Testing Matmul VJP ===")
    check_vjp("matmul", nb.matmul, jnp.matmul if HAS_JAX else None, (x, y))


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
