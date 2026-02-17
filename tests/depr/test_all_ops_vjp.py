#!/ plugins/python3
"""Comprehensive VJP testing against JAX."""

import numpy as np
import pytest

import nabla as nb
from nabla.core.autograd import backward_on_trace
from nabla.core.graph.tracing import trace

# Try to import JAX for comparison, but don't fail if not present
try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def check_vjp(fn, jax_fn, *args):
    """Generic checker for VJP against JAX."""
    # Nabla computation
    traced = trace(fn, *args)
    nabla_out = fn(*args)

    # Cotangent matching output shape - convert Dim to int for NumPy
    if isinstance(nabla_out, (list, tuple)):
        cotangent = [
            nb.Tensor.from_dlpack(
                np.ones(tuple(int(d) for d in t.shape), dtype=np.float32)
            )
            for t in nabla_out
        ]
    else:
        out_shape = tuple(int(d) for d in nabla_out.shape)
        cotangent = nb.Tensor.from_dlpack(np.ones(out_shape, dtype=np.float32))

    nabla_grads = backward_on_trace(traced, cotangent)

    if not HAS_JAX:
        pytest.skip("JAX not installed")

    # JAX computation
    jax_args = [jnp.array(a.to_numpy()) for a in args]

    if isinstance(nabla_out, (list, tuple)):
        # Match JAX's expected structure (list/tuple)
        jax_val, jax_vjp_fn = jax.vjp(jax_fn, *jax_args)
        if isinstance(jax_val, list):
            jax_cot = [jnp.ones(tuple(int(d) for d in v.shape)) for v in jax_val]
        else:
            jax_cot = tuple(jnp.ones(tuple(int(d) for d in v.shape)) for v in jax_val)
        jax_grads = jax_vjp_fn(jax_cot)
    else:
        jax_val, jax_vjp_fn = jax.vjp(jax_fn, *jax_args)
        jax_cot = jnp.ones(jax_val.shape)
        jax_grads = jax_vjp_fn(jax_cot)

    # Compare
    for i, arg in enumerate(args):
        n_grad = nabla_grads[arg].to_numpy()
        j_grad = np.array(jax_grads[i])
        np.testing.assert_allclose(n_grad, j_grad, rtol=1e-5, atol=1e-5)


def test_vjp_binary():
    x = nb.Tensor.from_dlpack(np.array([2.0, 3.0], dtype=np.float32))
    y = nb.Tensor.from_dlpack(np.array([4.0, 5.0], dtype=np.float32))

    check_vjp(lambda a, b: a + b, lambda a, b: a + b, x, y)
    check_vjp(lambda a, b: a * b, lambda a, b: a * b, x, y)
    check_vjp(lambda a, b: a - b, lambda a, b: a - b, x, y)
    check_vjp(lambda a, b: a / b, lambda a, b: a / b, x, y)


def test_vjp_unary():
    x = nb.Tensor.from_dlpack(np.array([1.0, -2.0, 3.0], dtype=np.float32))

    check_vjp(nb.ops.relu, jax.nn.relu, x)
    check_vjp(nb.ops.sigmoid, jax.nn.sigmoid, x)
    check_vjp(nb.ops.tanh, jnp.tanh, x)
    check_vjp(nb.ops.exp, jnp.exp, x)
    check_vjp(nb.ops.neg, jnp.negative, x)
    check_vjp(nb.ops.abs, jnp.abs, x)
    # Filter for positive for log/sqrt
    x_pos = nb.Tensor.from_dlpack(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    check_vjp(nb.ops.log, jnp.log, x_pos)
    check_vjp(nb.ops.sqrt, jnp.sqrt, x_pos)


def test_vjp_reduction():
    x = nb.Tensor.from_dlpack(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    check_vjp(lambda a: nb.reduce_sum(a, axis=0), lambda a: jnp.sum(a, axis=0), x)
    check_vjp(lambda a: nb.mean(a, axis=1), lambda a: jnp.mean(a, axis=1), x)


def test_vjp_view():
    x = nb.Tensor.from_dlpack(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    check_vjp(lambda a: nb.reshape(a, (2, 2)), lambda a: jnp.reshape(a, (2, 2)), x)
    check_vjp(
        lambda a: nb.unsqueeze(a, axis=0), lambda a: jnp.expand_dims(a, axis=0), x
    )


def test_vjp_multi_output():
    x = nb.Tensor.from_dlpack(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    def nb_split(a):
        return nb.ops.split(a, num_splits=2, axis=0)

    def jax_split(a):
        return jnp.split(a, 2, axis=0)

    check_vjp(nb_split, jax_split, x)
