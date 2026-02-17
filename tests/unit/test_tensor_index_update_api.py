# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import jax.numpy as jnp
import jax

import nabla as nb
from nabla import P

from .common import assert_allclose, tensor_from_jax


def test_setitem_slice_updates_tensor_binding():
    x = tensor_from_jax(jnp.arange(6, dtype=jnp.float32).reshape(2, 3))
    update = tensor_from_jax(jnp.array([[10.0, 11.0], [12.0, 13.0]], dtype=jnp.float32))

    x[:, 1:3] = update

    expected = jnp.array([[0.0, 10.0, 11.0], [3.0, 12.0, 13.0]], dtype=jnp.float32)
    assert_allclose(x, expected)


def test_setitem_tensor_indices_scalar_broadcast():
    x = tensor_from_jax(jnp.arange(5, dtype=jnp.float32))
    idx = tensor_from_jax(jnp.array([0, 2, 4], dtype=jnp.int32))

    x[idx] = 9.0

    expected = jnp.array([9.0, 1.0, 9.0, 3.0, 9.0], dtype=jnp.float32)
    assert_allclose(x, expected)


def test_at_set_is_functional():
    x = tensor_from_jax(jnp.arange(5, dtype=jnp.float32))
    idx = tensor_from_jax(jnp.array([1, 3], dtype=jnp.int32))

    y = x.at[idx].set(7.0)

    assert_allclose(x, jnp.arange(5, dtype=jnp.float32))
    assert_allclose(y, jnp.array([0.0, 7.0, 2.0, 7.0, 4.0], dtype=jnp.float32))


def test_at_slice_set_uses_functional_mode_flag():
    x = tensor_from_jax(jnp.arange(6, dtype=jnp.float32).reshape(2, 3))
    update = tensor_from_jax(jnp.array([[10.0, 11.0], [12.0, 13.0]], dtype=jnp.float32))

    y = x.at[:, 1:3].set(update)

    assert y.op_kwargs.get("use_buffer_ops") is False
    assert x.op_kwargs.get("use_buffer_ops") is None


def test_at_add_is_functional():
    x = tensor_from_jax(jnp.arange(5, dtype=jnp.float32))
    idx = tensor_from_jax(jnp.array([1, 3], dtype=jnp.int32))

    y = x.at[idx].add(2.0)

    assert_allclose(x, jnp.arange(5, dtype=jnp.float32))
    assert_allclose(y, jnp.array([0.0, 3.0, 2.0, 5.0, 4.0], dtype=jnp.float32))


def test_setitem_slice_in_traced_mode_uses_buffer_flag():
    x = tensor_from_jax(jnp.arange(6, dtype=jnp.float32).reshape(2, 3)).trace()
    update = tensor_from_jax(jnp.array([[10.0, 11.0], [12.0, 13.0]], dtype=jnp.float32))

    x[:, 1:3] = update

    expected = jnp.array([[0.0, 10.0, 11.0], [3.0, 12.0, 13.0]], dtype=jnp.float32)
    assert_allclose(x, expected)
    assert x.is_traced is True
    assert x.op_kwargs.get("use_buffer_ops") is True


def test_setitem_slice_sharded_runs_with_buffer_flag(mesh_1d_2):
    x = tensor_from_jax(jnp.arange(8, dtype=jnp.float32).reshape(2, 4)).shard(
        mesh_1d_2, P("dp", None)
    )
    update = tensor_from_jax(jnp.array([[50.0, 51.0], [52.0, 53.0]], dtype=jnp.float32))

    x[:, 2:4] = update

    expected = jnp.array(
        [[0.0, 1.0, 50.0, 51.0], [4.0, 5.0, 52.0, 53.0]], dtype=jnp.float32
    )
    assert_allclose(x.gather(), expected)


def test_setitem_slice_sharded_traced_uses_buffer_flag(mesh_1d_2):
    x = (
        tensor_from_jax(jnp.arange(8, dtype=jnp.float32).reshape(2, 4))
        .shard(mesh_1d_2, P("dp", None))
        .trace()
    )
    update = tensor_from_jax(jnp.array([[50.0, 51.0], [52.0, 53.0]], dtype=jnp.float32))

    x[:, 2:4] = update

    expected = jnp.array(
        [[0.0, 1.0, 50.0, 51.0], [4.0, 5.0, 52.0, 53.0]], dtype=jnp.float32
    )
    assert_allclose(x.gather(), expected)
    assert x.is_traced is True
    assert x.op_kwargs.get("use_buffer_ops") is True


def test_setitem_slice_vmap_batched():
    x = tensor_from_jax(jnp.arange(18, dtype=jnp.float32).reshape(3, 2, 3))
    upd = tensor_from_jax(
        jnp.array(
            [
                [[100.0, 101.0], [102.0, 103.0]],
                [[110.0, 111.0], [112.0, 113.0]],
                [[120.0, 121.0], [122.0, 123.0]],
            ],
            dtype=jnp.float32,
        )
    )

    def f(a, b):
        a[:, 1:3] = b
        return a

    out = nb.vmap(f)(x, upd)

    def jax_f(a, b):
        return a.at[:, 1:3].set(b)

    expected = jax.vmap(jax_f)(
        jnp.arange(18, dtype=jnp.float32).reshape(3, 2, 3),
        jnp.array(
            [
                [[100.0, 101.0], [102.0, 103.0]],
                [[110.0, 111.0], [112.0, 113.0]],
                [[120.0, 121.0], [122.0, 123.0]],
            ],
            dtype=jnp.float32,
        ),
    )
    assert_allclose(out, expected)


def test_chained_updates_setitem_matches_jax():
    x = tensor_from_jax(jnp.arange(12, dtype=jnp.float32).reshape(3, 4))
    col_update = tensor_from_jax(
        jnp.array([[100.0, 101.0], [110.0, 111.0], [120.0, 121.0]], dtype=jnp.float32)
    )
    row_idx = tensor_from_jax(jnp.array([0, 2], dtype=jnp.int32))
    row_update = tensor_from_jax(
        jnp.array([[-5.0, -6.0, -7.0, -8.0], [5.0, 6.0, 7.0, 8.0]])
    )

    x[:, 1:3] = col_update
    x[row_idx] = row_update

    expected = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    expected = expected.at[:, 1:3].set(
        jnp.array([[100.0, 101.0], [110.0, 111.0], [120.0, 121.0]], dtype=jnp.float32)
    )
    expected = expected.at[jnp.array([0, 2], dtype=jnp.int32)].set(
        jnp.array([[-5.0, -6.0, -7.0, -8.0], [5.0, 6.0, 7.0, 8.0]], dtype=jnp.float32)
    )
    assert_allclose(x, expected)


def test_chained_updates_at_matches_jax_and_leaves_input_unchanged():
    x = tensor_from_jax(jnp.arange(10, dtype=jnp.float32))
    idx = tensor_from_jax(jnp.array([0, 2, 7], dtype=jnp.int32))

    y = x.at[2:8].set(tensor_from_jax(jnp.array([30.0, 31.0, 32.0, 33.0, 34.0, 35.0])))
    z = y.at[idx].add(4.0)

    expected = jnp.arange(10, dtype=jnp.float32)
    expected = expected.at[2:8].set(jnp.array([30.0, 31.0, 32.0, 33.0, 34.0, 35.0]))
    expected = expected.at[jnp.array([0, 2, 7], dtype=jnp.int32)].add(4.0)
    assert_allclose(z, expected)
    assert_allclose(x, jnp.arange(10, dtype=jnp.float32))


def test_grad_setitem_loss_matches_jax():
    update = tensor_from_jax(jnp.array([5.0, 6.0], dtype=jnp.float32))

    def nb_loss(x):
        y = x + 0.0
        y[1:3] = update
        return nb.reduce_sum(y * y)

    def jax_loss(x):
        y = x + 0.0
        y = y.at[1:3].set(jnp.array([5.0, 6.0], dtype=jnp.float32))
        return jnp.sum(y * y)

    x_jax = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    gx_nb = nb.grad(nb_loss)(tensor_from_jax(x_jax))
    gx_jax = jax.grad(jax_loss)(x_jax)
    assert_allclose(gx_nb, gx_jax)


def test_grad_at_chained_updates_matches_jax():
    idx = tensor_from_jax(jnp.array([0, 3], dtype=jnp.int32))

    def nb_loss(x):
        y = x.at[1:4].set(
            tensor_from_jax(jnp.array([2.0, 4.0, 6.0], dtype=jnp.float32))
        )
        z = y.at[idx].add(1.5)
        return nb.reduce_sum(z * z)

    def jax_loss(x):
        y = x.at[1:4].set(jnp.array([2.0, 4.0, 6.0], dtype=jnp.float32))
        z = y.at[jnp.array([0, 3], dtype=jnp.int32)].add(1.5)
        return jnp.sum(z * z)

    x_jax = jnp.array([0.5, -1.0, 2.0, 3.5, -2.0], dtype=jnp.float32)
    gx_nb = nb.grad(nb_loss)(tensor_from_jax(x_jax))
    gx_jax = jax.grad(jax_loss)(x_jax)
    assert_allclose(gx_nb, gx_jax)


def test_vmap_grad_setitem_matches_jax():
    update = tensor_from_jax(jnp.array([9.0, 10.0], dtype=jnp.float32))

    def nb_loss_single(x):
        y = x + 0.0
        y[1:3] = update
        return nb.reduce_sum(y * y)

    def jax_loss_single(x):
        y = x + 0.0
        y = y.at[1:3].set(jnp.array([9.0, 10.0], dtype=jnp.float32))
        return jnp.sum(y * y)

    x_jax = jnp.arange(20, dtype=jnp.float32).reshape(5, 4)
    gx_nb = nb.vmap(nb.grad(nb_loss_single))(tensor_from_jax(x_jax))
    gx_jax = jax.vmap(jax.grad(jax_loss_single))(x_jax)
    assert_allclose(gx_nb, gx_jax)


def test_sharded_grad_at_matches_jax(mesh_1d_2):
    def nb_loss(x):
        y = x.at[:, 1:3].set(
            tensor_from_jax(jnp.array([[7.0, 8.0], [9.0, 10.0]], dtype=jnp.float32))
        )
        return nb.reduce_sum(y * y)

    def jax_loss(x):
        y = x.at[:, 1:3].set(jnp.array([[7.0, 8.0], [9.0, 10.0]], dtype=jnp.float32))
        return jnp.sum(y * y)

    x_jax = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)
    x_sharded = tensor_from_jax(x_jax).shard(mesh_1d_2, P("dp", None))

    gx_nb = nb.grad(nb_loss)(x_sharded)
    gx_jax = jax.grad(jax_loss)(x_jax)
    assert_allclose(gx_nb.gather(), gx_jax)


def test_sharded_grad_setitem_matches_jax(mesh_1d_2):
    update = tensor_from_jax(jnp.array([[7.0, 8.0], [9.0, 10.0]], dtype=jnp.float32))

    def nb_loss(x):
        y = x + 0.0
        y[:, 1:3] = update
        return nb.reduce_sum(y * y)

    def jax_loss(x):
        y = x + 0.0
        y = y.at[:, 1:3].set(jnp.array([[7.0, 8.0], [9.0, 10.0]], dtype=jnp.float32))
        return jnp.sum(y * y)

    x_jax = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)
    x_sharded = tensor_from_jax(x_jax).shard(mesh_1d_2, P("dp", None))

    gx_nb = nb.grad(nb_loss)(x_sharded)
    gx_jax = jax.grad(jax_loss)(x_jax)
    assert_allclose(gx_nb.gather(), gx_jax)
