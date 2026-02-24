# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nabla as nb
from tests.unit.common import cleanup_caches, tensor_from_jax, to_jax


def _close(nb_val, jax_val, rtol=1e-3, atol=1e-3):
    np.testing.assert_allclose(to_jax(nb_val), jax_val, rtol=rtol, atol=atol)


def _jax_avg_pool2d(x, *, kernel_size, stride=None, padding=(0, 0, 0, 0)):
    k_h, k_w = kernel_size
    s_h, s_w = stride if stride is not None else kernel_size
    p_t, p_b, p_l, p_r = padding

    window = (1, k_h, k_w, 1)
    strides = (1, s_h, s_w, 1)
    pad = ((0, 0), (p_t, p_b), (p_l, p_r), (0, 0))

    summed = jax.lax.reduce_window(x, 0.0, jax.lax.add, window, strides, pad)
    return summed / float(k_h * k_w)


def _jax_max_pool2d(x, *, kernel_size, stride=None, padding=(0, 0, 0, 0)):
    k_h, k_w = kernel_size
    s_h, s_w = stride if stride is not None else kernel_size
    p_t, p_b, p_l, p_r = padding

    window = (1, k_h, k_w, 1)
    strides = (1, s_h, s_w, 1)
    pad = ((0, 0), (p_t, p_b), (p_l, p_r), (0, 0))

    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, window, strides, pad)


class TestPooling2D:
    def test_avg_pool2d_forward_matches_jax(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 7, 6, 3), dtype=jnp.float32)
        x_nb = tensor_from_jax(x)

        kwargs = {"kernel_size": (3, 2), "stride": (2, 1), "padding": (1, 0, 2, 1)}
        y_nb = nb.avg_pool2d(x_nb, **kwargs)
        y_jax = _jax_avg_pool2d(x, **kwargs)

        _close(y_nb, y_jax)

    def test_max_pool2d_forward_matches_jax(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 5, 4), dtype=jnp.float32)
        x_nb = tensor_from_jax(x)

        kwargs = {"kernel_size": (2, 3), "stride": (2, 1), "padding": (1, 1, 0, 1)}
        y_nb = nb.max_pool2d(x_nb, **kwargs)
        y_jax = _jax_max_pool2d(x, **kwargs)

        _close(y_nb, y_jax)

    def test_avg_pool2d_vjp_jvp_match_jax(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(2), (1, 7, 6, 3), dtype=jnp.float32)
        tx = jax.random.normal(jax.random.PRNGKey(3), x.shape, dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        tx_nb = tensor_from_jax(tx)

        kwargs = {"kernel_size": (3, 2), "stride": (2, 1), "padding": (1, 0, 1, 1)}

        def f_nb(xp):
            return nb.avg_pool2d(xp, **kwargs)

        def f_jax(xp):
            return _jax_avg_pool2d(xp, **kwargs)

        out_nb, tan_nb = nb.jvp(f_nb, (x_nb,), (tx_nb,))
        out_jax, tan_jax = jax.jvp(f_jax, (x,), (tx,))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

        y_nb, vjp_nb = nb.vjp(f_nb, x_nb)
        (gx_nb,) = vjp_nb(nb.ones_like(y_nb))
        y_jax, vjp_jax = jax.vjp(f_jax, x)
        (gx_jax,) = vjp_jax(jnp.ones_like(y_jax))

        _close(y_nb, y_jax)
        _close(gx_nb, gx_jax)

    def test_max_pool2d_vjp_jvp_match_jax(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(4), (1, 7, 6, 3), dtype=jnp.float32)
        # Break ties to make max-gradient comparisons stable.
        x = x + jnp.arange(x.size, dtype=x.dtype).reshape(x.shape) * 1e-4
        tx = jax.random.normal(jax.random.PRNGKey(5), x.shape, dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        tx_nb = tensor_from_jax(tx)

        kwargs = {"kernel_size": (2, 2), "stride": (2, 1), "padding": (1, 0, 1, 0)}

        def f_nb(xp):
            return nb.max_pool2d(xp, **kwargs)

        def f_jax(xp):
            return _jax_max_pool2d(xp, **kwargs)

        out_nb, tan_nb = nb.jvp(f_nb, (x_nb,), (tx_nb,))
        out_jax, tan_jax = jax.jvp(f_jax, (x,), (tx,))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

        y_nb, vjp_nb = nb.vjp(f_nb, x_nb)
        (gx_nb,) = vjp_nb(nb.ones_like(y_nb))
        y_jax, vjp_jax = jax.vjp(f_jax, x)
        (gx_jax,) = vjp_jax(jnp.ones_like(y_jax))

        _close(y_nb, y_jax)
        _close(gx_nb, gx_jax)

    def test_avg_pool2d_nested_vmap_matches_jax(self):
        cleanup_caches()
        x = jax.random.normal(
            jax.random.PRNGKey(6), (2, 3, 1, 7, 6, 2), dtype=jnp.float32
        )
        tx = jax.random.normal(jax.random.PRNGKey(7), x.shape, dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        tx_nb = tensor_from_jax(tx)

        kwargs = {"kernel_size": (3, 2), "stride": (2, 1), "padding": (1, 0, 1, 1)}

        f_nb = nb.vmap(
            nb.vmap(lambda x_ij: nb.avg_pool2d(x_ij, **kwargs), in_axes=0, out_axes=0),
            in_axes=0,
            out_axes=0,
        )
        f_jax = jax.vmap(
            jax.vmap(lambda x_ij: _jax_avg_pool2d(x_ij, **kwargs), in_axes=0, out_axes=0),
            in_axes=0,
            out_axes=0,
        )

        out_nb, tan_nb = nb.jvp(f_nb, (x_nb,), (tx_nb,))
        out_jax, tan_jax = jax.jvp(f_jax, (x,), (tx,))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_pool2d_invalid_params_raise(self):
        cleanup_caches()
        x = tensor_from_jax(jax.random.normal(jax.random.PRNGKey(8), (1, 6, 6, 3)))

        with pytest.raises(ValueError, match="kernel_size values must be > 0"):
            _ = nb.avg_pool2d(x, kernel_size=(0, 2))

        with pytest.raises(ValueError, match="stride values must be > 0"):
            _ = nb.max_pool2d(x, kernel_size=2, stride=(1, 0))

        with pytest.raises(ValueError, match="padding values must be >= 0"):
            _ = nb.avg_pool2d(x, kernel_size=2, padding=(1, -1, 0, 0))

        with pytest.raises(ValueError, match=r"dilation=\(1, 1\)"):
            _ = nb.max_pool2d(x, kernel_size=2, dilation=(2, 1))
