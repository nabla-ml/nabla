# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nabla as nb
from tests.unit.common import (
    DeviceMesh,
    cleanup_caches,
    replicated,
    tensor_from_jax,
    to_jax,
)


def _close(nb_val, jax_val, rtol=8e-4, atol=8e-4):
    np.testing.assert_allclose(to_jax(nb_val), jax_val, rtol=rtol, atol=atol)


def _jax_conv2d(
    x,
    w,
    *,
    stride=(1, 1),
    dilation=(1, 1),
    padding=(0, 0, 0, 0),
    groups=1,
    bias=None,
):
    p_t, p_b, p_l, p_r = padding
    y = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=w,
        window_strides=stride,
        padding=((p_t, p_b), (p_l, p_r)),
        rhs_dilation=dilation,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=groups,
    )
    if bias is not None:
        y = y + bias
    return y


class TestConvolution2D:
    def test_conv2d_forward_matches_jax(self):
        cleanup_caches()
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 6, 5, 4), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(1), (3, 2, 4, 7), dtype=jnp.float32)
        b = jax.random.normal(jax.random.PRNGKey(2), (7,), dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        w_nb = tensor_from_jax(w)
        b_nb = tensor_from_jax(b)

        kwargs = {
            "stride": (2, 1),
            "dilation": (1, 1),
            "padding": (1, 0, 2, 1),
            "groups": 1,
        }

        y_nb = nb.conv2d(x_nb, w_nb, bias=b_nb, **kwargs)
        y_jax = _jax_conv2d(x, w, bias=b, **kwargs)

        _close(y_nb, y_jax)

    def test_conv2d_grouped_raises_clear_error(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(3), (2, 7, 6, 4), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(4), (3, 3, 2, 6), dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        w_nb = tensor_from_jax(w)

        with pytest.raises(ValueError, match="grouped mode"):
            _ = nb.conv2d(
                x_nb,
                w_nb,
                stride=(1, 2),
                dilation=(1, 1),
                padding=(1, 1, 0, 1),
                groups=2,
            )

    def test_conv2d_vjp_matches_jax(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(5), (2, 5, 5, 4), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(6), (3, 3, 4, 6), dtype=jnp.float32)
        b = jax.random.normal(jax.random.PRNGKey(7), (6,), dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        w_nb = tensor_from_jax(w)
        b_nb = tensor_from_jax(b)

        kwargs = {
            "stride": (1, 1),
            "dilation": (1, 1),
            "padding": (1, 1, 1, 1),
            "groups": 1,
        }

        def f_nb(xp, wp, bp):
            return nb.conv2d(xp, wp, bias=bp, **kwargs)

        def f_jax(xp, wp, bp):
            return _jax_conv2d(xp, wp, bias=bp, **kwargs)

        y_nb, vjp_nb = nb.vjp(f_nb, x_nb, w_nb, b_nb)
        cot_nb = nb.ones_like(y_nb)
        gx_nb, gw_nb, gb_nb = vjp_nb(cot_nb)

        y_jax, vjp_jax = jax.vjp(f_jax, x, w, b)
        gx_jax, gw_jax, gb_jax = vjp_jax(jnp.ones_like(y_jax))

        _close(y_nb, y_jax)
        _close(gx_nb, gx_jax)
        _close(gw_nb, gw_jax)
        _close(gb_nb, gb_jax)

    def test_conv2d_jvp_matches_jax(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(8), (1, 6, 7, 3), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(9), (3, 2, 3, 5), dtype=jnp.float32)
        b = jax.random.normal(jax.random.PRNGKey(10), (5,), dtype=jnp.float32)

        tx = jax.random.normal(jax.random.PRNGKey(11), x.shape, dtype=jnp.float32)
        tw = jax.random.normal(jax.random.PRNGKey(12), w.shape, dtype=jnp.float32)
        tb = jax.random.normal(jax.random.PRNGKey(13), b.shape, dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        w_nb = tensor_from_jax(w)
        b_nb = tensor_from_jax(b)
        tx_nb = tensor_from_jax(tx)
        tw_nb = tensor_from_jax(tw)
        tb_nb = tensor_from_jax(tb)

        kwargs = {
            "stride": (2, 1),
            "dilation": (1, 1),
            "padding": (1, 0, 1, 1),
            "groups": 1,
        }

        def f_nb(xp, wp, bp):
            return nb.conv2d(xp, wp, bias=bp, **kwargs)

        def f_jax(xp, wp, bp):
            return _jax_conv2d(xp, wp, bias=bp, **kwargs)

        out_nb, tan_nb = nb.jvp(f_nb, (x_nb, w_nb, b_nb), (tx_nb, tw_nb, tb_nb))
        out_jax, tan_jax = jax.jvp(f_jax, (x, w, b), (tx, tw, tb))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_conv2d_vjp_stride_gt_1_matches_jax(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(40), (2, 8, 7, 3), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(41), (3, 3, 3, 5), dtype=jnp.float32)
        b = jax.random.normal(jax.random.PRNGKey(42), (5,), dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        w_nb = tensor_from_jax(w)
        b_nb = tensor_from_jax(b)

        kwargs = {
            "stride": (2, 1),
            "dilation": (1, 1),
            "padding": (0, 0, 1, 1),
            "groups": 1,
        }

        def f_nb(xp, wp, bp):
            return nb.conv2d(xp, wp, bias=bp, **kwargs)

        def f_jax(xp, wp, bp):
            return _jax_conv2d(xp, wp, bias=bp, **kwargs)

        y_nb, vjp_nb = nb.vjp(f_nb, x_nb, w_nb, b_nb)
        gx_nb, gw_nb, gb_nb = vjp_nb(nb.ones_like(y_nb))

        y_jax, vjp_jax = jax.vjp(f_jax, x, w, b)
        gx_jax, gw_jax, gb_jax = vjp_jax(jnp.ones_like(y_jax))

        _close(y_nb, y_jax)
        _close(gx_nb, gx_jax)
        _close(gw_nb, gw_jax)
        _close(gb_nb, gb_jax)

    def test_conv2d_jvp_stride_gt_1_matches_jax(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(43), (1, 7, 5, 3), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(44), (3, 2, 3, 4), dtype=jnp.float32)
        tx = jax.random.normal(jax.random.PRNGKey(45), x.shape, dtype=jnp.float32)
        tw = jax.random.normal(jax.random.PRNGKey(46), w.shape, dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        w_nb = tensor_from_jax(w)
        tx_nb = tensor_from_jax(tx)
        tw_nb = tensor_from_jax(tw)

        kwargs = {
            "stride": (2, 1),
            "dilation": (1, 1),
            "padding": (1, 0, 0, 1),
            "groups": 1,
        }

        def f_nb(xp, wp):
            return nb.conv2d(xp, wp, **kwargs)

        def f_jax(xp, wp):
            return _jax_conv2d(xp, wp, **kwargs)

        out_nb, tan_nb = nb.jvp(f_nb, (x_nb, w_nb), (tx_nb, tw_nb))
        out_jax, tan_jax = jax.jvp(f_jax, (x, w), (tx, tw))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_conv2d_vmap_forward_matches_jax(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(54), (3, 2, 6, 5, 4), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(55), (3, 2, 4, 7), dtype=jnp.float32)
        b = jax.random.normal(jax.random.PRNGKey(56), (7,), dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        w_nb = tensor_from_jax(w)
        b_nb = tensor_from_jax(b)

        kwargs = {
            "stride": (2, 1),
            "dilation": (1, 1),
            "padding": (1, 0, 2, 1),
            "groups": 1,
        }

        f_nb = nb.vmap(lambda x_i: nb.conv2d(x_i, w_nb, bias=b_nb, **kwargs), in_axes=0, out_axes=0)
        y_nb = f_nb(x_nb)

        f_jax = jax.vmap(
            lambda x_i: _jax_conv2d(x_i, w, bias=b, **kwargs),
            in_axes=0,
            out_axes=0,
        )
        y_jax = f_jax(x)

        _close(y_nb, y_jax)

    def test_conv2d_vmap_vjp_matches_jax(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(57), (3, 2, 6, 5, 4), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(58), (3, 2, 4, 7), dtype=jnp.float32)
        b = jax.random.normal(jax.random.PRNGKey(59), (7,), dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        w_nb = tensor_from_jax(w)
        b_nb = tensor_from_jax(b)

        kwargs = {
            "stride": (2, 1),
            "dilation": (1, 1),
            "padding": (1, 0, 2, 1),
            "groups": 1,
        }

        f_nb = nb.vmap(
            lambda x_i: nb.conv2d(x_i, w_nb, bias=b_nb, **kwargs),
            in_axes=0,
            out_axes=0,
        )

        out_nb, vjp_fn = nb.vjp(f_nb, x_nb)
        (gx_nb,) = vjp_fn(nb.ones_like(out_nb))

        f_jax = jax.vmap(
            lambda x_i: _jax_conv2d(x_i, w, bias=b, **kwargs),
            in_axes=0,
            out_axes=0,
        )
        out_jax, vjp_jax = jax.vjp(f_jax, x)
        (gx_jax,) = vjp_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(gx_nb, gx_jax)

    def test_conv2d_vmap_jvp_matches_jax(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(66), (3, 2, 6, 5, 4), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(67), (3, 2, 4, 7), dtype=jnp.float32)
        b = jax.random.normal(jax.random.PRNGKey(68), (7,), dtype=jnp.float32)
        tx = jax.random.normal(jax.random.PRNGKey(69), x.shape, dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        w_nb = tensor_from_jax(w)
        b_nb = tensor_from_jax(b)
        tx_nb = tensor_from_jax(tx)

        kwargs = {
            "stride": (2, 1),
            "dilation": (1, 1),
            "padding": (1, 0, 2, 1),
            "groups": 1,
        }

        f_nb = nb.vmap(
            lambda x_i: nb.conv2d(x_i, w_nb, bias=b_nb, **kwargs),
            in_axes=0,
            out_axes=0,
        )
        f_jax = jax.vmap(
            lambda x_i: _jax_conv2d(x_i, w, bias=b, **kwargs),
            in_axes=0,
            out_axes=0,
        )

        out_nb, tan_nb = nb.jvp(f_nb, (x_nb,), (tx_nb,))
        out_jax, tan_jax = jax.jvp(f_jax, (x,), (tx,))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_conv2d_sharded_forward_jvp_matches_jax(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(70), (2, 6, 5, 4), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(71), (3, 2, 4, 7), dtype=jnp.float32)
        b = jax.random.normal(jax.random.PRNGKey(72), (7,), dtype=jnp.float32)
        tx = jax.random.normal(jax.random.PRNGKey(73), x.shape, dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        w_nb = tensor_from_jax(w)
        b_nb = tensor_from_jax(b)
        tx_nb = tensor_from_jax(tx)

        mesh = DeviceMesh("mesh_1", (1, 1), ("x", "y"))
        x_sharded = replicated(x_nb, mesh)

        kwargs = {
            "stride": (2, 1),
            "dilation": (1, 1),
            "padding": (1, 0, 2, 1),
            "groups": 1,
        }

        def f_nb(xp):
            return nb.reduce_sum(nb.conv2d(xp, w_nb, bias=b_nb, **kwargs))

        def f_jax(xp):
            return jnp.sum(_jax_conv2d(xp, w, bias=b, **kwargs))

        out_nb = f_nb(x_sharded)
        out_jax = f_jax(x)
        _close(out_nb, out_jax)

        out_nb_jvp, tan_nb = nb.jvp(f_nb, (x_sharded,), (tx_nb,))
        out_jax_jvp, tan_jax = jax.jvp(f_jax, (x,), (tx,))

        _close(out_nb_jvp, out_jax_jvp)
        _close(tan_nb, tan_jax)

    def test_conv2d_sharded_vjp_raises_current_device_mismatch(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(81), (2, 6, 5, 4), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(82), (3, 2, 4, 7), dtype=jnp.float32)
        b = jax.random.normal(jax.random.PRNGKey(83), (7,), dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        w_nb = tensor_from_jax(w)
        b_nb = tensor_from_jax(b)

        mesh = DeviceMesh("mesh_1", (1, 1), ("x", "y"))
        x_sharded = replicated(x_nb, mesh)

        kwargs = {
            "stride": (2, 1),
            "dilation": (1, 1),
            "padding": (1, 0, 2, 1),
            "groups": 1,
        }

        def f_nb(xp):
            return nb.reduce_sum(nb.conv2d(xp, w_nb, bias=b_nb, **kwargs))

        out_nb, vjp_nb = nb.vjp(f_nb, x_sharded)
        (gx_nb,) = vjp_nb(nb.ones_like(out_nb))

        with pytest.raises(ValueError, match="same device"):
            _ = to_jax(gx_nb)

    def test_conv2d_nested_vmap_jvp_vjp_matches_jax(self):
        cleanup_caches()
        x = jax.random.normal(
            jax.random.PRNGKey(90), (2, 3, 2, 6, 5, 4), dtype=jnp.float32
        )
        w = jax.random.normal(jax.random.PRNGKey(91), (3, 2, 4, 7), dtype=jnp.float32)
        b = jax.random.normal(jax.random.PRNGKey(92), (7,), dtype=jnp.float32)
        tx = jax.random.normal(jax.random.PRNGKey(93), x.shape, dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        w_nb = tensor_from_jax(w)
        b_nb = tensor_from_jax(b)
        tx_nb = tensor_from_jax(tx)

        kwargs = {
            "stride": (2, 1),
            "dilation": (1, 1),
            "padding": (1, 0, 2, 1),
            "groups": 1,
        }

        f_nb = nb.vmap(
            nb.vmap(
                lambda x_ij: nb.conv2d(x_ij, w_nb, bias=b_nb, **kwargs),
                in_axes=0,
                out_axes=0,
            ),
            in_axes=0,
            out_axes=0,
        )
        f_jax = jax.vmap(
            jax.vmap(
                lambda x_ij: _jax_conv2d(x_ij, w, bias=b, **kwargs),
                in_axes=0,
                out_axes=0,
            ),
            in_axes=0,
            out_axes=0,
        )

        out_nb, tan_nb = nb.jvp(f_nb, (x_nb,), (tx_nb,))
        out_jax, tan_jax = jax.jvp(f_jax, (x,), (tx,))
        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

        out_nb, vjp_nb = nb.vjp(f_nb, x_nb)
        (gx_nb,) = vjp_nb(nb.ones_like(out_nb))
        out_jax, vjp_jax = jax.vjp(f_jax, x)
        (gx_jax,) = vjp_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(gx_nb, gx_jax)

    def test_conv2d_invalid_params_raise_clear_errors(self):
        cleanup_caches()
        x = tensor_from_jax(jax.random.normal(jax.random.PRNGKey(94), (1, 5, 5, 3)))
        w = tensor_from_jax(jax.random.normal(jax.random.PRNGKey(95), (3, 3, 3, 4)))

        with pytest.raises(ValueError, match="stride values must be > 0"):
            _ = nb.conv2d(x, w, stride=(0, 1))

        with pytest.raises(ValueError, match="padding values must be >= 0"):
            _ = nb.conv2d(x, w, padding=(1, -1, 0, 0))

    def test_conv2d_channel_and_bias_mismatch_raise_clear_errors(self):
        cleanup_caches()
        x = tensor_from_jax(jax.random.normal(jax.random.PRNGKey(96), (1, 5, 5, 3)))
        w_bad_cin = tensor_from_jax(
            jax.random.normal(jax.random.PRNGKey(97), (3, 3, 2, 4))
        )
        w = tensor_from_jax(jax.random.normal(jax.random.PRNGKey(98), (3, 3, 3, 4)))
        b_bad = tensor_from_jax(jax.random.normal(jax.random.PRNGKey(99), (5,)))

        with pytest.raises(ValueError, match="Input channels"):
            _ = nb.conv2d(x, w_bad_cin)

        with pytest.raises(ValueError, match="bias size"):
            _ = nb.conv2d(x, w, bias=b_bad)


class TestConvolution2DTranspose:
    def test_conv2d_dilation_nonunit_raises_clear_error(self):
        cleanup_caches()
        x = tensor_from_jax(jax.random.normal(jax.random.PRNGKey(20), (1, 6, 6, 2)))
        w = tensor_from_jax(jax.random.normal(jax.random.PRNGKey(21), (3, 3, 2, 4)))

        with pytest.raises(ValueError, match="dilation"):
            _ = nb.conv2d(x, w, dilation=(2, 1))

    def test_conv2d_transpose_output_shape(self):
        cleanup_caches()
        x = tensor_from_jax(jax.random.normal(jax.random.PRNGKey(22), (2, 4, 5, 3)))
        w = tensor_from_jax(jax.random.normal(jax.random.PRNGKey(23), (3, 2, 7, 3)))

        y = nb.conv2d_transpose(
            x,
            w,
            stride=(2, 1),
            dilation=(1, 1),
            padding=(1, 0, 2, 1),
            output_paddings=(0, 0),
        )

        expected_h = (4 - 1) * 2 - 1 - 0 + (3 - 1) + 0 + 1
        expected_w = (5 - 1) * 1 - 2 - 1 + (2 - 1) + 0 + 1
        assert tuple(int(d) for d in y.shape) == (2, expected_h, expected_w, 7)

    def test_conv2d_transpose_jvp_matches_finite_difference(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(24), (1, 4, 4, 2), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(25), (3, 3, 5, 2), dtype=jnp.float32)

        tx = jax.random.normal(jax.random.PRNGKey(26), x.shape, dtype=jnp.float32)
        tw = jax.random.normal(jax.random.PRNGKey(27), w.shape, dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        w_nb = tensor_from_jax(w)
        tx_nb = tensor_from_jax(tx)
        tw_nb = tensor_from_jax(tw)

        kwargs = {
            "stride": (1, 1),
            "dilation": (1, 1),
            "padding": (1, 1, 1, 1),
            "output_paddings": (0, 0),
        }

        def f_nb(xp, wp):
            return nb.conv2d_transpose(xp, wp, **kwargs)

        _, tan_nb = nb.jvp(f_nb, (x_nb, w_nb), (tx_nb, tw_nb))

        eps = 5e-3
        f_plus = f_nb(
            x_nb + eps * tx_nb,
            w_nb + eps * tw_nb,
        )
        f_base = f_nb(x_nb, w_nb)
        tan_fd = (f_plus - f_base) / eps

        np.testing.assert_allclose(
            np.asarray(to_jax(tan_nb)),
            np.asarray(to_jax(tan_fd)),
            rtol=1.2e-1,
            atol=5e-2,
        )

    def test_conv2d_transpose_vjp_matches_finite_difference(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(47), (1, 4, 5, 2), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(48), (3, 2, 6, 2), dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        w_nb = tensor_from_jax(w)

        kwargs = {
            "stride": (2, 1),
            "dilation": (1, 1),
            "padding": (1, 0, 1, 0),
            "output_paddings": (0, 0),
        }

        def f_nb(xp, wp):
            return nb.conv2d_transpose(xp, wp, **kwargs)

        y_nb, vjp_nb = nb.vjp(f_nb, x_nb, w_nb)
        cot = tensor_from_jax(jax.random.normal(jax.random.PRNGKey(50), to_jax(y_nb).shape, dtype=jnp.float32))
        gx_nb, gw_nb = vjp_nb(cot)

        cot_np = np.asarray(to_jax(cot))
        gx_np = np.asarray(to_jax(gx_nb))
        gw_np = np.asarray(to_jax(gw_nb))

        eps = 1e-3

        def scalar_loss(x_in, w_in):
            y = np.asarray(to_jax(f_nb(tensor_from_jax(x_in), tensor_from_jax(w_in))))
            return float(np.sum(y * cot_np))

        # Check directional derivatives against VJP for each primal.
        vx = jax.random.normal(jax.random.PRNGKey(51), x.shape, dtype=jnp.float32)
        vw = jax.random.normal(jax.random.PRNGKey(52), w.shape, dtype=jnp.float32)

        fd_x = (scalar_loss(x + eps * vx, w) - scalar_loss(x - eps * vx, w)) / (2 * eps)
        fd_w = (scalar_loss(x, w + eps * vw) - scalar_loss(x, w - eps * vw)) / (2 * eps)

        ad_x = float(np.sum(gx_np * np.asarray(vx)))
        ad_w = float(np.sum(gw_np * np.asarray(vw)))

        np.testing.assert_allclose(ad_x, fd_x, rtol=1.5e-1, atol=8e-2)
        np.testing.assert_allclose(ad_w, fd_w, rtol=1.5e-1, atol=8e-2)

    @pytest.mark.xfail(
        reason="vmapped conv2d_transpose JVP does not match finite differences in current backend"
    )
    def test_conv2d_transpose_vmap_jvp_matches_finite_difference(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(74), (3, 1, 4, 4, 2), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(75), (3, 3, 5, 2), dtype=jnp.float32)
        tx = jax.random.normal(jax.random.PRNGKey(76), x.shape, dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        w_nb = tensor_from_jax(w)
        tx_nb = tensor_from_jax(tx)

        kwargs = {
            "stride": (1, 1),
            "dilation": (1, 1),
            "padding": (1, 1, 1, 1),
            "output_paddings": (0, 0),
        }

        f_nb = nb.vmap(
            lambda x_i: nb.conv2d_transpose(x_i, w_nb, **kwargs),
            in_axes=0,
            out_axes=0,
        )

        _, tan_nb = nb.jvp(f_nb, (x_nb,), (tx_nb,))
        eps = 5e-3
        tan_fd = (f_nb(x_nb + eps * tx_nb) - f_nb(x_nb)) / eps

        np.testing.assert_allclose(
            np.asarray(to_jax(tan_nb)),
            np.asarray(to_jax(tan_fd)),
            rtol=1.5e-1,
            atol=6e-2,
        )

    def test_conv2d_transpose_vmap_vjp_x_matches_finite_difference(self):
        cleanup_caches()
        x = jax.random.normal(jax.random.PRNGKey(77), (3, 1, 4, 5, 2), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(78), (3, 2, 6, 2), dtype=jnp.float32)

        x_nb = tensor_from_jax(x)
        w_nb = tensor_from_jax(w)

        kwargs = {
            "stride": (2, 1),
            "dilation": (1, 1),
            "padding": (1, 0, 1, 0),
            "output_paddings": (0, 0),
        }

        f_nb = nb.vmap(
            lambda x_i: nb.conv2d_transpose(x_i, w_nb, **kwargs),
            in_axes=0,
            out_axes=0,
        )

        y_nb, vjp_nb = nb.vjp(f_nb, x_nb)
        cot = tensor_from_jax(
            jax.random.normal(jax.random.PRNGKey(79), to_jax(y_nb).shape, dtype=jnp.float32)
        )
        (gx_nb,) = vjp_nb(cot)

        cot_np = np.asarray(to_jax(cot))
        gx_np = np.asarray(to_jax(gx_nb))
        eps = 1e-3

        def scalar_loss(x_in):
            y = np.asarray(to_jax(f_nb(tensor_from_jax(x_in))))
            return float(np.sum(y * cot_np))

        vx = jax.random.normal(jax.random.PRNGKey(80), x.shape, dtype=jnp.float32)
        fd_x = (scalar_loss(x + eps * vx) - scalar_loss(x - eps * vx)) / (2 * eps)
        ad_x = float(np.sum(gx_np * np.asarray(vx)))

        np.testing.assert_allclose(ad_x, fd_x, rtol=1.8e-1, atol=1.0e-1)

    def test_conv2d_transpose_output_padding_nonzero_raises(self):
        cleanup_caches()
        x = tensor_from_jax(jax.random.normal(jax.random.PRNGKey(31), (1, 4, 4, 2)))
        w = tensor_from_jax(jax.random.normal(jax.random.PRNGKey(32), (3, 3, 4, 2)))

        with pytest.raises(ValueError, match="output_paddings"):
            _ = nb.conv2d_transpose(x, w, output_paddings=(1, 0))

    def test_conv2d_transpose_invalid_params_channel_and_bias_raise(self):
        cleanup_caches()
        x = tensor_from_jax(jax.random.normal(jax.random.PRNGKey(101), (1, 4, 4, 2)))
        w_bad_cin = tensor_from_jax(
            jax.random.normal(jax.random.PRNGKey(102), (3, 3, 4, 3))
        )
        w = tensor_from_jax(jax.random.normal(jax.random.PRNGKey(103), (3, 3, 4, 2)))
        b_bad = tensor_from_jax(jax.random.normal(jax.random.PRNGKey(104), (5,)))

        with pytest.raises(ValueError, match="stride values must be > 0"):
            _ = nb.conv2d_transpose(x, w, stride=(1, 0))

        with pytest.raises(ValueError, match="padding values must be >= 0"):
            _ = nb.conv2d_transpose(x, w, padding=(0, -1, 0, 0))

        with pytest.raises(ValueError, match="Input channels"):
            _ = nb.conv2d_transpose(x, w_bad_cin)

        with pytest.raises(ValueError, match="bias size"):
            _ = nb.conv2d_transpose(x, w, bias=b_bad)
