# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ensure_tensor
from .convolution import (
    _flatten_batch_dims_into_n,
    _normalize_2tuple,
    _normalize_4padding,
    _require_all_nonnegative,
    _require_all_positive_2tuple,
    _restore_batch_dims_from_n,
)

if TYPE_CHECKING:
    from ..core.tensor import Tensor


def _pool2d_validate_and_normalize(
    x: "Tensor",
    *,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None,
    padding: int | tuple[int, int] | tuple[int, int, int, int],
    dilation: int | tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int], tuple[int, int]]:
    if len(x.shape) != 4:
        raise ValueError(
            f"pool2d expects rank-4 tensors, got x={tuple(int(d) for d in x.shape)}"
        )

    kernel = _normalize_2tuple(kernel_size, "kernel_size")
    stride_norm = _normalize_2tuple(stride if stride is not None else kernel, "stride")
    padding_norm = _normalize_4padding(padding)
    dilation_norm = _normalize_2tuple(dilation, "dilation")

    _require_all_positive_2tuple(kernel, "kernel_size")
    _require_all_positive_2tuple(stride_norm, "stride")
    _require_all_positive_2tuple(dilation_norm, "dilation")
    _require_all_nonnegative(padding_norm, "padding")

    if dilation_norm != (1, 1):
        raise ValueError("pool2d currently supports only dilation=(1, 1)")

    return kernel, stride_norm, padding_norm, dilation_norm


def _pool2d_impl(
    x: "Tensor",
    *,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None,
    padding: int | tuple[int, int] | tuple[int, int, int, int],
    dilation: int | tuple[int, int],
    mode: str,
) -> "Tensor":
    from .creation import full
    from .reduction import reduce_max, reduce_sum
    from .view import pad, slice_tensor, stack

    kernel, stride_norm, padding_norm, _ = _pool2d_validate_and_normalize(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    k_h, k_w = kernel
    s_h, s_w = stride_norm
    p_t, p_b, p_l, p_r = padding_norm

    x_flat, batch_shape, n_orig = _flatten_batch_dims_into_n(x)
    n, h_base, w_base, c = (int(d) for d in x_flat.shape)

    if mode == "avg":
        if p_t or p_b or p_l or p_r:
            x_work = pad(x_flat, paddings=[(0, 0), (p_t, p_b), (p_l, p_r), (0, 0)])
        else:
            x_work = x_flat
        _, h_in, w_in, _ = (int(d) for d in x_work.shape)
        h_out = (h_in - k_h) // s_h + 1
        w_out = (w_in - k_w) // s_w + 1
    else:
        h_out = (h_base + p_t + p_b - k_h) // s_h + 1
        w_out = (w_base + p_l + p_r - k_w) // s_w + 1

    if h_out <= 0 or w_out <= 0:
        raise ValueError(
            f"Non-positive pool2d output size H={h_out}, W={w_out}. "
            f"Input={(h_base, w_base)}, kernel={(k_h, k_w)}, stride={stride_norm}, padding={padding_norm}."
        )

    rows = []
    for out_h in range(h_out):
        cols = []
        h_start = out_h * s_h
        for out_w in range(w_out):
            w_start = out_w * s_w
            if mode == "avg":
                window = slice_tensor(
                    x_work,
                    start=[0, h_start, w_start, 0],
                    size=[n, k_h, k_w, c],
                )
                pooled = reduce_sum(window, axis=[1, 2], keepdims=False) / float(k_h * k_w)
            elif mode == "max":
                h0 = max(0, h_start - p_t)
                h1 = min(h_base, h_start - p_t + k_h)
                w0 = max(0, w_start - p_l)
                w1 = min(w_base, w_start - p_l + k_w)

                if h0 >= h1 or w0 >= w1:
                    pooled = full(
                        (n, c),
                        float("-inf"),
                        dtype=x_flat.dtype,
                        device=x_flat.device,
                    )
                else:
                    window = slice_tensor(
                        x_flat,
                        start=[0, h0, w0, 0],
                        size=[n, h1 - h0, w1 - w0, c],
                    )
                    pooled = reduce_max(window, axis=[1, 2], keepdims=False)
            else:
                raise ValueError(f"Unsupported pool2d mode: {mode}")
            cols.append(pooled)
        rows.append(stack(cols, axis=1))

    y_flat = stack(rows, axis=1)
    return _restore_batch_dims_from_n(y_flat, batch_shape, n_orig)


def avg_pool2d(
    x: "Tensor",
    *,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
    dilation: int | tuple[int, int] = (1, 1),
) -> "Tensor":
    x = ensure_tensor(x)
    return _pool2d_impl(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mode="avg",
    )


def max_pool2d(
    x: "Tensor",
    *,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
    dilation: int | tuple[int, int] = (1, 1),
) -> "Tensor":
    x = ensure_tensor(x)
    return _pool2d_impl(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mode="max",
    )


__all__ = ["avg_pool2d", "max_pool2d"]
