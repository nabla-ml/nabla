# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import ops

from .base import Operation, OpArgs, OpKwargs, OpResult, OpTensorValues, ensure_tensor

if TYPE_CHECKING:
    from ..core.tensor import Tensor


# ---------------------------------------------------------------------------
# Parameter normalization helpers
# ---------------------------------------------------------------------------


def _normalize_2tuple(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"{name} must be int or tuple/list of length 2, got {value}")


def _normalize_4padding(
    padding: int | tuple[int, int] | tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if isinstance(padding, int):
        p = int(padding)
        return (p, p, p, p)
    if isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            ph, pw = int(padding[0]), int(padding[1])
            return (ph, ph, pw, pw)
        if len(padding) == 4:
            return tuple(int(v) for v in padding)
    raise ValueError(
        "padding must be int, (pad_h, pad_w), or "
        "(pad_top, pad_bottom, pad_left, pad_right)"
    )


def _normalize_output_paddings(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(
        f"output_paddings must be int or tuple/list of length 2, got {value}"
    )


def _require_all_positive_2tuple(value: tuple[int, int], name: str) -> None:
    if value[0] <= 0 or value[1] <= 0:
        raise ValueError(f"{name} values must be > 0, got {value}")


def _require_all_nonnegative(value: tuple[int, ...], name: str) -> None:
    if any(v < 0 for v in value):
        raise ValueError(f"{name} values must be >= 0, got {value}")


# ---------------------------------------------------------------------------
# Backend workaround: spatial dilation via reshape+pad+slice
# ---------------------------------------------------------------------------


def _dilate_spatial_dims01(x: "Tensor", dilation_h: int, dilation_w: int) -> "Tensor":
    """Insert (dilation-1) zeros between elements in spatial dims 0 and 1.

    Input  shape: (H,  W,  D1, D2)
    Output shape: (H', W', D1, D2)   where H' = (H-1)*dh+1, W' = (W-1)*dw+1

    Used because MAX conv2d only supports dilation=(1,1); we explicitly dilate
    the cotangent tensor instead so that the weight-gradient conv can proceed
    with dilation=(1,1).
    """
    if dilation_h == 1 and dilation_w == 1:
        return x

    from .view import pad, reshape
    from .view.shape import slice_tensor

    H, W, D1, D2 = (int(d) for d in x.shape)

    if dilation_w > 1:
        x = reshape(x, (H, W, 1, D1 * D2))
        x = pad(x, paddings=[(0, 0), (0, 0), (0, dilation_w - 1), (0, 0)])
        x = reshape(x, (H, W * dilation_w, D1 * D2))
        W_new = (W - 1) * dilation_w + 1
        x = slice_tensor(x, start=[0, 0, 0], size=[H, W_new, D1 * D2])
        x = reshape(x, (H, W_new, D1, D2))
        W = W_new

    if dilation_h > 1:
        x = reshape(x, (H, 1, W * D1 * D2))
        x = pad(x, paddings=[(0, 0), (0, dilation_h - 1), (0, 0)])
        x = reshape(x, (H * dilation_h, W * D1 * D2))
        H_new = (H - 1) * dilation_h + 1
        x = slice_tensor(x, start=[0, 0], size=[H_new, W * D1 * D2])
        x = reshape(x, (H_new, W, D1, D2))

    return x


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def _conv2d_out_hw(
    h_in: int,
    w_in: int,
    k_h: int,
    k_w: int,
    stride: tuple[int, int],
    dilation: tuple[int, int],
    padding: tuple[int, int, int, int],
) -> tuple[int, int]:
    s_h, s_w = stride
    d_h, d_w = dilation
    p_t, p_b, p_l, p_r = padding
    h_out = (h_in + p_t + p_b - d_h * (k_h - 1) - 1) // s_h + 1
    w_out = (w_in + p_l + p_r - d_w * (k_w - 1) - 1) // s_w + 1
    if h_out <= 0 or w_out <= 0:
        raise ValueError(
            f"Non-positive conv2d output size H={h_out}, W={w_out}. "
            f"Input={(h_in, w_in)}, kernel={(k_h, k_w)}, stride={stride}, "
            f"dilation={dilation}, padding={padding}."
        )
    return h_out, w_out


def _conv2d_transpose_out_hw(
    h_in: int,
    w_in: int,
    k_h: int,
    k_w: int,
    stride: tuple[int, int],
    dilation: tuple[int, int],
    padding: tuple[int, int, int, int],
    output_paddings: tuple[int, int],
) -> tuple[int, int]:
    s_h, s_w = stride
    d_h, d_w = dilation
    p_t, p_b, p_l, p_r = padding
    op_h, op_w = output_paddings
    h_out = (h_in - 1) * s_h - p_t - p_b + d_h * (k_h - 1) + op_h + 1
    w_out = (w_in - 1) * s_w - p_l - p_r + d_w * (k_w - 1) + op_w + 1
    if h_out <= 0 or w_out <= 0:
        raise ValueError(
            f"Non-positive conv2d_transpose output size H={h_out}, W={w_out}."
        )
    return h_out, w_out


def _split_trailing_nhwc(shape: tuple[int, ...], op_name: str) -> tuple[tuple[int, ...], int, int, int]:
    if len(shape) < 4:
        raise ValueError(f"{op_name} expects tensor rank >= 4, got shape={shape}")
    prefix = tuple(shape[:-4])
    n, h, w, c = (int(v) for v in shape[-4:])
    return prefix, n, h, w, c


def _flatten_batch_dims_into_n(x: "Tensor") -> tuple["Tensor", tuple[int, ...], int]:
    """Convert [B..., N, H, W, C] with batch_dims>0 into [N_flat, H, W, C].

    Returns (flattened_tensor, batch_shape, n_orig).
    If x.batch_dims == 0, returns (x, (), N).
    """
    if x.batch_dims == 0:
        return x, (), int(x.shape[0])

    from .view import move_axis_from_batch_dims, reshape

    y = x
    for _ in range(x.batch_dims):
        y = move_axis_from_batch_dims(y, batch_axis=-1, logical_destination=0)

    y_shape = tuple(int(d) for d in y.shape)
    b = x.batch_dims
    batch_shape = y_shape[:b]
    n_orig = int(y_shape[b])
    n_flat = 1
    for dim in batch_shape:
        n_flat *= int(dim)
    n_flat *= n_orig

    y = reshape(y, (n_flat, *y_shape[b + 1 :]))
    return y, batch_shape, n_orig


def _restore_batch_dims_from_n(
    x_flat: "Tensor", batch_shape: tuple[int, ...], n_orig: int
) -> "Tensor":
    """Inverse of _flatten_batch_dims_into_n for NHWC tensors."""
    if not batch_shape:
        return x_flat

    from .view import incr_batch_dims, reshape

    shape = tuple(int(d) for d in x_flat.shape)
    y = reshape(x_flat, (*batch_shape, int(n_orig), *shape[1:]))
    for _ in batch_shape:
        y = incr_batch_dims(y)
    return y


# ---------------------------------------------------------------------------
# Conv2D
# ---------------------------------------------------------------------------


class Conv2DOp(Operation):
    @property
    def name(self) -> str:
        return "conv2d"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None

        x_shape = tuple(int(d) for d in x.shape)
        prefix_shape = x_shape[:-4]
        x_for_conv = x
        n_orig = h_in = w_in = c_in = None
        flat_batch = None

        if prefix_shape:
            n_orig, h_in, w_in, c_in = x_shape[-4:]
            flat_batch = 1
            for dim in prefix_shape:
                flat_batch *= int(dim)
            x_for_conv = ops.reshape(x, (flat_batch * n_orig, h_in, w_in, c_in))

        call_kwargs: dict[str, Any] = {
            "stride": tuple(kwargs["stride"]),
            "dilation": tuple(kwargs["dilation"]),
            "padding": tuple(kwargs["padding"]),
            "groups": int(kwargs.get("groups", 1)),
        }
        if bias is not None:
            call_kwargs["bias"] = bias

        y = ops.conv2d(x=x_for_conv, filter=weight, **call_kwargs)
        if prefix_shape:
            n_flat, h_out, w_out, c_out = (int(d) for d in y.shape)
            expected_n_flat = int(flat_batch) * int(n_orig)
            if n_flat != expected_n_flat:
                raise RuntimeError(
                    f"{self.name}: unexpected flattened batch size {n_flat}, expected {expected_n_flat}"
                )
            y = ops.reshape(y, (*prefix_shape, int(n_orig), h_out, w_out, c_out))

        return [y]

    def compute_physical_shape(
        self, args: OpArgs, kwargs: OpKwargs, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        from ..core.sharding import spmd

        x = args[0]
        w = args[1]
        stride = tuple(kwargs["stride"])
        dilation = tuple(kwargs["dilation"])
        padding = tuple(kwargs["padding"])
        groups = int(kwargs.get("groups", 1))

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            sx = x.physical_local_shape_ints(i if i < x.num_shards else 0)
            sw = w.physical_local_shape_ints(i if i < w.num_shards else 0)
            if sx is None or sw is None:
                raise RuntimeError(f"Could not determine physical shape for {self.name}")
            prefix, n, h_in, w_in, _ = _split_trailing_nhwc(sx, self.name)
            k_h, k_w, c_in_w, c_out = sw
            c_in = int(sx[-1])
            expected_c_in = int(c_in_w) * groups
            if c_in != expected_c_in:
                raise ValueError(
                    f"Input channels ({c_in}) must match filter channels*groups ({expected_c_in})"
                )
            h_out, w_out = _conv2d_out_hw(h_in, w_in, k_h, k_w, stride, dilation, padding)
            shapes.append((*prefix, n, h_out, w_out, c_out))

        dtypes, devices = self._build_shard_metadata(x, mesh, num_shards)
        return shapes, dtypes, devices

    def sharding_rule(self, input_shapes, output_shapes, **kwargs) -> Any:
        from ..core.sharding.propagation import OpShardingRuleTemplate

        in_x = {0: ["n"], 1: [], 2: [], 3: ["ci"]}
        in_w = {0: [], 1: [], 2: ["ci"], 3: ["co"]}
        out_y = {0: ["n"], 1: [], 2: [], 3: ["co"]}
        return OpShardingRuleTemplate([in_x, in_w], [out_y]).instantiate(
            input_shapes, output_shapes
        )

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        from . import conv2d, conv2d_transpose
        from .reduction import reduce_sum
        from .view import pad, permute
        from .view.shape import slice_tensor

        x = primals[0]
        w = primals[1]
        bias = primals[2] if len(primals) > 2 else None
        cot = cotangents[0]

        x_for_vjp, batch_shape_x, n_orig_x = _flatten_batch_dims_into_n(x)
        cot_for_vjp, batch_shape_cot, n_orig_cot = _flatten_batch_dims_into_n(cot)
        if batch_shape_x != batch_shape_cot or n_orig_x != n_orig_cot:
            raise ValueError(
                f"conv2d VJP batch_dims mismatch: x has {batch_shape_x} with N={n_orig_x}, "
                f"cot has {batch_shape_cot} with N={n_orig_cot}"
            )

        stride = tuple(kwargs["stride"])
        dilation = tuple(kwargs["dilation"])
        p_t, p_b, p_l, p_r = tuple(kwargs["padding"])

        n, h_in, w_in, c_in = (int(d) for d in x_for_vjp.shape)
        k_h, k_w, _, _ = (int(d) for d in w.shape)
        _, h_out, w_out, _ = (int(d) for d in cot_for_vjp.shape)
        s_h, s_w = stride
        d_h, d_w = dilation

        h_padded = h_in + p_t + p_b
        w_padded = w_in + p_l + p_r
        full_h = (h_out - 1) * s_h + d_h * (k_h - 1) + 1
        full_w = (w_out - 1) * s_w + d_w * (k_w - 1) + 1

        grad_x_full = conv2d_transpose(
            cot_for_vjp,
            w,
            stride=stride,
            dilation=dilation,
            padding=(0, 0, 0, 0),
            output_paddings=(0, 0),
        )
        if h_padded - full_h > 0 or w_padded - full_w > 0:
            grad_x_full = pad(
                grad_x_full,
                paddings=[(0, 0), (0, h_padded - full_h), (0, w_padded - full_w), (0, 0)],
            )
        grad_x_flat = slice_tensor(
            grad_x_full, start=[0, p_t, p_l, 0], size=[n, h_in, w_in, c_in]
        )
        grad_x = _restore_batch_dims_from_n(grad_x_flat, batch_shape_x, n_orig_x)

        x_pad = x_for_vjp
        if p_t or p_b or p_l or p_r:
            x_pad = pad(x_for_vjp, paddings=[(0, 0), (p_t, p_b), (p_l, p_r), (0, 0)])

        h_eff = d_h * (k_h - 1) + s_h * (h_out - 1) + 1
        w_eff = d_w * (k_w - 1) + s_w * (w_out - 1) + 1
        if int(x_pad.shape[1]) > h_eff or int(x_pad.shape[2]) > w_eff:
            x_pad = slice_tensor(x_pad, start=[0, 0, 0, 0], size=[n, h_eff, w_eff, c_in])

        x_perm = permute(x_pad, order=(3, 1, 2, 0))
        cot_perm = permute(cot_for_vjp, order=(1, 2, 0, 3))
        if s_h > 1 or s_w > 1:
            cot_perm = _dilate_spatial_dims01(cot_perm, s_h, s_w)

        grad_w_perm = conv2d(
            x_perm,
            cot_perm,
            stride=dilation,
            dilation=(1, 1),
            padding=(0, 0, 0, 0),
        )
        grad_w = permute(grad_w_perm, order=(1, 2, 0, 3))

        if bias is None:
            return [grad_x, grad_w]
        return [
            grad_x,
            grad_w,
            reduce_sum(cot_for_vjp, axis=[0, 1, 2], keepdims=False),
        ]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        from . import add, conv2d

        x, w = primals[0], primals[1]
        tx, tw = tangents[0], tangents[1]

        conv_kwargs = {
            "stride": tuple(kwargs["stride"]),
            "dilation": tuple(kwargs["dilation"]),
            "padding": tuple(kwargs["padding"]),
            "groups": int(kwargs.get("groups", 1)),
            "input_layout": kwargs.get("input_layout"),
            "filter_layout": kwargs.get("filter_layout"),
        }
        dy = add(
            conv2d(x=tx, filter=w, bias=None, **conv_kwargs),
            conv2d(x=x, filter=tw, bias=None, **conv_kwargs),
        )
        if len(primals) > 2:
            dy = add(dy, tangents[2])
        return [dy]


# ---------------------------------------------------------------------------
# Conv2DTranspose
# ---------------------------------------------------------------------------


class Conv2DTransposeOp(Operation):
    @property
    def name(self) -> str:
        return "conv2d_transpose"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None

        x_shape = tuple(int(d) for d in x.shape)
        prefix_shape = x_shape[:-4]
        x_for_conv = x
        n_orig = h_in = w_in = c_in = None
        flat_batch = None

        if prefix_shape:
            n_orig, h_in, w_in, c_in = x_shape[-4:]
            flat_batch = 1
            for dim in prefix_shape:
                flat_batch *= int(dim)
            x_for_conv = ops.reshape(x, (flat_batch * n_orig, h_in, w_in, c_in))

        call_kwargs: dict[str, Any] = {
            "stride": tuple(kwargs["stride"]),
            "dilation": tuple(kwargs["dilation"]),
            "padding": tuple(kwargs["padding"]),
            "output_paddings": tuple(kwargs["output_paddings"]),
        }
        if bias is not None:
            call_kwargs["bias"] = bias

        y = ops.conv2d_transpose(x=x_for_conv, filter=weight, **call_kwargs)
        if prefix_shape:
            n_flat, h_out, w_out, c_out = (int(d) for d in y.shape)
            expected_n_flat = int(flat_batch) * int(n_orig)
            if n_flat != expected_n_flat:
                raise RuntimeError(
                    f"{self.name}: unexpected flattened batch size {n_flat}, expected {expected_n_flat}"
                )
            y = ops.reshape(y, (*prefix_shape, int(n_orig), h_out, w_out, c_out))

        return [y]

    def compute_physical_shape(
        self, args: OpArgs, kwargs: OpKwargs, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        from ..core.sharding import spmd

        x = args[0]
        w = args[1]
        stride = tuple(kwargs["stride"])
        dilation = tuple(kwargs["dilation"])
        padding = tuple(kwargs["padding"])
        output_paddings = tuple(kwargs["output_paddings"])

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            sx = x.physical_local_shape_ints(i if i < x.num_shards else 0)
            sw = w.physical_local_shape_ints(i if i < w.num_shards else 0)
            if sx is None or sw is None:
                raise RuntimeError(f"Could not determine physical shape for {self.name}")
            prefix, n, h_in, w_in, c_in = _split_trailing_nhwc(sx, self.name)
            k_h, k_w, c_out, c_in_w = sw
            if c_in != c_in_w:
                raise ValueError(
                    f"Input channels ({c_in}) must match filter input channels ({c_in_w})"
                )
            h_out, w_out = _conv2d_transpose_out_hw(
                h_in, w_in, k_h, k_w, stride, dilation, padding, output_paddings
            )
            shapes.append((*prefix, n, h_out, w_out, c_out))

        dtypes, devices = self._build_shard_metadata(x, mesh, num_shards)
        return shapes, dtypes, devices

    def sharding_rule(self, input_shapes, output_shapes, **kwargs) -> Any:
        from ..core.sharding.propagation import OpShardingRuleTemplate

        in_x = {0: ["n"], 1: [], 2: [], 3: ["ci"]}
        in_w = {0: [], 1: [], 2: ["co"], 3: ["ci"]}
        out_y = {0: ["n"], 1: [], 2: [], 3: ["co"]}
        return OpShardingRuleTemplate([in_x, in_w], [out_y]).instantiate(
            input_shapes, output_shapes
        )

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        from . import conv2d
        from .reduction import reduce_sum
        from .view import pad, permute

        x = primals[0]
        w = primals[1]
        bias = primals[2] if len(primals) > 2 else None
        cot = cotangents[0]

        x_for_vjp, batch_shape_x, n_orig_x = _flatten_batch_dims_into_n(x)
        cot_for_vjp, batch_shape_cot, n_orig_cot = _flatten_batch_dims_into_n(cot)
        if batch_shape_x != batch_shape_cot or n_orig_x != n_orig_cot:
            raise ValueError(
                f"conv2d_transpose VJP batch_dims mismatch: x has {batch_shape_x} with N={n_orig_x}, "
                f"cot has {batch_shape_cot} with N={n_orig_cot}"
            )

        stride = tuple(kwargs["stride"])
        dilation = tuple(kwargs["dilation"])
        padding_vals = tuple(kwargs["padding"])
        p_t, p_b, p_l, p_r = padding_vals

        s_h, s_w = stride

        grad_x_flat = conv2d(
            cot_for_vjp, w, stride=stride, dilation=dilation, padding=padding_vals
        )
        grad_x = _restore_batch_dims_from_n(grad_x_flat, batch_shape_x, n_orig_x)

        cot_pad = pad(cot_for_vjp, paddings=[(0, 0), (p_t, p_b), (p_l, p_r), (0, 0)])
        x_perm = permute(x_for_vjp, order=(1, 2, 0, 3))
        if s_h > 1 or s_w > 1:
            x_perm = _dilate_spatial_dims01(x_perm, s_h, s_w)

        cot_pad_perm = permute(cot_pad, order=(3, 1, 2, 0))
        grad_w_perm = conv2d(
            cot_pad_perm,
            x_perm,
            stride=dilation,
            dilation=(1, 1),
            padding=(0, 0, 0, 0),
        )
        grad_w = permute(grad_w_perm, order=(1, 2, 0, 3))

        if bias is None:
            return [grad_x, grad_w]
        return [
            grad_x,
            grad_w,
            reduce_sum(cot_for_vjp, axis=[0, 1, 2], keepdims=False),
        ]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        from . import add, conv2d_transpose

        x, w = primals[0], primals[1]
        tx, tw = tangents[0], tangents[1]

        if x.batch_dims > 0 or tx.batch_dims > 0 or len(x.shape) != 4 or len(tx.shape) != 4:
            raise NotImplementedError(
                "conv2d_transpose JVP with batch_dims/transformed rank>4 inputs is not yet supported"
            )

        conv_kwargs = {
            "stride": tuple(kwargs["stride"]),
            "dilation": tuple(kwargs["dilation"]),
            "padding": tuple(kwargs["padding"]),
            "output_paddings": tuple(kwargs["output_paddings"]),
            "input_layout": kwargs.get("input_layout"),
            "filter_layout": kwargs.get("filter_layout"),
        }
        dy = add(
            conv2d_transpose(x=tx, filter=w, bias=None, **conv_kwargs),
            conv2d_transpose(x=x, filter=tw, bias=None, **conv_kwargs),
        )
        if len(primals) > 2:
            dy = add(dy, tangents[2])
        return [dy]


_conv2d_op = Conv2DOp()
_conv2d_transpose_op = Conv2DTransposeOp()


def conv2d(
    x: "Tensor",
    filter: "Tensor",
    *,
    stride: int | tuple[int, int] = (1, 1),
    dilation: int | tuple[int, int] = (1, 1),
    padding: int | tuple[int, int] | tuple[int, int, int, int] = (0, 0, 0, 0),
    groups: int = 1,
    bias: "Tensor | None" = None,
    input_layout: Any = None,
    filter_layout: Any = None,
) -> "Tensor":
    """Apply a 2D convolution over an input tensor.

    Operates on tensors in NHWC layout (batch, height, width, channels).
    Filters are expected in HWIO layout (kernel_h, kernel_w, in_channels, out_channels).
    Supports autograd (VJP and JVP).

    Args:
        x: Input tensor of shape ``(N, H, W, C_in)``.
        filter: Convolution kernel of shape ``(K_h, K_w, C_in, C_out)``.
        stride: Convolution stride as ``(s_h, s_w)`` or a single int.
            Default: ``(1, 1)``.
        dilation: Kernel dilation as ``(d_h, d_w)`` or a single int.
            Currently only ``(1, 1)`` is supported. Default: ``(1, 1)``.
        padding: Padding as an int, ``(pad_h, pad_w)``, or a 4-tuple
            ``(top, bottom, left, right)``. Default: ``(0, 0, 0, 0)``.
        groups: Number of blocked connections from input channels to output
            channels. Currently only ``1`` is supported. Default: ``1``.
        bias: Optional bias tensor of shape ``(C_out,)``.

    Returns:
        Output tensor of shape ``(N, H_out, W_out, C_out)``.
    """
    x = ensure_tensor(x)
    filter = ensure_tensor(filter)
    bias_t = ensure_tensor(bias) if bias is not None else None

    if len(x.shape) != 4 or len(filter.shape) != 4:
        raise ValueError(
            f"conv2d expects rank-4 tensors, got x={tuple(int(d) for d in x.shape)}, "
            f"filter={tuple(int(d) for d in filter.shape)}"
        )
    if input_layout is not None or filter_layout is not None:
        raise ValueError(
            "conv2d currently supports only default MAX NHWC/RSCF layouts in Nabla phase 1"
        )

    norm_stride = _normalize_2tuple(stride, "stride")
    norm_dilation = _normalize_2tuple(dilation, "dilation")
    norm_padding = _normalize_4padding(padding)
    groups_i = int(groups)

    _require_all_positive_2tuple(norm_stride, "stride")
    _require_all_positive_2tuple(norm_dilation, "dilation")
    _require_all_nonnegative(norm_padding, "padding")

    if groups_i <= 0:
        raise ValueError(f"groups must be > 0, got {groups_i}")

    if norm_dilation != (1, 1):
        raise ValueError("MAX conv2d currently supports only dilation=(1, 1)")
    if groups_i != 1:
        raise ValueError(
            "MAX conv2d grouped mode currently requires prepacked filters and is not yet exposed in Nabla"
        )

    c_in = int(x.shape[-1])
    c_in_w = int(filter.shape[2])
    c_out = int(filter.shape[3])
    expected_c_in = c_in_w * groups_i
    if c_in != expected_c_in:
        raise ValueError(
            f"Input channels ({c_in}) must match filter channels*groups ({expected_c_in})"
        )
    if bias_t is not None:
        if len(bias_t.shape) != 1:
            raise ValueError(
                f"conv2d bias must be rank-1 [C_out], got shape={tuple(int(d) for d in bias_t.shape)}"
            )
        if int(bias_t.shape[0]) != c_out:
            raise ValueError(
                f"conv2d bias size ({int(bias_t.shape[0])}) must match filter C_out ({c_out})"
            )

    kwargs: OpKwargs = {
        "stride": norm_stride,
        "dilation": norm_dilation,
        "padding": norm_padding,
        "groups": groups_i,
        "input_layout": input_layout,
        "filter_layout": filter_layout,
    }
    args = [x, filter] if bias_t is None else [x, filter, bias_t]
    return _conv2d_op(args, kwargs)[0]


def conv2d_transpose(
    x: "Tensor",
    filter: "Tensor",
    *,
    stride: int | tuple[int, int] = (1, 1),
    dilation: int | tuple[int, int] = (1, 1),
    padding: int | tuple[int, int] | tuple[int, int, int, int] = (0, 0, 0, 0),
    output_paddings: int | tuple[int, int] = (0, 0),
    bias: "Tensor | None" = None,
    input_layout: Any = None,
    filter_layout: Any = None,
) -> "Tensor":
    """Apply a 2D transposed convolution (fractionally-strided convolution).

    Also known as a ``deconvolution`` or ``upsample`` layer. Produces the
    gradient of :func:`conv2d` with respect to its input, enabling
    encoder–decoder architectures. Fully differentiable.

    Args:
        x: Input tensor of shape ``(N, H, W, C_in)``.
        filter: Kernel of shape ``(K_h, K_w, C_out, C_in)`` (note the
            transposed channel order compared to :func:`conv2d`).
        stride: Stride of the transposed convolution. Default: ``(1, 1)``.
        dilation: Kernel dilation. Currently only ``(1, 1)`` is supported.
            Default: ``(1, 1)``.
        padding: Amount of implicit zero-padding removed from the output.
            Same format as :func:`conv2d`. Default: ``(0, 0, 0, 0)``.
        output_paddings: Extra rows/columns added to the output shape.
            Currently only ``(0, 0)`` is supported. Default: ``(0, 0)``.
        bias: Optional bias tensor of shape ``(C_out,)``.

    Returns:
        Output tensor of shape ``(N, H_out, W_out, C_out)``.
    """
    x = ensure_tensor(x)
    filter = ensure_tensor(filter)
    bias_t = ensure_tensor(bias) if bias is not None else None

    if len(x.shape) != 4 or len(filter.shape) != 4:
        raise ValueError(
            "conv2d_transpose expects rank-4 tensors, got "
            f"x={tuple(int(d) for d in x.shape)}, filter={tuple(int(d) for d in filter.shape)}"
        )
    if input_layout is not None or filter_layout is not None:
        raise ValueError(
            "conv2d_transpose currently supports only default MAX NHWC layouts in Nabla phase 1"
        )

    norm_stride = _normalize_2tuple(stride, "stride")
    norm_dilation = _normalize_2tuple(dilation, "dilation")
    norm_padding = _normalize_4padding(padding)
    norm_output_paddings = _normalize_output_paddings(output_paddings)

    _require_all_positive_2tuple(norm_stride, "stride")
    _require_all_positive_2tuple(norm_dilation, "dilation")
    _require_all_nonnegative(norm_padding, "padding")
    _require_all_nonnegative(norm_output_paddings, "output_paddings")

    if norm_output_paddings != (0, 0):
        raise ValueError("MAX conv2d_transpose currently supports only output_paddings=(0, 0)")

    c_in = int(x.shape[-1])
    c_out = int(filter.shape[2])
    c_in_w = int(filter.shape[3])
    if c_in != c_in_w:
        raise ValueError(
            f"Input channels ({c_in}) must match filter input channels ({c_in_w})"
        )
    if bias_t is not None:
        if len(bias_t.shape) != 1:
            raise ValueError(
                "conv2d_transpose bias must be rank-1 [C_out], got "
                f"shape={tuple(int(d) for d in bias_t.shape)}"
            )
        if int(bias_t.shape[0]) != c_out:
            raise ValueError(
                f"conv2d_transpose bias size ({int(bias_t.shape[0])}) must match filter C_out ({c_out})"
            )

    kwargs: OpKwargs = {
        "stride": norm_stride,
        "dilation": norm_dilation,
        "padding": norm_padding,
        "output_paddings": norm_output_paddings,
        "input_layout": input_layout,
        "filter_layout": filter_layout,
    }
    args = [x, filter] if bias_t is None else [x, filter, bias_t]
    return _conv2d_transpose_op(args, kwargs)[0]


__all__ = ["conv2d", "conv2d_transpose"]
