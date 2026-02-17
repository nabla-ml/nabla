# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import ops

from .base import AxisOp, Operation, ReduceOperation

if TYPE_CHECKING:
    from ..core.tensor import Tensor
    from .base import OpArgs, OpKwargs, OpResult, OpTensorValues

from .view import SqueezePhysicalOp

_squeeze_physical_op = SqueezePhysicalOp()


class PhysicalReduceOp(Operation):
    """Base for physical reduction operations (sum, mean, max, min).

    Subclasses only need to define `name` and `kernel()`, and optionally
    override `collective_reduce_type`.
    """

    _infer_output_sharding: bool = False

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        from ..core.sharding import spmd

        x = args[0]
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            in_shape = x.physical_local_shape_ints(idx)
            if in_shape is None:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )
            norm_axis = axis if axis >= 0 else len(in_shape) + axis
            if keepdims:
                out_shape = tuple(
                    1 if j == norm_axis else d for j, d in enumerate(in_shape)
                )
            else:
                out_shape = tuple(d for j, d in enumerate(in_shape) if j != norm_axis)
            shapes.append(out_shape)

        dtypes, devices = self._build_shard_metadata(x, mesh, num_shards)
        return shapes, dtypes, devices

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Reduce: (d0, d1, ...) -> (d0, 1, ...) with reduce_dim kept as size 1."""
        from ..core.sharding.propagation import OpShardingRuleTemplate

        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)

        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)

        out_factors = list(factors)
        if 0 <= axis < rank:
            out_factors[axis] = "1"
        out_str = " ".join(out_factors)

        return OpShardingRuleTemplate.parse(
            f"{in_str} -> {out_str}", input_shapes
        ).instantiate(input_shapes, output_shapes)

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs
    ) -> tuple[int, ...]:
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        if keepdims:
            return tuple(1 if i == axis else d for i, d in enumerate(in_shape))
        else:
            return tuple(d for i, d in enumerate(in_shape) if i != axis)


class ReduceSumOp(ReduceOperation):
    @property
    def name(self) -> str:
        return "reduce_sum"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis = kwargs.get("axis", 0)
        return [ops.sum(x, axis)]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for reduce_sum: broadcast cotangent back to input shape."""
        x = primals[0]
        from ..ops.view.shape import broadcast_to

        return [broadcast_to(cotangents[0], tuple(x.shape))]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        t = tangents[0]
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        return [reduce_sum(t, axis=axis, keepdims=keepdims)]


class MeanOp(ReduceOperation):
    @property
    def name(self) -> str:
        return "mean"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis = kwargs.get("axis", 0)
        return [ops.mean(x, axis)]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for mean: broadcast cotangent / axis_size."""
        x = primals[0]
        axis = kwargs.get("axis", 0)
        axis_size = x.shape[axis]
        from ..ops.view.shape import broadcast_to

        target_shape = tuple(int(d) for d in x.shape)
        return [broadcast_to(cotangents[0], target_shape) / axis_size]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        return [mean(tangents[0], axis=axis, keepdims=keepdims)]


class ReduceMaxOp(ReduceOperation):
    @property
    def name(self) -> str:
        return "reduce_max"

    @property
    def collective_reduce_type(self) -> str:
        return "max"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis = kwargs.get("axis", 0)
        return [ops._reduce_max(x, axis=axis)]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        x = primals[0]
        from ..ops.binary import mul
        from ..ops.comparison import equal
        from ..ops.view.shape import broadcast_to

        max_broadcasted = broadcast_to(outputs[0], tuple(x.shape))
        mask = equal(x, max_broadcasted)
        cotangent_broadcasted = broadcast_to(cotangents[0], tuple(x.shape))
        return [mul(cotangent_broadcasted, mask)]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        x = primals[0]
        from ..ops.binary import mul
        from ..ops.comparison import equal
        from ..ops.reduction import reduce_sum
        from ..ops.view.shape import broadcast_to

        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        max_broadcasted = broadcast_to(outputs[0], tuple(x.shape))
        mask = equal(x, max_broadcasted)
        return [reduce_sum(mul(tangents[0], mask), axis=axis, keepdims=keepdims)]


class ReduceSumPhysicalOp(PhysicalReduceOp):
    @property
    def name(self) -> str:
        return "reduce_sum_physical"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis = kwargs.get("axis", 0)
        return [ops.sum(x, axis=axis)]


class MeanPhysicalOp(PhysicalReduceOp):
    @property
    def name(self) -> str:
        return "mean_physical"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis = kwargs.get("axis", 0)
        return [ops.mean(x, axis=axis)]


class ReduceMaxPhysicalOp(PhysicalReduceOp):
    @property
    def name(self) -> str:
        return "reduce_max_physical"

    @property
    def collective_reduce_type(self) -> str:
        return "max"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis = kwargs.get("axis", 0)
        return [ops._reduce_max(x, axis=axis)]


class ReduceMinOp(ReduceOperation):
    @property
    def name(self) -> str:
        return "reduce_min"

    @property
    def collective_reduce_type(self) -> str:
        return "min"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis = kwargs.get("axis", 0)
        return [ops._reduce_min(x, axis=axis)]


class ReduceMinPhysicalOp(PhysicalReduceOp):
    @property
    def name(self) -> str:
        return "reduce_min_physical"

    @property
    def collective_reduce_type(self) -> str:
        return "min"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis = kwargs.get("axis", 0)
        return [ops._reduce_min(x, axis=axis)]


_reduce_min_physical_op = ReduceMinPhysicalOp()
_reduce_min_op = ReduceMinOp()
_reduce_max_physical_op = ReduceMaxPhysicalOp()
_reduce_sum_physical_op = ReduceSumPhysicalOp()
_mean_physical_op = MeanPhysicalOp()
_reduce_sum_op = ReduceSumOp()
_mean_op = MeanOp()
_reduce_max_op = ReduceMaxOp()


def _multi_axis_reduce(op, x: Tensor, *, axis=None, keepdims: bool = False) -> Tensor:
    """Shared implementation for multi-axis reductions (reduce_sum, reduce_max, reduce_min)."""
    from .view import squeeze
    from .view.shape import reshape

    if axis is None:
        axis = tuple(range(len(x.shape)))

    if isinstance(axis, (list, tuple)):
        axes = sorted(
            [ax if ax >= 0 else len(x.shape) + ax for ax in axis], reverse=True
        )
        if not axes:
            # Empty axes (e.g. reducing a 0-d scalar) — nothing to reduce
            return x
        res = x
        for ax in axes:
            res = op([res], {"axis": ax, "keepdims": True})[0]

        if not keepdims:
            out_shape = tuple(
                int(d)
                for i, d in enumerate(x.shape)
                if i not in {ax if ax >= 0 else len(x.shape) + ax for ax in axis}
            )
            res = (
                reshape(res, out_shape if out_shape else (1,))
                if len(axes) > 1
                else squeeze(res, axis=axes[0])
            )
            if not out_shape and len(axes) > 1:
                res = squeeze(res, axis=0)
        return res

    result = op([x], {"axis": axis, "keepdims": True})[0]
    if not keepdims:
        result = squeeze(result, axis=axis)
    return result


def reduce_sum(
    x: Tensor,
    *,
    axis: int | tuple[int, ...] | list[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    return _multi_axis_reduce(_reduce_sum_op, x, axis=axis, keepdims=keepdims)


def mean(
    x: Tensor,
    *,
    axis: int | tuple[int, ...] | list[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute arithmetic mean along specified axis/axes.

    Implemented as sum(x) / product(shape[axes]) to correctly handle distributed sharding.
    """
    s = reduce_sum(x, axis=axis, keepdims=keepdims)

    shape = x.shape
    if axis is None:
        count = 1
        for d in shape:
            count *= int(d)
    elif isinstance(axis, (list, tuple)):
        count = 1
        for ax in axis:
            norm_ax = ax if ax >= 0 else len(shape) + ax
            count *= int(shape[norm_ax])
    else:
        norm_ax = axis if axis >= 0 else len(shape) + axis
        count = int(shape[norm_ax])

    return s / count


def reduce_max(
    x: Tensor,
    *,
    axis: int | tuple[int, ...] | list[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    return _multi_axis_reduce(_reduce_max_op, x, axis=axis, keepdims=keepdims)


def _physical_reduce(op, x: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    """Shared wrapper for physical reduce operations."""
    result = op([x], {"axis": axis, "keepdims": True})[0]
    if not keepdims:
        result = _squeeze_physical_op([result], {"axis": axis})[0]
    return result


def reduce_sum_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    return _physical_reduce(_reduce_sum_physical_op, x, axis, keepdims)


def mean_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    return _physical_reduce(_mean_physical_op, x, axis, keepdims)


def reduce_max_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    return _physical_reduce(_reduce_max_physical_op, x, axis, keepdims)


def reduce_min(
    x: Tensor,
    *,
    axis: int | tuple[int, ...] | list[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    return _multi_axis_reduce(_reduce_min_op, x, axis=axis, keepdims=keepdims)


def reduce_min_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    return _physical_reduce(_reduce_min_physical_op, x, axis, keepdims)


class _ArgReduceOp(AxisOp):
    """Base for argmax/argmin operations."""

    _op_name: str = ""

    def _get_reduce_fn(self):
        return ops.argmax if self._op_name == "argmax" else ops.argmin

    @property
    def name(self) -> str:
        return self._op_name

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis = kwargs.get("axis", -1)
        reduce_fn = self._get_reduce_fn()
        rank = len(x.shape)
        if axis < 0:
            axis += rank

        if axis != rank - 1:
            perm = list(range(rank))
            perm[axis], perm[-1] = perm[-1], perm[axis]
            x = ops.permute(x, tuple(perm))
            return [ops.squeeze(reduce_fn(x, axis=-1), -1)]

        return [ops.squeeze(reduce_fn(x, axis=axis), axis)]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        from max.dtype import DType

        from ..core.sharding import spmd

        x = args[0]
        axis = kwargs.get("axis", -1)
        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            in_shape = x.physical_local_shape_ints(idx)
            if in_shape is None:
                raise RuntimeError("Could not determine physical shape")
            norm_axis = axis if axis >= 0 else len(in_shape) + axis
            out_shape = tuple(d for j, d in enumerate(in_shape) if j != norm_axis)
            shapes.append(out_shape)

        return shapes, [DType.int64] * num_shards, [x.device] * num_shards

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        return [None]

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs: Any
    ) -> tuple[int, ...]:
        axis = kwargs.get("axis", -1)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        return tuple(d for i, d in enumerate(in_shape) if i != axis)


class ArgmaxOp(_ArgReduceOp):
    _op_name = "argmax"


class ArgminOp(_ArgReduceOp):
    _op_name = "argmin"


class CumsumOp(AxisOp):
    """Cumulative sum along an axis."""

    @property
    def name(self) -> str:
        return "cumsum"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis = kwargs.get("axis", -1)
        exclusive = kwargs.get("exclusive", False)
        reverse = kwargs.get("reverse", False)
        return [ops.cumsum(x, axis=axis, exclusive=exclusive, reverse=reverse)]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        x = args[0]
        shapes = []
        for i in range(x.num_shards):
            shapes.append(x.physical_local_shape_ints(i))
        return shapes, [x.dtype] * x.num_shards, [x.device] * x.num_shards

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP: flip(cumsum(flip(cotangent, axis), axis), axis)."""
        axis = kwargs.get("axis", -1)

        from ..ops.view.axes import flip

        return [flip(cumsum(flip(cotangents[0], axis=axis), axis=axis), axis=axis)]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP: cumsum(tangent, axis)."""
        axis = kwargs.get("axis", -1)
        return [cumsum(tangents[0], axis=axis)]

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        # Cumsum is elementwise identity in terms of rank/factors
        from ..core.sharding.propagation import OpShardingRuleTemplate

        rank = len(input_shapes[0])
        mapping = {i: [f"d{i}"] for i in range(rank)}
        return OpShardingRuleTemplate([mapping], [mapping]).instantiate(
            input_shapes, output_shapes
        )

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs: Any
    ) -> tuple[int, ...]:
        return input_shapes[0]


_argmax_op = ArgmaxOp()
_argmin_op = ArgminOp()
_cumsum_op = CumsumOp()


def argmax(x: Tensor, axis: int = -1, keepdims: bool = False) -> Tensor:
    res = _argmax_op([x], {"axis": axis})[0]
    if keepdims:
        from .view import unsqueeze

        res = unsqueeze(res, axis=axis)
    return res


def argmin(x: Tensor, axis: int = -1, keepdims: bool = False) -> Tensor:
    res = _argmin_op([x], {"axis": axis})[0]
    if keepdims:
        from .view import unsqueeze

        res = unsqueeze(res, axis=axis)
    return res


def cumsum(
    x: Tensor, axis: int = -1, exclusive: bool = False, reverse: bool = False
) -> Tensor:
    return _cumsum_op([x], {"axis": axis, "exclusive": exclusive, "reverse": reverse})[
        0
    ]


__all__ = [
    "ReduceSumOp",
    "reduce_sum",
    "MeanOp",
    "mean",
    "ReduceMaxOp",
    "reduce_max",
    "ReduceSumPhysicalOp",
    "reduce_sum_physical",
    "MeanPhysicalOp",
    "mean_physical",
    "ReduceMaxPhysicalOp",
    "reduce_max_physical",
    "ReduceMinOp",
    "reduce_min",
    "ReduceMinPhysicalOp",
    "reduce_min_physical",
]
