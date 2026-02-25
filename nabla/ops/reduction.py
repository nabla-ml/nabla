# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import os

from typing import TYPE_CHECKING, Any

from max.graph import ops

from .base import AxisOp, Operation, ReduceOperation

if TYPE_CHECKING:
    from ..core.tensor import Tensor
    from .base import OpArgs, OpKwargs, OpResult, OpTensorValues

from .view import SqueezePhysicalOp

_squeeze_physical_op = SqueezePhysicalOp()


def _debug_phys_vjp(tag: str, t: Tensor) -> None:
    if os.environ.get("NABLA_DEBUG_PHYS_VJP", "0") not in {
        "1",
        "true",
        "TRUE",
        "True",
    }:
        return
    phys = t.physical_global_shape or t.local_shape
    print(
        f"[NABLA_DEBUG_PHYS_VJP] {tag}: "
        f"shape={tuple(int(d) for d in t.shape)} "
        f"batch_dims={t.batch_dims} "
        f"phys={tuple(int(d) for d in phys)}"
    )


def _physical_shape_tuple(x: Tensor) -> tuple[int, ...]:
    return tuple(int(d) for d in (x.physical_global_shape or x.local_shape))


def _target_with_rank_prefix(
    like: Tensor, base_phys: tuple[int, ...]
) -> tuple[tuple[int, ...], int]:
    like_phys = _physical_shape_tuple(like)
    prefix = len(like_phys) - len(base_phys)
    if prefix <= 0:
        return base_phys, 0
    return tuple(int(d) for d in like_phys[:prefix]) + base_phys, prefix


def _normalize_axis(axis: int, rank: int) -> int:
    return axis if axis >= 0 else rank + axis


def _normalize_physical_axis_for_tensor(axis: int, x: Tensor) -> int:
    phys_rank = len(_physical_shape_tuple(x))
    if axis >= 0:
        return int(x.batch_dims) + axis
    return phys_rank + axis


class PhysicalReduceOp(AxisOp):
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
        # NOTE: MAX reduction ops ALWAYS produce keepdims=True output
        # (the reduced axis becomes size 1). The kernel never squeezes.
        # Squeezing for keepdims=False is handled at the _physical_reduce
        # level via an explicit traced squeeze_physical call.

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
            # Always keepdims=True to match MAX kernel output
            out_shape = tuple(
                1 if j == norm_axis else d for j, d in enumerate(in_shape)
            )
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

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        from ..ops.view import (
            broadcast_batch_dims,
            broadcast_to,
            broadcast_to_physical,
            unsqueeze,
        )

        x = primals[0]
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        x_phys = _physical_shape_tuple(x)
        _debug_phys_vjp("sum.vjp.x", x)

        cot = cotangents[0]
        _debug_phys_vjp("sum.vjp.cot.in", cot)
        if not keepdims:
            cot = unsqueeze(cot, axis=axis)
            _debug_phys_vjp("sum.vjp.cot.after_unsqueeze", cot)

        cot = broadcast_to(cot, tuple(int(d) for d in x.shape))
        _debug_phys_vjp("sum.vjp.cot.after_broadcast_to", cot)

        if cot.batch_dims < x.batch_dims:
            target_batch = x_phys[: x.batch_dims]
            cot = broadcast_batch_dims(cot, target_batch)
            _debug_phys_vjp("sum.vjp.cot.after_broadcast_batch_dims", cot)

        cot_phys = _physical_shape_tuple(cot)
        rank_prefix = max(0, len(cot_phys) - len(x_phys))
        target_phys = tuple(int(d) for d in cot_phys[:rank_prefix]) + x_phys

        out = broadcast_to_physical(cot, target_phys)
        _debug_phys_vjp("sum.vjp.out", out)
        return [out]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        return [
            reduce_sum_physical(
                tangents[0], axis=axis, keepdims=keepdims
            )
        ]


class MeanPhysicalOp(PhysicalReduceOp):
    @property
    def name(self) -> str:
        return "mean_physical"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis = kwargs.get("axis", 0)
        return [ops.mean(x, axis=axis)]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for mean_physical: broadcast cotangent / reduced axis size."""
        from ..ops.view import (
            broadcast_batch_dims,
            broadcast_to,
            broadcast_to_physical,
            unsqueeze,
        )

        x = primals[0]
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)

        x_phys = _physical_shape_tuple(x)
        norm_axis = _normalize_physical_axis_for_tensor(axis, x)
        axis_size = x_phys[norm_axis]

        cot = cotangents[0]
        if not keepdims:
            cot = unsqueeze(cot, axis=axis)

        cot = broadcast_to(cot, tuple(int(d) for d in x.shape))

        if cot.batch_dims < x.batch_dims:
            target_batch = x_phys[: x.batch_dims]
            cot = broadcast_batch_dims(cot, target_batch)

        cot_phys = _physical_shape_tuple(cot)
        rank_prefix = max(0, len(cot_phys) - len(x_phys))
        target_phys = tuple(int(d) for d in cot_phys[:rank_prefix]) + x_phys

        return [broadcast_to_physical(cot, target_phys) / axis_size]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        return [
            mean_physical(
                tangents[0], axis=axis, keepdims=keepdims
            )
        ]


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

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        from ..ops.binary import mul
        from ..ops.comparison import equal
        from ..ops.view import broadcast_to_physical, unsqueeze_physical

        x = primals[0]
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)

        x_phys = _physical_shape_tuple(x)
        norm_axis = _normalize_physical_axis_for_tensor(axis, x)

        max_target, _ = _target_with_rank_prefix(outputs[0], x_phys)
        max_broadcasted = broadcast_to_physical(outputs[0], max_target)
        mask = equal(x, max_broadcasted)

        cot = cotangents[0]
        cot_target, rank_prefix = _target_with_rank_prefix(cot, x_phys)
        cot_axis = rank_prefix + norm_axis
        if not keepdims:
            cot = unsqueeze_physical(cot, axis=cot_axis)
        cot_broadcasted = broadcast_to_physical(cot, cot_target)
        return [mul(cot_broadcasted, mask)]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP for reduce_max_physical: select tangent at argmax position.

        d/dt max(x + t*dx) = dx[argmax(x)] = sum(dx * mask) / sum(mask)
        where mask = (x == max(x)) along the reduced axis.
        """
        from ..ops.binary import mul
        from ..ops.comparison import equal
        from ..ops.view import broadcast_to_physical, unsqueeze_physical

        x = primals[0]
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)

        x_phys = _physical_shape_tuple(x)
        norm_axis = _normalize_physical_axis_for_tensor(axis, x)

        # Broadcast max output back to input shape, build mask
        max_target, _ = _target_with_rank_prefix(outputs[0], x_phys)
        max_broadcasted = broadcast_to_physical(outputs[0], max_target)
        mask = equal(x, max_broadcasted)

        # Select tangent at max positions & reduce
        tan = tangents[0]
        tan_target, rank_prefix = _target_with_rank_prefix(tan, x_phys)
        tan_axis = rank_prefix + norm_axis
        if not keepdims:
            tan = unsqueeze_physical(tan, axis=tan_axis)
        tan_broadcasted = broadcast_to_physical(tan, tan_target)

        # masked_tangent has shape x_phys — reduce along the reduction axis
        masked_tangent = mul(tan_broadcasted, mask)
        return [reduce_sum_physical(masked_tangent, axis=axis, keepdims=keepdims)]


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

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        from ..ops.binary import mul
        from ..ops.comparison import equal
        from ..ops.view import broadcast_to_physical, unsqueeze_physical

        x = primals[0]
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)

        x_phys = _physical_shape_tuple(x)
        norm_axis = _normalize_physical_axis_for_tensor(axis, x)

        min_target, _ = _target_with_rank_prefix(outputs[0], x_phys)
        min_broadcasted = broadcast_to_physical(outputs[0], min_target)
        mask = equal(x, min_broadcasted)

        cot = cotangents[0]
        cot_target, rank_prefix = _target_with_rank_prefix(cot, x_phys)
        cot_axis = rank_prefix + norm_axis
        if not keepdims:
            cot = unsqueeze_physical(cot, axis=cot_axis)
        cot_broadcasted = broadcast_to_physical(cot, cot_target)
        return [mul(cot_broadcasted, mask)]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP for reduce_min_physical: select tangent at argmin position.

        d/dt min(x + t*dx) = dx[argmin(x)] = sum(dx * mask) / sum(mask)
        where mask = (x == min(x)) along the reduced axis.
        """
        from ..ops.binary import mul
        from ..ops.comparison import equal
        from ..ops.view import broadcast_to_physical, unsqueeze_physical

        x = primals[0]
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)

        x_phys = _physical_shape_tuple(x)
        norm_axis = _normalize_physical_axis_for_tensor(axis, x)

        # Broadcast min output back to input shape, build mask
        min_target, _ = _target_with_rank_prefix(outputs[0], x_phys)
        min_broadcasted = broadcast_to_physical(outputs[0], min_target)
        mask = equal(x, min_broadcasted)

        # Select tangent at min positions & reduce
        tan = tangents[0]
        tan_target, rank_prefix = _target_with_rank_prefix(tan, x_phys)
        tan_axis = rank_prefix + norm_axis
        if not keepdims:
            tan = unsqueeze_physical(tan, axis=tan_axis)
        tan_broadcasted = broadcast_to_physical(tan, tan_target)

        # masked_tangent has shape x_phys — reduce along the reduction axis
        masked_tangent = mul(tan_broadcasted, mask)
        return [reduce_sum_physical(masked_tangent, axis=axis, keepdims=keepdims)]


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
    """Sum elements of *x* over the given axis (or axes).

    Args:
        x: Input tensor.
        axis: Axis or axes to reduce. ``None`` reduces over all elements.
        keepdims: If ``True``, the reduced axes are kept as size-1 dimensions.

    Returns:
        Reduced tensor. When *axis* is ``None`` and *keepdims* is ``False``,
        a scalar tensor is returned.
    """
    return _multi_axis_reduce(_reduce_sum_op, x, axis=axis, keepdims=keepdims)


def mean(
    x: Tensor,
    *,
    axis: int | tuple[int, ...] | list[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute the arithmetic mean of *x* along the given axis (or axes).

    Internally implemented as ``sum(x) / n`` where *n* is the product of
    the reduced axis sizes. This ensures correct results across sharded tensors.

    Args:
        x: Input tensor.
        axis: Axis or axes to reduce. ``None`` averages over all elements.
        keepdims: If ``True``, the reduced axes are kept as size-1 dimensions.

    Returns:
        Tensor with the mean values.
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
    """Return the maximum value of *x* along the given axis (or axes).

    Args:
        x: Input tensor.
        axis: Axis or axes to reduce. ``None`` reduces over all elements.
        keepdims: If ``True``, the reduced axes are kept as size-1 dimensions.

    Returns:
        Tensor with maximum values.
    """
    return _multi_axis_reduce(_reduce_max_op, x, axis=axis, keepdims=keepdims)


def _physical_reduce(op, x: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    """Shared wrapper for physical reduce operations.
    
    MAX reduction ops always produce keepdims=True output (reduced axis = 1).
    We always pass keepdims=True to the op so compute_physical_shape matches.
    If the user wants keepdims=False, we add an explicit traced squeeze_physical
    call AFTER the reduction — this must be traced for autograd.
    """
    from .view.axes import squeeze_physical

    result = op([x], {"axis": axis, "keepdims": True})[0]
    if not keepdims:
        result = squeeze_physical(result, axis=axis)
    return result


def reduce_sum_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    """Sum along *axis* in the physical (sharded) tensor representation.

    Unlike :func:`reduce_sum`, this operates directly on the physical shape
    (including batch dimensions added by ``vmap``). It is used internally by
    transforms that need fine-grained control over the reduction axis.

    Args:
        x: Input tensor.
        axis: Physical axis index to reduce along.
        keepdims: If ``True``, the reduced axis is kept as size 1.

    Returns:
        Physically-reduced tensor.
    """
    return _physical_reduce(_reduce_sum_physical_op, x, axis, keepdims)


def mean_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    """Compute the mean along *axis* in the physical (sharded) tensor representation.

    Analogous to :func:`reduce_sum_physical` but divides by the axis size.
    Used internally by transforms operating on the physical layout.

    Args:
        x: Input tensor.
        axis: Physical axis index to reduce along.
        keepdims: If ``True``, the reduced axis is kept as size 1.

    Returns:
        Physically-averaged tensor.
    """
    return _physical_reduce(_mean_physical_op, x, axis, keepdims)


def reduce_max_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    return _physical_reduce(_reduce_max_physical_op, x, axis, keepdims)


def reduce_min(
    x: Tensor,
    *,
    axis: int | tuple[int, ...] | list[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Return the minimum value of *x* along the given axis (or axes).

    Args:
        x: Input tensor.
        axis: Axis or axes to reduce. ``None`` reduces over all elements.
        keepdims: If ``True``, the reduced axes are kept as size-1 dimensions.

    Returns:
        Tensor with minimum values.
    """
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
    """Return the indices of the maximum values along *axis*.

    Args:
        x: Input tensor.
        axis: Axis along which to find the maximum. Default: ``-1``.
        keepdims: If ``True``, the reduced axis is kept as a size-1 dimension.

    Returns:
        Integer tensor of dtype ``int64`` with the argmax indices.
    """
    res = _argmax_op([x], {"axis": axis})[0]
    if keepdims:
        from .view import unsqueeze

        res = unsqueeze(res, axis=axis)
    return res


def argmin(x: Tensor, axis: int = -1, keepdims: bool = False) -> Tensor:
    """Return the indices of the minimum values along *axis*.

    Args:
        x: Input tensor.
        axis: Axis along which to find the minimum. Default: ``-1``.
        keepdims: If ``True``, the reduced axis is kept as a size-1 dimension.

    Returns:
        Integer tensor of dtype ``int64`` with the argmin indices.
    """
    res = _argmin_op([x], {"axis": axis})[0]
    if keepdims:
        from .view import unsqueeze

        res = unsqueeze(res, axis=axis)
    return res


def cumsum(
    x: Tensor, axis: int = -1, exclusive: bool = False, reverse: bool = False
) -> Tensor:
    """Compute the cumulative sum of *x* along *axis*.

    Args:
        x: Input tensor.
        axis: Axis along which to accumulate. Default: ``-1``.
        exclusive: If ``True``, each element is the sum of all *preceding*
            elements (the first output element is ``0``).
        reverse: If ``True``, accumulate from right to left.

    Returns:
        Tensor of the same shape as *x* with cumulative sums.
    """
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
