# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from .base import AxisOp, Operation, ReduceOperation

if TYPE_CHECKING:
    from ..core.tensor import Tensor

from .view import SqueezePhysicalOp

_squeeze_physical_op = SqueezePhysicalOp()


class ReduceSumOp(ReduceOperation):
    @property
    def name(self) -> str:
        return "reduce_sum"

    def kernel(
        self, x: TensorValue, *, axis: int, keepdims: bool = False
    ) -> TensorValue:
        return ops.sum(x, axis)

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs: Any
    ) -> tuple[int, ...]:
        """Compute output shape for reduction."""
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        if keepdims:
            return tuple(1 if i == axis else d for i, d in enumerate(in_shape))

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for reduce_sum: broadcast cotangent back to input shape."""
        x = primals
        from ..ops.view.shape import broadcast_to

        return broadcast_to(cotangent, tuple(x.shape))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for reduce_sum: sum the tangents along the same axis."""
        t = tangents
        # Sum of tangents is the JVP
        axis = output.op_kwargs.get("axis", 0)
        return reduce_sum(t, axis=axis, keepdims=True)


class MeanOp(ReduceOperation):
    @property
    def name(self) -> str:
        return "mean"

    def kernel(
        self, x: TensorValue, *, axis: int, keepdims: bool = False
    ) -> TensorValue:
        return ops.mean(x, axis)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for mean: broadcast cotangent / axis_size."""
        x = primals

        # Get axis from kwargs if available (from trace)
        axis = output.op_kwargs.get("axis", 0)

        axis_size = x.shape[axis]
        from ..ops.view.shape import broadcast_to

        # Create target shape for broadcasting cotangent back to x's shape
        target_shape = tuple(int(d) for d in x.shape)
        return broadcast_to(cotangent, target_shape) / axis_size

    def __call__(self, x, *, axis: int, keepdims: bool = False):
        return super().__call__(x, axis=axis, keepdims=keepdims)

    def compute_cost(
        self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]
    ) -> float:
        """Mean: 1 sum + 1 div per output element."""
        if not input_shapes:
            return 0.0
        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        return float(num_elements) + (
            float(num_elements) / input_shapes[0][0] if input_shapes[0] else 0
        )

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs: Any
    ) -> tuple[int, ...]:
        """Compute output shape for reduction."""
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        if keepdims:
            return tuple(1 if i == axis else d for i, d in enumerate(in_shape))
        else:
            return tuple(d for i, d in enumerate(in_shape) if i != axis)


class ReduceMaxOp(ReduceOperation):
    @property
    def name(self) -> str:
        return "reduce_max"

    @property
    def collective_reduce_type(self) -> str:
        return "max"

    def kernel(
        self, x: TensorValue, *, axis: int, keepdims: bool = False
    ) -> TensorValue:
        return ops._reduce_max(x, axis=axis)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for reduce_max: one-hot mask where input == max."""
        x = primals
        from ..ops.comparison import equal
        from ..ops.view.shape import broadcast_to
        from ..ops.binary import mul

        # Broadcast output back to input shape for comparison
        max_broadcasted = broadcast_to(output, tuple(x.shape))
        mask = equal(x, max_broadcasted)  # 1.0 where x == max, 0.0 elsewhere
        cotangent_broadcasted = broadcast_to(cotangent, tuple(x.shape))
        return mul(cotangent_broadcasted, mask)

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs: Any
    ) -> tuple[int, ...]:
        """Compute output shape for reduction."""
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        if keepdims:
            return tuple(1 if i == axis else d for i, d in enumerate(in_shape))
        else:
            return tuple(d for i, d in enumerate(in_shape) if i != axis)


class ReduceSumPhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "reduce_sum_physical"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for this physical reduction op."""
        from ..core.sharding import spmd

        x = args[0]
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is None:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )
            in_shape = tuple(int(d) for d in s)
            norm_axis = axis if axis >= 0 else len(in_shape) + axis
            if keepdims:
                out_shape = tuple(
                    1 if i == norm_axis else d for i, d in enumerate(in_shape)
                )
            else:
                out_shape = tuple(d for i, d in enumerate(in_shape) if i != norm_axis)
            shapes.append(out_shape)

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.device_refs]
            else:
                devices = [mesh.device_refs[0]] * num_shards
        else:
            devices = [x.device] * num_shards

        return shapes, dtypes, devices

    def kernel(
        self, x: TensorValue, *, axis: int, keepdims: bool = False
    ) -> TensorValue:

        return ops.sum(x, axis=axis)

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
        """Compute output shape for reduction."""
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        if keepdims:
            return tuple(1 if i == axis else d for i, d in enumerate(in_shape))
        else:
            return tuple(d for i, d in enumerate(in_shape) if i != axis)


class MeanPhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "mean_physical"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for mean_physical."""
        from ..core.sharding import spmd

        x = args[0]
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is None:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )
            in_shape = tuple(int(d) for d in s)
            norm_axis = axis if axis >= 0 else len(in_shape) + axis
            if keepdims:
                out_shape = tuple(
                    1 if i == norm_axis else d for i, d in enumerate(in_shape)
                )
            else:
                out_shape = tuple(d for i, d in enumerate(in_shape) if i != norm_axis)
            shapes.append(out_shape)

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.device_refs]
            else:
                devices = [mesh.device_refs[0]] * num_shards
        else:
            devices = [x.device] * num_shards

        return shapes, dtypes, devices

    def kernel(
        self, x: TensorValue, *, axis: int, keepdims: bool = False
    ) -> TensorValue:

        return ops.mean(x, axis=axis)

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
        """Compute output shape for reduction."""
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        if keepdims:
            return tuple(1 if i == axis else d for i, d in enumerate(in_shape))
        else:
            return tuple(d for i, d in enumerate(in_shape) if i != axis)


class ReduceMaxPhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "reduce_max_physical"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for reduce_max_physical."""
        from ..core.sharding import spmd

        x = args[0]
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is None:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )
            in_shape = tuple(int(d) for d in s)
            norm_axis = axis if axis >= 0 else len(in_shape) + axis
            if keepdims:
                out_shape = tuple(
                    1 if i == norm_axis else d for i, d in enumerate(in_shape)
                )
            else:
                out_shape = tuple(d for i, d in enumerate(in_shape) if i != norm_axis)
            shapes.append(out_shape)

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.device_refs]
            else:
                devices = [mesh.device_refs[0]] * num_shards
        else:
            devices = [x.device] * num_shards

        return shapes, dtypes, devices

    @property
    def collective_reduce_type(self) -> str:
        return "max"

    def kernel(
        self, x: TensorValue, *, axis: int, keepdims: bool = False
    ) -> TensorValue:

        return ops._reduce_max(x, axis=axis)

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
        """Compute output shape for reduction."""
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        if keepdims:
            return tuple(1 if i == axis else d for i, d in enumerate(in_shape))
        else:
            return tuple(d for i, d in enumerate(in_shape) if i != axis)


class ReduceMinOp(ReduceOperation):
    @property
    def name(self) -> str:
        return "reduce_min"

    @property
    def collective_reduce_type(self) -> str:
        return "min"

    def kernel(
        self, x: TensorValue, *, axis: int, keepdims: bool = False
    ) -> TensorValue:
        return ops._reduce_min(x, axis=axis)

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs: Any
    ) -> tuple[int, ...]:
        """Compute output shape for reduction."""
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        if keepdims:
            return tuple(1 if i == axis else d for i, d in enumerate(in_shape))
        else:
            return tuple(d for i, d in enumerate(in_shape) if i != axis)


class ReduceMinPhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "reduce_min_physical"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for reduce_min_physical."""
        from ..core.sharding import spmd

        x = args[0]
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is None:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )
            in_shape = tuple(int(d) for d in s)
            norm_axis = axis if axis >= 0 else len(in_shape) + axis
            if keepdims:
                out_shape = tuple(
                    1 if i == norm_axis else d for i, d in enumerate(in_shape)
                )
            else:
                out_shape = tuple(d for i, d in enumerate(in_shape) if i != norm_axis)
            shapes.append(out_shape)

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.device_refs]
            else:
                devices = [mesh.device_refs[0]] * num_shards
        else:
            devices = [x.device] * num_shards

        return shapes, dtypes, devices

    @property
    def collective_reduce_type(self) -> str:
        return "min"

    def kernel(
        self, x: TensorValue, *, axis: int, keepdims: bool = False
    ) -> TensorValue:
        return ops._reduce_min(x, axis=axis)

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
        """Compute output shape for reduction."""
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        if keepdims:
            return tuple(1 if i == axis else d for i, d in enumerate(in_shape))
        else:
            return tuple(d for i, d in enumerate(in_shape) if i != axis)


_reduce_min_physical_op = ReduceMinPhysicalOp()
_reduce_min_op = ReduceMinOp()
_reduce_max_physical_op = ReduceMaxPhysicalOp()
_reduce_sum_physical_op = ReduceSumPhysicalOp()
_mean_physical_op = MeanPhysicalOp()
_reduce_sum_op = ReduceSumOp()
_mean_op = MeanOp()
_reduce_max_op = ReduceMaxOp()


def reduce_sum(
    x: Tensor,
    *,
    axis: int | tuple[int, ...] | list[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    from .view import squeeze

    if axis is None:
        axis = tuple(range(len(x.shape)))

    if isinstance(axis, (list, tuple)):
        axes = sorted(
            [ax if ax >= 0 else len(x.shape) + ax for ax in axis], reverse=True
        )
        res = x
        for ax in axes:
            res = _reduce_sum_op(res, axis=ax, keepdims=True)

        if not keepdims:
            for ax in axes:
                res = squeeze(res, axis=ax)
        return res

    result = _reduce_sum_op(x, axis=axis, keepdims=True)
    if not keepdims:
        result = squeeze(result, axis=axis)
    return result


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
    from .view import squeeze

    if axis is None:
        axis = tuple(range(len(x.shape)))

    if isinstance(axis, (list, tuple)):
        axes = sorted(
            [ax if ax >= 0 else len(x.shape) + ax for ax in axis], reverse=True
        )
        res = x
        for ax in axes:
            res = _reduce_max_op(res, axis=ax, keepdims=True)

        if not keepdims:
            for ax in axes:
                res = squeeze(res, axis=ax)
        return res

    result = _reduce_max_op(x, axis=axis, keepdims=True)
    if not keepdims:
        result = squeeze(result, axis=axis)
    return result


def reduce_sum_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:

    result = _reduce_sum_physical_op(x, axis=axis, keepdims=True)
    if not keepdims:
        result = _squeeze_physical_op(result, axis=axis)
    return result


def mean_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:

    result = _mean_physical_op(x, axis=axis, keepdims=True)
    if not keepdims:
        result = _squeeze_physical_op(result, axis=axis)
    return result


def reduce_max_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:

    result = _reduce_max_physical_op(x, axis=axis, keepdims=True)
    if not keepdims:
        result = _squeeze_physical_op(result, axis=axis)
    return result


def reduce_min(
    x: Tensor,
    *,
    axis: int | tuple[int, ...] | list[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    from .view import squeeze

    if axis is None:
        axis = tuple(range(len(x.shape)))

    if isinstance(axis, (list, tuple)):
        axes = sorted(
            [ax if ax >= 0 else len(x.shape) + ax for ax in axis], reverse=True
        )
        res = x
        for ax in axes:
            res = _reduce_min_op(res, axis=ax, keepdims=True)

        if not keepdims:
            for ax in axes:
                res = squeeze(res, axis=ax)
        return res

    result = _reduce_min_op(x, axis=axis, keepdims=True)
    if not keepdims:
        result = squeeze(result, axis=axis)
    return result


def reduce_min_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:

    result = _reduce_min_physical_op(x, axis=axis, keepdims=True)
    if not keepdims:
        result = _squeeze_physical_op(result, axis=axis)
    return result



class ArgmaxOp(AxisOp):
    """Indices of the maximum value along an axis."""

    @property
    def name(self) -> str:
        return "argmax"

    def kernel(self, x: TensorValue, *, axis: int) -> TensorValue:
        rank = len(x.shape)
        if axis < 0:
            axis += rank
        
        if axis != rank - 1:
            perm = list(range(rank))
            perm[axis], perm[-1] = perm[-1], perm[axis]
            x = ops.permute(x, tuple(perm))
            # After swapping, the dimension to reduce is at -1
            res = ops.argmax(x, axis=-1)
            return ops.squeeze(res, -1)
            
        res = ops.argmax(x, axis=axis)
        return ops.squeeze(res, axis)

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
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
            s = x.physical_local_shape(idx)
            if s is None:
                raise RuntimeError("Could not determine physical shape")
            in_shape = tuple(int(d) for d in s)
            norm_axis = axis if axis >= 0 else len(in_shape) + axis
            out_shape = tuple(d for i, d in enumerate(in_shape) if i != norm_axis)
            shapes.append(out_shape)

        return shapes, [DType.int64] * num_shards, [x.device] * num_shards

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        return (None,)

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs: Any
    ) -> tuple[int, ...]:
        axis = kwargs.get("axis", -1)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        return tuple(d for i, d in enumerate(in_shape) if i != axis)


class ArgminOp(AxisOp):
    """Indices of the minimum value along an axis."""

    @property
    def name(self) -> str:
        return "argmin"

    def kernel(self, x: TensorValue, *, axis: int) -> TensorValue:
        rank = len(x.shape)
        if axis < 0:
            axis += rank
        
        if axis != rank - 1:
            perm = list(range(rank))
            perm[axis], perm[-1] = perm[-1], perm[axis]
            x = ops.permute(x, tuple(perm))
            return ops.squeeze(ops.argmin(x, axis=-1), -1)

        return ops.squeeze(ops.argmin(x, axis=axis), axis)

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
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
            s = x.physical_local_shape(idx)
            if s is None:
                raise RuntimeError("Could not determine physical shape")
            in_shape = tuple(int(d) for d in s)
            norm_axis = axis if axis >= 0 else len(in_shape) + axis
            out_shape = tuple(d for i, d in enumerate(in_shape) if i != norm_axis)
            shapes.append(out_shape)

        return shapes, [DType.int64] * num_shards, [x.device] * num_shards

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        return (None,)

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs: Any
    ) -> tuple[int, ...]:
        axis = kwargs.get("axis", -1)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        return tuple(d for i, d in enumerate(in_shape) if i != axis)


class CumsumOp(AxisOp):
    """Cumulative sum along an axis."""

    @property
    def name(self) -> str:
        return "cumsum"

    def kernel(
        self, x: TensorValue, *, axis: int, exclusive: bool = False, reverse: bool = False
    ) -> TensorValue:
        return ops.cumsum(x, axis=axis, exclusive=exclusive, reverse=reverse)

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        x = args[0]
        shapes = []
        for i in range(x.num_shards):
            shapes.append(tuple(int(d) for d in x.physical_local_shape(i)))
        return shapes, [x.dtype] * x.num_shards, [x.device] * x.num_shards

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP: flip(cumsum(flip(cotangent, axis), axis), axis)."""
        axis = output.op_kwargs.get("axis", -1)

        from ..ops.view.axes import flip

        return flip(cumsum(flip(cotangent, axis=axis), axis=axis), axis=axis)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP: cumsum(tangent, axis)."""
        axis = output.op_kwargs.get("axis", -1)
        return cumsum(tangents, axis=axis)

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
    from .view import squeeze

    res = _argmax_op(x, axis=axis)
    if keepdims:
        from .view import unsqueeze

        res = unsqueeze(res, axis=axis)
    return res


def argmin(x: Tensor, axis: int = -1, keepdims: bool = False) -> Tensor:
    res = _argmin_op(x, axis=axis)
    if keepdims:
        from .view import unsqueeze

        res = unsqueeze(res, axis=axis)
    return res


def cumsum(
    x: Tensor, axis: int = -1, exclusive: bool = False, reverse: bool = False
) -> Tensor:
    return _cumsum_op(x, axis=axis, exclusive=exclusive, reverse=reverse)


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