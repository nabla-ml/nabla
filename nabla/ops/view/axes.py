# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from max.graph import ops

from ..base import AxisOp, OpArgs, Operation, OpKwargs, OpResult, OpTensorValues

if TYPE_CHECKING:
    from ...core import Tensor


def _axis_compute_shapes(
    op: Operation, args: OpArgs, kwargs: OpKwargs, transform_fn: Any
) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
    """Shared boilerplate for axis-op compute_physical_shape.

    *transform_fn(in_shape_list, kwargs)* mutates/returns the output shape list.
    """
    from ...core.sharding import spmd

    x = args[0]
    mesh = spmd.get_mesh_from_args(args)
    num_shards = len(mesh.devices) if mesh else 1

    shapes = []
    for i in range(num_shards):
        idx = i if i < x.num_shards else 0
        in_shape = x.physical_local_shape_ints(idx)
        if in_shape is None:
            raise RuntimeError(f"Could not determine physical shape for {op.name}")
        shapes.append(tuple(transform_fn(list(in_shape), kwargs)))
    dtypes, devices = op._build_shard_metadata(x, mesh, num_shards)
    return shapes, dtypes, devices


class UnsqueezeOp(AxisOp):
    axis_offset_for_insert = True

    @property
    def name(self) -> str:
        return "unsqueeze"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis = kwargs.get("axis", 0)
        return [ops.unsqueeze(x, axis)]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for unsqueeze."""

        def _transform(s, kw):
            axis = kw.get("axis", 0)
            norm = axis if axis >= 0 else len(s) + 1 + axis
            s.insert(norm, 1)
            return s

        return _axis_compute_shapes(self, args, kwargs, _transform)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        from ...core.sharding.propagation import OpShardingRuleTemplate

        factors = [f"d{i}" for i in range(len(input_shapes[0]))]
        in_str = " ".join(factors)

        axis = kwargs.get("axis", 0)
        out_factors = list(factors)
        out_factors.insert(axis, "new_dim")
        out_str = " ".join(out_factors)

        return OpShardingRuleTemplate.parse(
            f"{in_str} -> {out_str}", input_shapes
        ).instantiate(input_shapes, output_shapes)

    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0]) + 1

    def infer_sharding_spec(
        self, args: tuple, mesh: Any, kwargs: dict = None
    ) -> tuple[Any | None, list[Any | None], bool]:
        """Insert empty dim spec at unsqueezed axis."""
        from ...core import Tensor
        from ...core.sharding.spec import DimSpec

        if not args:
            return None, [], False

        x = args[0]
        if isinstance(x, Tensor) and x.sharding:
            new_dim_specs = list(x.sharding.dim_specs)
            axis = kwargs.get("axis", 0)
            if axis < 0:
                axis += len(x.shape) + 1

            new_dim_specs.insert(axis, DimSpec([]))

            new_spec = x.sharding.clone()
            new_spec.dim_specs = new_dim_specs
            return new_spec, [x.sharding], False

        return None, [None], False

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for unsqueeze: squeeze the cotangent at the unsqueezed axis."""
        axis = kwargs.get("axis", 0)
        from . import squeeze

        return [squeeze(cotangents[0], axis=axis)]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP for unsqueeze: unsqueeze the tangent."""
        axis = kwargs.get("axis", 0)
        from . import unsqueeze

        return [unsqueeze(tangents[0], axis=axis)]


class SqueezeOp(AxisOp):
    axis_offset_for_insert = False

    @property
    def name(self) -> str:
        return "squeeze"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis = kwargs.get("axis", 0)
        if axis is None:
            return [ops.squeeze(x)]
        if isinstance(axis, int):
            return [ops.squeeze(x, axis)]

        # Squeeze multiple axes by chaining
        axes = sorted([a if a >= 0 else len(x.shape) + a for a in axis], reverse=True)
        res = x
        for a in axes:
            res = ops.squeeze(res, a)
        return [res]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for squeeze."""

        def _transform(s, kw):
            axis = kw.get("axis", 0)
            if axis is None:
                axes = [i for i, d in enumerate(s) if d == 1]
            elif isinstance(axis, int):
                norm = axis if axis >= 0 else len(s) + axis
                axes = [norm] if 0 <= norm < len(s) else []
            else:
                axes = []
                for a in axis:
                    norm = a if a >= 0 else len(s) + a
                    if 0 <= norm < len(s):
                        axes.append(norm)
            for a in sorted(axes, reverse=True):
                s.pop(a)
            return s

        return _axis_compute_shapes(self, args, kwargs, _transform)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        from ...core.sharding.propagation import OpShardingRuleTemplate

        factors = [f"d{i}" for i in range(len(input_shapes[0]))]
        in_str = " ".join(factors)

        axis = kwargs.get("axis", 0)
        out_factors = list(factors)
        if axis is None:
            # Squeeze all size-1 dims
            in_shape = input_shapes[0]
            axes = [i for i, d in enumerate(in_shape) if d == 1]
        elif isinstance(axis, int):
            axes = [axis if axis >= 0 else len(factors) + axis]
        else:
            axes = [a if a >= 0 else len(factors) + a for a in axis]

        for a in sorted(axes, reverse=True):
            out_factors.pop(a)

        out_str = " ".join(out_factors)

        return OpShardingRuleTemplate.parse(
            f"{in_str} -> {out_str}", input_shapes
        ).instantiate(input_shapes, output_shapes)

    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        axis = kwargs.get("axis", 0)
        if axis is None:
            in_shape = input_shapes[0]
            squeezed_count = sum(1 for d in in_shape if d == 1)
            return len(in_shape) - squeezed_count
        if isinstance(axis, int):
            return len(input_shapes[0]) - 1
        return len(input_shapes[0]) - len(axis)

    def infer_sharding_spec(
        self, args: tuple, mesh: Any, kwargs: dict = None
    ) -> tuple[Any | None, list[Any | None], bool]:
        """Remove dim spec at squeezed axis."""
        from ...core import Tensor

        if not args:
            return None, [], False

        x = args[0]
        if isinstance(x, Tensor) and x.sharding:
            axis = kwargs.get("axis", 0)
            if axis < 0:
                axis += len(x.shape)

            new_dim_specs = list(x.sharding.dim_specs)

            # Squeezing a sharded dimension is undefined unless it's handled
            # (e.g. implicitly replicated or partial).
            # For now we assume we just drop the spec.
            # If the dimension was sharded, we might lose info unless we track it?
            # But usually squeeze(1) implies size 1.
            if axis < len(new_dim_specs):
                new_dim_specs.pop(axis)

            new_spec = x.sharding.clone()
            new_spec.dim_specs = new_dim_specs
            return new_spec, [x.sharding], False

        return None, [None], False

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for squeeze: unsqueeze the cotangent at the squeezed axis."""
        axis = kwargs.get("axis", 0)
        from . import unsqueeze

        # axis may be a tuple when multiple axes are squeezed — unsqueeze each
        if isinstance(axis, (tuple, list)):
            result = cotangents[0]
            # Insert axes back in ascending order
            for a in sorted(axis):
                result = unsqueeze(result, axis=a)
            return [result]
        return [unsqueeze(cotangents[0], axis=axis)]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP for squeeze: squeeze the tangent."""
        axis = kwargs.get("axis", 0)
        from . import squeeze

        return [squeeze(tangents[0], axis=axis)]


class SwapAxesOp(AxisOp):
    axis_arg_names = ("axis1", "axis2")

    @property
    def name(self) -> str:
        return "swap_axes"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis1 = kwargs["axis1"]
        axis2 = kwargs["axis2"]
        return [ops.transpose(x, axis1, axis2)]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for swap_axes."""

        def _transform(s, kw):
            a1 = kw.get("axis1", 0)
            a2 = kw.get("axis2", 1)
            n1 = a1 if a1 >= 0 else len(s) + a1
            n2 = a2 if a2 >= 0 else len(s) + a2
            s[n1], s[n2] = s[n2], s[n1]
            return s

        return _axis_compute_shapes(self, args, kwargs, _transform)

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        axis1 = kwargs.get("axis1")
        axis2 = kwargs.get("axis2")
        return [swap_axes(cotangents[0], axis1=axis1, axis2=axis2)]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        axis1 = kwargs.get("axis1")
        axis2 = kwargs.get("axis2")
        return [swap_axes(tangents[0], axis1=axis1, axis2=axis2)]

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        return super().__call__(args, kwargs)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        from ...core.sharding.propagation import OpShardingRuleTemplate

        rank = len(input_shapes[0])
        axis1 = kwargs.get("axis1", 0)
        axis2 = kwargs.get("axis2", 1)
        if axis1 < 0:
            axis1 += rank
        if axis2 < 0:
            axis2 += rank

        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)

        factors[axis1], factors[axis2] = factors[axis2], factors[axis1]
        out_str = " ".join(factors)

        return OpShardingRuleTemplate.parse(
            f"{in_str} -> {out_str}", input_shapes
        ).instantiate(input_shapes, output_shapes)

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs
    ) -> tuple[int, ...]:
        """Swap dimensions at axis1 and axis2."""
        in_shape = list(input_shapes[0])
        axis1 = kwargs.get("axis1", 0)
        axis2 = kwargs.get("axis2", 1)
        if axis1 < 0:
            axis1 = len(in_shape) + axis1
        if axis2 < 0:
            axis2 = len(in_shape) + axis2
        in_shape[axis1], in_shape[axis2] = in_shape[axis2], in_shape[axis1]
        return tuple(in_shape)


_unsqueeze_op = UnsqueezeOp()
_squeeze_op = SqueezeOp()
_swap_axes_op = SwapAxesOp()

__all__ = [
    "unsqueeze",
    "squeeze",
    "swap_axes",
    "transpose",
    "MoveAxisOp",
    "UnsqueezePhysicalOp",
    "SqueezePhysicalOp",
    "FlipOp",
    "PermuteOp",
    "moveaxis",
    "unsqueeze_physical",
    "squeeze_physical",
    "flip",
    "permute",
]


def unsqueeze(x: Tensor, axis: int = 0) -> Tensor:
    """Insert a size-1 dimension at *axis* into *x*'s shape.

    Args:
        x: Input tensor.
        axis: Position at which to insert the new dimension.
            Supports negative indexing.

    Returns:
        Tensor with one additional dimension of size 1.
    """
    return _unsqueeze_op([x], {"axis": axis})[0]


def squeeze(x: Tensor, axis: int = 0) -> Tensor:
    """Remove the size-1 dimension at *axis* from *x*'s shape.

    Args:
        x: Input tensor. The dimension at *axis* must be 1.
        axis: Dimension to remove. Supports negative indexing.
            Pass ``None`` to squeeze all size-1 dimensions.

    Returns:
        Tensor with the specified dimension removed.
    """
    return _squeeze_op([x], {"axis": axis})[0]


def swap_axes(x: Tensor, axis1: int, axis2: int) -> Tensor:
    """Swap (transpose) two dimensions of *x*.

    Args:
        x: Input tensor.
        axis1: First axis. Supports negative indexing.
        axis2: Second axis. Supports negative indexing.

    Returns:
        View with *axis1* and *axis2* swapped.
    """
    return _swap_axes_op([x], {"axis1": axis1, "axis2": axis2})[0]


transpose = swap_axes


class MoveAxisOp(AxisOp):
    axis_arg_names = ("source", "destination")

    @property
    def name(self) -> str:
        return "moveaxis"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        source = kwargs["source"]
        destination = kwargs["destination"]
        rank = len(x.type.shape)
        if source < 0:
            source = rank + source
        if destination < 0:
            destination = rank + destination

        order = list(range(rank))
        order.pop(source)
        order.insert(destination, source)
        return [ops.permute(x, tuple(order))]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for moveaxis."""

        def _transform(s, kw):
            src = kw.get("source")
            dst = kw.get("destination")
            rank = len(s)
            ns = src if src >= 0 else rank + src
            nd = dst if dst >= 0 else rank + dst
            order = list(range(rank))
            order.pop(ns)
            order.insert(nd, ns)
            return [s[j] for j in order]

        return _axis_compute_shapes(self, args, kwargs, _transform)

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        return super().__call__(args, kwargs)

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        source = kwargs.get("source")
        destination = kwargs.get("destination")
        return [moveaxis(cotangents[0], source=destination, destination=source)]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        source = kwargs.get("source")
        destination = kwargs.get("destination")
        return [moveaxis(tangents[0], source=source, destination=destination)]

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        from ...core.sharding.propagation import OpShardingRuleTemplate

        rank = len(input_shapes[0])
        source = kwargs.get("source")
        destination = kwargs.get("destination")

        if source < 0:
            source += rank
        if destination < 0:
            destination += rank

        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)

        perm = list(factors)
        val = perm.pop(source)
        perm.insert(destination, val)
        out_str = " ".join(perm)

        return OpShardingRuleTemplate.parse(
            f"{in_str} -> {out_str}", input_shapes
        ).instantiate(input_shapes, output_shapes)


class UnsqueezePhysicalOp(AxisOp):
    @property
    def name(self) -> str:
        return "unsqueeze_physical"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis = kwargs.get("axis", 0)
        return [ops.unsqueeze(x, axis)]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for unsqueeze_physical."""

        def _transform(s, kw):
            axis = kw.get("axis", 0)
            norm = axis if axis >= 0 else len(s) + axis
            s.insert(norm, 1)
            return s

        return _axis_compute_shapes(self, args, kwargs, _transform)

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        return super().__call__(args, kwargs)

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        axis = kwargs.get("axis", 0)
        return [squeeze_physical(cotangents[0], axis=axis)]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        axis = kwargs.get("axis", 0)
        return [unsqueeze_physical(tangents[0], axis=axis)]

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        from ...core.sharding.propagation import OpShardingRuleTemplate

        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)

        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)

        out_factors = list(factors)
        out_factors.insert(axis, "new_dim")
        out_str = " ".join(out_factors)

        return OpShardingRuleTemplate.parse(
            f"{in_str} -> {out_str}", input_shapes
        ).instantiate(input_shapes, output_shapes)

    def infer_output_rank(
        self, input_shapes: tuple[tuple[int, ...], ...], **kwargs
    ) -> int:
        return len(input_shapes[0]) + 1


class SqueezePhysicalOp(AxisOp):
    @property
    def name(self) -> str:
        return "squeeze_physical"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis = kwargs.get("axis", 0)
        return [ops.squeeze(x, axis)]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for squeeze_physical."""

        def _transform(s, kw):
            axis = kw.get("axis", 0)
            norm = axis if axis >= 0 else len(s) + axis
            if os.environ.get("NABLA_DEBUG_PHYS", "0") == "1":
                print(f"[NABLA_DEBUG_PHYS] {self.name}._transform: in_shape={s} axis={axis} norm={norm}")
            s.pop(norm)
            return s

        return _axis_compute_shapes(self, args, kwargs, _transform)

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        return super().__call__(args, kwargs)

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        axis = kwargs.get("axis", 0)
        return [unsqueeze_physical(cotangents[0], axis=axis)]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        axis = kwargs.get("axis", 0)
        return [squeeze_physical(tangents[0], axis=axis)]

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        from ...core.sharding.propagation import OpShardingRuleTemplate

        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)

        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)

        out_factors = list(factors)
        out_factors.pop(axis)
        out_str = " ".join(out_factors)

        return OpShardingRuleTemplate.parse(
            f"{in_str} -> {out_str}", input_shapes
        ).instantiate(input_shapes, output_shapes)

    def infer_output_rank(
        self, input_shapes: tuple[tuple[int, ...], ...], **kwargs
    ) -> int:
        return len(input_shapes[0]) - 1


_moveaxis_op = MoveAxisOp()
_unsqueeze_physical_op = UnsqueezePhysicalOp()
_squeeze_physical_op = SqueezePhysicalOp()


def moveaxis(x: Tensor, source: int, destination: int) -> Tensor:
    """Move axis *source* to position *destination*.

    Args:
        x: Input tensor.
        source: Original axis position. Supports negative indexing.
        destination: Target axis position. Supports negative indexing.

    Returns:
        Tensor with the axis at *source* moved to *destination*.
    """
    return _moveaxis_op([x], {"source": source, "destination": destination})[0]


def unsqueeze_physical(x: Tensor, axis: int = 0) -> Tensor:
    """Insert a size-1 dimension at *axis* in the **physical** tensor layout.

    Unlike :func:`unsqueeze`, this operates on the physical shape (which
    includes batch dimensions added by ``vmap``). Used internally by
    transforms that manipulate the physical layout directly.
    """
    return _unsqueeze_physical_op([x], {"axis": axis})[0]


def squeeze_physical(x: Tensor, axis: int = 0) -> Tensor:
    """Remove the size-1 dimension at *axis* in the **physical** tensor layout.

    Counterpart to :func:`unsqueeze_physical`. Used internally by transforms.
    """
    return _squeeze_physical_op([x], {"axis": axis})[0]


class FlipOp(AxisOp):
    """Flip a tensor along a specified axis."""

    @property
    def name(self) -> str:
        return "flip"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        axis = kwargs["axis"]
        return [ops.flip(x, axis)]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        x = args[0]
        shapes = [x.physical_local_shape_ints(i) for i in range(x.num_shards)]
        return shapes, [x.dtype] * x.num_shards, [x.device] * x.num_shards

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        axis = kwargs.get("axis")
        return [flip(cotangents[0], axis=axis)]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        axis = kwargs.get("axis")
        return [flip(tangents[0], axis=axis)]

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        # Flip is elementwise, sharding is preserved
        from ...core.sharding.propagation import OpShardingRuleTemplate

        rank = len(input_shapes[0])
        mapping = {i: [f"d{i}"] for i in range(rank)}
        return OpShardingRuleTemplate([mapping], [mapping]).instantiate(
            input_shapes, output_shapes
        )

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs: Any
    ) -> tuple[int, ...]:
        return input_shapes[0]


class PermuteOp(Operation):
    """Permute the dimensions of a tensor according to a given order."""

    @property
    def name(self) -> str:
        return "permute"

    def adapt_kwargs(self, args: OpArgs, kwargs: OpKwargs, batch_dims: int) -> dict:
        order = kwargs.get("order")
        if order is None:
            return kwargs
        if batch_dims > 0:
            order = tuple(range(batch_dims)) + tuple(batch_dims + i for i in order)
        return {**kwargs, "order": order}

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        order = kwargs["order"]
        return [ops.permute(x, order)]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        x = args[0]
        order = kwargs.get("order")
        shapes = []
        for i in range(x.num_shards):
            local_shape = x.physical_local_shape_ints(i)
            shapes.append(tuple(local_shape[j] for j in order))
        return shapes, [x.dtype] * x.num_shards, [x.device] * x.num_shards

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        order = kwargs.get("order")
        inv_order = [0] * len(order)
        for i, p in enumerate(order):
            inv_order[p] = i
        return [permute(cotangents[0], order=tuple(inv_order))]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        order = kwargs.get("order")
        return [permute(tangents[0], order=order)]

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        from ...core.sharding.propagation import OpShardingRuleTemplate

        order = kwargs.get("order")
        rank = len(input_shapes[0])
        in_factors = [f"d{i}" for i in range(rank)]
        out_factors = [in_factors[i] for i in order]

        in_mapping = {i: [in_factors[i]] for i in range(rank)}
        out_mapping = {i: [out_factors[i]] for i in range(rank)}

        return OpShardingRuleTemplate([in_mapping], [out_mapping]).instantiate(
            input_shapes, output_shapes
        )

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs: Any
    ) -> tuple[int, ...]:
        in_shape = input_shapes[0]
        order = kwargs.get("order")
        return tuple(in_shape[i] for i in order)


_flip_op = FlipOp()
_permute_op = PermuteOp()


def flip(x: Tensor, axis: int) -> Tensor:
    """Reverse the elements of *x* along the specified axis.

    Args:
        x: Input tensor.
        axis: The axis along which to reverse. Supports negative indexing.

    Returns:
        Tensor with elements reversed along *axis*. Shape is unchanged.
    """
    return _flip_op([x], {"axis": axis})[0]


def permute(x: Tensor, order: tuple[int, ...]) -> Tensor:
    """Reorder the dimensions of *x* according to *order*.

    Args:
        x: Input tensor of rank ``N``.
        order: A permutation of ``(0, 1, ..., N-1)`` giving the new dimension
            ordering. Equivalent to NumPy's ``transpose(axes=order)``.

    Returns:
        Tensor with dimensions reordered as specified.
    """
    return _permute_op([x], {"order": order})[0]
