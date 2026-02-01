# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from ..base import AxisOp, Operation

if TYPE_CHECKING:
    from ...core import Tensor


class UnsqueezeOp(AxisOp):
    axis_offset_for_insert = True

    @property
    def name(self) -> str:
        return "unsqueeze"

    def kernel(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.unsqueeze(x, axis)

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for unsqueeze."""
        from ...core.sharding import spmd

        x = args[0]
        # Kwargs are already adapted in Operation.__call__
        axis = kwargs.get("axis", 0)

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is not None:
                in_shape = list(int(d) for d in s)
                norm_axis = axis if axis >= 0 else len(in_shape) + 1 + axis
                in_shape.insert(norm_axis, 1)
                shapes.append(tuple(in_shape))
            else:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.devices]
            else:
                devices = [mesh.devices[0]] * num_shards
        else:
            devices = [x.device] * (num_shards or 1)

        return shapes, dtypes, devices

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

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for unsqueeze: squeeze the cotangent at the unsqueezed axis."""
        axis = output.op_kwargs.get("axis", 0)
        from . import squeeze

        return squeeze(cotangent, axis=axis)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for unsqueeze: unsqueeze the tangent."""
        axis = output.op_kwargs.get("axis", 0)
        from . import unsqueeze

        return unsqueeze(tangents, axis=axis)


class SqueezeOp(AxisOp):
    axis_offset_for_insert = False

    @property
    def name(self) -> str:
        return "squeeze"

    def kernel(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.squeeze(x, axis)

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for squeeze."""
        from ...core.sharding import spmd

        x = args[0]
        # Kwargs are already adapted in Operation.__call__
        axis = kwargs.get("axis", 0)

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is not None:
                in_shape = list(int(d) for d in s)
                norm_axis = axis if axis >= 0 else len(in_shape) + axis
                in_shape.pop(norm_axis)
                shapes.append(tuple(in_shape))
            else:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.devices]
            else:
                devices = [mesh.devices[0]] * num_shards
        else:
            devices = [x.device] * (num_shards or 1)

        return shapes, dtypes, devices

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
        out_factors.pop(axis)
        out_str = " ".join(out_factors)

        return OpShardingRuleTemplate.parse(
            f"{in_str} -> {out_str}", input_shapes
        ).instantiate(input_shapes, output_shapes)

    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0]) - 1

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

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for squeeze: unsqueeze the cotangent at the squeezed axis."""
        axis = output.op_kwargs.get("axis", 0)
        from . import unsqueeze

        return unsqueeze(cotangent, axis=axis)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for squeeze: squeeze the tangent."""
        axis = output.op_kwargs.get("axis", 0)
        from . import squeeze

        return squeeze(tangents, axis=axis)


class SwapAxesOp(AxisOp):
    axis_arg_names = ("axis1", "axis2")

    @property
    def name(self) -> str:
        return "swap_axes"

    def kernel(self, x: TensorValue, *, axis1: int, axis2: int) -> TensorValue:
        return ops.transpose(x, axis1, axis2)

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for swap_axes."""
        from ...core.sharding import spmd

        x = args[0]
        # Kwargs are already adapted in Operation.__call__
        axis1 = kwargs.get("axis1", 0)
        axis2 = kwargs.get("axis2", 1)

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is not None:
                in_shape = list(int(d) for d in s)
                norm_axis1 = axis1 if axis1 >= 0 else len(in_shape) + axis1
                norm_axis2 = axis2 if axis2 >= 0 else len(in_shape) + axis2
                in_shape[norm_axis1], in_shape[norm_axis2] = (
                    in_shape[norm_axis2],
                    in_shape[norm_axis1],
                )
                shapes.append(tuple(in_shape))
            else:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.devices]
            else:
                devices = [mesh.devices[0]] * num_shards
        else:
            devices = [x.device] * (num_shards or 1)

        return shapes, dtypes, devices

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for swap_axes: swap back."""
        # Transpose is self-inverse
        axis1 = output.op_kwargs.get("axis1")
        axis2 = output.op_kwargs.get("axis2")
        return swap_axes(cotangent, axis1=axis1, axis2=axis2)

    def __call__(self, x, *, axis1: int, axis2: int):
        return super().__call__(x, axis1=axis1, axis2=axis2)

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
    "MoveAxisOp",
    "UnsqueezePhysicalOp",
    "SqueezePhysicalOp",
    "moveaxis",
    "unsqueeze_physical",
    "squeeze_physical",
]


def unsqueeze(x: Tensor, axis: int = 0) -> Tensor:
    return _unsqueeze_op(x, axis=axis)


def squeeze(x: Tensor, axis: int = 0) -> Tensor:
    return _squeeze_op(x, axis=axis)


def swap_axes(x: Tensor, axis1: int, axis2: int) -> Tensor:
    return _swap_axes_op(x, axis1=axis1, axis2=axis2)


class MoveAxisOp(AxisOp):
    axis_arg_names = ("source", "destination")

    @property
    def name(self) -> str:
        return "moveaxis"

    def kernel(self, x: TensorValue, *, source: int, destination: int) -> TensorValue:
        rank = len(x.type.shape)
        if source < 0:
            source = rank + source
        if destination < 0:
            destination = rank + destination

        order = list(range(rank))
        order.pop(source)
        order.insert(destination, source)
        return ops.permute(x, tuple(order))

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for moveaxis."""
        from ...core.sharding import spmd

        x = args[0]
        source = kwargs.get("source")
        destination = kwargs.get("destination")

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is not None:
                in_shape = list(int(d) for d in s)
                rank = len(in_shape)
                norm_source = source if source >= 0 else rank + source
                norm_dest = destination if destination >= 0 else rank + destination
                order = list(range(rank))
                order.pop(norm_source)
                order.insert(norm_dest, norm_source)
                out_shape = tuple(in_shape[j] for j in order)
                shapes.append(out_shape)
            else:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.devices]
            else:
                devices = [mesh.devices[0]] * num_shards
        else:
            devices = [x.device] * (num_shards or 1)

        return shapes, dtypes, devices

    def __call__(self, x: Tensor, *, source: int, destination: int) -> Tensor:
        return super().__call__(x, source=source, destination=destination)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        source = output.op_kwargs.get("source")
        destination = output.op_kwargs.get("destination")
        # Inverse: move from destination back to source
        return moveaxis(cotangent, source=destination, destination=source)

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


class UnsqueezePhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "unsqueeze_physical"

    def kernel(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.unsqueeze(x, axis)

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for unsqueeze_physical."""
        from ...core.sharding import spmd

        x = args[0]
        axis = kwargs.get("axis", 0)

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is not None:
                in_shape = list(int(d) for d in s)
                norm_axis = axis if axis >= 0 else len(in_shape) + axis
                in_shape.insert(norm_axis, 1)
                shapes.append(tuple(in_shape))
            else:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.devices]
            else:
                devices = [mesh.devices[0]] * num_shards
        else:
            devices = [x.device] * (num_shards or 1)

        return shapes, dtypes, devices

    def __call__(self, x: Tensor, *, axis: int = 0) -> Tensor:
        return super().__call__(x, axis=axis)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        axis = output.op_kwargs.get("axis", 0)
        return squeeze_physical(cotangent, axis=axis)

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


class SqueezePhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "squeeze_physical"

    def kernel(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.squeeze(x, axis)

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for squeeze_physical."""
        from ...core.sharding import spmd

        x = args[0]
        axis = kwargs.get("axis", 0)

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is not None:
                in_shape = list(int(d) for d in s)
                norm_axis = axis if axis >= 0 else len(in_shape) + axis
                in_shape.pop(norm_axis)
                shapes.append(tuple(in_shape))
            else:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.devices]
            else:
                devices = [mesh.devices[0]] * num_shards
        else:
            devices = [x.device] * (num_shards or 1)

        return shapes, dtypes, devices

    def __call__(self, x: Tensor, *, axis: int = 0) -> Tensor:
        return super().__call__(x, axis=axis)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        axis = output.op_kwargs.get("axis", 0)
        return unsqueeze_physical(cotangent, axis=axis)

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
    return _moveaxis_op(x, source=source, destination=destination)


def unsqueeze_physical(x: Tensor, axis: int = 0) -> Tensor:
    return _unsqueeze_physical_op(x, axis=axis)


def squeeze_physical(x: Tensor, axis: int = 0) -> Tensor:
    return _squeeze_physical_op(x, axis=axis)
