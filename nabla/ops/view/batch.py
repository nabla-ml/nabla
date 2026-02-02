# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Batch dimension manipulation operations.

These operations manipulate the batch_dims metadata and perform the necessary
physical axis permutations to move axes in/out of the batch dimension region.

Key insight: move_axis_to/from_batch_dims are composed from simpler primitives:
- moveaxis_physical (physical axis permutation)
- incr_batch_dims / decr_batch_dims (batch_dims metadata adjustment)

This means sharding rules are inherited from the underlying moveaxis operation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from ..base import Operation

if TYPE_CHECKING:
    from ...core.tensor import Tensor


class IncrBatchDimsOp(Operation):
    """Increment batch_dims counter without changing data layout."""

    @property
    def name(self) -> str:
        return "incr_batch_dims"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Physical shape is unchanged; only batch_dims metadata changes."""
        from ...core.sharding import spmd

        x = args[0]
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
            shapes.append(tuple(int(d) for d in s))

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.devices]
            else:
                devices = [mesh.devices[0]] * num_shards
        else:
            devices = [x.device] * num_shards

        return shapes, dtypes, devices

    def kernel(self, x: TensorValue) -> TensorValue:
        return x

    def __call__(self, x: Tensor) -> Tensor:
        result = super().__call__(x)
        result._impl.batch_dims = x.batch_dims + 1
        return result

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        return decr_batch_dims(cotangent)


class DecrBatchDimsOp(Operation):
    """Decrement batch_dims counter without changing data layout."""

    @property
    def name(self) -> str:
        return "decr_batch_dims"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Physical shape is unchanged; only batch_dims metadata changes."""
        from ...core.sharding import spmd

        x = args[0]
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
            shapes.append(tuple(int(d) for d in s))

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.devices]
            else:
                devices = [mesh.devices[0]] * num_shards
        else:
            devices = [x.device] * num_shards

        return shapes, dtypes, devices

    def kernel(self, x: TensorValue) -> TensorValue:
        return x

    def __call__(self, x: Tensor) -> Tensor:
        if x.batch_dims <= 0:
            raise ValueError("Cannot decrement batch_dims below 0")
        result = super().__call__(x)
        result._impl.batch_dims = x.batch_dims - 1
        return result

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        return incr_batch_dims(cotangent)


class MoveAxisPhysicalOp(Operation):
    """Move a physical axis to another physical position, preserving batch_dims."""

    @property
    def name(self) -> str:
        return "moveaxis_physical"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for moveaxis_physical."""
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
            if s is None:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )

            in_shape = list(int(d) for d in s)
            rank = len(in_shape)
            norm_source = source if source >= 0 else rank + source
            norm_dest = destination if destination >= 0 else rank + destination

            order = list(range(rank))
            order.pop(norm_source)
            order.insert(norm_dest, norm_source)
            out_shape = tuple(in_shape[j] for j in order)
            shapes.append(out_shape)

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.devices]
            else:
                devices = [mesh.devices[0]] * num_shards
        else:
            devices = [x.device] * num_shards

        return shapes, dtypes, devices

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

    def __call__(self, x: Tensor, *, source: int, destination: int) -> Tensor:
        # Preserve batch_dims through the permutation
        original_batch_dims = x.batch_dims
        result = super().__call__(x, source=source, destination=destination)
        result._impl.batch_dims = original_batch_dims
        return result

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        source = output.op_kwargs.get("source")
        destination = output.op_kwargs.get("destination")
        # Inverse: move from destination back to source
        return moveaxis_physical(cotangent, source=destination, destination=source)

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


class BroadcastBatchDimsOp(Operation):
    """Broadcast tensor to have specified batch dimensions."""

    @property
    def name(self) -> str:
        return "broadcast_batch_dims"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for broadcast_batch_dims."""
        from ...core.sharding import spmd, spec

        x = args[0]
        target_shape = kwargs.get("shape")

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        if target_shape is None:
            raise RuntimeError(f"Could not determine target shape for {self.name}")

        shapes = []
        if output_sharding and mesh:
            for i in range(num_shards):
                local = spec.compute_local_shape(
                    target_shape, output_sharding, device_id=i
                )
                shapes.append(tuple(int(d) for d in local))
        else:
            shapes = [tuple(int(d) for d in target_shape)] * num_shards

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.devices]
            else:
                devices = [mesh.devices[0]] * num_shards
        else:
            devices = [x.device] * num_shards

        return shapes, dtypes, devices

    def kernel(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        return ops.broadcast_to(x, shape)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        """Broadcast adds batch dims as prefix, input dims shift to suffix."""
        from ...core.sharding.propagation import OpShardingRuleTemplate

        if not input_shapes or not output_shapes:
            return None

        in_shape = input_shapes[0]
        out_shape = output_shapes[0]

        in_rank = len(in_shape)
        out_rank = len(out_shape)
        n_batch_dims = out_rank - in_rank

        # Output: batch dims get independent factors, then input dims
        out_factors = [f"d{i}" for i in range(out_rank)]
        out_mapping = {i: [out_factors[i]] for i in range(out_rank)}

        # Input: dimensions map to output dimensions after batch prefix
        in_mapping = {}
        for i in range(in_rank):
            out_idx = i + n_batch_dims
            in_mapping[i] = [out_factors[out_idx]]

        return OpShardingRuleTemplate([in_mapping], [out_mapping]).instantiate(
            input_shapes, output_shapes
        )

    def __call__(self, x: Tensor, *, batch_shape: tuple[int, ...]) -> Tensor:
        from ...core import Tensor

        logical_shape = tuple(x.shape)
        physical_shape = tuple(batch_shape) + logical_shape

        result = super().__call__(x, shape=physical_shape)

        # Update batch_dims to match the new batch shape
        output = Tensor._create_unsafe(
            bufferss=result._buffers,
            values=result._graph_values,
            is_traced=result.is_traced,
            batch_dims=len(batch_shape),
        )
        output.sharding = result.sharding
        return output


# Singleton instances
_incr_batch_dims_op = IncrBatchDimsOp()
_decr_batch_dims_op = DecrBatchDimsOp()
_moveaxis_physical_op = MoveAxisPhysicalOp()
_broadcast_batch_dims_op = BroadcastBatchDimsOp()


def incr_batch_dims(x: Tensor) -> Tensor:
    """Increment batch_dims counter (first physical dim becomes batch dim)."""
    return _incr_batch_dims_op(x)


def decr_batch_dims(x: Tensor) -> Tensor:
    """Decrement batch_dims counter (first batch dim becomes logical dim)."""
    return _decr_batch_dims_op(x)


def moveaxis_physical(x: Tensor, source: int, destination: int) -> Tensor:
    """Move a physical axis to another physical position."""
    return _moveaxis_physical_op(x, source=source, destination=destination)


def move_axis_to_batch_dims(x: Tensor, axis: int) -> Tensor:
    """Move a logical axis into the batch dimensions (3 ops: calc + moveaxis_physical + incr)."""
    physical_axis = x.batch_dims + (axis if axis >= 0 else len(x.shape) + axis)
    if physical_axis != 0:
        x = moveaxis_physical(x, source=physical_axis, destination=0)
    return incr_batch_dims(x)


def move_axis_from_batch_dims(
    x: Tensor, batch_axis: int = 0, logical_destination: int = 0
) -> Tensor:
    """Move a batch dimension to logical axis (3 ops: calc + moveaxis_physical + decr)."""
    if x.batch_dims <= 0:
        raise ValueError("No batch dims to move from")

    batch_axis = batch_axis if batch_axis >= 0 else x.batch_dims + batch_axis
    new_batch_dims = x.batch_dims - 1
    logical_destination = (
        logical_destination
        if logical_destination >= 0
        else (len(x.shape) + 1) + logical_destination
    )
    physical_dest = new_batch_dims + logical_destination

    if batch_axis != physical_dest:
        x = moveaxis_physical(x, source=batch_axis, destination=physical_dest)
    return decr_batch_dims(x)


def broadcast_batch_dims(x: Tensor, batch_shape: tuple[int, ...]) -> Tensor:
    """Broadcast tensor to have specified batch shape."""
    return _broadcast_batch_dims_op(x, batch_shape=batch_shape)


__all__ = [
    "IncrBatchDimsOp",
    "DecrBatchDimsOp",
    "MoveAxisPhysicalOp",
    "BroadcastBatchDimsOp",
    "incr_batch_dims",
    "decr_batch_dims",
    "moveaxis_physical",
    "move_axis_to_batch_dims",
    "move_axis_from_batch_dims",
    "broadcast_batch_dims",
]
