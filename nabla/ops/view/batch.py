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

from max.graph import ops

from ..base import OpArgs, Operation, OpKwargs, OpResult, OpTensorValues

if TYPE_CHECKING:
    from ...core.tensor import Tensor


def _identity_physical_shape(op, args, kwargs):
    """Return input shard shapes unchanged (shared by Incr/DecrBatchDimsOp)."""
    from ...core.sharding import spmd

    x = args[0]
    mesh = spmd.get_mesh_from_args(args)
    num_shards = len(mesh.devices) if mesh else 1

    shapes = []
    for i in range(num_shards):
        idx = i if i < x.num_shards else 0
        s = x.physical_local_shape(idx)
        if s is None:
            raise RuntimeError(f"Could not determine physical shape for {op.name}")
        shapes.append(tuple(int(d) for d in s))

    dtypes = [x.dtype] * num_shards
    if mesh:
        if mesh.is_distributed:
            devices = list(mesh.devices)
        else:
            devices = [mesh.devices[0]] * num_shards
    else:
        devices = [x.device] * num_shards

    return shapes, dtypes, devices


class IncrBatchDimsOp(Operation):
    """Increment batch_dims counter without changing data layout."""

    @property
    def name(self) -> str:
        return "incr_batch_dims"

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Physical shape is unchanged; only batch_dims metadata changes."""
        return _identity_physical_shape(self, args, kwargs)

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [args[0]]

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        x = args[0]
        results = super().__call__(args, kwargs)
        for r in results:
            r._impl.batch_dims = x.batch_dims + 1
        return results

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        return [decr_batch_dims(cotangents[0])]


class DecrBatchDimsOp(Operation):
    """Decrement batch_dims counter without changing data layout."""

    @property
    def name(self) -> str:
        return "decr_batch_dims"

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Physical shape is unchanged; only batch_dims metadata changes."""
        return _identity_physical_shape(self, args, kwargs)

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [args[0]]

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        x = args[0]
        if x.batch_dims <= 0:
            raise ValueError("Cannot decrement batch_dims below 0")
        results = super().__call__(args, kwargs)
        for r in results:
            r._impl.batch_dims = x.batch_dims - 1
        return results

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        return [incr_batch_dims(cotangents[0])]


class MoveAxisPhysicalOp(Operation):
    """Move a physical axis to another physical position, preserving batch_dims."""

    @property
    def name(self) -> str:
        return "moveaxis_physical"

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
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

            in_shape = [int(d) for d in s]
            rank = len(in_shape)
            norm_source = source if source >= 0 else rank + source
            norm_dest = destination if destination >= 0 else rank + destination

            order = list(range(rank))
            order.pop(norm_source)
            order.insert(norm_dest, norm_source)
            shapes.append(tuple(in_shape[j] for j in order))

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = list(mesh.devices)
            else:
                devices = [mesh.devices[0]] * num_shards
        else:
            devices = [x.device] * num_shards

        return shapes, dtypes, devices

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

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        # Preserve batch_dims through the permutation
        x = args[0]
        original_batch_dims = x.batch_dims
        results = super().__call__(args, kwargs)
        for r in results:
            r._impl.batch_dims = original_batch_dims
        return results

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        source = kwargs.get("source")
        destination = kwargs.get("destination")
        # Inverse: move from destination back to source
        return [
            moveaxis_physical(cotangents[0], source=destination, destination=source)
        ]

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
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for broadcast_batch_dims."""
        from ...core.sharding import spec, spmd

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
                devices = list(mesh.devices)
            else:
                devices = [mesh.devices[0]] * num_shards
        else:
            devices = [x.device] * num_shards

        return shapes, dtypes, devices

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        shape = kwargs["shape"]
        return [ops.broadcast_to(x, shape)]

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

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        x = args[0]
        batch_shape = kwargs["batch_shape"]
        logical_shape = tuple(x.shape)
        physical_shape = tuple(batch_shape) + logical_shape

        results = super().__call__(args, {"shape": physical_shape})

        # Update batch_dims to match the new batch shape.
        for r in results:
            r._impl.batch_dims = len(batch_shape)
        return results

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP: sum over added batch dimensions."""
        x = primals[0]
        added = outputs[0].batch_dims - x.batch_dims
        if added <= 0:
            return [cotangents[0]]

        from ..reduction import reduce_sum_physical
        from .batch import decr_batch_dims

        result = cotangents[0]
        for _ in range(added):
            result = reduce_sum_physical(result, axis=0, keepdims=False)
            if result.batch_dims > 0:
                result = decr_batch_dims(result)
        return [result]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP: broadcast tangent across batch dimensions."""
        phys = outputs[0].physical_global_shape
        batch_shape = tuple(int(d) for d in phys[: outputs[0].batch_dims])
        return [broadcast_batch_dims(tangents[0], batch_shape)]


# Singleton instances
_incr_batch_dims_op = IncrBatchDimsOp()
_decr_batch_dims_op = DecrBatchDimsOp()
_moveaxis_physical_op = MoveAxisPhysicalOp()
_broadcast_batch_dims_op = BroadcastBatchDimsOp()


def incr_batch_dims(x: Tensor) -> Tensor:
    """Increment batch_dims counter (first physical dim becomes batch dim)."""
    return _incr_batch_dims_op([x], {})[0]


def decr_batch_dims(x: Tensor) -> Tensor:
    """Decrement batch_dims counter (first batch dim becomes logical dim)."""
    return _decr_batch_dims_op([x], {})[0]


def moveaxis_physical(x: Tensor, source: int, destination: int) -> Tensor:
    """Move a physical axis to another physical position."""
    return _moveaxis_physical_op([x], {"source": source, "destination": destination})[0]


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
    return _broadcast_batch_dims_op([x], {"batch_shape": batch_shape})[0]


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
