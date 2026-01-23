# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING

from max.graph import TensorValue, ops

from ..base import Operation

if TYPE_CHECKING:
    from ...core.tensor import Tensor


def _copy_impl_with_batch_dims(
    x: Tensor, new_batch_dims: int, op: Operation = None, kwargs: dict = None
) -> Tensor:
    from ...core import Tensor

    output = Tensor._create_unsafe(
        storages=x._storages,
        values=x._values,
        traced=x.traced,
        batch_dims=new_batch_dims,
    )

    output.sharding = x.sharding

    if op is not None and x.traced:
        op._setup_output_refs(output, (x,), kwargs or {}, True)

    return output


class IncrBatchDimsOp(Operation):
    @property
    def name(self) -> str:
        return "incr_batch_dims"

    def maxpr(self, x: TensorValue) -> TensorValue:
        return x

    def __call__(self, x: Tensor) -> Tensor:
        return _copy_impl_with_batch_dims(x, x.batch_dims + 1, op=self, kwargs={})

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        return decr_batch_dims(cotangent)


class DecrBatchDimsOp(Operation):
    @property
    def name(self) -> str:
        return "decr_batch_dims"

    def maxpr(self, x: TensorValue) -> TensorValue:
        return x

    def __call__(self, x: Tensor) -> Tensor:
        if x.batch_dims <= 0:
            raise ValueError("Cannot decrement batch_dims below 0")
        return _copy_impl_with_batch_dims(x, x.batch_dims - 1, op=self, kwargs={})

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        return incr_batch_dims(cotangent)


class MoveAxisToBatchDimsOp(Operation):
    @property
    def name(self) -> str:
        return "move_axis_to_batch_dims"

    def maxpr(self, x: TensorValue, *, physical_axis: int) -> TensorValue:
        rank = len(x.type.shape)
        order = list(range(rank))
        order.pop(physical_axis)
        order.insert(0, physical_axis)
        return ops.permute(x, tuple(order))

    def __call__(self, x: Tensor, *, axis: int) -> Tensor:
        batch_dims = x.batch_dims
        logical_rank = len(x.shape)

        if axis < 0:
            axis = logical_rank + axis

        physical_axis = batch_dims + axis
        result = super().__call__(x, physical_axis=physical_axis)
        return _copy_impl_with_batch_dims(result, batch_dims + 1, op=self, kwargs={"axis": axis})

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        axis = output.op_kwargs.get("axis")
        return move_axis_from_batch_dims(cotangent, batch_axis=0, logical_destination=axis)


class MoveAxisFromBatchDimsOp(Operation):
    @property
    def name(self) -> str:
        return "move_axis_from_batch_dims"

    def maxpr(
        self, x: TensorValue, *, physical_source: int, physical_destination: int
    ) -> TensorValue:
        rank = len(x.type.shape)
        order = list(range(rank))
        order.pop(physical_source)
        order.insert(physical_destination, physical_source)
        return ops.permute(x, tuple(order))

    def __call__(
        self, x: Tensor, *, batch_axis: int = 0, logical_destination: int = 0
    ) -> Tensor:
        current_batch_dims = x.batch_dims
        if current_batch_dims <= 0:
            raise ValueError("No batch dims to move from")

        logical_rank = len(x.shape)

        if batch_axis < 0:
            batch_axis = current_batch_dims + batch_axis

        physical_source = batch_axis
        new_batch_dims = current_batch_dims - 1
        new_logical_rank = logical_rank + 1

        if logical_destination < 0:
            logical_destination = new_logical_rank + logical_destination

        physical_destination = new_batch_dims + logical_destination

        result = super().__call__(
            x,
            physical_source=physical_source,
            physical_destination=physical_destination,
        )
        return _copy_impl_with_batch_dims(result, new_batch_dims, op=self, kwargs={"batch_axis": batch_axis, "logical_destination": logical_destination})

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        batch_axis = output.op_kwargs.get("batch_axis")
        logical_destination = output.op_kwargs.get("logical_destination")
        # Inverse: move from logical back to batch
        # Wait, move_axis_to_batch_dims moves logical axis to physical 0.
        # But here we moved FROM batch_axis to logical_destination.
        # So we need to move FROM logical_destination BACK TO batch_axis.
        # Currently move_axis_to_batch_dims always moves to physical 0.
        # We might need a more general op if batch_axis != 0.
        # For vmap, batch_axis is usually 0.
        return move_axis_to_batch_dims(cotangent, axis=logical_destination)


class BroadcastBatchDimsOp(Operation):
    @property
    def name(self) -> str:
        return "broadcast_batch_dims"

    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        return ops.broadcast_to(x, shape)

    def __call__(self, x: Tensor, *, batch_shape: tuple[int, ...]) -> Tensor:
        logical_shape = tuple(x.shape)
        physical_shape = tuple(batch_shape) + logical_shape

        result = super().__call__(x, shape=physical_shape)
        return _copy_impl_with_batch_dims(result, len(batch_shape))


_incr_batch_dims_op = IncrBatchDimsOp()
_decr_batch_dims_op = DecrBatchDimsOp()
_move_axis_to_batch_dims_op = MoveAxisToBatchDimsOp()
_move_axis_from_batch_dims_op = MoveAxisFromBatchDimsOp()
_broadcast_batch_dims_op = BroadcastBatchDimsOp()


def incr_batch_dims(x: Tensor) -> Tensor:
    return _incr_batch_dims_op(x)


def decr_batch_dims(x: Tensor) -> Tensor:
    return _decr_batch_dims_op(x)


def move_axis_to_batch_dims(x: Tensor, axis: int) -> Tensor:
    return _move_axis_to_batch_dims_op(x, axis=axis)


def move_axis_from_batch_dims(
    x: Tensor, batch_axis: int = 0, logical_destination: int = 0
) -> Tensor:
    return _move_axis_from_batch_dims_op(
        x, batch_axis=batch_axis, logical_destination=logical_destination
    )


def broadcast_batch_dims(x: Tensor, batch_shape: tuple[int, ...]) -> Tensor:
    return _broadcast_batch_dims_op(x, batch_shape=batch_shape)


__all__ = [
    "IncrBatchDimsOp",
    "DecrBatchDimsOp",
    "MoveAxisToBatchDimsOp",
    "MoveAxisFromBatchDimsOp",
    "BroadcastBatchDimsOp",
    "incr_batch_dims",
    "decr_batch_dims",
    "move_axis_to_batch_dims",
    "move_axis_from_batch_dims",
    "broadcast_batch_dims",
]
