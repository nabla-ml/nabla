# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING

from max.graph import TensorValue, ops

from ..base import Operation

if TYPE_CHECKING:
    from ...core import Tensor


class GatherOp(Operation):
    """Gather elements from data tensor along an axis using indices."""

    @property
    def name(self) -> str:
        return "gather"

    def __call__(self, x: Tensor, indices: Tensor, *, axis: int = 0) -> Tensor:
        """Call gather with logical axis translation."""
        data_batch_dims = x.batch_dims
        indices_batch_dims = indices.batch_dims
        logical_ndim = len(x.shape)

        if axis < 0:
            axis = logical_ndim + axis
        phys_axis = data_batch_dims + axis

        use_batched = (
            data_batch_dims > 0
            and indices_batch_dims > 0
            and data_batch_dims == indices_batch_dims
        )
        batch_dims = data_batch_dims if use_batched else 0

        return super().__call__(x, indices, axis=phys_axis, batch_dims=batch_dims)

    def maxpr(
        self,
        x: TensorValue,
        indices: TensorValue,
        *,
        axis: int = 0,
        batch_dims: int = 0,
    ) -> TensorValue:
        from max.dtype import DType

        if batch_dims > 0:
            indices_shape = list(indices.shape)
            new_shape = indices_shape + [1]
            indices = ops.reshape(indices, new_shape)

            if indices.dtype != DType.int64:
                indices = ops.cast(indices, DType.int64)

            return ops.gather_nd(x, indices, batch_dims=batch_dims)
        else:
            return ops.gather(x, indices, axis)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        x, indices = primals
        axis = output.op_kwargs.get("axis", 0)
        from ..creation import zeros_like

        gx = scatter(zeros_like(x), indices, cotangent, axis=axis)
        return (gx, None)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Gather sharding rule: Data(d...), Indices(i...) -> Output(d_prefix..., i..., d_suffix...)."""
        from ...core.sharding.propagation import OpShardingRuleTemplate

        if not input_shapes or len(input_shapes) < 2:
            return None

        data_rank = len(input_shapes[0])
        indices_rank = len(input_shapes[1])
        axis = kwargs.get("axis", 0)
        if axis < 0:
            axis += data_rank

        data_factors = [f"d{i}" for i in range(data_rank)]
        data_str = " ".join(data_factors)

        indices_factors = [f"i{i}" for i in range(indices_rank)]
        indices_str = " ".join(indices_factors)

        out_factors = data_factors[:axis] + indices_factors + data_factors[axis + 1 :]
        out_str = " ".join(out_factors)

        return OpShardingRuleTemplate.parse(
            f"{data_str}, {indices_str} -> {out_str}", input_shapes
        ).instantiate(input_shapes, output_shapes)


class ScatterOp(Operation):
    """Scatter updates into data tensor at indices along an axis."""

    @property
    def name(self) -> str:
        return "scatter"

    def __call__(
        self, x: Tensor, indices: Tensor, updates: Tensor, *, axis: int = 0
    ) -> Tensor:
        """Call scatter with logical axis translation."""
        batch_dims = x.batch_dims
        logical_ndim = len(x.shape)

        if axis < 0:
            axis = logical_ndim + axis
        phys_axis = batch_dims + axis

        return super().__call__(x, indices, updates, axis=phys_axis)

    def maxpr(
        self,
        x: TensorValue,
        indices: TensorValue,
        updates: TensorValue,
        *,
        axis: int = 0,
    ) -> TensorValue:
        from max.dtype import DType

        indices_shape = list(indices.shape)

        if axis == 0:
            new_indices_shape = indices_shape + [1]
            indices = ops.reshape(indices, new_indices_shape)
            if indices.dtype != DType.int64:
                indices = ops.cast(indices, DType.int64)
            return ops.scatter_nd(x, updates, indices)
        else:
            leading_dims = [int(d) for d in x.shape[:axis]]
            trailing_update_dims = [int(d) for d in indices_shape]
            full_shape = leading_dims + trailing_update_dims

            coord_list = []

            for d, dim_size in enumerate(leading_dims):
                from max.graph import DeviceRef

                coord = ops.range(
                    0, dim_size, 1, dtype=DType.int64, device=DeviceRef.CPU()
                )
                shape = [1] * len(full_shape)
                shape[d] = dim_size
                coord = ops.reshape(coord, shape)
                coord = ops.broadcast_to(coord, full_shape)
                coord_list.append(coord)

            idx = indices
            if idx.dtype != DType.int64:
                idx = ops.cast(idx, DType.int64)

            shape = [1] * len(leading_dims) + trailing_update_dims
            idx = ops.reshape(idx, shape)
            idx = ops.broadcast_to(idx, full_shape)
            coord_list.append(idx)

            stacked = ops.stack(coord_list, axis=-1)

            return ops.scatter_nd(x, updates, stacked)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        x, indices, updates = primals
        axis = output.op_kwargs.get("axis", 0)
        from ..creation import zeros_like

        gx = scatter(cotangent, indices, zeros_like(updates), axis=axis)
        g_updates = gather(cotangent, indices, axis=axis)

        return (gx, None, g_updates)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Scatter sharding rule: Data(d...), Indices(i...), Updates(i..., d_suffix...) -> Data."""
        from ...core.sharding.propagation import OpShardingRuleTemplate

        if not input_shapes or len(input_shapes) < 3:
            return None

        data_rank = len(input_shapes[0])
        indices_rank = len(input_shapes[1])
        axis = kwargs.get("axis", 0)
        if axis < 0:
            axis += data_rank

        data_factors = [f"d{i}" for i in range(data_rank)]
        data_str = " ".join(data_factors)

        indices_factors = [f"i{i}" for i in range(indices_rank)]
        indices_str = " ".join(indices_factors)

        updates_factors = (
            data_factors[:axis] + indices_factors + data_factors[axis + 1 :]
        )
        updates_str = " ".join(updates_factors)

        out_str = data_str

        return OpShardingRuleTemplate.parse(
            f"{data_str}, {indices_str}, {updates_str} -> {out_str}", input_shapes
        ).instantiate(input_shapes, output_shapes)


_gather_op = GatherOp()
_scatter_op = ScatterOp()


def gather(x: Tensor, indices: Tensor, axis: int = 0) -> Tensor:
    """Gather elements from x along axis using indices."""
    from ..base import ensure_tensor

    indices = ensure_tensor(indices)
    return _gather_op(x, indices, axis=axis)


def scatter(x: Tensor, indices: Tensor, updates: Tensor, axis: int = 0) -> Tensor:
    """Scatter updates into x at indices along axis."""
    from ..base import ensure_tensor

    x = ensure_tensor(x)
    indices = ensure_tensor(indices)
    updates = ensure_tensor(updates)
    return _scatter_op(x, indices, updates, axis=axis)
