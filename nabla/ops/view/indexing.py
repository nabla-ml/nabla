# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from ..base import OpArgs, Operation, OpKwargs, OpResult, OpTensorValues

if TYPE_CHECKING:
    from ...core import Tensor


def _adapt_axis_kwargs(kwargs: OpKwargs, batch_dims: int) -> OpKwargs:
    """Translate logical axis to physical axis for gather/scatter."""
    if batch_dims == 0:
        return kwargs
    axis = kwargs.get("axis", 0)
    adapted_axis = axis + batch_dims if axis >= 0 else axis
    return {**kwargs, "axis": adapted_axis, "batch_dims": batch_dims}


class GatherOp(Operation):
    """Gather elements from data tensor along an axis using indices."""

    @property
    def name(self) -> str:
        return "gather"

    def adapt_kwargs(self, args: OpArgs, kwargs: OpKwargs, batch_dims: int) -> dict:
        return _adapt_axis_kwargs(kwargs, batch_dims)

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for gather.

        kwargs here are ADAPTED (physical axis, batch_dims set).
        """
        from ...core.sharding import spmd

        x = args[0]
        indices = args[1]
        axis = kwargs.get("axis", 0)
        batch_dims = kwargs.get("batch_dims", 0)

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx_x = i if i < x.num_shards else 0
            idx_i = i if i < indices.num_shards else 0

            s_x = x.physical_local_shape(idx_x)
            s_i = indices.physical_local_shape(idx_i)
            if s_x is None or s_i is None:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )

            x_shape = [int(d) for d in s_x]
            i_shape = [int(d) for d in s_i]

            # Physical axis: gather replaces axis dim with index dims (after batch)
            norm_axis = axis if axis >= 0 else len(x_shape) + axis
            # Output: x[:axis] + indices[batch_dims:] + x[axis+1:]
            out_shape = (
                x_shape[:norm_axis] + i_shape[batch_dims:] + x_shape[norm_axis + 1 :]
            )

            shapes.append(tuple(out_shape))

        dtypes, devices = self._build_shard_metadata(x, mesh, num_shards)

        return shapes, dtypes, devices

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        """Call gather with LOGICAL axis (no translation here)."""
        x = args[0]
        axis = kwargs.get("axis", 0)
        logical_ndim = len(x.shape)
        if axis < 0:
            axis = logical_ndim + axis
        return super().__call__(args, {**kwargs, "axis": axis})

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        from max.dtype import DType

        x = args[0]
        indices = args[1]
        axis = kwargs.get("axis", 0)
        batch_dims = kwargs.get("batch_dims", 0)

        if batch_dims > 0 and axis != batch_dims:
            # Need to permute x so the gather axis is at position batch_dims,
            # then use gather_nd, then permute the result back.
            x_rank = len(x.shape)
            # Build permutation: move axis to batch_dims position
            perm_fwd = list(range(x_rank))
            perm_fwd.remove(axis)
            perm_fwd.insert(batch_dims, axis)
            x_perm = ops.permute(x, perm_fwd)

            # gather_nd along position batch_dims
            indices_shape = list(indices.shape)
            new_shape = indices_shape + [1]
            idx = ops.reshape(indices, new_shape)
            if idx.dtype != DType.int64:
                idx = ops.cast(idx, DType.int64)
            result = ops.gather_nd(x_perm, idx, batch_dims=batch_dims)

            # Permute result back: move batch_dims position back to where axis was
            # result shape: batch_dims_shape + idx_shape[batch_dims:] + remaining
            # We need to move the gathered dim(s) back to the right position
            r_rank = len(result.shape)
            n_idx_dims = len(indices_shape) - batch_dims  # number of index dims
            # In the result: [0..batch_dims-1, batch_dims..batch_dims+n_idx-1, rest...]
            # In the original output: [0..batch_dims-1, <dims before axis>, idx_dims, <dims after axis>]
            # The 'rest' in result corresponds to dims that were NOT at the gather axis
            # We moved axis to batch_dims, so rest = dims before axis (batch_dims..axis-1) + dims after axis
            # We need to put them back
            perm_inv = list(range(r_rank))
            # The result has: batch | idx_dims | (dims_before_axis) | (dims_after_axis)
            # We want: batch | (dims_before_axis) | idx_dims | (dims_after_axis)
            n_before = axis - batch_dims  # dims between batch and original axis
            if n_before > 0:
                # Move idx_dims (at positions batch_dims..batch_dims+n_idx-1)
                # after the n_before dims
                idx_dim_positions = list(range(batch_dims, batch_dims + n_idx_dims))
                before_positions = list(
                    range(batch_dims + n_idx_dims, batch_dims + n_idx_dims + n_before)
                )
                after_positions = list(
                    range(batch_dims + n_idx_dims + n_before, r_rank)
                )
                perm_inv = (
                    list(range(batch_dims))
                    + before_positions
                    + idx_dim_positions
                    + after_positions
                )
                result = ops.permute(result, perm_inv)

            return [result]

        elif batch_dims > 0:
            # axis == batch_dims: gather_nd directly
            indices_shape = list(indices.shape)
            new_shape = indices_shape + [1]
            indices = ops.reshape(indices, new_shape)

            if indices.dtype != DType.int64:
                indices = ops.cast(indices, DType.int64)

            return [ops.gather_nd(x, indices, batch_dims=batch_dims)]
        else:
            return [ops.gather(x, indices, axis)]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP rule uses LOGICAL axis from kwargs."""
        x, indices = primals[0], primals[1]
        axis = kwargs.get("axis", 0)
        from ..creation import zeros_like

        gx = scatter(zeros_like(x), indices, cotangents[0], axis=axis)
        return [gx, None]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP rule uses LOGICAL axis from kwargs."""
        _x, indices = primals[0], primals[1]
        tx = tangents[0]
        axis = kwargs.get("axis", 0)
        return [gather(tx, indices, axis=axis)]

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

    def adapt_kwargs(self, args: OpArgs, kwargs: OpKwargs, batch_dims: int) -> dict:
        return _adapt_axis_kwargs(kwargs, batch_dims)

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for scatter (same as input x)."""
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

        dtypes, devices = self._build_shard_metadata(x, mesh, num_shards)

        return shapes, dtypes, devices

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        """Call scatter with LOGICAL axis (no translation here)."""
        x = args[0]
        axis = kwargs.get("axis", 0)
        logical_ndim = len(x.shape)
        if axis < 0:
            axis = logical_ndim + axis
        return super().__call__(args, {**kwargs, "axis": axis})

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        from max.dtype import DType

        x = args[0]
        indices = args[1]
        updates = args[2]
        axis = kwargs.get("axis", 0)
        batch_dims = kwargs.get("batch_dims", 0)

        if batch_dims > 0 and axis != batch_dims:
            # Permute x and updates so the scatter axis is at position batch_dims,
            # then scatter_nd, then permute back.
            x_rank = len(x.shape)
            u_rank = len(updates.shape)

            # Build permutation for x: move axis to batch_dims
            perm_x = list(range(x_rank))
            perm_x.remove(axis)
            perm_x.insert(batch_dims, axis)
            x_perm = ops.permute(x, perm_x)

            # Build permutation for updates: same movement
            perm_u = list(range(u_rank))
            perm_u.remove(axis)
            perm_u.insert(batch_dims, axis)
            updates_perm = ops.permute(updates, perm_u)

            # scatter_nd at position batch_dims
            _indices_shape = list(indices.shape)
            idx = indices
            if idx.dtype != DType.int64:
                idx = ops.cast(idx, DType.int64)

            # Use axis=0 scatter_nd approach on the permuted tensors
            # For batch_dims > 0, we need per-batch scatter
            result = self._scatter_at_axis(
                x_perm, idx, updates_perm, axis=batch_dims, batch_dims=batch_dims
            )

            # Permute result back
            perm_inv = [0] * x_rank
            for i, p in enumerate(perm_x):
                perm_inv[p] = i
            result = ops.permute(result, perm_inv)

            return [result]

        elif batch_dims > 0:
            # axis == batch_dims: scatter directly at that position
            return [
                self._scatter_at_axis(
                    x, indices, updates, axis=axis, batch_dims=batch_dims
                )
            ]
        else:
            return [self._scatter_at_axis(x, indices, updates, axis=axis, batch_dims=0)]

    def _scatter_at_axis(
        self,
        x: TensorValue,
        indices: TensorValue,
        updates: TensorValue,
        *,
        axis: int,
        batch_dims: int,
    ) -> TensorValue:
        """Core scatter_nd implementation for a given physical axis.

        Builds full coordinate tensors for all dims before the scatter axis
        (including batch dims), then uses scatter_nd with those coordinates.
        """
        from max.dtype import DType
        from max.graph import DeviceRef

        indices_shape = list(indices.shape)

        # Leading dims: everything before the scatter axis in x
        leading_dims = [int(d) for d in x.shape[:axis]]
        # Scatter (index) dims: indices shape after batch dims
        scatter_dims = [int(d) for d in indices_shape[batch_dims:]]
        # Full coordinate space
        full_shape = leading_dims + scatter_dims

        if not full_shape:
            # Scalar index into 1D tensor — direct scatter_nd
            idx = indices
            if idx.dtype != DType.int64:
                idx = ops.cast(idx, DType.int64)
            idx = ops.reshape(idx, list(indices_shape) + [1])
            return ops.scatter_nd(x, updates, idx)

        coord_list = []

        # Build coordinate for each leading dimension (including batch dims)
        for d in range(axis):
            dim_size = int(x.shape[d])
            coord = ops.range(0, dim_size, 1, dtype=DType.int64, device=DeviceRef.CPU())
            shape = [1] * len(full_shape)
            shape[d] = dim_size
            coord = ops.reshape(coord, shape)
            coord = ops.broadcast_to(coord, full_shape)
            coord_list.append(coord)

        # Coordinate for the scatter axis: the actual indices
        idx = indices
        if idx.dtype != DType.int64:
            idx = ops.cast(idx, DType.int64)

        # Reshape to broadcast across non-batch leading dims
        n_non_batch_leading = axis - batch_dims
        if n_non_batch_leading > 0:
            # Insert singletons: (B..., K) → (B..., 1, ..., 1, K)
            idx_shape = (
                list(indices_shape[:batch_dims])
                + [1] * n_non_batch_leading
                + list(indices_shape[batch_dims:])
            )
            idx = ops.reshape(idx, idx_shape)
        idx = ops.broadcast_to(idx, full_shape)
        coord_list.append(idx)

        stacked = ops.stack(coord_list, axis=-1)
        return ops.scatter_nd(x, updates, stacked)

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP rule uses LOGICAL axis from kwargs."""
        _x, indices, updates = primals[0], primals[1], primals[2]
        axis = kwargs.get("axis", 0)
        from ..creation import zeros_like

        gx = scatter(cotangents[0], indices, zeros_like(updates), axis=axis)
        g_updates = gather(cotangents[0], indices, axis=axis)

        return [gx, None, g_updates]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP rule uses LOGICAL axis from kwargs."""
        _x, indices, _updates = primals[0], primals[1], primals[2]
        tx, _, t_updates = tangents[0], tangents[1], tangents[2]
        axis = kwargs.get("axis", 0)
        return [scatter(tx, indices, t_updates, axis=axis)]

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
    return _gather_op([x, indices], {"axis": axis})[0]


def scatter(x: Tensor, indices: Tensor, updates: Tensor, axis: int = 0) -> Tensor:
    """Scatter updates into x at indices along axis."""
    from ..base import ensure_tensor

    x = ensure_tensor(x)
    indices = ensure_tensor(indices)
    updates = ensure_tensor(updates)
    return _scatter_op([x, indices, updates], {"axis": axis})[0]
