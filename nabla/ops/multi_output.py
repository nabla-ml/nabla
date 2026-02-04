# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Any

from max.graph import TensorValue, ops

from .base import AxisOp, Operation


class SplitOp(AxisOp):
    """Split a tensor into multiple equal chunks along an axis.

    Example:
        >>> x = Tensor.arange(0, 12).reshape((3, 4))
        >>> a, b = split(x, num_splits=2, axis=1)
    """

    @property
    def name(self) -> str:
        return "split"

    output_container_type = tuple

    def __call__(self, x: Tensor, **kwargs: Any) -> tuple[Tensor, ...]:
        from ..core.sharding.spec import DimSpec

        rank = len(x.shape)
        batch_dims = x.batch_dims
        axis = kwargs.get("axis", 0)

        if axis < 0:
            axis = rank + axis

        phys_axis = batch_dims + axis

        spec = x.sharding
        if spec and phys_axis < len(spec.dim_specs):
            ds = spec.dim_specs[phys_axis]
            if ds.axes:
                new_dim_specs = list(spec.dim_specs)
                new_dim_specs[phys_axis] = DimSpec([])

                x = x.with_sharding(spec.mesh, new_dim_specs)

        return super().__call__(x, **kwargs)

    def kernel(
        self, x: TensorValue, *, num_splits: int, axis: int = 0
    ) -> tuple[TensorValue, ...]:
        """Split tensor into num_splits equal parts along axis."""
        shape = list(x.type.shape)
        axis_size = int(shape[axis])

        if axis_size % num_splits != 0:
            raise ValueError(
                f"Cannot split axis of size {axis_size} into {num_splits} equal parts"
            )

        chunk_size = axis_size // num_splits
        split_sizes = [chunk_size] * num_splits

        result_list = ops.split(x, split_sizes, axis)
        return tuple(result_list)

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[list[tuple[int, ...]]], list[list[Any]], list[list[Any]]]:
        """Infer physical shapes for split (multi-output)."""
        from ..core.sharding import spmd

        x = args[0]
        num_splits = kwargs.get("num_splits")
        axis = kwargs.get("axis", 0)

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        all_outputs_shapes = []
        for _ in range(num_splits):
            out_shapes = []
            for shard_idx in range(num_shards):
                idx = shard_idx if shard_idx < x.num_shards else 0
                s = x.physical_local_shape(idx)
                if s is not None:
                    in_shape = list(int(d) for d in s)
                    norm_axis = axis if axis >= 0 else len(in_shape) + axis
                    in_shape[norm_axis] //= num_splits
                    out_shapes.append(tuple(in_shape))
                else:
                    raise RuntimeError(
                        f"Could not determine physical shape in {self.name}"
                    )
            all_outputs_shapes.append(out_shapes)

        dtypes = [[x.dtype] * num_shards] * num_splits
        if mesh:
            if mesh.is_distributed:
                devs = [d for d in mesh.device_refs]
            else:
                devs = [mesh.device_refs[0]] * num_shards
            devices = [devs] * num_splits
        else:
            devices = [[x.device] * num_shards] * num_splits

        return all_outputs_shapes, dtypes, devices

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for split: concatenate cotangents along split axis."""
        from .view.shape import concatenate

        # output is a tuple/list of tensors
        target = output[0] if isinstance(output, (list, tuple)) else output
        axis = target.op_kwargs.get("axis", 0)
        return concatenate(cotangent, axis=axis)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Split preserves sharding on non-split dimensions.

        The split axis changes size, so we must assign a new factor to it
        in the output to avoid size mismatch conflicts.
        """
        from ..core.sharding.propagation import OpShardingRuleTemplate

        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        if axis < 0:
            axis += rank

        in_factors = [chr(97 + i) for i in range(rank)]

        out_factors = list(in_factors)
        out_factors[axis] = "z"

        in_mapping = {i: [in_factors[i]] for i in range(rank)}
        out_mapping = {i: [out_factors[i]] for i in range(rank)}

        if output_shapes is not None:
            count = len(output_shapes)
        else:
            count = kwargs.get("num_splits", 2)

        return OpShardingRuleTemplate(
            input_mappings=[in_mapping], output_mappings=[out_mapping] * count
        ).instantiate(input_shapes, output_shapes)

    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0])


class ChunkOp(AxisOp):
    """Split a tensor into a specified number of chunks.

    Similar to SplitOp but returns a list instead of tuple.

    Example:
        >>> x = Tensor.arange(0, 12)
        >>> chunks = chunk(x, chunks=3)
    """

    @property
    def name(self) -> str:
        return "chunk"

    output_container_type = list

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[list[tuple[int, ...]]], list[list[Any]], list[list[Any]]]:
        """Infer physical shapes for chunk (multi-output)."""
        from ..core.sharding import spmd

        x = args[0]
        chunks = kwargs.get("chunks")
        axis = kwargs.get("axis", 0)

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        all_outputs_shapes = []
        for shard_idx in range(num_shards):
            idx = shard_idx if shard_idx < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is None:
                raise RuntimeError(f"Could not determine physical shape in {self.name}")

            in_shape = list(int(d) for d in s)
            norm_axis = axis if axis >= 0 else len(in_shape) + axis
            axis_size = in_shape[norm_axis]

            div = axis_size // chunks
            rem = axis_size % chunks

            split_sizes = []
            for i in range(chunks):
                size = div + 1 if i < rem else div
                if size > 0:
                    split_sizes.append(size)

            for out_idx, size in enumerate(split_sizes):
                out_shape = list(in_shape)
                out_shape[norm_axis] = size
                if len(all_outputs_shapes) <= out_idx:
                    all_outputs_shapes.append([])
                all_outputs_shapes[out_idx].append(tuple(out_shape))

        num_splits = len(all_outputs_shapes)
        dtypes = [[x.dtype] * num_shards] * num_splits
        if mesh:
            if mesh.is_distributed:
                devs = [d for d in mesh.device_refs]
            else:
                devs = [mesh.device_refs[0]] * num_shards
            devices = [devs] * num_splits
        else:
            devices = [[x.device] * num_shards] * num_splits

        return all_outputs_shapes, dtypes, devices

    def __call__(self, x: Tensor, **kwargs: Any) -> list[Tensor]:
        from ..core.sharding.spec import DimSpec

        rank = len(x.shape)
        batch_dims = x.batch_dims
        axis = kwargs.get("axis", 0)

        if axis < 0:
            axis = rank + axis

        phys_axis = batch_dims + axis

        spec = x.sharding
        if spec and phys_axis < len(spec.dim_specs):
            ds = spec.dim_specs[phys_axis]
            if ds.axes:
                new_dim_specs = list(spec.dim_specs)
                new_dim_specs[phys_axis] = DimSpec([])

                x = x.with_sharding(spec.mesh, new_dim_specs)

        return super().__call__(x, **kwargs)

    def kernel(
        self, x: TensorValue, *, chunks: int, axis: int = 0
    ) -> list[TensorValue]:
        """Split tensor into specified number of chunks."""
        shape = list(x.type.shape)
        axis_size = int(shape[axis])

        div = axis_size // chunks
        rem = axis_size % chunks

        split_sizes = []
        for i in range(chunks):
            size = div + 1 if i < rem else div

            if size > 0:
                split_sizes.append(size)

        return ops.split(x, split_sizes, axis)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for chunk: concatenate cotangents along chunk axis."""
        from .view.shape import concatenate

        target = output[0] if isinstance(output, (list, tuple)) else output
        axis = target.op_kwargs.get("axis", 0)
        return concatenate(cotangent, axis=axis)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Chunk preserves sharding on non-split dimensions."""
        from ..core.sharding.propagation import OpShardingRuleTemplate

        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        if axis < 0:
            axis += rank

        in_factors = [chr(97 + i) for i in range(rank)]

        out_factors = list(in_factors)
        out_factors[axis] = "z"

        in_mapping = {i: [in_factors[i]] for i in range(rank)}
        out_mapping = {i: [out_factors[i]] for i in range(rank)}

        if output_shapes is not None:
            count = len(output_shapes)
        else:
            chunks = kwargs.get("chunks", 1)
            dim_size = input_shapes[0][axis]

            div = dim_size // chunks
            rem = dim_size % chunks
            count = 0
            for i in range(chunks):
                s = div + 1 if i < rem else div
                if s > 0:
                    count += 1

        return OpShardingRuleTemplate(
            input_mappings=[in_mapping], output_mappings=[out_mapping] * count
        ).instantiate(input_shapes, output_shapes)

    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0])


class UnbindOp(AxisOp):
    """Remove a dimension and return tuple of slices.

    Example:
        >>> x = Tensor.zeros((3, 4, 5))
        >>> slices = unbind(x, axis=0)
    """

    @property
    def name(self) -> str:
        return "unbind"

    def __call__(self, x: Tensor, **kwargs: Any) -> tuple[Tensor, ...]:
        from ..core.sharding.spec import DimSpec

        rank = len(x.shape)
        batch_dims = x.batch_dims
        axis = kwargs.get("axis", 0)

        if axis < 0:
            axis = rank + axis

        phys_axis = batch_dims + axis

        spec = x.sharding
        if spec and phys_axis < len(spec.dim_specs):
            ds = spec.dim_specs[phys_axis]
            if ds.axes:
                new_dim_specs = list(spec.dim_specs)
                new_dim_specs[phys_axis] = DimSpec([])

                x = x.with_sharding(spec.mesh, new_dim_specs)

        return super().__call__(x, **kwargs)

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[list[tuple[int, ...]]], list[list[Any]], list[list[Any]]]:
        """Infer physical shapes for unbind (multi-output)."""
        from ..core.sharding import spmd

        x = args[0]
        axis = kwargs.get("axis", 0)

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        all_outputs_shapes = []
        for shard_idx in range(num_shards):
            idx = shard_idx if shard_idx < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is None:
                raise RuntimeError(f"Could not determine physical shape in {self.name}")

            in_shape = list(int(d) for d in s)
            norm_axis = axis if axis >= 0 else len(in_shape) + axis
            axis_size = in_shape[norm_axis]

            out_shape = [d for i, d in enumerate(in_shape) if i != norm_axis]

            for out_idx in range(axis_size):
                if len(all_outputs_shapes) <= out_idx:
                    all_outputs_shapes.append([])
                all_outputs_shapes[out_idx].append(tuple(out_shape))

        num_splits = len(all_outputs_shapes)
        dtypes = [[x.dtype] * num_shards] * num_splits
        if mesh:
            if mesh.is_distributed:
                devs = [d for d in mesh.device_refs]
            else:
                devs = [mesh.device_refs[0]] * num_shards
            devices = [devs] * num_splits
        else:
            devices = [[x.device] * num_shards] * num_splits

        return all_outputs_shapes, dtypes, devices

    def kernel(self, x: TensorValue, *, axis: int = 0) -> tuple[TensorValue, ...]:
        """Remove dimension and return slices."""
        shape = list(x.type.shape)
        axis_size = int(shape[axis])

        split_sizes = [1] * axis_size
        sliced = ops.split(x, split_sizes, axis)

        results = [ops.squeeze(s, axis) for s in sliced]
        return tuple(results)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for unbind: stack cotangents and unsqueeze along unbound axis."""
        from .view.shape import stack

        target = output[0] if isinstance(output, (list, tuple)) else output
        axis = target.op_kwargs.get("axis", 0)
        return stack(cotangent, axis=axis)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Unbind: the unbound axis is removed, other dims shift.

        Outputs (N slices) all share the same sharding: factor at axis is gone.
        """
        from ..core.sharding.propagation import OpShardingRuleTemplate

        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)

        factors = [f"d{i}" for i in range(rank)]
        in_mapping = {i: [factors[i]] for i in range(rank)}

        out_factors = [factors[i] for i in range(rank) if i != axis]
        out_mapping = {i: [out_factors[i]] for i in range(len(out_factors))}

        if output_shapes:
            count = len(output_shapes)
        else:
            count = input_shapes[0][axis]

        return OpShardingRuleTemplate([in_mapping], [out_mapping] * count).instantiate(
            input_shapes, output_shapes
        )

    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0]) - 1


class MinMaxOp(Operation):
    """Return both min and max of a tensor (example of dict output).

    Example:
        >>> x = Tensor.arange(0, 10)
        >>> result = minmax(x)
        >>> result['min'], result['max']
    """

    @property
    def name(self) -> str:
        return "minmax"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]] | None, Any]:
        """MinMaxOp returns a dict; skip explicit physical shape inference."""
        return None, None, None

    def __call__(self, x: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        """Compute global min and max by reducing all axes."""
        from ..ops.reduction import reduce_min, reduce_max

        # Reduce along all axes to get scalar
        result_min = x
        result_max = x
        for axis in reversed(range(len(x.shape))):
            result_min = reduce_min(result_min, axis=axis, keepdims=False)
            result_max = reduce_max(result_max, axis=axis, keepdims=False)

        return {
            "min": result_min,
            "max": result_max,
        }

    def kernel(self, x: TensorValue, **kwargs: Any) -> dict[str, TensorValue]:
        """Compute min and max simultaneously."""
        return {
            "min": ops.min(x),
            "max": ops.max(x),
        }


split = SplitOp()
chunk = ChunkOp()
unbind = UnbindOp()
minmax = MinMaxOp()


__all__ = [
    "split",
    "chunk",
    "unbind",
    "minmax",
]
