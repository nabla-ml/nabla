# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from max import graph

    from ..core import Tensor


from .utils import (
    _get_tensor_hash,
    _make_hashable,
    _clear_hashable_cache,
    ensure_tensor,
    collect_metadata,
    adapt_and_reshard,
    compute_structural_hash,
    eager_execute,
    verify_eager_shapes,
    package_outputs,
    apply_auto_reduction,
    apply_jvp,
)


class Operation(ABC):
    """Base class for all operations.

    Auto-propagates batch_dims.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    def kernel(self, *args: graph.TensorValue, **kwargs: Any) -> Any:
        """Returns TensorValue or pytree of TensorValues."""
        ...

    @property
    def collective_reduce_type(self) -> str:
        """Type of reduction to use for cross-shard communication (sum, max, min, prod)."""
        return "sum"

    def compute_cost(
        self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]
    ) -> float:
        """Estimate compute cost (FLOPs)."""
        return 0.0

    def memory_cost(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        dtype_bytes: int = 4,
    ) -> int:
        """Estimate memory usage (bytes) for output tensors."""
        total = 0
        for shape in output_shapes:
            elements = 1
            for dim in shape:
                elements *= dim
            total += elements * dtype_bytes
        return total

    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Default physical execution: execute kernel on each shard independently.

        This is the standard pattern used by most operations. Operations with
        specialized execution logic (communication ops, reductions, etc.) can override.

        Returns:
            tuple: (shard_results, output_sharding, mesh)
        """
        from ..core import GRAPH
        from ..core.sharding import spmd

        mesh = spmd.get_mesh_from_args(args)

        with GRAPH.graph:
            shard_results = spmd.execute_on_shards(
                self.kernel, args, kwargs, mesh, op=self
            )

        output_sharding, _, _ = spmd.infer_output_sharding(
            self, args, mesh, kwargs or {}
        )

        return (shard_results, output_sharding, mesh)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # === New Path: Physical Execution ===

        # 1. Collect Metadata (for Adaptation logic)
        max_batch_dims, any_traced, any_sharded, any_has_tangent = collect_metadata(
            args
        )

        # 2. Adaptation: Reshard Inputs
        resharded_args, adapted_kwargs, predicted_output_spec, mesh, reduce_axes = (
            adapt_and_reshard(self, args, kwargs, any_sharded, max_batch_dims)
        )

        # 3. Compute Hash for Caching
        op_hash = compute_structural_hash(self.name, resharded_args, adapted_kwargs)

        # 4. Eager Execution (if enabled)
        execution_results = eager_execute(self, resharded_args, kwargs, adapted_kwargs)

        # 5. Determine Physical Metadata (Shapes, Dtypes, Devices)
        from ..config import VERIFY_EAGER_SHAPES

        if execution_results is None or VERIFY_EAGER_SHAPES:
            # We must compute them manually if no physical execution happened,
            # or if the user explicitly requested verification against the backend.
            if type(self).compute_physical_shape is Operation.compute_physical_shape:
                raise RuntimeError(
                    f"{self.__class__.__name__} must implement compute_physical_shape"
                )

            output_physical_shapes, output_shard_dtypes, output_shard_devices = (
                self.compute_physical_shape(
                    resharded_args, adapted_kwargs, predicted_output_spec
                )
            )

            # Optional cross-verification
            if execution_results is not None:
                verify_eager_shapes(self, execution_results, output_physical_shapes)
        else:
            # OPTIMIZATION: Trust the TensorValues produced by the backend.
            # package_outputs will let the Tensors infer shape/dtype from their own values.
            output_physical_shapes, output_shard_dtypes, output_shard_devices = (
                None,
                None,
                None,
            )

        # 6. Packaging (Create Tensor(s))
        output = package_outputs(
            self,
            execution_results,
            output_physical_shapes,
            output_shard_dtypes,
            output_shard_devices,
            predicted_output_spec,
            mesh,
            any_traced,
            max_batch_dims,
        )

        # 7. Tracing & Post-Processing
        self._setup_output_refs(output, resharded_args, kwargs, op_hash=op_hash)
        output = apply_auto_reduction(self, output, mesh, reduce_axes)

        if any_has_tangent:
            apply_jvp(self, args, output)

        return output

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]] | None, list[Any] | None, list[Any] | None]:
        """Infer per-shard physical shapes for outputs.

        Subclasses must override this when used with physical execution.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement compute_physical_shape"
        )

    def adapt_kwargs(self, args: tuple, kwargs: dict, batch_dims: int) -> dict:
        return kwargs

    def _transform_shard_kwargs(
        self, kwargs: dict, output_sharding: Any, shard_idx: int, args: tuple
    ) -> dict:
        """Transform kwargs for per-shard kernel execution."""

        return kwargs

    def infer_output_rank(
        self, input_shapes: tuple[tuple[int, ...], ...], **kwargs
    ) -> int:
        """Infer output rank from input shapes."""
        if not input_shapes:
            return 0
        return len(input_shapes[0])

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Default sharding rule: elementwise for same-rank ops."""
        from ..core.sharding.propagation import OpShardingRuleTemplate

        n_inputs = len(input_shapes)
        output_rank = len(output_shapes[0]) if output_shapes else len(input_shapes[0])

        mapping = {i: [f"d{i}"] for i in range(output_rank)}

        if n_inputs == 1:

            return OpShardingRuleTemplate([mapping], [mapping]).instantiate(
                input_shapes, output_shapes
            )
        else:

            return OpShardingRuleTemplate([mapping] * n_inputs, [mapping]).instantiate(
                input_shapes, output_shapes
            )

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        raise NotImplementedError(f"'{self.name}' does not implement vjp_rule")

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        raise NotImplementedError(f"'{self.name}' does not implement jvp_rule")

    def get_sharding_rule_template(self) -> Any:
        return None

    def _setup_output_refs(
        self,
        output: Any,
        args: tuple,
        logical_kwargs: dict,
        op_hash: tuple[Any, ...] | None = None,
    ) -> None:
        """Set up OpNode for tracing support."""
        from ..core import Tensor, pytree

        # Fast path: most outputs are single tensors
        if isinstance(output, Tensor):
            output_impls = [output._impl]
            output_tree_def = pytree.PyTreeDef(pytree._K_LEAF, None, (), 1)
        elif isinstance(output, (list, tuple)) and all(
            isinstance(x, Tensor) for x in output
        ):
            output_impls = [x._impl for x in output]
            output_tree_def = pytree.tree_structure(output)
        else:
            output_impls = [
                x._impl for x in pytree.tree_leaves(output) if isinstance(x, Tensor)
            ]
            if not output_impls:
                return
            _, output_tree_def = pytree.tree_flatten(output, is_leaf=pytree.is_tensor)

        from ..core import OpNode

        # Optimized to_impl: only use tree_map if args contain nested structures
        def to_impl(x: Any) -> Any:
            return x._impl if isinstance(x, Tensor) else x

        # Fast path for simple tuple of tensors (very common case)
        if isinstance(args, tuple) and all(
            isinstance(a, Tensor) or not isinstance(a, (list, dict, tuple))
            for a in args
        ):
            stored_args = tuple(to_impl(a) for a in args)
        else:
            stored_args = pytree.tree_map(to_impl, args)

        # Fast path for kwargs (usually None or simple dict)
        if logical_kwargs is None:
            stored_logical_kwargs = None
        elif isinstance(logical_kwargs, dict) and all(
            not isinstance(v, (list, tuple)) for v in logical_kwargs.values()
        ):
            stored_logical_kwargs = {k: to_impl(v) for k, v in logical_kwargs.items()}
        else:
            stored_logical_kwargs = pytree.tree_map(to_impl, logical_kwargs)

        output_refs = OpNode(
            _refs=tuple(output_impls),
            tree_def=output_tree_def,
            op=self,
            op_args=stored_args,
            op_kwargs=stored_logical_kwargs,
            _op_hash=op_hash,
        )

        for idx, impl in enumerate(output_impls):
            impl.output_refs = output_refs
            impl.output_index = idx

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class BinaryOperation(Operation):
    """Base for binary element-wise ops with batch_dims-aware broadcasting."""

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for binary elementwise ops (same as input 0)."""
        from ..core import Tensor
        from ..core.sharding import spmd

        x = args[0]
        y = args[1]

        # We rely on previous broadcasting steps to ensure x and y have compatible shapes
        # Or at least compatible for the operation logic.
        # For elementwise binary, inputs are typically broadcasted to match.

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            # Prefer x, but if x is scalar/smaller rank, y might determine shape?
            # In BinaryOperation.__call__, inputs are broadcasted to target_physical.
            # So args passed here (resharded_args) should have correct shapes.

            # Handle replicated inputs (1 shard) used in multi-shard mesh
            idx_x = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx_x)

            if s is None:
                # Fallback to y if x doesn't help? or error?
                if isinstance(y, Tensor):
                    idx_y = i if i < y.num_shards else 0
                    s = y.physical_local_shape(idx_y)

            if s is not None:
                shapes.append(tuple(int(d) for d in s))
            else:
                # Should not happen if inputs are realized/valid
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )

        # Dtype promotion
        # Simple rule: preserve float32, etc.
        # Max usually follows numpy promotion.
        # For simple cases (same dtype), return x.dtype.
        # If different, we might need logic.
        # Let's assume same dtype for now or pick wider.
        dtype = x.dtype
        if hasattr(y, "dtype") and y.dtype != dtype:
            # Very basic promotion: float > int
            pass

        dtypes = [dtype] * num_shards

        # Device placement
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.device_refs]
            else:
                devices = [mesh.device_refs[0]] * num_shards
        else:
            devices = [x.device] * num_shards

        return shapes, dtypes, devices

    def compute_cost(
        self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]
    ) -> float:
        """Elementwise binary op: 1 FLOP per output element."""
        if not output_shapes:
            return 0.0
        num_elements = 1
        for d in output_shapes[0]:
            num_elements *= d
        return float(num_elements)

    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Physical execution for Binary Ops."""
        from ..core import GRAPH
        from ..core.sharding import spmd

        mesh = spmd.get_mesh_from_args(args)

        with GRAPH.graph:
            shard_results = spmd.execute_on_shards(
                self.kernel, args, kwargs, mesh, op=self
            )

        # Infer output sharding from inputs
        output_sharding, _, _ = spmd.infer_output_sharding(
            self, args, mesh, kwargs or {}
        )

        return (shard_results, output_sharding, mesh)

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        from . import view as view_ops
        from .base import ensure_tensor

        x = ensure_tensor(x)
        y = ensure_tensor(y)

        x_logical = tuple(int(d) for d in x.shape)
        y_logical = tuple(int(d) for d in y.shape)
        target_logical = self._broadcast_shapes(x_logical, y_logical)

        if x_logical != target_logical:
            x = view_ops.broadcast_to(x, target_logical)
        if y_logical != target_logical:
            y = view_ops.broadcast_to(y, target_logical)

        x_batch_dims = x.batch_dims
        y_batch_dims = y.batch_dims

        if x_batch_dims >= y_batch_dims:
            global_phys = x.physical_global_shape or x.local_shape
            batch_shape = tuple(int(d) for d in global_phys[:x_batch_dims])
        else:
            global_phys = y.physical_global_shape or y.local_shape
            batch_shape = tuple(int(d) for d in global_phys[:y_batch_dims])

        target_physical = batch_shape + target_logical

        x_global = x.physical_global_shape or x.local_shape
        y_global = y.physical_global_shape or y.local_shape

        if tuple(int(d) for d in x_global) != target_physical:
            x = view_ops.broadcast_to_physical(x, target_physical)

        if tuple(int(d) for d in y_global) != target_physical:
            y = view_ops.broadcast_to_physical(y, target_physical)

        return super().__call__(x, y)

    def _broadcast_shapes(
        self, s1: tuple[int, ...], s2: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Compute broadcast shape of two shapes (numpy-style right-aligned)."""
        if len(s1) > len(s2):
            s2 = (1,) * (len(s1) - len(s2)) + s2
        elif len(s2) > len(s1):
            s1 = (1,) * (len(s2) - len(s1)) + s1

        result = []
        for d1, d2 in zip(s1, s2, strict=False):
            if d1 == d2:
                result.append(d1)
            elif d1 == 1:
                result.append(d2)
            elif d2 == 1:
                result.append(d1)
            else:
                raise ValueError(f"Cannot broadcast shapes {s1} and {s2}")
        return tuple(result)


class AxisOp(Operation):
    """Base for ops that take LOGICAL axis/axes kwargs.

    Translates integer kwargs by batch_dims offset.
    """

    axis_offset_for_insert: bool = False

    axis_arg_names: set[str] = {"axis", "dim"}

    def adapt_kwargs(self, args: tuple, kwargs: dict, batch_dims: int) -> dict:
        if batch_dims == 0:
            return kwargs

        translated = {}
        for key, value in kwargs.items():
            if key in self.axis_arg_names and isinstance(value, int):
                # Negative indices are preserved (they index from end of both logical and physical)
                # Only POSITIVE indices need shifting by batch_dims.
                if value >= 0:
                    translated[key] = value + batch_dims
                else:
                    translated[key] = value
            else:
                translated[key] = value
        return translated

    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Physical execution for AxisOp.

        Executes the kernel on each shard independently.
        Subclasses like ReduceOperation may override for specialized behavior.

        NOTE: This method receives RAW kwargs and performs adaptation internally.
        """
        from ..core import GRAPH
        from ..core.sharding import spmd

        # 1. Collect Metadata
        max_batch_dims = collect_metadata(args)[0]

        # 2. Adapt kwargs
        adapted_kwargs = self.adapt_kwargs(args, kwargs, max_batch_dims)
        mesh = spmd.get_mesh_from_args(args)

        with GRAPH.graph:
            shard_results = spmd.execute_on_shards(
                self.kernel, args, adapted_kwargs, mesh, op=self
            )

        return (shard_results, None, mesh)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Reduce sharding rule: (d0, d1, ...) -> (d0, ...) with axis dim removed."""
        from ..core.sharding.propagation import OpShardingRuleTemplate

        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)

        in_mapping = {i: [f"d{i}"] for i in range(rank)}

        out_mapping = {}
        for i in range(rank):
            if i == axis:
                out_mapping[i] = []
            else:
                out_mapping[i] = [f"d{i}"]

        return OpShardingRuleTemplate([in_mapping], [out_mapping]).instantiate(
            input_shapes, output_shapes
        )


class ReduceOperation(AxisOp):
    """Base for reduction operations (sum, mean, max, min, etc.)."""

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for reduction operations."""
        from ..core.sharding import spmd

        x = args[0]
        # Kwargs are already adapted in Operation.__call__
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is not None:
                in_shape = tuple(int(d) for d in s)
                # Normalize axis relative to the physical rank
                norm_axis = axis if axis >= 0 else len(in_shape) + axis
                if keepdims:
                    out_shape = tuple(
                        1 if i == norm_axis else d for i, d in enumerate(in_shape)
                    )
                else:
                    out_shape = tuple(
                        d for i, d in enumerate(in_shape) if i != norm_axis
                    )
                shapes.append(out_shape)
            else:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.device_refs]
            else:
                devices = [mesh.device_refs[0]] * num_shards
        else:
            devices = [x.device] * num_shards

        return shapes, dtypes, devices

    def compute_cost(
        self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]
    ) -> float:
        """Reduction: 1 FLOP per input element (for summing/comparing)."""
        if not input_shapes:
            return 0.0
        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        return float(num_elements)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Reduction: reduce dims are removed or sized 1."""
        from ..core.sharding.propagation import OpShardingRuleTemplate

        if not input_shapes:
            return None

        rank = len(input_shapes[0])
        reduce_axes = kwargs.get("axis")
        keepdims = kwargs.get("keepdims", False)

        if reduce_axes is None:
            reduce_axes = tuple(range(rank))
        elif isinstance(reduce_axes, int):
            reduce_axes = (reduce_axes,)
        else:
            reduce_axes = tuple(reduce_axes)

        reduce_axes = tuple(ax + rank if ax < 0 else ax for ax in reduce_axes)

        factors = [f"d{i}" for i in range(rank)]
        in_mapping = {i: [factors[i]] for i in range(rank)}
        out_mapping = {}

        if keepdims:
            for i in range(rank):
                if i in reduce_axes:
                    out_mapping[i] = []
                else:
                    out_mapping[i] = [factors[i]]
        else:
            out_idx = 0
            for i in range(rank):
                if i not in reduce_axes:
                    out_mapping[out_idx] = [factors[i]]
                    out_idx += 1

        return OpShardingRuleTemplate([in_mapping], [out_mapping]).instantiate(
            input_shapes, output_shapes
        )

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

    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Physical execution for Reduction Ops.

        Executes the reduction kernel on each shard independently.
        Cross-shard reductions (when reducing over sharded axes) are handled
        by the auto-AllReduce mechanism in Operation.__call__.

        NOTE: This method receives RAW kwargs and performs adaptation internally.
        This ensures consistency with trace rehydration.
        """
        from ..core import GRAPH
        from ..core.sharding import spmd

        # Collect Metadata and Adapt
        max_batch_dims = collect_metadata(args)[0]
        adapted_kwargs = self.adapt_kwargs(args, kwargs, max_batch_dims)
        mesh = spmd.get_mesh_from_args(args)

        with GRAPH.graph:
            shard_results = spmd.execute_on_shards(
                self.kernel, args, adapted_kwargs, mesh, op=self
            )

        return (shard_results, None, mesh)


class UnaryOperation(Operation):
    """Base for unary element-wise operations."""

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for unary elementwise ops (same as input)."""
        from ..core.sharding import spmd

        x = args[0]
        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is not None:
                shapes.append(tuple(int(d) for d in s))
            else:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.device_refs]
            else:
                devices = [mesh.device_refs[0]] * num_shards
        else:
            devices = [x.device] * num_shards

        return shapes, dtypes, devices

    def compute_cost(
        self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]
    ) -> float:
        """Unary elementwise op: 1 FLOP per element by default."""
        if not input_shapes:
            return 0.0
        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        return float(num_elements)

    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Physical execution for Unary Ops."""
        from ..core import GRAPH
        from ..core.sharding import spmd

        mesh = spmd.get_mesh_from_args(args)

        with GRAPH.graph:
            shard_results = spmd.execute_on_shards(
                self.kernel, args, kwargs, mesh, op=self
            )

        return (shard_results, None, mesh)


class ShapeOp(Operation):
    """Base for ops that take LOGICAL shape kwargs."""

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for shape operations."""
        from ..core.sharding import spmd, spec

        x = args[0]
        # Kwargs are already adapted in Operation.__call__
        global_phys_shape = kwargs.get("shape")

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        if output_sharding and mesh:
            for i in range(num_shards):
                # Use sharding spec to determine local physical shape from global physical shape
                local = spec.compute_local_shape(
                    global_phys_shape, output_sharding, device_id=i
                )
                shapes.append(tuple(int(d) for d in local))
        else:
            # Unsharded / replicated case
            shapes = [tuple(int(d) for d in global_phys_shape)] * num_shards

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.device_refs]
            else:
                devices = [mesh.device_refs[0]] * num_shards
        else:
            devices = [x.device] * num_shards

        return shapes, dtypes, devices

    def adapt_kwargs(self, args: tuple, kwargs: dict, batch_dims: int) -> dict:
        if batch_dims == 0:
            return kwargs

        shape = kwargs.get("shape")
        if shape is not None and len(args) > 0:
            x = args[0]
            from ..core import Tensor

            if isinstance(x, Tensor):
                if x.physical_global_shape is not None:
                    global_batch_shape = tuple(
                        int(d) for d in x.physical_global_shape[:batch_dims]
                    )
                else:
                    global_phys = x.local_shape
                    # Fallback if local_shape is None? Should not happen if hydrated or determined.
                    if global_phys is None:
                        # If we can't determine physical shape, we might be in trouble or it's not realized.
                        # But for shape ops we usually need it.
                        # Let's assume input has shape info.
                        global_phys = (
                            x.shape
                        )  # Fallback to logical if physical missing? No.
                        # If physical is missing it might be (B, ...).
                        # Let's rely on standard property behavior.
                        pass

                    if global_phys:
                        global_batch_shape = tuple(
                            int(d) for d in global_phys[:batch_dims]
                        )
                    else:
                        # Best effort: assume standard layout
                        global_batch_shape = x.shape[
                            :batch_dims
                        ]  # This is wrong if x.shape is logical.
                        # But x.shape IS logical.
                        # Logic in original __call__:
                        # global_phys = x.local_shape
                        # global_batch_shape = tuple(int(d) for d in global_phys[:batch_dims])
                        # If local_shape is None, we can't do much.
                        return kwargs

                physical_shape = global_batch_shape + tuple(shape)
                new_kwargs = kwargs.copy()
                new_kwargs["shape"] = physical_shape
                return new_kwargs
        return kwargs

    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Physical execution for ShapeOp.

        Executes kernel on each shard with shape kwargs adapted for batch_dims.

        NOTE: This method receives RAW kwargs and performs adaptation internally.
        """
        from ..core import GRAPH
        from ..core.sharding import spmd

        # Collect Metadata and Adapt
        max_batch_dims = collect_metadata(args)[0]
        adapted_kwargs = self.adapt_kwargs(args, kwargs, max_batch_dims)
        mesh = spmd.get_mesh_from_args(args)

        with GRAPH.graph:
            shard_results = spmd.execute_on_shards(
                self.kernel, args, adapted_kwargs, mesh, op=self
            )

        return (shard_results, None, mesh)
