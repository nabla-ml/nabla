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


def ensure_tensor(x: Any) -> Tensor:
    """Convert scalar or array-like to Tensor."""
    from ..core import Tensor

    if isinstance(x, Tensor):
        return x

    return Tensor.constant(x)


def _make_hashable(obj: Any) -> Any:
    """Convert objects to stable, hashable keys for graph caching."""
    from ..core import Tensor

    if isinstance(obj, Tensor):
        sharding = obj.sharding
        sharding_key = None
        if sharding is not None:
            mesh = getattr(sharding, "mesh", None)
            mesh_key = None
            if mesh is not None:
                mesh_key = (
                    getattr(mesh, "name", None),
                    tuple(getattr(mesh, "shape", ()) or ()),
                    tuple(getattr(mesh, "axis_names", ()) or ()),
                )

            dim_specs = []
            for ds in getattr(sharding, "dim_specs", []) or []:
                dim_specs.append(
                    (
                        tuple(getattr(ds, "axes", ()) or ()),
                        bool(getattr(ds, "partial", False)),
                    )
                )

            sharding_key = (
                mesh_key,
                tuple(dim_specs),
                tuple(sorted(getattr(sharding, "replicated_axes", set()) or set())),
                tuple(sorted(getattr(sharding, "partial_sum_axes", set()) or set())),
            )

        physical_shapes = getattr(obj._impl, "_physical_shapes", None)
        shard_dtypes = getattr(obj._impl, "_shard_dtypes", None)
        shard_devices = getattr(obj._impl, "_shard_devices", None)

        return (
            "tensor",
            str(obj.dtype),
            tuple(int(d) for d in obj.shape),  # Convert Dim to int
            sharding_key,
            tuple(physical_shapes) if physical_shapes else None,
            tuple(shard_dtypes) if shard_dtypes else None,
            tuple(str(d) for d in shard_devices) if shard_devices else None,
        )
    if isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(x) for x in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


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
        if hasattr(self, "execute"):
            from ..core import GRAPH, Tensor, pytree
            from ..core.sharding import spmd

            # 1. Collect Metadata (for Adaptation logic)
            max_batch_dims = 0
            any_traced = False
            any_sharded = False

            def collect_metadata(x):
                nonlocal max_batch_dims, any_traced, any_sharded
                if isinstance(x, Tensor):
                    if x.batch_dims > max_batch_dims:
                        max_batch_dims = x.batch_dims
                    if x.is_traced:
                        any_traced = True
                    if x.is_sharded:
                        any_sharded = True

            pytree.tree_map(collect_metadata, args)

            # 2. Adaptation: Reshard Inputs
            mesh = spmd.get_mesh_from_args(args) if any_sharded else None
            # Also check kwargs for mesh (e.g., shard/reshard ops pass mesh as kwarg)
            if mesh is None and kwargs.get("mesh") is not None:
                mesh = kwargs["mesh"]

            adapted_kwargs = self.adapt_kwargs(args, kwargs, max_batch_dims)
            args = spmd.ensure_specs(args, mesh)
            predicted_output_spec, input_shardings, reduce_axes = (
                spmd.infer_output_sharding(self, args, mesh, adapted_kwargs or {})
            )

            # Perform the data movement (Logical Adaptation)
            resharded_args = spmd.reshard_inputs(args, input_shardings, mesh)

            # Compute hash AFTER resharding - use resharded_args for cache key
            def get_tensor_hash(x):
                """Get hash from tensor - shape/dtype/sharding for realized, OpNode hash+index for unrealized."""
                if isinstance(x, Tensor):
                    # Check if tensor has physical storage (realized)
                    buffers = x._impl._buffers
                    has_output_refs = x._impl.output_refs is not None
                    
                    sharding_key = _make_hashable(x.sharding) if x.sharding else None
                    
                    if buffers:
                        # Realized tensor - hash based on shape/dtype/sharding (data-independent)
                        shape_tuple = tuple(int(d) for d in x.shape)
                        return ("realized", str(x.dtype), shape_tuple, sharding_key)
                    elif has_output_refs and x._impl.output_refs._op_hash is not None:
                        # Unrealized tensor - use its OpNode's hash AND its output index
                        return (x._impl.output_refs._op_hash, x._impl.output_index, sharding_key)
                    else:
                        # Leaf tensor without storage (shouldn't happen in normal flow)
                        shape_tuple = tuple(int(d) for d in x.shape)
                        return ("leaf", str(x.dtype), shape_tuple, sharding_key)
                return _make_hashable(x)
            
            arg_hashes = tuple(get_tensor_hash(x) for x in resharded_args)
            kwarg_hashes = tuple(sorted((k, get_tensor_hash(v)) for k, v in (adapted_kwargs or kwargs).items()))
            op_hash = (self.name, arg_hashes, kwarg_hashes)

            # 3. Physical Execution (DEFERRED)
            # Instead of executing now, just compute shape metadata
            if type(self).compute_physical_shape is Operation.compute_physical_shape:
                raise RuntimeError(
                    f"{self.__class__.__name__} must implement compute_physical_shape "
                    "for physical execution."
                )

            output_physical_shapes, output_shard_dtypes, output_shard_devices = (
                self.compute_physical_shape(
                    resharded_args, adapted_kwargs, predicted_output_spec
                )
            )

            # 4. Packaging (Create promise tensor with metadata only, no graph values yet)
            output = spmd.create_sharded_output(
                [],  # Empty - will be populated during evaluate()
                predicted_output_spec,
                any_traced,
                max_batch_dims,
                mesh=mesh,
                physical_shapes=output_physical_shapes,
                shard_dtypes=output_shard_dtypes,
                shard_devices=output_shard_devices,
            )
            output._impl.graph_values_epoch = -1  # Mark as unrealized
            GRAPH.add_unrealized(output._impl)

            # 6. Tracing - set up OpNode for THIS op FIRST (with hash)
            # Store resharded_args (not original args) so rehydration has correctly sharded inputs
            self._setup_output_refs(output, resharded_args, kwargs, op_hash=op_hash)

            # 5. SPMD: Post-Op Collectives (Automatic Reductions)
            # This may wrap output with an all_reduce op (which sets up its own refs)
            if reduce_axes and mesh:
                from .execution_utils import apply_auto_reduction

                output = apply_auto_reduction(self, output, mesh, reduce_axes)

            # 7. Post-Processing (JVP)
            # Check for tangents in ORIGINAL args (before resharding)
            any_has_tangent = False
            for x in pytree.tree_leaves(args):
                if isinstance(x, Tensor) and x.tangent is not None:
                    any_has_tangent = True

            if any_has_tangent:
                from .execution_utils import apply_jvp

                apply_jvp(self, args, output)

            return output

        # This should never happen since all operations now have execute
        # (either their own implementation or the default one from the base class)
        raise RuntimeError(
            f"Operation '{self.name}' reached unreachable code path. "
            f"This is a bug in the execution model."
        )

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

        output_impls = [
            x._impl for x in pytree.tree_leaves(output) if isinstance(x, Tensor)
        ]

        if not output_impls:
            return

        from ..core import OpNode

        _, output_tree_def = pytree.tree_flatten(output, is_leaf=pytree.is_tensor)
        output_refs = tuple(output_impls)

        def to_impl(x: Any) -> Any:
            return x._impl if isinstance(x, Tensor) else x

        stored_args = pytree.tree_map(to_impl, args)

        stored_logical_kwargs = (
            pytree.tree_map(to_impl, logical_kwargs) if logical_kwargs else None
        )

        output_refs = OpNode(
            _refs=output_refs,
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
                 raise RuntimeError(f"Could not determine physical shape for {self.name}")

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
        from ..core import GRAPH, Tensor, pytree
        from ..core.sharding import spmd

        # Compute batch_dims from args for kwargs adaptation
        max_batch_dims = 0
        for x in pytree.tree_leaves(args):
            if isinstance(x, Tensor) and x.batch_dims > max_batch_dims:
                max_batch_dims = x.batch_dims

        # Adapt kwargs (e.g., shift axis indices by batch_dims)
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
        from ..core import GRAPH, Tensor, pytree
        from ..core.sharding import spmd

        # Compute batch_dims from args for kwargs adaptation
        max_batch_dims = 0
        for x in pytree.tree_leaves(args):
            if isinstance(x, Tensor) and x.batch_dims > max_batch_dims:
                max_batch_dims = x.batch_dims

        # Adapt kwargs (e.g., shift axis indices by batch_dims)
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
                 raise RuntimeError(f"Could not determine physical shape for {self.name}")
        
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
        from ..core import GRAPH, Tensor, pytree
        from ..core.sharding import spmd

        # Compute batch_dims from args for kwargs adaptation
        max_batch_dims = 0
        for x in pytree.tree_leaves(args):
            if isinstance(x, Tensor) and x.batch_dims > max_batch_dims:
                max_batch_dims = x.batch_dims

        # Adapt kwargs (prepend batch shape to target shape)
        adapted_kwargs = self.adapt_kwargs(args, kwargs, max_batch_dims)

        mesh = spmd.get_mesh_from_args(args)

        with GRAPH.graph:
            shard_results = spmd.execute_on_shards(
                self.kernel, args, adapted_kwargs, mesh, op=self
            )

        return (shard_results, None, mesh)
