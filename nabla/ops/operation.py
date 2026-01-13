# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Updated Operations Base Classes
# ===----------------------------------------------------------------------=== #

"""Base classes for all operations with automatic batch_dims propagation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from max import graph
    from ..core.tensor import Tensor


def ensure_tensor(x: Any) -> "Tensor":
    """Convert scalar (int/float) or array-like to Tensor.
    
    This should be called at the start of any operation that accepts
    multiple inputs to ensure all inputs are proper Tensors.
    """
    from ..core.tensor import Tensor
    import numpy as np
    
    if isinstance(x, Tensor):
        return x
    
    # Convert scalar or array-like to numpy then to Tensor
    arr = np.asarray(x, dtype=np.float32)
    return Tensor.from_dlpack(arr)


class Operation(ABC):
    """Base class for all operations.
    
    Auto-propagates batch_dims = max(all input batch_dims) to all outputs.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        ...
    
    @abstractmethod
    def maxpr(self, *args: graph.TensorValue, **kwargs: Any) -> Any:
        """Returns TensorValue or pytree of TensorValues."""
        ...
    
    def cost_model(self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]) -> float:
        """Estimate compute cost (FLOPs). Default is 0.0 (negligible)."""
        return 0.0
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Unified dispatch for all operations.
        
        Handles sharded and unsharded tensors uniformly via SPMD logic.
        Unsharded execution is treated as a special case with mesh=None (implicit 1-shard).
        """
        from ..core.tensor import Tensor
        from ..core.tensor_impl import TensorImpl
        from ..core.compute_graph import GRAPH
        from ..core import pytree
        from ..sharding import spmd
        from max import graph as g
        
        # 1. Collect metadata from inputs
        any_traced = False
        any_has_tangent = False
        max_batch_dims = 0
        any_sharded = False
        
        def collect_metadata(x: Any) -> Any:
            nonlocal any_traced, any_has_tangent, max_batch_dims, any_sharded
            if isinstance(x, Tensor):
                any_traced = any_traced or x._impl.traced
                any_has_tangent = any_has_tangent or (x._impl.tangent is not None)
                max_batch_dims = max(max_batch_dims, x._impl.batch_dims)
                any_sharded = any_sharded or x._impl.is_sharded
            return x
        
        pytree.tree_map(collect_metadata, args)
        
        # 2. Determine execution mode (Implicit)
        mesh = spmd.get_mesh_from_args(args) if any_sharded else None
        
        # 3. Setup for execution (Common)
        # Ensure proper specifications (Pass-through if mesh is None)
        args = spmd.ensure_specs(args, mesh)
        
        # Infer sharding (Returns None/[]/False if mesh is None)
        output_sharding, input_shardings, reduce_axes = spmd.infer_output_sharding(
            self, args, mesh, kwargs or {}
        )
        
        # Eagerly reshard inputs (Pass-through if mesh is None)
        args = spmd.reshard_inputs(args, input_shardings, mesh)
        
        num_shards = len(mesh.devices) if mesh else 1
        
        # 4. Execute operation (Common Loop)
        with GRAPH.graph:
            shard_results = []
            for shard_idx in range(num_shards):
                # Retrieve shard arguments (Handles slicing or value extraction)
                shard_args = spmd.get_shard_args(
                    args, shard_idx, input_shardings or [], g, Tensor, pytree
                )
                
                shard_kwargs = self._transform_shard_kwargs(kwargs, output_sharding, shard_idx)
                if shard_kwargs is None:
                    shard_kwargs = {}
                shard_results.append(self.maxpr(*shard_args, **shard_kwargs))
        
        # 5. Create output tensors (Unified & Pytree-aware)
        if not shard_results:
            return None # Should not happen
            
        # Reconstruct output structure from first result
        flat_results_per_shard = [pytree.tree_leaves(res) for res in shard_results]
        treedef = pytree.tree_structure(shard_results[0])
        num_leaves = len(flat_results_per_shard[0])
        
        output_leaves = []
        for i in range(num_leaves):
            # Collect this leaf's values across all shards
            leaf_shards = [shard[i] for shard in flat_results_per_shard]
            
            # Create sharded/unsharded tensor from list of values
            tensor = spmd.create_sharded_output(
                leaf_shards, output_sharding, any_traced, max_batch_dims, mesh=mesh
            )
            output_leaves.append(tensor)
            
        output = pytree.tree_unflatten(treedef, output_leaves)
        
        # 7. Common post-processing - set up refs for the main op output
        # This must happen before all_reduce modifies the output
        self._setup_output_refs(output, args, kwargs, any_traced)
        
        
        # 8. AllReduce partial results if contracting dimension was sharded
        if reduce_axes and mesh:
            output = self._apply_auto_reduction(output, mesh, reduce_axes)
        
        if any_has_tangent:
            self._apply_jvp(args, output)
        
        return output

    def _apply_auto_reduction(self, output: Any, mesh: "DeviceMesh", reduce_axes: set[str]) -> Any:
        """Apply automatic AllReduce to partial results if needed."""
        from ..core.tensor import Tensor
        from ..core import pytree
        from .communication import all_reduce_op, simulate_grouped_all_reduce
        from ..sharding.spec import ShardingSpec, DimSpec
        from ..core.tensor_impl import TensorImpl
        from ..core.compute_graph import GRAPH

        def apply_grouped_all_reduce(t):
            if not isinstance(t, Tensor) or not t._impl._values:
                return t
            
            # Apply graph-level grouped all-reduce
            with GRAPH.graph:
                reduced_values = simulate_grouped_all_reduce(
                    t._impl._values, mesh, reduce_axes, all_reduce_op
                )
            
            # Create new Tensor with reduced values (replicated output)
            # Note: We assume full replication for now, but this might need refinement
            # based on which axes were actually reduced.
            # If we reduced over ALL axes in the mesh, it's fully replicated.
            # If we reduced over a SUBSET (e.g. TP), it might still be sharded on others (e.g. DP).
            
            # Logic: Start with the current sharding spec of 't' (the partial result)
            # The partial result 't' has "complex" sharding where reduced axes are technically 
            # sharded but contain partial sums.
            # The output should have those axes marked as replicated (empty set).
            
            current_spec = t._impl.sharding
            if current_spec:
                new_dim_specs = []
                for ds in current_spec.dim_specs:
                    # Remove any axes that were reduced
                    new_axes = sorted(list(set(ds.axes) - reduce_axes))
                    new_dim_specs.append(DimSpec(new_axes))
                new_spec = ShardingSpec(mesh, new_dim_specs)
            else:
                # Fallback to full replication if no spec (shouldn't happen)
                rank = len(t.shape)
                new_spec = ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)])

            # Create new Tensor with reduced values
            # Use shared helper to ensure correct global shape metadata and initialization
            from ..sharding import spmd
            reduced_tensor = spmd.create_sharded_output(
                reduced_values, new_spec, t.traced, t.batch_dims, mesh
            )
            
            # Setup tracing refs so ALL_REDUCE appears in trace
            trace_kwargs = {'mesh': mesh, 'reduce_axes': list(reduce_axes)}
            all_reduce_op._setup_output_refs(reduced_tensor, (t,), trace_kwargs, t.traced)
            
            return reduced_tensor
            

        
        return pytree.tree_map(apply_grouped_all_reduce, output)
    
    def _setup_output_refs(self, output: Any, args: tuple, kwargs: dict, traced: bool) -> None:
        """Set up OutputRefs for tracing support."""
        from ..core.tensor import Tensor
        from ..core import pytree
        
        output_impls = [x._impl for x in pytree.tree_leaves(output) if isinstance(x, Tensor)]
        
        if not output_impls:
            return
        
        import weakref
        from ..core.tracing import OutputRefs
        
        _, output_tree_def = pytree.tree_flatten(output, is_leaf=pytree.is_tensor)
        weak_refs = tuple(weakref.ref(impl) for impl in output_impls)
        
        def to_impl(x: Any) -> Any:
            return x._impl if isinstance(x, Tensor) else x
        
        stored_args = pytree.tree_map(to_impl, args) if traced else ()
        # Always store kwargs for debugging (xpr) - they're cheap
        stored_kwargs = pytree.tree_map(to_impl, kwargs) if kwargs else None
        
        output_refs = OutputRefs(
            _refs=weak_refs,
            tree_def=output_tree_def,
            op=self,
            op_args=stored_args,
            op_kwargs=stored_kwargs
        )
        
        for idx, impl in enumerate(output_impls):
            impl.output_refs = output_refs
            impl.output_index = idx
    
    def _apply_jvp(self, args: tuple, output: Any) -> None:
        """Apply JVP rule to compute output tangents."""
        from ..core.tensor import Tensor
        from ..core import pytree
        
        tangents = pytree.tree_map(
            lambda x: Tensor(impl=x._impl.tangent) if isinstance(x, Tensor) and x._impl.tangent else None,
            args
        )
        output_tangent = self.jvp_rule(args, tangents, output)
        if output_tangent is not None:
            pytree.tree_map(
                lambda o, t: setattr(o._impl, 'tangent', t._impl) if isinstance(o, Tensor) and isinstance(t, Tensor) else None,
                output, output_tangent
            )
    
    def _transform_shard_kwargs(
        self, 
        kwargs: dict, 
        output_sharding: Any, 
        shard_idx: int
    ) -> dict:
        """Transform kwargs for per-shard maxpr execution.
        
        Override this in operations that have shape/target kwargs that need
        to be converted from GLOBAL to LOCAL for sharded execution.
        
        Examples:
            - ReshapeOp: converts 'shape' from global to local based on output_sharding
            - BroadcastToOp: converts 'shape' from global to local
        
        Args:
            kwargs: Original kwargs passed to __call__
            output_sharding: Inferred output ShardingSpec (or None if replicated)
            shard_idx: Current shard index being processed
            
        Returns:
            kwargs suitable for this shard's maxpr call
        """
        # Default: pass kwargs unchanged (works for elementwise, reduce, etc.)
        return kwargs

    def infer_output_rank(self, input_shapes: tuple[tuple[int, ...], ...], **kwargs) -> int:
        """Infer output rank from input shapes.
        
        Default implementation assumes element-wise operation (output rank equals input rank).
        """
        if not input_shapes:
            return 0
        return len(input_shapes[0])

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Default sharding rule: elementwise for same-rank ops.
        
        All inputs and output share the same factors (d0, d1, ...).
        """
        from ..sharding.propagation import OpShardingRuleTemplate
        
        n_inputs = len(input_shapes)
        output_rank = len(output_shapes[0]) if output_shapes else len(input_shapes[0])
        
        # All dims share factors d0, d1, ..., d{rank-1}
        mapping = {i: [f"d{i}"] for i in range(output_rank)}
        
        if n_inputs == 1:
            # Unary: (d0, d1, ...) -> (d0, d1, ...)
            return OpShardingRuleTemplate([mapping], [mapping]).instantiate(
                input_shapes, output_shapes
            )
        else:
            # Elementwise: (d0, d1, ...), (d0, d1, ...) -> (d0, d1, ...)
            return OpShardingRuleTemplate(
                [mapping] * n_inputs, [mapping]
            ).instantiate(input_shapes, output_shapes)
    
    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        raise NotImplementedError(f"'{self.name}' does not implement vjp_rule")
    
    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        raise NotImplementedError(f"'{self.name}' does not implement jvp_rule")
    
    def get_sharding_rule_template(self) -> Any:
        return None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class BinaryOperation(Operation):
    """Base for binary element-wise ops with batch_dims-aware broadcasting."""
    
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        from . import view as view_ops
        from . import _physical as physical_ops
        
        # Ensure both inputs are Tensors (converts scalars/arrays)
        x = ensure_tensor(x)
        y = ensure_tensor(y)
        
        # Step 1: Broadcast LOGICAL shapes
        x_logical = tuple(int(d) for d in x.shape)
        y_logical = tuple(int(d) for d in y.shape)
        target_logical = self._broadcast_shapes(x_logical, y_logical)
        
        # Optimize: Skip if shapes already match target
        if x_logical != target_logical:
            x = view_ops.broadcast_to(x, target_logical)
        if y_logical != target_logical:
            y = view_ops.broadcast_to(y, target_logical)
        
        # Step 2: Broadcast PHYSICAL shapes (batch dims) - use GLOBAL shapes
        x_batch_dims = x._impl.batch_dims
        y_batch_dims = y._impl.batch_dims

        # Get GLOBAL batch shape from cached_shape (which stores global physical shape)
        if x_batch_dims >= y_batch_dims:
            global_phys = x.global_shape or x.local_shape
            batch_shape = tuple(int(d) for d in global_phys[:x_batch_dims])
        else:
            global_phys = y.global_shape or y.local_shape
            batch_shape = tuple(int(d) for d in global_phys[:y_batch_dims])
        
        target_physical = batch_shape + target_logical
        
        # Optimize: Skip physical broadcast if specs match target
        # Note: We must check if cached_shape matches target to be safe
        x_global = x.global_shape or x.local_shape
        y_global = y.global_shape or y.local_shape
        
        # We need to broadcast strict physical shapes including batch dims
        if tuple(int(d) for d in x_global) != target_physical:
             x = physical_ops.broadcast_to_physical(x, target_physical)
             
        if tuple(int(d) for d in y_global) != target_physical:
             y = physical_ops.broadcast_to_physical(y, target_physical)
        
        return super().__call__(x, y)

    def _broadcast_shapes(self, s1: tuple[int, ...], s2: tuple[int, ...]) -> tuple[int, ...]:
        """Compute broadcast shape of two shapes (numpy-style right-aligned)."""
        if len(s1) > len(s2):
            s2 = (1,) * (len(s1) - len(s2)) + s2
        elif len(s2) > len(s1):
            s1 = (1,) * (len(s2) - len(s1)) + s1
            
        result = []
        for d1, d2 in zip(s1, s2):
            if d1 == d2:
                result.append(d1)
            elif d1 == 1:
                result.append(d2)
            elif d2 == 1:
                result.append(d1)
            else:
                raise ValueError(f"Cannot broadcast shapes {s1} and {s2}")
        return tuple(result)




class LogicalAxisOperation(Operation):
    """Base for ops that take LOGICAL axis/axes kwargs.
    
    Translates ALL integer kwargs by batch_dims offset.
    Works for: unsqueeze, squeeze, transpose, reduce_sum, etc.
    """
    
    # True for unsqueeze (uses ndim+1 for negative axis normalization)
    axis_offset_for_insert: bool = False
    
    # Arguments that should be treated as axes and translated
    axis_arg_names: set[str] = {'axis', 'dim'}
    
    def __call__(self, x: Tensor, **kwargs: Any) -> Tensor:
        batch_dims = x._impl.batch_dims
        logical_ndim = len(x.shape)
        
        translated = {}
        for key, value in kwargs.items():
            # Only translate if key is identified as an axis argument
            if key in self.axis_arg_names and isinstance(value, int):
                if value < 0:
                    offset = 1 if self.axis_offset_for_insert else 0
                    value = logical_ndim + offset + value
                translated[key] = batch_dims + value
            else:
                translated[key] = value
        
        return super().__call__(x, **translated)
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Reduce sharding rule: (d0, d1, ...) -> (d0, ...) with axis dim removed.
        
        The reduced dimension gets no factor in the output.
        """
        from ..sharding.propagation import OpShardingRuleTemplate
        
        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        
        # Input: all dims have factors
        in_mapping = {i: [f"d{i}"] for i in range(rank)}
        
        # Output: reduced dim has no factor (empty list)
        out_mapping = {}
        for i in range(rank):
            if i == axis:
                out_mapping[i] = []  # Reduced dim - no factor
            else:
                out_mapping[i] = [f"d{i}"]
        
        return OpShardingRuleTemplate([in_mapping], [out_mapping]).instantiate(
            input_shapes, output_shapes
        )

# Proper ABC for reduction operations with reduce sharding rule
class ReduceOperation(LogicalAxisOperation):
    """Base for reduction operations (sum, mean, max, min, etc.).
    
    Inherits axis translation from LogicalAxisOperation.
    Provides reduce sharding rule: reduced dim gets no factor.
    """
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Reduction: reduce dims are removed or sized 1.
        
        If keepdims=True, reduced dim becomes size 1 but retains a factor (mapped to nothing/replicated).
        If keepdims=False, reduced dim is removed from output string.
        """
        from ..sharding.propagation import OpShardingRuleTemplate
        
        if not input_shapes:
            return None
            
        rank = len(input_shapes[0])
        reduce_axes = kwargs.get("axis", None)
        keepdims = kwargs.get("keepdims", False)
        
        if reduce_axes is None:
            reduce_axes = tuple(range(rank))
        elif isinstance(reduce_axes, int):
            reduce_axes = (reduce_axes,)
        else:
            reduce_axes = tuple(reduce_axes)
            
        # Normalize reduce axes
        reduce_axes = tuple(
            ax + rank if ax < 0 else ax 
            for ax in reduce_axes
        )
        
        factors = [f"d{i}" for i in range(rank)]
        in_mapping = {i: [factors[i]] for i in range(rank)}
        out_mapping = {}
        
        if keepdims:
            # Output has same rank, but reduced dims are size 1.
            # We map reduced dims to empty factors -> enforced replication or contraction
            for i in range(rank):
                if i in reduce_axes:
                    out_mapping[i] = [] # Reduced factor
                else:
                    out_mapping[i] = [factors[i]]
        else:
            # Output has lower rank. Reduced dims are skipped.
            out_idx = 0
            for i in range(rank):
                if i not in reduce_axes:
                    out_mapping[out_idx] = [factors[i]]
                    out_idx += 1
                    
        return OpShardingRuleTemplate([in_mapping], [out_mapping]).instantiate(
            input_shapes, output_shapes
        )
    
    def infer_output_shape(self, input_shapes: list[tuple[int, ...]], **kwargs) -> tuple[int, ...]:
        """Compute output shape for reduction."""
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]
        
        # Normalize negative axis
        if axis < 0:
            axis = len(in_shape) + axis
        
        if keepdims:
            return tuple(1 if i == axis else d for i, d in enumerate(in_shape))
        else:
            return tuple(d for i, d in enumerate(in_shape) if i != axis)

UnaryOperation = Operation


class LogicalShapeOperation(Operation):
    """Base for ops that take LOGICAL shape kwargs.
    
    Prepends batch_shape to the shape kwarg.
    Works for: reshape, broadcast_to, etc.
    """
    
    def __call__(self, x: Tensor, *, shape: tuple[int, ...]) -> Tensor:
        batch_dims = x._impl.batch_dims
        
        if batch_dims > 0:
            # For sharded tensors, use GLOBAL batch shape from cached_shape
            # (not local batch_shape which is per-shard)
            # This ensures consistency with infer_output_sharding which uses cached_shape
            if x._impl.cached_shape is not None:
                # Use cached global physical shape (includes batch dims)
                global_batch_shape = tuple(int(d) for d in x._impl.cached_shape[:batch_dims])
            else:
                # Fallback to local physical shape for unsharded tensors
                # (For unsharded tensors, local physical == global physical)
                global_phys = x.local_shape
                global_batch_shape = tuple(int(d) for d in global_phys[:batch_dims])
            physical_shape = global_batch_shape + tuple(shape)
        else:
            physical_shape = tuple(shape)
        
        return super().__call__(x, shape=physical_shape)


__all__ = [
    "Operation",
    "BinaryOperation",
    "LogicalAxisOperation",
    "LogicalShapeOperation",
    "ReduceOperation",
    "UnaryOperation",
]
