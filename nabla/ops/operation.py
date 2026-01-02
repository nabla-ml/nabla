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
                shard_results.append(self.maxpr(*shard_args, **shard_kwargs))
            
            # AllReduce partial results if contracting dimension was sharded
            if reduce_axes and mesh:
                from .communication import all_reduce_op, simulate_grouped_all_reduce
                shard_results = simulate_grouped_all_reduce(
                    shard_results, mesh, reduce_axes, all_reduce_op
                )
        
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
        
        # 6. Common post-processing
        self._setup_output_refs(output, args, kwargs, any_traced)
        
        if any_has_tangent:
            self._apply_jvp(args, output)
        
        return output
    
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
        stored_kwargs = pytree.tree_map(to_impl, kwargs) if traced and kwargs else None
        
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
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Default sharding rule: elementwise for same-rank ops.
        
        Operations can override this to provide custom factor mappings.
        By default, assumes all inputs and output share the same factors
        (elementwise behavior).
        """
        from ..sharding.propagation import elementwise_template, unary_template
        output_rank = len(output_shapes[0]) if output_shapes else len(input_shapes[0])
        n_inputs = len(input_shapes)
        if n_inputs == 1:
            return unary_template(output_rank).instantiate(input_shapes, output_shapes)
        else:
            return elementwise_template(output_rank).instantiate(input_shapes, output_shapes)
    
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
        from ..core.tensor import Tensor
        
        x_batch = x._impl.batch_dims
        y_batch = y._impl.batch_dims
        out_batch_dims = max(x_batch, y_batch)
        
        # Explicit broadcasting when shapes differ
        if len(x.shape) != len(y.shape) or x.shape != y.shape:
            x, y = self._prepare_for_broadcast(x, y, out_batch_dims)
        
        return super().__call__(x, y)
    
    def _prepare_for_broadcast(self, x: Tensor, y: Tensor, out_batch_dims: int) -> tuple[Tensor, Tensor]:
        from . import view as view_ops
        
        # Use GLOBAL shapes for broadcast logic (not shard-local)
        # This ensures sharded tensors broadcast correctly based on logical dims
        x_global = tuple(d for d in x._impl.global_shape or x._impl.physical_shape)
        y_global = tuple(d for d in y._impl.global_shape or y._impl.physical_shape)
        x_batch = x._impl.batch_dims
        y_batch = y._impl.batch_dims
        
        # Split into batch and logical (using global shapes)
        x_batch_shape = x_global[:x_batch]
        x_logical_shape = x_global[x_batch:]
        y_batch_shape = y_global[:y_batch]
        y_logical_shape = y_global[y_batch:]
        
        # Compute broadcasted shapes (using global shapes)
        out_batch_shape = self._broadcast_shapes(x_batch_shape, y_batch_shape)
        out_logical_shape = self._broadcast_shapes(x_logical_shape, y_logical_shape)
        out_physical_shape = out_batch_shape + out_logical_shape
        
        # Prepare x - unsqueeze then broadcast
        x = self._unsqueeze_to_rank(x, len(out_physical_shape), x_batch, out_batch_dims)
        current = tuple(d for d in (x._impl.global_shape or x._impl.physical_shape))
        if current != out_physical_shape:
            x = view_ops.broadcast_to(x, out_logical_shape)
        
        # Prepare y - unsqueeze then broadcast
        y = self._unsqueeze_to_rank(y, len(out_physical_shape), y_batch, out_batch_dims)
        current = tuple(d for d in (y._impl.global_shape or y._impl.physical_shape))
        if current != out_physical_shape:
            y = view_ops.broadcast_to(y, out_logical_shape)
        
        return x, y
    
    def _unsqueeze_to_rank(self, t: Tensor, target_rank: int, current_batch_dims: int, target_batch_dims: int) -> Tensor:
        from . import view as view_ops
        
        current_rank = len(t._impl.physical_shape)
        batch_dims_to_add = target_batch_dims - current_batch_dims
        current_logical_rank = current_rank - current_batch_dims
        target_logical_rank = target_rank - target_batch_dims
        logical_dims_to_add = target_logical_rank - current_logical_rank
        
        # Add batch dims at front (logical axis 0 repeatedly)
        for _ in range(batch_dims_to_add):
            t = view_ops.unsqueeze(t, axis=0)
            
        # Update batch_dims if we added any, so that subsequent logical ops 
        # (like unsqueeze for logical dims) respect the new batch structure.
        if batch_dims_to_add > 0:
            t._impl.batch_dims = target_batch_dims
            
        # Add logical dims (at position = target_batch_dims, which is logical 0 after batch adds)
        for _ in range(logical_dims_to_add):
            t = view_ops.unsqueeze(t, axis=0)
        
        return t
    
    @staticmethod
    def _broadcast_shapes(shape1: tuple, shape2: tuple) -> tuple:
        max_len = max(len(shape1), len(shape2))
        s1 = (1,) * (max_len - len(shape1)) + tuple(shape1)
        s2 = (1,) * (max_len - len(shape2)) + tuple(shape2)
        
        result = []
        for d1, d2 in zip(s1, s2):
            if d1 == d2:
                result.append(d1)
            elif d1 == 1:
                result.append(d2)
            elif d2 == 1:
                result.append(d1)
            else:
                raise ValueError(f"Cannot broadcast shapes {shape1} and {shape2}")
        return tuple(result)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
    ) -> Any:
        # Default to elementwise behavior for binary ops
        from ..sharding.propagation import elementwise_template
        # Use output rank to determine template
        if not output_shapes:
            return None
        rank = len(output_shapes[0])
        return elementwise_template(rank).instantiate(input_shapes, output_shapes)


class LogicalAxisOperation(Operation):
    """Base for ops that take LOGICAL axis/axes kwargs.
    
    Translates ALL integer kwargs by batch_dims offset.
    Works for: unsqueeze, squeeze, transpose, reduce_sum, etc.
    """
    
    # True for unsqueeze (uses ndim+1 for negative axis normalization)
    axis_offset_for_insert: bool = False
    
    def __call__(self, x: Tensor, **kwargs: Any) -> Tensor:
        batch_dims = x._impl.batch_dims
        logical_ndim = len(x.shape)
        
        translated = {}
        for key, value in kwargs.items():
            if isinstance(value, int) and not isinstance(value, bool):
                if value < 0:
                    offset = 1 if self.axis_offset_for_insert else 0
                    value = logical_ndim + offset + value
                translated[key] = batch_dims + value
            else:
                translated[key] = value
        
        return super().__call__(x, **translated)


# Alias for backward compatibility and semantic clarity
ReduceOperation = LogicalAxisOperation
UnaryOperation = Operation


class LogicalShapeOperation(Operation):
    """Base for ops that take LOGICAL shape kwargs.
    
    Prepends batch_shape to the shape kwarg.
    Works for: reshape, broadcast_to, etc.
    """
    
    def __call__(self, x: Tensor, *, shape: tuple[int, ...]) -> Tensor:
        batch_shape = x._impl.batch_shape
        physical_shape = tuple(batch_shape) + tuple(shape) if batch_shape else tuple(shape)
        return super().__call__(x, shape=physical_shape)


__all__ = [
    "Operation",
    "BinaryOperation",
    "LogicalAxisOperation",
    "LogicalShapeOperation",
    "ReduceOperation",
    "UnaryOperation",
]
