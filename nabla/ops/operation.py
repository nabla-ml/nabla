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
        from ..core.tensor import Tensor
        from ..core.tensor_impl import TensorImpl
        from ..core.compute_graph import GRAPH
        from ..core import pytree
        from max import graph as g
        
        # Check for sharded inputs - dispatch to SPMD path if found
        if self._has_sharded_inputs(args):
            return self._call_spmd(*args, **kwargs)
        
        any_traced = False
        any_has_tangent = False
        max_batch_dims = 0
        
        def tensor_to_value(x: Any) -> Any:
            nonlocal any_traced, any_has_tangent, max_batch_dims
            if isinstance(x, Tensor):
                if x._impl.traced:
                    any_traced = True
                if x._impl.tangent is not None:
                    any_has_tangent = True
                max_batch_dims = max(max_batch_dims, x._impl.batch_dims)
                return g.TensorValue(x)
            return x
        
        def value_to_tensor(x: Any) -> Any:
            if pytree.is_tensor_value(x):
                impl = TensorImpl(values=x, traced=any_traced, batch_dims=max_batch_dims)
                impl.cache_metadata(x)
                return Tensor(impl=impl)
            return x
        
        with GRAPH.graph:
            converted_args = pytree.tree_map(tensor_to_value, args)
            result_tree = self.maxpr(*converted_args, **kwargs)
        
        output = pytree.tree_map(value_to_tensor, result_tree)
        
        # OutputRefs for tracing
        output_impls = [x._impl for x in pytree.tree_leaves(output) if isinstance(x, Tensor)]
        
        if output_impls:
            import weakref
            from ..core.tracing import OutputRefs
            
            _, output_tree_def = pytree.tree_flatten(output, is_leaf=pytree.is_tensor)
            weak_refs = tuple(weakref.ref(impl) for impl in output_impls)
            
            def to_impl(x: Any) -> Any:
                return x._impl if isinstance(x, Tensor) else x
            
            stored_args = pytree.tree_map(to_impl, args) if any_traced else ()
            stored_kwargs = pytree.tree_map(to_impl, kwargs) if any_traced and kwargs else None
            
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
        
        # JVP mode
        if any_has_tangent:
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
        
        return output
    
    # ===== SPMD Sharding Support =====
    
    def _has_sharded_inputs(self, args: tuple) -> bool:
        """Check if any input tensor is sharded."""
        from ..sharding import spmd
        return spmd.has_sharded_inputs(args)
    
    def _get_mesh_from_args(self, args: tuple):
        """Extract the DeviceMesh from sharded inputs."""
        from ..sharding import spmd
        return spmd.get_mesh_from_args(args)
    
    def _call_spmd(self, *args: Any, **kwargs: Any) -> Any:
        """SPMD execution: run maxpr per-shard and collect results."""
        from ..core.tensor import Tensor
        from ..core.compute_graph import GRAPH
        from ..core import pytree
        from ..sharding.spec import ShardingSpec, DimSpec
        from ..sharding import spmd
        from max import graph as g
        
        mesh = spmd.get_mesh_from_args(args)
        if mesh is None:
            raise ValueError("SPMD dispatch called but no mesh found")
        
        num_shards = len(mesh.devices)
        
        # Annotate unsharded inputs as replicated (OPEN so propagation can assign sharding)
        def ensure_sharding(x: Any) -> Any:
            if isinstance(x, Tensor) and x._impl.sharding is None:
                rank = len(x.shape)
                x._impl.sharding = ShardingSpec(mesh, [DimSpec([], is_open=True) for _ in range(rank)])
            return x
        
        args = pytree.tree_map(ensure_sharding, args)
        
        # Collect metadata
        traced, batch_dims = self._collect_input_metadata(args)
        
        # Infer shardings: returns (output_sharding, per_input_shardings, needs_allreduce)
        output_sharding, input_shardings, needs_allreduce = self._infer_output_sharding(args, mesh, kwargs)
        
        # Note: align_input_shardings is not needed because propagation already
        # updates input_shardings to reflect resolved factor shardings. The
        # get_shard_args function uses input_shardings (not output_sharding) to
        # slice each input correctly.
        
        # Execute per-shard, using each input's RESOLVED sharding for slicing
        with GRAPH.graph:
            shard_results = []
            for shard_idx in range(num_shards):
                shard_args = spmd.get_shard_args(
                    args, shard_idx, input_shardings, g, Tensor, pytree
                )
                shard_results.append(self.maxpr(*shard_args, **kwargs))
            
            # AllReduce partial results if contracting dimension was sharded
            if needs_allreduce and len(shard_results) > 1:
                from .communication import all_reduce_op
                shard_results = all_reduce_op.maxpr(shard_results, reduction="sum")
        
        # Create output
        output = spmd.create_sharded_output(
            shard_results, output_sharding, traced, batch_dims
        )
        
        # Reshard if output has pre-annotated constraint
        return self._check_and_reshard_output(output, output_sharding, kwargs)
    
    def _collect_input_metadata(self, args: tuple) -> tuple:
        """Collect traced and batch_dims info from inputs."""
        from ..core.tensor import Tensor
        from ..core import pytree
        
        any_traced = False
        max_batch_dims = 0
        
        for x in pytree.tree_leaves(args):
            if isinstance(x, Tensor):
                any_traced = any_traced or x._impl.traced
                max_batch_dims = max(max_batch_dims, x._impl.batch_dims)
        
        return any_traced, max_batch_dims
    
    def _wrap_shard_result(self, template: Any, all_shard_results: list, any_traced: bool, max_batch_dims: int, sharding) -> Any:
        """Wrap shard results into a sharded Tensor."""
        from ..core.tensor import Tensor
        from ..core.tensor_impl import TensorImpl
        from ..core import pytree
        from max import graph as g
        
        if not pytree.is_tensor_value(template):
            return template
        
        # Collect corresponding values from all shards
        values = [r for r in all_shard_results if pytree.is_tensor_value(r)]
        impl = TensorImpl(
            values=values,
            traced=any_traced,
            batch_dims=max_batch_dims,
            sharding=sharding,
        )
        if values:
            impl.cache_metadata(values[0])
        return Tensor(impl=impl)
    
    def _infer_output_sharding(self, args: tuple, mesh, kwargs: dict = None):
        """Infer output sharding using factor-based propagation (Shardy algorithm)."""
        from ..sharding import spmd
        return spmd.infer_output_sharding(self, args, mesh, kwargs or {})
    
    def _needs_reshard(self, from_spec, to_spec) -> bool:
        """Check if resharding is needed between two sharding specs."""
        from ..sharding import spmd
        return spmd.needs_reshard(from_spec, to_spec)
    
    def _reshard_output(self, output, from_spec, to_spec, mesh):
        """Reshard output tensor from one sharding to another."""
        from ..sharding import spmd
        return spmd.reshard_tensor(output, from_spec, to_spec, mesh)
    
    def _check_and_reshard_output(self, output, computed_sharding, kwargs):
        """Check if output has pre-annotated sharding and reshard if needed.
        
        If the output already has a sharding annotation (from user constraint),
        and it differs from the computed sharding, insert resharding.
        """
        from ..core.tensor import Tensor
        
        if not isinstance(output, Tensor):
            return output
        
        # Check if there's a pre-annotated target sharding
        # This would be set by user via output.shard() before calling op
        target_sharding = kwargs.get('_target_output_sharding', None)
        
        if target_sharding is None:
            return output
        
        # Check if resharding is needed
        if self._needs_reshard(computed_sharding, target_sharding):
            mesh = computed_sharding.mesh if computed_sharding else target_sharding.mesh
            return self._reshard_output(output, computed_sharding, target_sharding, mesh)
        
        return output
    
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
        
        # Always do explicit broadcasting when ranks differ
        # This ensures consistent behavior and simplifies sharding
        # (binary ops become same-rank elementwise after broadcasting)
        x_rank = len(x.shape)
        y_rank = len(y.shape)
        if x_rank != y_rank or x.shape != y.shape:
            x, y = self._prepare_for_broadcast(x, y, out_batch_dims)
        
        # Check for sharded inputs - handled by SPMD dispatch
        if x._impl.is_sharded or y._impl.is_sharded:
            return super().__call__(x, y)
        
        result = super().__call__(x, y)
        return result
    
    def _prepare_for_broadcast(self, x: Tensor, y: Tensor, out_batch_dims: int) -> tuple[Tensor, Tensor]:
        from . import view as view_ops
        
        # Use GLOBAL shapes for broadcast logic (not shard-local)
        # This ensures sharded tensors broadcast correctly based on logical dims
        x_global = tuple(int(d) for d in x._impl.global_shape or x._impl.physical_shape)
        y_global = tuple(int(d) for d in y._impl.global_shape or y._impl.physical_shape)
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
        current = tuple(int(d) for d in (x._impl.global_shape or x._impl.physical_shape))
        if current != out_physical_shape:
            x = view_ops.broadcast_to(x, out_logical_shape)
        
        # Prepare y - unsqueeze then broadcast
        y = self._unsqueeze_to_rank(y, len(out_physical_shape), y_batch, out_batch_dims)
        current = tuple(int(d) for d in (y._impl.global_shape or y._impl.physical_shape))
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
