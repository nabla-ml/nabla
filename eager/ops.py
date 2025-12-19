# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Updated Operations Base Classes
# ===----------------------------------------------------------------------=== #

"""Base classes for all operations with automatic batch_dims propagation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from max import graph
    from .tensor import Tensor


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
        from .tensor import Tensor
        from .tensor_impl import TensorImpl
        from .compute_graph import GRAPH
        from . import pytree
        from max import graph as g
        
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
            from .tracing import OutputRefs
            
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
    
    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        raise NotImplementedError(f"'{self.name}' does not implement vjp_rule")
    
    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        raise NotImplementedError(f"'{self.name}' does not implement jvp_rule")
    
    def sharding_rule(self, inputs: Any, output: Any) -> Any:
        raise NotImplementedError(f"'{self.name}' does not implement sharding_rule")
    
    def get_sharding_rule_template(self) -> Any:
        return None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class BinaryOperation(Operation):
    """Base for binary element-wise ops with batch_dims-aware broadcasting."""
    
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        from .tensor import Tensor
        
        x_batch = x._impl.batch_dims
        y_batch = y._impl.batch_dims
        out_batch_dims = max(x_batch, y_batch)
        
        # Explicit broadcasting for traced tensors (needed for correct gradients)
        if x._impl.traced or y._impl.traced:
            x, y = self._prepare_for_broadcast(x, y, out_batch_dims)
        
        result = super().__call__(x, y)
        return result
    
    def _prepare_for_broadcast(self, x: Tensor, y: Tensor, out_batch_dims: int) -> tuple[Tensor, Tensor]:
        from . import logical_view_ops as view_ops
        
        # Get PHYSICAL shapes
        x_physical = tuple(x._impl.physical_shape)
        y_physical = tuple(y._impl.physical_shape)
        x_batch = x._impl.batch_dims
        y_batch = y._impl.batch_dims
        
        # Split into batch and logical
        x_batch_shape = x_physical[:x_batch]
        x_logical_shape = x_physical[x_batch:]
        y_batch_shape = y_physical[:y_batch]
        y_logical_shape = y_physical[y_batch:]
        
        # Compute broadcasted shapes
        out_batch_shape = self._broadcast_shapes(x_batch_shape, y_batch_shape)
        out_logical_shape = self._broadcast_shapes(x_logical_shape, y_logical_shape)
        out_physical_shape = out_batch_shape + out_logical_shape
        
        # Prepare x
        if x._impl.traced:
            x = self._unsqueeze_to_rank(x, len(out_physical_shape), x_batch, out_batch_dims)
            current = tuple(x._impl.physical_shape)
            if current != out_physical_shape:
                x = view_ops.broadcast_to(x, out_logical_shape)
        
        # Prepare y
        if y._impl.traced:
            y = self._unsqueeze_to_rank(y, len(out_physical_shape), y_batch, out_batch_dims)
            current = tuple(y._impl.physical_shape)
            if current != out_physical_shape:
                y = view_ops.broadcast_to(y, out_logical_shape)
        
        return x, y
    
    def _unsqueeze_to_rank(self, t: Tensor, target_rank: int, current_batch_dims: int, target_batch_dims: int) -> Tensor:
        from . import logical_view_ops as view_ops
        
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
