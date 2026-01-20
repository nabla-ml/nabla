# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from max import graph
    from ..core import Tensor


def ensure_tensor(x: Any) -> "Tensor":
    """Convert scalar or array-like to Tensor."""
    from ..core import Tensor
    import numpy as np
    
    if isinstance(x, Tensor):
        return x
    
    # Convert scalar or array-like to numpy then to Tensor
    arr = np.asarray(x, dtype=np.float32)
    return Tensor.from_dlpack(arr)


class Operation(ABC):
    """Base class for all operations.
    
    Auto-propagates batch_dims.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        ...
    
    def maxpr(self, *args: graph.TensorValue, **kwargs: Any) -> Any:
        """Returns TensorValue or pytree of TensorValues."""
        ...
        
    @property
    def collective_reduce_type(self) -> str:
        """Type of reduction to use for cross-shard communication (sum, max, min, prod)."""
        return "sum"
    
    def compute_cost(self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]) -> float:
        """Estimate compute cost (FLOPs)."""
        return 0.0
    
    def memory_cost(
        self, 
        input_shapes: list[tuple[int, ...]], 
        output_shapes: list[tuple[int, ...]], 
        dtype_bytes: int = 4
    ) -> int:
        """Estimate memory usage (bytes) for output tensors."""
        total = 0
        for shape in output_shapes:
            elements = 1
            for dim in shape:
                elements *= dim
            total += elements * dtype_bytes
        return total
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Unified dispatch for all operations."""
        from .dispatch import execute_operation
        return execute_operation(self, *args, **kwargs)
    
    def _transform_shard_kwargs(
        self, 
        kwargs: dict, 
        output_sharding: Any, 
        shard_idx: int
    ) -> dict:
        """Transform kwargs for per-shard maxpr execution."""
        # Default: pass kwargs unchanged (works for elementwise, reduce, etc.)
        return kwargs

    def infer_output_rank(self, input_shapes: tuple[tuple[int, ...], ...], **kwargs) -> int:
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
    
    def _setup_output_refs(self, output: Any, args: tuple, kwargs: dict, traced: bool) -> None:
        """Set up OutputRefs for tracing support."""
        from ..core import Tensor
        from ..core import pytree
        
        output_impls = [x._impl for x in pytree.tree_leaves(output) if isinstance(x, Tensor)]
        
        if not output_impls:
            return
        
        import weakref
        from ..core import OutputRefs
        
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ===----------------------------------------------------------------------=== #
# Specialized Operation Types (merged from types.py)
# ===----------------------------------------------------------------------=== #

class BinaryOperation(Operation):
    """Base for binary element-wise ops with batch_dims-aware broadcasting."""
    
    def compute_cost(self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]) -> float:
        """Elementwise binary op: 1 FLOP per output element."""
        if not output_shapes:
            return 0.0
        num_elements = 1
        for d in output_shapes[0]:
            num_elements *= d
        return float(num_elements)
    
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        from ..core.tensor import Tensor
        from . import view as view_ops
        # from . import _physical as physical_ops # REMOVED
        from .base import ensure_tensor
        
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
        x_batch_dims = x.batch_dims
        y_batch_dims = y.batch_dims

        # Get GLOBAL batch shape from physical_global_shape
        if x_batch_dims >= y_batch_dims:
            global_phys = x.physical_global_shape or x.local_shape
            batch_shape = tuple(int(d) for d in global_phys[:x_batch_dims])
        else:
            global_phys = y.physical_global_shape or y.local_shape
            batch_shape = tuple(int(d) for d in global_phys[:y_batch_dims])
        
        target_physical = batch_shape + target_logical
        
        # Optimize: Skip physical broadcast if specs match target
        x_global = x.physical_global_shape or x.local_shape
        y_global = y.physical_global_shape or y.local_shape
        
        # We need to broadcast strict physical shapes including batch dims
        if tuple(int(d) for d in x_global) != target_physical:
             x = view_ops.broadcast_to_physical(x, target_physical)
             
        if tuple(int(d) for d in y_global) != target_physical:
             y = view_ops.broadcast_to_physical(y, target_physical)
        
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
    
    Translates integer kwargs by batch_dims offset.
    """
    
    # True for unsqueeze (uses ndim+1 for negative axis normalization)
    axis_offset_for_insert: bool = False
    
    # Arguments that should be treated as axes and translated
    axis_arg_names: set[str] = {'axis', 'dim'}
    
    def __call__(self, x: Tensor, **kwargs: Any) -> Tensor:
        batch_dims = x.batch_dims
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
        """Reduce sharding rule: (d0, d1, ...) -> (d0, ...) with axis dim removed."""
        from ..core.sharding.propagation import OpShardingRuleTemplate
        
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


class ReduceOperation(LogicalAxisOperation):
    """Base for reduction operations (sum, mean, max, min, etc.)."""
    
    def compute_cost(self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]) -> float:
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
            for i in range(rank):
                if i in reduce_axes:
                    out_mapping[i] = [] # Reduced factor
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


class UnaryOperation(Operation):
    """Base for unary element-wise operations."""
    
    def compute_cost(self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]) -> float:
        """Unary elementwise op: 1 FLOP per element by default."""
        if not input_shapes:
            return 0.0
        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        return float(num_elements)


class LogicalShapeOperation(Operation):
    """Base for ops that take LOGICAL shape kwargs."""
    
    def __call__(self, x: Tensor, *, shape: tuple[int, ...]) -> Tensor:
        batch_dims = x.batch_dims
        
        if batch_dims > 0:
            if x.physical_global_shape is not None:
                global_batch_shape = tuple(int(d) for d in x.physical_global_shape[:batch_dims])
            else:
                global_phys = x.local_shape
                global_batch_shape = tuple(int(d) for d in global_phys[:batch_dims])
            physical_shape = global_batch_shape + tuple(shape)
        else:
            physical_shape = tuple(shape)
        
        return super().__call__(x, shape=physical_shape)
