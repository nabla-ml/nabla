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
    """Convert scalar (int/float) or array-like to Tensor.
    
    This should be called at the start of any operation that accepts
    multiple inputs to ensure all inputs are proper Tensors.
    """
    from ..core import Tensor
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
    
    def compute_cost(self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]) -> float:
        """Estimate compute cost (FLOPs) for this operation.
        
        Override in subclasses to provide accurate FLOP estimates.
        Communication costs are NOT included here - they are computed from
        the CollectiveOperations (AllReduce, AllGather, etc.) that get
        inserted by the SPMD execution based on factor propagation.
        
        Returns:
            Estimated FLOPs. Default is 0.0 (negligible compute).
        """
        return 0.0
    
    def memory_cost(
        self, 
        input_shapes: list[tuple[int, ...]], 
        output_shapes: list[tuple[int, ...]], 
        dtype_bytes: int = 4
    ) -> int:
        """Estimate memory usage (bytes) for output tensors.
        
        Used by the solver for memory-aware sharding decisions. This estimates
        the memory required to store the operation's output tensors.
        
        Args:
            input_shapes: Shapes of input tensors
            output_shapes: Shapes of output tensors
            dtype_bytes: Bytes per element (default 4 for float32)
        
        Returns:
            Total bytes for all output tensors.
        """
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
