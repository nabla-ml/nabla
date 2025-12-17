# ===----------------------------------------------------------------------=== #
# Nabla 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Base class for all operations in the computation graph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from max import graph
    from .tensor_impl import TensorImpl
    from .tensor import Tensor
    from .sharding import ShardingSpec
    from .sharding_propagation import OpShardingRule


class Operation(ABC):
    """Base class for all operations in the computation graph.
    
    Each operation defines:
    - name: Unique identifier (e.g., 'add', 'matmul', 'sum')
    - maxpr(): Symbolic lowering to MAX graph (the ONLY execution path)
    - __call__(): Execute the operation on Tensors (auto-detects JVP mode)
    - vjp_rule(): Reverse-mode autodiff (optional)
    - jvp_rule(): Forward-mode autodiff (optional)
    
    JVP Mode:
        When any input Tensor has its ._impl.tangent set, __call__ automatically:
        1. Computes the primal output normally via maxpr
        2. Calls jvp_rule to compute the output tangent
        3. Attaches the output tangent to the output Tensor's ._impl.tangent
    
    Batch dimensions are always the prefix of the physical shape.
    In maxpr(), translate logical axes to physical by adding batch_dims:
        physical_axis = logical_axis + batch_dims (for positive axes)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique operation name."""
        ...
    
    @abstractmethod
    def maxpr(self, *args: graph.TensorValue, **kwargs: Any) -> Any:
        """Symbolic lowering to MAX graph operations.
        
        This is the ONLY execution path - we are always lazy.
        
        Args:
            *args: Input TensorValues from the graph
            **kwargs: Operation-specific parameters
            
        Returns:
            A pytree (single value, tuple, list, or dict) of TensorValues.
            For single-output ops, return a single TensorValue.
            For multi-output ops (split, svd, topk), return tuple/list/dict.
        """
        ...
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the operation on Tensors.
        
        Uses symmetric tree_map for input/output conversion:
        - Input: Tensors -> TensorValues, everything else unchanged
        - Output: TensorValues -> Tensors, everything else unchanged
        
        JVP Mode:
            If any input Tensor has ._impl.tangent set, this method:
            1. Computes primal output normally
            2. Extracts tangents from inputs
            3. Calls jvp_rule(primals, tangents, output) to get output tangent
            4. Attaches output tangent to output._impl.tangent
        
        Args:
            *args: Pytree of inputs (Tensors and non-Tensors)
            **kwargs: Operation-specific parameters
            
        Returns:
            Pytree matching maxpr's output, with TensorValues wrapped as Tensors.
        """
        from .tensor import Tensor
        from .tensor_impl import TensorImpl
        from .compute_graph import GRAPH
        from . import pytree
        from max import graph as g
        
        # Track if any input Tensor is traced (for graph building)
        any_traced = False
        # Track if any input has tangent (for JVP mode)
        any_has_tangent = False
        
        def tensor_to_value(x: Any) -> Any:
            """Convert Tensor -> TensorValue, pass through everything else."""
            nonlocal any_traced, any_has_tangent
            if isinstance(x, Tensor):
                if x._impl.traced:
                    any_traced = True
                if x._impl.tangent is not None:
                    any_has_tangent = True
                return g.TensorValue(x)
            return x
        
        def value_to_tensor(x: Any) -> Any:
            """Convert TensorValue -> Tensor, pass through everything else."""
            if pytree.is_tensor_value(x):
                # NOTE: We don't pass op/args to TensorImpl anymore.
                # They are now stored in the shared OutputRefs instance.
                impl = TensorImpl(
                    values=x,
                    traced=any_traced,
                )
                impl.cache_metadata(x)
                return Tensor(impl=impl)
            return x
        
        # Convert inputs: Tensor -> TensorValue
        with GRAPH.graph:
            converted_args = pytree.tree_map(tensor_to_value, args)
            result_tree = self.maxpr(*converted_args, **kwargs)
        
        # Convert outputs: TensorValue -> Tensor
        output = pytree.tree_map(value_to_tensor, result_tree)
        
        # Populate OutputRefs for multi-output operation tracking
        # Flatten outputs to get all TensorImpl leaves
        output_impls = [
            x._impl for x in pytree.tree_leaves(output) 
            if isinstance(x, Tensor)
        ]
        
        if output_impls:
            # Create OutputRefs with weak references to avoid cycles
            import weakref
            from .tracing import OutputRefs
            
            # Get the output tree structure
            _, output_tree_def = pytree.tree_flatten(output, is_leaf=pytree.is_tensor)
            
            # Create weak references to all output TensorImpls
            weak_refs = tuple(weakref.ref(impl) for impl in output_impls)
            
            # Create shared OutputRefs instance with operation metadata
            # Only store op_args/kwargs if traced (to allow GC in untraced mode)
            # CRITICAL: Store TensorImpl refs, not Tensor wrappers, to preserve
            # the weak-ref GC strategy (Tensor holds weak ref to TensorImpl)
            def to_impl(x: Any) -> Any:
                """Convert Tensor -> TensorImpl, pass through everything else."""
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
            
            # Populate each output with the shared OutputRefs and its index
            for idx, impl in enumerate(output_impls):
                impl.output_refs = output_refs
                impl.output_index = idx
        
        # JVP mode: if any input has tangent, propagate to output
        if any_has_tangent:
            # Extract tangents (structure matches args)
            tangents = pytree.tree_map(
                lambda x: Tensor(impl=x._impl.tangent) if isinstance(x, Tensor) and x._impl.tangent else None,
                args
            )
            # Compute output tangent
            output_tangent = self.jvp_rule(args, tangents, output)
            # Attach to output (mutates output._impl.tangent in place)
            if output_tangent is not None:
                pytree.tree_map(
                    lambda o, t: setattr(o._impl, 'tangent', t._impl) if isinstance(o, Tensor) and isinstance(t, Tensor) else None,
                    output, output_tangent
                )
        
        return output
    
    # ===== Autodiff Rules (Optional) =====
    
    def vjp_rule(
        self, 
        primals: Any,
        cotangent: Any,
        output: Any,
    ) -> Any:
        """Vector-Jacobian product for reverse-mode autodiff.
        
        Given the output cotangent (grad w.r.t. output), compute cotangents
        for each input primal. Return None for inputs that don't need gradients.
        
        All arguments and return values are pytrees (nested dict/list/tuple)
        with Tensors at the leaves. This matches __call__'s flexibility.
        
        Args:
            primals: Original input args (same structure as passed to __call__).
                This is the full pytree including non-Tensor static params.
            cotangent: Gradient flowing back from the output. Structure matches
                the output pytree.
            output: The forward pass output pytree.
            
        Returns:
            Pytree of gradients matching the primals structure. Use None for
            leaves that are not differentiable (non-Tensors or Tensors without
            gradients).
            
        Accessing Static Params:
            Static parameters are available in the primals pytree directly.
            Additional context available via:
            - output._impl.op_args: Original positional args tuple
            - output._impl.op_kwargs: Keyword arguments dict
            
        Note:
            Use pytree utilities for processing:
                from . import pytree
                leaves = pytree.tree_leaves(primals)
                grads = pytree.tree_map(compute_grad, primals, cotangent)
        """
        raise NotImplementedError(
            f"Operation '{self.name}' does not implement vjp_rule"
        )
    
    def jvp_rule(
        self,
        primals: Any,
        tangents: Any,
        output: Any,
    ) -> Any:
        """Jacobian-vector product for forward-mode autodiff.
        
        Given input tangents (directional derivatives), compute the output tangent.
        The primal output is already computed and provided for use as a "residual".
        
        All arguments are pytrees (nested dict/list/tuple) with Tensors at leaves.
        This matches __call__'s flexibility.
        
        Args:
            primals: Original input args (same structure as passed to __call__).
            tangents: Tangent vectors matching primals structure. None for
                leaves that are not differentiated (non-Tensors or constants).
            output: The already-computed forward pass output pytree.
                Often useful as a "free residual" (e.g., exp(x) tangent = output * tangent_in).
            
        Returns:
            Output tangent pytree matching the output structure.
            Return None for outputs that don't have tangents.
            
        Example for exp:
            def jvp_rule(self, primals, tangents, output):
                # d/dt exp(x + t*dx) = exp(x) * dx = output * tangent
                return output * tangents[0]
        """
        raise NotImplementedError(
            f"Operation '{self.name}' does not implement jvp_rule"
        )
    
    # ===== Sharding (Optional) =====
    
    def sharding_rule(
        self,
        inputs: Any,
        output: Any,
    ) -> Any:
        """Propagate or infer sharding annotations.
        
        Given the input and output tensors (as pytrees), this rule can:
        - Propagate sharding from inputs to output (forward)
        - Propagate sharding from output to inputs (backward)
        - Infer optimal sharding based on operation semantics
        
        All arguments are pytrees matching the structure of __call__ args/outputs.
        
        Args:
            inputs: Input pytree with current sharding annotations.
                Access sharding via tensor._impl.sharding for each Tensor leaf.
            output: Output pytree with current sharding annotation.
            
        Returns:
            An OpShardingRule or None. The rule may also mutate sharding
            annotations in-place on the input/output tensors.
            
        Note:
            This is bidirectional - the rule may update sharding on
            either inputs or output depending on what's already annotated.
        """
        raise NotImplementedError(
            f"Operation '{self.name}' does not implement sharding_rule"
        )
    
    def get_sharding_rule_template(self) -> Any:
        """Get the sharding rule template for this operation.
        
        Returns:
            OpShardingRuleTemplate or None
        """
        return None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class BinaryOperation(Operation):
    """Abstract base class for binary element-wise operations.
    
    Handles batch_dims-aware broadcasting for vmap support:
    1. Compute output batch_dims = max(input1.batch_dims, input2.batch_dims)
    2. For traced inputs, unsqueeze and broadcast to match output shapes
    
    Subclasses only need to implement:
    - name property
    - maxpr(x, y) -> TensorValue (the actual element-wise operation)
    - jvp_rule (optional)
    - vjp_rule (optional)
    """
    
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """Execute binary operation with batch_dims-aware broadcasting.
        
        For traced tensors, explicitly broadcasts to ensure correct gradient shapes.
        For untraced tensors, relies on MAX's implicit broadcasting.
        
        Args:
            x: First input tensor
            y: Second input tensor
            
        Returns:
            Result tensor with batch_dims = max(x.batch_dims, y.batch_dims)
        """
        from .tensor import Tensor
        
        # Determine output batch_dims
        x_batch = x._impl.batch_dims
        y_batch = y._impl.batch_dims
        out_batch_dims = max(x_batch, y_batch)
        
        # Check if any input is traced (need explicit broadcasting for gradients)
        any_traced = x._impl.traced or y._impl.traced
        
        # If traced, we need to unsqueeze and broadcast for correct gradient shapes
        if any_traced:
            x, y = self._prepare_for_broadcast(x, y, out_batch_dims)
        
        # Delegate to parent's __call__ which handles maxpr execution and JVP
        result = super().__call__(x, y)
        
        # Set output batch_dims
        if isinstance(result, Tensor):
            result._impl.batch_dims = out_batch_dims
        
        return result
    
    def _prepare_for_broadcast(
        self, 
        x: Tensor, 
        y: Tensor, 
        out_batch_dims: int
    ) -> tuple[Tensor, Tensor]:
        """Prepare tensors for broadcasting when traced.
        
        This ensures correct gradient shapes by:
        1. Unsqueezing to match ranks (batch dims and logical dims separately)
        2. Broadcasting to output shape
        
        Args:
            x: First tensor
            y: Second tensor  
            out_batch_dims: Target batch dims for output
            
        Returns:
            Tuple of (prepared_x, prepared_y)
        """
        # Import view ops lazily to avoid circular imports
        from . import view_ops
        
        # Use PHYSICAL shapes (includes batch dims) for correct splitting
        x_physical = x._impl.physical_shape
        y_physical = y._impl.physical_shape
        x_batch = x._impl.batch_dims
        y_batch = y._impl.batch_dims
        
        # Split into batch and logical shapes
        x_batch_shape = x_physical[:x_batch]
        x_logical_shape = x_physical[x_batch:]
        y_batch_shape = y_physical[:y_batch]
        y_logical_shape = y_physical[y_batch:]
        
        # Compute broadcasted shapes for batch and logical separately
        out_batch_shape = self._broadcast_shapes(x_batch_shape, y_batch_shape)
        out_logical_shape = self._broadcast_shapes(x_logical_shape, y_logical_shape)
        out_physical_shape = out_batch_shape + out_logical_shape
        
        # Prepare x: unsqueeze then broadcast if needed
        if x._impl.traced:
            x = self._unsqueeze_to_rank(x, len(out_physical_shape), x_batch, out_batch_dims)
            current_physical = x._impl.physical_shape
            if current_physical != out_physical_shape:
                x = view_ops.broadcast_to(x, out_physical_shape)
        
        # Prepare y: unsqueeze then broadcast if needed  
        if y._impl.traced:
            y = self._unsqueeze_to_rank(y, len(out_physical_shape), y_batch, out_batch_dims)
            current_physical = y._impl.physical_shape
            if current_physical != out_physical_shape:
                y = view_ops.broadcast_to(y, out_physical_shape)
        
        return x, y
    
    def _unsqueeze_to_rank(
        self, 
        t: Tensor, 
        target_rank: int,
        current_batch_dims: int,
        target_batch_dims: int
    ) -> Tensor:
        """Unsqueeze tensor to target rank, adding dims at correct positions.
        
        Adds dimensions at:
        - The start of batch dims (to match target_batch_dims)
        - The start of logical dims (to match target logical rank)
        """
        from . import view_ops
        
        current_shape = t.shape
        current_rank = len(current_shape)
        
        # Number of batch dims to add at front
        batch_dims_to_add = target_batch_dims - current_batch_dims
        
        # Number of logical dims to add (after batch dims)
        current_logical_rank = current_rank - current_batch_dims
        target_logical_rank = target_rank - target_batch_dims
        logical_dims_to_add = target_logical_rank - current_logical_rank
        
        # Add batch dims at front
        for _ in range(batch_dims_to_add):
            t = view_ops.unsqueeze(t, axis=0)
        
        # Add logical dims after batch (at position = new batch_dims)
        for _ in range(logical_dims_to_add):
            t = view_ops.unsqueeze(t, axis=target_batch_dims)
        
        return t
    
    @staticmethod
    def _broadcast_shapes(
        shape1: tuple[int, ...], 
        shape2: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Compute broadcasted shape for two shapes.
        
        Standard numpy-style broadcasting rules.
        """
        # Pad shorter shape with 1s at front
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
                raise ValueError(
                    f"Cannot broadcast shapes {shape1} and {shape2}"
                )
        return tuple(result)

