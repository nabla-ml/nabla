# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
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


class Operation(ABC):
    """Base class for all operations in the computation graph.
    
    Each operation defines:
    - name: Unique identifier (e.g., 'add', 'matmul', 'sum')
    - maxpr(): Symbolic lowering to MAX graph (the ONLY execution path)
    - __call__(): Execute the operation on Tensors
    - vjp_rule(): Reverse-mode autodiff (optional)
    - jvp_rule(): Forward-mode autodiff (optional)
    
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
    
    def __call__(self, *args: Tensor, **kwargs: Any) -> Any:
        """Execute the operation on Tensors.
        
        This method:
        1. Extracts graph values from input Tensors
        2. Calls maxpr() within the graph context
        3. Wraps the result pytree in Tensors with proper tracing
        
        Args:
            *args: Input Tensors
            **kwargs: Operation-specific parameters
            
        Returns:
            A pytree of Tensors matching the structure returned by maxpr().
            For single-output ops, returns a single Tensor.
            For multi-output ops, returns tuple/list/dict of Tensors.
        """
        from .tensor import Tensor
        from .tensor_impl import TensorImpl
        from .compute_graph import GRAPH
        from . import pytree
        from max import graph as g
        
        # Determine if any input is traced
        parent_impls: list[TensorImpl] = []
        any_traced = False
        for arg in args:
            if isinstance(arg, Tensor):
                parent_impls.append(arg._impl)
                if arg._impl.traced:
                    any_traced = True
        
        # Execute maxpr within graph context
        with GRAPH.graph:
            # Get TensorValues from Tensors
            tensor_values = [g.TensorValue(arg) for arg in args]
            result_tree = self.maxpr(*tensor_values, **kwargs)
        
        # Handle single TensorValue output (common case, optimize for it)
        if pytree.is_tensor_value(result_tree):
            # Single output - create TensorImpl directly (no sibling tracking needed)
            output_impl = TensorImpl(
                values=result_tree,
                parents=parent_impls,
                op=self,
                op_kwargs=kwargs if kwargs else None,
                traced=any_traced,
            )
            return Tensor(impl=output_impl)
        
        # Multi-output case - use pytree wrapping with sibling tracking
        return pytree.wrap_tensor_values(
            result_tree,
            parent_impls=parent_impls,
            op=self,
            op_kwargs=kwargs if kwargs else None,
            traced=any_traced,
        )
    
    # ===== Autodiff Rules (Optional) =====
    
    def vjp_rule(
        self, 
        primals: list[Tensor],
        cotangent: Any,
        output: Any,
    ) -> list[Tensor | None]:
        """Vector-Jacobian product for reverse-mode autodiff.
        
        Given the output cotangent (grad w.r.t. output), compute cotangents
        for each input primal. Return None for inputs that don't need gradients.
        
        Args:
            primals: Original input Tensors to this operation
            cotangent: Gradient flowing back from the output. For single-output
                ops, this is a Tensor. For multi-output ops, this is a pytree
                matching the output structure (with Tensors at each leaf).
            output: The forward pass output. For single-output ops, this is a
                Tensor. For multi-output ops, this is a pytree of Tensors.
                Access operation kwargs via output.op_kwargs (or first element
                if pytree).
            
        Returns:
            List of gradients for each primal (None if not differentiable)
            
        Note:
            For multi-output ops, you can use pytree utilities:
                from . import pytree
                cot_leaves = pytree.tree_leaves(cotangent)
                out_leaves = pytree.tensor_leaves(output)
        """
        raise NotImplementedError(
            f"Operation '{self.name}' does not implement vjp_rule"
        )
    
    def jvp_rule(
        self,
        primals: list[Tensor],
        tangents: list[Tensor | None],
        output: Any,
    ) -> tuple[Any, Any]:
        """Jacobian-vector product for forward-mode autodiff.
        
        Given input tangents (directional derivatives), compute the output tangent.
        
        Args:
            primals: Original input Tensors to this operation
            tangents: Tangent vectors for each primal (None if not differentiated)
            output: The forward pass output (single Tensor or pytree of Tensors).
                Access operation kwargs via output.op_kwargs (or first element
                if pytree).
            
        Returns:
            Tuple of (output pytree, output tangent pytree). Both should have
            the same structure as the forward pass output.
            
        Note:
            For multi-output ops, return tangents as a matching pytree.
        """
        raise NotImplementedError(
            f"Operation '{self.name}' does not implement jvp_rule"
        )
    
    # ===== Sharding (Optional) =====
    
    def sharding_rule(
        self,
        inputs: list[Tensor],
        output: Tensor,
    ) -> None:
        """Propagate or infer sharding annotations.
        
        Given the input and output tensors, this rule can:
        - Propagate sharding from inputs to output (forward)
        - Propagate sharding from output to inputs (backward)
        - Infer optimal sharding based on operation semantics
        
        The rule reads/writes sharding annotations via tensor._impl.sharding.
        Operation kwargs are accessible via output.op_kwargs.
        
        Args:
            inputs: Input Tensors with their current sharding annotations
            output: Output Tensor with its current sharding annotation
            
        Note:
            This is bidirectional - the rule may update sharding on
            either inputs or output depending on what's already annotated.
        """
        raise NotImplementedError(
            f"Operation '{self.name}' does not implement sharding_rule"
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

