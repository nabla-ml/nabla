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
    from .sharding import ShardingSpec
    from .sharding_propagation import OpShardingRule


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
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the operation on Tensors.
        
        Uses symmetric tree_map for input/output conversion:
        - Input: Tensors -> TensorValues, everything else unchanged
        - Output: TensorValues -> Tensors, everything else unchanged
        
        maxpr receives the SAME pytree structure as __call__, just with
        TensorValues instead of Tensors. Output structure must match.
        
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
        
        def tensor_to_value(x: Any) -> Any:
            """Convert Tensor -> TensorValue, pass through everything else."""
            nonlocal any_traced
            if isinstance(x, Tensor):
                if x._impl.traced:
                    any_traced = True
                return g.TensorValue(x)
            return x
        
        def value_to_tensor(x: Any) -> Any:
            """Convert TensorValue -> Tensor, pass through everything else."""
            if pytree.is_tensor_value(x):
                impl = TensorImpl(
                    values=x,
                    op=self,
                    op_args=args,
                    op_kwargs=kwargs if kwargs else None,
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
        return pytree.tree_map(value_to_tensor, result_tree)
    
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
    ) -> tuple[Any, Any]:
        """Jacobian-vector product for forward-mode autodiff.
        
        Given input tangents (directional derivatives), compute the output tangent.
        
        All arguments are pytrees (nested dict/list/tuple) with Tensors at leaves.
        This matches __call__'s flexibility.
        
        Args:
            primals: Original input args (same structure as passed to __call__).
            tangents: Tangent vectors matching primals structure. Use None for
                leaves that are not differentiated (non-Tensors or constants).
            output: The forward pass output pytree.
            
        Returns:
            Tuple of (output pytree, output tangent pytree). Both should have
            the same structure as the forward pass output.
            
        Accessing Static Params:
            Static parameters are available in the primals pytree directly.
            
        Note:
            Use pytree utilities for processing:
                from . import pytree
                out_tangent = pytree.tree_map(compute_tangent, primals, tangents)
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

