# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Updated Binary Operations
# ===----------------------------------------------------------------------=== #

"""Binary ops using updated Operation base with auto batch_dims."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from max.graph import TensorValue, ops

from .operation import BinaryOperation, Operation

if TYPE_CHECKING:
    from ..core.tensor import Tensor


class AddOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "add"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.add(args[0], args[1])


class MulOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "mul"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.mul(args[0], args[1])


class SubOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "sub"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.sub(args[0], args[1])


class DivOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "div"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.div(args[0], args[1])


class MatmulOp(Operation):
    """Matmul with 1D promotion handling."""
    
    @property
    def name(self) -> str:
        return "matmul"
    
    def cost_model(self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]) -> float:
        """Estimate FLOPs for matmul: 2 * M * N * K."""
        # Standard case: A[..., M, K], B[..., K, N] -> C[..., M, N]
        if not input_shapes or len(input_shapes) < 2:
            return 0.0
            
        shape_a = input_shapes[0]
        shape_b = input_shapes[1]
        
        if len(shape_a) < 2 or len(shape_b) < 2:
            return 0.0
            
        # Last two dims are M, K and K, N
        m = shape_a[-2]
        k = shape_a[-1]
        n = shape_b[-1]
        
        # Batch size
        batch_size = 1
        for d in shape_a[:-2]:
            batch_size *= d
            
        return 2.0 * batch_size * m * n * k

    def communication_cost(
        self, 
        input_specs: list["ShardingSpec"], 
        output_specs: list["ShardingSpec"], 
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        mesh: "DeviceMesh"
    ) -> float:
        """Estimate communication cost for MatMul (e.g. MP AllReduce)."""
        from .communication import AllReduceOp
        
        if not input_specs or len(input_specs) < 2:
            return 0.0
            
        # MatMul Rule: ... mk, ... kn -> ... mn
        # Contracting dimension is 'k' (inner dimension)
        
        spec_a = input_specs[0]
        spec_b = input_specs[1]
        
        if not spec_a.dim_specs or not spec_b.dim_specs:
            return 0.0
            
        # A: [..., M, K] -> K is index -1
        # B: [..., K, N] -> K is index -2
        k_sharding_a = spec_a.dim_specs[-1].axes
        k_sharding_b = spec_b.dim_specs[-2].axes if len(spec_b.dim_specs) >= 2 else []
        
        # If K is sharded, we likely need an AllReduce on the output
        if k_sharding_a or k_sharding_b:
            if not output_shapes:
                return 0.0
            
            # Calculate output size in bytes (assuming float32)
            out_shape = output_shapes[0]
            num_elements = 1
            for d in out_shape:
                num_elements *= d
            output_bytes = num_elements * 4
            
            # Axes to reduce over: union of sharding axes on K
            reduce_axes = list(set(k_sharding_a) | set(k_sharding_b))
            
            return AllReduceOp.estimate_cost(output_bytes, mesh, reduce_axes)
            
        return 0.0
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.matmul(args[0], args[1])
    
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        from ..core.tensor import Tensor
        from . import view as view_ops
        from .operation import ensure_tensor
        
        # Ensure both inputs are Tensors (converts scalars/arrays)
        x = ensure_tensor(x)
        y = ensure_tensor(y)
        
        x_was_1d = len(x.shape) == 1
        y_was_1d = len(y.shape) == 1
        
        if x_was_1d:
            x = view_ops.unsqueeze(x, axis=0)
        if y_was_1d:
            y = view_ops.unsqueeze(y, axis=-1)
        
        result = super().__call__(x, y)
        
        if x_was_1d and y_was_1d:
            result = view_ops.squeeze(result, axis=-1)
            result = view_ops.squeeze(result, axis=-1)
        elif x_was_1d:
            result = view_ops.squeeze(result, axis=0)
        elif y_was_1d:
            result = view_ops.squeeze(result, axis=-1)
        
        return result
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
    ) -> Any:
        """Matmul: (batch..., m, k) @ (batch..., k, n) -> (batch..., m, n).
        
        k is the contracting factor (appears only in inputs).
        Also handles broadcast case where one input lacks batch dims.
        """
        from ..sharding.propagation import OpShardingRuleTemplate
        return OpShardingRuleTemplate.parse("... m k, ... k n -> ... m n", input_shapes).instantiate(
            input_shapes, output_shapes
        )
    
    # NOTE: No custom _infer_output_sharding needed - the generic factor-based
    # propagation handles matmul correctly via sharding_rule() above.



add = AddOp()
mul = MulOp()
sub = SubOp()
div = DivOp()
matmul = MatmulOp()


__all__ = [
    "AddOp", "MulOp", "SubOp", "DivOp", "MatmulOp",
    "add", "mul", "sub", "div", "matmul",
]
