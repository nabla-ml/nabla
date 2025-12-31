# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Custom Sharding Op
# ===----------------------------------------------------------------------=== #

from typing import Any, List, Tuple
from max.graph import TensorValue, ops
from nabla.ops.operation import Operation
from nabla.core.tensor import Tensor

class CustomSumReduceOp(Operation):
    """Custom reduction op with manual sharding rule."""

    @property
    def name(self) -> str:
        return "custom_sum_reduce"

    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        from nabla.ops.custom_mojo import call_custom_kernel

        # Output shape is (N,) if input is (N, M)
        # We need to manually compute output shape/type here for the helper
        # Retrieve input shape from tensor value type
        in_shape = x.type.shape
        out_shape = (in_shape[0],)
        
        # We need to construct the output type
        # Assuming we can just pass the shape to expected type
        from max.graph import TensorType
        out_type = TensorType(x.type.dtype, out_shape, x.type.device)

        result = call_custom_kernel(
            func_name="custom_sum_reduce",
            kernel_path="custom_kernels",
            values=x,
            out_types=out_type,
        )
        return result

    def sharding_rule(
        self,
        input_shapes: List[Tuple[int, ...]],
        output_shapes: List[Tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        """
        Generic Sharding Rule for reduction.
        Reads 'axis' and 'keepdims' from kwargs.
        Defaults to axis=1 to match custom_sum_reduce kernel behavior.
        """
        from nabla.sharding.propagation import reduce_template
        
        # Get reduction parameters from op_kwargs
        # Default to axis=1 because the underlying Mojo kernel is hardcoded for axis 1
        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 1)
        keepdims = kwargs.get("keepdims", False)
        
        return reduce_template(rank, [axis], keepdims).instantiate(input_shapes, output_shapes)

custom_sum_reduce = CustomSumReduceOp()

__all__ = ["custom_sum_reduce"]
