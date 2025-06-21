"""
Take and Give operations for tensor indexing.

This module implements complementary operations:
- "take": Select elements from input using index tensor along specified axis (like gather)
- "give": Scatter/accumulate values into target tensor at specified indices (inverse of take)

These operations are designed to work with MAX graph operations and provide
proper gradient computation through VJP/JVP rules.
"""

__all__ = ["gather", "scatter"]


import numpy as np
from max.driver import Tensor
from max.dtype import DType
from max.graph import TensorValue, ops

from ..core.array import Array
from .operation import Operation


class GatherOp(Operation):
    def __init__(self, axis: int = -1):
        """
        Initialize gather operation.

        Args:
            axis: The dimension which indices indexes from input.
                  If negative, indexes relative to the end of the input tensor.
        """
        super().__init__("gather")
        self.axis = axis

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """
        Compute the output shape for take operation.

        The output shape replaces the indexed dimension with the indices shape.

        Args:
            input_shapes: (input_shape, indices_shape)

        Returns:
            Output shape tuple
        """
        input_shape, indices_shape = input_shapes

        # Normalize negative axis
        axis = self.axis
        if axis < 0:
            axis += len(input_shape)

        # For gather: input_shape with axis dimension replaced by indices_shape
        # The indices_shape should be inserted at the axis position
        output_shape = input_shape[:axis] + indices_shape + input_shape[axis + 1 :]

        return output_shape

    def compute_output_batch_dims(self, *input_batch_dims: tuple) -> tuple:
        """
        Compute output batch dims for gather operation.

        For gather with vmap, the output batch dims should be the broadcasted
        batch dimensions of both input array and indices.

        Args:
            input_batch_dims: (input_batch_dims, indices_batch_dims)

        Returns:
            Broadcasted batch dims of input array and indices
        """
        if len(input_batch_dims) != 2:
            raise ValueError(
                f"Gather operation requires 2 input batch dims, got {len(input_batch_dims)}"
            )

        input_batch_dims_val, indices_batch_dims_val = (
            input_batch_dims[0],
            input_batch_dims[1],
        )

        # Use the standard broadcasting logic for batch dimensions
        from ..utils.shape_utils import get_broadcasted_shape

        return get_broadcasted_shape(input_batch_dims_val, indices_batch_dims_val)

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        """
        MAX graph implementation using max.graph.ops.gather.

        Args:
            args: [input_tensor, indices_tensor]
            output: Output array to store result
        """
        input_tensor, indices_tensor = args

        # Import MAX ops

        # Ensure indices are integers for MAX
        if indices_tensor.type.dtype.name != "int64":
            indices_tensor = ops.cast(indices_tensor, ops.DType.int64)

        # Use MAX's gather operation
        result = ops.gather(input_tensor, indices_tensor, axis=self.axis)
        output.tensor_value = result

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        """
        Eager computation using NumPy take operation.

        Args:
            args: [input_array, indices_array]
            output: Output array to store result
        """
        input_array, indices_array = args

        input_np = input_array.to_numpy()
        indices_np = indices_array.to_numpy()

        if indices_np.dtype.kind not in {"i", "u"}:
            raise ValueError(
                f"Indices array must be of integer type, got {indices_np.dtype}"
            )

        # Handle batched case properly
        if input_array.batch_dims or indices_array.batch_dims:
            # For batched operations, we need to apply gather to each batch element
            batch_size = input_np.shape[0]
            results = []

            for i in range(batch_size):
                # Extract the i-th batch element from both arrays
                input_batch_i = input_np[i]  # Shape: input_array.shape
                indices_batch_i = indices_np[i]  # Shape: indices_array.shape

                # Apply gather to this batch element
                result_batch_i = np.take(input_batch_i, indices_batch_i, axis=self.axis)
                results.append(result_batch_i)

            # Stack the results back into a batch
            result_np = np.stack(results, axis=0)
        else:
            # Non-batched case - direct numpy take
            result_np = np.take(input_np, indices_np, axis=self.axis)

        output.impl = Tensor.from_numpy(result_np)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        """
        Vector-Jacobian product rule for gather operation.

        Args:
            primals: [input_array, indices_array] - the forward pass inputs
            cotangent: Gradient flowing back from output
            output: Forward pass output (for reference)

        Returns:
            [input_grad, indices_grad] where indices_grad is zero array
        """
        input_array, indices_array = primals

        target_shape = input_array.shape

        # Use scatter from this module instead of view
        input_grad = scatter(target_shape, indices_array, cotangent, axis=self.axis)

        # Indices don't need gradients, but we need to return a zero array of the same shape
        from ..ops.creation import zeros

        indices_grad = zeros(indices_array.shape, dtype=input_array.dtype)

        return [input_grad, indices_grad]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        """
        Jacobian-vector product rule for gather operation.

        Args:
            primals: [input_array, indices_array] - the forward pass inputs
            tangents: [input_tangent, indices_tangent] - the tangent vectors
            output: Forward pass output (for reference)

        Returns:
            Output tangent
        """
        input_tangent, indices_tangent = tangents
        # Indices tangents are ignored (indices are discrete)
        # Apply the same gather operation to input tangents
        return gather(input_tangent, indices=primals[1], axis=self.axis)

    def compute_output_dtype(self, input_array: Array, indices: Array) -> DType:
        """Output dtype same as input array dtype."""
        return input_array.dtype

    def forward(self, *args: Array) -> Array:
        """Forward pass for gather operation."""
        if len(args) != 2:
            raise ValueError(f"Gather operation requires 2 arguments, got {len(args)}")

        # Move arrays to best device (like BinaryOperation does)
        from .operation import move_to_best_device

        args = move_to_best_device(*args)
        input_array, indices = args

        # Validate inputs
        if not isinstance(input_array, Array) or not isinstance(indices, Array):
            raise TypeError("Both arguments must be Array instances")

        output_shape = self.compute_output_shape(input_array.shape, indices.shape)
        output_batch_dims = self.compute_output_batch_dims(
            input_array.batch_dims, indices.batch_dims
        )
        output_dtype = self.compute_output_dtype(input_array, indices)

        res = Array(
            shape=output_shape,
            dtype=output_dtype,
            device=input_array.device,
            materialize=False,
            name=self.name,
            batch_dims=output_batch_dims,
        )

        res.set_maxpr(self.maxpr)
        res.add_arguments(input_array, indices)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule
        res.custom_kernel_path = self.custom_kernel_path()

        if not res.stage_realization:
            self.eagerxpr([input_array, indices], res)

        return res


def gather(input_array: Array, indices: Array, axis: int = -1) -> Array:
    """
    Gather operation - select elements from input_array using indices along specified axis.

    Args:
        input_array: Input tensor to gather from
        indices: Index tensor specifying which elements to gather
        axis: Axis along which to gather (default: -1)

    Returns:
        Output tensor with gathered elements
    """
    # Normalize axis to be negative (following the codebase pattern)
    if axis >= 0:
        # Convert positive axis to negative relative to input shape
        axis = axis - len(input_array.shape)

    op = GatherOp(axis)
    return op.forward(input_array, indices)


class ScatterOp(Operation):
    def __init__(self, target_shape: tuple, axis: int = -1):
        """
        Initialize scatter operation.

        Args:
            target_shape: Shape of the output tensor
            axis: The dimension along which to scatter indices
        """
        super().__init__("scatter")
        self.target_shape = target_shape
        self.axis = axis

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """
        Compute the output shape for give operation.

        Args:
            input_shapes: (indices_shape, values_shape)

        Returns:
            target_shape (fixed by constructor)
        """
        # Convert Array objects to plain integers if needed (for JIT compatibility)
        shape_list = []
        for dim in self.target_shape:
            if hasattr(dim, "to_numpy"):
                # It's an Array object, convert to scalar
                shape_list.append(int(dim.to_numpy().item()))
            else:
                # It's already a plain integer
                shape_list.append(int(dim))
        return tuple(shape_list)

    def compute_output_batch_dims(self, *input_batch_dims: tuple) -> tuple:
        """
        Compute output batch dims for scatter operation.

        Args:
            input_batch_dims: (indices_batch_dims, values_batch_dims)

        Returns:
            Broadcasted batch dims
        """
        if len(input_batch_dims) != 2:
            raise ValueError(
                f"Scatter operation requires 2 input batch dims, got {len(input_batch_dims)}"
            )

        indices_batch_dims, values_batch_dims = input_batch_dims[0], input_batch_dims[1]

        from ..utils.shape_utils import get_broadcasted_shape

        return get_broadcasted_shape(indices_batch_dims, values_batch_dims)

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        """
        MAX graph implementation using max.graph.ops.scatter_nd for flexible indexing.

        Args:
            args: [indices_tensor, values_tensor]
            output: Output array to store result
        """
        indices_tensor, values_tensor = args

        # Import MAX ops
        from max.graph import DeviceRef

        # Ensure indices are integers for MAX
        if indices_tensor.type.dtype.name not in ["int32", "int64"]:
            indices_tensor = ops.cast(indices_tensor, ops.DType.int64)

        # Create zero tensor with target shape
        zero_scalar = ops.constant(
            0.0, dtype=output.dtype, device=DeviceRef.from_device(output.device)
        )
        target_shape = list(output.batch_dims) + list(output.shape)
        zero_tensor = ops.broadcast_to(zero_scalar, target_shape)

        # Normalize axis (convert from negative to positive)
        axis = self.axis
        if axis < 0:
            axis += len(output.shape)

        # Adjust axis to account for batch dimensions
        axis_with_batch = axis + len(output.batch_dims)

        # For scatter_nd, we need to create multi-dimensional indices
        # Based on the axis, we need to reshape indices appropriately
        if axis_with_batch == 0:
            # For axis=0, reshape indices from [...] to [..., 1] 
            # This specifies single coordinate along the first dimension
            indices_nd = ops.unsqueeze(indices_tensor, -1)
        else:
            # For other axes, we need to create full coordinate tuples
            # For now, let's fall back to regular scatter for non-zero axes
            # to maintain compatibility
            
            # Expand indices to have the same rank as values if needed
            values_shape = values_tensor.type.shape
            indices_shape = indices_tensor.type.shape

            # If indices has fewer dimensions than values, broadcast it to match
            if len(indices_shape) < len(values_shape):
                # Add dimensions of size 1 to match values rank
                for _ in range(len(values_shape) - len(indices_shape)):
                    indices_tensor = ops.unsqueeze(indices_tensor, -1)

                # Now broadcast to match values shape
                indices_tensor = ops.broadcast_to(indices_tensor, values_shape)

            # Use regular scatter for non-zero axes
            result = ops.scatter(
                zero_tensor, values_tensor, indices_tensor, axis=axis_with_batch
            )
            output.tensor_value = result
            return

        # Use scatter_nd for axis=0 case
        result = ops.scatter_nd(zero_tensor, values_tensor, indices_nd)
        output.tensor_value = result

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        """
        Eager computation using NumPy scatter operation.

        Args:
            args: [indices_array, values_array]
            output: Output array to store result
        """
        indices_array, values_array = args

        indices_np = indices_array.to_numpy()
        values_np = values_array.to_numpy()

        if indices_np.dtype.kind not in {"i", "u"}:
            raise ValueError(
                f"Indices array must be of integer type, got {indices_np.dtype}"
            )

        # Create zero array with target shape
        if indices_array.batch_dims or values_array.batch_dims:
            # Batched case: create zeros with batch dimensions included
            full_shape = list(output.batch_dims) + list(output.shape)
        else:
            # Non-batched case: use target shape directly
            full_shape = list(self.target_shape)

        result_np = np.zeros(full_shape, dtype=values_np.dtype)

        # Handle batched case properly
        if indices_array.batch_dims or values_array.batch_dims:
            # For batched operations, we need to apply scatter to each batch element
            batch_size = indices_np.shape[0]

            for i in range(batch_size):
                # Extract the i-th batch element from both arrays
                indices_batch_i = indices_np[i]  # Shape: indices_array.shape
                values_batch_i = values_np[i]  # Shape: values_array.shape

                # Apply scatter to this batch element using advanced indexing
                self._scatter_single(
                    result_np[i], indices_batch_i, values_batch_i, self.axis
                )
        else:
            # Non-batched case - direct numpy scatter
            self._scatter_single(result_np, indices_np, values_np, self.axis)

        output.impl = Tensor.from_numpy(result_np)

    def _scatter_single(
        self,
        target_array: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
        axis: int,
    ) -> None:
        """Helper method to scatter values into target_array at given indices along axis."""
        # Normalize negative axis
        if axis < 0:
            axis += target_array.ndim

        # Build indexing tuple for advanced indexing
        idx = [slice(None)] * target_array.ndim
        idx[axis] = indices  # type: ignore

        # Use advanced indexing to set values
        target_array[tuple(idx)] = values

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        """
        Vector-Jacobian product rule for give operation.

        Args:
            primals: [indices, values] - the forward pass inputs
            cotangent: Gradient flowing back from output
            output: Forward pass output (for reference)

        Returns:
            [indices_grad, values_grad] where indices_grad is zero array
        """
        indices_array, values_array = primals

        # Indices don't need gradients, but we need to return a zero array of the same shape
        from ..ops.creation import zeros

        indices_grad = zeros(
            indices_array.shape, dtype=values_array.dtype
        )  # Use values dtype

        # Values gradient: gather the cotangent at the same indices
        values_grad = gather(cotangent, indices_array, axis=self.axis)

        return [indices_grad, values_grad]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        """
        Jacobian-vector product rule for give operation.

        Args:
            primals: [indices, values] - the forward pass inputs
            tangents: [indices_tangent, values_tangent] - the tangent vectors
            output: Forward pass output (for reference)

        Returns:
            Output tangent
        """
        indices_tangent, values_tangent = tangents

        # Indices tangents are ignored (indices are discrete)
        # Apply the same scatter operation to values tangents
        return scatter(
            self.target_shape, primals[0], values_tangent, axis=self.axis
        )  # Use original indices

    def compute_output_dtype(self, indices: Array, values: Array) -> DType:
        """Output dtype same as values array dtype."""
        return values.dtype

    def forward(self, *args: Array) -> Array:
        """Forward pass for scatter operation."""
        if len(args) != 2:
            raise ValueError(f"Scatter operation requires 2 arguments, got {len(args)}")

        # Move arrays to best device (like BinaryOperation does)
        from .operation import move_to_best_device

        args = move_to_best_device(*args)
        indices, values = args

        # Validate inputs
        if not isinstance(indices, Array) or not isinstance(values, Array):
            raise TypeError("Both arguments must be Array instances")

        output_shape = self.compute_output_shape(indices.shape, values.shape)
        output_batch_dims = self.compute_output_batch_dims(
            indices.batch_dims, values.batch_dims
        )
        output_dtype = self.compute_output_dtype(indices, values)

        res = Array(
            shape=output_shape,
            dtype=output_dtype,
            device=values.device,  # Use values device since that's the data
            materialize=False,
            name=self.name,
            batch_dims=output_batch_dims,
        )

        res.set_maxpr(self.maxpr)
        res.add_arguments(indices, values)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule
        res.custom_kernel_path = self.custom_kernel_path()

        if not res.stage_realization:
            self.eagerxpr([indices, values], res)

        return res


def scatter(
    target_shape: tuple, indices: Array, values: Array, axis: int = -1
) -> Array:
    if axis >= 0:
        # make negative
        axis = axis - len(target_shape)
    op = ScatterOp(target_shape, axis)
    return op.forward(indices, values)
