# ===----------------------------------------------------------------------=== #
# Nabla 2025
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

"""View and shape manipulation operations."""

import numpy as np
from max.driver import Tensor
from max.graph import Value, ops

from ..core.array import Array, Shape
from .operation import Operation, ViewOperation

# Public API
__all__ = [
    "transpose",
    "permute",
    "move_axis_to_front",
    "move_axis_from_front",
    "reshape",
    "broadcast_to",
    "broadcast_batch_dims",
    "squeeze",
    "unsqueeze",
    "squeeze_batch_dims",
    "unsqueeze_batch_dims",
    "shallow_copy",
    "array_slice",
    "pad",
    "concatenate",
]


class TransposeOp(ViewOperation):
    """Matrix/tensor transpose operation."""

    def __init__(self, axis_1: int = -2, axis_2: int = -1):
        super().__init__(f"transpose[permutation=({axis_1},{axis_2})]")
        self.axis_1 = axis_1
        self.axis_2 = axis_2

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compute output shape for transpose operation with compatible signature."""
        if len(input_shapes) != 1:
            raise ValueError(
                f"Transpose operation requires 1 input shape, got {len(input_shapes)}"
            )
        arg_shape = input_shapes[0]

        if not arg_shape:
            raise ValueError("Cannot transpose an empty shape")

        axis_1 = self.axis_1 if self.axis_1 >= 0 else len(arg_shape) + self.axis_1
        axis_2 = self.axis_2 if self.axis_2 >= 0 else len(arg_shape) + self.axis_2

        if axis_1 < 0 or axis_1 >= len(arg_shape):
            raise ValueError(f"axis_1 {axis_1} is out of bounds for shape {arg_shape}")
        if axis_2 < 0 or axis_2 >= len(arg_shape):
            raise ValueError(f"axis_2 {axis_2} is out of bounds for shape {arg_shape}")

        new_shape = list(arg_shape)
        new_shape[axis_1], new_shape[axis_2] = new_shape[axis_2], new_shape[axis_1]
        return tuple(new_shape)

    def maxpr(self, args: list[Value], output: Array) -> None:
        output.tensor_value = ops.transpose(args[0], self.axis_1, self.axis_2)

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        offset = len(args[0].batch_dims)
        axes = list(range(-offset - len(args[0].shape), 0))
        axes[self.axis_1], axes[self.axis_2] = axes[self.axis_2], axes[self.axis_1]

        np_result = np.transpose(args[0].to_numpy(), axes)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [transpose(cotangent, self.axis_1, self.axis_2)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return transpose(tangents[0], self.axis_1, self.axis_2)


def transpose(arg: Array, axis_1: int = -2, axis_2: int = -1) -> Array:
    """Transpose array along two axes."""
    axis_1 = axis_1 if axis_1 < 0 else -len(arg.shape) + axis_1
    axis_2 = axis_2 if axis_2 < 0 else -len(arg.shape) + axis_2
    op = TransposeOp(axis_1, axis_2)
    return op.forward(arg)


class PermuteOp(ViewOperation):
    """Permute (reorder) the dimensions of a tensor according to given axes."""

    def __init__(self, axes: tuple[int, ...]):
        """Initialize permute operation.
        
        Args:
            axes: Tuple of axis indices specifying the new order.
                  Must be a permutation of range(ndim).
        """
        super().__init__(f"permute[axes={axes}]")
        self.axes = tuple(axes)

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compute output shape after permutation."""
        if len(input_shapes) != 1:
            raise ValueError(f"Permute operation requires 1 input shape, got {len(input_shapes)}")
        
        input_shape = input_shapes[0]
        
        # Validate axes - should now be all negative and same length as input
        if len(self.axes) != len(input_shape):
            raise ValueError(f"Number of axes {len(self.axes)} must match input dimensions {len(input_shape)}")
        
        # Convert to positive indices for validation
        positive_axes = [ax + len(input_shape) for ax in self.axes]
        if sorted(positive_axes) != list(range(len(input_shape))):
            raise ValueError(f"Axes {self.axes} must be a permutation of negative indices corresponding to {list(range(len(input_shape)))}")
        
        # Reorder dimensions according to axes (convert negative to positive for indexing)
        return tuple(input_shape[axis + len(input_shape)] for axis in self.axes)

    def maxpr(self, args: list[Value], output: Array) -> None:
        """Max computation: permute the tensor."""
        # Convert negative axes to positive for ops.transpose
        positive_axes = [ax + len(args[0].shape) for ax in self.axes]
        output.tensor_value = ops.transpose(args[0], tuple(positive_axes))

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        """Eager computation: permute using numpy."""
        # Handle batch dimensions properly like transpose does
        offset = len(args[0].batch_dims)
        
        # Convert our negative axes (relative to array shape) to work with full numpy array
        numpy_axes = []
        for ax in self.axes:
            # ax is negative relative to args[0].shape, convert to positive
            array_pos_ax = ax + len(args[0].shape)
            # Now convert to position in full numpy array (including batch dims)
            numpy_pos_ax = offset + array_pos_ax
            numpy_axes.append(numpy_pos_ax)
        
        # Prepend batch dimension indices (they stay in their original positions)
        full_axes = list(range(offset)) + numpy_axes
        
        np_result = np.transpose(args[0].to_numpy(), full_axes)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(self, primals: list[Array], cotangent: Array, output: Array) -> list[Array]:
        """VJP rule: reverse the permutation."""
        # Create inverse permutation for negative indices
        inv_axes = [0] * len(self.axes)
        for i, axis in enumerate(self.axes):
            # Convert negative axis to positive index for inverse mapping
            pos_axis = axis + len(self.axes)
            inv_axes[pos_axis] = i
        
        # Convert back to negative indices
        inv_axes_negative = [-len(self.axes) + ax for ax in inv_axes]
        
        return [permute(cotangent, tuple(inv_axes_negative))]

    def jvp_rule(self, primals: list[Array], tangents: list[Array], output: Array) -> Array:
        """JVP rule: apply same permutation to tangent."""
        return permute(tangents[0], self.axes)


def permute(input_array: Array, axes: tuple[int, ...]) -> Array:
    """Permute (reorder) the dimensions of a tensor.
    
    Args:
        input_array: Input tensor
        axes: Tuple specifying the new order of dimensions
        
    Returns:
        Tensor with reordered dimensions
        
    Example:
        >>> x = nb.ones((2, 3, 4))  # shape (2, 3, 4)
        >>> y = permute(x, (2, 0, 1))  # shape (4, 2, 3)
        >>> # Dimension 2 -> position 0, dimension 0 -> position 1, dimension 1 -> position 2
    """
    # always store axes to be fully negative 
    axes = tuple(-len(input_array.shape) + ax if ax >= 0 else ax for ax in axes)
    # but first we add oentailly missing axes which we treat as unpemruted
    axes_new = []
    for i in range(-len(input_array.shape), -len(axes)):
        axes_new.append(i)

    axes = axes_new + list(axes)  # prepend missing axes to the front

    op = PermuteOp(axes)
    return op.forward(input_array)


def move_axis_to_front(input_array: Array, axis: int) -> Array:
    """Move specified axis to the front (position 0), shifting others right.
    
    Args:
        input_array: Input tensor
        axis: Axis to move to front
        
    Returns:
        Tensor with specified axis moved to front
        
    Example:
        >>> x = nb.ones((2, 3, 4))  # shape (2, 3, 4)  
        >>> y = move_axis_to_front(x, 2)  # shape (4, 2, 3)
        >>> # axis 2 moved to front, others shifted: [2, 0, 1]
    """
    ndim = len(input_array.shape)
    
    # Normalize negative axis
    if axis < 0:
        axis = ndim + axis
        
    if axis < 0 or axis >= ndim:
        raise ValueError(f"Axis {axis} out of bounds for array of dimension {ndim}")
    
    # Generate permutation: [axis, 0, 1, ..., axis-1, axis+1, ..., ndim-1]
    axes = [axis] + [i for i in range(ndim) if i != axis]
    
    return permute(input_array, tuple(axes))


def move_axis_from_front(input_array: Array, target_axis: int) -> Array:
    """Move front axis (position 0) to specified target position.
    
    Args:
        input_array: Input tensor (assumes front axis is the one to move)
        target_axis: Target position for the front axis
        
    Returns:
        Tensor with front axis moved to target position
        
    Example:
        >>> x = nb.ones((4, 2, 3))  # front axis has size 4
        >>> y = move_axis_from_front(x, 2)  # shape (2, 3, 4)
        >>> # front axis moved to position 2: [1, 2, 0]
    """
    ndim = len(input_array.shape)
    
    # Normalize negative axis
    if target_axis < 0:
        target_axis = ndim + target_axis
        
    if target_axis < 0 or target_axis >= ndim:
        raise ValueError(f"Target axis {target_axis} out of bounds for array of dimension {ndim}")
    
    if target_axis == 0:
        return input_array  # Already at front
    
    # Generate permutation to move front to target_axis
    # [1, 2, ..., target_axis, 0, target_axis+1, ..., ndim-1]  
    axes = list(range(1, target_axis + 1)) + [0] + list(range(target_axis + 1, ndim))
    
    return permute(input_array, tuple(axes))

class ReshapeOp(ViewOperation):
    """Reshape operation."""

    def __init__(self, arg_shape: Shape, target_shape: Shape):
        super().__init__(f"reshape[new_sizes={target_shape}]")
        self.arg_shape = arg_shape
        self.target_shape = target_shape

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compatible signature."""
        if len(input_shapes) != 1:
            raise ValueError(
                f"Reshape operation requires 1 input shape, got {len(input_shapes)}"
            )
        return self.target_shape

    def forward(self, *args: Array) -> Array:
        """Override forward to validate size compatibility with compatible signature."""
        if len(args) != 1:
            raise ValueError(f"Reshape operation requires 1 argument, got {len(args)}")
        arg = args[0]

        old_size = np.prod(arg.shape) if arg.shape else 1
        new_size = np.prod(self.target_shape) if self.target_shape else 1
        if old_size != new_size:
            raise ValueError(
                f"Cannot reshape array of size {old_size} to shape {self.target_shape} of size {new_size}"
            )

        return super().forward(arg)

    def maxpr(self, args: list[Value], output: Array) -> None:
        output.tensor_value = ops.reshape(
            args[0], output.batch_dims + self.target_shape
        )

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.reshape(
            args[0].to_numpy(), output.batch_dims + self.target_shape
        )
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [reshape(cotangent, self.arg_shape)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return reshape(tangents[0], self.target_shape)


def reshape(arg: Array, shape: Shape) -> Array:
    """Reshape array to given shape."""
    op = ReshapeOp(arg.shape, shape)
    return op.forward(arg)


class BroadcastToOp(ViewOperation):
    """Broadcast array to target shape."""

    def __init__(self, target_shape: Shape):
        super().__init__(f"broadcast[shape={target_shape}]")
        self.target_shape = target_shape

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compatible signature."""
        if len(input_shapes) != 1:
            raise ValueError(
                f"Broadcast operation requires 1 input shape, got {len(input_shapes)}"
            )
        return self.target_shape

    def forward(self, *args: Array) -> Array:
        """Override forward to handle case where no broadcasting needed with compatible signature."""
        if len(args) != 1:
            raise ValueError(
                f"Broadcast operation requires 1 argument, got {len(args)}"
            )
        arg = args[0]
        if arg.shape == self.target_shape:
            return arg
        return super().forward(*args)

    @staticmethod
    def get_broadcasted_axes(input_shape: Shape, target_shape: Shape) -> list[int]:
        """Get axes that were broadcasted (for VJP)."""
        if len(input_shape) > len(target_shape):
            raise ValueError(
                f"Input shape {input_shape} cannot be broadcast to {target_shape}"
            )

        broadcasted_axes = []
        padded_input = (1,) * (len(target_shape) - len(input_shape)) + input_shape

        for i in range(len(target_shape)):
            if padded_input[i] == 1 and target_shape[i] > 1:
                # Return negative index to reference from the right side
                # This ensures we sum over the correct dimension
                broadcasted_axes.append(i - len(target_shape))
            elif padded_input[i] != target_shape[i] and padded_input[i] != 1:
                raise ValueError(f"Cannot broadcast {input_shape} to {target_shape}")

        return broadcasted_axes

    def maxpr(self, args: list[Value], output: Array) -> None:
        output.tensor_value = ops.broadcast_to(
            args[0], output.batch_dims + self.target_shape
        )

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.broadcast_to(
            args[0].to_numpy(), shape=output.batch_dims + self.target_shape
        )
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        broadcasted_axes = self.get_broadcasted_axes(
            primals[0].shape, self.target_shape
        )
        from .reduce import sum as sum_op  # Renamed to avoid shadowing built-in

        return [sum_op(cotangent, axes=broadcasted_axes, keep_dims=True)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return broadcast_to(tangents[0], self.target_shape)


def broadcast_to(arg: Array, shape: Shape) -> Array:
    """Broadcast array to target shape."""
    if arg.shape == shape:
        return arg
    for _ in range(len(shape) - len(arg.shape)):
        arg = unsqueeze(arg, [0])
    op = BroadcastToOp(shape)
    return op.forward(arg)


class BroadcastBatchDimsOp(ViewOperation):
    """Broadcast array to target batch_dims."""

    def __init__(self, target_batch_dims: Shape):
        super().__init__(f"broadcast_batch_dims[shape={target_batch_dims}]")
        self.target_batch_dims = target_batch_dims

    def compute_output_batch_dims(self, *input_batch_dimss: tuple) -> tuple:
        """Compatible signature."""
        if len(input_batch_dimss) != 1:
            raise ValueError(
                f"Broadcast operation requires 1 input batch_dims, got {len(input_batch_dimss)}"
            )
        return self.target_batch_dims

    def forward(self, *args: Array) -> Array:
        """Override forward to handle case where no broadcasting needed with compatible signature."""
        if len(args) != 1:
            raise ValueError(
                f"Broadcast operation requires 1 argument, got {len(args)}"
            )
        arg = args[0]
        if arg.batch_dims == self.target_batch_dims:
            return arg
        return super().forward(*args)

    @staticmethod
    def get_broadcasted_axes(
        input_batch_dims: Shape, target_batch_dims: Shape
    ) -> list[int]:
        """Get axes that were broadcasted (for VJP)."""
        if len(input_batch_dims) > len(target_batch_dims):
            raise ValueError(
                f"Input batch_dims {input_batch_dims} cannot be broadcast to {target_batch_dims}"
            )

        broadcasted_axes = []
        padded_input = (1,) * (
            len(target_batch_dims) - len(input_batch_dims)
        ) + input_batch_dims

        for i in range(len(target_batch_dims)):
            if padded_input[i] == 1 and i < len(target_batch_dims) - len(
                input_batch_dims
            ):
                # This dimension was added by padding (broadcasted from non-existent to size 1 or more)
                broadcasted_axes.append(i - len(target_batch_dims))
            elif padded_input[i] == 1 and target_batch_dims[i] > 1:
                # This dimension was broadcasted from size 1 to larger size
                broadcasted_axes.append(i - len(target_batch_dims))
            elif padded_input[i] != target_batch_dims[i] and padded_input[i] != 1:
                raise ValueError(
                    f"Cannot broadcast {input_batch_dims} to {target_batch_dims}"
                )

        return broadcasted_axes

    def maxpr(self, args: list[Value], output: Array) -> None:
        output.tensor_value = ops.broadcast_to(
            args[0], self.target_batch_dims + output.shape
        )

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.broadcast_to(
            args[0].to_numpy(), shape=self.target_batch_dims + output.shape
        )
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .reduce import sum_batch_dims

        broadcasted_axes = self.get_broadcasted_axes(
            primals[0].batch_dims, output.batch_dims
        )
        return [sum_batch_dims(cotangent, axes=broadcasted_axes, keep_dims=True)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return broadcast_batch_dims(tangents[0], self.target_batch_dims)


def broadcast_batch_dims(arg: Array, batch_dims: Shape) -> Array:
    """Broadcast array to target batch_dims."""
    if arg.batch_dims == batch_dims:
        return arg
    
    for _ in range(len(batch_dims) - len(arg.batch_dims)):
        arg = unsqueeze_batch_dims(arg, [0])

    op = BroadcastBatchDimsOp(batch_dims)
    return op.forward(arg)


class SqueezeOp(ViewOperation):
    """Squeeze operation to remove dimensions of size 1."""

    def __init__(self, axes: list[int] = None):
        super().__init__(f"squeeze[axes={axes}]")
        self.axes = sorted(axes) if axes is not None else []

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compatible signature."""
        if len(input_shapes) != 1:
            raise ValueError(
                f"Squeeze operation requires 1 input shape, got {len(input_shapes)}"
            )
        input_shape = input_shapes[0]

        new_shape = list(input_shape)
        for ax in self.axes:
            if ax < -len(new_shape) or ax >= len(new_shape):
                raise ValueError(f"Axis {ax} is out of bounds for squeeze operation")
            if input_shape[ax] == 1:
                new_shape[ax] = None
            else:
                raise ValueError(
                    f"Cannot squeeze axis {ax} of size {input_shape[ax]} (must be 1)"
                )

        new_shape = [dim for dim in new_shape if dim is not None]
        return tuple(new_shape)

    def forward(self, *args: Array) -> Array:
        """Override forward to handle case where no squeezing needed with compatible signature."""
        if len(args) != 1:
            raise ValueError(f"Squeeze operation requires 1 argument, got {len(args)}")
        return super().forward(*args)

    def maxpr(self, args: list[Value], output: Array) -> None:
        res_value = args[0]
        for i, ax in enumerate(self.axes):
            adjusted_axis = ax - i
            res_value = ops.squeeze(res_value, adjusted_axis)
        output.tensor_value = res_value

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        axis = tuple(self.axes) if self.axes else None
        np_result = np.squeeze(args[0].to_numpy(), axis=axis)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, _primals: list[Array], cotangent: Array, _output: Array
    ) -> list[Array]:
        return [unsqueeze(cotangent, self.axes)]

    def jvp_rule(
        self, _primals: list[Array], tangents: list[Array], _output: Array
    ) -> Array:
        return squeeze(tangents[0], self.axes)


def squeeze(arg: Array, axes: list[int] = None) -> Array:
    """Squeeze array by removing dimensions of size 1."""
    if axes is None:
        return arg
    axes = [ax if ax < 0 else -len(arg.shape) + ax for ax in axes]

    op = SqueezeOp(axes)
    res = op.forward(arg)

    return res


class UnsqueezeOp(ViewOperation):
    """Unsqueeze operation to add dimensions of size 1."""

    def __init__(self, axes: list[int] = None):
        super().__init__(f"unsqueeze[axes={axes}]")
        self.axes = sorted(axes) if axes is not None else []

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compatible signature."""
        if len(input_shapes) != 1:
            raise ValueError(
                f"Unsqueeze operation requires 1 input shape, got {len(input_shapes)}"
            )
        input_shape = input_shapes[0]

        new_shape = list(input_shape)
        for ax in self.axes:
            if ax < -len(new_shape) - 1:
                raise ValueError(f"Axis {ax} is out of bounds for unsqueeze operation")
            if ax + 1 <= -1:
                new_shape.insert(ax + 1, 1)
            else:
                new_shape.append(1)

        return tuple(new_shape)

    def forward(self, *args: Array) -> Array:
        """Override forward to handle case where no unsqueezing needed with compatible signature."""
        if len(args) != 1:
            raise ValueError(
                f"Unsqueeze operation requires 1 argument, got {len(args)}"
            )
        return super().forward(*args)

    def maxpr(self, args: list[Value], output: Array) -> None:
        res_value = args[0]
        for ax in self.axes:
            res_value = ops.unsqueeze(res_value, ax)
        output.tensor_value = res_value

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.expand_dims(args[0].to_numpy(), axis=self.axes)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [squeeze(cotangent, self.axes)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return unsqueeze(tangents[0], self.axes)


def unsqueeze(arg: Array, axes: list[int] = None) -> Array:
    """Unsqueeze array by adding dimensions of size 1."""
    if axes is None:
        return arg

    axes = [ax if ax < 0 else -len(arg.shape) - 1 + ax for ax in axes]
    op = UnsqueezeOp(axes)
    return op.forward(arg)


class ShallowCopyOp(ViewOperation):
    """Copy operation to create a new array with the same data."""

    def __init__(self):
        super().__init__("shallow_copy")

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compatible signature."""
        if len(input_shapes) != 1:
            raise ValueError(
                f"Copy operation requires 1 input shape, got {len(input_shapes)}"
            )
        return input_shapes[0]

    def maxpr(self, args: list[Value], output: Array) -> None:
        output.tensor_value = args[0]

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        output.impl = args[0].impl

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [cotangent]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return tangents[0]


def shallow_copy(arg: Array) -> Array:
    """Create a shallow copy of the array."""
    op = ShallowCopyOp()
    return op.forward(arg)


class ConcatenateOp(Operation):
    """Concatenate operation to join arrays along an existing axis."""

    def __init__(self, axis: int = 0):
        super().__init__(f"concatenate[axis={axis}]")
        self.axis = axis

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compute output shape for concatenate operation."""
        if len(input_shapes) == 0:
            raise ValueError("Concatenate operation requires at least 1 input")

        # All input shapes must be the same except along the concatenation axis
        first_shape = input_shapes[0]
        if not first_shape:
            raise ValueError("Cannot concatenate empty shapes")

        # Normalize axis
        axis = self.axis if self.axis >= 0 else len(first_shape) + self.axis
        if axis < 0 or axis >= len(first_shape):
            raise ValueError(
                f"Axis {self.axis} is out of bounds for array with {len(first_shape)} dimensions"
            )

        # Check that all shapes are compatible
        total_size_along_axis = 0
        for i, shape in enumerate(input_shapes):
            if len(shape) != len(first_shape):
                raise ValueError(
                    f"All inputs must have the same number of dimensions for concatenate operation. "
                    f"Input 0 has {len(first_shape)} dimensions, input {i} has {len(shape)} dimensions"
                )

            for j, (dim1, dim2) in enumerate(zip(first_shape, shape, strict=False)):
                if j != axis and dim1 != dim2:
                    raise ValueError(
                        f"All inputs must have the same shape except along axis {axis}. "
                        f"Input 0 has shape {first_shape}, input {i} has shape {shape}"
                    )

            total_size_along_axis += shape[axis]

        # Compute output shape
        output_shape = list(first_shape)
        output_shape[axis] = total_size_along_axis
        return tuple(output_shape)

    def maxpr(self, args: list[Value], output: Array) -> None:
        """MAX graph implementation using ops.concat."""
        # Normalize axis for MAX operations, considering batch_dims
        full_output_shape = output.batch_dims + output.shape
        axis = self.axis if self.axis >= 0 else len(output.shape) + self.axis

        # Adjust axis to account for batch_dims in the actual tensor
        axis_in_tensor = axis + len(output.batch_dims)
        output.tensor_value = ops.concat(args, axis=axis_in_tensor)

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        """Eager execution using NumPy concatenate."""
        import numpy as np

        numpy_arrays = [arg.to_numpy() for arg in args]
        # Normalize axis for NumPy operations, considering batch_dims
        axis = self.axis if self.axis >= 0 else len(output.shape) + self.axis

        # Adjust axis to account for batch_dims in the actual tensor
        axis_in_tensor = axis + len(output.batch_dims)
        result = np.concatenate(numpy_arrays, axis=axis_in_tensor)
        output.impl = Tensor.from_numpy(result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        """Vector-Jacobian product rule for concatenate operation.

        The VJP of concatenate is slicing the cotangent back into pieces.
        """
        # Normalize axis
        axis = self.axis if self.axis >= 0 else len(cotangent.shape) + self.axis

        # Split the cotangent along the concatenated axis
        result = []
        start_idx = 0

        for primal in primals:
            size_along_axis = primal.shape[axis]
            end_idx = start_idx + size_along_axis

            # Create slice that selects this input's portion along the concatenated axis
            slices = [slice(None)] * len(cotangent.shape)
            slices[axis] = slice(start_idx, end_idx)

            # Slice the cotangent
            sliced = array_slice(cotangent, slices)
            result.append(sliced)

            start_idx = end_idx

        return result

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        """Jacobian-vector product rule for concatenate operation.

        The JVP of concatenate is concatenating the tangents along the same axis.
        """
        # Use the ConcatenateOp directly to avoid circular import
        op = ConcatenateOp(axis=self.axis)
        return op.forward(*tangents)

    def forward(self, *args: Array) -> Array:
        """Forward pass for concatenate operation with multiple inputs."""
        if len(args) == 0:
            raise ValueError("Concatenate operation requires at least 1 argument")

        # Validate inputs have compatible properties
        first_arg = args[0]
        for i, arg in enumerate(args[1:], 1):
            if arg.dtype != first_arg.dtype:
                raise ValueError(
                    f"All inputs must have the same dtype. Got {arg.dtype} vs {first_arg.dtype}"
                )
            if arg.device != first_arg.device:
                raise ValueError(
                    f"All inputs must be on the same device. Got {arg.device} vs {first_arg.device}"
                )

        # Compute output properties
        input_shapes = [arg.shape for arg in args]
        output_shape = self.compute_output_shape(*input_shapes)

        # All inputs should have the same batch_dims
        output_batch_dims = first_arg.batch_dims
        for i, arg in enumerate(args[1:], 1):
            if arg.batch_dims != output_batch_dims:
                raise ValueError(
                    f"All inputs must have the same batch_dims for concatenate operation. "
                    f"Input 0 has batch_dims {output_batch_dims}, input {i} has batch_dims {arg.batch_dims}"
                )

        # Create result array
        res = Array(
            shape=output_shape,
            dtype=first_arg.dtype,
            device=first_arg.device,
            materialize=False,
            name=self.name,
            batch_dims=output_batch_dims,
        )

        # Set up computation
        res.set_maxpr(self.maxpr)
        res.add_arguments(*args)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule

        # Execute eager computation if needed
        if not res.stage_realization:
            self.eagerxpr(list(args), res)

        return res


def concatenate(args: list[Array], axis: int = 0) -> Array:
    """Concatenate arrays along an existing axis.

    Args:
        args: List of arrays to concatenate
        axis: Axis along which to concatenate arrays (default: 0)

    Returns:
        Concatenated array
    """
    if not args:
        raise ValueError("Concatenate operation requires at least one array")

    op = ConcatenateOp(axis)
    return op.forward(*args)


class ArraySliceOp(ViewOperation):
    """Array slicing operation."""

    def __init__(self, slices: list[slice]):
        # Convert slices to a more manageable format
        slice_strs = []
        for s in slices:
            start = s.start if s.start is not None else ""
            stop = s.stop if s.stop is not None else ""
            step = s.step if s.step is not None else ""
            if step and step != 1:
                slice_strs.append(f"{start}:{stop}:{step}")
            else:
                slice_strs.append(f"{start}:{stop}")

        super().__init__(f"array_slice[{','.join(slice_strs)}]")
        self.slices = slices

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compute output shape for array slice operation."""
        if len(input_shapes) != 1:
            raise ValueError(
                f"Array slice operation requires 1 input shape, got {len(input_shapes)}"
            )

        input_shape = input_shapes[0]
        output_shape = []

        # Process each dimension
        for i, dim_size in enumerate(input_shape):
            if i < len(self.slices):
                s = self.slices[i]
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else dim_size
                step = s.step if s.step is not None else 1

                # Handle negative indices
                if start < 0:
                    start = max(0, dim_size + start)
                if stop < 0:
                    stop = max(0, dim_size + stop)

                # Clamp to valid range
                start = max(0, min(start, dim_size))
                stop = max(start, min(stop, dim_size))

                # Calculate output size for this dimension
                if step > 0:
                    output_size = max(0, (stop - start + step - 1) // step)
                else:
                    raise ValueError("Negative step not supported")

                output_shape.append(output_size)
            else:
                # No slice for this dimension, keep original size
                output_shape.append(dim_size)

        return tuple(output_shape)

    def maxpr(self, args: list[Value], output: Array) -> None:
        """MAX graph implementation using ops.slice_tensor."""
        # Build slice indices for MAX ops.slice_tensor
        # Need to account for batch_dims - slicing only applies to shape dimensions
        slice_indices = []

        # Add full slices for batch dimensions
        for _ in range(len(output.batch_dims)):
            slice_indices.append(slice(None))

        # Add the actual slices for shape dimensions
        for i in range(len(self.slices)):
            s = self.slices[i]
            slice_indices.append(slice(s.start, s.stop, s.step))

        # Use ops.slice_tensor which follows NumPy slicing semantics
        output.tensor_value = ops.slice_tensor(args[0], slice_indices)

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        """Eager execution using NumPy slicing."""
        input_array = args[0].to_numpy()

        # Build numpy slice tuple
        # Need to account for batch_dims - slicing only applies to shape dimensions
        numpy_slices = []

        # Add full slices for batch dimensions
        for _ in range(len(args[0].batch_dims)):
            numpy_slices.append(slice(None))

        # Add the actual slices for shape dimensions
        for i in range(len(args[0].shape)):
            if i < len(self.slices):
                numpy_slices.append(self.slices[i])
            else:
                numpy_slices.append(slice(None))  # Full slice for remaining dimensions

        result = input_array[tuple(numpy_slices)]
        output.impl = Tensor.from_numpy(result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        """Vector-Jacobian product rule for array slice."""
        return [pad(cotangent, self.slices, primals[0].shape)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        """Jacobian-vector product rule for array slice."""
        return array_slice(tangents[0], self.slices)


def array_slice(arg: Array, slices: list[slice]) -> Array:
    """Slice an array along specified dimensions.

    Args:
        arg: Input array to slice
        slices: List of slice objects defining the slicing for each dimension

    Returns:
        Sliced array
    """
    op = ArraySliceOp(slices)
    return op.forward(arg)


def split(arg: Array, sizes: list[int], axis: int = 0) -> list[Array]:
    """Split an array into multiple sub-arrays along a specified axis.

    Args:
        arg: Input array to split
        sizes: List of sizes for each split along the specified axis
        axis: Axis along which to split the array (default: 0)
    Returns:
        List of sub-arrays resulting from the split
    """
    if not sizes:
        raise ValueError("Sizes list must not be empty")

    if axis < 0:
        axis += len(arg.shape)

    if axis < 0 or axis >= len(arg.shape):
        raise ValueError(
            f"Axis {axis} is out of bounds for array with {len(arg.shape)} dimensions"
        )

    # Compute the total size along the specified axis
    total_size = sum(sizes)
    if total_size != arg.shape[axis]:
        raise ValueError(
            f"Total size {total_size} along axis {axis} does not match input shape {arg.shape[axis]}"
        )

    # Create slices for each split
    slices = []
    idx = 0
    for size in sizes:
        slices.append(slice(idx, idx + size))
        idx += size

    # Create the result arrays
    results = []
    for s in slices:
        slice_obj = [slice(None)] * len(arg.shape)  # Full slice for all dimensions
        slice_obj[axis] = s  # Set the slice for the specified axis
        results.append(array_slice(arg, slice_obj))

    return results


class PadOp(Operation):
    """Inverse slice operation - places a smaller array into a larger zero-filled array."""

    def __init__(self, slices: list[slice], target_shape: Shape):
        # Convert slices to string representation for name
        slice_strs = []
        for s in slices:
            start = s.start if s.start is not None else ""
            stop = s.stop if s.stop is not None else ""
            step = s.step if s.step is not None else ""
            if step and step != 1:
                slice_strs.append(f"{start}:{stop}:{step}")
            else:
                slice_strs.append(f"{start}:{stop}")

        super().__init__(f"pad[{','.join(slice_strs)}]")
        self.slices = slices
        self.target_shape = target_shape

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compute output shape for inverse slice operation."""
        if len(input_shapes) != 1:
            raise ValueError(
                f"Inverse slice operation requires 1 input shape, got {len(input_shapes)}"
            )

        # Validate that applying slices to target_shape would yield input_shape
        input_shape = input_shapes[0]

        # Simulate slicing target_shape with self.slices to verify consistency
        expected_shape = []
        for i, dim_size in enumerate(self.target_shape):
            if i < len(self.slices):
                s = self.slices[i]
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else dim_size
                step = s.step if s.step is not None else 1

                if step != 1:
                    raise NotImplementedError(
                        "Stepped slicing not yet supported in pad"
                    )

                # Handle negative indices
                if start < 0:
                    start = max(0, dim_size + start)
                if stop < 0:
                    stop = max(0, dim_size + stop)

                # Clamp to valid range
                start = max(0, min(start, dim_size))
                stop = max(start, min(stop, dim_size))

                # Calculate output size for this dimension
                output_size = stop - start
                expected_shape.append(output_size)
            else:
                # No slice for this dimension, keep original size
                expected_shape.append(dim_size)

        expected_shape = tuple(expected_shape)
        if expected_shape != input_shape:
            raise ValueError(
                f"Slicing target_shape {self.target_shape} with {self.slices} "
                f"would produce shape {expected_shape}, but input has shape {input_shape}"
            )

        return self.target_shape

    def maxpr(self, args: list[Value], output: Array) -> None:
        """MAX graph implementation using scatter operation."""
        raise NotImplementedError(
            "MAX graph implementation for pad is not yet implemented."
        )
    

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        """Eager execution using NumPy."""
        small_array = args[0]

        # Create zero-filled target array
        target_shape = output.batch_dims + output.shape
        result_np = np.zeros(target_shape, dtype=small_array.to_numpy().dtype)

        # Build slice indices (accounting for batch_dims)
        slice_indices = []

        # Add full slices for batch dimensions
        for _ in range(len(output.batch_dims)):
            slice_indices.append(slice(None))

        # Add the actual slices for shape dimensions
        slice_indices.extend(self.slices)

        # Add full slices for any remaining dimensions
        for i in range(len(self.slices), len(output.shape)):
            slice_indices.append(slice(None))

        # Place small array into the target location
        result_np[tuple(slice_indices)] = small_array.to_numpy()

        output.impl = Tensor.from_numpy(result_np)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        """VJP rule: slice the cotangent back to original size."""
        # The VJP of pad is just a regular slice!
        from nabla.ops.view import array_slice

        return [array_slice(cotangent, self.slices)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        """JVP rule: apply pad to tangents."""
        return pad(tangents[0], self.slices, self.target_shape)

    def forward(self, *args: Array) -> Array:
        """Forward pass for inverse slice operation."""
        if len(args) != 1:
            raise ValueError(
                f"Inverse slice operation requires 1 argument, got {len(args)}"
            )

        input_array = args[0]

        # Compute output properties
        output_shape = self.compute_output_shape(input_array.shape)

        # Create result array
        res = Array(
            shape=output_shape,
            dtype=input_array.dtype,
            device=input_array.device,
            materialize=False,
            name=self.name,
            batch_dims=input_array.batch_dims,
        )

        # Set up computation
        res.set_maxpr(self.maxpr)
        res.add_arguments(input_array)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule

        # Execute eager computation if needed
        if not res.stage_realization:
            self.eagerxpr([input_array], res)

        return res


def pad(arg: Array, slices: list[slice], target_shape: Shape) -> Array:
    """Place a smaller array into a larger zero-filled array at the location specified by slices.

    This is the inverse operation of array slicing - given slices, a small array, and target shape,
    it creates a larger array where the small array is placed at the sliced location
    and everything else is zero.

    Args:
        arg: Input array (the smaller array to be placed)
        slices: List of slice objects defining where to place the array
        target_shape: The shape of the output array

    Returns:
        Larger array with input placed at sliced location, zeros elsewhere
    """
    op = PadOp(slices, target_shape)
    return op.forward(arg)




class SqueezeBatchDimsOp(ViewOperation):
    """Squeeze operation to remove batch dimensions of size 1."""

    def __init__(self, axes: list[int] = None):
        super().__init__(f"squeeze_batch_dims[axes={axes}]")
        self.axes = sorted(axes) if axes is not None else []

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Shape stays the same for batch dimension operations."""
        if len(input_shapes) != 1:
            raise ValueError(
                f"Squeeze batch dims operation requires 1 input shape, got {len(input_shapes)}"
            )
        return input_shapes[0]

    def compute_output_batch_dims(self, *input_batch_dimss: tuple) -> tuple:
        """Compute output batch_dims for squeeze operation."""
        if len(input_batch_dimss) != 1:
            raise ValueError(
                f"Squeeze batch dims operation requires 1 input batch_dims, got {len(input_batch_dimss)}"
            )
        input_batch_dims = input_batch_dimss[0]

        new_batch_dims = list(input_batch_dims)
        for ax in self.axes:
            if ax < -len(new_batch_dims) or ax >= len(new_batch_dims):
                raise ValueError(f"Axis {ax} is out of bounds for squeeze batch dims operation")
            if input_batch_dims[ax] == 1:
                new_batch_dims[ax] = None
            else:
                raise ValueError(
                    f"Cannot squeeze batch axis {ax} of size {input_batch_dims[ax]} (must be 1)"
                )

        new_batch_dims = [dim for dim in new_batch_dims if dim is not None]
        return tuple(new_batch_dims)

    def forward(self, *args: Array) -> Array:
        """Override forward to handle case where no squeezing needed."""
        if len(args) != 1:
            raise ValueError(f"Squeeze batch dims operation requires 1 argument, got {len(args)}")
        return super().forward(*args)

    def maxpr(self, args: list[Value], output: Array) -> None:
        """MAX graph implementation using ops.squeeze."""
        axes = [ax - len(args[0].shape) for ax in self.axes]
        output.tensor_value = ops.squeeze(args[0], axes=axes)

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        """Eager execution using NumPy squeeze."""
        axes = [ax - len(args[0].shape) for ax in self.axes]
        np_result = np.squeeze(args[0].to_numpy(), axis=tuple(axes))
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, _primals: list[Array], cotangent: Array, _output: Array
    ) -> list[Array]:
        """VJP rule: unsqueeze the cotangent back to original batch dimensions."""
        return [unsqueeze_batch_dims(cotangent, self.axes)]

    def jvp_rule(
        self, _primals: list[Array], tangents: list[Array], _output: Array
    ) -> Array:
        """JVP rule: apply squeeze to tangents."""
        return squeeze_batch_dims(tangents[0], self.axes)


def squeeze_batch_dims(arg: Array, axes: list[int] = None) -> Array:
    """Squeeze array by removing batch dimensions of size 1.
    
    Args:
        arg: Input array
        axes: List of batch dimension axes to squeeze. If None, returns array unchanged.
        
    Returns:
        Array with specified batch dimensions of size 1 removed
    """
    if axes is None:
        return arg
    # Convert to negative indices for consistency with batch dimension handling
    axes = [ax if ax < 0 else -len(arg.batch_dims) + ax for ax in axes]
    op = SqueezeBatchDimsOp(axes)
    return op.forward(arg)


class UnsqueezeBatchDimsOp(ViewOperation):
    """Unsqueeze operation to add batch dimensions of size 1."""

    def __init__(self, axes: list[int] = None):
        super().__init__(f"unsqueeze_batch_dims[axes={axes}]")
        self.axes = sorted(axes) if axes is not None else []

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Shape stays the same for batch dimension operations."""
        if len(input_shapes) != 1:
            raise ValueError(
                f"Unsqueeze batch dims operation requires 1 input shape, got {len(input_shapes)}"
            )
        return input_shapes[0]

    def compute_output_batch_dims(self, *input_batch_dimss: tuple) -> tuple:
        """Compute output batch_dims for unsqueeze operation."""
        if len(input_batch_dimss) != 1:
            raise ValueError(
                f"Unsqueeze batch dims operation requires 1 input batch_dims, got {len(input_batch_dimss)}"
            )
        input_batch_dims = input_batch_dimss[0]

        new_batch_dims = list(input_batch_dims)
        for ax in self.axes:
            if ax < -len(new_batch_dims) - 1:
                raise ValueError(f"Axis {ax} is out of bounds for unsqueeze batch dims operation")
            if ax + 1 <= -1:
                new_batch_dims.insert(ax + 1, 1)
            else:
                new_batch_dims.append(1)

        return tuple(new_batch_dims)

    def forward(self, *args: Array) -> Array:
        """Override forward to handle case where no unsqueezing needed."""
        if len(args) != 1:
            raise ValueError(
                f"Unsqueeze batch dims operation requires 1 argument, got {len(args)}"
            )
        return super().forward(*args)

    def maxpr(self, args: list[Value], output: Array) -> None:
        """MAX graph implementation using ops.unsqueeze."""
        axes = [ax - len(args[0].shape) for ax in self.axes] if self.axes else None
        output.tensor_value = ops.unsqueeze(args[0], axes=axes)

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        """Eager execution using NumPy expand_dims."""
        axes = [ax - len(args[0].shape) for ax in self.axes] if self.axes else None
        np_result = np.expand_dims(args[0].to_numpy(), axis=axes)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        """VJP rule: squeeze the cotangent back to original batch dimensions."""
        return [squeeze_batch_dims(cotangent, self.axes)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        """JVP rule: apply unsqueeze to tangents."""
        return unsqueeze_batch_dims(tangents[0], self.axes)


def unsqueeze_batch_dims(arg: Array, axes: list[int] = None) -> Array:
    """Unsqueeze array by adding batch dimensions of size 1.
    
    Args:
        arg: Input array
        axes: List of positions where to insert batch dimensions of size 1. 
              If None, returns array unchanged.
        
    Returns:
        Array with batch dimensions of size 1 added at specified positions
    """
    if axes is None:
        return arg

    # Convert to negative indices for consistency with batch dimension handling
    axes = [ax if ax < 0 else -len(arg.batch_dims) - 1 + ax for ax in axes]
    
    op = UnsqueezeBatchDimsOp(axes)
    return op.forward(arg)