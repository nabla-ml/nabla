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
from .operation import ViewOperation


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
        super().__init__(f"broadcast_in_dim[shape={target_shape}]")
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
                broadcasted_axes.append(i)
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
        from .reduce import reduce_sum

        return [reduce_sum(cotangent, axes=broadcasted_axes)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return broadcast_to(tangents[0], self.target_shape)


def broadcast_to(arg: Array, shape: Shape) -> Array:
    """Broadcast array to target shape."""
    if len(arg.shape) < len(shape):
        new_shape = (1,) * (len(shape) - len(arg.shape)) + arg.shape
        arg = reshape(arg, new_shape)
    op = BroadcastToOp(shape)
    return op.forward(arg)


class BroadcastBatchDimsOp(ViewOperation):
    """Broadcast array to target batch_dims."""

    def __init__(self, target_batch_dims: Shape):
        super().__init__(f"broadcast_in_dim[batch_dims={target_batch_dims}]")
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
            if padded_input[i] == 1 and target_batch_dims[i] > 1:
                broadcasted_axes.append(i)
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
            primals[0].batch_dims, self.target_batch_dims
        )
        return [sum_batch_dims(cotangent, axes=broadcasted_axes)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return broadcast_batch_dims(tangents[0], self.target_batch_dims)


def broadcast_batch_dims(arg: Array, batch_dims: Shape) -> Array:
    """Broadcast array to target batch_dims."""
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
    return op.forward(arg)


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
        super().__init__("copy")

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