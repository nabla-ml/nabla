# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or beautiful, software
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

        # Normalize negative axes
        axis_1 = self.axis_1 if self.axis_1 >= 0 else len(arg_shape) + self.axis_1
        axis_2 = self.axis_2 if self.axis_2 >= 0 else len(arg_shape) + self.axis_2

        if axis_1 < 0 or axis_1 >= len(arg_shape):
            raise ValueError(f"axis_1 {axis_1} is out of bounds for shape {arg_shape}")
        if axis_2 < 0 or axis_2 >= len(arg_shape):
            raise ValueError(f"axis_2 {axis_2} is out of bounds for shape {arg_shape}")

        # Create new shape with axes swapped
        new_shape = list(arg_shape)
        new_shape[axis_1], new_shape[axis_2] = new_shape[axis_2], new_shape[axis_1]
        return tuple(new_shape)

    def maxpr(self, args: list[Value], output: Array) -> None:
        output.tensor_value = ops.transpose(args[0], self.axis_1, self.axis_2)

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        # Create permutation axes
        axes = list(range(len(args[0].shape)))
        axes[self.axis_1], axes[self.axis_2] = axes[self.axis_2], axes[self.axis_1]

        np_result = np.transpose(args[0].get_numpy(), axes)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [transpose(cotangent, self.axis_1, self.axis_2)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return transpose(tangents[0], self.axis_1, self.axis_2)


# Global operation instances for efficiency
# _transpose_op = TransposeOp()


def transpose(arg: Array, axis_1: int = -2, axis_2: int = -1) -> Array:
    """Transpose array along two axes."""
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

        # Validate that total size remains the same
        old_size = np.prod(arg.shape) if arg.shape else 1
        new_size = np.prod(self.target_shape) if self.target_shape else 1
        if old_size != new_size:
            raise ValueError(
                f"Cannot reshape array of size {old_size} to shape {self.target_shape} of size {new_size}"
            )

        return super().forward(arg)

    def maxpr(self, args: list[Value], output: Array) -> None:
        output.tensor_value = ops.reshape(args[0], self.target_shape)

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.reshape(args[0].get_numpy(), self.target_shape)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [reshape(cotangent, self.arg_shape)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return reshape(tangents[0], self.target_shape)


# Global operation instances for efficiency
# Note: ReshapeOp requires target_shape parameter, so no global instance


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
        # Pad input shape with leading 1s
        padded_input = (1,) * (len(target_shape) - len(input_shape)) + input_shape

        for i in range(len(target_shape)):
            if padded_input[i] == 1 and target_shape[i] > 1:
                broadcasted_axes.append(i)
            elif padded_input[i] != target_shape[i] and padded_input[i] != 1:
                raise ValueError(f"Cannot broadcast {input_shape} to {target_shape}")

        return broadcasted_axes

    def maxpr(self, args: list[Value], output: Array) -> None:
        output.tensor_value = ops.broadcast_to(args[0], self.target_shape)

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.broadcast_to(args[0].get_numpy(), shape=self.target_shape)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        broadcasted_axes = self.get_broadcasted_axes(
            primals[0].shape, self.target_shape
        )

        # Import reduce_sum here to avoid circular imports
        from .reduce import reduce_sum

        return [reduce_sum(cotangent, axes=broadcasted_axes)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return broadcast_to(tangents[0], self.target_shape)


def broadcast_to(arg: Array, shape: Shape) -> Array:
    """Broadcast array to target shape."""
    op = BroadcastToOp(shape)
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

        # Normalize axes
        normalized_axes = [ax if ax >= 0 else len(input_shape) + ax for ax in self.axes]

        # Remove dimensions of size 1
        new_shape = [
            dim
            for i, dim in enumerate(input_shape)
            if i not in normalized_axes or dim > 1
        ]
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
        np_result = np.squeeze(args[0].get_numpy(), axis=axis)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, _primals: list[Array], cotangent: Array, _output: Array
    ) -> list[Array]:
        return [unsqueeze(cotangent, self.axes)]

    def jvp_rule(
        self, _primals: list[Array], tangents: list[Array], _output: Array
    ) -> Array:
        return squeeze(tangents[0], self.axes)
        return squeeze(tangents[0], self.axes)


def squeeze(arg: Array, axes: list[int] = None) -> Array:
    """Squeeze array by removing dimensions of size 1."""
    if axes is None:
        return arg
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

        # Normalize axes
        normalized_axes = [ax if ax >= 0 else len(input_shape) + ax for ax in self.axes]

        # Add dimensions of size 1 at specified axes
        new_shape = list(input_shape)
        for ax in sorted(normalized_axes):
            new_shape.insert(ax, 1)
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
        for i, ax in enumerate(self.axes):
            adjusted_axis = ax - i
            res_value = ops.unsqueeze(res_value, adjusted_axis)
        output.tensor_value = res_value

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.expand_dims(args[0].get_numpy(), axis=self.axes)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        # Unsqueeze does not change the size, so we can just return the cotangent
        return [squeeze(cotangent, self.axes)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        # Unsqueeze does not change the size, so we can just return the tangent
        return unsqueeze(tangents[0], self.axes)


def unsqueeze(arg: Array, axes: list[int] = None) -> Array:
    """Unsqueeze array by adding dimensions of size 1."""
    if axes is None:
        return arg  # No axes specified, return original array
    op = UnsqueezeOp(axes)
    return op.forward(arg)
