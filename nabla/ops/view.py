"""View and shape manipulation operations."""

import numpy as np
from max.driver import Tensor
from max.graph import Value, ops

from ..core.array import Array, Shape
from .operation import ViewOperation


class TransposeOp(ViewOperation):
    """Matrix/tensor transpose operation."""

    def __init__(self, axis_1: int = -2, axis_2: int = -1):
        super().__init__("transpose")  # Corrected super call
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
        super().__init__("reshape")  # Corrected super call
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
        super().__init__("broadcast_to")  # Corrected super call
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
