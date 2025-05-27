"""Linear algebra operations."""

import numpy as np
from max.driver import Tensor
from max.graph import Value, ops

from ..core.array import Array
from ..utils.broadcasting import get_broadcasted_shape
from .operation import BinaryOperation


class MatMulOp(BinaryOperation):
    """Matrix multiplication operation with batching support."""

    def __init__(self):
        super().__init__("matmul")

    def forward(self, *args: Array) -> Array:
        """Forward pass for matrix multiplication with compatible signature."""
        if len(args) != 2:
            raise ValueError(
                f"Matrix multiplication requires 2 arguments, got {len(args)}"
            )
        arg1, arg2 = args[0], args[1]

        # Validate inputs
        self._validate_inputs(arg1, arg2)

        # Compute output shape
        output_shape = self.compute_output_shape(arg1.shape, arg2.shape)

        res = Array(
            shape=output_shape,
            dtype=arg1.dtype,
            device=arg1.device,
            materialize=False,
            name=self.name,
        )

        res.set_maxpr(self.maxpr)
        res.add_argument(arg1)
        res.add_argument(arg2)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule

        from .base import EAGERMODE

        if EAGERMODE:
            self.eagerxpr([arg1, arg2], res)

        return res

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compute output shape for matrix multiplication with compatible signature."""
        if len(input_shapes) != 2:
            raise ValueError(
                f"Matrix multiplication requires 2 input shapes, got {len(input_shapes)}"
            )
        shape1, shape2 = input_shapes[0], input_shapes[1]

        if shape1[-1] != shape2[-2]:
            raise ValueError(
                f"Shapes {shape1} and {shape2} are not compatible for matrix multiplication"
            )

        return get_broadcasted_shape(
            shape1,
            shape2,
            ignore_axes=[-2, -1],
            replace_ignored_dims=[shape1[-2], shape2[-1]],
        )

    def _validate_inputs(self, arg1: Array, arg2: Array) -> None:
        """Validate matrix multiplication inputs."""
        if not isinstance(arg1, Array) or not isinstance(arg2, Array):
            raise TypeError("Both arguments must be Array instances")
        if arg1.dtype != arg2.dtype:
            raise ValueError(f"Dtypes {arg1.dtype} and {arg2.dtype} are incompatible")
        if arg1.device != arg2.device:
            raise ValueError(
                f"Devices {arg1.device} and {arg2.device} are incompatible"
            )
        if arg1.shape[-1] != arg2.shape[-2]:
            raise ValueError(
                f"Shapes {arg1.shape} and {arg2.shape} are not compatible for matrix multiplication"
            )

    def maxpr(self, args: list[Value], output: Array) -> None:
        x_val, y_val = args[0], args[1]
        x_shape_orig, y_shape_orig = x_val.shape, y_val.shape

        # Validate K-dimension compatibility
        if x_shape_orig[-1] != y_shape_orig[-2]:
            raise ValueError(
                f"Shapes {x_shape_orig} and {y_shape_orig} are not compatible for matrix multiplication "
                f"(K-dimension mismatch: {x_shape_orig[-1]} vs {y_shape_orig[-2]})"
            )

        # Compute broadcasted output shape
        output_shape_tuple = get_broadcasted_shape(
            x_shape_orig,
            y_shape_orig,
            ignore_axes=[-2, -1],
            replace_ignored_dims=[x_shape_orig[-2], y_shape_orig[-1]],
        )

        # Extract dimensions
        m_dim = output_shape_tuple[-2]
        n_dim = output_shape_tuple[-1]
        k_dim = x_shape_orig[-1]
        output_batch_shape = output_shape_tuple[:-2]

        # Broadcast inputs to align with output batch shape
        x_target_shape = output_batch_shape + (m_dim, k_dim)
        y_target_shape = output_batch_shape + (k_dim, n_dim)

        x_val_b = (
            ops.broadcast_to(x_val, x_target_shape)
            if x_val.shape != x_target_shape
            else x_val
        )
        y_val_b = (
            ops.broadcast_to(y_val, y_target_shape)
            if y_val.shape != y_target_shape
            else y_val
        )

        # Reshape for 4D matmul if needed
        num_batch_dims = len(output_batch_shape)

        if num_batch_dims == 0:
            shape_for_x = (1, 1, m_dim, k_dim)
            shape_for_y = (1, 1, k_dim, n_dim)
        elif num_batch_dims == 1:
            b0 = int(output_batch_shape[0])
            shape_for_x = (b0, 1, m_dim, k_dim)
            shape_for_y = (b0, 1, k_dim, n_dim)
        elif num_batch_dims == 2:
            shape_for_x = x_val_b.shape
            shape_for_y = y_val_b.shape
        else:
            # Flatten batch dimensions for ops.matmul
            b_eff_1 = int(np.prod(output_batch_shape[:-1]))
            b_eff_2 = int(output_batch_shape[-1])
            shape_for_x = (b_eff_1, b_eff_2, m_dim, k_dim)
            shape_for_y = (b_eff_1, b_eff_2, k_dim, n_dim)

        x_for_matmul = (
            ops.reshape(x_val_b, shape_for_x)
            if x_val_b.shape != shape_for_x
            else x_val_b
        )
        y_for_matmul = (
            ops.reshape(y_val_b, shape_for_y)
            if y_val_b.shape != shape_for_y
            else y_val_b
        )

        # Perform 4D matrix multiplication
        matmul_result = ops.matmul(x_for_matmul, y_for_matmul)

        # Reshape back to final output shape if needed
        output.tensor_value = (
            ops.reshape(matmul_result, output_shape_tuple)
            if matmul_result.shape != output_shape_tuple
            else matmul_result
        )

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        arg0_numpy = args[0].get_numpy()
        arg1_numpy = args[1].get_numpy()
        np_result = np.matmul(arg0_numpy, arg1_numpy)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        x, y = primals
        # Import transpose function here to avoid circular imports
        from .view import transpose

        return [matmul(cotangent, transpose(y)), matmul(transpose(x), cotangent)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        x, y = primals
        tx, ty = tangents
        # Import add here to avoid circular imports
        from .binary import add

        return add(matmul(x, ty), matmul(tx, y))


# Global operation instance for efficiency
_matmul_op = MatMulOp()


def matmul(arg0: Array, arg1: Array) -> Array:
    """Matrix multiplication with broadcasting support."""
    return _matmul_op.forward(arg0, arg1)
