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

"""
Linear algebra operations for Nabla tensors.

This module provides fundamental linear algebra functions, starting with matrix
multiplication (`matmul`). The operations support broadcasting over batch
dimensions and are equipped with differentiation rules for use in gradient-based
optimization.
"""

import numpy as np
from max.graph import TensorValue, ops

from ..core.tensor import Tensor
from ..utils.shape_utils import get_broadcasted_shape
from .operation import Operation, BinaryOperation

# Public API
__all__ = ["matmul", "conv2d", "conv2d_transpose"]


class MatMulOp(BinaryOperation):
    """
    Implements the matrix multiplication operation, supporting batched inputs.

    This operation class encapsulates the logic for matrix multiplication,
    including shape computation, validation, execution in both eager and graph
    modes, and the rules for automatic differentiation (VJP and JVP).
    """

    def __init__(self):
        """Initializes the MatMulOp."""
        super().__init__("dot_general")

    def forward(self, *args: Tensor) -> Tensor:
        """
        Executes the forward pass for matrix multiplication.

        This method handles the core logic, including promoting 1D tensors to 2D
        for the multiplication, performing broadcasting, and then reshaping the
        output back to the expected rank.

        Parameters
        ----------
        *args : Tensor
            A tuple containing the two input tensors to be multiplied, `(arg1, arg2)`.

        Returns
        -------
        Tensor
            The result of the matrix multiplication.
        """
        if len(args) != 2:
            raise ValueError(f"Binary operation requires 2 arguments, got {len(args)}")

        # Move tensors to best device
        from .operation import move_to_best_device

        args = move_to_best_device(*args)
        arg1, arg2 = args[0], args[1]

        from ..ops.view import broadcast_batch_dims, broadcast_to, reshape

        arg1_has_rank_1 = len(arg1.shape) == 1
        arg2_has_rank_1 = len(arg2.shape) == 1

        # Promote 1D tensors to 2D for matmul computation
        if arg1_has_rank_1:
            arg1 = reshape(arg1, (1, arg1.shape[0]))

        if arg2_has_rank_1:
            arg2 = reshape(arg2, (arg2.shape[0], 1))

        self._validate_inputs(arg1, arg2)

        output_shape = self.compute_output_shape(arg1.shape, arg2.shape)
        output_batch_dims = self.compute_output_batch_dims(
            arg1.batch_dims, arg2.batch_dims
        )
        output_dtype = self.compute_output_dtype(arg1, arg2)
        if arg1.traced or arg1.requires_grad:
            arg1 = broadcast_to(arg1, output_shape[:-2] + arg1.shape[-2:])
            arg1 = broadcast_batch_dims(arg1, output_batch_dims)
        if arg2.traced or arg2.requires_grad:
            arg2 = broadcast_to(arg2, output_shape[:-2] + arg2.shape[-2:])
            arg2 = broadcast_batch_dims(arg2, output_batch_dims)

        res = Tensor(
            shape=output_shape,
            dtype=output_dtype,
            device=arg1.logical_device,
            materialize=False,
            name=self.name,
            batch_dims=output_batch_dims,
        )

        res.set_maxpr(self.maxpr)
        res.add_arguments(arg1, arg2)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule
        res.custom_kernel_path = self.custom_kernel_path()

        if not res.stage_realization:
            self.eagerxpr([arg1, arg2], res)

        # Reshape output back to the correct rank if inputs were 1D
        if arg1_has_rank_1 and arg2_has_rank_1:
            # Vector dot product results in a scalar-like (1,1) shape, squeeze it
            res = reshape(res, output_shape[:-2] + (1, 1))
        elif arg1_has_rank_1:
            # Squeeze the first dimension
            res = reshape(res, output_shape[:-2] + (res.shape[1],))
        elif arg2_has_rank_1:
            # Squeeze the second dimension
            res = reshape(res, output_shape[:-2] + (res.shape[0],))

        res.creator_op = self
        return res

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """
        Computes the output shape for matrix multiplication.

        The batch dimensions are broadcasted, and the last two dimensions follow
        standard matrix multiplication rules (M, K) @ (K, N) -> (M, N).

        Parameters
        ----------
        input_shapes : tuple
            A tuple of two shapes, `(shape1, shape2)`.

        Returns
        -------
        tuple
            The shape of the resulting tensor.
        """
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

    def _validate_inputs(self, arg1: Tensor, arg2: Tensor) -> None:
        """
        Validates inputs for matrix multiplication.

        Checks for type, dtype, device, and shape compatibility.

        Parameters
        ----------
        arg1 : Tensor
            The first input tensor.
        arg2 : Tensor
            The second input tensor.

        Raises
        ------
        TypeError
            If inputs are not Tensor instances.
        ValueError
            If dtypes, devices, or shapes are incompatible.
        """
        if not isinstance(arg1, Tensor) or not isinstance(arg2, Tensor):
            raise TypeError("Both arguments must be Tensor instances")
        if arg1.dtype != arg2.dtype:
            raise ValueError(f"Dtypes {arg1.dtype} and {arg2.dtype} are incompatible")
        if arg1.logical_device != arg2.logical_device:
            raise ValueError(
                f"Devices {arg1.logical_device} and {arg2.logical_device} are incompatible"
            )
        if arg1.shape[-1] != arg2.shape[-2]:
            raise ValueError(
                f"Shapes {arg1.shape} and {arg2.shape} are not compatible for matrix multiplication"
            )

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        """
        Defines the MAX graph implementation for matrix multiplication.

        For inputs with more than 4 dimensions, it reshapes them to 3D for
        batched matmul and then reshapes the result back.

        Parameters
        ----------
        args : list[TensorValue]
            A list containing the two input tensor values.
        output : Tensor
            The output tensor to store the result in.
        """
        x_val, y_val = args[0], args[1]
        x_shape = x_val.shape
        y_shape = y_val.shape
        output_shape = output.batch_dims + output.shape

        if len(output_shape) <= 4:
            output.tensor_value = ops.matmul(args[0], args[1])
        else:
            if x_shape[:-2] != y_shape[:-2]:
                raise ValueError(
                    f"Shapes {x_shape} and {y_shape} are not compatible for matrix multiplication "
                    f"(batch dimensions mismatch: {x_shape[:-2]} vs {y_shape[:-2]})"
                )
            # Reshape high-rank tensors to 3D for batched matmul, then reshape back
            batch_dims_x = [int(dim) for dim in x_shape[:-2]]
            batch_dims_y = [int(dim) for dim in y_shape[:-2]]
            new_shape_x = (
                np.prod(batch_dims_x).item(),
                int(x_shape[-2]),
                int(x_shape[-1]),
            )
            new_shape_y = (
                np.prod(batch_dims_y).item(),
                int(y_shape[-2]),
                int(y_shape[-1]),
            )
            x_val_b = ops.reshape(x_val, new_shape_x)
            y_val_b = ops.reshape(y_val, new_shape_y)
            matmul_result = ops.matmul(x_val_b, y_val_b)
            reshaped_result = ops.reshape(
                matmul_result,
                tuple(args[0].shape[:-2])
                + (matmul_result.shape[-2], matmul_result.shape[-1]),
            )
            output.tensor_value = reshaped_result

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        """
        Defines the eager mode execution using `numpy.matmul`.

        Parameters
        ----------
        args : list[Tensor]
            A list containing the two input tensors.
        output : Tensor
            The output tensor to store the result in.
        """
        arg0_numpy = args[0].to_numpy()
        arg1_numpy = args[1].to_numpy()
        np_result = np.matmul(arg0_numpy, arg1_numpy)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        """
        Defines the vector-Jacobian product (VJP) rule for matmul.

        The gradients are `g @ y.T` and `x.T @ g` for inputs `x` and `y` and
        gradient `g`.

        Parameters
        ----------
        primals : list[Tensor]
            The original inputs to the operation, `(x, y)`.
        cotangent : Tensor
            The gradient of the loss with respect to the output.
        output : Tensor
            The output of the forward pass.

        Returns
        -------
        list[Tensor]
            A list containing the gradients with respect to each input.
        """
        x, y = primals
        from .view import transpose

        grad_x = matmul(cotangent, transpose(y))
        grad_y = matmul(transpose(x), cotangent)
        return [grad_x, grad_y]

    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
        """
        Defines the Jacobian-vector product (JVP) rule for matmul.

        Based on the product rule, the tangent is `(dx @ y) + (x @ dy)`.

        Parameters
        ----------
        primals : list[Tensor]
            The original inputs to the operation, `(x, y)`.
        tangents : list[Tensor]
            The tangents of the inputs, `(tx, ty)`.
        output : Tensor
            The output of the forward pass.

        Returns
        -------
        Tensor
            The tangent of the output.
        """
        x, y = primals
        tx, ty = tangents

        from .binary import add

        return add(matmul(x, ty), matmul(tx, y))


# Global operation instance for efficiency
_matmul_op = MatMulOp()


def matmul(arg0: Tensor | float | int, arg1: Tensor | float | int) -> Tensor:
    """
    Performs matrix multiplication on two tensors.

    This function follows the semantics of `numpy.matmul`, supporting
    multiplication of 1D vectors, 2D matrices, and stacks of matrices.

    - If both arguments are 1D tensors of size `N`, it computes the inner
      (dot) product and returns a scalar-like tensor.
    - If one argument is a 2D tensor (M, K) and the other is a 1D tensor (K),
      it promotes the vector to a matrix (1, K) or (K, 1) for the
      multiplication, then squeezes the result back to a 1D tensor.
    - If both arguments are 2D tensors, `(M, K) @ (K, N)`, it performs standard
      matrix multiplication, resulting in an tensor of shape `(M, N)`.
    - If either argument has more than 2 dimensions, it is treated as a stack
      of matrices residing in the last two dimensions and is broadcast accordingly.

    Parameters
    ----------
    arg0 : Tensor | float | int
        The first input tensor.
    arg1 : Tensor | float | int
        The second input tensor.

    Returns
    -------
    Tensor
        The result of the matrix multiplication.

    Examples
    --------
    >>> import nabla as nb
    >>> # Vector-vector product (dot product)
    >>> v1 = nb.tensor([1, 2, 3])
    >>> v2 = nb.tensor([4, 5, 6])
    >>> nb.matmul(v1, v2)
    Tensor([32], dtype=int32)

    >>> # Matrix-vector product
    >>> M = nb.tensor([[1, 2], [3, 4]])
    >>> v = nb.tensor([5, 6])
    >>> nb.matmul(M, v)
    Tensor([17, 39], dtype=int32)

    >>> # Batched matrix-matrix product
    >>> M1 = nb.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) # Shape (2, 2, 2)
    >>> M2 = nb.tensor([[[9, 1], [2, 3]], [[4, 5], [6, 7]]]) # Shape (2, 2, 2)
    >>> nb.matmul(M1, M2)
    Tensor([[[ 13,   7],
            [ 35,  15]],
    <BLANKLINE>
           [[ 56,  47],
            [ 76,  67]]], dtype=int32)
    """
    from .binary import _ensure_tensor

    arg0 = _ensure_tensor(arg0)
    arg1 = _ensure_tensor(arg1)
    return _matmul_op.forward(arg0, arg1)


# ===----------------------------------------------------------------------=== #
# Convolution Operations
# ===----------------------------------------------------------------------=== #

# Global operation caches for efficiency
_conv2d_op_cache = {}
_conv2d_transpose_op_cache = {}


def _normalize_tuple(value, n: int, name: str) -> tuple:
    """Normalize a parameter to a tuple of length n.
    
    Parameters
    ----------
    value : int or tuple
        The value to normalize
    n : int
        The desired tuple length
    name : str
        Parameter name for error messages
        
    Returns
    -------
    tuple
        Normalized tuple of length n
    """
    if isinstance(value, int):
        return (value,) * n
    elif isinstance(value, (tuple, list)):
        if len(value) == n:
            return tuple(value)
        else:
            raise ValueError(
                f"{name} must be an int or a tuple of {n} ints, got {value}"
            )
    else:
        raise TypeError(
            f"{name} must be an int or a tuple, got {type(value)}"
        )


def _normalize_padding(padding, name: str = "padding") -> tuple[tuple[int, int], tuple[int, int]]:
    """Normalize padding argument to ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)).
    
    Supports several formats matching PyTorch:
    - int: same padding on all sides
    - (pad_h, pad_w): symmetric padding for height and width
    - (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right): explicit padding
    - ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)): already normalized
    - "valid": no padding
    - "same": padding to preserve spatial dimensions (only for stride=1)
    
    Parameters
    ----------
    padding : int, tuple, or str
        Padding specification
    name : str
        Parameter name for error messages
        
    Returns
    -------
    tuple
        ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right))
    """
    if isinstance(padding, str):
        if padding.lower() == "valid":
            return ((0, 0), (0, 0))
        elif padding.lower() == "same":
            # For "same" padding, we need kernel size and dilation which we don't have here
            # This will be handled in the operation class
            return "same"  # Return as-is, will be computed later
        else:
            raise ValueError(f"Unknown string padding '{padding}'. Use 'valid' or 'same'.")
    
    if isinstance(padding, int):
        return ((padding, padding), (padding, padding))
    
    if isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            # (pad_h, pad_w) format
            pad_h, pad_w = padding
            return ((pad_h, pad_h), (pad_w, pad_w))
        elif len(padding) == 4:
            # (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right) format
            return ((padding[0], padding[1]), (padding[2], padding[3]))
        elif len(padding) == 2 and isinstance(padding[0], (tuple, list)):
            # ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)) format
            return tuple(tuple(p) for p in padding)
        else:
            raise ValueError(
                f"{name} tuple must have length 2 or 4, got {len(padding)}"
            )
    
    raise TypeError(f"{name} must be an int, tuple, or 'valid'/'same', got {type(padding)}")


class Conv2DOp(Operation):
    """2D Convolution operation with PyTorch-compatible NCHW layout.
    
    Input Layout: NCHW (batch, channels, height, width)
    Weight Layout: (out_channels, in_channels/groups, kernel_height, kernel_width)
    
    This matches PyTorch's nn.Conv2d exactly.
    """
    
    def __init__(
        self,
        stride: tuple[int, int],
        padding: tuple[tuple[int, int], tuple[int, int]] | str,
        dilation: tuple[int, int],
        groups: int,
    ):
        super().__init__("conv2d")
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
    
    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compute output shape for NCHW layout."""
        if len(input_shapes) != 2:
            raise ValueError(f"Conv2D requires 2 input shapes, got {len(input_shapes)}")
        
        input_shape, weight_shape = input_shapes
        n, c_in, h_in, w_in = input_shape
        c_out, c_in_per_group, k_h, k_w = weight_shape
        
        # Validate channel dimensions
        if c_in != c_in_per_group * self.groups:
            raise ValueError(
                f"Input channels ({c_in}) must equal weight in_channels/groups "
                f"({c_in_per_group}) * groups ({self.groups}) = {c_in_per_group * self.groups}"
            )
        
        # Handle "same" padding
        if self.padding == "same":
            if self.stride != (1, 1):
                raise ValueError("padding='same' is only supported for stride=1")
            # For "same" padding with stride=1, output size equals input size
            h_out, w_out = h_in, w_in
        else:
            # Regular padding calculation
            (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right) = self.padding
            dil_h, dil_w = self.dilation
            s_h, s_w = self.stride
            
            # PyTorch formula: floor((H + 2*pad - dilation*(kernel-1) - 1) / stride + 1)
            h_out = (h_in + pad_h_top + pad_h_bottom - dil_h * (k_h - 1) - 1) // s_h + 1
            w_out = (w_in + pad_w_left + pad_w_right - dil_w * (k_w - 1) - 1) // s_w + 1
        
        if h_out <= 0 or w_out <= 0:
            raise ValueError(
                f"Computed non-positive output dimensions: ({n}, {c_out}, {h_out}, {w_out})"
            )
        
        return (n, c_out, h_out, w_out)
    
    def _validate_inputs(self, input_tensor: Tensor, weight_tensor: Tensor) -> None:
        """Validate input tensors."""
        if not isinstance(input_tensor, Tensor) or not isinstance(weight_tensor, Tensor):
            raise TypeError("Both arguments must be Tensor instances")
        
        if len(input_tensor.shape) != 4:
            raise ValueError(f"Input must be 4D (NCHW), got shape {input_tensor.shape}")
        
        if len(weight_tensor.shape) != 4:
            raise ValueError(f"Weight must be 4D, got shape {weight_tensor.shape}")
        
        if input_tensor.dtype != weight_tensor.dtype:
            raise ValueError(
                f"Input and weight dtypes must match: {input_tensor.dtype} vs {weight_tensor.dtype}"
            )
        
        if input_tensor.logical_device != weight_tensor.logical_device:
            raise ValueError(
                f"Input and weight must be on same device: "
                f"{input_tensor.logical_device} vs {weight_tensor.logical_device}"
            )
    
    def forward(self, *args: Tensor) -> Tensor:
        """Forward pass for convolution."""
        if len(args) != 2:
            raise ValueError(f"Conv2D requires 2 arguments, got {len(args)}")
        
        from .operation import move_to_best_device
        args = move_to_best_device(*args)
        input_tensor, weight_tensor = args
        
        self._validate_inputs(input_tensor, weight_tensor)
        
        output_shape = self.compute_output_shape(input_tensor.shape, weight_tensor.shape)
        output_dtype = input_tensor.dtype
        output_batch_dims = input_tensor.batch_dims
        
        res = Tensor(
            shape=output_shape,
            dtype=output_dtype,
            device=input_tensor.logical_device,
            materialize=False,
            name=self.name,
            batch_dims=output_batch_dims,
        )
        
        res.set_maxpr(self.maxpr)
        res.add_arguments(input_tensor, weight_tensor)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule
        res.custom_kernel_path = self.custom_kernel_path()
        
        if not res.stage_realization:
            self.eagerxpr([input_tensor, weight_tensor], res)
        
        res.creator_op = self
        return res
    
    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        """MAX graph implementation - convert NCHW to NHWC for MAX ops.
        
        Note: MAX backend runtime currently does not support dilation > 1.
        This is a known limitation of the MAX engine (as of 2025). Operations with
        dilation > 1 will work in eager mode (using PyTorch) but will fail in JIT mode.
        """
        input_val, weight_val = args
        
        # Convert NCHW -> NHWC for input: (N,C,H,W) -> (N,H,W,C)
        input_nhwc = ops.permute(input_val, (0, 2, 3, 1))
        
        # Convert weight from (C_out, C_in/g, H, W) to (H, W, C_in/g, C_out)
        weight_hwio = ops.permute(weight_val, (2, 3, 1, 0))
        
        # Compute "same" padding if needed
        if self.padding == "same":
            k_h = weight_val.shape[2]
            k_w = weight_val.shape[3]
            dil_h, dil_w = self.dilation
            # For "same" padding: total_pad = dilation * (kernel - 1)
            total_pad_h = dil_h * (k_h - 1)
            total_pad_w = dil_w * (k_w - 1)
            pad_h_top = total_pad_h // 2
            pad_h_bottom = total_pad_h - pad_h_top
            pad_w_left = total_pad_w // 2
            pad_w_right = total_pad_w - pad_w_left
            padding_for_max = (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)
        else:
            (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right) = self.padding
            padding_for_max = (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)
        
        # Call MAX conv2d with NHWC layout
        result_nhwc = ops.conv2d(
            x=input_nhwc,
            filter=weight_hwio,
            stride=self.stride,
            dilation=self.dilation,
            padding=padding_for_max,
            groups=self.groups,
        )
        
        # Convert result back from NHWC -> NCHW: (N,H,W,C) -> (N,C,H,W)
        output.tensor_value = ops.permute(result_nhwc, (0, 3, 1, 2))
    
    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        """Eager execution using pure NumPy with im2col."""
        input_tensor, weight_tensor = args
        
        # Get input as numpy arrays (already in NCHW format)
        input_np = input_tensor.to_numpy()
        weight_np = weight_tensor.to_numpy()
        
        # Extract dimensions
        N, C_in, H_in, W_in = input_np.shape
        C_out, _, K_h, K_w = weight_np.shape
        
        # Compute padding
        if self.padding == "same":
            # For 'same', compute padding to keep spatial dims constant (stride=1 assumed)
            pad_h = ((H_in - 1) * self.stride[0] + self.dilation[0] * (K_h - 1) + 1 - H_in)
            pad_w = ((W_in - 1) * self.stride[1] + self.dilation[1] * (K_w - 1) + 1 - W_in)
            pad_h = max(0, pad_h)
            pad_w = max(0, pad_w)
            pad_h_top = pad_h // 2
            pad_h_bottom = pad_h - pad_h_top
            pad_w_left = pad_w // 2
            pad_w_right = pad_w - pad_w_left
        else:
            (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right) = self.padding
        
        # Apply padding
        if pad_h_top > 0 or pad_h_bottom > 0 or pad_w_left > 0 or pad_w_right > 0:
            input_padded = np.pad(
                input_np,
                ((0, 0), (0, 0), (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)),
                mode='constant',
                constant_values=0
            )
        else:
            input_padded = input_np
        
        _, _, H_padded, W_padded = input_padded.shape
        
        # Compute output dimensions
        H_out = (H_padded - self.dilation[0] * (K_h - 1) - 1) // self.stride[0] + 1
        W_out = (W_padded - self.dilation[1] * (K_w - 1) - 1) // self.stride[1] + 1
        
        # Handle groups
        if self.groups > 1:
            # Grouped convolution
            C_in_per_group = C_in // self.groups
            C_out_per_group = C_out // self.groups
            output_np = np.zeros((N, C_out, H_out, W_out), dtype=input_np.dtype)
            
            for g in range(self.groups):
                # Extract group inputs and weights
                input_g = input_padded[:, g * C_in_per_group:(g + 1) * C_in_per_group, :, :]
                weight_g = weight_np[g * C_out_per_group:(g + 1) * C_out_per_group, :, :, :]
                
                # Perform convolution for this group
                output_g = self._conv2d_numpy_single_group(
                    input_g, weight_g, H_out, W_out, K_h, K_w
                )
                
                output_np[:, g * C_out_per_group:(g + 1) * C_out_per_group, :, :] = output_g
        else:
            # Standard convolution (groups=1)
            output_np = self._conv2d_numpy_single_group(
                input_padded, weight_np, H_out, W_out, K_h, K_w
            )
        
        output.impl_(output_np)
    
    def _conv2d_numpy_single_group(self, input_np, weight_np, H_out, W_out, K_h, K_w):
        """Helper function to perform conv2d for a single group using vectorized im2col."""
        N, C_in, H_padded, W_padded = input_np.shape
        C_out = weight_np.shape[0]
        
        # Fast path for common case: stride=1, dilation=1
        if self.stride == (1, 1) and self.dilation == (1, 1):
            # Use numpy's stride tricks for efficient patch extraction
            from numpy.lib.stride_tricks import as_strided
            
            # Create sliding window view of input
            # Shape: (N, C_in, H_out, W_out, K_h, K_w)
            shape = (N, C_in, H_out, W_out, K_h, K_w)
            strides = (
                input_np.strides[0],  # N
                input_np.strides[1],  # C_in
                input_np.strides[2],  # H step
                input_np.strides[3],  # W step
                input_np.strides[2],  # K_h step
                input_np.strides[3],  # K_w step
            )
            patches = as_strided(input_np, shape=shape, strides=strides)
            
            # Reshape for matrix multiplication
            # patches: (N, C_in, H_out, W_out, K_h, K_w) -> (N, H_out, W_out, C_in * K_h * K_w)
            patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N, H_out, W_out, -1)
            
            # weight: (C_out, C_in, K_h, K_w) -> (C_out, C_in * K_h * K_w)
            weight_reshaped = weight_np.reshape(C_out, -1)
            
            # Matrix multiply: (N, H_out, W_out, C_in*K_h*K_w) @ (C_in*K_h*K_w, C_out) -> (N, H_out, W_out, C_out)
            output_np = np.dot(patches, weight_reshaped.T)
            
            # Transpose back to NCHW: (N, H_out, W_out, C_out) -> (N, C_out, H_out, W_out)
            output_np = output_np.transpose(0, 3, 1, 2)
            
        else:
            # General case with stride/dilation: use optimized loop with einsum
            output_np = np.zeros((N, C_out, H_out, W_out), dtype=input_np.dtype)
            
            # Extract all patches at once for each output position
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * self.stride[0]
                    w_start = w_out * self.stride[1]
                    
                    # Extract patch with dilation (vectorized)
                    h_indices = h_start + np.arange(K_h) * self.dilation[0]
                    w_indices = w_start + np.arange(K_w) * self.dilation[1]
                    
                    # Check bounds
                    h_valid = h_indices < H_padded
                    w_valid = w_indices < W_padded
                    
                    if h_valid.all() and w_valid.all():
                        # All indices valid - fast path
                        # Extract patch: (N, C_in, K_h, K_w)
                        patch = input_np[:, :, h_indices[:, None], w_indices]
                        
                        # Compute conv: einsum 'nchw,ochw->no' then assign to output
                        # (N, C_in, K_h, K_w) * (C_out, C_in, K_h, K_w) -> (N, C_out)
                        output_np[:, :, h_out, w_out] = np.einsum('nchw,ochw->no', patch, weight_np)
                    else:
                        # Handle boundary conditions
                        for kh in range(K_h):
                            for kw in range(K_w):
                                h_in = h_start + kh * self.dilation[0]
                                w_in = w_start + kw * self.dilation[1]
                                
                                if h_in < H_padded and w_in < W_padded:
                                    input_patch = input_np[:, :, h_in, w_in]  # (N, C_in)
                                    weight_patch = weight_np[:, :, kh, kw]    # (C_out, C_in)
                                    output_np[:, :, h_out, w_out] += np.dot(input_patch, weight_patch.T)
        
        return output_np
    
    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        """VJP rule for conv2d.
        
        Given cotangent ∂L/∂output, computes:
        - ∂L/∂input: using conv2d_transpose with weight
        - ∂L/∂weight: using permuted conv2d between input and cotangent
        
        Based on the mathematical derivation in conv2d_manual_grad.py.
        """
        input_arr, weight_arr = primals
        
        # Get dimensions
        N, C_in, H_in, W_in = input_arr.shape
        C_out, _, K_H, K_W = weight_arr.shape
        _, _, H_out, W_out = output.shape
        
        # Parse stride and dilation
        s_h, s_w = self.stride
        d_h, d_w = self.dilation
        
        # Get normalized padding - handle "same" and "valid" specially
        if self.padding == "same":
            # For "same" padding with stride=1, we need to compute actual padding
            # Same padding formula: total_pad = (kernel - 1) * dilation
            pad_h_total = (K_H - 1) * d_h
            pad_w_total = (K_W - 1) * d_w
            pad_h_top = pad_h_total // 2
            pad_h_bottom = pad_h_total - pad_h_top
            pad_w_left = pad_w_total // 2
            pad_w_right = pad_w_total - pad_w_left
        elif self.padding == "valid":
            pad_h_top = pad_h_bottom = pad_w_left = pad_w_right = 0
        else:
            (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right) = self.padding
        
        # === 1. Gradient w.r.t. Input (∂L/∂X) ===
        # Strategy: Use conv2d_transpose to "undo" the convolution
        
        # First, compute the padded input dimensions
        H_padded = H_in + pad_h_top + pad_h_bottom
        W_padded = W_in + pad_w_left + pad_w_right
        
        # Compute the dimensions after conv_transpose2d
        H_prime = (H_out - 1) * s_h + d_h * (K_H - 1) + 1
        W_prime = (W_out - 1) * s_w + d_w * (K_W - 1) + 1
        
        # Compute output_padding needed
        op_h_padded = H_padded - H_prime
        op_w_padded = W_padded - W_prime
        
        # Compute cropping needed (if output_padding would be negative)
        crop_h_end = max(0, -op_h_padded)
        crop_w_end = max(0, -op_w_padded)
        op_h_padded = max(0, op_h_padded)
        op_w_padded = max(0, op_w_padded)
        
        # Apply conv_transpose2d
        grad_input_padded = conv2d_transpose(
            cotangent, weight_arr,
            stride=self.stride,
            padding=(0, 0),  # No padding in conv_transpose2d itself
            output_padding=(op_h_padded, op_w_padded),
            dilation=self.dilation,
            groups=self.groups
        )
        
        # Crop if needed
        if crop_h_end > 0:
            grad_input_padded = grad_input_padded[:, :, :-crop_h_end, :]
        if crop_w_end > 0:
            grad_input_padded = grad_input_padded[:, :, :, :-crop_w_end]
        
        # Remove padding to get gradient w.r.t. original input
        h_end = H_padded - pad_h_bottom if pad_h_bottom > 0 else H_padded
        w_end = W_padded - pad_w_right if pad_w_right > 0 else W_padded
        grad_input = grad_input_padded[:, :, pad_h_top:h_end, pad_w_left:w_end]
        
        # === 2. Gradient w.r.t. Weight (∂L/∂W) ===
        # Strategy: Permute input and cotangent to use conv2d
        # For now, only implement groups=1 case
        
        if self.groups != 1:
            raise NotImplementedError("VJP for grouped convolution (groups > 1) not yet implemented")
        
        # Pad input manually using numpy (for now - TODO: make this work in graph mode)
        import numpy as np
        input_np = input_arr.to_numpy()
        if pad_h_top > 0 or pad_h_bottom > 0 or pad_w_left > 0 or pad_w_right > 0:
            input_np = np.pad(
                input_np,
                ((0, 0), (0, 0), (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)),
                mode='constant',
                constant_values=0
            )
        input_padded = Tensor.from_numpy(input_np)
        
        # Compute effective padded dimensions for cropping
        H_pad_eff = d_h * (K_H - 1) + s_h * (H_out - 1) + 1
        W_pad_eff = d_w * (K_W - 1) + s_w * (W_out - 1) + 1
        
        # Crop input if needed
        input_cropped = input_padded[:, :, :H_pad_eff, :W_pad_eff]
        
        # Permute: (N, C, H, W) -> (C, N, H, W)
        from ..ops.view import permute
        input_perm = permute(input_cropped, (1, 0, 2, 3))
        cotangent_perm = permute(cotangent, (1, 0, 2, 3))
        
        # Convolve with swapped stride/dilation
        grad_weight_perm = conv2d(
            input_perm, cotangent_perm,
            stride=(d_h, d_w),
            dilation=(s_h, s_w),
            padding=(0, 0),
            groups=1
        )
        
        # Permute back: (C_out, C_in, K_H, K_W) <- (C_in, C_out, K_H, K_W)  
        grad_weight = permute(grad_weight_perm, (1, 0, 2, 3))
        
        return [grad_input, grad_weight]
    
    
    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
        """JVP rule - to be implemented after testing."""
        raise NotImplementedError("JVP for conv2d will be implemented after testing")


def conv2d(
    input_tensor: Tensor,
    weight: Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple | str = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> Tensor:
    """2D convolution with PyTorch-compatible NCHW layout.
    
    Applies a 2D convolution over an input tensor. This function matches
    PyTorch's F.conv2d exactly in terms of input/output shapes and semantics.
    
    Parameters
    ----------
    input_tensor : Tensor
        Input tensor of shape (N, C_in, H, W)
    weight : Tensor
        Convolution kernel of shape (C_out, C_in/groups, K_H, K_W)
    stride : int or tuple, optional
        Stride of the convolution. Default: 1
    padding : int, tuple, or str, optional
        Padding added to input. Can be:
        - int: same padding on all sides
        - (pad_h, pad_w): symmetric padding
        - 'valid': no padding
        - 'same': padding to preserve size (stride=1 only)
        Default: 0
    dilation : int or tuple, optional
        Spacing between kernel elements. Default: 1
    groups : int, optional
        Number of blocked connections. Default: 1
        
    Returns
    -------
    Tensor
        Output tensor of shape (N, C_out, H_out, W_out)
        
    Examples
    --------
    >>> import nabla as nb
    >>> # Simple 2D convolution
    >>> x = nb.zeros((1, 3, 32, 32))  # NCHW
    >>> w = nb.zeros((64, 3, 3, 3))   # (out_ch, in_ch, H, W)
    >>> y = nb.conv2d(x, w)
    >>> y.shape
    (1, 64, 30, 30)
    """
    # Normalize parameters
    norm_stride = _normalize_tuple(stride, 2, "stride")
    norm_dilation = _normalize_tuple(dilation, 2, "dilation")
    norm_padding = _normalize_padding(padding, "padding")
    
    # Cache operation instances for efficiency
    cache_key = (norm_stride, norm_padding, norm_dilation, groups)
    if cache_key not in _conv2d_op_cache:
        _conv2d_op_cache[cache_key] = Conv2DOp(
            stride=norm_stride,
            padding=norm_padding,
            dilation=norm_dilation,
            groups=groups,
        )
    
    op = _conv2d_op_cache[cache_key]
    return op.forward(input_tensor, weight)


class Conv2DTransposeOp(Operation):
    """2D Transposed Convolution with PyTorch-compatible NCHW layout.
    
    Input Layout: NCHW (batch, channels, height, width)
    Weight Layout: (in_channels, out_channels/groups, kernel_height, kernel_width)
    
    Note: For transposed convolution, the weight layout is transposed compared to conv2d.
    This matches PyTorch's nn.ConvTranspose2d.
    """
    
    def __init__(
        self,
        stride: tuple[int, int],
        padding: tuple[tuple[int, int], tuple[int, int]],
        output_padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
    ):
        super().__init__("conv2d_transpose")
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
    
    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compute output shape for NCHW layout transposed convolution."""
        if len(input_shapes) != 2:
            raise ValueError(f"Conv2DTranspose requires 2 input shapes, got {len(input_shapes)}")
        
        input_shape, weight_shape = input_shapes
        n, c_in, h_in, w_in = input_shape
        c_in_w, c_out_per_group, k_h, k_w = weight_shape
        
        # Validate channel dimensions
        if c_in != c_in_w:
            raise ValueError(
                f"Input channels ({c_in}) must match weight input channels ({c_in_w})"
            )
        
        c_out = c_out_per_group * self.groups
        
        # PyTorch formula for transposed convolution:
        # H_out = (H_in - 1) * stride - 2*padding + dilation*(kernel-1) + output_padding + 1
        (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right) = self.padding
        out_pad_h, out_pad_w = self.output_padding
        dil_h, dil_w = self.dilation
        s_h, s_w = self.stride
        
        # Use symmetric padding for the formula (PyTorch assumes symmetric)
        pad_h = pad_h_top  # Assuming symmetric
        pad_w = pad_w_left
        
        h_out = (h_in - 1) * s_h - 2 * pad_h + dil_h * (k_h - 1) + out_pad_h + 1
        w_out = (w_in - 1) * s_w - 2 * pad_w + dil_w * (k_w - 1) + out_pad_w + 1
        
        if h_out <= 0 or w_out <= 0:
            raise ValueError(
                f"Computed non-positive output dimensions: ({n}, {c_out}, {h_out}, {w_out})"
            )
        
        return (n, c_out, h_out, w_out)
    
    def _validate_inputs(self, input_tensor: Tensor, weight_tensor: Tensor) -> None:
        """Validate input tensors."""
        if not isinstance(input_tensor, Tensor) or not isinstance(weight_tensor, Tensor):
            raise TypeError("Both arguments must be Tensor instances")
        
        if len(input_tensor.shape) != 4:
            raise ValueError(f"Input must be 4D (NCHW), got shape {input_tensor.shape}")
        
        if len(weight_tensor.shape) != 4:
            raise ValueError(f"Weight must be 4D, got shape {weight_tensor.shape}")
        
        if input_tensor.dtype != weight_tensor.dtype:
            raise ValueError(
                f"Input and weight dtypes must match: {input_tensor.dtype} vs {weight_tensor.dtype}"
            )
        
        if input_tensor.logical_device != weight_tensor.logical_device:
            raise ValueError(
                f"Input and weight must be on same device: "
                f"{input_tensor.logical_device} vs {weight_tensor.logical_device}"
            )
    
    def forward(self, *args: Tensor) -> Tensor:
        """Forward pass for transposed convolution."""
        if len(args) != 2:
            raise ValueError(f"Conv2DTranspose requires 2 arguments, got {len(args)}")
        
        from .operation import move_to_best_device
        args = move_to_best_device(*args)
        input_tensor, weight_tensor = args
        
        self._validate_inputs(input_tensor, weight_tensor)
        
        output_shape = self.compute_output_shape(input_tensor.shape, weight_tensor.shape)
        output_dtype = input_tensor.dtype
        output_batch_dims = input_tensor.batch_dims
        
        res = Tensor(
            shape=output_shape,
            dtype=output_dtype,
            device=input_tensor.logical_device,
            materialize=False,
            name=self.name,
            batch_dims=output_batch_dims,
        )
        
        res.set_maxpr(self.maxpr)
        res.add_arguments(input_tensor, weight_tensor)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule
        res.custom_kernel_path = self.custom_kernel_path()
        
        if not res.stage_realization:
            self.eagerxpr([input_tensor, weight_tensor], res)
        
        res.creator_op = self
        return res
    
    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        """MAX graph implementation - convert NCHW to NHWC for MAX ops."""
        input_val, weight_val = args
        
        # Convert NCHW -> NHWC for input: (N,C,H,W) -> (N,H,W,C)
        input_nhwc = ops.permute(input_val, (0, 2, 3, 1))
        
        # For transposed conv, weight is (C_in, C_out/g, H, W)
        # MAX expects (H, W, C_out, C_in) for conv2d_transpose
        # So we transpose: (C_in, C_out/g, H, W) -> (H, W, C_out/g, C_in)
        weight_hwoi = ops.permute(weight_val, (2, 3, 1, 0))
        
        (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right) = self.padding
        padding_for_max = (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)
        
        # Call MAX conv2d_transpose with NHWC layout
        result_nhwc = ops.conv2d_transpose(
            x=input_nhwc,
            filter=weight_hwoi,
            stride=self.stride,
            dilation=self.dilation,
            padding=padding_for_max,
            output_paddings=self.output_padding,
        )
        
        # Convert result back from NHWC -> NCHW: (N,H,W,C) -> (N,C,H,W)
        output.tensor_value = ops.permute(result_nhwc, (0, 3, 1, 2))
    
    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        """Eager execution using pure NumPy."""
        input_tensor, weight_tensor = args
        
        # Get input as numpy arrays (already in NCHW format)
        input_np = input_tensor.to_numpy()
        weight_np = weight_tensor.to_numpy()
        
        # Extract dimensions
        N, C_in, H_in, W_in = input_np.shape
        # Weight shape: (C_in, C_out // groups, K_h, K_w) for transposed conv
        _, C_out_per_group, K_h, K_w = weight_np.shape
        C_out = C_out_per_group * self.groups
        
        # Compute output dimensions
        (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right) = self.padding
        
        H_out = (H_in - 1) * self.stride[0] - pad_h_top - pad_h_bottom + self.dilation[0] * (K_h - 1) + self.output_padding[0] + 1
        W_out = (W_in - 1) * self.stride[1] - pad_w_left - pad_w_right + self.dilation[1] * (K_w - 1) + self.output_padding[1] + 1
        
        # Initialize output
        output_np = np.zeros((N, C_out, H_out, W_out), dtype=input_np.dtype)
        
        # Handle groups
        C_in_per_group = C_in // self.groups
        
        # Optimized transposed convolution using vectorized operations
        # Fast path for common case: groups=1, stride=(1,1) or (2,2), dilation=(1,1)
        if self.groups == 1 and self.dilation == (1, 1):
            # Vectorized approach: process all input positions at once
            for h_in in range(H_in):
                h_out_start = h_in * self.stride[0] - pad_h_top
                h_indices = h_out_start + np.arange(K_h) * self.dilation[0]
                h_valid = (h_indices >= 0) & (h_indices < H_out)
                
                for w_in in range(W_in):
                    w_out_start = w_in * self.stride[1] - pad_w_left
                    w_indices = w_out_start + np.arange(K_w) * self.dilation[1]
                    w_valid = (w_indices >= 0) & (w_indices < W_out)
                    
                    # Create meshgrid for valid positions
                    h_valid_idx = np.where(h_valid)[0]
                    w_valid_idx = np.where(w_valid)[0]
                    
                    if len(h_valid_idx) > 0 and len(w_valid_idx) > 0:
                        # Extract input slice: (N, C_in)
                        input_slice = input_np[:, :, h_in, w_in]  # (N, C_in)
                        
                        # Extract weight slice for valid kernel positions
                        # weight_np: (C_in, C_out, K_h, K_w)
                        weight_slice = weight_np[:, :, h_valid_idx[:, None], w_valid_idx]  # (C_in, C_out, len(h), len(w))
                        
                        # Compute contribution: einsum 'nc,cohw->nohw'
                        # (N, C_in) * (C_in, C_out, H_k, W_k) -> (N, C_out, H_k, W_k)
                        contrib = np.einsum('nc,cohw->nohw', input_slice, weight_slice)
                        
                        # Scatter to output
                        h_out_idx = h_indices[h_valid_idx]
                        w_out_idx = w_indices[w_valid_idx]
                        for i, h_out in enumerate(h_out_idx):
                            for j, w_out in enumerate(w_out_idx):
                                output_np[:, :, h_out, w_out] += contrib[:, :, i, j]
        else:
            # General case: handle groups and arbitrary dilation
            for n in range(N):
                for g in range(self.groups):
                    c_in_start = g * C_in_per_group
                    c_out_start = g * C_out_per_group
                    
                    for c_in_local in range(C_in_per_group):
                        c_in = c_in_start + c_in_local
                        
                        for h_in in range(H_in):
                            for w_in in range(W_in):
                                input_val = input_np[n, c_in, h_in, w_in]
                                
                                h_out_start = h_in * self.stride[0] - pad_h_top
                                w_out_start = w_in * self.stride[1] - pad_w_left
                                
                                # Vectorize the kernel loop
                                h_indices = h_out_start + np.arange(K_h) * self.dilation[0]
                                w_indices = w_out_start + np.arange(K_w) * self.dilation[1]
                                
                                h_valid = (h_indices >= 0) & (h_indices < H_out)
                                w_valid = (w_indices >= 0) & (w_indices < W_out)
                                
                                for kh in np.where(h_valid)[0]:
                                    h_out = h_indices[kh]
                                    for kw in np.where(w_valid)[0]:
                                        w_out = w_indices[kw]
                                        
                                        # Vectorize output channel dimension
                                        weight_vals = weight_np[c_in, :, kh, kw]  # (C_out_per_group,)
                                        output_np[n, c_out_start:c_out_start + C_out_per_group, h_out, w_out] += input_val * weight_vals
        
        output.impl_(output_np)
    
    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        """VJP rule - to be implemented after testing."""
        raise NotImplementedError("VJP for conv2d_transpose will be implemented after testing")
    
    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
        """JVP rule - to be implemented after testing."""
        raise NotImplementedError("JVP for conv2d_transpose will be implemented after testing")


def conv2d_transpose(
    input_tensor: Tensor,
    weight: Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple = 0,
    output_padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> Tensor:
    """2D transposed convolution with PyTorch-compatible NCHW layout.
    
    Applies a 2D transposed convolution (also called deconvolution) over an input tensor.
    This function matches PyTorch's F.conv_transpose2d exactly.
    
    Parameters
    ----------
    input_tensor : Tensor
        Input tensor of shape (N, C_in, H, W)
    weight : Tensor
        Convolution kernel of shape (C_in, C_out/groups, K_H, K_W)
        Note: Different from conv2d! First dim is input channels.
    stride : int or tuple, optional
        Stride of the convolution. Default: 1
    padding : int or tuple, optional
        Padding added to input. Can be:
        - int: same padding on all sides
        - (pad_h, pad_w): symmetric padding
        Default: 0
    output_padding : int or tuple, optional
        Additional size added to output shape. Default: 0
    dilation : int or tuple, optional
        Spacing between kernel elements. Default: 1
    groups : int, optional
        Number of blocked connections. Default: 1
        
    Returns
    -------
    Tensor
        Output tensor of shape (N, C_out, H_out, W_out)
        
    Examples
    --------
    >>> import nabla as nb
    >>> # Simple 2D transposed convolution
    >>> x = nb.zeros((1, 64, 8, 8))    # NCHW
    >>> w = nb.zeros((64, 3, 3, 3))    # (in_ch, out_ch, H, W) - note the order!
    >>> y = nb.conv2d_transpose(x, w, stride=2)
    >>> y.shape
    (1, 3, 17, 17)
    """
    # Normalize parameters
    norm_stride = _normalize_tuple(stride, 2, "stride")
    norm_dilation = _normalize_tuple(dilation, 2, "dilation")
    norm_output_padding = _normalize_tuple(output_padding, 2, "output_padding")
    
    # For transpose conv, "same" padding doesn't make sense, so we don't support it
    if isinstance(padding, str):
        raise ValueError("String padding ('same', 'valid') not supported for conv2d_transpose")
    norm_padding = _normalize_padding(padding, "padding")
    
    # Cache operation instances for efficiency
    cache_key = (norm_stride, norm_padding, norm_output_padding, norm_dilation, groups)
    if cache_key not in _conv2d_transpose_op_cache:
        _conv2d_transpose_op_cache[cache_key] = Conv2DTransposeOp(
            stride=norm_stride,
            padding=norm_padding,
            output_padding=norm_output_padding,
            dilation=norm_dilation,
            groups=groups,
        )
    
    op = _conv2d_transpose_op_cache[cache_key]
    return op.forward(input_tensor, weight)


# # --- Old commented code below this line ---

# # --- Commented placeholder for conv2d_transpose ---
# # This will be implemented after conv2d is tested

# class Conv2DTransposeOp(Operation):
#     """2D Transposed Convolution - to be implemented."""
#     pass

# def conv2d_transpose(...):
#     """2D transposed convolution - to be implemented."""
#     pass


# # --- Old commented code below this line ---

# def flip(x: Tensor, axis: int | tuple[int, ...]) -> Tensor:
#     if isinstance(value, int):
#         return (value,) * n
#     elif isinstance(value, tuple | list):
#         if len(value) == n:
#             return tuple(value)
#         else:
#             raise ValueError(
#                 f"{name} must be an int or a tuple of {n} ints, got {value}"
#             )
#     else:
#         raise TypeError(
#             f"{name} must be an int or a tuple, got {type(value)} for {name}"
#         )


# def _normalize_padding_arg(padding_arg, name="padding"):
#     if isinstance(padding_arg, int):  # single int for all sides
#         return ((padding_arg, padding_arg), (padding_arg, padding_arg))
#     if isinstance(padding_arg, tuple | list):
#         if len(padding_arg) == 2:
#             if all(isinstance(x, int) for x in padding_arg):  # (symmetric_H, symmetric_W)
#                 ph, pw = padding_arg
#                 return ((ph, ph), (pw, pw))
#             elif all(isinstance(x, tuple | list) and len(x) == 2 and all(isinstance(y, int) for y in x) for x in padding_arg):
#                 # ((H_top, H_bottom), (W_left, W_right))
#                 return tuple(map(tuple, padding_arg))
#         elif len(padding_arg) == 4 and all(isinstance(x, int) for x in padding_arg):
#             # (H_top, H_bottom, W_left, W_right)
#             pt, pb, pl, pr = padding_arg
#             return ((pt, pb), (pl, pr))
#     raise ValueError(
#         f"{name} format is not recognized. Use int, (ph,pw), (pt,pb,pl,pr), or ((pt,pb),(pl,pr)). Got {padding_arg}"
#     )


# def flip(x: Tensor, axis: int | tuple[int, ...]) -> Tensor:
#     """
#     Reverses the order of elements in an tensor along the given axes.
#     This is an implementation of np.flip using fundamental slicing.
#     """
#     if isinstance(axis, int):
#         axes_to_flip = (axis,)
#     else:
#         axes_to_flip = axis

#     # Create a list of slice(None) objects, one for each dimension
#     slicer = [slice(None)] * len(x.shape)

#     # For each axis to be flipped, set the corresponding slice to ::-1
#     for ax in axes_to_flip:
#         slicer[ax] = slice(None, None, -1)

#     # Use tuple slicing on the tensor. The Nabla Tensor class's __getitem__
#     # must support this to be Python-idiomatic.
#     return x[tuple(slicer)]

# def _conv2d_filter_gradient(
#     x: Tensor, dy: Tensor, stride: tuple, dilation: tuple, padding: tuple, groups: int
# ) -> Tensor:
#     """
#     Computes `grad_W = conv(permute(x), permute(dy))` for a standard conv2d.
#     Returns a filter gradient in HWIO layout.
#     """
#     from ..ops import view

#     # Permute input x (NHWC) to be the data for the new conv: (Cin, H, W, N)
#     x_perm = view.transpose(x, (3, 1, 2, 0))

#     # Permute grad_output dy (NH'W'Cout) to be the filter for the new conv: (H', W', N, Cout)
#     dy_perm = view.transpose(dy, (1, 2, 0, 3))

#     # The new convolution's stride is the original's dilation, and vice versa.
#     # This is a standard identity for this gradient formulation.
#     grad_filter_permuted = conv2d(
#         x_perm, dy_perm, stride=dilation, dilation=stride, padding=padding, groups=groups
#     )

#     # The output is (Cin, kH, kW, Cout). Permute back to standard filter layout.
#     return view.transpose

# class Conv2DOp(BinaryOperation):
#     # ... This class is likely correct, but its VJP depends on the functions below ...
#     # Keep the version from my previous answer. The key fix is in Conv2DTransposeOp's VJP.
#     # ... For completeness, I'll include it with the corrected VJP rule call ...
#     """2D Convolution operation.

#     Data Layout: NHWC (batch, height, width, in_channels)
#     Filter Layout: HWIO (height, width, in_channels/groups, out_channels)
#     """

#     def __init__(self, stride, dilation, padding, groups):
#         super().__init__("conv2d")
#         self.stride = stride
#         self.dilation = dilation
#         self.padding = padding
#         self.groups = groups

#     def compute_output_shape(self, *input_shapes: tuple) -> tuple:
#         input_shape, filter_shape = input_shapes
#         n, h_in, w_in, c_in = input_shape
#         k_h, k_w, f_cin_div_g, f_cout = filter_shape

#         if c_in != f_cin_div_g * self.groups:
#             raise ValueError(
#                 f"Input channels ({c_in}) must match filter's effective input channels "
#                 f"({f_cin_div_g} * {self.groups} groups = {f_cin_div_g * self.groups}). "
#                 f"Input shape: {input_shape}, Filter shape: {filter_shape}"
#             )

#         (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right) = self.padding
#         dil_h, dil_w = self.dilation
#         s_h, s_w = self.stride

#         h_out = (h_in + pad_h_top + pad_h_bottom - dil_h * (k_h - 1) - 1) // s_h + 1
#         w_out = (w_in + pad_w_left + pad_w_right - dil_w * (k_w - 1) - 1) // s_w + 1
#         c_out = f_cout

#         if h_out <= 0 or w_out <= 0:
#             raise ValueError(f"Computed non-positive output dimensions for Conv2D: {(n, h_out, w_out, c_out)}")

#         return (n, h_out, w_out, c_out)

#     def forward(self, *args: Tensor) -> Tensor:
#         # Standard forward pass logic
#         from .operation import move_to_best_device
#         input_arr, filter_arr = move_to_best_device(*args)
#         self._validate_inputs(input_arr, filter_arr)

#         output_shape = self.compute_output_shape(input_arr.shape, filter_arr.shape)
#         res = Tensor(
#             shape=output_shape, dtype=self.compute_output_dtype(input_arr, filter_arr),
#             device=input_arr.logical_device, materialize=False, name=self.name,
#             batch_dims=input_arr.batch_dims,
#         )
#         res.set_maxpr(self.maxpr)
#         res.add_arguments(input_arr, filter_arr)
#         res.vjp_rule = self.vjp_rule
#         res.jvp_rule = self.jvp_rule
#         if not res.stage_realization:
#             self.eagerxpr([input_arr, filter_arr], res)
#         return res

#     def _validate_inputs(self, input_arr: Tensor, filter_arr: Tensor) -> None:
#         if len(input_arr.shape) != 4 or len(filter_arr.shape) != 4:
#             raise ValueError("Conv2D requires 4D input and filter tensors.")
#         if input_arr.logical_device != filter_arr.logical_device:
#             raise ValueError(f"Devices {input_arr.logical_device} and {filter_arr.logical_device} are incompatible")

#     def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
#         input_val, filter_val = args
#         (pt, pb), (pl, pr) = self.padding
#         output.tensor_value = ops.conv2d(
#             x=input_val, filter=filter_val, stride=self.stride,
#             dilation=self.dilation, padding=(pt, pb, pl, pr), groups=self.groups
#         )

#     def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
#         input_arr, filter_arr = args
#         input_torch = torch.from_numpy(np.transpose(input_arr.to_numpy(), (0, 3, 1, 2)))
#         filter_torch = torch.from_numpy(np.transpose(filter_arr.to_numpy(), (3, 2, 0, 1)))
#         (pad_h, _), (pad_w, _) = self.padding
#         result_torch = F.conv2d(
#             input=input_torch, weight=filter_torch, bias=None, stride=self.stride,
#             padding=(pad_h, pad_w), dilation=self.dilation, groups=self.groups
#         )
#         result_nhwc = np.transpose(result_torch.numpy(), (0, 2, 3, 1))
#         output.impl_(result_nhwc)

#     def vjp_rule(self, primals: list[Tensor], cotangent: Tensor, output: Tensor) -> list[Tensor]:
#         """VJP of Y = conv(X, W)"""
#         input_arr, filter_arr = primals # filter_arr is HWIO

#         # 1. grad_input = conv_transpose(dY, W_flipped_180)
#         flipped_filter = flip(filter_arr, axis=(0, 1))
#         # Filter for conv_transpose must be HWOI. Swap channels of our HWIO filter.
#         filter_for_grad_input = flipped_filter.transpose((0, 1, 3, 2))

#         # Calculate output_padding to restore original input shape
#         h_in, w_in = input_arr.shape[1:3]
#         h_out, w_out = cotangent.shape[1:3]
#         k_h, k_w = filter_arr.shape[0:2]
#         (pt, pb), (pl, pr) = self.padding
#         sh, sw = self.stride
#         dh, dw = self.dilation
#         out_pad_h = h_in - ((h_out - 1) * sh - (pt + pb) + (k_h - 1) * dh + 1)
#         out_pad_w = w_in - ((w_out - 1) * sw - (pl + pr) + (k_w - 1) * dw + 1)

#         grad_input = conv2d_transpose(
#             cotangent, filter_for_grad_input, stride=self.stride, dilation=self.dilation,
#             padding=self.padding, output_padding=(max(0,out_pad_h), max(0,out_pad_w)), groups=self.groups
#         )

#         # 2. grad_filter = conv(permute(X), permute(dY))
#         grad_filter = _conv2d_filter_gradient(
#             input_arr, cotangent, self.stride, self.dilation, self.padding, self.groups
#         )

#         return [grad_input, grad_filter]


#     def jvp_rule(self, primals: list[Tensor], tangents: list[Tensor], output: Tensor) -> Tensor:
#         input_arr, filter_arr = primals
#         input_tangent, filter_tangent = tangents
#         from .binary import add
#         res1 = conv2d(input_tangent, filter_arr, stride=self.stride, dilation=self.dilation, padding=self.padding, groups=self.groups)
#         res2 = conv2d(input_arr, filter_tangent, stride=self.stride, dilation=self.dilation, padding=self.padding, groups=self.groups)
#         return add(res1, res2)

# class Conv2DTransposeOp(BinaryOperation):
#     # This is the class with the key fixes.
#     # ... (Keep the __init__, compute_output_shape, forward, _validate_inputs, maxpr, eagerxpr from my PREVIOUS answer)
#     # The only change is in the VJP RULE.

#     def __init__(self, stride, dilation, padding, output_padding, groups):
#         super().__init__("conv2d_transpose")
#         self.stride = stride
#         self.dilation = dilation
#         self.padding = padding
#         self.output_padding = output_padding
#         self.groups = groups

#     def compute_output_shape(self, *input_shapes: tuple) -> tuple:
#         input_shape, filter_shape = input_shapes
#         n, h_in, w_in, c_in = input_shape
#         k_h, k_w, f_cout, f_cin_div_g = filter_shape
#         if c_in != f_cin_div_g * self.groups:
#              raise ValueError(
#                 f"Input channels ({c_in}) must match filter's effective input channels "
#                 f"({f_cin_div_g} * {self.groups} groups = {f_cin_div_g * self.groups}). "
#                 f"This is the 'I' in HWOI. "
#                 f"Input shape: {input_shape}, Filter shape: {filter_shape}"
#             )
#         (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right) = self.padding
#         out_pad_h, out_pad_w = self.output_padding
#         dil_h, dil_w = self.dilation
#         s_h, s_w = self.stride
#         h_out = (h_in - 1) * s_h - (pad_h_top + pad_h_bottom) + dil_h * (k_h - 1) + 1 + out_pad_h
#         w_out = (w_in - 1) * s_w - (pad_w_left + pad_w_right) + dil_w * (k_w - 1) + 1 + out_pad_w
#         c_out = f_cout
#         if h_out <= 0 or w_out <= 0:
#             raise ValueError(f"Computed non-positive output dimensions for Conv2DTranspose: {(n, h_out, w_out, c_out)}")
#         return (n, h_out, w_out, c_out)

#     def forward(self, *args: Tensor) -> Tensor:
#         from .operation import move_to_best_device
#         input_arr, filter_arr = move_to_best_device(*args)
#         self._validate_inputs(input_arr, filter_arr)
#         output_shape = self.compute_output_shape(input_arr.shape, filter_arr.shape)
#         res = Tensor(
#             shape=output_shape, dtype=self.compute_output_dtype(input_arr, filter_arr),
#             device=input_arr.logical_device, materialize=False, name=self.name,
#             batch_dims=input_arr.batch_dims,
#         )
#         res.set_maxpr(self.maxpr)
#         res.add_arguments(input_arr, filter_arr)
#         res.vjp_rule = self.vjp_rule
#         res.jvp_rule = self.jvp_rule
#         if not res.stage_realization:
#             self.eagerxpr([input_arr, filter_arr], res)
#         return res

#     def _validate_inputs(self, input_arr: Tensor, filter_arr: Tensor) -> None:
#         if len(input_arr.shape) != 4 or len(filter_arr.shape) != 4:
#             raise ValueError("Conv2DTranspose requires 4D input and filter tensors.")
#         if input_arr.logical_device != filter_arr.logical_device:
#             raise ValueError(f"Devices {input_arr.logical_device} and {filter_arr.logical_device} are incompatible")

#     def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
#         input_val, filter_val = args
#         (pt, pb), (pl, pr) = self.padding
#         if self.groups > 1:
#             from ..ops.view import split, concatenate
#             input_chunks = split(input_val, self.groups, axis=3)
#             filter_chunks = split(filter_val, self.groups, axis=3)
#             output_chunks = []
#             for i in range(self.groups):
#                 chunk_out = ops.conv2d_transpose(
#                     input_chunks[i], filter_chunks[i], stride=self.stride,
#                     dilation=self.dilation, padding=(pt, pb, pl, pr),
#                     output_paddings=self.output_padding
#                 )
#                 output_chunks.append(chunk_out)
#             output.tensor_value = concatenate(output_chunks, axis=3)
#         else:
#             output.tensor_value = ops.conv2d_transpose(
#                 input_val, filter_val, stride=self.stride, dilation=self.dilation,
#                 padding=(pt, pb, pl, pr), output_paddings=self.output_padding
#             )

#     def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
#         input_arr, filter_arr = args
#         input_torch = torch.from_numpy(np.transpose(input_arr.to_numpy(), (0, 3, 1, 2)))
#         filter_torch = torch.from_numpy(np.transpose(filter_arr.to_numpy(), (3, 2, 0, 1)))
#         (pad_h, _), (pad_w, _) = self.padding
#         result_torch = F.conv_transpose2d(
#             input=input_torch, weight=filter_torch, bias=None, stride=self.stride,
#             padding=(pad_h, pad_w), output_padding=self.output_padding,
#             groups=self.groups, dilation=self.dilation
#         )
#         result_nhwc = np.transpose(result_torch.numpy(), (0, 2, 3, 1))
#         output.impl_(result_nhwc)

#     def vjp_rule(self, primals: list[Tensor], cotangent: Tensor, output: Tensor) -> list[Tensor]:
#         """VJP of Y = conv_transpose(X, W)"""
#         input_arr, filter_arr = primals # filter_arr is HWOI

#         # 1. grad_input = conv(dY, W_flipped_180)
#         flipped_filter = flip(filter_arr, axis=(0, 1))
#         # Filter for conv2d must be HWIO. Swap channels of our HWOI filter.
#         filter_for_grad_input = flipped_filter.transpose((0, 1, 3, 2))

#         grad_input = conv2d(
#             cotangent, filter_for_grad_input, stride=self.stride,
#             dilation=self.dilation, padding=self.padding, groups=self.groups
#         )

#         # 2. grad_filter = conv(permute(dY), permute(X))
#         # Note the swapped arguments compared to the conv2d VJP.
#         grad_filter_HWIO = _conv2d_filter_gradient(
#             cotangent, input_arr, self.stride, self.dilation, self.padding, self.groups
#         )

#         # The helper returns HWIO. The gradient must match the primal filter's HWOI layout.
#         grad_filter = grad_filter_HWIO.transpose((0, 1, 3, 2))

#         return [grad_input, grad_filter]

#     def jvp_rule(self, primals: list[Tensor], tangents: list[Tensor], output: Tensor) -> Tensor:
#         input_arr, filter_arr = primals
#         input_tangent, filter_tangent = tangents
#         from .binary import add
#         res1 = conv2d_transpose(
#             input_tangent, filter_arr, stride=self.stride, dilation=self.dilation,
#             padding=self.padding, output_padding=self.output_padding, groups=self.groups)
#         res2 = conv2d_transpose(
#             input_arr, filter_tangent, stride=self.stride, dilation=self.dilation,
#             padding=self.padding, output_padding=self.output_padding, groups=self.groups)
#         return add(res1, res2)

# def conv2d(
#     input_arr: Tensor, filter_arr: Tensor, stride=(1, 1),
#     dilation=(1, 1), padding=0, groups=1
# ) -> Tensor:
#     """Applies a 2D convolution."""
#     norm_stride = _normalize_tuple(stride, 2, "stride")
#     norm_dilation = _normalize_tuple(dilation, 2, "dilation")
#     norm_padding = _normalize_padding_arg(padding, "padding")

#     cache_key = (norm_stride, norm_dilation, norm_padding, groups)
#     if cache_key not in _conv2d_op_cache:
#         _conv2d_op_cache[cache_key] = Conv2DOp(norm_stride, norm_dilation, norm_padding, groups)
#     op = _conv2d_op_cache[cache_key]
#     return op.forward(input_arr, filter_arr)


# def conv2d_transpose(
#     input_arr: Tensor, filter_arr: Tensor, stride=(1, 1),
#     dilation=(1, 1), padding=0, output_padding=0, groups=1
# ) -> Tensor:
#     """Applies a 2D transposed convolution."""
#     norm_stride = _normalize_tuple(stride, 2, "stride")
#     norm_dilation = _normalize_tuple(dilation, 2, "dilation")
#     norm_padding = _normalize_padding_arg(padding, "padding")
#     norm_output_padding = _normalize_tuple(output_padding, 2, "output_padding")

#     cache_key = (norm_stride, norm_dilation, norm_padding, norm_output_padding, groups)
#     if cache_key not in _conv2d_transpose_op_cache:
#         _conv2d_transpose_op_cache[cache_key] = Conv2DTransposeOp(
#             norm_stride, norm_dilation, norm_padding, norm_output_padding, groups
#         )
#     op = _conv2d_transpose_op_cache[cache_key]
#     return op.forward(input_arr, filter_arr)


# # ===----------------------------------------------------------------------=== #
# # Nabla 2025
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ===----------------------------------------------------------------------=== #

# """Numpy-based convolution utilities for eager execution."""

# from typing import Union

# import numpy as np


# def im2col(
#     input_data: np.ndarray,
#     filter_h: int,
#     filter_w: int,
#     stride: Union[int, tuple[int, int]] = 1,
#     dilation: Union[int, tuple[int, int]] = 1,
#     pad: Union[int, tuple[int, int]] = 0,
# ) -> np.ndarray:
#     """
#     Convert input data to column matrix for convolution.

#     Parameters:
#     -----------
#     input_data : ndtensor
#         Input data with shape (N, C, H, W)
#     filter_h : int
#         Filter height
#     filter_w : int
#         Filter width
#     stride : int or tuple
#         Stride for convolution
#     dilation : int or tuple
#         Dilation for convolution
#     pad : int or tuple
#         Padding for input

#     Returns:
#     --------
#     col : ndtensor
#         Column matrix with shape (N, C, filter_h, filter_w, out_h, out_w)
#     """
#     n, c, h, w = input_data.shape

#     # Handle stride and dilation as tuples
#     if isinstance(stride, int):
#         stride_h, stride_w = stride, stride
#     else:
#         stride_h, stride_w = stride

#     if isinstance(dilation, int):
#         dilation_h, dilation_w = dilation, dilation
#     else:
#         dilation_h, dilation_w = dilation

#     if isinstance(pad, int):
#         pad_h, pad_w = pad, pad
#     else:
#         pad_h, pad_w = pad

#     out_h = (h + 2 * pad_h - dilation_h * (filter_h - 1) - 1) // stride_h + 1
#     out_w = (w + 2 * pad_w - dilation_w * (filter_w - 1) - 1) // stride_w + 1

#     img = np.pad(
#         input_data, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
#     )
#     col = np.ndarray((n, c, filter_h, filter_w, out_h, out_w), dtype=input_data.dtype)

#     for j in range(filter_h):
#         j_lim = j * dilation_h + stride_h * out_h
#         for i in range(filter_w):
#             i_lim = i * dilation_w + stride_w * out_w
#             col[:, :, j, i, :, :] = img[
#                 :,
#                 :,
#                 j * dilation_h : j_lim : stride_h,
#                 i * dilation_w : i_lim : stride_w,
#             ]

#     return col


# def col2im(
#     col: np.ndarray,
#     input_shape: tuple[int, int, int, int],
#     filter_h: int,
#     filter_w: int,
#     stride: Union[int, tuple[int, int]] = 1,
#     dilation: Union[int, tuple[int, int]] = 1,
#     pad: Union[int, tuple[int, int]] = 0,
# ) -> np.ndarray:
#     """
#     Convert column matrix back to input data shape.

#     Parameters:
#     -----------
#     col : ndtensor
#         Column matrix with shape (N, C, filter_h, filter_w, out_h, out_w)
#     input_shape : tuple
#         Original input shape (N, C, H, W)
#     filter_h : int
#         Filter height
#     filter_w : int
#         Filter width
#     stride : int or tuple
#         Stride for convolution
#     dilation : int or tuple
#         Dilation for convolution
#     pad : int or tuple
#         Padding for input

#     Returns:
#     --------
#     img : ndtensor
#         Reconstructed input data with shape (N, C, H, W)
#     """
#     n, c, h, w = input_shape

#     # Handle stride and dilation as tuples
#     if isinstance(stride, int):
#         stride_h, stride_w = stride, stride
#     else:
#         stride_h, stride_w = stride

#     if isinstance(dilation, int):
#         dilation_h, dilation_w = dilation, dilation
#     else:
#         dilation_h, dilation_w = dilation

#     if isinstance(pad, int):
#         pad_h, pad_w = pad, pad
#     else:
#         pad_h, pad_w = pad

#     out_h = (h + 2 * pad_h - dilation_h * (filter_h - 1) - 1) // stride_h + 1
#     out_w = (w + 2 * pad_w - dilation_w * (filter_w - 1) - 1) // stride_w + 1

#     img = np.zeros(
#         (n, c, h + 2 * pad_h + stride_h - 1, w + 2 * pad_w + stride_w - 1),
#         dtype=col.dtype,
#     )

#     for j in range(filter_h):
#         j_lim = j * dilation_h + stride_h * out_h
#         for i in range(filter_w):
#             i_lim = i * dilation_w + stride_w * out_w
#             img[
#                 :,
#                 :,
#                 j * dilation_h : j_lim : stride_h,
#                 i * dilation_w : i_lim : stride_w,
#             ] += col[:, :, j, i, :, :]

#     return img[:, :, pad_h : h + pad_h, pad_w : w + pad_w]


# def conv2d(input_data, filters, dilation=(1, 1), stride=(1, 1), padding=(0, 0)):
#     """
#     2D convolution using im2col method.

#     Parameters:
#     -----------
#     input_data : ndtensor
#         Input data with shape (N, C_in, H, W)
#     filters : ndtensor
#         Filters with shape (C_out, C_in, filter_h, filter_w)
#     dilation : tuple
#         Dilation factors (dilation_h, dilation_w)
#     stride : tuple
#         Stride values (stride_h, stride_w)
#     padding : tuple
#         Padding values (pad_h, pad_w)

#     Returns:
#     --------
#     output : ndtensor
#         Convolution output with shape (N, C_out, out_h, out_w)
#     """
#     n, c_in, h, w = input_data.shape
#     c_out, c_in_f, filter_h, filter_w = filters.shape

#     assert c_in == c_in_f, f"Input channels {c_in} != filter input channels {c_in_f}"

#     # Calculate output dimensions
#     pad_h, pad_w = padding
#     stride_h, stride_w = stride
#     dilation_h, dilation_w = dilation

#     out_h = (h + 2 * pad_h - dilation_h * (filter_h - 1) - 1) // stride_h + 1
#     out_w = (w + 2 * pad_w - dilation_w * (filter_w - 1) - 1) // stride_w + 1

#     # Convert input to column matrix
#     col = im2col(input_data, filter_h, filter_w, stride, dilation, padding)
#     col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)

#     # Reshape filters
#     w_col = filters.reshape(c_out, -1)

#     # Perform convolution via matrix multiplication
#     out = np.dot(col, w_col.T)
#     out = out.reshape(n, out_h, out_w, c_out).transpose(0, 3, 1, 2)

#     return out


# def transposed_conv2d(
#     input_data,
#     filters,
#     dilation=(1, 1),
#     stride=(1, 1),
#     padding=(0, 0),
#     output_padding=(0, 0),
# ):
#     """
#     2D transposed convolution using JAX-compatible algorithm.

#     JAX's conv_transpose implementation:
#     1. Upsample input by inserting (stride-1) zeros between elements
#     2. Apply regular convolution with effective padding

#     For transposed convolution, the effective padding is:
#     effective_pad = kernel_size - 1 - original_pad

#     Parameters:
#     -----------
#     input_data : ndtensor
#         Input data with shape (N, C_in, H, W)
#     filters : ndtensor
#         Filters with shape (C_out, C_in, filter_h, filter_w)
#     dilation : tuple
#         Dilation factors (dilation_h, dilation_w)
#     stride : tuple
#         Stride values (stride_h, stride_w)
#     padding : tuple
#         Original padding values (pad_h, pad_w) from the forward convolution
#     output_padding : tuple
#         Output padding values (out_pad_h, out_pad_w) - not used in JAX-compatible mode

#     Returns:
#     --------
#     output : ndtensor
#         Transposed convolution output
#     """
#     n, c_in, h, w = input_data.shape
#     c_out, c_in_f, filter_h, filter_w = filters.shape

#     assert c_in == c_in_f, f"Input channels {c_in} != filter input channels {c_in_f}"

#     pad_h, pad_w = padding
#     stride_h, stride_w = stride
#     dilation_h, dilation_w = dilation

#     # Step 1: Upsample input by inserting (stride-1) zeros between elements
#     if stride_h > 1 or stride_w > 1:
#         # Calculate upsampled dimensions
#         upsampled_h = h + (h - 1) * (stride_h - 1)
#         upsampled_w = w + (w - 1) * (stride_w - 1)

#         # Create upsampled tensor filled with zeros
#         upsampled = np.zeros(
#             (n, c_in, upsampled_h, upsampled_w), dtype=input_data.dtype
#         )

#         # Insert original values at strided positions
#         upsampled[:, :, ::stride_h, ::stride_w] = input_data
#     else:
#         # No upsampling needed for stride=1
#         upsampled = input_data

#     # Step 2: Calculate effective padding for transposed convolution
#     # For transposed conv, if original conv had padding P and kernel size K,
#     # the effective padding for the underlying regular conv is (K-1-P)
#     effective_pad_h = filter_h - 1 - pad_h
#     effective_pad_w = filter_w - 1 - pad_w

#     # Step 3: Apply regular convolution with effective padding
#     # Use stride=1 since upsampling already handled the stride effect
#     result = conv2d(
#         upsampled,
#         filters,
#         dilation=dilation,
#         stride=(1, 1),
#         padding=(effective_pad_h, effective_pad_w),
#     )

#     # Step 4: Apply output_padding if specified
#     # Output padding adds zeros to the right and bottom of the output
#     out_pad_h, out_pad_w = output_padding
#     if out_pad_h > 0 or out_pad_w > 0:
#         n, c_out, h_out, w_out = result.shape
#         padded_result = np.zeros(
#             (n, c_out, h_out + out_pad_h, w_out + out_pad_w), dtype=result.dtype
#         )
#         padded_result[:, :, :h_out, :w_out] = result
#         result = padded_result

#     return result
