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
"""Special functions for neural networks."""

from collections.abc import Callable

from ..core.tensor import Tensor


# from __future__ import annotations

from pathlib import Path

from max.graph.dim import DimLike, StaticDim
from max.graph.shape import Shape
from max.graph.type import TensorType
from max.graph.value import TensorValue, TensorValueLike
from max.graph.ops import custom
from max.graph.ops import shape_to_tensor
from .operation import Operation
import numpy as np
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops

# Public API
__all__ = ["softmax", "logsumexp", "where", "cond", "fold", "unfold"]


def logsumexp(arg: Tensor, axis: int | None = None, keep_dims: bool = False) -> Tensor:
    """Computes the log of the sum of exponentials of input elements.

    This function computes `log(sum(exp(x)))` in a numerically stable way by using
    the identity: `logsumexp(x) = max(x) + log(sum(exp(x - max(x))))`. This
    avoids overflow errors that can occur when `exp(x)` is very large.

    Parameters
    ----------
    arg : Tensor
        The input tensor.
    axis : int | None, optional
        The axis or axes along which to compute the `logsumexp`. If None (the
        default), the operation is performed over all elements of the tensor.
    keep_dims : bool, optional
        If True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast
        correctly against the input tensor. Defaults to False.

    Returns
    -------
    Tensor
        An tensor containing the result of the `logsumexp` operation.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([1.0, 2.0, 3.0])
    >>> nb.logsumexp(x)
    Tensor([3.407606], dtype=float32)

    >>> data = nb.tensor([[1, 2, 3], [4, 5, 6]])
    >>> nb.logsumexp(data, axis=1)
    Tensor([3.407606, 6.407606], dtype=float32)
    """
    from .binary import add, sub
    from .reduce import max as tensor_max
    from .reduce import sum as tensor_sum
    from .unary import exp, log

    # For numerical stability, subtract the max before exponentiating
    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))

    # Find max along specified axis, keeping dimensions for broadcasting
    x_max = tensor_max(arg, axes=axis, keep_dims=True)

    # Subtract max and exponentiate
    shifted = sub(arg, x_max)
    exp_shifted = exp(shifted)

    # Sum and take log
    sum_exp = tensor_sum(exp_shifted, axes=axis, keep_dims=True)
    log_sum_exp = log(sum_exp)

    # Add back the max
    result = add(x_max, log_sum_exp)

    # Remove extra dimensions if not keeping them
    if not keep_dims and axis is not None:
        from .view import squeeze

        axes_to_squeeze = [axis] if isinstance(axis, int) else list(axis)

        for ax in sorted(axes_to_squeeze, reverse=True):
            result = squeeze(result, [ax])  # Pass as list

    return result


def softmax(arg: Tensor, axis: int = -1) -> Tensor:
    """Computes the softmax function for an tensor.

    The softmax function transforms a vector of real numbers into a probability
    distribution. Each element in the output is in the range (0, 1), and the
    elements along the specified axis sum to 1. It is calculated in a
    numerically stable way as `exp(x - logsumexp(x))`.

    Parameters
    ----------
    arg : Tensor
        The input tensor.
    axis : int, optional
        The axis along which the softmax computation is performed. The default
        is -1, which is the last axis.

    Returns
    -------
    Tensor
        An tensor of the same shape as the input, containing the softmax
        probabilities.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([1.0, 2.0, 3.0])
    >>> nb.softmax(x)
    Tensor([0.09003057, 0.24472848, 0.66524094], dtype=float32)

    >>> logits = nb.tensor([[1, 2, 3], [1, 1, 1]])
    >>> nb.softmax(logits, axis=1)
    Tensor([[0.09003057, 0.24472848, 0.66524094],
           [0.33333334, 0.33333334, 0.33333334]], dtype=float32)
    """
    from .binary import sub
    from .unary import exp

    # For numerical stability: softmax(x) = exp(x - logsumexp(x))
    log_sum_exp = logsumexp(arg, axis=axis, keep_dims=True)

    # Compute softmax: exp(x - logsumexp(x))
    normalized = sub(arg, log_sum_exp)
    return exp(normalized)


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Selects elements from two tensors based on a condition.

    This function returns an tensor with elements chosen from `x` where the
    corresponding element in `condition` is True, and from `y` otherwise.
    The function supports broadcasting among the three input tensors.

    Parameters
    ----------
    condition : Tensor
        A boolean tensor. Where True, yield `x`, otherwise yield `y`.
    x : Tensor
        The tensor from which to take values when `condition` is True.
    y : Tensor
        The tensor from which to take values when `condition` is False.

    Returns
    -------
    Tensor
        An tensor with elements from `x` and `y`, depending on `condition`.

    Examples
    --------
    >>> import nabla as nb
    >>> condition = nb.tensor([True, False, True])
    >>> x = nb.tensor([1, 2, 3])
    >>> y = nb.tensor([10, 20, 30])
    >>> nb.where(condition, x, y)
    Tensor([1, 20, 3], dtype=int32)

    Broadcasting example:
    >>> nb.where(nb.tensor([True, False]), nb.tensor(5), nb.tensor([10, 20]))
    Tensor([5, 20], dtype=int32)
    """
    from .binary import add, mul
    from .unary import cast, logical_not

    # where(c, x, y) = c * x + (1 - c) * y
    # Convert boolean condition to float for arithmetic
    cond_float = cast(condition, x.dtype)
    inv_cond = cast(logical_not(condition), x.dtype)

    x_part = mul(cond_float, x)
    y_part = mul(inv_cond, y)

    return add(x_part, y_part)


def cond(
    condition: Tensor, true_fn: Callable, false_fn: Callable, *args, **kwargs
) -> Tensor:
    """Conditionally executes one of two functions.

    If `condition` is True, `true_fn` is called; otherwise, `false_fn` is
    called. This is a control-flow primitive that allows for conditional
    execution within a computational graph. Unlike `nabla.where`, which
    evaluates both branches, `cond` only executes the selected function.

    Parameters
    ----------
    condition : Tensor
        A scalar boolean tensor that determines which function to execute.
    true_fn : Callable
        The function to be called if `condition` is True.
    false_fn : Callable
        The function to be called if `condition` is False.
    *args
        Positional arguments to be passed to the selected function.
    **kwargs
        Keyword arguments to be passed to the selected function.

    Returns
    -------
    Tensor
        The result of calling either `true_fn` or `false_fn`.

    Examples
    --------
    >>> import nabla as nb
    >>> def f(x):
    ...     return x * 2
    ...
    >>> def g(x):
    ...     return x + 10
    ...
    >>> x = nb.tensor(5)
    >>> # Executes g(x) because the condition is False
    >>> nb.cond(nb.tensor(False), f, g, x)
    Tensor([15], dtype=int32)
    """
    from max.dtype import DType

    from .unary import cast

    # Convert condition to boolean if necessary
    bool_condition = cast(condition, DType.bool)

    return where(bool_condition, true_fn(*args, **kwargs), false_fn(*args, **kwargs))


def fold_custom(
    input: TensorValueLike,
    output_size: tuple[DimLike, DimLike],
    kernel_size: tuple[DimLike, DimLike],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> TensorValue:
    """Combines an array of sliding blocks into a larger containing tensor.

    The input tensor must have shape ``(N, C * kernel_sizes, L)`` where ``N`` is
    the batch dimension, ``C`` is the number of channels, ``kernel_sizes`` is
    the product of the kernel sizes, and ``L`` is the number of local blocks.

    The resulting output tensor will have shape
    ``(N, C, output_shape[0], output_shape[1])``.

    ``L``, the number of blocks, must be equivalent to:
    ``prod((output_size[d] + 2 * padding[d] - dilation[d] * (kernel_size[d] - 1) - 1) / stride[d] + 1)``

    where ``d`` is over all spatial dimensions.

    Args:
        input: The 3D tensor to fold with shape ``(N, C * kernel sizes, L)``.
        output_size: Spatial dimensions of the output tensor. Must be a tuple of two ints.
        kernel_size: The size of the sliding blocks. Must be a tuple of two ints.
        stride: The stride of the sliding blocks in the input dimension
            (can be an int or a tuple of two ints).
        dilation: The spacing between the kernel elements.
            (can be an int or a tuple of two ints).
        padding: 0-paddings to be added on both sides of the inputs.
            (can be an int or a tuple of two ints).

    Returns:
        The folded 4D tensor with shape ``(N, C, output_shape[0], output_shape[1])``.
    """
    input = TensorValue(input)

    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    if not isinstance(padding, tuple):
        padding = (padding, padding)

    if isinstance(kernel_size[0], int) and isinstance(kernel_size[1], int):
        channels = input.shape[1] // (kernel_size[0] * kernel_size[1])
        output_shape = Shape(
            [input.shape[0], channels, output_size[0], output_size[1]]
        )
    else:
        output_shape = Shape(
            [input.shape[0], "channels", output_size[0], output_size[1]]
        )

    # Run early shape checks if the shapes are statically known.
    if isinstance(kernel_size[0], int) and isinstance(kernel_size[1], int):
        if (
            isinstance(input.shape[1], StaticDim)
            and int(input.shape[1]) % (kernel_size[0] * kernel_size[1]) != 0
        ):
            raise ValueError(
                f"Dim 1 of the input tensor ({input.shape[1]}) must be a multiple "
                "of the product of the total kernel size"
                f" ({kernel_size[0]} * {kernel_size[1]})."
            )

        if (
            isinstance(input.shape[2], StaticDim)
            and isinstance(output_size[0], int)
            and isinstance(output_size[1], int)
        ):
            L = 1
            for n, (o, k) in enumerate(
                zip(output_size, kernel_size, strict=True)
            ):
                L_d = int(
                    (int(o) + 2 * padding[n] - dilation[n] * (int(k) - 1) - 1)
                    // stride[n]
                    + 1
                )
                L *= L_d
            if int(input.shape[2]) != L:
                raise ValueError(
                    f"Last dimension of input tensor ({input.shape[2]}) must match "
                    f"the calculated number of blocks ({L})."
                )

    parameters: dict[str, int] = {
        "stride_h": stride[0],
        "stride_w": stride[1],
        "dilation_h": dilation[0],
        "dilation_w": dilation[1],
        "padding_h": padding[0],
        "padding_w": padding[1],
    }

    return custom(
        "fold_custom",
        input.device,
        [
            input,
            shape_to_tensor(output_size),
            shape_to_tensor(kernel_size),
        ],
        [TensorType(input.dtype, output_shape, input.device)],
        parameters=parameters,
    )[0].tensor



class FoldOp(Operation):
    """Fold operation that combines sliding blocks into a larger tensor.
    
    The fold operation is the inverse of unfold. It takes a 3D tensor of shape
    (N, C * kernel_sizes, L) and reconstructs a 4D tensor of shape
    (N, C, output_size[0], output_size[1]).
    """

    def __init__(self,
        output_size: tuple[DimLike, DimLike],
        kernel_size: tuple[DimLike, DimLike],
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
    ):
        super().__init__("custom_fold")
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, *args: Tensor) -> Tensor:
        """Forward pass for fold operation."""
        if len(args) != 1:
            raise ValueError(f"Fold operation requires 1 argument, got {len(args)}")
        arg = args[0]

        output_shape = self.compute_output_shape(arg.shape)
        output_batch_dims = self.compute_output_batch_dims(arg.batch_dims)
        output_dtype = self.compute_output_dtype(arg)

        res = Tensor(
            shape=output_shape,
            dtype=output_dtype,
            device=arg.logical_device,
            materialize=False,
            name=self.name,
            batch_dims=output_batch_dims,
        )

        res.set_maxpr(self.maxpr)
        res.add_arguments(arg)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule
        res.custom_kernel_path = self.custom_kernel_path()

        if not res.stage_realization:
            self.eagerxpr([arg], res)

        res.creator_op = self
        return res

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        output.tensor_value = fold_custom(
            args[0], 
            self.output_size, 
            self.kernel_size,
            self.stride, 
            self.dilation, 
            self.padding
        )

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        """Eager computation using NumPy for fold operation."""
        input_np = args[0].to_numpy()
        
        N, C_kernel, L = input_np.shape
        kH, kW = self.kernel_size
        OH, OW = self.output_size
        
        # Calculate number of channels
        C = C_kernel // (kH * kW)
        
        # Initialize output with zeros
        output_np = np.zeros((N, C, OH, OW), dtype=input_np.dtype)
        
        # Reshape input to (N, C, kH*kW, L)
        input_reshaped = input_np.reshape(N, C, kH * kW, L)
        
        # Iterate over each block
        block_idx = 0
        for h in range(0, OH - kH + 1 + 2 * self.padding[0], self.stride[0]):
            for w in range(0, OW - kW + 1 + 2 * self.padding[1], self.stride[1]):
                if block_idx >= L:
                    break
                
                # Extract the block from input
                block = input_reshaped[:, :, :, block_idx].reshape(N, C, kH, kW)
                
                # Add block to output at the correct position
                for kh in range(kH):
                    for kw in range(kW):
                        out_h = h - self.padding[0] + kh * self.dilation[0]
                        out_w = w - self.padding[1] + kw * self.dilation[1]
                        
                        if 0 <= out_h < OH and 0 <= out_w < OW:
                            output_np[:, :, out_h, out_w] += block[:, :, kh, kw]
                
                block_idx += 1
        
        output.impl_(output_np)

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compute output shape for fold operation."""
        if len(input_shapes) != 1:
            raise ValueError(f"Fold operation requires 1 input shape, got {len(input_shapes)}")
        
        input_shape = input_shapes[0]
        kH, kW = self.kernel_size
        
        return (
            input_shape[0],  # batch size
            input_shape[1] // (kH * kW),  # channels
            self.output_size[0],  # output height
            self.output_size[1],  # output width
        )

    def compute_output_dtype(self, arg: Tensor) -> DType:
        """Output dtype is same as input dtype."""
        return arg.dtype

    def compute_output_batch_dims(self, input_batch_dims: tuple[int, ...]) -> tuple[int, ...]:
        """Output batch dims are same as input batch dims."""
        return input_batch_dims

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        """VJP rule for fold operation.
        
        The gradient of fold w.r.t. its input is unfold applied to the cotangent.
        This is because fold and unfold are adjoint operations:
        - fold sums sliding blocks into output
        - unfold (adjoint) extracts gradients from output back to blocks
        
        Args:
            primals: [input] where input has shape (N, C*k*k, L)
            cotangent: gradient w.r.t. output with shape (N, C, H, W)
            output: the output tensor from forward pass
            
        Returns:
            [grad_input] where grad_input has shape (N, C*k*k, L)
        """
        # Apply unfold to the cotangent (gradient w.r.t. output)
        grad_input = unfold(
            cotangent,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.padding
        )
        return [grad_input,]

    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
        # raise NotImplementedError("JVP rule for FoldOp is not implemented.")
        return fold(
            tangents[0],
            output.shape,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.padding
        )

    def custom_kernel_path(self) -> Path | None:
        """Path to custom kernel implementation."""
        # Return path to the examples/custom_kernels directory where fold.mojo is located
        return Path(__file__).parent / "kernels"


def fold(arg: Tensor,
    output_size: tuple[DimLike, DimLike], 
    kernel_size: tuple[DimLike, DimLike],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> Tensor:
    """Fold operation wrapper."""
    return FoldOp(
        output_size,
        kernel_size,
        stride,
        dilation,
        padding,
    ).forward(
        arg,
    )


def unfold_custom(
    input: TensorValueLike,
    kernel_size: tuple[DimLike, DimLike],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> TensorValue:
    """Extracts sliding local blocks from a batched input tensor.

    The input tensor must have shape ``(N, C, H, W)`` where ``N`` is
    the batch dimension, ``C`` is the number of channels, and ``H`` and ``W``
    are the spatial dimensions.

    The resulting output tensor will have shape
    ``(N, C * kernel_sizes, L)`` where ``L`` is the number of blocks.

    Args:
        input: Input tensor of shape ``(N, C, H, W)``.
        kernel_size: Size of the sliding blocks ``(kH, kW)``.
        stride: Stride of the sliding blocks. Default: 1.
        dilation: Dilation of the sliding blocks. Default: 1.
        padding: Padding added to input. Default: 0.

    Returns:
        Output tensor of shape ``(N, C * kernel_sizes, L)``.
    """
    # Convert to graph tensors
    input = TensorValue(input)

    # Normalize parameters
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding)

    # Validate input shape
    if len(input.shape) != 4:
        raise ValueError(
            f"Expected 4D input (batch, channels, height, width), got {len(input.shape)}D"
        )

    # Extract input dimensions
    N = input.shape[0]
    C = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]

    # Calculate output dimensions
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    padding_h, padding_w = padding

    height_col = (
        int(H) + 2 * padding_h - dilation_h * (int(kernel_h) - 1) - 1
    ) // stride_h + 1
    width_col = (
        int(W) + 2 * padding_w - dilation_w * (int(kernel_w) - 1) - 1
    ) // stride_w + 1

    L = height_col * width_col
    output_shape = [N, int(C) * int(kernel_h) * int(kernel_w), L]

    # Validate calculated dimensions
    if height_col <= 0 or width_col <= 0:
        raise ValueError(
            f"Calculated output dimensions are invalid: height_col={height_col}, width_col={width_col}. "
            f"Check your input dimensions (H={H}, W={W}), kernel_size={kernel_size}, "
            f"stride={stride}, dilation={dilation}, padding={padding}"
        )

    parameters: dict[str, int] = {
        "stride_h": stride_h,
        "stride_w": stride_w,
        "dilation_h": dilation_h,
        "dilation_w": dilation_w,
        "padding_h": padding_h,
        "padding_w": padding_w,
    }

    return custom(
        "unfold_custom",
        input.device,
        [
            input,
            shape_to_tensor(kernel_size),
        ],
        [TensorType(input.dtype, output_shape, input.device)],
        parameters=parameters,
    )[0].tensor


class UnfoldOp(Operation):
    """Unfold operation that extracts sliding blocks from a larger tensor.
    
    The unfold operation is the inverse of fold. It takes a 4D tensor of shape
    (N, C, H, W) and extracts sliding blocks into a 3D tensor of shape
    (N, C * kernel_sizes, L) where L is the number of blocks.
    
    This operation is also known as im2col in the context of convolution operations.
    """

    def __init__(
        self,
        kernel_size: tuple[DimLike, DimLike],
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
    ):
        """Initialize the unfold operation.

        Args:
            kernel_size: Size of the sliding blocks (kH, kW).
            stride: Stride of the sliding blocks. Can be a single int or tuple (sH, sW).
            dilation: Dilation of the sliding blocks. Can be a single int or tuple (dH, dW).
            padding: Padding added to input. Can be a single int or tuple (pH, pW).
        """
        super().__init__("custom_unfold")
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation
            
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

    def compute_output_shape(self, input_shape: Shape) -> tuple[DimLike, ...]:
        """Compute the output shape of the unfold operation."""
        N = input_shape[0]
        C = input_shape[1]
        H = input_shape[2]
        W = input_shape[3]
        
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        dilation_h, dilation_w = self.dilation
        padding_h, padding_w = self.padding

        height_col = (
            int(H) + 2 * padding_h - dilation_h * (int(kernel_h) - 1) - 1
        ) // stride_h + 1
        width_col = (
            int(W) + 2 * padding_w - dilation_w * (int(kernel_w) - 1) - 1
        ) // stride_w + 1

        L = height_col * width_col
        
        return (N, int(C) * int(kernel_h) * int(kernel_w), L)

    def compute_output_dtype(self, *input_dtypes: DType) -> DType:
        """Compute output dtype."""
        return input_dtypes[0]

    def compute_output_batch_dims(self, *input_batch_dims: int) -> int:
        """Compute output batch dimensions."""
        return input_batch_dims[0]

    def forward(self, *args: Tensor) -> Tensor:
        """Forward pass of the unfold operation."""
        if len(args) != 1:
            raise ValueError(f"Unfold operation requires 1 argument, got {len(args)}")
        arg = args[0]

        output_shape = self.compute_output_shape(arg.shape)
        output_batch_dims = self.compute_output_batch_dims(arg.batch_dims)
        output_dtype = self.compute_output_dtype(arg.dtype)

        res = Tensor(
            shape=output_shape,
            dtype=output_dtype,
            device=arg.logical_device,
            materialize=False,
            name=self.name,
            batch_dims=output_batch_dims,
        )

        res.set_maxpr(self.maxpr)
        res.add_arguments(arg)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule
        res.custom_kernel_path = self.custom_kernel_path()

        if not res.stage_realization:
            self.eagerxpr([arg], res)

        res.creator_op = self
        return res

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        """MAX graph construction."""
        output.tensor_value = unfold_custom(
            args[0],
            self.kernel_size,
            self.stride,
            self.dilation,
            self.padding
        )

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        """Eager execution using NumPy."""
        # Get input as numpy array
        input_np = args[0].to_numpy()
        
        N, C, H, W = input_np.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        dilation_h, dilation_w = self.dilation
        padding_h, padding_w = self.padding
        
        # Add padding
        if padding_h > 0 or padding_w > 0:
            input_np = np.pad(
                input_np,
                ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)),
                mode='constant',
                constant_values=0
            )
            H_padded = H + 2 * padding_h
            W_padded = W + 2 * padding_w
        else:
            H_padded = H
            W_padded = W
        
        # Calculate output dimensions
        height_col = (H_padded - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        width_col = (W_padded - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        L = height_col * width_col
        
        # Create output array
        output_np = np.zeros(
            (N, C * kernel_h * kernel_w, L),
            dtype=input_np.dtype
        )
        
        # Extract patches
        for n in range(N):
            for c in range(C):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        c_out = c * kernel_h * kernel_w + kh * kernel_w + kw
                        
                        for h_col in range(height_col):
                            for w_col in range(width_col):
                                h_in = h_col * stride_h + kh * dilation_h
                                w_in = w_col * stride_w + kw * dilation_w
                                
                                block_idx = h_col * width_col + w_col
                                output_np[n, c_out, block_idx] = input_np[n, c, h_in, w_in]
        
        # Set output value
        output.impl_(output_np)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        """VJP rule for unfold operation.
        
        The gradient of unfold w.r.t. its input is fold applied to the cotangent.
        This is because fold and unfold are adjoint operations:
        - unfold extracts sliding blocks from input
        - fold (adjoint) sums gradients from blocks back to original positions
        
        Args:
            primals: [input] where input has shape (N, C, H, W)
            cotangent: gradient w.r.t. output with shape (N, C*k*k, L)
            output: the output tensor from forward pass
            
        Returns:
            [grad_input] where grad_input has shape (N, C, H, W)
        """
        # Extract original spatial dimensions from the input
        input_shape = primals[0].shape
        H = input_shape[2]
        W = input_shape[3]
        output_size = (H, W)
        
        # Apply fold to the cotangent (gradient w.r.t. output)
        # This sums the gradients back to their original spatial positions
        grad_input = fold(
            cotangent,
            output_size,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.padding
        )
        return [grad_input]

    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], node: Tensor
    ) -> Tensor:
        """Jacobian-vector product (forward-mode autodiff)."""
        # raise NotImplementedError("JVP rule for UnfoldOp is not implemented.")
        return unfold(
            primals[0],
            self.kernel_size,
            self.stride,
            self.dilation,
            self.padding
        )

    def custom_kernel_path(self) -> Path | None:
        """Path to custom kernel implementation."""
        return Path(__file__).parent / "kernels"


def unfold(
    arg: Tensor,
    kernel_size: tuple[DimLike, DimLike],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> Tensor:
    """Unfold operation wrapper."""
    return UnfoldOp(
        kernel_size,
        stride,
        dilation,
        padding,
    ).forward(
        arg,
    )