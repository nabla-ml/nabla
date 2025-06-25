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

"""Linear algebra operations."""

import numpy as np
from max.driver import Tensor
from max.graph import TensorValue, ops

from ..core.array import Array
from ..utils.shape_utils import get_broadcasted_shape
from .operation import BinaryOperation

# Public API
__all__ = ["matmul"]


class MatMulOp(BinaryOperation):
    """Matrix multiplication operation with batching support."""

    def __init__(self):
        super().__init__("dot_general")

    def forward(self, *args: Array) -> Array:
        """Forward pass for binary operations."""
        if len(args) != 2:
            raise ValueError(f"Binary operation requires 2 arguments, got {len(args)}")

        # Move arrays to best device
        from .operation import move_to_best_device

        args = move_to_best_device(*args)
        arg1, arg2 = args[0], args[1]

        from ..ops.view import broadcast_batch_dims, broadcast_to, reshape

        arg1_has_rank_1 = len(arg1.shape) == 1
        arg2_has_rank_1 = len(arg2.shape) == 1

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
        if arg1.traced:
            arg1 = broadcast_to(arg1, output_shape[:-2] + arg1.shape[-2:])
            arg1 = broadcast_batch_dims(arg1, output_batch_dims)
        if arg2.traced:
            arg2 = broadcast_to(arg2, output_shape[:-2] + arg2.shape[-2:])
            arg2 = broadcast_batch_dims(arg2, output_batch_dims)

        res = Array(
            shape=output_shape,
            dtype=output_dtype,
            device=arg1.device,
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

        # If the inputs were rank 1, we need to reshape the output
        if arg1_has_rank_1 and arg2_has_rank_1:
            res = reshape(res, output_shape[:-2] + (1, 1))
        elif arg1_has_rank_1:
            res = reshape(res, output_shape[:-2] + (res.shape[1],))
        elif arg2_has_rank_1:
            res = reshape(res, output_shape[:-2] + (res.shape[0],))

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

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
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
            # now we can simpply reshape the args to a rank3 tensor respecitvely and then do a batche dmamtul on this one
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

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        arg0_numpy = args[0].to_numpy()
        arg1_numpy = args[1].to_numpy()
        np_result = np.matmul(arg0_numpy, arg1_numpy)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        x, y = primals
        from .view import transpose

        return [matmul(cotangent, transpose(y)), matmul(transpose(x), cotangent)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        x, y = primals
        tx, ty = tangents

        from .binary import add

        return add(matmul(x, ty), matmul(tx, y))


# Global operation instance for efficiency
_matmul_op = MatMulOp()


def matmul(arg0, arg1) -> Array:
    """Matrix multiplication with broadcasting support."""
    from .binary import _ensure_array

    arg0 = _ensure_array(arg0)
    arg1 = _ensure_array(arg1)
    return _matmul_op.forward(arg0, arg1)


# # --- Convolution operations using im2col and col2im ---
# # Global operation instances
# _conv2d_op_cache = {}
# _conv2d_transpose_op_cache = {}


# # --- Helper functions for normalization ---
# def _normalize_tuple(value, n, name):
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


# def flip(x: Array, axis: int | tuple[int, ...]) -> Array:
#     """
#     Reverses the order of elements in an array along the given axes.
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

#     # Use tuple slicing on the array. The Nabla Array class's __getitem__
#     # must support this to be Python-idiomatic.
#     return x[tuple(slicer)]

# def _conv2d_filter_gradient(
#     x: Array, dy: Array, stride: tuple, dilation: tuple, padding: tuple, groups: int
# ) -> Array:
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

#     def forward(self, *args: Array) -> Array:
#         # Standard forward pass logic
#         from .operation import move_to_best_device
#         input_arr, filter_arr = move_to_best_device(*args)
#         self._validate_inputs(input_arr, filter_arr)

#         output_shape = self.compute_output_shape(input_arr.shape, filter_arr.shape)
#         res = Array(
#             shape=output_shape, dtype=self.compute_output_dtype(input_arr, filter_arr),
#             device=input_arr.device, materialize=False, name=self.name,
#             batch_dims=input_arr.batch_dims,
#         )
#         res.set_maxpr(self.maxpr)
#         res.add_arguments(input_arr, filter_arr)
#         res.vjp_rule = self.vjp_rule
#         res.jvp_rule = self.jvp_rule
#         if not res.stage_realization:
#             self.eagerxpr([input_arr, filter_arr], res)
#         return res

#     def _validate_inputs(self, input_arr: Array, filter_arr: Array) -> None:
#         if len(input_arr.shape) != 4 or len(filter_arr.shape) != 4:
#             raise ValueError("Conv2D requires 4D input and filter tensors.")
#         if input_arr.device != filter_arr.device:
#             raise ValueError(f"Devices {input_arr.device} and {filter_arr.device} are incompatible")

#     def maxpr(self, args: list[TensorValue], output: Array) -> None:
#         input_val, filter_val = args
#         (pt, pb), (pl, pr) = self.padding
#         output.tensor_value = ops.conv2d(
#             x=input_val, filter=filter_val, stride=self.stride,
#             dilation=self.dilation, padding=(pt, pb, pl, pr), groups=self.groups
#         )

#     def eagerxpr(self, args: list[Array], output: Array) -> None:
#         input_arr, filter_arr = args
#         input_torch = torch.from_numpy(np.transpose(input_arr.to_numpy(), (0, 3, 1, 2)))
#         filter_torch = torch.from_numpy(np.transpose(filter_arr.to_numpy(), (3, 2, 0, 1)))
#         (pad_h, _), (pad_w, _) = self.padding
#         result_torch = F.conv2d(
#             input=input_torch, weight=filter_torch, bias=None, stride=self.stride,
#             padding=(pad_h, pad_w), dilation=self.dilation, groups=self.groups
#         )
#         result_nhwc = np.transpose(result_torch.numpy(), (0, 2, 3, 1))
#         output.impl = Tensor.from_numpy(result_nhwc)

#     def vjp_rule(self, primals: list[Array], cotangent: Array, output: Array) -> list[Array]:
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


#     def jvp_rule(self, primals: list[Array], tangents: list[Array], output: Array) -> Array:
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

#     def forward(self, *args: Array) -> Array:
#         from .operation import move_to_best_device
#         input_arr, filter_arr = move_to_best_device(*args)
#         self._validate_inputs(input_arr, filter_arr)
#         output_shape = self.compute_output_shape(input_arr.shape, filter_arr.shape)
#         res = Array(
#             shape=output_shape, dtype=self.compute_output_dtype(input_arr, filter_arr),
#             device=input_arr.device, materialize=False, name=self.name,
#             batch_dims=input_arr.batch_dims,
#         )
#         res.set_maxpr(self.maxpr)
#         res.add_arguments(input_arr, filter_arr)
#         res.vjp_rule = self.vjp_rule
#         res.jvp_rule = self.jvp_rule
#         if not res.stage_realization:
#             self.eagerxpr([input_arr, filter_arr], res)
#         return res

#     def _validate_inputs(self, input_arr: Array, filter_arr: Array) -> None:
#         if len(input_arr.shape) != 4 or len(filter_arr.shape) != 4:
#             raise ValueError("Conv2DTranspose requires 4D input and filter tensors.")
#         if input_arr.device != filter_arr.device:
#             raise ValueError(f"Devices {input_arr.device} and {filter_arr.device} are incompatible")

#     def maxpr(self, args: list[TensorValue], output: Array) -> None:
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

#     def eagerxpr(self, args: list[Array], output: Array) -> None:
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
#         output.impl = Tensor.from_numpy(result_nhwc)

#     def vjp_rule(self, primals: list[Array], cotangent: Array, output: Array) -> list[Array]:
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

#     def jvp_rule(self, primals: list[Array], tangents: list[Array], output: Array) -> Array:
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
#     input_arr: Array, filter_arr: Array, stride=(1, 1),
#     dilation=(1, 1), padding=0, groups=1
# ) -> Array:
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
#     input_arr: Array, filter_arr: Array, stride=(1, 1),
#     dilation=(1, 1), padding=0, output_padding=0, groups=1
# ) -> Array:
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
#     input_data : ndarray
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
#     col : ndarray
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
#     col : ndarray
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
#     img : ndarray
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
#     input_data : ndarray
#         Input data with shape (N, C_in, H, W)
#     filters : ndarray
#         Filters with shape (C_out, C_in, filter_h, filter_w)
#     dilation : tuple
#         Dilation factors (dilation_h, dilation_w)
#     stride : tuple
#         Stride values (stride_h, stride_w)
#     padding : tuple
#         Padding values (pad_h, pad_w)

#     Returns:
#     --------
#     output : ndarray
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
#     input_data : ndarray
#         Input data with shape (N, C_in, H, W)
#     filters : ndarray
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
#     output : ndarray
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

#         # Create upsampled array filled with zeros
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
