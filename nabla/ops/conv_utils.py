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

"""Numpy-based convolution utilities for eager execution."""

import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, dilation=1, pad=0):
    """
    Convert input data to column matrix for convolution.

    Parameters:
    -----------
    input_data : ndarray
        Input data with shape (N, C, H, W)
    filter_h : int
        Filter height
    filter_w : int
        Filter width
    stride : int or tuple
        Stride for convolution
    dilation : int or tuple
        Dilation for convolution
    pad : int or tuple
        Padding for input

    Returns:
    --------
    col : ndarray
        Column matrix with shape (N, C, filter_h, filter_w, out_h, out_w)
    """
    N, C, H, W = input_data.shape

    # Handle stride and dilation as tuples
    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h, dilation_w = dilation, dilation
    else:
        dilation_h, dilation_w = dilation

    if isinstance(pad, int):
        pad_h, pad_w = pad, pad
    else:
        pad_h, pad_w = pad

    out_h = (H + 2 * pad_h - dilation_h * (filter_h - 1) - 1) // stride_h + 1
    out_w = (W + 2 * pad_w - dilation_w * (filter_w - 1) - 1) // stride_w + 1

    img = np.pad(
        input_data, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
    )
    col = np.ndarray((N, C, filter_h, filter_w, out_h, out_w), dtype=input_data.dtype)

    for j in range(filter_h):
        j_lim = j * dilation_h + stride_h * out_h
        for i in range(filter_w):
            i_lim = i * dilation_w + stride_w * out_w
            col[:, :, j, i, :, :] = img[
                :,
                :,
                j * dilation_h : j_lim : stride_h,
                i * dilation_w : i_lim : stride_w,
            ]

    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, dilation=1, pad=0):
    """
    Convert column matrix back to input data shape.

    Parameters:
    -----------
    col : ndarray
        Column matrix with shape (N, C, filter_h, filter_w, out_h, out_w)
    input_shape : tuple
        Original input shape (N, C, H, W)
    filter_h : int
        Filter height
    filter_w : int
        Filter width
    stride : int or tuple
        Stride for convolution
    dilation : int or tuple
        Dilation for convolution
    pad : int or tuple
        Padding for input

    Returns:
    --------
    img : ndarray
        Reconstructed input data with shape (N, C, H, W)
    """
    N, C, H, W = input_shape

    # Handle stride and dilation as tuples
    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h, dilation_w = dilation, dilation
    else:
        dilation_h, dilation_w = dilation

    if isinstance(pad, int):
        pad_h, pad_w = pad, pad
    else:
        pad_h, pad_w = pad

    out_h = (H + 2 * pad_h - dilation_h * (filter_h - 1) - 1) // stride_h + 1
    out_w = (W + 2 * pad_w - dilation_w * (filter_w - 1) - 1) // stride_w + 1

    img = np.zeros(
        (N, C, H + 2 * pad_h + stride_h - 1, W + 2 * pad_w + stride_w - 1),
        dtype=col.dtype,
    )

    for j in range(filter_h):
        j_lim = j * dilation_h + stride_h * out_h
        for i in range(filter_w):
            i_lim = i * dilation_w + stride_w * out_w
            img[
                :,
                :,
                j * dilation_h : j_lim : stride_h,
                i * dilation_w : i_lim : stride_w,
            ] += col[:, :, j, i, :, :]

    return img[:, :, pad_h : H + pad_h, pad_w : W + pad_w]


def conv2d(input_data, filters, dilation=(1, 1), stride=(1, 1), padding=(0, 0)):
    """
    2D convolution using im2col method.

    Parameters:
    -----------
    input_data : ndarray
        Input data with shape (N, C_in, H, W)
    filters : ndarray
        Filters with shape (C_out, C_in, filter_h, filter_w)
    dilation : tuple
        Dilation factors (dilation_h, dilation_w)
    stride : tuple
        Stride values (stride_h, stride_w)
    padding : tuple
        Padding values (pad_h, pad_w)

    Returns:
    --------
    output : ndarray
        Convolution output with shape (N, C_out, out_h, out_w)
    """
    N, C_in, H, W = input_data.shape
    C_out, C_in_f, filter_h, filter_w = filters.shape

    assert C_in == C_in_f, f"Input channels {C_in} != filter input channels {C_in_f}"

    # Calculate output dimensions
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    out_h = (H + 2 * pad_h - dilation_h * (filter_h - 1) - 1) // stride_h + 1
    out_w = (W + 2 * pad_w - dilation_w * (filter_w - 1) - 1) // stride_w + 1

    # Convert input to column matrix
    col = im2col(input_data, filter_h, filter_w, stride, dilation, padding)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    # Reshape filters
    W_col = filters.reshape(C_out, -1)

    # Perform convolution via matrix multiplication
    out = np.dot(col, W_col.T)
    out = out.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)

    return out


def transposed_conv2d(
    input_data,
    filters,
    dilation=(1, 1),
    stride=(1, 1),
    padding=(0, 0),
    output_padding=(0, 0),
):
    """
    2D transposed convolution using JAX-compatible algorithm.

    JAX's conv_transpose implementation:
    1. Upsample input by inserting (stride-1) zeros between elements
    2. Apply regular convolution with effective padding

    For transposed convolution, the effective padding is:
    effective_pad = kernel_size - 1 - original_pad

    Parameters:
    -----------
    input_data : ndarray
        Input data with shape (N, C_in, H, W)
    filters : ndarray
        Filters with shape (C_out, C_in, filter_h, filter_w)
    dilation : tuple
        Dilation factors (dilation_h, dilation_w)
    stride : tuple
        Stride values (stride_h, stride_w)
    padding : tuple
        Original padding values (pad_h, pad_w) from the forward convolution
    output_padding : tuple
        Output padding values (out_pad_h, out_pad_w) - not used in JAX-compatible mode

    Returns:
    --------
    output : ndarray
        Transposed convolution output
    """
    N, C_in, H, W = input_data.shape
    C_out, C_in_f, filter_h, filter_w = filters.shape

    assert C_in == C_in_f, f"Input channels {C_in} != filter input channels {C_in_f}"

    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    # Step 1: Upsample input by inserting (stride-1) zeros between elements
    if stride_h > 1 or stride_w > 1:
        # Calculate upsampled dimensions
        upsampled_h = H + (H - 1) * (stride_h - 1)
        upsampled_w = W + (W - 1) * (stride_w - 1)

        # Create upsampled array filled with zeros
        upsampled = np.zeros(
            (N, C_in, upsampled_h, upsampled_w), dtype=input_data.dtype
        )

        # Insert original values at strided positions
        upsampled[:, :, ::stride_h, ::stride_w] = input_data
    else:
        # No upsampling needed for stride=1
        upsampled = input_data

    # Step 2: Calculate effective padding for transposed convolution
    # For transposed conv, if original conv had padding P and kernel size K,
    # the effective padding for the underlying regular conv is (K-1-P)
    effective_pad_h = filter_h - 1 - pad_h
    effective_pad_w = filter_w - 1 - pad_w

    # Step 3: Apply regular convolution with effective padding
    # Use stride=1 since upsampling already handled the stride effect
    result = conv2d(
        upsampled,
        filters,
        dilation=dilation,
        stride=(1, 1),
        padding=(effective_pad_h, effective_pad_w),
    )

    # Step 4: Apply output_padding if specified
    # Output padding adds zeros to the right and bottom of the output
    out_pad_h, out_pad_w = output_padding
    if out_pad_h > 0 or out_pad_w > 0:
        N, C_out, H_out, W_out = result.shape
        padded_result = np.zeros(
            (N, C_out, H_out + out_pad_h, W_out + out_pad_w), dtype=result.dtype
        )
        padded_result[:, :, :H_out, :W_out] = result
        result = padded_result

    return result
