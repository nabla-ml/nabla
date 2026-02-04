# MAX Operations Reference (Categorized)

## Types and Helpers
These are non-tensor operations, help definitions, or graph management utilities.

max.graph.ops.Callable()
> A symbolic callable representation.

max.graph.ops.DeviceRef(device_type, id=0)
> A symbolic device representation (e.g., CPU, GPU).

max.graph.ops.DeviceRef.CPU(id=0)
> Returns a CPU device reference.

max.graph.ops.DeviceRef.GPU(id=0)
> Returns a GPU device reference.

max.graph.ops.DeviceRef.from_device(device)
> Creates a device reference from a concrete device.

max.graph.ops.DeviceRef.from_mlir(attr)
> Creates a device reference from an MLIR attribute.

max.graph.ops.DeviceRef.is_cpu()
> Checks if the device is a CPU.

max.graph.ops.DeviceRef.is_gpu()
> Checks if the device is a GPU.

max.graph.ops.DeviceRef.to_device()
> Converts a device reference to a concrete device.

max.graph.ops.DeviceRef.to_mlir()
> Converts a device reference to an MLIR attribute.

max.graph.ops.TensorType(dtype, shape, device, _layout=None)
> Representational type for a symbolic tensor (dtype, shape, device).

max.graph.ops.TensorType.as_buffer()
> Returns the corresponding buffer type.

max.graph.ops.TensorType.from_mlir(type)
> Creates a tensor type from an MLIR type.

max.graph.ops.TensorType.to_mlir()
> Converts a tensor type to an MLIR type.

max.graph.ops.InterpolationMode(value, names=<not given>, *values, module=None, qualname=None, type=None, start=1, boundary=None)
> Enum defining image interpolation modes (e.g., BICUBIC, BILINEAR).

max.graph.ops.assert_same_device(*values, **named_values)
> Asserts that all provided tensors or buffers reside on the same device.

max.graph.ops.print(value, label='debug_tensor')
> Prints a tensor or string during graph execution for debugging.

max.graph.ops.shape_to_tensor(shape)
> Converts a shape attribute to a tensor value.

max.graph.ops.buffer_create(type)
> Creates a mutable buffer of a given type.

max.graph.ops.buffer_load(x)
> Loads a mutable buffer into an immutable tensor.

max.graph.ops.buffer_store(destination, source)
> Stores an immutable tensor into a mutable buffer.

max.graph.ops.buffer_store_slice(destination, source, indices)
> Stores a tensor into a specific slice of a mutable buffer.

## Creation Operations
Operations that create new tensors from constant values or sequences.

max.graph.ops.constant(value, dtype=None, device=None)
> Creates a constant tensor from a provided value.

max.graph.ops.constant_external(name, type)
> Registers an external weight name for compile-time lookup.

max.graph.ops.range(start, stop, step=1, out_dim=None, *, dtype, device)
> Generates a sequence of numbers from start to stop with a given step.

max.graph.ops.hann_window(window_length, device, periodic=True, dtype=float32)
> Generates a 1D Hann window tensor.

## Unary Operations
Elementwise operations that take a single tensor as input.

max.graph.ops.abs(x)
> Elementwise absolute value.
> vjp: cotangent * sign(x)
> jvp: tangent * sign(x)

max.graph.ops.acos(x)
> Elementwise arccosine.
> vjp: -cotangent / sqrt(1 - x^2)
> jvp: -tangent / sqrt(1 - x^2)

max.graph.ops.atanh(x)
> Elementwise inverse hyperbolic tangent.
> vjp: cotangent / (1 - x^2)
> jvp: tangent / (1 - x^2)

max.graph.ops.cos(x)
> Elementwise cosine.
> vjp: -cotangent * sin(x)
> jvp: -tangent * sin(x)

max.graph.ops.erf(x)
> Elementwise error function.
> vjp: cotangent * (2 / sqrt(pi)) * exp(-x^2)
> jvp: tangent * (2 / sqrt(pi)) * exp(-x^2)

max.graph.ops.exp(x)
> Elementwise exponential.
> vjp: cotangent * output
> jvp: tangent * output

max.graph.ops.floor(x)
> Elementwise floor (round down).
> vjp: None
> jvp: None

max.graph.ops.is_inf(x)
> Elementwise check for infinity.
> vjp: None
> jvp: None

max.graph.ops.is_nan(x)
> Elementwise check for NaN.
> vjp: None
> jvp: None

max.graph.ops.log(x)
> Elementwise natural logarithm.
> vjp: cotangent / x
> jvp: tangent / x

max.graph.ops.log1p(x)
> Elementwise log(1 + x).
> vjp: cotangent / (1 + x)
> jvp: tangent / (1 + x)

max.graph.ops.logical_not(x)
> Elementwise logical NOT.
> vjp: None
> jvp: None

max.graph.ops.negate(x)
> Elementwise negation (-x).
> vjp: -cotangent
> jvp: -tangent

max.graph.ops.relu(x)
> Elementwise Rectified Linear Unit activation.
> vjp: cotangent * (x > 0)
> jvp: tangent * (x > 0)

max.graph.ops.round(x)
> Elementwise rounding to the nearest even number.
> vjp: None
> jvp: None

max.graph.ops.rsqrt(x)
> Elementwise reciprocal square root (1 / sqrt(x)).
> vjp: -0.5 * cotangent * output^3
> jvp: -0.5 * tangent * output^3

max.graph.ops.sigmoid(x)
> Elementwise sigmoid activation.
> vjp: cotangent * output * (1 - output)
> jvp: tangent * output * (1 - output)

max.graph.ops.silu(x)
> Elementwise Sigmoid Linear Unit (swish) activation.
> vjp: cotangent * (sigmoid(x) + output * (1 - sigmoid(x)))
> jvp: tangent * (sigmoid(x) + output * (1 - sigmoid(x)))

max.graph.ops.sin(x)
> Elementwise sine.
> vjp: cotangent * cos(x)
> jvp: tangent * cos(x)

max.graph.ops.sqrt(x)
> Elementwise square root.
> vjp: cotangent / (2 * output)
> jvp: tangent / (2 * output)

max.graph.ops.tanh(x)
> Elementwise hyperbolic tangent activation.
> vjp: cotangent * (1 - output^2)
> jvp: tangent * (1 - output^2)

max.graph.ops.trunc(x)
> Elementwise truncation (rounds towards zero).
> vjp: None
> jvp: None

## Binary Operations
Elementwise operations that take two tensors as input.

max.graph.ops.add(lhs, rhs)
> Elementwise addition.
> vjp: (cotangent, cotangent)
> jvp: tangent_lhs + tangent_rhs

max.graph.ops.div(lhs, rhs)
> Elementwise true division (promotes to float for integers).
> vjp: (cotangent / rhs, -cotangent * lhs / rhs^2)
> jvp: (rhs * tangent_lhs - lhs * tangent_rhs) / rhs^2

max.graph.ops.logical_and(lhs, rhs)
> Elementwise logical AND.
> vjp: None
> jvp: None

max.graph.ops.logical_or(lhs, rhs)
> Elementwise logical OR.
> vjp: None
> jvp: None

max.graph.ops.logical_xor(lhs, rhs)
> Elementwise logical XOR.
> vjp: None
> jvp: None

max.graph.ops.matmul(lhs, rhs)
> Matrix multiplication with broadcasting support.
> vjp: (cotangent @ rhs.T, lhs.T @ cotangent)
> jvp: lhs @ tangent_rhs + tangent_lhs @ rhs

max.graph.ops.mod(lhs, rhs)
> Elementwise modulus (remainder).
> vjp: (cotangent, -cotangent * floor(lhs / rhs))
> jvp: tangent_lhs - tangent_rhs * floor(lhs / rhs)

max.graph.ops.mul(lhs, rhs)
> Elementwise multiplication.
> vjp: (cotangent * rhs, cotangent * lhs)
> jvp: lhs * tangent_rhs + rhs * tangent_lhs

max.graph.ops.outer(lhs, rhs)
> Outer product of two vectors.
> vjp: (matmul(cotangent, rhs), matmul(transpose(cotangent), lhs))
> jvp: outer(tangent_lhs, rhs) + outer(lhs, tangent_rhs)

max.graph.ops.pow(lhs, rhs)
> Elementwise exponentiation (lhs ^ rhs).
> vjp: (cotangent * rhs * lhs^(rhs-1), cotangent * output * log(lhs))
> jvp: rhs * lhs^(rhs-1) * tangent_lhs + output * log(lhs) * tangent_rhs

max.graph.ops.sub(lhs, rhs)
> Elementwise subtraction.
> vjp: (cotangent, -cotangent)
> jvp: tangent_lhs - tangent_rhs

## Comparison Operations
Operations that compare tensors elementwise and usually return boolean results.

max.graph.ops.equal(lhs, rhs)
> Elementwise equality comparison.
> vjp: None
> jvp: None

max.graph.ops.greater(lhs, rhs)
> Elementwise greater-than comparison.
> vjp: None
> jvp: None

max.graph.ops.greater_equal(lhs, rhs)
> Elementwise greater-than-or-equal comparison.
> vjp: None
> jvp: None

max.graph.ops.not_equal(lhs, rhs)
> Elementwise inequality comparison.
> vjp: None
> jvp: None

max.graph.ops.where(condition, x, y)
> Elementwise selection: returns x where condition is true, otherwise y.
> vjp: (None, where(condition, cotangent, 0), where(condition, 0, cotangent))
> jvp: where(condition, tangent_x, tangent_y)

## Reduction Operations
Operations that reduce one or more dimensions of a tensor.

max.graph.ops.argmax(x, axis=-1)
> Indices of the maximum value along an axis.
> vjp: None
> jvp: None

max.graph.ops.argmin(x, axis=-1)
> Indices of the minimum value along an axis.
> vjp: None
> jvp: None

max.graph.ops.cumsum(x, axis=-1, exclusive=False, reverse=False)
> Cumulative sum along an axis.
> vjp: flip(cumsum(flip(cotangent, axis), axis), axis)
> jvp: cumsum(tangent, axis)

max.graph.ops.max(x, y=None, /, axis=None)
> Overload for elementwise maximum or reduction maximum along an axis.
> vjp: cotangent * (x == broadcast(output, x.shape))
> jvp: where(x == output, tangent, 0)

max.graph.ops.mean(x, axis=-1)
> Mean reduction along an axis.
> vjp: broadcast_to(cotangent, input_shape) / size
> jvp: mean(tangent, axis)

max.graph.ops.min(x, y=None, /, axis=None)
> Overload for elementwise minimum or reduction minimum along an axis.
> vjp: cotangent * (x == broadcast(output, x.shape))
> jvp: where(x == output, tangent, 0)

max.graph.ops.sum(x, axis=-1)
> Sum reduction along an axis.
> vjp: broadcast_to(cotangent, input_shape)
> jvp: sum(tangent, axis)

## View and Shape Operations
Operations that change the shape, layout, or indexing of a tensor without changing its data.

max.graph.ops.as_interleaved_complex(x)
> Interprets alternating real/imag pairs as complex numbers.
> vjp: view_as_real_nested(cotangent)
> jvp: as_interleaved_complex(tangent)

max.graph.ops.band_part(x, num_lower=None, num_upper=None, exclude=False)
> Extracts a diagonal band from a matrix or batch of matrices.
> vjp: band_part(cotangent, num_lower, num_upper, exclude)
> jvp: band_part(tangent, num_lower, num_upper, exclude)

max.graph.ops.broadcast_to(x, shape, out_dims=None)
> Broadcasts a tensor to a target shape.
> vjp: sum(cotangent, axis=broadcast_dims)
> jvp: broadcast_to(tangent, shape)

max.graph.ops.cast(x, dtype)
> Casts a tensor to a different data type.
> vjp: cast(cotangent, input_dtype)
> jvp: cast(tangent, dtype)

max.graph.ops.concat(original_vals, axis=0)
> Concatenates a sequence of tensors along an existing axis.
> vjp: split(cotangent, [v.shape[axis] for v in original_vals], axis)
> jvp: concat(tangents, axis)

max.graph.ops.flatten(x, start_dim=0, end_dim=-1)
> Flattens a range of dimensions into a single dimension.
> vjp: reshape(cotangent, input_shape)
> jvp: reshape(tangent, output_shape)

max.graph.ops.pad(input, paddings, mode='constant', value=0)
> Pads a tensor with a constant value.
> vjp: slice_tensor(cotangent, unpadded_region)
> jvp: pad(tangent, paddings, mode, value=0)

max.graph.ops.permute(x, dims)
> Reorders all dimensions of a tensor.
> vjp: permute(cotangent, inverse_dims)
> jvp: permute(tangent, dims)

max.graph.ops.rebind(x, shape, message='', layout=None)
> Asserts or updates the symbolic shape/layout of a tensor.
> vjp: cotangent
> jvp: tangent

max.graph.ops.repeat_interleave(x, repeats, axis=None, out_dim=None)
> Repeats elements of a tensor along a dimension.
> vjp: reshape(sum(reshape(cotangent, expanded_shape), axis=new_axes), input_shape)
> jvp: repeat_interleave(tangent, repeats, axis)

max.graph.ops.reshape(x, shape)
> Changes the shape of a tensor without altering its data.
> vjp: reshape(cotangent, input_shape)
> jvp: reshape(tangent, shape)

max.graph.ops.resize(input, shape, interpolation=InterpolationMode.BILINEAR)
> Resizes an image tensor using interpolation.
> vjp: resize_grad(cotangent, input.shape, interpolation)
> jvp: resize(tangent, shape, interpolation)

max.graph.ops.slice_tensor(x, indices)
> Slices out a subtensor based on multidimensional indices.
> vjp: buffer_store_slice(buffer_create(x.shape), cotangent, indices)
> jvp: slice_tensor(tangent, indices)

max.graph.ops.squeeze(x, axis)
> Removes a size-1 dimension from a tensor.
> vjp: unsqueeze(cotangent, axis)
> jvp: squeeze(tangent, axis)

max.graph.ops.stack(values, axis=0)
> Joins a sequence of tensors along a new axis.
> vjp: split(cotangent, axis)
> jvp: stack(tangents, axis)

max.graph.ops.tile(x, repeats)
> Repeats a tensor multiple times along each dimension.
> vjp: reshape(sum(reshape(cotangent, tiled_split_shape), axis=reduction_axes), input_shape)
> jvp: tile(tangent, repeats)

max.graph.ops.transpose(x, axis_1, axis_2)
> Transposes two axes of a tensor.
> vjp: transpose(cotangent, axis_1, axis_2)
> jvp: transpose(tangent, axis_1, axis_2)

max.graph.ops.unsqueeze(x, axis)
> Inserts a size-1 dimension at a specific index.
> vjp: squeeze(cotangent, axis)
> jvp: unsqueeze(tangent, axis)

## Multi-Output Operations
Operations that return multiple tensor outputs.

max.graph.ops.chunk(x, chunks, axis=0)
> Splits a tensor into a specified number of equal-sized chunks along an axis.
> vjp: concat(cotangents, axis)
> jvp: chunk(tangent, chunks, axis)

max.graph.ops.split(x, split_sizes, axis=0)
> Splits a tensor into multiple tensors along an axis.
> vjp: concat(cotangents, axis)
> jvp: split(tangent, split_sizes, axis)

max.graph.ops.top_k(input, k, axis=-1)
> Returns the K largest values and their indices along an axis.
> vjp: scatter(zeros_like(input), indices, cotangent, axis)
> jvp: gather(tangent, indices, axis)

## Communication and Distributed Operations
Operations that involve multi-device communication.

max.graph.ops.allgather(inputs, signal_buffers, axis=0)
> Collects tensors from multiple devices and concatenates them along an axis.
> vjp: slice_tensor(cotangent, rank_indices)
> jvp: allgather(tangent, signal_buffers, axis)

max.graph.ops.distributed_broadcast(input, signal_buffers)
> Broadcasts a tensor from one device to all participating devices.
> vjp: all_reduce(cotangent, op=SUM)
> jvp: distributed_broadcast(tangent, signal_buffers)

max.graph.ops.shard_and_stack(inputs, devices, axis=0)
> Shards input tensors across devices for tensor parallelism.
> vjp: allgather(cotangent, signal_buffers, axis)
> jvp: shard_and_stack(tangent, devices, axis)

max.graph.ops.transfer_to(x, device)
> Transfers a tensor between devices.
> vjp: transfer_to(cotangent, src_device)
> jvp: transfer_to(tangent, device)

## Control Flow Operations
Operations that control graph execution.

max.graph.ops.call(graph, *args, prefix='')
> Call a graph with the provided arguments and return its results.
> vjp: call(vjp_graph, *args, cotangents)
> jvp: call(jvp_graph, *args, tangents)

max.graph.ops.cond(pred, out_types, then_fn, else_fn)
> Conditional execution of two branches based on a boolean predicate.
> vjp: cond(pred, vjp_then, vjp_else, cotangents)
> jvp: cond(pred, jvp_then, jvp_else, tangents)

max.graph.ops.while_loop(initial_values, predicate, body)
> Executes a loop until the predicate evaluates to false.
> vjp: while_loop_vjp(initial_values, body, cotangents)
> jvp: while_loop_jvp(initial_values, body, tangents)

## Custom and Specialized Operations
Custom kernels, quantization, and specialized domain-specific ops.

max.graph.ops.custom(name, device, values, out_types, parameters=None)
> Executes a custom registered kernel.
> vjp: custom_vjp(name, cotangent, values, parameters)
> jvp: custom_jvp(name, tangent, values, parameters)

max.graph.ops.inplace_custom(name, device, values, out_types=None, parameters=None)
> Executes a custom registered kernel with in-place mutations.
> vjp: custom_vjp(name, cotangent, values, parameters)
> jvp: custom_jvp(name, tangent, values, parameters)

max.graph.ops.dequantize(encoding, quantized)
> Dequantizes a tensor using a specific encoding (e.g., Q4_0, Q4_K).
> vjp: (dequant_grad_encoding(cotangent, quantized), None)
> jvp: dequantize(encoding_tangent, quantized)

max.graph.ops.qmatmul(encoding, config, lhs, *rhs)
> Matrix multiplication between floating point and quantized tensors.
> vjp: (None, None, matmul(cotangent, dequantize(rhs).T), *dequant_grad(matmul(lhs.T, cotangent), *rhs))
> jvp: matmul(tangent, dequantize(rhs))

max.graph.ops.irfft(input_tensor, n=None, axis=-1, normalization=Normalization.BACKWARD, input_is_complex=False, buffer_size_mb=512)
> Inverse Real Fast Fourier Transform.
> vjp: rfft(cotangent, ...)
> jvp: irfft(tangent, ...)

max.graph.ops.gather(input, indices, axis)
> Selects elements along an axis using integer indices.
> vjp: (scatter_add(zeros_like(input), indices, cotangent, axis=axis), None)
> jvp: gather(tangent, indices, axis)

max.graph.ops.gather_nd(input, indices, batch_dims=0)
> Selects elements using N-dimensional indexing.
> vjp: scatter_nd(zeros_like(input), indices, cotangent, batch_dims)
> jvp: gather_nd(tangent, indices, batch_dims)

max.graph.ops.masked_scatter(input, mask, updates, out_dim)
> Updates tensor elements where a boolean mask is true.
> vjp: (masked_scatter(cotangent, mask, 0), None, slice(cotangent, mask))
> jvp: masked_scatter(tangent, mask, tangent_updates)

max.graph.ops.nonzero(x, out_dim)
> Returns the indices of nonzero elements.
> vjp: None
> jvp: None

max.graph.ops.scatter(input, updates, indices, axis=-1)
> Writes updates to a tensor along an axis at specified indices.
> vjp: (scatter(cotangent, indices, zeros_like(updates), axis=axis), None, gather(cotangent, indices, axis=axis))
> jvp: scatter(tangent, tangent_updates, indices, axis)

max.graph.ops.scatter_nd(input, updates, indices, batch_dims=0)
> Writes updates to a tensor using N-dimensional indexing.
> vjp: (scatter_nd(cotangent, indices, 0, batch_dims), gather_nd(cotangent, indices, batch_dims), None)
> jvp: scatter_nd(tangent_input, tangent_updates, indices, batch_dims)

## NN and High-level Operations
Standard neural network layers and activations.

max.graph.ops.avg_pool2d(input, kernel_size, stride=1, dilation=1, padding=0, ceil_mode=False, count_boundary=True)
> 2D average pooling.
> vjp: avg_unpool2d(cotangent, kernel_size, stride, ...)
> jvp: avg_pool2d(tangent, kernel_size, stride, ...)

max.graph.ops.max_pool2d(input, kernel_size, stride=1, dilation=1, padding=0, ceil_mode=False)
> 2D max pooling.
> vjp: max_unpool2d(cotangent, indices, kernel_size, ...)
> jvp: max_pool2d_with_indices(tangent, indices, ...)

max.graph.ops.conv2d(x, filter, stride=(1, 1), dilation=(1, 1), padding=(0, 0, 0, 0), groups=1, bias=None, input_layout=ConvInputLayout.NHWC, filter_layout=FilterLayout.RSCF)
> 2D convolution.
> vjp: (conv2d_transpose(cotangent, filter, stride, dilation, padding), conv2d_filter_grad(x, cotangent))
> jvp: conv2d(tangent_x, filter) + conv2d(x, tangent_filter)

max.graph.ops.conv2d_transpose(x, filter, stride=(1, 1), dilation=(1, 1), padding=(0, 0, 0, 0), output_paddings=(0, 0), bias=None, input_layout=ConvInputLayout.NHWC, filter_layout=FilterLayout.RSCF)
> 2D transposed convolution (deconvolution).
> vjp: (conv2d(cotangent, filter, ...), conv2d_transpose_filter_grad(x, cotangent))
> jvp: conv2d_transpose(tangent_x, filter) + conv2d_transpose(x, tangent_filter)

max.graph.ops.conv3d(x, filter, stride=(1, 1, 1), dilation=(1, 1, 1), padding=(0, 0, 0, 0, 0, 0), groups=1, bias=None, input_layout=ConvInputLayout.NHWC, filter_layout=FilterLayout.QRSCF)
> 3D convolution.
> vjp: (conv3d_input_grad(cotangent, filter), conv3d_filter_grad(x, cotangent))
> jvp: conv3d(tangent_x, filter) + conv3d(x, tangent_filter)

max.graph.ops.layer_norm(input, gamma, beta, epsilon)
> Applies layer normalization.
> vjp: (ln_input_grad(cotangent, ...), ln_gamma_grad(cotangent, ...), ln_beta_grad(cotangent))
> jvp: layer_norm_jvp(tangent, ...)

max.graph.ops.gelu(x, approximate='none')
> Elementwise Gaussian Error Linear Unit activation.
> vjp: cotangent * (0.5 * (1 + erf(x / sqrt(2))) + x * exp(-x^2 / 2) / sqrt(2 * pi))
> jvp: tangent * (0.5 * (1 + erf(x / sqrt(2))) + x * exp(-x^2 / 2) / sqrt(2 * pi))

max.graph.ops.softmax(value, axis=-1)
> Softmax activation along an axis.
> vjp: output * (cotangent - sum(cotangent * output, axis, keepdims=True))
> jvp: output * (tangent - sum(tangent * output, axis, keepdims=True))

max.graph.ops.logsoftmax(value, axis=-1)
> Logarithm of the softmax activation along an axis.
> vjp: cotangent - softmax(value) * sum(cotangent, axis, keepdims=True)
> jvp: tangent - exp(output) * sum(tangent, axis, keepdims=True)

max.graph.ops.fold(input, output_size, kernel_size, stride=1, dilation=1, padding=0)
> Combines sliding blocks into a larger tensor (inverse of im2col).
> vjp: unfold(cotangent, kernel_size, stride, ...)
> jvp: fold(tangent, output_size, kernel_size, ...)

max.graph.ops.argsort(x, ascending=True)
> Indices that would sort the tensor along its first dimension.
> vjp: None
> jvp: None
