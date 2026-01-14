Python module

ops

Implements operations used when staging a graph.

This module provides operations for building computational graphs in MAX. These operations create, transform, and manipulate tensor values within the graph.

You can also use functions in Graph to add constant values to your graph with operations like constant().

The TensorValue type (returned by most operations) implements various dunder methods to support operations between TensorValues, such as + for addition, * for multiplication, and @ for matrix multiplication. It also provides convenience methods like reshape() and flatten().

Callable
class max.graph.ops.Callable

InterpolationMode
class max.graph.ops.InterpolationMode(value, names=<not given>, *values, module=None, qualname=None, type=None, start=1, boundary=None)

Interpolation modes for image resize operations.

This enum defines the available interpolation methods that can be used when resizing tensors. Currently only BICUBIC is implemented, with BILINEAR and NEAREST planned for future support.

BICUBIC
BICUBIC = 'bicubic'

BILINEAR
BILINEAR = 'bilinear'

NEAREST
NEAREST = 'nearest'

TensorType
class max.graph.ops.TensorType(dtype, shape, device, _layout=None)

A symbolic TensorType.

This is not an eager tensor type! This contains no actual data, but instead represents the type of a value at some point in time during model execution.

Most internal values in a model will be tensors. This type represents their element type (dtype) and dimensions (dims) at a specific point during model computation. It allows us to do some optimistic optimizations and shape inference during graph construction, and to provide more detailed shape information to the compiler for further optimization passes.

The following example shows how to create a tensor type with static dimensions and access its properties:

from max.graph import TensorType
from max.dtype import DType
tensor_type = TensorType(DType.float32, (2, 3))
print(tensor_type.dtype)  # Outputs: DType.float32
print(tensor_type.shape)  # Outputs: [2, 3]

It can also represent a fully dynamic rank tensor. The presence of dynamic rank tensors in a graph will often degrade performance dramatically and prevents many classes of optimizations.

An optional device (device) can also be provided to indicate the explicit device the tensor is associated with.

Constructs a tensor type.

Parameters:

dtype (DType) – The element type of the tensor data.
dims – The shape dimensions of the tensor. The number of dims is the rank of the tensor.
shape (Shape)
device (DeviceRef)
_layout (FilterLayout | None)
as_buffer()
as_buffer()

Returns the analogous buffer type.

Return type:

BufferType

from_mlir()
classmethod from_mlir(type)

Constructs a tensor type from an MLIR type.

Parameters:

t – The MLIR Type object to parse into a tensor type.
type (TensorType)
Returns:

The tensor type represented by the MLIR Type value.

Return type:

TensorType

to_mlir()
to_mlir()

Converts to an mlir.Type instance.

Returns:

An mlir.Type in the specified Context.

Return type:

TensorType

abs()
max.graph.ops.abs(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

add()
max.graph.ops.add(lhs, rhs)

Parameters:

lhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
rhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
Return type:

TensorValue

allgather()
max.graph.ops.allgather(inputs, signal_buffers, axis=0)

Collective allgather operation.

This op is a collective op which takes in tensors from different devices and outputs tensors on different devices. In particular, this operation will gather the inputs across different devices and concatenates them along the specified dimension. The result is then broadcasted back to the same devices that the inputs came from.

Parameters:

inputs (Iterable[TensorValue]) – The input tensors to gather.
signal_buffers (Iterable[BufferValue]) – Device buffer values used for synchronization.
axis (int) – Dimension to concatenate the input tensors. Defaults to 0.
Returns:

An iterable outputs which all hold the gathered output. Each output tensor contains the concatenation of all inputs along the specified dimension.

Return type:

list[TensorValue]

argmax()
max.graph.ops.argmax(x, axis=-1)

Reduces a symbolic tensor using an argmax operation.

When provided with a tensor with all identical elements, on CPU this will return the first element index in the tensor, on GPU this will return an arbitrary index.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input tensor for the operation.
axis (int) – The axis along which to compute the reduction. If negative, indexes from the last dimension. For example, a value of -1 will compute the reduction along the last dimension.
Returns:

A symbolic tensor representing the result of the argmax operation. The tensor will have the same rank as the input tensor, and the same shape except along the axis dimension which will have size 1.

Return type:

TensorValue

argmin()
max.graph.ops.argmin(x, axis=-1)

Reduces a symbolic tensor using an argmin operation.

When provided with a tensor with all identical elements, on CPU this will return the first element index in the tensor, on GPU this will return an arbitrary index.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input tensor for the operation.
axis (int) – The axis along which to compute the reduction. If negative, indexes from the last dimension. For example, a value of -1 will compute the reduction along the last dimension.
Returns:

A symbolic tensor representing the result of the argmin operation. The tensor will have the same rank as the input tensor, and the same shape except along the axis dimension which will have size 1.

Return type:

TensorValue

argsort()
max.graph.ops.argsort(x, ascending=True)

Returns the indices that would sort a tensor.

This function returns the indices that would sort the input tensor along its first dimension. The returned indices are of type int64.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue) – Input tensor to be sorted.
ascending (bool) – If True (default), sort in ascending order. If False, sort in descending order.
Returns:

A tensor of indices of the same shape as the input tensor.

Return type:

TensorValue

as_interleaved_complex()
max.graph.ops.as_interleaved_complex(x)

Reshapes the input symbolic tensor as complex from alternating (real, imag).

Parameters:

interleaved – A symbolic tensor representing complex numbers as alternating pairs of (real, imag) real-valued numbers. Its last dimension must have an even size.
x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
Returns:

A symbolic tensor representing the complex-valued tensor, but with the values pulled out as complex numbers. The result has the same dimensions for all dimensions except the last dimension, which is halved, and then a final dimension of size 2 representing the complex value.

Return type:

TensorValue

assert_same_device()
max.graph.ops.assert_same_device(*values, **named_values)

Parameters:

values (TensorValue | BufferValue)
named_values (TensorValue | BufferValue)
Return type:

None

atanh()
max.graph.ops.atanh(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

avg_pool2d()
max.graph.ops.avg_pool2d(input, kernel_size, stride=1, dilation=1, padding=0, ceil_mode=False, count_boundary=True)

Perform a 2D average pooling operation on the input tensor.

This function applies a 2D average pooling operation to the input tensor [N, H, W, C]. The pooling operation slides a window of size kernel_size over the input tensor, and computes the average value within each window.

Parameters:

input (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input tensor to perform the pooling operation on.
kernel_size (tuple[int | str | Dim | integer[Any], int | str | Dim | integer[Any]]) – The size of the sliding blocks.
stride (int | tuple[int, int]) – The stride of the sliding blocks in the input dimension.
dilation (int | tuple[int, int]) – The spacing between the kernel elements.
padding (int | tuple[int, int]) – 0-paddings to be added on both sides of the inputs.
ceil_mode (bool) – If true, use ceil instead of floor to compute the output shape.
count_boundary (bool) – If true, count the padding elements when computing the average.
Return type:

TensorValue

band_part()
max.graph.ops.band_part(x, num_lower=None, num_upper=None, exclude=False)

Masks out everything except a diagonal band of an input matrix.

Copies a tensor setting everything outside the central diagonal band of the matrices to zero, where all but the last two axes are effectively batches, and the last two axes define sub matrices.

Assumes the input has dimensions [I, J, …, M, N], then the output tensor has the same shape as the input, and the values are given by

out[i, j, ..., m, n] = in_band(m, n) * input[i, j,  ..., m, n].

with the indicator function:

in_band(m, n) = ((num_lower is None || (m - n) <= num_lower)) &&
                (num_upper is None || (n - m) <= num_upper))

Parameters:

input – The input to mask out.
num_lower (int | None) – The number of diagonal bands to include below the central diagonal. If None, include the entire lower triangle.
num_upper (int | None) – The number of diagonal bands to include above the central diagonal. If None, include the entire upper triangle.
exclude (bool) – If true, invert the selection of elements to mask. Elements in the band are set to zero.
x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
Returns:

A symbolic tensor value with the configured selection masked out to 0 values, and the remaining values copied from the input tensor.

Raises:

ValueError – If the input tensor rank is less than 2, or if num_lower/num_upper are out of bounds for statically known dimensions.

Return type:

TensorValue

broadcast_to()
max.graph.ops.broadcast_to(x, shape, out_dims=None)

Broadcasts a symbolic tensor.

Broadcasts the input tensor to the specified shape. Dimensions in the input must be one or match the target dimension.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue) – The input symbolic tensor to broadcast. This tensor may not contain any dynamic dimensions.
shape (TensorValue | Iterable[int | str | Dim | integer[Any]]) – The new shape as a list of dimensions. Dynamic dimensions are not allowed.
out_dims (Iterable[int | str | Dim | integer[Any]] | None) – Output dims used only for tensor-valued shape.
Returns:

A symbolic tensor with the same elements as the original tensor, but in a new shape. Its symbolic shape is the same as shape.

Raises:

ValueError – if a tensor-valued shape is passed without out_dims.

Return type:

TensorValue

buffer_create()
max.graph.ops.buffer_create(type)

Creates a buffer of the given type.

Parameters:

type (BufferType) – The type of the resulting BufferValue

Returns:

A new BufferValue of the requested type.

Return type:

BufferValue

buffer_load()
max.graph.ops.buffer_load(x)

Loads the input buffer into a tensor.

It loads the in-place mutable tensor to an immutable tensor graph value. This is semantically equivalent to a copy from the mutable tensor x to the mutable value-semantic tensor output.

Parameters:

x (BufferValue) – The buffer to be loaded to a tensor.

Returns:

A tensor graph value representing a copy of the buffer loaded.

Return type:

TensorValue

buffer_store()
max.graph.ops.buffer_store(destination, source)

Stores the input tensor into the in-out buffer.

It stores the immutable input tensor x in the mutable tensor y. This is semantically equivalent to a copy from x tensor to the y buffer.

Parameters:

x – The tensor to be stored in the buffer.
y – The buffer to store the tensor in.
destination (BufferValue | HasBufferValue)
source (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
Return type:

None

buffer_store_slice()
max.graph.ops.buffer_store_slice(destination, source, indices)

Stores the input tensor to into a slice in the input buffer.

It stores the immutable input tensor source in the mutable tensor destination. This is semantically equivalent to a copy from source tensor to a slice in the destination buffer at index specified by indices.

Parameters:

destination (BufferValue | HasBufferValue) – The buffer to store the tensor in.
source (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The tensor to be stored in the buffer.
indices (Sequence[TensorValue | int | slice | tuple[slice, int | str | Dim | integer[Any]] | builtins.ellipsis]) – The index in the buffer where the tensor should be stored
Return type:

None

call()
max.graph.ops.call(graph, *args, prefix='')

Call a graph with the provided arguments and return its results.

This function invokes a previously defined graph, passing in the provided arguments and the current chain value, and returns the results.

The body of the graph is ultimately inlined into the caller, so the chain value is only used for serialization if the subgraph’s body contains an operation that makes use of it in the first place.

The current advantage of using subgraphs is that it offers a way to improve compile times for operations that are used repeatedly in a model. As a secondary benefit, it also makes the IR more readable by allowing control flow to be expressed in a more natural way.

Parameters:

graph (Graph) – The graph to call
*args (Value[Any]) – Arguments to pass to the called graph
prefix (str) – Prefix to add to the names of any weights in the subgraph
Returns:

Either a single Value or a list of Values representing the graph outputs (excluding the chain value which is handled internally)

Return type:

list[Value[Any]]

cast()
max.graph.ops.cast(x, dtype)

Casts a symbolic tensor to a different data type.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue) – The input tensor to cast.
dtype (DType) – The target dtype to which the tensor is cast.
Returns:

A new symbolic tensor with the same shape as the input and the specified dtype.

Return type:

TensorValue

chunk()
max.graph.ops.chunk(x, chunks, axis=0)

Chunk the tensor into an exact number of chunks along the specified dim.

Example:

>>> a = TensorValue([1, 2, 3, 4, 5])
>>> chunk(a, 2, 0)
[TensorValue([1, 2]), TensorValue([3, 4])]

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The tensor to chunk.
chunks (int) – The number of chunks to split the tensor into. chunks must statically evenly divide x.shape[axis].
axis (int) – The axis to split the tensor along.
Returns:

A list of chunks tensors.

Return type:

list[TensorValue]

concat()
max.graph.ops.concat(original_vals, axis=0)

Concatenates a list of symbolic tensors along an axis.

Parameters:

original_vals (Iterable[Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray]) – A list of symbolic tensor values. Each tensor must have the same dtype and rank, and must have the same dimension size for each dimension other than axis.
axis (int) – The axis to concatenate along. If negative, indexes relative to the end of the tensor shape. For instance, concat(vs, -1) will concat along the last dimension.
Returns:

A new symbolic tensor representing the concatenation result. It will have the same rank as each input tensor, and its dimensions will be the same as each input tensor’s for each dimension other than axis, which will have size equal to the sum of all tensor’s size for that dimension.

Return type:

TensorValue

cond()
max.graph.ops.cond(pred, out_types, then_fn, else_fn)

Conditionally execute one of two branches based on a boolean predicate.

Both branches must return the same number and types of values as specified in out_types. Buffer mutations in branches are tracked automatically through the chain mechanism.

Examples:

Basic conditional with return values:
def then_fn():
    return ops.constant(1, DType.int32, device=DeviceRef.CPU())
def else_fn():
    return ops.constant(0, DType.int32, device=DeviceRef.CPU())
​
result = ops.cond(
    pred,
    [TensorType(DType.int32, [], device=device)],
    then_fn,
    else_fn
)

Conditional with buffer mutations:
def then_fn():
    ops.inplace_custom("increment", device=buffer.device, values=[buffer])
def else_fn():
    ops.inplace_custom("decrement", device=buffer.device, values=[buffer])
​
ops.cond(pred, None, then_fn, else_fn)

:: :param pred: Boolean scalar tensor of type DType.bool determining branch execution :param out_types: Expected output types for both branches. Use None for branches that don’t return values :param then_fn: Callable executed when pred is True. Must return values matching out_types if out_types is not None :param else_fn: Callable executed when pred is False. Must return values matching out_types if out_types is not None

Returns:

List of output values from executed branch. Returns empty list when out_types is None

Raises:

ValueError – If branches return different numbers of results or result types don’t match out_types

Parameters:

pred (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
out_types (Iterable[Type[Any]] | None)
then_fn (Callable[[], Iterable[Value[Any]] | Value[Any] | None])
else_fn (Callable[[], Iterable[Value[Any]] | Value[Any] | None])
Return type:

list[TensorValue]

NOTE
Buffer operations in branches automatically update the global chain state to maintain mutation ordering constraints

constant()
max.graph.ops.constant(value, dtype=None, device=None)

Adds a node representing a constant operation.

The value of this constant will have the type TensorType with the same shape as value. If value is a scalar type, it will create a TensorType with 0 dimensions.

The constant will be loaded with the specified dtype. If the constant does not fit within the specified dtype, an error is raised.

Warning: Loading the constant could result in precision loss. For example, loading 16777217 as a float32 will result in 16777216.0.

Parameters:

value (DLPackArray | Sequence[float | number[Any] | Sequence[Number | NestedArray]] | float | number[Any]) – The constant’s value.
dtype (DType | None) – The constant tensor’s element type.
device (Device | DeviceRef | None) – The device the constant lives on.
Returns:

A graph value containing the constant data as an attribute.

Return type:

TensorValue

constant_external()
max.graph.ops.constant_external(name, type)

Registers an external constant (weight) in the graph of a given type.

Two external constants with the same name and type refer to the same weight.

Two external constants with the same name and different types are incompatible and will fail compilation.

Parameters:

name (str) – The name of the external constant. This should be the fully-qualified weight name and must be unique.
type (TensorType) – The type of the constant value.
Returns:

A tensor value of the specified type, representing the weight value associated with the name at compile time.

Return type:

TensorValue

conv2d()
max.graph.ops.conv2d(x, filter, stride=(1, 1), dilation=(1, 1), padding=(0, 0, 0, 0), groups=1, bias=None, input_layout=ConvInputLayout.NHWC, filter_layout=FilterLayout.RSCF)

Computes the 2-D convolution product of the input with the given filter, bias, strides, dilations, paddings, and groups.

The op supports 2-D convolution, with the following layout assumptions:

input x has NHWC layout, i.e., (batch_size, height, width, in_channels)
filter has layout RSCF, i.e., (height, width, in_channels / num_groups, out_channels)
bias has shape (out_channels,)
The padding values are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before, pad_dim2_after…) and represent padding 0’s before and after the indicated spatial dimensions in input. In 2-D convolution, dim1 here represents H and dim2 represents W. In Python like syntax, padding a 2x3 spatial input with [0, 1, 2, 1] would yield:

input = [
  [1, 2, 3],
  [4, 5, 6]
]
## Shape is 2x3

padded_input = [
  [0, 0, 1, 2, 3, 0],
  [0, 0, 4, 5, 6, 0],
  [0, 0, 0, 0, 0, 0]
]
## Shape is 3x6

This op currently only supports strides and padding on the input.

Parameters:

input – An NHWC input tensor to perform the convolution upon.
filter (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The convolution filter in RSCF layout: (height, width, in_channels / num_groups, out_channels).
stride (tuple[int, int]) – The stride of the convolution operation.
dilation (tuple[int, int]) – The spacing between the kernel points.
padding (tuple[int, int, int, int]) – The amount of padding applied to the input.
groups (int) – When greater than 1, divides the convolution into multiple parallel convolutions. The number of input and output channels must both be divisible by the number of groups.
x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
bias (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray | None)
input_layout (ConvInputLayout)
filter_layout (FilterLayout)
Returns:

A symbolic tensor value with the convolution applied.

Return type:

TensorValue

conv2d_transpose()
max.graph.ops.conv2d_transpose(x, filter, stride=(1, 1), dilation=(1, 1), padding=(0, 0, 0, 0), output_paddings=(0, 0), bias=None, input_layout=ConvInputLayout.NHWC, filter_layout=FilterLayout.RSCF)

Computes the 2-D deconvolution of the input with the given filter, strides, dilations, paddings, and groups.

The op supports the transpose (gradient) of convolution, with the following layout assumptions: (note the out_channel is w.r.t. the original convolution)

input x has NHWC layout, i.e., (batch_size, height, width, in_channels)
filter has layout RSCF, i.e., (kernel_height, kernel_width, out_channels, in_channels)
bias has shape (out_channels,)
The padding values are expected to take the form in the form [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]].

This op effectively computes the gradient of a convolution with respect to its input (as if the original convolution operation had the same filter and hyperparameters as this op). A visualization of the computation can be found in https://d2l.ai/chapter_computer-vision/transposed-conv.html.

The padding values are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before, pad_dim2_after…) and represent padding 0’s before and after the indicated spatial dimensions in input. In 2D ConvTranspose, dim1 here represents H_out and dim2 represents W_out. In python like syntax, padding a 2x4 spatial output with [0, 1, 2, 1] would yield:

output = [
  [1, 2, 3, 4],
  [5, 6, 7, 8]
]
## Shape is 2x4

padded_input = [
  [3],
]
## Shape is 1x1

Parameters:

input – An NHWC input tensor to perform the convolution upon.
filter (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The convolution filter in RSCF layout: (height, width, out_channels, in_channels).
stride (tuple[int, int]) – The stride of the sliding window for each dimension of input. If a single value is given it is replicated in the H and W dimension. By default the N and C dimensions are set to 0.
dilation (tuple[int, int]) – The spacing between the kernel points.
padding (tuple[int, int, int, int]) – The amount of padding applied to the input.
output_paddings (tuple[int, int]) – this argument is meant to resolve the ambiguity of multiple potential output shapes when any stride is greater than 1. Basically, we’ll add output_paddings[i] number of zeros at the end of output’s ith axis. We only support output_paddings = 0.
bias (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray | None) – tensor of shape (out_channels,)
x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
input_layout (ConvInputLayout)
filter_layout (FilterLayout)
Returns:

A symbolic tensor value with the convolution applied.

Return type:

TensorValue

conv3d()
max.graph.ops.conv3d(x, filter, stride=(1, 1, 1), dilation=(1, 1, 1), padding=(0, 0, 0, 0, 0, 0), groups=1, bias=None, input_layout=ConvInputLayout.NHWC, filter_layout=FilterLayout.QRSCF)

Computes the 3-D convolution product of the input with the given filter, strides, dilations, paddings, and groups.

The op supports 3-D convolution, with the following layout assumptions:

input has NDHWC layout, i.e., (batch_size, depth, height, width, in_channels)
filter has layout RSCF, i.e., (depth, height, width, in_channels / num_groups, out_channels)
The padding values are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before, pad_dim2_after…) and represent padding 0’s before and after the indicated spatial dimensions in input. In 3-D convolution, dim1 here represents D, dim2 represents H and dim3 represents W. In Python like syntax, padding a 2x3 spatial input with [0, 1, 2, 1] would yield:

input = [
  [1, 2, 3],
  [4, 5, 6]
]
## Shape is 2x3

padded_input = [
  [0, 0, 1, 2, 3, 0],
  [0, 0, 4, 5, 6, 0],
  [0, 0, 0, 0, 0, 0]
]
## Shape is 3x6

This op currently only supports strides and padding on the input.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – An NDHWC input tensor to perform the convolution upon.
filter (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The convolution filter in RSCF layout: (depth, height, width, in_channels / num_groups, out_channels).
stride (tuple[int, int, int]) – The stride of the convolution operation.
dilation (tuple[int, int, int]) – The spacing between the kernel points.
padding (tuple[int, int, int, int, int, int]) – The amount of padding applied to the input.
groups (int) – When greater than 1, divides the convolution into multiple parallel convolutions. The number of input and output channels must both be divisible by the number of groups.
bias (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray | None)
input_layout (ConvInputLayout)
filter_layout (FilterLayout)
Returns:

A symbolic tensor value with the convolution applied. Output shape = (batch_size, depth, height, width, out_channels).

Return type:

TensorValue

cos()
max.graph.ops.cos(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

cumsum()
max.graph.ops.cumsum(x, axis=-1, exclusive=False, reverse=False)

Computes the cumulative sum of the input tensor along the given axis.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input tensor to sum over.
axis (int) – The axis along which to compute the sum. If negative, indexes from the last dimension. For example, a value of -1 will compute the sum along the last dimension.
exclusive (bool) – If set, start at 0 and exclude the final element. Otherwise, start with the first element. Said another way, cumsum computes [sum(x[…, :i, …]) for i in range(x.shape[axis])]. If exclusive is set, the bounds are instead range(1, x.shape[axis]).
reverse (bool) – If set, start from the end. In other words, the first element will be the total sum, with each element following counting downwards; or [sum(x[…, i:, …]) for i in range(x.shape[axis])].
Returns:

A symbolic tensor representing the result of the cumsum operation. The tensor will have the same type as the input tensor. The computed values will be the cumulative sum of the values along the given axis, according to the specified parameters:

if exclusive is set, the first value will be 0, and the last value will be excluded from the sum
if reverse is set, the sum will be computed starting at the back of the axis back to the front, rather than front-to-back
Return type:

TensorValue

custom()
max.graph.ops.custom(name, device, values, out_types, parameters=None)

Creates a node to execute a custom graph operation in the graph.

The custom op should be registered by annotating a function with the @compiler.register decorator.

Parameters:

name (str) – The op name provided to @compiler.register.
values (Sequence[Value[Any]]) – The op function’s arguments.
out_types (Sequence[Type[Any]]) – The list of op function’s return type.
parameters (Mapping[str, bool | int | str | DType] | None) – Dictionary of extra parameters expected by the kernel.
device (Device | DeviceRef) – Device that the op is assigned to. This becomes a target parameter to the kernel.
Returns:

Symbolic values representing the outputs of the op in the graph. These correspond 1:1 with the types passed as out_types.

Return type:

list[Value[Any]]

dequantize()
max.graph.ops.dequantize(encoding, quantized)

Dequantizes a quantized tensor to floating point.

NOTE: Currently this supports Q4_0, Q4_K, and Q6_K encodings only.

Parameters:

encoding (QuantizationEncoding) – The quantization encoding to use.
quantized (TensorValue) – The quantized tensor to dequantize.
Returns:

The dequantized result (a floating point tensor).

Return type:

TensorValue

div()
max.graph.ops.div(lhs, rhs)

Divides two symbolic tensors using true division (Python operator /).

For integer operands, this performs true division by promoting to float, matching Python’s / operator behavior. For floating-point operands, this performs standard floating-point division.

Creates a new op node to compute the division of two symbol tensor values and adds it to the graph, returning the symbolic result.

Parameters:

lhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The symbol to use as left side of the division.
rhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The symbol to use as right side of the division.
Returns:

A symbolic tensor value representing the output of the division. The result will have: : - floating-point dtype for integer operands, promoted dtype for mixed types

the same shape as the broadcast of the two input shapes.
Raises:

Error – If the input values’ shapes are not compatible for broadcasting.
Error – If one of the input values has an unsupported dtype.
Error – If the two symbols are parts of different graphs.
Return type:

TensorValue

equal()
max.graph.ops.equal(lhs, rhs)

Parameters:

lhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
rhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
Return type:

TensorValue

erf()
max.graph.ops.erf(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

exp()
max.graph.ops.exp(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

flatten()
max.graph.ops.flatten(x, start_dim=0, end_dim=-1)

Flattens the specified dims of a symbolic tensor.

The number and order of the elements in the tensor is unchanged. All dimensions from start_dim to end_dim (inclusive) are merged into a single output dim.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
start_dim (int)
end_dim (int)
Return type:

TensorValue

floor()
max.graph.ops.floor(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

fold()
max.graph.ops.fold(input, output_size, kernel_size, stride=1, dilation=1, padding=0)

Combines an array of sliding blocks into a larger containing tensor.

The input tensor must have shape (N, C * kernel_sizes, L) where N is the batch dimension, C is the number of channels, kernel_sizes is the product of the kernel sizes, and L is the number of local blocks.

The resulting output tensor will have shape (N, C, output_shape[0], output_shape[1]).

L, the number of blocks, must be equivalent to: prod((output_size[d] + 2 * padding[d] - dilation[d] * (kernel_size[d] - 1) - 1) / stride[d] + 1)

where d is over all spatial dimensions.

Parameters:

input (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The 3D tensor to fold with shape (N, C * kernel sizes, L).
output_size (tuple[int | str | Dim | integer[Any], int | str | Dim | integer[Any]]) – Spatial dimensions of the output tensor. Must be a tuple of two ints.
kernel_size (tuple[int | str | Dim | integer[Any], int | str | Dim | integer[Any]]) – The size of the sliding blocks. Must be a tuple of two ints.
stride (int | tuple[int, int]) – The stride of the sliding blocks in the input dimension (can be an int or a tuple of two ints).
dilation (int | tuple[int, int]) – The spacing between the kernel elements. (can be an int or a tuple of two ints).
padding (int | tuple[int, int]) – 0-paddings to be added on both sides of the inputs. (can be an int or a tuple of two ints).
Returns:

The folded 4D tensor with shape (N, C, output_shape[0], output_shape[1]).

Return type:

TensorValue

gather()
max.graph.ops.gather(input, indices, axis)

Selects elements out of an input tensor by index.

Parameters:

input (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input symbolic tensor to select elements from.
indices (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – A symbolic tensor of index values to use for selection.
axis (int) – The dimension which indices indexes from input. If negative, indexes relative to the end of the input tensor. For instance, gather(input, indices, axis=-1) will index against the last dimension of input.
Returns:

A new symbolic tensor representing the result of the gather operation.

Return type:

TensorValue

gather_nd()
max.graph.ops.gather_nd(input, indices, batch_dims=0)

Selects elements out of an input tensor by N-dimensional index.

This operation performs N-dimensional indexing into input using indices. Unlike gather(), which indexes along a single axis, gather_nd() allows indexing along multiple dimensions simultaneously.

input_shape = ["a", "b", "c", "d", "e"]
indices_shape = ["a", "f", 3]
input_type = TensorType(DType.bfloat16, input_shape)
indices_type = TensorType(DType.int32, indices_shape)
with Graph("gather_nd", input_types=[input_type, indices_type]) as graph:
    input, indices = graph.inputs
    gathered = ops.gather_nd(input, indices, batch_dims=1)
    print(gathered.type)
## Output: TensorType(dtype=DType.bfloat16, shape=["a", "f", "e"])

In this example:

batch_dims is 1, so there’s 1 shared dimension at the beginning.
indices has an additional dimension “f” which becomes part of the output.
The last dimension of indices is the index vector; values in this vector are interpreted to be indices into “b”, “c”, and “d”.
Since batch_dims (1) + index size (3) < input.rank (5), the remaining dimensions (in this case “e”) are sliced into the output as features.
Parameters:

input (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input symbolic tensor to select elements from.
indices (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – A symbolic tensor of index values to use for selection. The last dimension of this tensor must be static. This dimension will be used to index or slice into input immediately following batch_dims initial dimensions. The size of this index dimension is the number of dimensions it specifies.
batch_dims (int) – The number of leading batch dimensions shared by input and indices; 0 by default. input and indices must exactly match up to their first batch_dims dimensions. This function does not broadcast.
Returns:

A new symbolic tensor representing the result of the gather operation. The output will have the same dtype as input, and will have shape depending on the inputs, in this order:

input.shape[:batch_dims] – The “broadcast” dimensions (though note that this function does not broadcast). These dimensions must be identical between input and indices.
indices.shape[batch_dims:-1] – The “gather” dimensions; this allows multi-dimensional tensors of indices. The last dimension is the index vector.
input.shape[batch_dims + indices.shape[-1]:] – The “slice” dimensions. If batch_dims < input.rank - indices.shape[-1] (again, this last is the index vector), then any following dimensions of the inputs are taken entirely as though slicing.
Return type:

TensorValue

gelu()
max.graph.ops.gelu(x, approximate='none')

Computes the elementwise gelu of a symbolic tensor.

Creates a new op node to compute the elementwise gelu of a symbolic tensor and adds it to the graph, returning the symbolic result.

For approximate == "none", the exact gelu function is computed.

For approximate == "tanh", the approximation:

g
e
l
u
(
x
)
=
0.5
∗
x
∗
(
1.0
+
t
a
n
h
(
0.7978845608028654
∗
(
x
+
0.044715
∗
x
∗
∗
3
)
)
)
gelu(x)=0.5∗x∗(1.0+tanh(0.7978845608028654∗(x+0.044715∗x∗∗3)))
is used.

For approximate == "quick", the approximation:

g
e
l
u
(
x
)
=
s
i
g
m
o
i
d
(
1.702
∗
x
)
∗
x
gelu(x)=sigmoid(1.702∗x)∗x
is used.

Parameters:

value – The symbolic tensor to use as the input to the gelu computation.
x (TensorValue)
approximate (str)
Returns:

A new symbolic tensor value representing the output of the gelu value computation.

Raises:

Error – If the symbol doesn’t represent a tensor value.
ValueError – If the approximation method is invalid.
greater()
max.graph.ops.greater(lhs, rhs)

Parameters:

lhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
rhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
Return type:

TensorValue

greater_equal()
max.graph.ops.greater_equal(lhs, rhs)

Parameters:

lhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
rhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
Return type:

TensorValue

hann_window()
max.graph.ops.hann_window(window_length, device, periodic=True, dtype=float32)

Calculate a Hann window for a given length.

Hann window function:

H
[
n
]
=
1
/
2
[
1
−
c
o
s
(
2
∗
p
i
∗
n
/
(
N
−
1
)
)
]
H[n]=1/2[1−cos(2∗pi∗n/(N−1))]
where N is window_length.

Parameters:

window_length (int) – The length of the window.
device (DeviceRef) – The device to run the operation on.
periodic (bool) – bool flag determines whether the returned window trims off the last duplicate value from the symmetric window and is ready to be used as a periodic window with functions like stft(). hann_window(L, periodic=True) == hann_window(L + 1, periodic=False)[:-1])
dtype (DType) – The desired data type of the output tensor.
Returns:

A 1-D tensor of size (window_length,) containing the window.

Raises:

ValueError – If window_length is negative.
TypeError – If window_length is not an integer.
Return type:

TensorValue

inplace_custom()
max.graph.ops.inplace_custom(name, device, values, out_types=None, parameters=None)

Creates a node to execute an in-place custom graph operation in the graph.

The custom op should be registered by annotating a function with the @compiler.register decorator.

Parameters:

name (str) – The op name provided to @compiler.register.
device (DeviceRef) – Device that the op is assigned to. This becomes a target parameter to the kernel.
values (Sequence[Value[Any]]) – The op function’s arguments.
parameters (dict[str, bool | int | str | DType] | None) – Dictionary of extra parameters expected by the kernel.
out_types (Sequence[Type[Any]] | None)
Return type:

list[Value[Any]]

irfft()
max.graph.ops.irfft(input_tensor, n=None, axis=-1, normalization=Normalization.BACKWARD, input_is_complex=False, buffer_size_mb=512)

Compute the inverse real FFT of the input tensor.

Parameters:

input_tensor (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue) – The input tensor to compute the inverse real FFT of.
n (int | None) – The size of the output tensor. Must be an int, and cannot be a symbolic Tensor. The input tensor will be padded or truncated to n // 2 + 1 along the specified axis.
axis (int) – The axis to compute the inverse real FFT of.
normalization (Normalization | str) – The normalization to apply to the output tensor. Can be “backward”, “ortho”, or “forward”. When “backward”, the output is divided by n. When “ortho”, the output is divided by sqrt(n). When “forward”, no normalization is applied.
input_is_complex (bool) – Whether the input tensor is already interleaved complex. The last dimension of the input tensor must be 2, and is excluded from the dimension referred to by axis.
buffer_size_mb (int) – The estimated size of a persistent buffer to use for storage of intermediate results. Needs to be the same across multiple calls to irfft within the same graph. Otherwise, multiple buffers will be allocated.
Returns:

The inverse real FFT of the input tensor. The shape of the output tensor is the same as the shape of the input tensor, except for the axis that the inverse real FFT is computed over, which is replaced by n.

is_inf()
max.graph.ops.is_inf(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

is_nan()
max.graph.ops.is_nan(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

layer_norm()
max.graph.ops.layer_norm(input, gamma, beta, epsilon)

Performs layer normalization.

Parameters:

input (TensorValue) – The input tensor to normalize.
gamma (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The gamma parameter of the normalization.
beta (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The beta parameter of the normalization.
epsilon (float) – The epsilon parameter of the normalization.
Returns:

A graph tensor value with the normalization applied.

Raises:

ValueError – If gamma size doesn’t match the last dimension of input.
ValueError – If beta size doesn’t match the last dimension of input.
ValueError – If epsilon is not positive.
Return type:

TensorValue

log()
max.graph.ops.log(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

log1p()
max.graph.ops.log1p(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

logical_and()
max.graph.ops.logical_and(lhs, rhs)

Parameters:

lhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
rhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
Return type:

TensorValue

logical_not()
max.graph.ops.logical_not(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

logical_or()
max.graph.ops.logical_or(lhs, rhs)

Parameters:

lhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
rhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
Return type:

TensorValue

logical_xor()
max.graph.ops.logical_xor(lhs, rhs)

Parameters:

lhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
rhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
Return type:

TensorValue

logsoftmax()
max.graph.ops.logsoftmax(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

masked_scatter()
max.graph.ops.masked_scatter(input, mask, updates, out_dim)

Creates a new symbolic tensor where the updates are written to input where mask is true.

Parameters:

input (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input symbolic tensor to write elements to.
mask (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – A symbolic tensor of boolean values to update.
updates (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – A symbolic tensor of elements to write to input.
out_dim (int | str | Dim | integer[Any]) – The new data-dependent dimension.
Returns:

A new symbolic tensor representing the result of the masked_scatter operation.

Return type:

TensorValue

matmul()
max.graph.ops.matmul(lhs, rhs)

Computes the matrix multiplication of two tensor graph values.

Performs general matrix multiplication with broadcasting.

If the lhs is 1D, it will be reshaped to 1xD. If the rhs is 1D, it will be reshaped to Dx1. In both cases, the additional 1 dimensions will be removed from the output shape.

For the multiplication, the innermost (rightmost) 2 dimensions are treated as a matrix. The lhs matrix will have the shape MxK. The rhs matrix will have the shape KxN. The output will have the shape MxN The K dimensions must be equivalent in both matrices.

The remaining outer dimensions will be broadcasted.

Parameters:

lhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The left-hand-side of the matmul.
rhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The right-hand-side of the matmul.
location – An optional location for a more specific error message.
Returns:

A tensor graph value representing he result of broadcasting the two matrices together and then performing a matrix multiply along the innermost two dimension of each tensor.

Return type:

TensorValue

max()
max.graph.ops.max(x, y=None, /, axis=None)

Overload for ops.elementwise.max and ops.reduction.max.

If two tensors are provided, axis is ignored and returns an elementwise maximum.
If one tensor is provided, compute ops.reduction.max on the tensor and axis.
Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
y (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray | None)
axis (int | None)
Return type:

TensorValue

max_pool2d()
max.graph.ops.max_pool2d(input, kernel_size, stride=1, dilation=1, padding=0, ceil_mode=False)

Perform a 2D max pooling operation on the input tensor.

This function applies a 2D max pooling operation to the input tensor [N, H, W, C]. The pooling operation slides a window of size kernel_size over the input tensor, and selects the maximum value within each window.

Parameters:

input (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input tensor to perform the pooling operation on.
kernel_size (tuple[int | str | Dim | integer[Any], int | str | Dim | integer[Any]]) – The size of the sliding blocks.
stride (int | tuple[int, int]) – The stride of the sliding blocks in the input dimension.
dilation (int | tuple[int, int]) – The spacing between the kernel elements.
padding (int | tuple[int, int]) – 0-paddings to be added on both sides of the inputs.
ceil_mode (bool) – If true, use ceil instead of floor to compute the output shape.
Return type:

TensorValue

mean()
max.graph.ops.mean(x, axis=-1)

Reduces a symbolic tensor using a mean operation.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input tensor for the operation.
axis (int) – The axis along which to compute the reduction. If negative, indexes from the last dimension. For example, a value of -1 will compute the reduction along the last dimension.
Returns:

A symbolic tensor representing the result of the mean operation. The tensor will have the same rank as the input tensor, and the same shape except along the axis dimension which will have size 1.

Return type:

TensorValue

min()
max.graph.ops.min(x, y=None, /, axis=None)

Overload for ops.elementwise.min and ops.reduction.min.

If two tensors are provided, axis is ignored and returns an elementwise minimum.
If one tensor is provided, compute ops.reduction.min on the tensor and axis.
Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
y (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray | None)
axis (int | None)
Return type:

TensorValue

mod()
max.graph.ops.mod(lhs, rhs)

Parameters:

lhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
rhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
Return type:

TensorValue

mul()
max.graph.ops.mul(lhs, rhs)

Parameters:

lhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
rhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
Return type:

TensorValue

negate()
max.graph.ops.negate(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

nonzero()
max.graph.ops.nonzero(x, out_dim)

Returns the indices of all nozero elements in a tensor.

Returns a tensor of indices of the nonzero values in the given tensor. The return value is a 2D tensor of shape [out_dim x rank_in], where out_dim is the number of nonzero elements in the input tensor, and rank_in is the rank of the input tensor. Indices are generated in row-major order.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input symbolic tensor.
out_dim (int | str | Dim | integer[Any]) – The newly generated dimension that is sized for the number of nonzero elements.
Returns:

A symbolic tensor of indices

Return type:

TensorValue

not_equal()
max.graph.ops.not_equal(lhs, rhs)

Parameters:

lhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
rhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
Return type:

TensorValue

outer()
max.graph.ops.outer(lhs, rhs)

Computes the outer product of two symbolic vectors.

Parameters:

lhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The left side of the product. Whatever its shape, it will be flattened to a rank-1 vector.
rhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The right side of the product. Whatever its shape, it will be flattened to a rank-1 vector. Must have the same number of elements as lhs.
Returns:

A symbolic tensor representing the outer product of the two input vectors. It will have rank 2, with the dimension sizes being the number of elements of lhs and rhs respectively.

Return type:

TensorValue

pad()
max.graph.ops.pad(input, paddings, mode='constant', value=0)

Pads a tensor with constant values.

Adds padding to the input tensor using the specified padding values. Currently only constant padding mode is supported.

Parameters:

input (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input tensor to pad.
paddings (Iterable[int]) – Sequence of padding values. The padding values are applied symmetrically to each dimension. For a tensor with rank N, paddings should contain 2*N values: [pad_before_dim0, pad_after_dim0, pad_before_dim1, pad_after_dim1, …].
mode (Literal['constant']) – The padding mode. Currently only “constant” is supported.
value (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The constant value to use for padding.
Return type:

TensorValue

permute()
max.graph.ops.permute(x, dims)

Permutes all dimensions of a symbolic tensor.

Parameters:

input – The input symbolic tensor to transpose.
dims (list[int]) – The desired ordering of the dimensions in the output tensor.
x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
Returns:

A new symbolic tensor with the dimensions permuted to match the passed in order. It has the same elements and dtype, but the order of the elements is different according to the permutation.

Return type:

TensorValue

pow()
max.graph.ops.pow(lhs, rhs)

Parameters:

lhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
rhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
Return type:

TensorValue

print()
max.graph.ops.print(value, label='debug_tensor')

Prints the value of a tensor or a string during graph execution.

This function is used to output the current value of a tensor and is primarily used for debugging purposes within the context of the Max Engine and its graph execution framework. This is particularly useful to verify the intermediate results of your computations are as expected.

By printing the tensor values, you can visualize the data flowing through the graph, which helps in understanding how the operations are transforming the data.

When labeling the function you can assign the output, making it easier to identify which tensor’s value is being printed, especially when there are multiple print statements in a complex graph.

def add_tensors(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    input_type = TensorType(dtype=DType.float32, shape=(1,), device=DeviceRef.CPU())
    with Graph(
        "simple_add_graph", input_types=(input_type, input_type)
    ) as graph:
        lhs, rhs = graph.inputs
        out = ops.add(lhs, rhs)
        ops.print(out, label="addition_output")  # Pass the output tensor here

        graph.output(out)
        print("final graph:", graph)

Parameters:

value (str | TensorValue) – The value to print. Can be either a string or a TensorValue.
label (str) – A label to identify the printed value. Defaults to debug_tensor.
Return type:

None

qmatmul()
max.graph.ops.qmatmul(encoding, config, lhs, *rhs)

Performs matrix multiplication between floating point and quantized tensors.

This quantizes the lhs floating point value to match the encoding of the rhs quantized value, performs matmul, and then dequantizes the result. Beware that, compared to a regular matmul op, this one expects the rhs value to be transposed. For example, if the lhs shape is [32, 64], and the quantized rhs shape is also [32, 64], then the output shape is [32, 32].

That is, this function returns the result from:

dequantize(quantize(lhs) @ transpose(rhs))

The last two dimensions in lhs are treated as matrices and multiplied by rhs (which must be a 2D tensor). Any remaining dimensions in lhs are broadcast dimensions.

NOTE: Currently this supports Q4_0, Q4_K, and Q6_K encodings only.

Parameters:

encoding (QuantizationEncoding) – The quantization encoding to use.
lhs (TensorValue) – The non-quantized, left-hand-side of the matmul.
*rhs (TensorValue) – The transposed and quantized right-hand-side of the matmul and auxiliary tensor (if has). Must be rank 2 and in a supported [quantization encoding] (/max/api/mojo/graph/quantization/).
config (QuantizationConfig | None)
Returns:

The dequantized result (a floating point tensor).

Return type:

TensorValue

range()
max.graph.ops.range(start, stop, step=1, out_dim=None, *, dtype, device)

Creates a sequence of numbers. The sequence goes from start with increments of size step up to (but not including) stop. All arguments are mandatory and must have the same element type.

Note the following restrictions on input values:

step must be non-zero
stop - start must be zero or have the same sign as step
Parameters:

start (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The start of the range to generate.
stop (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The range will be generated up to, but not including, this value.
step (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The step size for the range.
out_dim (int | str | Dim | integer[Any] | None) – The expected output dimensions returned by the range op. These will be assert at graph execution time to be correct.
device (Device | DeviceRef) – Device of the result tensor.
dtype (DType) – Data type of the result tensor. If not specified, defaults to float32 for numeric inputs or infers from tensor inputs.
Returns:

A symbolic tensor value containing the defined range of values.

Return type:

TensorValue

rebind()
max.graph.ops.rebind(x, shape, message='', layout=None)

Rebinds a symbolic tensor to a specified set of dimensions.

This does not mutate the symbolic tensor passed in, but instead adds a runtime assert that the input symbolic shape is equivalent to out_dims shape. For example, if the input tensor shape has dynamic/unknown sizes, this will assert a fixed sizes that may be required for a subsequent operation.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input symbolic tensor to rebind.
shape (Iterable[int | str | Dim | integer[Any]]) – The symbolic shape to assert for x, as a list of Dim values.
message (str) – The message printed if the rebind fails at runtime.
layout (FilterLayout | None) – A layout of the weights used by some operations like conv.
Returns:

A symbolic tensor with the same elements and shape as the given tensor, but with the symbolic shape asserted to out_dims.

Return type:

TensorValue

relu()
max.graph.ops.relu(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

repeat_interleave()
max.graph.ops.repeat_interleave(x, repeats, axis=None, out_dim=None)

Repeats elements of a tensor along the given dimension.

Modeled after torch.repeat_interleave, with the constraint that

For example, given repeats=2 and the following input:

## Input tensor with shape (2, 2)
input = TensorValue(x)  # Contains [[1.0, 2.0], [3.0, 4.0]]

repeat_interleave with axis=0:

## Output tensor with shape (4, 2)
output = repeat_interleave(input, repeats=2, axis=0)
## Contains [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]

repeat_interleave with axis=1:

## Output tensor with shape (2, 4)
output = repeat_interleave(input, repeats=2, axis=1)
## Contains [[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]]

repeat_interleave with axis=None (the default):

repeat_interleave with repeats=[2, 3] and axis=0:

repeat_value = TensorValue([2, 3])

## Output tensor with shape (5, 2)
output = repeat_interleave(input, repeats=repeat_value, axis=0)
## Contains [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]

## Output tensor with shape (8,)
output = repeat_interleave(input, repeats=2)  # axis = None
## Contains [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input tensor.
repeats (int | TensorValue) – The number of repetitions for each element.
axis (int | None) – The dimension along which to repeat values. If axis is not specified or None (the default), flatten the input array and repeat the flattened values.
out_dim (int | str | Dim | integer[Any] | None)
Returns:

A symbolic tensor with the elements interleaved.

Raises:

ValueError – If repeats non-positive or if axis is out of range.

Return type:

TensorValue

reshape()
max.graph.ops.reshape(x, shape)

Reshapes a symbolic tensor.

The number and order of the elements in the tensor is unchanged. In other words, if you were to iterate over elements in the tensor by major dimension to minor dimension, the iteration order would stay the same.

If a value of -1 is present in the shape, that dimension becomes an automatically calculated dimension collecting all unspecified dimensions. Its length becomes the number of elements in the original tensor divided by the product of elements of the reshape.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input symbolic tensor to reshape. This tensor may not contain any dynamic dimensions.
shape (Iterable[int | str | Dim | integer[Any]]) – The new shape as a list of dimensions. Dynamic dimensions are not allowed. A single dimension may be -1.
Returns:

A symbolic tensor with the same elements as the original tensor, but in a new shape. Its symbolic shape is the same as shape.

Raises:

ValueError – if input and target shapes’ number of elements mismatch.

Return type:

TensorValue

resize()
max.graph.ops.resize(input, shape, interpolation=InterpolationMode.BILINEAR)

Resize the input tensor to the given shape.

This function resizes a tensor using the specified interpolation method. The tensor is expected to have NCHW format (batch, channels, height, width).

Parameters:

input (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input tensor to resize. Must have rank 4 in NCHW format.
shape (Iterable[int | str | Dim | integer[Any]]) – Desired output shape of length 4 corresponding to (N, C, H, W).
interpolation (InterpolationMode) – Desired interpolation enum defined by InterpolationMode. Default is InterpolationMode.BILINEAR. Currently only BICUBIC is supported.
Returns:

A resized tensor with the shape specified by the shape argument.

Raises:

ValueError – If the input doesn’t have rank 4, shape has wrong number of elements, or unsupported interpolation mode is specified.
NotImplementedError – If single integer size or non-BICUBIC interpolation mode is specified.
Return type:

TensorValue

round()
max.graph.ops.round(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

rsqrt()
max.graph.ops.rsqrt(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

scatter()
max.graph.ops.scatter(input, updates, indices, axis=-1)

Creates a new symbolic tensor where the updates are written to input according to indices.

Parameters:

input (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input symbolic tensor to write elements to.
updates (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – A symbolic tensor of elements to write to input.
indices (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The positions in input to update.
axis (int) – The axis along which indices indexes into.
Returns:

A new symbolic tensor representing the result of the scatter operation.

Return type:

TensorValue

scatter_nd()
max.graph.ops.scatter_nd(input, updates, indices)

Creates a new symbolic tensor where the updates are scattered into input at specified indices.

Parameters:

input (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input symbolic tensor to write elements to.
updates (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – A symbolic tensor of elements to write to input.
indices (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – A tensor of indices specifying where to write updates. Shape should be [num_updates, rank] for full indexing or [num_updates, k] for partial indexing where k < rank.
Returns:

A new symbolic tensor representing the result of the scatter_nd operation.

Return type:

TensorValue

shape_to_tensor()
max.graph.ops.shape_to_tensor(shape)

Converts a shape to a tensor.

This is useful for using a shape attribute in an op that expects a tensor value.

Parameters:

shape (Iterable[int | str | Dim | integer[Any]]) – the shape attribute of a tensor value.

Returns:

The TensorValue containing the same value as shape.

Return type:

TensorValue

Example:

>>> x = ops.constant(np.zeros((1,)), DType.int64, device=DeviceRef.CPU())
>>> result = ops.stack([
...     x,
...     ops.shape_to_tensor(x.shape),
... ])
TensorValue(dtype=int64, shape=[StaticDim(dim=2), StaticDim(dim=1)])

sigmoid()
max.graph.ops.sigmoid(x)

Computes the elementwise sigmoid of a symbolic tensor.

Creates a new op node to compute the elementwise sigmoid of a symbolic tensor and adds it to the graph, returning the symbolic result.

Parameters:

value – The symbolic tensor to use as the input to the sigmoid computation.
x (TensorValue)
Returns:

A new symbolic tensor value representing the output of the sigmoid value computation.

Raises:

Error – If the symbol doesn’t represent a tensor value.

Return type:

TensorValue

silu()
max.graph.ops.silu(x)

Computes the elementwise silu of a symbolic tensor.

Creates a new op node to compute the elementwise silu of a symbolic tensor and adds it to the graph, returning the symbolic result.

silu is defined as silu(x) = x * sigmoid(x).

Parameters:

value – The symbolic tensor to use as the input to the silu computation.
x (TensorValue)
Returns:

A new symbolic tensor value representing the output of the silu value computation.

Raises:

Error – If the symbol doesn’t represent a tensor value.

sin()
max.graph.ops.sin(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

slice_tensor()
max.graph.ops.slice_tensor(x, indices)

Slices out a subtensor view of the input tensor based on indices.

The semantics of slice_tensor() follow NumPy slicing semantics with the following restrictions:

Slice indices must not index out of [-dim - 1, dim - 1] for negative step, or [-dim, dim] for positive step.
## Reverse a tensor.
slice_tensor(x, [slice(None, None, -1)])
## Unsqueeze the second last dimension of a tensor.
slice_tensor(x, [..., None, slice(None)])

Returns:

The sliced subtensor of x.

Parameters:

x (TensorValue)
indices (SliceIndices)
Return type:

TensorValue

softmax()
max.graph.ops.softmax(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

split()
max.graph.ops.split(x, split_sizes, axis=0)

Splits the input tensor into multiple tensors along a given dimension.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input symbolic tensor to split.
split_sizes (Sequence[int | str | Dim | integer[Any]]) – Sizes of each output tensor. Must add up to the split dimension axis.
axis (int) – Dimension to split the input tensor. Must have a statically known dimension size.
Returns:

A list of tensors with the same length as split_sizes, where each tensor has the same shape as the input except along the split dimension axis, where the size is given by the corresponding element in split_sizes.

Return type:

list[TensorValue]

sqrt()
max.graph.ops.sqrt(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

squeeze()
max.graph.ops.squeeze(x, axis)

Removes a size-1 dimension from a symbolic tensor.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input symbolic tensor to squeeze.
axis (int) – The dimension to remove from the input’s shape. If negative, this indexes from the end of the tensor. For example, squeeze(v, -1) squeezes the last dimension.
Returns:

A symbolic tensor with the same number of elements as the input tensor, and whose rank is 1 less than the rank of the input tensor.

Return type:

TensorValue

stack()
max.graph.ops.stack(values, axis=0)

Stacks a list of tensors along a new axis.

Parameters:

values (Iterable[Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray]) – A list of symbolic tensor values. Each tensor must have the same dtype and rank, and must have the same dimension size for each dimension.
axis (int) – The axis to concatenate along. If negative, indexes relative to the end of the tensor shape plus 1. For instance, stack(vs, -1) will create and stack along a new axis as the last dimension, aad stack(vs, -2) will create and stack along a new dimension which is inserted immediately before the last dimension.
Returns:

A new symbolic tensor representing the result of the stack. It will have rank n+1 where n is the rank of each input tensor. Its size on each dimension other than axis will be the same as each input tensors’, with the new axis inserted. Along the new dimension it will have size len(values).

Return type:

TensorValue

sub()
max.graph.ops.sub(lhs, rhs)

Parameters:

lhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
rhs (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
Return type:

TensorValue

sum()
max.graph.ops.sum(x, axis=-1)

Reduces a symbolic tensor using a sum operation.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input tensor for the operation.
axis (int) – The axis along which to compute the reduction. If negative, indexes from the last dimension. For example, a value of -1 will compute the reduction along the last dimension.
Returns:

A symbolic tensor representing the result of the sum operation. The tensor will have the same rank as the input tensor, and the same shape except along the axis dimension which will have size 1.

Return type:

TensorValue

tanh()
max.graph.ops.tanh(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

tile()
max.graph.ops.tile(x, repeats)

Returns a new Tensor as the result of copying the input tensor N_i times on each dimension, where N_i = repeats[i].

The i-th dimension of output shape will be the ith dimension of input shape multiplied by N_i.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)
repeats (Iterable[int | str | Dim | integer[Any]])
Return type:

TensorValue

top_k()
max.graph.ops.top_k(input, k, axis=-1)

Returns tensor with only top K values along given axis.

Parameters:

input (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input tensor from which to select top k.
k (int) – The number of values to select from input.
axis (int) – The axis from which to select top k.
Returns:

Top K values, Top K indices

Return type:

tuple[TensorValue, TensorValue]

transfer_to()
max.graph.ops.transfer_to(x, device)

Device-to-Device transfer operation.

This op transfers the input tensor from its current device over to another. A device represents a computation unit, like CPU, GPU, etc. This op is useful for instance when working with accelerators, like GPU, where for instance one may need to move data from GPU to GPU, or from one GPU to CPU.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue) – The input tensor to transfer.
device (Device | DeviceRef) – The device to transfer to.
Returns:

A tensor transferred to device specified.

Return type:

TensorValue

transpose()
max.graph.ops.transpose(x, axis_1, axis_2)

Transposes two axes of a symbolic tensor. For more information, see transpose().

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input symbolic tensor to transpose.
axis_1 (int) – One of the two axes to transpose. If negative, this indexes from the end of the tensor. For example, transpose(v, -1, -2) transposes the last two axes.
axis_2 (int) – The other axis to transpose. May also be negative to index from the end of the tensor.
Returns:

A new symbolic tensor with the two specified axes transposed. It has the same elements and dtype, but the order of the elements is different according to the transposition.

Return type:

TensorValue

trunc()
max.graph.ops.trunc(x)

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray)

Return type:

TensorValue

unsqueeze()
max.graph.ops.unsqueeze(x, axis)

Inserts a size-1 dimension into a symbolic tensor.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The input symbolic tensor to unsqueeze.
axis (int) – The index at which to insert a new dimension into the input’s shape. Elements at that index or higher are shifted back. If negative, it indexes relative 1 plus the rank of the tensor. For example, unsqueeze(v, -1) adds a new dimension at the end, and unsqueeze(v, -2) inserts the dimension immediately before the last dimension.
Returns:

A symbolic tensor with the same number of elements as the input tensor, whose rank is 1 larger than the rank of the input tensor. The result’s shape at the axis dimension is a static dimension of size 1.

Return type:

TensorValue

where()
max.graph.ops.where(condition, x, y)

Returns condition ? x : y (element-wise), where cond, x and y are input tensors.

Parameters:

condition (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – The condition tensor to use for selecting elementwise values. This tensor must have a boolean dtype.
x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – If the condition is true at a position, the value from the same position in this tensor will be selected.
y (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – If the condition is false at a position, the value from the same position in this tensor will be selected.
Returns:

A new symbolic tensor holding either values from either x or y, based on the elements in condition.

Return type:

TensorValue

while_loop()
max.graph.ops.while_loop(initial_values, predicate, body)

Execute a loop until the predicate evaluates to false.

Both the predicate and body functions must take in as arguments the same number and types of values as specified in the init_args. The predication function must return only a boolean scalar tensor of type DType.bool. The body function must return a list of values matching the types of init_args, (or may return a value directly if there is only one).

The following example demonstrates a basic while loop with a single argument:

from max.graph import Graph, ops
from max.dtype import DType

with Graph("while_loop_example") as g:
    x = ops.constant(0, dtype=DType.int32, device=DeviceRef.CPU())

    def pred(x):
        return x < 10

    def body(x):
        return x + 1

    result = ops.while_loop(x, pred, body)
    print(result)

The following example shows a while loop with multiple arguments:

from max.graph import Graph, ops
from max.dtype import DType

with Graph("while_loop_example") as g:
    x = ops.constant(0, dtype=DType.int32, device=DeviceRef.CPU())
    y = ops.constant(5, dtype=DType.int32, device=DeviceRef.CPU())

    def pred(x, y):
        return ops.logical_and(x < 10, y < 15)

    def body(x, y):
        return [x + 1, y + 1]

    results = ops.while_loop((x, y), pred, body)
    print(results)

Parameters:

initial_values (Iterable[Value[Any]] | Value[Any]) – Initial values for loop arguments. Must be non-empty.
predicate (Callable[[...], TensorValue]) – Callable that takes loop arguments and returns a boolean scalar tensor of type DType.bool determining loop continuation.
body (Callable[[...], Value[Any] | Iterable[Value[Any]]]) – Callable that takes loop arguments and returns updated values matching the types of init_args.
Returns:

List of output values from the final loop iteration.

Raises:

ValueError – If init_args is empty.
NotImplementedError – If any init_arg is a BufferValue.
Return type:

list[TensorValue]

NOTE
Buffer operations are currently not supported.