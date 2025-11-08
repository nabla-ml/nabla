"""
MAX Engine Mojo Bindings - Modular Architecture
================================================

This module provides a clean, modular interface to MAX Engine from Mojo.

Architecture:
1. PythonBridge: Centralized Python module management
2. Device: Abstract device handling  
3. TensorType: Type definitions for symbolic tensors
4. TensorValue: Runtime tensor values with graph context
5. Graph: Graph construction API
6. MaxModel: Model compilation and execution
7. ops: Operation registry for extensible operations
"""

from python import Python, PythonObject
from memory import ArcPointer
from .utils import PythonBridge, OpRegistry, Graph
from .types import DType, DTypeConverter, TensorType, TensorValue, DeviceType, Device


# ============================================================================
# BINARY OPERATIONS
# ============================================================================

fn add(a: TensorValue, b: TensorValue) raises -> TensorValue:
    """Element-wise addition of two tensors."""
    return OpRegistry.binary_op("add", a, b)


fn matmul(a: TensorValue, b: TensorValue) raises -> TensorValue:
    """Matrix multiplication of two tensors."""
    return OpRegistry.binary_op("matmul", a, b)


fn mul(a: TensorValue, b: TensorValue) raises -> TensorValue:
    """Element-wise multiplication of two tensors."""
    return OpRegistry.binary_op("mul", a, b)


fn sub(a: TensorValue, b: TensorValue) raises -> TensorValue:
    """Element-wise subtraction of two tensors."""
    return OpRegistry.binary_op("sub", a, b)


fn div(a: TensorValue, b: TensorValue) raises -> TensorValue:
    """Element-wise division of two tensors."""
    return OpRegistry.binary_op("div", a, b)

fn mod(a: TensorValue, b: TensorValue) raises -> TensorValue:
    """Element-wise modulo operation."""
    return OpRegistry.binary_op("mod", a, b)


fn pow(a: TensorValue, b: TensorValue) raises -> TensorValue:
    """Element-wise power operation."""
    return OpRegistry.binary_op("pow", a, b)


# ============================================================================
# UNARY OPERATIONS
# ============================================================================

fn abs(x: TensorValue) raises -> TensorValue:
    """Absolute value of tensor."""
    return OpRegistry.unary_op("abs", x)


fn negate(x: TensorValue) raises -> TensorValue:
    """Negate tensor (multiply by -1)."""
    return OpRegistry.unary_op("negate", x)


fn relu(x: TensorValue) raises -> TensorValue:
    """ReLU activation: max(0, x)."""
    return OpRegistry.unary_op("relu", x)


fn sigmoid(x: TensorValue) raises -> TensorValue:
    """Sigmoid activation: 1 / (1 + exp(-x))."""
    return OpRegistry.unary_op("sigmoid", x)


fn tanh(x: TensorValue) raises -> TensorValue:
    """Hyperbolic tangent activation."""
    return OpRegistry.unary_op("tanh", x)


fn exp(x: TensorValue) raises -> TensorValue:
    """Exponential function: e^x."""
    return OpRegistry.unary_op("exp", x)


fn log(x: TensorValue) raises -> TensorValue:
    """Natural logarithm."""
    return OpRegistry.unary_op("log", x)


fn sqrt(x: TensorValue) raises -> TensorValue:
    """Square root."""
    return OpRegistry.unary_op("sqrt", x)



fn log1p(x: TensorValue) raises -> TensorValue:
    """Compute log(1 + x) element-wise."""
    return OpRegistry.unary_op("log1p", x)


fn logsoftmax(x: TensorValue, axis: Int = -1) raises -> TensorValue:
    """Log-softmax activation along specified axis.
    
    Args:
        x: Input tensor.
        axis: Axis along which to compute log-softmax. Default is -1 (last axis).
    
    Returns:
        Log-softmax activated tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.logsoftmax(x.get_graph()[].graph, x.to_python(), axis)
    return TensorValue(x.get_graph(), result_tensor)


fn atanh(x: TensorValue) raises -> TensorValue:
    """Inverse hyperbolic tangent."""
    return OpRegistry.unary_op("atanh", x)


fn cos(x: TensorValue) raises -> TensorValue:
    """Cosine function."""
    return OpRegistry.unary_op("cos", x)


fn erf(x: TensorValue) raises -> TensorValue:
    """Error function."""
    return OpRegistry.unary_op("erf", x)


fn floor(x: TensorValue) raises -> TensorValue:
    """Floor function (round down to nearest integer)."""
    return OpRegistry.unary_op("floor", x)


fn is_inf(x: TensorValue) raises -> TensorValue:
    """Check if elements are infinity."""
    return OpRegistry.unary_op("is_inf", x)


fn is_nan(x: TensorValue) raises -> TensorValue:
    """Check if elements are NaN."""
    return OpRegistry.unary_op("is_nan", x)



fn sin(x: TensorValue) raises -> TensorValue:
    """Sine function."""
    return OpRegistry.unary_op("sin", x)


fn rsqrt(x: TensorValue) raises -> TensorValue:
    """Reciprocal square root: 1/sqrt(x)."""
    return OpRegistry.unary_op("rsqrt", x)


fn round(x: TensorValue) raises -> TensorValue:
    """Round to nearest integer."""
    return OpRegistry.unary_op("round", x)


fn trunc(x: TensorValue) raises -> TensorValue:
    """Truncate to integer (round toward zero)."""
    return OpRegistry.unary_op("trunc", x)



# ============================================================================
# REDUCTION OPERATIONS (with axis parameter)
# ============================================================================

fn mean(x: TensorValue, axis: Int = -1) raises -> TensorValue:
    """Mean reduction along specified axis.
    
    Args:
        x: Input tensor.
        axis: Axis along which to compute the mean. Default is -1 (last axis).
    
    Returns:
        Reduced tensor with mean values.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.mean(x.get_graph()[].graph, x.to_python(), axis)
    return TensorValue(x.get_graph(), result_tensor)


fn sum(x: TensorValue, axis: Int = -1) raises -> TensorValue:
    """Sum reduction along specified axis.
    
    Args:
        x: Input tensor.
        axis: Axis along which to compute the sum. Default is -1 (last axis).
    
    Returns:
        Reduced tensor with sum values.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.sum(x.get_graph()[].graph, x.to_python(), axis)
    return TensorValue(x.get_graph(), result_tensor)


fn max(x: TensorValue, axis: Int = -1) raises -> TensorValue:
    """Max reduction along specified axis.
    
    Args:
        x: Input tensor.
        axis: Axis along which to compute the max. Default is -1 (last axis).
    
    Returns:
        Reduced tensor with max values.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.max(x.get_graph()[].graph, x.to_python(), axis)
    return TensorValue(x.get_graph(), result_tensor)


fn min(x: TensorValue, axis: Int = -1) raises -> TensorValue:
    """Min reduction along specified axis.
    
    Args:
        x: Input tensor.
        axis: Axis along which to compute the min. Default is -1 (last axis).
    
    Returns:
        Reduced tensor with min values.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.min(x.get_graph()[].graph, x.to_python(), axis)
    return TensorValue(x.get_graph(), result_tensor)


fn argmax(x: TensorValue, axis: Int = -1) raises -> TensorValue:
    """Argmax reduction along specified axis.
    
    Args:
        x: Input tensor.
        axis: Axis along which to compute argmax. Default is -1 (last axis).
    
    Returns:
        Tensor with indices of maximum values.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.argmax(x.get_graph()[].graph, x.to_python(), axis)
    return TensorValue(x.get_graph(), result_tensor)


fn argmin(x: TensorValue, axis: Int = -1) raises -> TensorValue:
    """Argmin reduction along specified axis.
    
    Args:
        x: Input tensor.
        axis: Axis along which to compute argmin. Default is -1 (last axis).
    
    Returns:
        Tensor with indices of minimum values.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.argmin(x.get_graph()[].graph, x.to_python(), axis)
    return TensorValue(x.get_graph(), result_tensor)


# ============================================================================
# ACTIVATION FUNCTIONS (Batch 2) - with parameters
# ============================================================================

fn gelu(x: TensorValue, approximate: String = "none") raises -> TensorValue:
    """GELU activation function.
    
    Args:
        x: Input tensor.
        approximate: Approximation mode - "none", "tanh", or "quick".
    
    Returns:
        GELU activated tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.gelu(x.get_graph()[].graph, x.to_python(), approximate)
    return TensorValue(x.get_graph(), result_tensor)


fn softmax(x: TensorValue, axis: Int = -1) raises -> TensorValue:
    """Softmax activation along specified axis.
    
    Args:
        x: Input tensor.
        axis: Axis along which to compute softmax. Default is -1 (last axis).
    
    Returns:
        Softmax activated tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.softmax(x.get_graph()[].graph, x.to_python(), axis)
    return TensorValue(x.get_graph(), result_tensor)


fn silu(x: TensorValue) raises -> TensorValue:
    """SiLU (Swish) activation: x * sigmoid(x)."""
    return OpRegistry.unary_op("silu", x)


# ============================================================================
# COMPARISON OPERATIONS
# ============================================================================

fn equal(a: TensorValue, b: TensorValue) raises -> TensorValue:
    """Element-wise equality comparison."""
    return OpRegistry.binary_op("equal", a, b)


fn greater(a: TensorValue, b: TensorValue) raises -> TensorValue:
    """Element-wise greater than comparison."""
    return OpRegistry.binary_op("greater", a, b)


fn greater_equal(a: TensorValue, b: TensorValue) raises -> TensorValue:
    """Element-wise greater than or equal comparison."""
    return OpRegistry.binary_op("greater_equal", a, b)


fn not_equal(a: TensorValue, b: TensorValue) raises -> TensorValue:
    """Element-wise not equal comparison."""
    return OpRegistry.binary_op("not_equal", a, b)


# ============================================================================
# LOGICAL OPERATIONS
# ============================================================================

fn logical_and(a: TensorValue, b: TensorValue) raises -> TensorValue:
    """Element-wise logical AND."""
    return OpRegistry.binary_op("logical_and", a, b)


fn logical_or(a: TensorValue, b: TensorValue) raises -> TensorValue:
    """Element-wise logical OR."""
    return OpRegistry.binary_op("logical_or", a, b)


fn logical_xor(a: TensorValue, b: TensorValue) raises -> TensorValue:
    """Element-wise logical XOR."""
    return OpRegistry.binary_op("logical_xor", a, b)


fn logical_not(x: TensorValue) raises -> TensorValue:
    """Element-wise logical NOT."""
    return OpRegistry.unary_op("logical_not", x)



fn less(lhs: TensorValue, rhs: TensorValue) raises -> TensorValue:
    """Element-wise less than comparison.
    
    Args:
        lhs: Left operand.
        rhs: Right operand.
    
    Returns:
        Boolean tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.less(lhs.get_graph()[].graph, lhs.to_python(), rhs.to_python())
    return TensorValue(lhs.get_graph(), result_tensor)


fn less_equal(lhs: TensorValue, rhs: TensorValue) raises -> TensorValue:
    """Element-wise less than or equal comparison.
    
    Args:
        lhs: Left operand.
        rhs: Right operand.
    
    Returns:
        Boolean tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.less_equal(lhs.get_graph()[].graph, lhs.to_python(), rhs.to_python())
    return TensorValue(lhs.get_graph(), result_tensor)


# ============================================================================
# BATCH 5: SHAPE OPERATIONS
# ============================================================================

fn reshape(x: TensorValue, shape: List[Int]) raises -> TensorValue:
    """Reshape tensor to new shape.
    
    Args:
        x: Input tensor.
        shape: Target shape as List[Int].
    
    Returns:
        Reshaped tensor.
    """
    var ops = PythonBridge.get_module("main")
    var py_shape = PythonBridge.shape_to_python(shape)
    var result_tensor = ops.reshape(x.get_graph()[].graph, x.to_python(), py_shape)
    return TensorValue(x.get_graph(), result_tensor)


fn flatten(x: TensorValue, start_dim: Int = 0, end_dim: Int = -1) raises -> TensorValue:
    """Flatten tensor dimensions from start_dim to end_dim.
    
    Args:
        x: Input tensor.
        start_dim: First dimension to flatten. Default is 0.
        end_dim: Last dimension to flatten. Default is -1.
    
    Returns:
        Flattened tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.flatten(x.get_graph()[].graph, x.to_python(), start_dim, end_dim)
    return TensorValue(x.get_graph(), result_tensor)


fn squeeze(x: TensorValue, axis: Int) raises -> TensorValue:
    """Remove dimension of size 1 at specified axis.
    
    Args:
        x: Input tensor.
        axis: Axis to squeeze.
    
    Returns:
        Squeezed tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.squeeze(x.get_graph()[].graph, x.to_python(), axis)
    return TensorValue(x.get_graph(), result_tensor)


fn unsqueeze(x: TensorValue, axis: Int) raises -> TensorValue:
    """Add dimension of size 1 at specified axis.
    
    Args:
        x: Input tensor.
        axis: Axis to add dimension.
    
    Returns:
        Unsqueezed tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.unsqueeze(x.get_graph()[].graph, x.to_python(), axis)
    return TensorValue(x.get_graph(), result_tensor)


fn permute(x: TensorValue, dims: List[Int]) raises -> TensorValue:
    """Permute dimensions of tensor.
    
    Args:
        x: Input tensor.
        dims: New ordering of dimensions as List[Int].
    
    Returns:
        Permuted tensor.
    """
    var ops = PythonBridge.get_module("main")
    var py_dims = PythonBridge.shape_to_python(dims)
    var result_tensor = ops.permute(x.get_graph()[].graph, x.to_python(), py_dims)
    return TensorValue(x.get_graph(), result_tensor)


fn broadcast_to(x: TensorValue, shape: List[Int]) raises -> TensorValue:
    """Broadcast tensor to new shape.
    
    Args:
        x: Input tensor.
        shape: Target shape as List[Int].
    
    Returns:
        Broadcasted tensor.
    """
    var ops = PythonBridge.get_module("main")
    var py_shape = PythonBridge.shape_to_python(shape)
    var result_tensor = ops.broadcast_to(x.get_graph()[].graph, x.to_python(), py_shape)
    return TensorValue(x.get_graph(), result_tensor)


fn cast(x: TensorValue, dtype: DType) raises -> TensorValue:
    """Cast tensor to different data type.
    
    Args:
        x: Input tensor.
        dtype: Target DType.
    
    Returns:
        Casted tensor.
    """
    var ops = PythonBridge.get_module("main")
    
    # Convert DType to string for Python backend
    var dtype_str: String
    if dtype == DType.float32:
        dtype_str = "float32"
    elif dtype == DType.float64:
        dtype_str = "float64"
    elif dtype == DType.int32:
        dtype_str = "int32"
    elif dtype == DType.int64:
        dtype_str = "int64"
    elif dtype == DType.int8:
        dtype_str = "int8"
    elif dtype == DType.uint8:
        dtype_str = "uint8"
    elif dtype == DType.bool:
        dtype_str = "bool"
    else:
        raise Error("Unsupported DType")
    
    var result_tensor = ops.cast(x.get_graph()[].graph, x.to_python(), dtype_str)
    return TensorValue(x.get_graph(), result_tensor)


fn transpose(x: TensorValue, axis_1: Int, axis_2: Int) raises -> TensorValue:
    """Transpose two axes of tensor.
    
    Args:
        x: Input tensor.
        axis_1: First axis.
        axis_2: Second axis.
    
    Returns:
        Transposed tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.transpose(x.get_graph()[].graph, x.to_python(), axis_1, axis_2)
    return TensorValue(x.get_graph(), result_tensor)



fn resize(input: TensorValue, shape: List[Int], interpolation: String = "BILINEAR") raises -> TensorValue:
    """Resize tensor to target shape.
    
    Args:
        input: Input tensor.
        shape: Target shape as List[Int].
        interpolation: Interpolation mode ('BILINEAR', 'NEAREST', etc.). Default is 'BILINEAR'.
    
    Returns:
        Resized tensor.
    """
    var ops = PythonBridge.get_module("main")
    var py_shape = PythonBridge.shape_to_python(shape)
    var result_tensor = ops.resize(input.get_graph()[].graph, input.to_python(), py_shape, interpolation)
    return TensorValue(input.get_graph(), result_tensor)


# ============================================================================
# BATCH 6: TENSOR MANIPULATION
# ============================================================================

fn concat(vals: List[TensorValue], axis: Int = 0) raises -> TensorValue:
    """Concatenate tensors along specified axis.
    
    Args:
        vals: List of tensors to concatenate.
        axis: Axis along which to concatenate. Default is 0.
    
    Returns:
        Concatenated tensor.
    """
    if len(vals) == 0:
        raise Error("concat requires at least one tensor")
    
    var ops = PythonBridge.get_module("main")
    var builtins = Python.import_module("builtins")
    
    # Convert List[TensorValue] to Python list of tensors
    var py_list = builtins.list()
    for i in range(len(vals)):
        _ = py_list.append(vals[i].to_python())
    
    var result_tensor = ops.concat(vals[0].get_graph()[].graph, py_list, axis)
    return TensorValue(vals[0].get_graph(), result_tensor)


fn split(x: TensorValue, split_sizes: List[Int], axis: Int = 0) raises -> List[TensorValue]:
    """Split tensor into multiple tensors.
    
    Args:
        x: Input tensor.
        split_sizes: Sizes of each split as List[Int].
        axis: Axis along which to split. Default is 0.
    
    Returns:
        List of split tensors.
    """
    var ops = PythonBridge.get_module("main")
    var py_split_sizes = PythonBridge.shape_to_python(split_sizes)
    var result_list = ops.split(x.get_graph()[].graph, x.to_python(), py_split_sizes, axis)
    
    # Convert Python list of tensors to List[TensorValue]
    var results = List[TensorValue]()
    for i in range(len(result_list)):
        results.append(TensorValue(x.get_graph(), result_list[i]))
    
    return results^


fn chunk(x: TensorValue, chunks: Int, axis: Int = 0) raises -> List[TensorValue]:
    """Split tensor into equal chunks.
    
    Args:
        x: Input tensor.
        chunks: Number of chunks.
        axis: Axis along which to chunk. Default is 0.
    
    Returns:
        List of chunked tensors.
    """
    var ops = PythonBridge.get_module("main")
    var result_list = ops.chunk(x.get_graph()[].graph, x.to_python(), chunks, axis)
    
    # Convert Python list of tensors to List[TensorValue]
    var results = List[TensorValue]()
    for i in range(len(result_list)):
        results.append(TensorValue(x.get_graph(), result_list[i]))
    
    return results^


fn gather(input: TensorValue, indices: TensorValue, axis: Int) raises -> TensorValue:
    """Gather values along an axis specified by indices.
    
    Args:
        input: Input tensor.
        indices: Indices tensor.
        axis: Axis along which to gather.
    
    Returns:
        Gathered tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.gather(input.get_graph()[].graph, input.to_python(), indices.to_python(), axis)
    return TensorValue(input.get_graph(), result_tensor)


fn pad(input: TensorValue, paddings: List[Int], mode: String = "constant", value: Float32 = 0.0) raises -> TensorValue:
    """Pad tensor with specified padding.
    
    Args:
        input: Input tensor.
        paddings: Padding sizes as List[Int].
        mode: Padding mode ('constant', 'reflect', 'replicate', 'circular'). Default is 'constant'.
        value: Fill value for constant padding. Default is 0.0.
    
    Returns:
        Padded tensor.
    """
    var ops = PythonBridge.get_module("main")
    var py_paddings = PythonBridge.shape_to_python(paddings)
    var result_tensor = ops.pad(input.get_graph()[].graph, input.to_python(), py_paddings, mode, value)
    return TensorValue(input.get_graph(), result_tensor)


# ============================================================================
# Batch 7: Common Missing Operations (12 ops)
# ============================================================================

fn stack(tensors: List[TensorValue], axis: Int = 0) raises -> TensorValue:
    """Stack tensors along a new axis.
    
    Args:
        tensors: List of tensors to stack.
        axis: Axis along which to stack. Default is 0.
    
    Returns:
        Stacked tensor.
    """
    if len(tensors) == 0:
        raise Error("Cannot stack empty list of tensors")
    
    var ops = PythonBridge.get_module("main")
    var builtins = PythonBridge.get_builtins()
    
    # Convert List[TensorValue] to Python list
    var py_list = builtins.list()
    for i in range(len(tensors)):
        _ = py_list.append(tensors[i].to_python())
    
    var result_tensor = ops.stack(tensors[0].get_graph()[].graph, py_list, axis)
    return TensorValue(tensors[0].get_graph(), result_tensor)


fn tile(input: TensorValue, repeats: List[Int]) raises -> TensorValue:
    """Tile/repeat tensor along each dimension.
    
    Args:
        input: Input tensor.
        repeats: Number of repetitions for each dimension.
    
    Returns:
        Tiled tensor.
    """
    var ops = PythonBridge.get_module("main")
    var py_repeats = PythonBridge.shape_to_python(repeats)
    var result_tensor = ops.tile(input.get_graph()[].graph, input.to_python(), py_repeats)
    return TensorValue(input.get_graph(), result_tensor)


fn repeat_interleave(input: TensorValue, repeats: Int, axis: Int = 0) raises -> TensorValue:
    """Repeat elements of a tensor.
    
    Args:
        input: Input tensor.
        repeats: Number of repetitions for each element.
        axis: Axis along which to repeat. Default is 0.
    
    Returns:
        Tensor with repeated elements.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.repeat_interleave(input.get_graph()[].graph, input.to_python(), repeats, axis)
    return TensorValue(input.get_graph(), result_tensor)


fn slice_tensor(input: TensorValue, indices: PythonObject) raises -> TensorValue:
    """Slice tensor using index specifications.
    
    Args:
        input: Input tensor.
        indices: Python list of slice specifications (can include slice(), integers, Ellipsis, None).
    
    Returns:
        Sliced tensor.
    
    Example:
        Reverse a tensor along first dimension:
        ```
        var builtins = PythonBridge.get_builtins()
        var idx = builtins.list([builtins.slice(None, None, -1)])
        var result = slice_tensor(x, idx)
        ```
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.slice_tensor(input.get_graph()[].graph, input.to_python(), indices)
    return TensorValue(input.get_graph(), result_tensor)


fn where(condition: TensorValue, x: TensorValue, y: TensorValue) raises -> TensorValue:
    """Select elements from x or y depending on condition.
    
    Args:
        condition: Boolean condition tensor.
        x: Values to select when condition is True.
        y: Values to select when condition is False.
    
    Returns:
        Tensor with selected values.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.where(condition.get_graph()[].graph, condition.to_python(), x.to_python(), y.to_python())
    return TensorValue(condition.get_graph(), result_tensor)


fn outer(lhs: TensorValue, rhs: TensorValue) raises -> TensorValue:
    """Compute outer product of two vectors.
    
    Args:
        lhs: First vector.
        rhs: Second vector.
    
    Returns:
        Outer product matrix.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.outer(lhs.get_graph()[].graph, lhs.to_python(), rhs.to_python())
    return TensorValue(lhs.get_graph(), result_tensor)


fn cumsum(input: TensorValue, axis: Int = -1, exclusive: Bool = False, reverse: Bool = False) raises -> TensorValue:
    """Cumulative sum along an axis.
    
    Args:
        input: Input tensor.
        axis: Axis along which to compute cumsum. Default is -1.
        exclusive: If True, exclude current element. Default is False.
        reverse: If True, compute cumsum in reverse. Default is False.
    
    Returns:
        Cumulative sum tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.cumsum(input.get_graph()[].graph, input.to_python(), axis, exclusive, reverse)
    return TensorValue(input.get_graph(), result_tensor)


fn argsort(input: TensorValue, ascending: Bool = True) raises -> TensorValue:
    """Return indices that would sort the tensor.
    
    Args:
        input: Input tensor.
        ascending: Sort in ascending order if True. Default is True.
    
    Returns:
        Indices tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.argsort(input.get_graph()[].graph, input.to_python(), ascending)
    return TensorValue(input.get_graph(), result_tensor)


fn top_k(input: TensorValue, k: Int, axis: Int = -1) raises -> PythonObject:
    """Find top k values and their indices.
    
    Args:
        input: Input tensor.
        k: Number of top elements.
        axis: Axis along which to find top k. Default is -1.
    
    Returns:
        Python tuple of (values_tensor, indices_tensor).
        Access as: result[0] for values, result[1] for indices.
    """
    var ops = PythonBridge.get_module("main")
    return ops.top_k(input.get_graph()[].graph, input.to_python(), k, axis)


fn scatter(input: TensorValue, updates: TensorValue, indices: TensorValue, axis: Int = -1) raises -> TensorValue:
    """Scatter updates into input tensor at specified indices.
    
    Args:
        input: Input tensor.
        updates: Update values.
        indices: Indices where to scatter updates.
        axis: Axis along which to scatter. Default is -1.
    
    Returns:
        Scattered tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.scatter(input.get_graph()[].graph, input.to_python(), updates.to_python(), indices.to_python(), axis)
    return TensorValue(input.get_graph(), result_tensor)


fn scatter_nd(input: TensorValue, updates: TensorValue, indices: TensorValue) raises -> TensorValue:
    """Scatter updates into input tensor using N-dimensional indices.
    
    Args:
        input: Input tensor.
        updates: Update values.
        indices: N-dimensional indices.
    
    Returns:
        Scattered tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.scatter_nd(input.get_graph()[].graph, input.to_python(), updates.to_python(), indices.to_python())
    return TensorValue(input.get_graph(), result_tensor)


fn gather_nd(input: TensorValue, indices: TensorValue, batch_dims: Int = 0) raises -> TensorValue:
    """Gather values from input using N-dimensional indices.
    
    Args:
        input: Input tensor.
        indices: N-dimensional indices.
        batch_dims: Number of batch dimensions. Default is 0.
    
    Returns:
        Gathered tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.gather_nd(input.get_graph()[].graph, input.to_python(), indices.to_python(), batch_dims)
    return TensorValue(input.get_graph(), result_tensor)


# ============================================================================
# LINEAR ALGEBRA OPERATIONS
# ============================================================================

fn conv2d(x: TensorValue, filter: TensorValue, stride: List[Int], dilation: List[Int], 
          padding: List[Int], groups: Int = 1) raises -> TensorValue:
    """2D convolution operation.
    
    Args:
        x: Input tensor.
        filter: Filter/kernel tensor.
        stride: Stride for height and width as List[Int].
        dilation: Dilation for height and width as List[Int].
        padding: Padding (top, bottom, left, right) as List[Int].
        groups: Number of groups for grouped convolution. Default is 1.
    
    Returns:
        Convolution result.
    
    Note: For bias, use the version with bias parameter separately.
    """
    var ops = PythonBridge.get_module("main")
    var py_stride = PythonBridge.shape_to_python(stride)
    var py_dilation = PythonBridge.shape_to_python(dilation)
    var py_padding = PythonBridge.shape_to_python(padding)
    var result_tensor = ops.conv2d(x.get_graph()[].graph, x.to_python(), filter.to_python(),
                                  py_stride, py_dilation, py_padding, groups)
    return TensorValue(x.get_graph(), result_tensor)


fn conv2d_transpose(x: TensorValue, filter: TensorValue, stride: List[Int], dilation: List[Int],
                    padding: List[Int], output_paddings: List[Int]) raises -> TensorValue:
    """2D transposed convolution operation.
    
    Args:
        x: Input tensor.
        filter: Filter/kernel tensor.
        stride: Stride for height and width as List[Int].
        dilation: Dilation for height and width as List[Int].
        padding: Padding (top, bottom, left, right) as List[Int].
        output_paddings: Output padding for height and width as List[Int].
    
    Returns:
        Transposed convolution result.
    """
    var ops = PythonBridge.get_module("main")
    var py_stride = PythonBridge.shape_to_python(stride)
    var py_dilation = PythonBridge.shape_to_python(dilation)
    var py_padding = PythonBridge.shape_to_python(padding)
    var py_output_paddings = PythonBridge.shape_to_python(output_paddings)
    var result_tensor = ops.conv2d_transpose(x.get_graph()[].graph, x.to_python(), filter.to_python(),
                                            py_stride, py_dilation, py_padding, py_output_paddings)
    return TensorValue(x.get_graph(), result_tensor)


fn max_pool2d(input: TensorValue, kernel_size: List[Int], stride: List[Int], dilation: List[Int],
              padding: List[Int], ceil_mode: Bool = False) raises -> TensorValue:
    """2D max pooling operation.
    
    Args:
        input: Input tensor.
        kernel_size: Size of pooling kernel (height, width) as List[Int].
        stride: Stride for height and width as List[Int].
        dilation: Dilation for height and width as List[Int].
        padding: Padding (top, bottom, left, right) as List[Int].
        ceil_mode: Use ceil instead of floor for output shape. Default is False.
    
    Returns:
        Max pooled tensor.
    """
    var ops = PythonBridge.get_module("main")
    var py_kernel_size = PythonBridge.shape_to_python(kernel_size)
    var py_stride = PythonBridge.shape_to_python(stride)
    var py_dilation = PythonBridge.shape_to_python(dilation)
    var py_padding = PythonBridge.shape_to_python(padding)
    var result_tensor = ops.max_pool2d(input.get_graph()[].graph, input.to_python(), 
                                      py_kernel_size, py_stride, py_dilation, py_padding, ceil_mode)
    return TensorValue(input.get_graph(), result_tensor)


fn avg_pool2d(input: TensorValue, kernel_size: List[Int], stride: List[Int], dilation: List[Int],
              padding: List[Int], ceil_mode: Bool = False, count_boundary: Bool = True) raises -> TensorValue:
    """2D average pooling operation.
    
    Args:
        input: Input tensor.
        kernel_size: Size of pooling kernel (height, width) as List[Int].
        stride: Stride for height and width as List[Int].
        dilation: Dilation for height and width as List[Int].
        padding: Padding (top, bottom, left, right) as List[Int].
        ceil_mode: Use ceil instead of floor for output shape. Default is False.
        count_boundary: Include padding in average calculation. Default is True.
    
    Returns:
        Average pooled tensor.
    """
    var ops = PythonBridge.get_module("main")
    var py_kernel_size = PythonBridge.shape_to_python(kernel_size)
    var py_stride = PythonBridge.shape_to_python(stride)
    var py_dilation = PythonBridge.shape_to_python(dilation)
    var py_padding = PythonBridge.shape_to_python(padding)
    var result_tensor = ops.avg_pool2d(input.get_graph()[].graph, input.to_python(),
                                      py_kernel_size, py_stride, py_dilation, py_padding, ceil_mode, count_boundary)
    return TensorValue(input.get_graph(), result_tensor)


fn layer_norm(input: TensorValue, gamma: TensorValue, beta: TensorValue, epsilon: Float32) raises -> TensorValue:
    """Layer normalization.
    
    Args:
        input: Input tensor.
        gamma: Scale parameter.
        beta: Shift parameter.
        epsilon: Small value for numerical stability.
    
    Returns:
        Normalized tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.layer_norm(input.get_graph()[].graph, input.to_python(), 
                                      gamma.to_python(), beta.to_python(), epsilon)
    return TensorValue(input.get_graph(), result_tensor)


fn clip_by_value(x: TensorValue, min_val: TensorValue, max_val: TensorValue) raises -> TensorValue:
    """Clip tensor values element-wise.
    
    Args:
        x: Input tensor.
        min_val: Minimum value tensor.
        max_val: Maximum value tensor.
    
    Returns:
        Clipped tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.clip_by_value(x.get_graph()[].graph, x.to_python(), 
                                         min_val.to_python(), max_val.to_python())
    return TensorValue(x.get_graph(), result_tensor)


fn band_part(x: TensorValue, num_lower: Int, num_upper: Int) raises -> TensorValue:
    """Extract band from tensor.
    
    Args:
        x: Input tensor.
        num_lower: Number of lower diagonals to keep.
        num_upper: Number of upper diagonals to keep.
    
    Returns:
        Band-extracted tensor.
    """
    var max_graph = PythonBridge.get_module("main")
    var result_tensor = max_graph.band_part(
        x.get_graph()[].graph,
        x.to_python(),
        num_lower,
        num_upper
    )
    return TensorValue(x.get_graph(), result_tensor)


fn conv3d(input: TensorValue, weight: TensorValue, 
          stride: List[Int], padding: List[Int], dilation: List[Int]) raises -> TensorValue:
    """3D convolution operation.
    
    Args:
        input: Input tensor (NDHWC format).
        weight: Filter/kernel tensor.
        stride: Stride for each dimension (3 values).
        padding: Padding for each dimension (6 values: before/after for each dim).
        dilation: Dilation for each dimension (3 values).
    
    Returns:
        Convolved tensor.
    """
    OpRegistry.check_same_context([input, weight])
    var max_graph = PythonBridge.get_module("main")
    var stride_tuple = PythonBridge.shape_to_python(stride)
    var padding_tuple = PythonBridge.shape_to_python(padding)
    var dilation_tuple = PythonBridge.shape_to_python(dilation)
    
    var result_tensor = max_graph.conv3d(
        input.get_graph()[].graph,
        input.to_python(),
        weight.to_python(),
        stride_tuple,
        padding_tuple,
        dilation_tuple
    )
    return TensorValue(input.get_graph(), result_tensor)


fn hann_window(size: Int, periodic: Bool, device: String = "cpu:0") raises -> TensorValue:
    """Generate Hann window.
    
    Args:
        size: Window size (integer constant).
        periodic: Whether window is periodic.
        device: Device string (default: "cpu:0").
    
    Returns:
        Hann window tensor (requires a graph context).
    """
    # This operation needs a graph context but takes an integer size
    # It's special and needs to be called within a graph builder context
    raise Error("hann_window requires graph context - use directly in Python backend")


fn masked_scatter(x: TensorValue, mask: TensorValue, source: TensorValue, out_dim: Int = 0) raises -> TensorValue:
    """Scatter source values into x where mask is true.
    
    Args:
        x: Input tensor.
        mask: Boolean mask tensor.
        source: Source values to scatter.
        out_dim: Output dimension (default: 0).
    
    Returns:
        Scattered tensor.
    """
    OpRegistry.check_same_context([x, mask, source])
    var max_graph = PythonBridge.get_module("main")
    var result_tensor = max_graph.masked_scatter(
        x.get_graph()[].graph,
        x.to_python(),
        mask.to_python(),
        source.to_python(),
        out_dim
    )
    return TensorValue(x.get_graph(), result_tensor)


fn as_interleaved_complex(x: TensorValue) raises -> TensorValue:
    """Reshape input tensor as complex from alternating (real, imag).
    
    Args:
        x: Input tensor with alternating real/imag values.
    
    Returns:
        Complex-valued tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.as_interleaved_complex(x.get_graph()[].graph, x.to_python())
    return TensorValue(x.get_graph(), result_tensor)


fn assert_same_device(values: List[TensorValue]) raises:
    """Assert that all tensor values are on the same device.
    
    Args:
        values: List of tensor values to check.
    
    Raises:
        Error if tensors are not on the same device.
    
    Note: This checks all tensors in the list for device compatibility.
    """
    if len(values) == 0:
        return
    
    # Note: MAX ops.assert_same_device requires actual TensorValue objects
    # This is a simplified version that just checks context
    OpRegistry.check_same_context(values)


fn cond(pred: TensorValue, out_types: PythonObject, then_fn: PythonObject, else_fn: PythonObject) raises -> List[TensorValue]:
    """Conditionally execute one of two branches based on a boolean predicate.
    
    Args:
        pred: Boolean scalar tensor determining branch execution.
        out_types: Expected output types for both branches (Python list or None).
        then_fn: Callable executed when pred is True (Python callable).
        else_fn: Callable executed when pred is False (Python callable).
    
    Returns:
        List of output values from executed branch.
    
    Note: then_fn and else_fn must be Python callables.
    """
    var ops = PythonBridge.get_module("main")
    var result_list = ops.cond(pred.get_graph()[].graph, pred.to_python(), out_types, then_fn, else_fn)
    
    var results = List[TensorValue]()
    for i in range(len(result_list)):
        results.append(TensorValue(pred.get_graph(), result_list[i]))
    
    return results^


fn constant(ctx: ArcPointer[Graph], value: PythonObject, dtype: DType, device: Device) raises -> TensorValue:
    """Add a constant operation node.
    
    Args:
        ctx: Graph context.
        value: The constant's value (as Python object).
        dtype: The constant tensor's element type.
        device: The device the constant lives on.
    
    Returns:
        Graph value containing the constant data.
    """
    var ops = PythonBridge.get_module("main")
    var python_dtype = DTypeConverter.to_python(dtype)
    # Note: DeviceType.Accelerator() or DeviceType.GPU() both map to "gpu:0"
    var device_str = String("cpu:0") if device.device_type == DeviceType.CPU() else String("gpu:0")
    var result_tensor = ops.constant(ctx[].graph, value, python_dtype, device_str)
    return TensorValue(ctx, result_tensor)


fn constant_external(ctx: ArcPointer[Graph], name: String, tensor_type: TensorType) raises -> TensorValue:
    """Register an external constant (weight) in the graph.
    
    Args:
        ctx: Graph context.
        name: The name of the external constant.
        tensor_type: The type of the constant value.
    
    Returns:
        Tensor value representing the weight.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.constant_external(ctx[].graph, name, tensor_type.to_python())
    return TensorValue(ctx, result_tensor)


fn custom(name: String, device: Device, values: List[TensorValue], 
          out_types: List[TensorType], parameters: Optional[PythonObject] = None) raises -> List[TensorValue]:
    """Create a node to execute a custom graph operation.
    
    Args:
        name: The op name provided to @compiler.register.
        device: Device that the op is assigned to.
        values: The op function's arguments.
        out_types: The list of op function's return types.
        parameters: Dictionary of extra parameters expected by the kernel.
    
    Returns:
        List of symbolic values representing the outputs of the op.
    """
    if len(values) == 0:
        raise Error("custom operation requires at least one input value")
    
    var ctx = values[0].get_graph()
    OpRegistry.check_same_context(values)
    
    var ops = PythonBridge.get_module("main")
    var builtins = Python.import_module("builtins")
    
    # Convert values to Python list
    var py_values = builtins.list()
    for i in range(len(values)):
        _ = py_values.append(values[i].tensor_value)
    
    # Convert out_types to Python list
    var py_out_types = builtins.list()
    for i in range(len(out_types)):
        _ = py_out_types.append(out_types[i].to_python())
    
    # Call the Python backend
    var result: PythonObject
    if parameters:
        result = ops.custom(ctx[].graph, name, device.to_python(), py_values, py_out_types, parameters.value())
    else:
        result = ops.custom(ctx[].graph, name, device.to_python(), py_values, py_out_types)
    
    # Convert result to List[TensorValue]
    var output = List[TensorValue]()
    for i in range(len(result)):
        output.append(TensorValue(ctx, result[i]))
    return output^


fn allgather(inputs: List[TensorValue], signal_buffers: List[TensorValue], axis: Int = 0) raises -> List[TensorValue]:
    """Collective allgather operation.
    
    Args:
        inputs: The input tensors to gather.
        signal_buffers: Device buffer values used for synchronization.
        axis: Dimension to concatenate the input tensors (default: 0).
    
    Returns:
        List of gathered output tensors.
    """
    if len(inputs) == 0:
        raise Error("allgather requires at least one input")
    
    var ctx = inputs[0].get_graph()
    OpRegistry.check_same_context(inputs)
    
    var ops = PythonBridge.get_module("main")
    var builtins = Python.import_module("builtins")
    
    # Convert inputs to Python list
    var py_inputs = builtins.list()
    for i in range(len(inputs)):
        _ = py_inputs.append(inputs[i].tensor_value)
    
    # Convert signal_buffers to Python list
    var py_buffers = builtins.list()
    for i in range(len(signal_buffers)):
        _ = py_buffers.append(signal_buffers[i].tensor_value)
    
    var result = ops.allgather(ctx[].graph, py_inputs, py_buffers, axis)
    
    # Convert result to List[TensorValue]
    var output = List[TensorValue]()
    for i in range(len(result)):
        output.append(TensorValue(ctx, result[i]))
    return output^


fn fold(input: TensorValue, output_size: List[Int], kernel_size: List[Int], 
        stride: List[Int], dilation: List[Int], padding: List[Int]) raises -> TensorValue:
    """Combine array of sliding blocks into larger containing tensor.
    
    Args:
        input: The 3D tensor to fold.
        output_size: Spatial dimensions of the output tensor (2 values).
        kernel_size: The size of the sliding blocks (2 values).
        stride: The stride of the sliding blocks (2 values).
        dilation: The spacing between the kernel elements (2 values).
        padding: 0-paddings to be added on both sides (2 values).
    
    Returns:
        The folded 4D tensor.
    """
    var ops = PythonBridge.get_module("main")
    var py_output_size = PythonBridge.shape_to_python(output_size)
    var py_kernel_size = PythonBridge.shape_to_python(kernel_size)
    var py_stride = PythonBridge.shape_to_python(stride)
    var py_dilation = PythonBridge.shape_to_python(dilation)
    var py_padding = PythonBridge.shape_to_python(padding)
    
    var result_tensor = ops.fold(input.get_graph()[].graph, input.to_python(), 
                                 py_output_size, py_kernel_size, py_stride, py_dilation, py_padding)
    return TensorValue(input.get_graph(), result_tensor)


fn nonzero(x: TensorValue, out_dim: Int) raises -> TensorValue:
    """Get indices of non-zero elements.
    
    Args:
        x: Input tensor.
        out_dim: The newly generated dimension that is sized for the number of nonzero elements.
    
    Returns:
        Tensor of indices.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.nonzero(x.get_graph()[].graph, x.to_python(), out_dim)
    return TensorValue(x.get_graph(), result_tensor)


fn range_op(ctx: ArcPointer[Graph], start: Int, stop: Int, step: Int, dtype: DType, device: Device) raises -> TensorValue:
    """Create a sequence of numbers.
    
    Args:
        ctx: Graph context.
        start: The start of the range.
        stop: The range will be generated up to, but not including, this value.
        step: The step size for the range.
        dtype: Data type of the result tensor.
        device: Device of the result tensor.
    
    Returns:
        Tensor containing the defined range of values.
    """
    var ops = PythonBridge.get_module("main")
    
    # Convert dtype to string
    var dtype_str: String
    if dtype == DType.float32:
        dtype_str = "float32"
    elif dtype == DType.float64:
        dtype_str = "float64"
    elif dtype == DType.int32:
        dtype_str = "int32"
    elif dtype == DType.int64:
        dtype_str = "int64"
    else:
        dtype_str = "float32"
    
    # Note: DeviceType.Accelerator() or DeviceType.GPU() both map to "gpu:0"
    var device_str = String("cpu:0") if device.device_type == DeviceType.CPU() else String("gpu:0")
    
    var result_tensor = ops.range_op(ctx[].graph, start, stop, step, None, dtype_str, device_str)
    return TensorValue(ctx, result_tensor)


fn rebind(x: TensorValue, shape: List[Int], message: String = "") raises -> TensorValue:
    """Rebind a symbolic tensor to a specified set of dimensions.
    
    Args:
        x: The input symbolic tensor to rebind.
        shape: The symbolic shape to assert for x.
        message: The message printed if the rebind fails at runtime.
    
    Returns:
        Tensor with the symbolic shape asserted.
    """
    var ops = PythonBridge.get_module("main")
    var py_shape = PythonBridge.shape_to_python(shape)
    var result_tensor = ops.rebind(x.get_graph()[].graph, x.to_python(), py_shape, message)
    return TensorValue(x.get_graph(), result_tensor)


fn transfer_to(x: TensorValue, device: Device) raises -> TensorValue:
    """Device-to-Device transfer operation.
    
    Args:
        x: The input tensor to transfer.
        device: The device to transfer to.
    
    Returns:
        Tensor transferred to specified device.
    """
    var ops = PythonBridge.get_module("main")
    # Note: DeviceType.Accelerator() or DeviceType.GPU() both map to "gpu:0"
    var device_str = String("cpu:0") if device.device_type == DeviceType.CPU() else String("gpu:0")
    var result_tensor = ops.transfer_to(x.get_graph()[].graph, x.to_python(), device_str)
    return TensorValue(x.get_graph(), result_tensor)


fn irfft(input_tensor: TensorValue, n: Int, axis: Int = -1, normalization: String = "backward", 
         input_is_complex: Bool = False, buffer_size_mb: Int = 512) raises -> TensorValue:
    """Compute the inverse real FFT of the input tensor.
    
    Args:
        input_tensor: The input tensor to compute the inverse real FFT of.
        n: The size of the output tensor.
        axis: The axis to compute the inverse real FFT of.
        normalization: The normalization to apply ("backward", "ortho", or "forward").
        input_is_complex: Whether the input tensor is already interleaved complex.
        buffer_size_mb: The estimated size of a persistent buffer.
    
    Returns:
        The inverse real FFT of the input tensor.
    """
    var ops = PythonBridge.get_module("main")
    var result_tensor = ops.irfft(input_tensor.get_graph()[].graph, input_tensor.to_python(), 
                                  n, axis, normalization, input_is_complex, buffer_size_mb)
    return TensorValue(input_tensor.get_graph(), result_tensor)
