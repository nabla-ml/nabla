import numpy as np

def gen_random_values(size, base):
    # generate a size x size array of random numbers between base and base+1
    random_array = np.random.rand(size, size)
    return random_array + base


# Build the graph
# Now with our environment and packages setup, lets create the graph. This graph will define a computational workflow that adds two tensors together.

# Let's start by creating a new file called addition.py inside of your working directory and add the following libraries:

from max import engine
from max.driver import CPU, Tensor
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops, TensorValue

class MaxGraph:
    """Wrapper for MAX Graph that simplifies graph construction
    
    Attributes:
        graph: The underlying MAX Graph object
        name: Name of the graph
    """
    graph: Graph 
    name: str
    _device: CPU | None

    # MAX Graph itself has the following parameters:
    # name (str) – A name for the graph.
    # input_types (Iterable[Type[Any]]) – The data type(s) for the input tensor(s).
    # path (Path | None) – The path to a saved graph (internal use only).
    # custom_extensions (Iterable[Path]) – The extensions to load for the model. Supports paths to .mojopkg or .mojo sources with custom ops.
    # context (mlir.Context | None)
    # kernel_library (KernelLibrary | None)
    # module (mlir.Module | None)

    def __init__(self, 
        name: str, 
        input_types,  # Accept either a single TensorType or an iterable
        device: CPU | None = None,
        custom_extensions: list = None
        ) -> None:
        """Initialize a MAX Graph wrapper
        
        Args:
            name: Name for the graph
            input_types: Single TensorType or list/tuple of TensorTypes
            device: Device to use (defaults to CPU)
            custom_extensions: List of paths to custom op extensions (.mojopkg or .mojo files)
        """
        # Check if input_types is already a list/tuple, if not wrap it
        if isinstance(input_types, (list, tuple)):
            types_list = input_types
        else:
            types_list = [input_types]
        
        self.name = name
        self._device = device if device is not None else CPU()
        
        # Create graph with or without custom extensions
        if custom_extensions:
            from pathlib import Path
            # Convert string paths to Path objects if needed
            extension_paths = [Path(p) if isinstance(p, str) else p for p in custom_extensions]
            self.graph = Graph(
                name,
                input_types=types_list,
                custom_extensions=extension_paths
            )
        else:
            self.graph = Graph(
                name,
                input_types=types_list,
            )

    def output(self, args: list[TensorValue]) -> None:
        """Set the graph's outputs
        
        Args:
            args: List of TensorValue outputs from operations
        """
        with self.graph as g:
            g.output(*args)

    @property
    def inputs(self):
        """Expose the graph's inputs as symbolic tensors"""
        return self.graph.inputs
    
    def compile(self, devices: list | None = None):
        """Compile the graph into a model ready for inference, wrapped in MaxModel
        
        Args:
            devices: List of devices to use for inference (defaults to [CPU()])
            
        Returns:
            MaxModel: Wrapped model ready for execution
        """
        if devices is None:
            devices = [self._device]
        
        session = engine.InferenceSession(devices=devices)
        model = session.load(self.graph)
        return MaxModel(model)


class MaxModel:
    """Wrapper for MAX Model that accepts list of tensors for execution"""
    model: engine.Model
    
    def __init__(self, model: engine.Model) -> None:
        self.model = model
        self._num_inputs = len(model.input_metadata)
    
    def __call__(self, inputs: list[Tensor]) -> list[Tensor]:
        """Execute the model with a list of input tensors"""
        return self.execute(inputs)
    
    def execute(self, inputs: list[Tensor]) -> list[Tensor]:
        """Execute the model with a list of input tensors
        
        Args:
            inputs: List of input tensors matching the model's input specifications
            
        Returns:
            List of output tensors
            
        Raises:
            ValueError: If input count doesn't match model requirements
            TypeError: If inputs are not tensors
        """
        # Validate input count
        if len(inputs) != self._num_inputs:
            raise ValueError(
                f"Model expects {self._num_inputs} inputs, got {len(inputs)}"
            )
        
        # Validate input types
        for i, inp in enumerate(inputs):
            if not isinstance(inp, Tensor):
                raise TypeError(
                    f"Input {i} must be a Tensor, got {type(inp).__name__}"
                )
        
        # Execute with unpacked inputs
        try:
            return self.model(*inputs)
        except Exception as e:
            raise RuntimeError(f"Model execution failed: {str(e)}") from e
    
    @property
    def input_metadata(self):
        """Get input specifications"""
        return self.model.input_metadata
    
    @property
    def num_inputs(self) -> int:
        """Get number of expected inputs"""
        return self._num_inputs


def add(ctx: MaxGraph, a: TensorValue, b: TensorValue) -> TensorValue:
    """Add two tensors within the given graph context."""
    with ctx.graph:
        return ops.add(a, b)
    
def matmul(ctx: MaxGraph, a: TensorValue, b: TensorValue) -> TensorValue:
    """Matrix multiply two tensors within the given graph context."""
    with ctx.graph:
        return ops.matmul(a, b)

def sub(ctx: MaxGraph, a: TensorValue, b: TensorValue) -> TensorValue:
    """Subtract two tensors within the given graph context."""
    with ctx.graph:
        return ops.sub(a, b)

def mul(ctx: MaxGraph, a: TensorValue, b: TensorValue) -> TensorValue:
    """Element-wise multiply two tensors within the given graph context."""
    with ctx.graph:
        return ops.mul(a, b)

def div(ctx: MaxGraph, a: TensorValue, b: TensorValue) -> TensorValue:
    """Element-wise divide two tensors within the given graph context."""
    with ctx.graph:
        return ops.div(a, b)


# ============================================================================
# UNARY OPERATIONS
# ============================================================================

def abs(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Absolute value of tensor."""
    with ctx.graph:
        return ops.abs(x)

def negate(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Negate tensor (multiply by -1)."""
    with ctx.graph:
        return ops.negate(x)

def relu(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """ReLU activation: max(0, x)."""
    with ctx.graph:
        return ops.relu(x)

def sigmoid(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Sigmoid activation: 1 / (1 + exp(-x))."""
    with ctx.graph:
        return ops.sigmoid(x)

def tanh(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Hyperbolic tangent activation."""
    with ctx.graph:
        return ops.tanh(x)

def exp(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Exponential function: e^x."""
    with ctx.graph:
        return ops.exp(x)

def log(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Natural logarithm."""
    with ctx.graph:
        return ops.log(x)

def sqrt(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Square root."""
    with ctx.graph:
        return ops.sqrt(x)


# ============================================================================
# REDUCTION OPERATIONS (with axis parameter)
# ============================================================================

def mean(ctx: MaxGraph, x: TensorValue, axis: int = -1) -> TensorValue:
    """Mean reduction along specified axis."""
    with ctx.graph:
        return ops.mean(x, axis=axis)


# ============================================================================
# MATHEMATICAL FUNCTIONS (Batch 2)
# ============================================================================

def atanh(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Inverse hyperbolic tangent."""
    with ctx.graph:
        return ops.atanh(x)

def cos(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Cosine function."""
    with ctx.graph:
        return ops.cos(x)

def erf(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Error function."""
    with ctx.graph:
        return ops.erf(x)

def floor(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Floor function (round down to nearest integer)."""
    with ctx.graph:
        return ops.floor(x)

def is_inf(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Check if elements are infinity."""
    with ctx.graph:
        return ops.is_inf(x)

def is_nan(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Check if elements are NaN."""
    with ctx.graph:
        return ops.is_nan(x)


# ============================================================================
# ACTIVATION FUNCTIONS (Batch 2)
# ============================================================================

def gelu(ctx: MaxGraph, x: TensorValue, approximate: str = "none") -> TensorValue:
    """GELU activation function.
    
    Args:
        ctx: Graph context
        x: Input tensor
        approximate: Approximation mode - "none", "tanh", or "quick"
    """
    with ctx.graph:
        return ops.gelu(x, approximate=approximate)


# ============================================================================
# COMPARISON OPERATIONS
# ============================================================================

def equal(ctx: MaxGraph, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    """Element-wise equality comparison."""
    with ctx.graph:
        return ops.equal(lhs, rhs)

def greater(ctx: MaxGraph, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    """Element-wise greater than comparison."""
    with ctx.graph:
        return ops.greater(lhs, rhs)

def greater_equal(ctx: MaxGraph, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    """Element-wise greater than or equal comparison."""
    with ctx.graph:
        return ops.greater_equal(lhs, rhs)


def not_equal(ctx: MaxGraph, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    """Element-wise not equal comparison."""
    with ctx.graph:
        return ops.not_equal(lhs, rhs)


# ============================================================================
# LOGICAL OPERATIONS
# ============================================================================

def logical_and(ctx: MaxGraph, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    """Element-wise logical AND."""
    with ctx.graph:
        return ops.logical_and(lhs, rhs)


def logical_or(ctx: MaxGraph, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    """Element-wise logical OR."""
    with ctx.graph:
        return ops.logical_or(lhs, rhs)


def logical_xor(ctx: MaxGraph, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    """Element-wise logical XOR."""
    with ctx.graph:
        return ops.logical_xor(lhs, rhs)


def logical_not(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Element-wise logical NOT."""
    with ctx.graph:
        return ops.logical_not(x)


# ============================================================================
# ADDITIONAL MATH FUNCTIONS (Batch 3)
# ============================================================================

def log1p(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Compute log(1 + x) element-wise."""
    with ctx.graph:
        return ops.log1p(x)


def logsoftmax(ctx: MaxGraph, x: TensorValue, axis: int = -1) -> TensorValue:
    """Log-softmax activation along specified axis."""
    with ctx.graph:
        return ops.logsoftmax(x, axis=axis)


# ============================================================================
# ADDITIONAL BINARY OPERATIONS (Batch 3)
# ============================================================================

def mod(ctx: MaxGraph, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    """Element-wise modulo operation."""
    with ctx.graph:
        return ops.mod(lhs, rhs)


def pow(ctx: MaxGraph, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    """Element-wise power operation."""
    with ctx.graph:
        return ops.pow(lhs, rhs)


# ============================================================================
# BATCH 4: REDUCTION OPERATIONS
# ============================================================================

def sum(ctx: MaxGraph, x: TensorValue, axis: int = -1) -> TensorValue:
    """Sum reduction along specified axis."""
    with ctx.graph:
        return ops.sum(x, axis=axis)


def max(ctx: MaxGraph, x: TensorValue, axis: int = -1) -> TensorValue:
    """Max reduction along specified axis."""
    with ctx.graph:
        return ops.max(x, axis=axis)


def min(ctx: MaxGraph, x: TensorValue, axis: int = -1) -> TensorValue:
    """Min reduction along specified axis."""
    with ctx.graph:
        return ops.min(x, axis=axis)


def argmax(ctx: MaxGraph, x: TensorValue, axis: int = -1) -> TensorValue:
    """Argmax reduction along specified axis."""
    with ctx.graph:
        return ops.argmax(x, axis=axis)


def argmin(ctx: MaxGraph, x: TensorValue, axis: int = -1) -> TensorValue:
    """Argmin reduction along specified axis."""
    with ctx.graph:
        return ops.argmin(x, axis=axis)


# ============================================================================
# BATCH 4: MORE UNARY MATH FUNCTIONS
# ============================================================================

def sin(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Sine function."""
    with ctx.graph:
        return ops.sin(x)


def rsqrt(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Reciprocal square root: 1/sqrt(x)."""
    with ctx.graph:
        return ops.rsqrt(x)


def round(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Round to nearest integer."""
    with ctx.graph:
        return ops.round(x)


def trunc(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Truncate to integer (round toward zero)."""
    with ctx.graph:
        return ops.trunc(x)


# ============================================================================
# BATCH 4: MORE ACTIVATION FUNCTIONS
# ============================================================================

def softmax(ctx: MaxGraph, x: TensorValue, axis: int = -1) -> TensorValue:
    """Softmax activation along specified axis."""
    with ctx.graph:
        return ops.softmax(x, axis=axis)


def silu(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """SiLU (Swish) activation: x * sigmoid(x)."""
    with ctx.graph:
        return ops.silu(x)


# ============================================================================
# SHAPE OPERATIONS (Batch 5)
# ============================================================================

def reshape(ctx: MaxGraph, x: TensorValue, shape: tuple) -> TensorValue:
    """Reshape tensor to new shape.
    
    Args:
        x: Input tensor
        shape: Target shape as Python tuple
    """
    with ctx.graph:
        return ops.reshape(x, shape)


def flatten(ctx: MaxGraph, x: TensorValue, start_dim: int = 0, end_dim: int = -1) -> TensorValue:
    """Flatten tensor dimensions from start_dim to end_dim.
    
    Args:
        x: Input tensor
        start_dim: First dimension to flatten (default: 0)
        end_dim: Last dimension to flatten (default: -1)
    """
    with ctx.graph:
        return ops.flatten(x, start_dim=start_dim, end_dim=end_dim)


def squeeze(ctx: MaxGraph, x: TensorValue, axis: int) -> TensorValue:
    """Remove dimension of size 1 at specified axis.
    
    Args:
        x: Input tensor
        axis: Axis to squeeze
    """
    with ctx.graph:
        return ops.squeeze(x, axis=axis)


def unsqueeze(ctx: MaxGraph, x: TensorValue, axis: int) -> TensorValue:
    """Add dimension of size 1 at specified axis.
    
    Args:
        x: Input tensor
        axis: Axis to add dimension
    """
    with ctx.graph:
        return ops.unsqueeze(x, axis=axis)


def permute(ctx: MaxGraph, x: TensorValue, dims: tuple) -> TensorValue:
    """Permute dimensions of tensor.
    
    Args:
        x: Input tensor
        dims: New ordering of dimensions as Python tuple
    """
    with ctx.graph:
        return ops.permute(x, dims)


def broadcast_to(ctx: MaxGraph, x: TensorValue, shape: tuple) -> TensorValue:
    """Broadcast tensor to new shape.
    
    Args:
        x: Input tensor
        shape: Target shape as Python tuple
    """
    with ctx.graph:
        return ops.broadcast_to(x, shape)


def cast(ctx: MaxGraph, x: TensorValue, dtype: str) -> TensorValue:
    """Cast tensor to different data type.
    
    Args:
        x: Input tensor
        dtype: Target dtype as string (e.g., 'float32', 'int64')
    """
    with ctx.graph:
        # Convert string dtype to MAX DType
        from max.dtype import DType as MAX_DType
        dtype_map = {
            'float32': MAX_DType.float32,
            'float64': MAX_DType.float64,
            'int32': MAX_DType.int32,
            'int64': MAX_DType.int64,
            'int8': MAX_DType.int8,
            'uint8': MAX_DType.uint8,
            'bool': MAX_DType.bool,
        }
        target_dtype = dtype_map.get(dtype)
        if target_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return ops.cast(x, target_dtype)


def transpose(ctx: MaxGraph, x: TensorValue, axis_1: int, axis_2: int) -> TensorValue:
    """Transpose two axes of tensor.
    
    Args:
        x: Input tensor
        axis_1: First axis
        axis_2: Second axis
    """
    with ctx.graph:
        return ops.transpose(x, axis_1, axis_2)


# ============================================================================
# TENSOR MANIPULATION (Batch 6)
# ============================================================================

def concat(ctx: MaxGraph, vals: list, axis: int = 0) -> TensorValue:
    """Concatenate tensors along specified axis.
    
    Args:
        vals: List of tensors to concatenate
        axis: Axis along which to concatenate (default: 0)
    """
    with ctx.graph:
        return ops.concat(vals, axis=axis)


def split(ctx: MaxGraph, x: TensorValue, split_sizes: tuple, axis: int = 0) -> list:
    """Split tensor into multiple tensors.
    
    Args:
        x: Input tensor
        split_sizes: Sizes of each split as tuple
        axis: Axis along which to split (default: 0)
    
    Returns:
        List of split tensors
    """
    with ctx.graph:
        return ops.split(x, split_sizes, axis=axis)


def chunk(ctx: MaxGraph, x: TensorValue, chunks: int, axis: int = 0) -> list:
    """Split tensor into equal chunks.
    
    Args:
        x: Input tensor
        chunks: Number of chunks
        axis: Axis along which to chunk (default: 0)
    
    Returns:
        List of chunked tensors
    """
    with ctx.graph:
        return ops.chunk(x, chunks, axis=axis)


def gather(ctx: MaxGraph, input: TensorValue, indices: TensorValue, axis: int) -> TensorValue:
    """Gather values along an axis specified by indices.
    
    Args:
        input: Input tensor
        indices: Indices tensor
        axis: Axis along which to gather
    """
    with ctx.graph:
        return ops.gather(input, indices, axis=axis)


def pad(ctx: MaxGraph, input: TensorValue, paddings: tuple, mode: str = 'constant', value: float = 0.0) -> TensorValue:
    """Pad tensor with specified padding.
    
    Args:
        input: Input tensor
        paddings: Padding sizes as tuple
        mode: Padding mode ('constant', 'reflect', 'replicate', 'circular')
        value: Fill value for constant padding (default: 0.0)
    """
    with ctx.graph:
        return ops.pad(input, paddings, mode=mode, value=value)


# ============================================================================
# Batch 7: Common Missing Operations (10 ops)
# ============================================================================

def stack(ctx: MaxGraph, tensors: list, axis: int = 0) -> TensorValue:
    """Stack tensors along a new axis.
    
    Args:
        tensors: List of tensors to stack
        axis: Axis along which to stack (default: 0)
    """
    with ctx.graph:
        return ops.stack(tensors, axis=axis)


def tile(ctx: MaxGraph, x: TensorValue, repeats: tuple) -> TensorValue:
    """Tile/repeat tensor along each dimension.
    
    Args:
        x: Input tensor
        repeats: Number of repetitions for each dimension
    """
    with ctx.graph:
        return ops.tile(x, repeats)


def repeat_interleave(ctx: MaxGraph, x: TensorValue, repeats: int, axis: int = 0) -> TensorValue:
    """Repeat elements of a tensor.
    
    Args:
        x: Input tensor
        repeats: Number of repetitions for each element
        axis: Axis along which to repeat (default: 0)
    """
    with ctx.graph:
        return ops.repeat_interleave(x, repeats, axis=axis)


def slice_tensor(ctx: MaxGraph, x: TensorValue, indices: list) -> TensorValue:
    """Slice tensor using index specifications.
    
    Args:
        x: Input tensor
        indices: List of slice specifications (can include slice objects, integers, ellipsis, None)
    """
    with ctx.graph:
        return ops.slice_tensor(x, indices)


def where(ctx: MaxGraph, condition: TensorValue, x: TensorValue, y: TensorValue) -> TensorValue:
    """Select elements from x or y depending on condition.
    
    Args:
        condition: Boolean condition tensor
        x: Values to select when condition is True
        y: Values to select when condition is False
    """
    with ctx.graph:
        return ops.where(condition, x, y)


def outer(ctx: MaxGraph, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    """Compute outer product of two vectors.
    
    Args:
        lhs: First vector
        rhs: Second vector
    """
    with ctx.graph:
        return ops.outer(lhs, rhs)


def cumsum(ctx: MaxGraph, x: TensorValue, axis: int = -1, exclusive: bool = False, reverse: bool = False) -> TensorValue:
    """Cumulative sum along an axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to compute cumsum (default: -1)
        exclusive: If True, exclude current element (default: False)
        reverse: If True, compute cumsum in reverse (default: False)
    """
    with ctx.graph:
        return ops.cumsum(x, axis=axis, exclusive=exclusive, reverse=reverse)


def argsort(ctx: MaxGraph, x: TensorValue, ascending: bool = True) -> TensorValue:
    """Return indices that would sort the tensor.
    
    Args:
        x: Input tensor
        ascending: Sort in ascending order if True (default: True)
    """
    with ctx.graph:
        return ops.argsort(x, ascending=ascending)


def top_k(ctx: MaxGraph, input: TensorValue, k: int, axis: int = -1) -> tuple:
    """Find top k values and their indices.
    
    Args:
        input: Input tensor
        k: Number of top elements
        axis: Axis along which to find top k (default: -1)
    
    Returns:
        Tuple of (values, indices)
    """
    with ctx.graph:
        result = ops.top_k(input, k, axis=axis)
        # top_k returns a tuple of (values, indices)
        return result


def scatter(ctx: MaxGraph, input: TensorValue, updates: TensorValue, indices: TensorValue, axis: int = -1) -> TensorValue:
    """Scatter updates into input tensor at specified indices.
    
    Args:
        input: Input tensor
        updates: Update values
        indices: Indices where to scatter updates
        axis: Axis along which to scatter (default: -1)
    """
    with ctx.graph:
        return ops.scatter(input, updates, indices, axis=axis)


def scatter_nd(ctx: MaxGraph, input: TensorValue, updates: TensorValue, indices: TensorValue) -> TensorValue:
    """Scatter updates into input tensor using N-dimensional indices.
    
    Args:
        input: Input tensor
        updates: Update values
        indices: N-dimensional indices
    """
    with ctx.graph:
        return ops.scatter_nd(input, updates, indices)


def gather_nd(ctx: MaxGraph, input: TensorValue, indices: TensorValue, batch_dims: int = 0) -> TensorValue:
    """Gather values from input using N-dimensional indices.
    
    Args:
        input: Input tensor
        indices: N-dimensional indices
        batch_dims: Number of batch dimensions (default: 0)
    """
    with ctx.graph:
        return ops.gather_nd(input, indices, batch_dims=batch_dims)


# ============================================================================
# Batch 8: Conv, Pooling & Normalization (5 ops)
# ============================================================================

def conv2d(ctx: MaxGraph, x: TensorValue, filter: TensorValue, stride: tuple = (1, 1), 
           dilation: tuple = (1, 1), padding: tuple = (0, 0, 0, 0), groups: int = 1, 
           bias: TensorValue = None) -> TensorValue:
    """2D convolution operation.
    
    Args:
        x: Input tensor
        filter: Filter/kernel tensor
        stride: Stride for height and width (default: (1, 1))
        dilation: Dilation for height and width (default: (1, 1))
        padding: Padding (top, bottom, left, right) (default: (0, 0, 0, 0))
        groups: Number of groups for grouped convolution (default: 1)
        bias: Optional bias tensor (default: None)
    """
    with ctx.graph:
        if bias is not None:
            return ops.conv2d(x, filter, stride=stride, dilation=dilation, 
                            padding=padding, groups=groups, bias=bias)
        else:
            return ops.conv2d(x, filter, stride=stride, dilation=dilation, 
                            padding=padding, groups=groups)


def conv2d_transpose(ctx: MaxGraph, x: TensorValue, filter: TensorValue, stride: tuple = (1, 1),
                     dilation: tuple = (1, 1), padding: tuple = (0, 0, 0, 0), 
                     output_paddings: tuple = (0, 0), bias: TensorValue = None) -> TensorValue:
    """2D transposed convolution (deconvolution) operation.
    
    Args:
        x: Input tensor
        filter: Filter/kernel tensor
        stride: Stride for height and width (default: (1, 1))
        dilation: Dilation for height and width (default: (1, 1))
        padding: Padding (top, bottom, left, right) (default: (0, 0, 0, 0))
        output_paddings: Output padding for height and width (default: (0, 0))
        bias: Optional bias tensor (default: None)
    """
    with ctx.graph:
        if bias is not None:
            return ops.conv2d_transpose(x, filter, stride=stride, dilation=dilation,
                                      padding=padding, output_paddings=output_paddings, bias=bias)
        else:
            return ops.conv2d_transpose(x, filter, stride=stride, dilation=dilation,
                                      padding=padding, output_paddings=output_paddings)


def max_pool2d(ctx: MaxGraph, input: TensorValue, kernel_size: tuple, stride: tuple = (1, 1),
               dilation: tuple = (1, 1), padding: tuple = (0, 0, 0, 0), ceil_mode: bool = False) -> TensorValue:
    """2D max pooling operation.
    
    Args:
        input: Input tensor
        kernel_size: Size of pooling kernel (height, width)
        stride: Stride for height and width (default: (1, 1))
        dilation: Dilation for height and width (default: (1, 1))
        padding: Padding (top, bottom, left, right) (default: (0, 0, 0, 0))
        ceil_mode: Use ceil instead of floor for output shape (default: False)
    """
    with ctx.graph:
        return ops.max_pool2d(input, kernel_size, stride=stride, dilation=dilation,
                            padding=padding, ceil_mode=ceil_mode)


def avg_pool2d(ctx: MaxGraph, input: TensorValue, kernel_size: tuple, stride: tuple = (1, 1),
               dilation: tuple = (1, 1), padding: tuple = (0, 0, 0, 0), 
               ceil_mode: bool = False, count_boundary: bool = True) -> TensorValue:
    """2D average pooling operation.
    
    Args:
        input: Input tensor
        kernel_size: Size of pooling kernel (height, width)
        stride: Stride for height and width (default: (1, 1))
        dilation: Dilation for height and width (default: (1, 1))
        padding: Padding (top, bottom, left, right) (default: (0, 0, 0, 0))
        ceil_mode: Use ceil instead of floor for output shape (default: False)
        count_boundary: Include padding in average calculation (default: True)
    """
    with ctx.graph:
        return ops.avg_pool2d(input, kernel_size, stride=stride, dilation=dilation,
                            padding=padding, ceil_mode=ceil_mode, count_boundary=count_boundary)


def layer_norm(ctx: MaxGraph, input: TensorValue, gamma: TensorValue, beta: TensorValue, 
               epsilon: float) -> TensorValue:
    """Layer normalization.
    
    Args:
        input: Input tensor
        gamma: Scale parameter
        beta: Shift parameter
        epsilon: Small value for numerical stability
    """
    with ctx.graph:
        return ops.layer_norm(input, gamma, beta, epsilon)


# ============================================================================
# Batch 9: Additional Utilities (6 ops)
# ============================================================================

def resize(ctx: MaxGraph, input: TensorValue, shape: tuple, interpolation: str = 'BILINEAR') -> TensorValue:
    """Resize tensor to target shape.
    
    Args:
        input: Input tensor
        shape: Target shape
        interpolation: Interpolation mode ('BILINEAR', 'NEAREST', etc.) (default: 'BILINEAR')
    """
    with ctx.graph:
        # Import InterpolationMode enum if needed
        from max.graph.ops import InterpolationMode
        mode = getattr(InterpolationMode, interpolation.upper())
        return ops.resize(input, shape, interpolation=mode)


def nonzero(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Get indices of non-zero elements.
    
    Args:
        x: Input tensor
    
    Note: Requires out_dim parameter which specifies output dimension.
    """
    with ctx.graph:
        # out_dim needs to be provided - for now, use a symbolic dimension
        return ops.nonzero(x, out_dim=None)


def clamp(ctx: MaxGraph, x: TensorValue, min_val: float = None, max_val: float = None) -> TensorValue:
    """Clamp tensor values to a range.
    
    Args:
        x: Input tensor
        min_val: Minimum value (optional)
        max_val: Maximum value (optional)
    """
    with ctx.graph:
        # MAX doesn't have clamp directly, but we can use min/max ops
        result = x
        if min_val is not None:
            result = ops.max(result, ops.constant(min_val, dtype=x.tensor.dtype))
        if max_val is not None:
            result = ops.min(result, ops.constant(max_val, dtype=x.tensor.dtype))
        return result


def less(ctx: MaxGraph, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    """Element-wise less than comparison.
    
    Args:
        lhs: Left operand
        rhs: Right operand
    """
    with ctx.graph:
        # MAX may not have direct 'less', but can use: lhs < rhs = not(lhs >= rhs)
        return ops.logical_not(ops.greater_equal(lhs, rhs))


def less_equal(ctx: MaxGraph, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    """Element-wise less than or equal comparison.
    
    Args:
        lhs: Left operand
        rhs: Right operand
    """
    with ctx.graph:
        # lhs <= rhs = not(lhs > rhs)
        return ops.logical_not(ops.greater(lhs, rhs))


def clip_by_value(ctx: MaxGraph, x: TensorValue, min_val: TensorValue, max_val: TensorValue) -> TensorValue:
    """Clip tensor values element-wise.
    
    Args:
        x: Input tensor
        min_val: Minimum value tensor
        max_val: Maximum value tensor
    """
    with ctx.graph:
        # Clip using min and max operations
        result = ops.max(x, min_val)
        result = ops.min(result, max_val)
        return result


def band_part(ctx: MaxGraph, x: TensorValue, num_lower: int, num_upper: int) -> TensorValue:
    """Extract band from tensor.
    
    Args:
        x: Input tensor
        num_lower: Number of lower diagonals to keep
        num_upper: Number of upper diagonals to keep
    """
    with ctx.graph:
        return ops.band_part(x, num_lower, num_upper)


def conv3d(ctx: MaxGraph, input: TensorValue, weight: TensorValue,
           stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0, 0, 0, 0), dilation: tuple = (1, 1, 1)) -> TensorValue:
    """3D convolution operation.
    
    Args:
        input: Input tensor (NDHWC format)
        weight: Filter/kernel tensor
        stride: Stride for each dimension (3 values)
        padding: Padding for each dimension (6 values: before/after for each dim)
        dilation: Dilation for each dimension (3 values)
    """
    with ctx.graph:
        return ops.conv3d(input, weight, stride=stride, padding=padding, dilation=dilation)


def hann_window(ctx: MaxGraph, size: int, periodic: bool, device: str = "cpu:0") -> TensorValue:
    """Generate Hann window.
    
    Args:
        size: Window size (integer)
        periodic: Whether window is periodic
        device: Device string
    """
    with ctx.graph:
        from max.graph import DeviceRef
        dev = DeviceRef.CPU()
        return ops.hann_window(size, periodic=periodic, device=dev)


def masked_scatter(ctx: MaxGraph, x: TensorValue, mask: TensorValue, source: TensorValue, out_dim: int = 0) -> TensorValue:
    """Scatter source values into x where mask is true.
    
    Args:
        x: Input tensor
        mask: Boolean mask tensor
        source: Source values to scatter
        out_dim: Output dimension
    """
    with ctx.graph:
        return ops.masked_scatter(x, mask, source, out_dim=out_dim)


# ============================================================================
# Batch 11: Additional Missing Core Operations
# ============================================================================

def allgather(ctx: MaxGraph, inputs: list, signal_buffers: list, axis: int = 0) -> list:
    """Collective allgather operation.
    
    Args:
        inputs: The input tensors to gather
        signal_buffers: Device buffer values used for synchronization
        axis: Dimension to concatenate the input tensors (default: 0)
    
    Returns:
        List of gathered output tensors
    """
    with ctx.graph:
        return ops.allgather(inputs, signal_buffers, axis=axis)


def as_interleaved_complex(ctx: MaxGraph, x: TensorValue) -> TensorValue:
    """Reshape input tensor as complex from alternating (real, imag).
    
    Args:
        x: Input tensor with alternating real/imag values
    
    Returns:
        Complex-valued tensor
    """
    with ctx.graph:
        return ops.as_interleaved_complex(x)


def assert_same_device(ctx: MaxGraph, *values, **named_values) -> None:
    """Assert that all tensor values are on the same device.
    
    Args:
        *values: Variable number of tensor values to check
        **named_values: Named tensor values to check
    
    Raises:
        Error if tensors are not on the same device
    """
    with ctx.graph:
        ops.assert_same_device(*values, **named_values)


def buffer_create(ctx: MaxGraph, buffer_type):
    """Create a buffer of the given type.
    
    Args:
        buffer_type: The type of the resulting BufferValue
    
    Returns:
        A new BufferValue of the requested type
    """
    with ctx.graph:
        return ops.buffer_create(buffer_type)


def buffer_load(ctx: MaxGraph, x) -> TensorValue:
    """Load buffer into a tensor.
    
    Args:
        x: The buffer to be loaded
    
    Returns:
        Tensor graph value representing a copy of the buffer
    """
    with ctx.graph:
        return ops.buffer_load(x)


def buffer_store(ctx: MaxGraph, destination, source: TensorValue) -> None:
    """Store tensor into buffer.
    
    Args:
        destination: The buffer to store the tensor in
        source: The tensor to be stored in the buffer
    """
    with ctx.graph:
        ops.buffer_store(destination, source)


def buffer_store_slice(ctx: MaxGraph, destination, source: TensorValue, indices) -> None:
    """Store tensor into a slice in the buffer.
    
    Args:
        destination: The buffer to store the tensor in
        source: The tensor to be stored
        indices: The index in the buffer where the tensor should be stored
    """
    with ctx.graph:
        ops.buffer_store_slice(destination, source, indices)


def cond(ctx: MaxGraph, pred: TensorValue, out_types, then_fn, else_fn) -> list:
    """Conditionally execute one of two branches based on a boolean predicate.
    
    Args:
        pred: Boolean scalar tensor determining branch execution
        out_types: Expected output types for both branches
        then_fn: Callable executed when pred is True
        else_fn: Callable executed when pred is False
    
    Returns:
        List of output values from executed branch
    """
    with ctx.graph:
        return ops.cond(pred, out_types, then_fn, else_fn)


def constant(ctx: MaxGraph, value, dtype=None, device=None) -> TensorValue:
    """Add a constant operation node.
    
    Args:
        value: The constant's value
        dtype: The constant tensor's element type
        device: The device the constant lives on
    
    Returns:
        Graph value containing the constant data
    """
    with ctx.graph:
        if device is not None:
            from max.graph import DeviceRef
            if isinstance(device, str):
                # Parse device string like "cpu:0" or "gpu:0"
                parts = device.split(':')
                if parts[0].lower() == 'cpu':
                    device_ref = DeviceRef.CPU(id=int(parts[1]) if len(parts) > 1 else 0)
                elif parts[0].lower() == 'gpu':
                    device_ref = DeviceRef.GPU(id=int(parts[1]) if len(parts) > 1 else 0)
                else:
                    device_ref = device
            else:
                device_ref = device
            return ops.constant(value, dtype=dtype, device=device_ref)
        else:
            return ops.constant(value, dtype=dtype)


def constant_external(ctx: MaxGraph, name: str, tensor_type) -> TensorValue:
    """Register an external constant (weight) in the graph.
    
    Args:
        name: The name of the external constant
        tensor_type: The type of the constant value
    
    Returns:
        Tensor value representing the weight
    """
    with ctx.graph:
        return ops.constant_external(name, tensor_type)


def custom(ctx: MaxGraph, name: str, device, values: list, out_types: list, parameters: dict = None) -> list:
    """Create a node to execute a custom graph operation.
    
    Args:
        name: The op name provided to @compiler.register
        device: Device that the op is assigned to
        values: The op function's arguments
        out_types: The list of op function's return type
        parameters: Dictionary of extra parameters expected by the kernel
    
    Returns:
        Symbolic values representing the outputs of the op
    """
    with ctx.graph:
        if parameters is not None:
            return ops.custom(name, device=device, values=values, out_types=out_types, parameters=parameters)
        else:
            return ops.custom(name, device=device, values=values, out_types=out_types)


def dequantize(ctx: MaxGraph, encoding, quantized: TensorValue) -> TensorValue:
    """Dequantize a quantized tensor to floating point.
    
    Args:
        encoding: The quantization encoding to use
        quantized: The quantized tensor to dequantize
    
    Returns:
        The dequantized result (floating point tensor)
    """
    with ctx.graph:
        return ops.dequantize(encoding, quantized)


def fold(ctx: MaxGraph, input: TensorValue, output_size: tuple, kernel_size: tuple, 
         stride: int | tuple = 1, dilation: int | tuple = 1, padding: int | tuple = 0) -> TensorValue:
    """Combine array of sliding blocks into larger containing tensor.
    
    Args:
        input: The 3D tensor to fold with shape (N, C * kernel sizes, L)
        output_size: Spatial dimensions of the output tensor (2 ints)
        kernel_size: The size of the sliding blocks (2 ints)
        stride: The stride of the sliding blocks
        dilation: The spacing between the kernel elements
        padding: 0-paddings to be added on both sides
    
    Returns:
        The folded 4D tensor
    """
    with ctx.graph:
        return ops.fold(input, output_size, kernel_size, stride=stride, dilation=dilation, padding=padding)


def inplace_custom(ctx: MaxGraph, name: str, device, values: list, out_types=None, parameters: dict = None) -> list:
    """Create a node to execute an in-place custom graph operation.
    
    Args:
        name: The op name provided to @compiler.register
        device: Device that the op is assigned to
        values: The op function's arguments
        out_types: The list of op function's return type (optional)
        parameters: Dictionary of extra parameters expected by the kernel
    
    Returns:
        Symbolic values representing the outputs of the op
    """
    with ctx.graph:
        if parameters is not None:
            return ops.inplace_custom(name, device=device, values=values, out_types=out_types, parameters=parameters)
        else:
            return ops.inplace_custom(name, device=device, values=values, out_types=out_types)


def irfft(ctx: MaxGraph, input_tensor: TensorValue, n: int = None, axis: int = -1, 
          normalization: str = 'backward', input_is_complex: bool = False, 
          buffer_size_mb: int = 512) -> TensorValue:
    """Compute the inverse real FFT of the input tensor.
    
    Args:
        input_tensor: The input tensor to compute the inverse real FFT of
        n: The size of the output tensor
        axis: The axis to compute the inverse real FFT of
        normalization: The normalization to apply ("backward", "ortho", or "forward")
        input_is_complex: Whether the input tensor is already interleaved complex
        buffer_size_mb: The estimated size of a persistent buffer
    
    Returns:
        The inverse real FFT of the input tensor
    """
    with ctx.graph:
        from max.graph.ops import Normalization
        norm_map = {
            'backward': Normalization.BACKWARD,
            'ortho': Normalization.ORTHO,
            'forward': Normalization.FORWARD
        }
        norm = norm_map.get(normalization.lower(), Normalization.BACKWARD)
        
        if n is not None:
            return ops.irfft(input_tensor, n=n, axis=axis, normalization=norm, 
                           input_is_complex=input_is_complex, buffer_size_mb=buffer_size_mb)
        else:
            return ops.irfft(input_tensor, axis=axis, normalization=norm,
                           input_is_complex=input_is_complex, buffer_size_mb=buffer_size_mb)


def nonzero(ctx: MaxGraph, x: TensorValue, out_dim) -> TensorValue:
    """Get indices of non-zero elements.
    
    Args:
        x: Input tensor
        out_dim: The newly generated dimension that is sized for the number of nonzero elements
    
    Returns:
        Tensor of indices
    """
    with ctx.graph:
        return ops.nonzero(x, out_dim=out_dim)


def qmatmul(ctx: MaxGraph, encoding, config, lhs: TensorValue, *rhs) -> TensorValue:
    """Perform matrix multiplication between floating point and quantized tensors.
    
    Args:
        encoding: The quantization encoding to use
        config: The quantization config
        lhs: The non-quantized, left-hand-side of the matmul
        *rhs: The transposed and quantized right-hand-side of the matmul
    
    Returns:
        The dequantized result (floating point tensor)
    """
    with ctx.graph:
        return ops.qmatmul(encoding, config, lhs, *rhs)


def range_op(ctx: MaxGraph, start, stop, step=1, out_dim=None, dtype=None, device=None) -> TensorValue:
    """Create a sequence of numbers.
    
    Args:
        start: The start of the range
        stop: The range will be generated up to, but not including, this value
        step: The step size for the range
        out_dim: The expected output dimensions
        dtype: Data type of the result tensor
        device: Device of the result tensor
    
    Returns:
        Tensor containing the defined range of values
    """
    with ctx.graph:
        from max.graph import DeviceRef
        from max.dtype import DType as MAX_DType
        
        # Handle device
        if device is None:
            device_ref = DeviceRef.CPU()
        elif isinstance(device, str):
            parts = device.split(':')
            if parts[0].lower() == 'cpu':
                device_ref = DeviceRef.CPU(id=int(parts[1]) if len(parts) > 1 else 0)
            elif parts[0].lower() == 'gpu':
                device_ref = DeviceRef.GPU(id=int(parts[1]) if len(parts) > 1 else 0)
            else:
                device_ref = device
        else:
            device_ref = device
        
        # Handle dtype
        if dtype is None:
            target_dtype = MAX_DType.float32
        elif isinstance(dtype, str):
            dtype_map = {
                'float32': MAX_DType.float32,
                'float64': MAX_DType.float64,
                'int32': MAX_DType.int32,
                'int64': MAX_DType.int64,
            }
            target_dtype = dtype_map.get(dtype, MAX_DType.float32)
        else:
            target_dtype = dtype
        
        if out_dim is not None:
            return ops.range(start, stop, step=step, out_dim=out_dim, dtype=target_dtype, device=device_ref)
        else:
            return ops.range(start, stop, step=step, dtype=target_dtype, device=device_ref)


def rebind(ctx: MaxGraph, x: TensorValue, shape, message: str = '', layout=None) -> TensorValue:
    """Rebind a symbolic tensor to a specified set of dimensions.
    
    Args:
        x: The input symbolic tensor to rebind
        shape: The symbolic shape to assert for x
        message: The message printed if the rebind fails at runtime
        layout: A layout of the weights used by some operations
    
    Returns:
        Tensor with the symbolic shape asserted
    """
    with ctx.graph:
        if layout is not None:
            return ops.rebind(x, shape, message=message, layout=layout)
        else:
            return ops.rebind(x, shape, message=message)


def transfer_to(ctx: MaxGraph, x: TensorValue, device) -> TensorValue:
    """Device-to-Device transfer operation.
    
    Args:
        x: The input tensor to transfer
        device: The device to transfer to
    
    Returns:
        Tensor transferred to specified device
    """
    with ctx.graph:
        from max.graph import DeviceRef
        if isinstance(device, str):
            parts = device.split(':')
            if parts[0].lower() == 'cpu':
                device_ref = DeviceRef.CPU(id=int(parts[1]) if len(parts) > 1 else 0)
            elif parts[0].lower() == 'gpu':
                device_ref = DeviceRef.GPU(id=int(parts[1]) if len(parts) > 1 else 0)
            else:
                device_ref = device
        else:
            device_ref = device
        return ops.transfer_to(x, device_ref)


def while_loop(ctx: MaxGraph, initial_values, predicate, body) -> list:
    """Execute a loop until the predicate evaluates to false.
    
    Args:
        initial_values: Initial values for loop arguments
        predicate: Callable that takes loop arguments and returns a boolean scalar tensor
        body: Callable that takes loop arguments and returns updated values
    
    Returns:
        List of output values from the final loop iteration
    """
    with ctx.graph:
        return ops.while_loop(initial_values, predicate, body)

