# ===----------------------------------------------------------------------===
# Nabla 2025
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------===

"""Array creation and initialization operations."""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Device, Tensor
from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops

from ..core.array import Array, Shape
from .operation import Operation
from .view import broadcast_batch_dims, broadcast_to

# Public API
all = [
    "array",
    "arange",
    "ndarange",
    "ndarange_like",
    "randn",
    "randn_like",
    "rand",
    "rand_like",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    "full_like",
    "xavier_uniform",
    "xavier_normal",
    "he_uniform",
    "he_normal",
    "lecun_uniform",
    "lecun_normal",
    "glorot_uniform",
    "triu",
]

# Constants
_DEFAULT_CPU = CPU()
_DEFAULT_SEED = 0
_DEFAULT_DTYPE = DType.float32


def _validate_shape(shape: Shape) -> None:
    """Validate shape parameter."""
    if not isinstance(shape, tuple):
        raise TypeError(f"Shape must be a tuple, got {type(shape)}")


def _validate_numeric(value: float | int, name: str) -> None:
    """Validate numeric parameter."""
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric, got {type(value)}")


def _create_filled_array(
    shape: Shape,
    fill_value: float,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Create array filled with constant value using broadcasting."""
    _validate_shape(shape)
    _validate_shape(batch_dims)
    # WORKAROUND: Handle scalar boolean tensors (MAX tensor bug)
    # Workaround for MAX boolean tensor bug: ANY boolean tensor creation fails in MAX
    # when creating the scalar seed value, so we need special handling for all boolean cases
    if dtype == DType.bool:
        # Create boolean array by starting with float and converting
        try:
            # Try creating (1,) boolean array first
            scalar_1d = Array.from_numpy(
                np.array([fill_value], dtype=DType.to_numpy(dtype))
            ).to(device)
            scalar_1d.traced = traced

            if not shape:
                # For scalar boolean, reshape (1,) to ()
                from .view import reshape

                array = reshape(scalar_1d, ())
            else:
                # For non-scalar boolean, broadcast (1,) to target shape
                array = broadcast_to(scalar_1d, shape)
        except Exception:
            # Fallback: create as float and convert to bool
            scalar_float = Array.from_numpy(
                np.array([fill_value], dtype=np.float32)
            ).to(device)
            scalar_float.traced = traced

            if not shape:
                # Convert scalar float to scalar bool
                array = scalar_float.astype(dtype)
            else:
                # Broadcast float to shape, then convert to bool
                float_array = broadcast_to(scalar_float, shape)
                array = float_array.astype(dtype)
    else:
        # Original implementation for non-boolean types
        scalar = Array.from_numpy(np.array(fill_value, dtype=DType.to_numpy(dtype))).to(
            device
        )
        scalar.traced = traced

        if not shape:
            array = scalar
        else:
            array = broadcast_to(scalar, shape)

    if batch_dims:
        array = broadcast_batch_dims(array, batch_dims)

    return array


class RandomOp(Operation):
    """Base class for random number generators."""

    def __init__(
        self, shape: Shape, dtype: DType, device: Device, seed: int, op_name: str
    ):
        super().__init__(f"rng_{op_name}[shape={shape}]")
        self.shape = shape
        self.dtype = dtype
        self.logical_device = device
        self.seed = seed

        # Validate common parameters
        _validate_shape(shape)
        if not isinstance(seed, int):
            raise TypeError(f"Seed must be int, got {type(seed)}")

    def forward(self, *args: Array) -> Array:
        """Forward pass for creation operations."""
        if args:
            raise ValueError(
                f"Creation operation requires 0 arguments, got {len(args)}"
            )

        res = Array(
            shape=self.shape,
            dtype=self.dtype,
            device=self.logical_device,
            materialize=False,
            name=self.name,
        )

        res.set_maxpr(self.maxpr)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule
        res.custom_kernel_path = self.custom_kernel_path()

        if not res.stage_realization:
            self.eagerxpr([], res)

        res.creator_op = self
        return res

    def compute_output_shape(self, *input_shapes) -> tuple:
        return self.shape

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        raise NotImplementedError("VJP for random creation operations is not defined.")

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        raise NotImplementedError("JVP for random creation operations is not defined.")


class RandNOp(RandomOp):
    """Normal distribution random number generator."""

    def __init__(
        self,
        shape: Shape,
        dtype: DType = _DEFAULT_DTYPE,
        mean: float = 0.0,
        std: float = 1.0,
        device: Device = _DEFAULT_CPU,
        seed: int = _DEFAULT_SEED,
    ):
        super().__init__(shape, dtype, device, seed, "normal")
        self.mean = mean
        self.std = std

        _validate_numeric(mean, "Mean")
        _validate_numeric(std, "Std")
        if std <= 0:
            raise ValueError(f"Std must be positive, got {std}")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        ops.random.set_seed(self.seed)
        output.tensor_value = ops.random.normal(
            TensorType(
                output.dtype, output.shape, DeviceRef.from_device(output.logical_device)
            ),
            mean=self.mean,
            std=self.std,
        )

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np.random.seed(self.seed)
        np_result = np.random.normal(
            loc=self.mean, scale=self.std, size=output.shape
        ).astype(DType.to_numpy(output.dtype))
        output.impl_(Tensor.from_numpy(np_result).to(output.logical_device))


class RandUniformOp(RandomOp):
    """Uniform distribution random number generator."""

    def __init__(
        self,
        shape: Shape,
        dtype: DType = _DEFAULT_DTYPE,
        lower: float = 0.0,
        upper: float = 1.0,
        device: Device = _DEFAULT_CPU,
        seed: int = _DEFAULT_SEED,
    ):
        super().__init__(shape, dtype, device, seed, "uniform")
        self.lower = lower
        self.upper = upper

        _validate_numeric(lower, "Lower bound")
        _validate_numeric(upper, "Upper bound")
        if upper <= lower:
            raise ValueError(
                f"Upper bound must be greater than lower bound, got {lower} and {upper}"
            )

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        ops.random.set_seed(self.seed)
        output.tensor_value = ops.random.uniform(
            TensorType(
                output.dtype, output.shape, DeviceRef.from_device(output.logical_device)
            ),
            range=(self.lower, self.upper),
        )

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np.random.seed(self.seed)
        np_result = np.random.uniform(
            low=self.lower, high=self.upper, size=output.shape
        ).astype(DType.to_numpy(output.dtype))
        output.impl_(Tensor.from_numpy(np_result).to(output.logical_device))


def array(
    data: list | np.ndarray | float | int,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Creates an array from a Python list, NumPy array, or scalar.

    This function is the primary way to create a Nabla array from existing
    data. It converts the input data into a Nabla array on the specified
    device and with the given data type.

    Parameters
    ----------
    data : list | np.ndarray | float | int
        The input data to convert to an array.
    dtype : DType, optional
        The desired data type for the array. Defaults to DType.float32.
    device : Device, optional
        The computational device where the array will be stored. Defaults
        to the CPU.
    batch_dims : Shape, optional
        Specifies leading dimensions to be treated as batch dimensions.
        Defaults to an empty tuple.
    traced : bool, optional
        Whether the operation should be traced in the graph. Defaults to False.

    Returns
    -------
    Array
        A new Nabla array containing the provided data.

    Examples
    --------
    >>> import nabla as nb
    >>> import numpy as np
    >>> # Create from a Python list
    >>> nb.array([1, 2, 3])
    Array([1, 2, 3], dtype=int32)
    <BLANKLINE>
    >>> # Create from a NumPy array
    >>> np_arr = np.array([[4.0, 5.0], [6.0, 7.0]])
    >>> nb.array(np_arr)
    Array([[4., 5.],
           [6., 7.]], dtype=float32)
    <BLANKLINE>
    >>> # Create a scalar array
    >>> nb.array(100, dtype=nb.DType.int64)
    Array(100, dtype=int64)
    """
    if isinstance(data, list):
        np_data = np.array(data, dtype=DType.to_numpy(dtype))
    elif isinstance(data, np.ndarray):
        np_data = data.astype(DType.to_numpy(dtype))
    elif isinstance(data, int | float):
        # Handle scalar values
        np_data = np.array(data, dtype=DType.to_numpy(dtype))
    elif isinstance(data, (np.bool_, bool)):
        # Handle numpy boolean and Python boolean scalars
        np_data = np.array(data, dtype=DType.to_numpy(dtype))
    else:
        raise TypeError(
            f"Data must be a list, numpy array, or scalar, got {type(data)}"
        )
    # Special handling for boolean scalar tensors (MAX bug workaround)
    if np_data.shape == () and dtype == DType.bool:
        # For scalar boolean, create as float and convert
        float_array = Array.from_numpy(np_data.astype(np.float32)).to(device)
        float_array.traced = traced
        arr = float_array.astype(DType.bool)

    else:
        arr = Array.from_numpy(np_data).to(device)
        arr.traced = traced

    return broadcast_batch_dims(arr, batch_dims) if batch_dims else arr


class ArangeOp(Operation):
    """Operation to create a 1D array with evenly spaced values."""

    def __init__(
        self,
        start: float | int,
        stop: float | int,
        step: float | int,
        dtype: DType,
        device: Device,
    ):
        super().__init__(f"arange[start={start},stop={stop},step={step}]")
        self.start = start
        self.stop = stop
        self.step = step
        self.dtype = dtype
        self.logical_device = device

        # Pre-compute the output shape using numpy's robust implementation
        # This handles all edge cases like float steps, negative steps, etc.
        self._np_arange_for_shape = np.arange(
            start, stop, step, dtype=DType.to_numpy(dtype)
        )
        self.shape = self._np_arange_for_shape.shape

    def forward(self, *args: Array) -> Array:
        """Forward pass for the arange creation operation."""
        if args:
            raise ValueError(
                f"Creation operation 'arange' requires 0 arguments, got {len(args)}"
            )

        res = Array(
            shape=self.shape,
            dtype=self.dtype,
            device=self.logical_device,
            materialize=False,
            name=self.name,
        )

        res.set_maxpr(self.maxpr)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule

        if not res.stage_realization:
            self.eagerxpr([], res)

        res.creator_op = self
        return res

    def compute_output_shape(self, *input_shapes) -> tuple:
        return self.shape

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        """Graph-mode execution using max.ops.arange."""
        # This assumes an equivalent ops.arange exists in the MAX graph library.
        # This is a common and expected operation for a backend.
        output.tensor_value = ops.range(
            start=self.start,
            stop=self.stop,
            step=self.step,
            dtype=output.dtype,
            device=DeviceRef.from_device(output.logical_device),
        )

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        """Eager-mode execution using numpy."""
        # We can reuse the numpy array we created for the shape calculation
        output.impl_(
            Tensor.from_numpy(self._np_arange_for_shape).to(output.logical_device)
        )

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        # The arange operation does not depend on any Array inputs,
        # so its gradient is not defined in this context.
        raise NotImplementedError("VJP for 'arange' creation operation is not defined.")

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        # The arange operation does not depend on any Array inputs,
        # so its gradient is not defined in this context.
        raise NotImplementedError("JVP for 'arange' creation operation is not defined.")


def arange(
    start: int | float,
    stop: int | float | None = None,
    step: int | float | None = None,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    traced: bool = False,
    batch_dims: Shape = (),
) -> Array:
    """Returns evenly spaced values within a given interval.

    Values are generated within the half-open interval `[start, stop)`.
    In other words, the interval includes `start` but excludes `stop`.
    This function follows the JAX/NumPy `arange` API.

    Parameters
    ----------
    start : int | float
        Start of interval. If `stop` is None, `start` is treated as `stop`
        and the starting value is 0.
    stop : int | float, optional
        End of interval. The interval does not include this value.
        Defaults to None.
    step : int | float, optional
        Spacing between values. The default step size is 1.
    dtype : DType, optional
        The data type of the output array. Defaults to DType.float32.
    device : Device, optional
        The device to place the array on. Defaults to the CPU.
    traced : bool, optional
        Whether the operation should be traced in the graph. Defaults to False.
    batch_dims : Shape, optional
        Specifies leading dimensions to be treated as batch dimensions.
        Defaults to an empty tuple.

    Returns
    -------
    Array
        A 1D array of evenly spaced values.

    Examples
    --------
    >>> import nabla as nb
    >>> # nb.arange(stop)
    >>> nb.arange(5)
    Array([0., 1., 2., 3., 4.], dtype=float32)
    <BLANKLINE>
    >>> # nb.arange(start, stop)
    >>> nb.arange(5, 10)
    Array([5., 6., 7., 8., 9.], dtype=float32)
    <BLANKLINE>
    >>> # nb.arange(start, stop, step)
    >>> nb.arange(10, 20, 2, dtype=nb.DType.int32)
    Array([10, 12, 14, 16, 18], dtype=int32)
    """
    # Handle the case where only one positional argument is provided, e.g., arange(5)
    if stop is None:
        stop = start
        start = 0

    if step is None:
        step = 1

    _validate_numeric(start, "start")
    _validate_numeric(stop, "stop")
    _validate_numeric(step, "step")

    if step == 0:
        raise ValueError("arange: step cannot be zero.")

    op = ArangeOp(start=start, stop=stop, step=step, dtype=dtype, device=device)
    arr = op.forward()
    arr.traced = traced
    if batch_dims:
        arr = broadcast_batch_dims(arr, batch_dims)
    return arr


def ndarange(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Creates an array of a given shape with sequential values.

    The array is filled with values from 0 to N-1, where N is the total
    number of elements (the product of the shape dimensions).

    Parameters
    ----------
    shape : Shape
        The shape of the output array.
    dtype : DType, optional
        The desired data type for the array. Defaults to DType.float32.
    device : Device, optional
        The device to place the array on. Defaults to the CPU.
    batch_dims : Shape, optional
        Specifies leading dimensions to be treated as batch dimensions.
        Defaults to an empty tuple.
    traced : bool, optional
        Whether the operation should be traced in the graph. Defaults to False.

    Returns
    -------
    Array
        An array of the specified shape containing values from 0 to N-1.

    Examples
    --------
    >>> import nabla as nb
    >>> nb.ndarange((2, 3), dtype=nb.DType.int32)
    Array([[0, 1, 2],
           [3, 4, 5]], dtype=int32)
    """
    return arange(
        0, int(np.prod(shape)), 1, dtype=dtype, device=device, traced=traced
    ).reshape(shape)


def ndarange_like(template: Array) -> Array:
    """Creates an array with sequential values like a template array.

    The new array will have the same shape, dtype, device, and batch
    dimensions as the template array. It is filled with values from 0 to
    N-1, where N is the total number of elements.

    Parameters
    ----------
    template : Array
        The template array to match properties from.

    Returns
    -------
    Array
        A new array with the same properties as the template, filled with
        sequential values.

    Examples
    --------
    >>> import nabla as nb
    >>> template = nb.zeros((2, 2), dtype=nb.DType.int32)
    >>> nb.ndarange_like(template)
    Array([[0, 1],
           [2, 3]], dtype=int32)
    """
    return ndarange(
        template.shape,
        template.dtype,
        template.logical_device,
        template.batch_dims,
        template.traced,
    )


def randn(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    mean: float = 0.0,
    std: float = 1.0,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Creates an array with normally distributed random values.

    The values are drawn from a normal (Gaussian) distribution with the
    specified mean and standard deviation.

    Parameters
    ----------
    shape : Shape
        The shape of the output array.
    dtype : DType, optional
        The desired data type for the array. Defaults to DType.float32.
    mean : float, optional
        The mean of the normal distribution. Defaults to 0.0.
    std : float, optional
        The standard deviation of the normal distribution. Defaults to 1.0.
    device : Device, optional
        The device to place the array on. Defaults to the CPU.
    seed : int, optional
        The seed for the random number generator for reproducibility.
        Defaults to 0.
    batch_dims : Shape, optional
        Specifies leading dimensions to be treated as batch dimensions.
        Defaults to an empty tuple.
    traced : bool, optional
        Whether the operation should be traced in the graph. Defaults to False.

    Returns
    -------
    Array
        An array of the specified shape filled with random values.
    """
    arr = RandNOp(shape, dtype, mean, std, device, seed).forward()
    arr.traced = traced
    return broadcast_batch_dims(arr, batch_dims) if batch_dims else arr


def randn_like(
    template: Array, mean: float = 0.0, std: float = 1.0, seed: int = _DEFAULT_SEED
) -> Array:
    """Creates an array with normally distributed random values like a template.

    The new array will have the same shape, dtype, device, and batch
    dimensions as the template array.

    Parameters
    ----------
    template : Array
        The template array to match properties from.
    mean : float, optional
        The mean of the normal distribution. Defaults to 0.0.
    std : float, optional
        The standard deviation of the normal distribution. Defaults to 1.0.
    seed : int, optional
        The seed for the random number generator. Defaults to 0.

    Returns
    -------
    Array
        A new array with the same properties as the template, filled with
        normally distributed random values.
    """
    res = randn(
        template.shape,
        template.dtype,
        mean,
        std,
        template.logical_device,
        seed,
        template.batch_dims,
        traced=template.traced,
    )
    return res


def rand(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    lower: float = 0.0,
    upper: float = 1.0,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Creates an array with uniformly distributed random values.

    The values are drawn from a continuous uniform distribution over the
    interval `[lower, upper)`.

    Parameters
    ----------
    shape : Shape
        The shape of the output array.
    dtype : DType, optional
        The desired data type for the array. Defaults to DType.float32.
    lower : float, optional
        The lower boundary of the output interval. Defaults to 0.0.
    upper : float, optional
        The upper boundary of the output interval. Defaults to 1.0.
    device : Device, optional
        The device to place the array on. Defaults to the CPU.
    seed : int, optional
        The seed for the random number generator. Defaults to 0.
    batch_dims : Shape, optional
        Specifies leading dimensions to be treated as batch dimensions.
        Defaults to an empty tuple.
    traced : bool, optional
        Whether the operation should be traced in the graph. Defaults to False.

    Returns
    -------
    Array
        An array of the specified shape filled with random values.
    """
    arr = RandUniformOp(shape, dtype, lower, upper, device, seed).forward()
    arr.traced = traced
    return broadcast_batch_dims(arr, batch_dims) if batch_dims else arr


def rand_like(
    template: Array, lower: float = 0.0, upper: float = 1.0, seed: int = _DEFAULT_SEED
) -> Array:
    """Creates an array with uniformly distributed random values like a template.

    The new array will have the same shape, dtype, device, and batch
    dimensions as the template array.

    Parameters
    ----------
    template : Array
        The template array to match properties from.
    lower : float, optional
        The lower boundary of the output interval. Defaults to 0.0.
    upper : float, optional
        The upper boundary of the output interval. Defaults to 1.0.
    seed : int, optional
        The seed for the random number generator. Defaults to 0.

    Returns
    -------
    Array
        A new array with the same properties as the template, filled with
        uniformly distributed random values.
    """
    res = rand(
        template.shape,
        template.dtype,
        lower,
        upper,
        template.logical_device,
        seed,
        template.batch_dims,
        traced=template.traced,
    )
    return res


def zeros(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Creates an array of a given shape filled with zeros.

    Parameters
    ----------
    shape : Shape
        The shape of the new array, e.g., `(2, 3)` or `(5,)`.
    dtype : DType, optional
        The desired data type for the array. Defaults to DType.float32.
    device : Device, optional
        The device to place the array on. Defaults to the CPU.
    batch_dims : Shape, optional
        Specifies leading dimensions to be treated as batch dimensions.
        Defaults to an empty tuple.
    traced : bool, optional
        Whether the operation should be traced in the graph. Defaults to False.

    Returns
    -------
    Array
        An array of the specified shape and dtype, filled with zeros.

    Examples
    --------
    >>> import nabla as nb
    >>> # Create a 2x3 matrix of zeros
    >>> nb.zeros((2, 3), dtype=nb.DType.int32)
    Array([[0, 0, 0],
           [0, 0, 0]], dtype=int32)
    """
    return _create_filled_array(shape, 0.0, dtype, device, batch_dims, traced=traced)


def ones(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Creates an array of a given shape filled with ones.

    Parameters
    ----------
    shape : Shape
        The shape of the new array, e.g., `(2, 3)` or `(5,)`.
    dtype : DType, optional
        The desired data type for the array. Defaults to DType.float32.
    device : Device, optional
        The device to place the array on. Defaults to the CPU.
    batch_dims : Shape, optional
        Specifies leading dimensions to be treated as batch dimensions.
        Defaults to an empty tuple.
    traced : bool, optional
        Whether the operation should be traced in the graph. Defaults to False.

    Returns
    -------
    Array
        An array of the specified shape and dtype, filled with ones.

    Examples
    --------
    >>> import nabla as nb
    >>> # Create a vector of ones
    >>> nb.ones((4,), dtype=nb.DType.float32)
    Array([1., 1., 1., 1.], dtype=float32)
    """
    return _create_filled_array(shape, 1.0, dtype, device, batch_dims, traced=traced)


def zeros_like(template: Array) -> Array:
    """Creates an array of zeros with the same properties as a template array.

    The new array will have the same shape, dtype, device, and batch
    dimensions as the template array.

    Parameters
    ----------
    template : Array
        The template array to match properties from.

    Returns
    -------
    Array
        A new array of zeros with the same properties as the template.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.array([[1, 2], [3, 4]], dtype=nb.DType.int32)
    >>> nb.zeros_like(x)
    Array([[0, 0],
           [0, 0]], dtype=int32)
    """
    return zeros(
        template.shape,
        template.dtype,
        template.logical_device,
        template.batch_dims,
        traced=template.traced,
    )


def ones_like(template: Array) -> Array:
    """Creates an array of ones with the same properties as a template array.

    The new array will have the same shape, dtype, device, and batch
    dimensions as the template array.

    Parameters
    ----------
    template : Array
        The template array to match properties from.

    Returns
    -------
    Array
        A new array of ones with the same properties as the template.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.array([[1., 2.], [3., 4.]])
    >>> nb.ones_like(x)
    Array([[1., 1.],
           [1., 1.]], dtype=float32)
    """
    return ones(
        template.shape,
        template.dtype,
        template.logical_device,
        template.batch_dims,
        traced=template.traced,
    )


def full_like(template: Array, fill_value: float) -> Array:
    """Creates a filled array with the same properties as a template array.

    The new array will have the same shape, dtype, device, and batch
    dimensions as the template array, filled with `fill_value`.

    Parameters
    ----------
    template : Array
        The template array to match properties from.
    fill_value : float
        The value to fill the new array with.

    Returns
    -------
    Array
        A new array filled with `fill_value` and with the same properties
        as the template.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.zeros((2, 2))
    >>> nb.full_like(x, 7.0)
    Array([[7., 7.],
           [7., 7.]], dtype=float32)
    """
    return _create_filled_array(
        template.shape,
        fill_value,
        template.dtype,
        template.logical_device,
        template.batch_dims,
        template.traced,
    )


# Neural Network Initialization Methods


def xavier_uniform(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    gain: float = 1.0,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Fills an array with values according to the Xavier uniform initializer.

    Also known as Glorot uniform initialization, this method is designed to
    keep the variance of activations the same across every layer in a network.
    It samples from a uniform distribution U(-a, a) where
    a = gain * sqrt(6 / (fan_in + fan_out)).

    Parameters
    ----------
    shape : Shape
        The shape of the output array. Must be at least 2D.
    dtype : DType, optional
        The desired data type for the array. Defaults to DType.float32.
    gain : float, optional
        An optional scaling factor. Defaults to 1.0.
    device : Device, optional
        The device to place the array on. Defaults to the CPU.
    seed : int, optional
        The seed for the random number generator. Defaults to 0.
    batch_dims : Shape, optional
        Specifies leading dimensions to be treated as batch dimensions.
        Defaults to an empty tuple.
    traced : bool, optional
        Whether the operation should be traced in the graph. Defaults to False.

    Returns
    -------
    Array
        An array initialized with the Xavier uniform distribution.
    """
    _validate_shape(shape)
    if len(shape) < 2:
        raise ValueError(
            f"Xavier initialization requires at least 2D shape, got {shape}"
        )

    fan_in, fan_out = shape[-2], shape[-1]
    std = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rand(shape, dtype, -std, std, device, seed, batch_dims, traced=traced)


def xavier_normal(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    gain: float = 1.0,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Fills an array with values according to the Xavier normal initializer.

    Also known as Glorot normal initialization. It samples from a normal
    distribution N(0, std^2) where std = gain * sqrt(2 / (fan_in + fan_out)).

    Parameters
    ----------
    shape : Shape
        The shape of the output array. Must be at least 2D.
    dtype : DType, optional
        The desired data type for the array. Defaults to DType.float32.
    gain : float, optional
        An optional scaling factor. Defaults to 1.0.
    device : Device, optional
        The device to place the array on. Defaults to the CPU.
    seed : int, optional
        The seed for the random number generator. Defaults to 0.
    batch_dims : Shape, optional
        Specifies leading dimensions to be treated as batch dimensions.
        Defaults to an empty tuple.
    traced : bool, optional
        Whether the operation should be traced in the graph. Defaults to False.

    Returns
    -------
    Array
        An array initialized with the Xavier normal distribution.
    """
    _validate_shape(shape)
    if len(shape) < 2:
        raise ValueError(
            f"Xavier initialization requires at least 2D shape, got {shape}"
        )

    fan_in, fan_out = shape[-2], shape[-1]
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return randn(shape, dtype, 0.0, std, device, seed, batch_dims, traced=traced)


def he_uniform(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Fills an array with values according to the He uniform initializer.

    This method is designed for layers with ReLU activations. It samples from
    a uniform distribution U(-a, a) where a = sqrt(6 / fan_in).

    Parameters
    ----------
    shape : Shape
        The shape of the output array. Must be at least 2D.
    dtype : DType, optional
        The desired data type for the array. Defaults to DType.float32.
    device : Device, optional
        The device to place the array on. Defaults to the CPU.
    seed : int, optional
        The seed for the random number generator. Defaults to 0.
    batch_dims : Shape, optional
        Specifies leading dimensions to be treated as batch dimensions.
        Defaults to an empty tuple.
    traced : bool, optional
        Whether the operation should be traced in the graph. Defaults to False.

    Returns
    -------
    Array
        An array initialized with the He uniform distribution.
    """
    _validate_shape(shape)
    if len(shape) < 2:
        raise ValueError(f"He initialization requires at least 2D shape, got {shape}")

    fan_in = shape[-2]
    bound = np.sqrt(6.0 / fan_in)
    return rand(shape, dtype, -bound, bound, device, seed, batch_dims, traced=traced)


def he_normal(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Fills an array with values according to the He normal initializer.

    This method is designed for layers with ReLU activations. It samples from
    a normal distribution N(0, std^2) where std = sqrt(2 / fan_in).

    Parameters
    ----------
    shape : Shape
        The shape of the output array. Must be at least 2D.
    dtype : DType, optional
        The desired data type for the array. Defaults to DType.float32.
    device : Device, optional
        The device to place the array on. Defaults to the CPU.
    seed : int, optional
        The seed for the random number generator. Defaults to 0.
    batch_dims : Shape, optional
        Specifies leading dimensions to be treated as batch dimensions.
        Defaults to an empty tuple.
    traced : bool, optional
        Whether the operation should be traced in the graph. Defaults to False.

    Returns
    -------
    Array
        An array initialized with the He normal distribution.
    """
    _validate_shape(shape)
    if len(shape) < 2:
        raise ValueError(f"He initialization requires at least 2D shape, got {shape}")

    fan_in = shape[-2]
    std = np.sqrt(2.0 / fan_in)
    return randn(shape, dtype, 0.0, std, device, seed, batch_dims, traced=traced)


def lecun_uniform(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Fills an array with values according to the LeCun uniform initializer.

    This method is often used for layers with SELU activations. It samples from
    a uniform distribution U(-a, a) where a = sqrt(3 / fan_in).

    Parameters
    ----------
    shape : Shape
        The shape of the output array. Must be at least 2D.
    dtype : DType, optional
        The desired data type for the array. Defaults to DType.float32.
    device : Device, optional
        The device to place the array on. Defaults to the CPU.
    seed : int, optional
        The seed for the random number generator. Defaults to 0.
    batch_dims : Shape, optional
        Specifies leading dimensions to be treated as batch dimensions.
        Defaults to an empty tuple.
    traced : bool, optional
        Whether the operation should be traced in the graph. Defaults to False.

    Returns
    -------
    Array
        An array initialized with the LeCun uniform distribution.
    """
    _validate_shape(shape)
    if len(shape) < 2:
        raise ValueError(
            f"LeCun initialization requires at least 2D shape, got {shape}"
        )

    fan_in = shape[-2]
    bound = np.sqrt(3.0 / fan_in)
    return rand(shape, dtype, -bound, bound, device, seed, batch_dims, traced=traced)


def lecun_normal(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Fills an array with values according to the LeCun normal initializer.

    This method is often used for layers with SELU activations. It samples from
    a normal distribution N(0, std^2) where std = sqrt(1 / fan_in).

    Parameters
    ----------
    shape : Shape
        The shape of the output array. Must be at least 2D.
    dtype : DType, optional
        The desired data type for the array. Defaults to DType.float32.
    device : Device, optional
        The device to place the array on. Defaults to the CPU.
    seed : int, optional
        The seed for the random number generator. Defaults to 0.
    batch_dims : Shape, optional
        Specifies leading dimensions to be treated as batch dimensions.
        Defaults to an empty tuple.
    traced : bool, optional
        Whether the operation should be traced in the graph. Defaults to False.

    Returns
    -------
    Array
        An array initialized with the LeCun normal distribution.
    """
    _validate_shape(shape)
    if len(shape) < 2:
        raise ValueError(
            f"LeCun initialization requires at least 2D shape, got {shape}"
        )

    fan_in = shape[-2]
    std = np.sqrt(1.0 / fan_in)
    return randn(shape, dtype, 0.0, std, device, seed, batch_dims, traced=traced)


def glorot_uniform(
    shape: Shape,
    dtype: DType = _DEFAULT_DTYPE,
    gain: float = 1.0,
    device: Device = _DEFAULT_CPU,
    seed: int = _DEFAULT_SEED,
    batch_dims: Shape = (),
    traced: bool = False,
) -> Array:
    """Fills an array with values according to the Glorot uniform initializer.

    This is an alias for `xavier_uniform`. It samples from a uniform
    distribution U(-a, a) where a = sqrt(6 / (fan_in + fan_out)).

    Parameters
    ----------
    shape : Shape
        The shape of the output array. Must be at least 2D.
    dtype : DType, optional
        The desired data type for the array. Defaults to DType.float32.
    gain : float, optional
        An optional scaling factor. Defaults to 1.0.
    device : Device, optional
        The device to place the array on. Defaults to the CPU.
    seed : int, optional
        The seed for the random number generator. Defaults to 0.
    batch_dims : Shape, optional
        Specifies leading dimensions to be treated as batch dimensions.
        Defaults to an empty tuple.
    traced : bool, optional
        Whether the operation should be traced in the graph. Defaults to False.

    Returns
    -------
    Array
        An array initialized with the Glorot uniform distribution.
    """
    _validate_shape(shape)
    if len(shape) < 2:
        raise ValueError(
            f"Glorot initialization requires at least 2D shape, got {shape}"
        )

    fan_in, fan_out = shape[-2], shape[-1]
    bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rand(shape, dtype, -bound, bound, device, seed, batch_dims, traced=traced)


def triu(x: Array, k: int = 0) -> Array:
    """Returns the upper triangular part of a matrix or batch of matrices.

    The elements below the k-th diagonal are zeroed out. The input is
    expected to be at least 2-dimensional.

    Parameters
    ----------
    x : Array
        Input array with shape (..., M, N).
    k : int, optional
        Diagonal offset. `k = 0` is the main diagonal. `k > 0` is above the
        main diagonal, and `k < 0` is below the main diagonal. Defaults to 0.

    Returns
    -------
    Array
        An array with the lower triangular part zeroed out, with the same
        shape and dtype as `x`.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.ndarange((3, 3), dtype=nb.DType.int32)
    >>> x
    Array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]], dtype=int32)
    <BLANKLINE>
    >>> # Upper triangle with the main diagonal
    >>> nb.triu(x, k=0)
    Array([[0, 1, 2],
           [0, 4, 5],
           [0, 0, 8]], dtype=int32)
    <BLANKLINE>
    >>> # Upper triangle above the main diagonal
    >>> nb.triu(x, k=1)
    Array([[0, 1, 2],
           [0, 0, 5],
           [0, 0, 0]], dtype=int32)
    """
    from .special import where

    mask = ndarange((x.shape[-1],)) < ndarange((x.shape[-1],))[:, None] + k
    return where(mask, x, array(0, dtype=x.dtype, device=x.logical_device))