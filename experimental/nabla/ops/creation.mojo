from nabla.core.tensor import Tensor
from nabla.utils.max_bindings import (
    MaxTensor,
    MaxDType,
    _list_to_py_tuple,
    _list_to_py_list,
)
from nabla.utils.debug import err_loc
from python import Python


@always_inline
fn full(
    value: Float32, shape: List[Int], dtype: DType = DType.float32
) raises -> Tensor:
    """Create a tensor full of a scalar value."""
    try:
        var tensor = Tensor(shape, dtype, stage_realization=True)
        var np_full = Python.import_module("numpy").full(
            _list_to_py_tuple(shape),
            value,
            dtype=MaxDType.from_dtype(dtype).to_numpy_dtype(),
        )
        tensor.set_data(MaxTensor.from_numpy(np_full))
        tensor.set_stage_realization(False)
        return tensor
    except err:
        raise Error("\nError in full operation: " + String(err) + err_loc())


@always_inline
fn zeros(shape: List[Int], dtype: DType = DType.float32) raises -> Tensor:
    """Create a tensor full of zeros using the full operation."""
    return full(0.0, shape, dtype)


@always_inline
fn ones(shape: List[Int], dtype: DType = DType.float32) raises -> Tensor:
    """Create a tensor full of ones using the full operation."""
    return full(1.0, shape, dtype)


@always_inline
fn arange(
    start: Int, stop: Int, step: Int = 1, dtype: DType = DType.float32
) raises -> Tensor:
    try:
        var np_arange = Python.import_module("numpy").arange(
            start, stop, step, dtype=MaxDType.from_dtype(dtype).to_numpy_dtype()
        )
        var shape = [Int(np_arange.shape[0])]
        var tensor = Tensor(shape, dtype, stage_realization=True)
        tensor.set_data(MaxTensor.from_numpy(np_arange))
        tensor.set_stage_realization(False)
        return tensor
    except err:
        raise Error("\nError in arange operation: " + String(err) + err_loc())


@always_inline
fn ndarange(shape: List[Int], dtype: DType = DType.float32) raises -> Tensor:
    try:
        var end = 1
        for dim in shape:
            end *= dim
        var np_shape = _list_to_py_list(shape)
        var np_arange = (
            Python.import_module("numpy")
            .arange(0, end, 1, dtype=MaxDType.from_dtype(dtype).to_numpy_dtype())
            .reshape(np_shape)
        )
        var tensor = Tensor(shape, dtype, stage_realization=True)
        tensor.set_data(MaxTensor.from_numpy(np_arange))
        tensor.set_stage_realization(False)
        return tensor
    except err:
        raise Error("\nError in ndarange operation: " + String(err) + err_loc())


@always_inline
fn randn(
    shape: List[Int],
    mean: Float32 = 0.0,
    std: Float32 = 1.0,
    dtype: DType = DType.float32,
) raises -> Tensor:
    """Create a tensor with random values from a normal distribution.

    Args:
        shape: The shape of the tensor.
        mean: Mean of the normal distribution (default: 0.0).
        std: Standard deviation of the normal distribution (default: 1.0).
        dtype: Data type of the tensor (default: DType.float32).
    """
    try:
        var np_randn = (
            Python.import_module("numpy")
            .random.normal(mean, std, _list_to_py_tuple(shape))
            .astype(MaxDType.from_dtype(dtype).to_numpy_dtype())
        )
        var tensor = Tensor(shape, dtype, stage_realization=True)
        tensor.set_data(MaxTensor.from_numpy(np_randn))
        tensor.set_stage_realization(False)
        return tensor
    except err:
        raise Error("\nError in randn operation: " + String(err) + err_loc())


@always_inline
fn randu(
    shape: List[Int],
    low: Float32 = 0.0,
    high: Float32 = 1.0,
    dtype: DType = DType.float32,
) raises -> Tensor:
    """Create a tensor with random values from a uniform distribution.

    Args:
        shape: The shape of the tensor.
        low: Lower bound of the uniform distribution (inclusive, default: 0.0).
        high: Upper bound of the uniform distribution (exclusive, default: 1.0).
        dtype: Data type of the tensor (default: DType.float32).
    """
    try:
        var np_randu = (
            Python.import_module("numpy")
            .random.uniform(low, high, _list_to_py_tuple(shape))
            .astype(MaxDType.from_dtype(dtype).to_numpy_dtype())
        )
        var tensor = Tensor(shape, dtype, stage_realization=True)
        tensor.set_data(MaxTensor.from_numpy(np_randu))
        tensor.set_stage_realization(False)
        return tensor
    except err:
        raise Error("\nError in randu operation: " + String(err) + err_loc())
