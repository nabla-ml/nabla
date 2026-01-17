# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Any

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, ShapeLike, TensorType, TensorValue, ops
from max.graph.ops.constant import NestedArray, Number

from .operation import Operation
from ..core import defaults


class ConstantOp(Operation):
    """Create a tensor from a constant value (number, array, nested list)."""
    
    @property
    def name(self) -> str:
        return "constant"
    
    def maxpr(
        self,
        value: NestedArray | Number,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        return ops.constant(value, dtype, device)


class FullOp(Operation):
    """Create a tensor filled with a constant value."""
    
    @property
    def name(self) -> str:
        return "full"
    
    def maxpr(
        self,
        shape: ShapeLike,
        value: Number,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        const = ops.constant(value, dtype, device)
        return ops.broadcast_to(const, shape)


class ZerosOp(Operation):
    """Create a tensor filled with zeros."""
    
    @property
    def name(self) -> str:
        return "zeros"
    
    def maxpr(
        self,
        shape: ShapeLike,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        const = ops.constant(0, dtype, device)
        return ops.broadcast_to(const, shape)


class OnesOp(Operation):
    """Create a tensor filled with ones."""
    
    @property
    def name(self) -> str:
        return "ones"
    
    def maxpr(
        self,
        shape: ShapeLike,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        const = ops.constant(1, dtype, device)
        return ops.broadcast_to(const, shape)


class ArangeOp(Operation):
    """Create a tensor with evenly spaced values."""
    
    @property
    def name(self) -> str:
        return "arange"
    
    def maxpr(
        self,
        start: int,
        stop: int,
        step: int,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        return ops.range(start, stop, step, dtype=dtype, device=device)


# =============================================================================
# Random creation ops
# =============================================================================

class UniformOp(Operation):
    """Create a tensor with uniform random values."""
    
    @property
    def name(self) -> str:
        return "uniform"
    
    def maxpr(
        self,
        shape: ShapeLike,
        low: float,
        high: float,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        tensor_type = TensorType(dtype, shape, device=DeviceRef.from_device(device))
        return ops.random.uniform(tensor_type, range=(low, high))


class GaussianOp(Operation):
    """Create a tensor with Gaussian (normal) random values."""
    
    @property
    def name(self) -> str:
        return "gaussian"
    
    def maxpr(
        self,
        shape: ShapeLike,
        mean: float,
        std: float,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        tensor_type = TensorType(dtype, shape, device=DeviceRef.from_device(device))
        return ops.random.gaussian(tensor_type, mean=mean, std=std)


# =============================================================================
# Singleton instances
# =============================================================================

_constant_op = ConstantOp()
_full_op = FullOp()
_zeros_op = ZerosOp()
_ones_op = OnesOp()
_arange_op = ArangeOp()
_uniform_op = UniformOp()
_gaussian_op = GaussianOp()


# =============================================================================
# Public API functions
# =============================================================================

def constant(
    value: NestedArray | Number,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
):
    """Create a tensor from a constant value.
    
    Args:
        value: A number, numpy array, or nested list/tuple of numbers.
        dtype: Data type (default: float32).
        device: Device to create tensor on (default: CPU).
        
    Returns:
        A Tensor containing the constant value.
    """
    dtype, device = defaults(dtype, device)
    return _constant_op(value, dtype, device)


def full(
    shape: ShapeLike,
    value: Number,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    traced: bool = False,
):
    """Create a tensor filled with a constant value.
    
    Args:
        shape: Shape of the tensor.
        value: Value to fill the tensor with.
        dtype: Data type (default: float32).
        device: Device to create tensor on (default: CPU).
        traced: Whether to enable tracing for autograd.
        
    Returns:
        A Tensor filled with the specified value.
    """
    dtype, device = defaults(dtype, device)
    t = _full_op(shape, value, dtype, device)
    if traced:
        t._impl.traced = True
    return t


def zeros(
    shape: ShapeLike,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    traced: bool = False,
):
    """Create a tensor filled with zeros.
    
    Args:
        shape: Shape of the tensor.
        dtype: Data type (default: float32).
        device: Device to create tensor on (default: CPU).
        traced: Whether to enable tracing for autograd.
        
    Returns:
        A Tensor filled with zeros.
    """
    dtype, device = defaults(dtype, device)
    t = _zeros_op(shape, dtype, device)
    if traced:
        t._impl.traced = True
    return t


def ones(
    shape: ShapeLike,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    traced: bool = False,
):
    """Create a tensor filled with ones.
    
    Args:
        shape: Shape of the tensor.
        dtype: Data type (default: float32).
        device: Device to create tensor on (default: CPU).
        traced: Whether to enable tracing for autograd.
        
    Returns:
        A Tensor filled with ones.
    """
    dtype, device = defaults(dtype, device)
    t = _ones_op(shape, dtype, device)
    if traced:
        t._impl.traced = True
    return t


def arange(
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
):
    """Create a tensor with evenly spaced values.
    
    Args:
        start: Start of interval (or stop if stop is None).
        stop: End of interval.
        step: Spacing between values.
        dtype: Data type (default: float32).
        device: Device to create tensor on (default: CPU).
        
    Returns:
        A 1-D Tensor with evenly spaced values.
    """
    dtype, device = defaults(dtype, device)
    if stop is None:
        start, stop = 0, start
    return _arange_op(start, stop, step, dtype, device)


def uniform(
    shape: ShapeLike = (),
    low: float = 0.0,
    high: float = 1.0,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
):
    """Create a tensor with uniform random values.
    
    Args:
        shape: Shape of the tensor.
        low: Lower bound of the uniform distribution.
        high: Upper bound of the uniform distribution.
        dtype: Data type (default: float32).
        device: Device to create tensor on (default: CPU).
        
    Returns:
        A Tensor with values uniformly distributed in [low, high).
    """
    dtype, device = defaults(dtype, device)
    return _uniform_op(shape, low, high, dtype, device)


def gaussian(
    shape: ShapeLike = (),
    mean: float = 0.0,
    std: float = 1.0,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
):
    """Create a tensor with Gaussian (normal) random values.
    
    Args:
        shape: Shape of the tensor.
        mean: Mean of the Gaussian distribution.
        std: Standard deviation of the Gaussian distribution.
        dtype: Data type (default: float32).
        device: Device to create tensor on (default: CPU).
        
    Returns:
        A Tensor with values from a Gaussian distribution.
    """
    dtype, device = defaults(dtype, device)
    return _gaussian_op(shape, mean, std, dtype, device)


# Alias for gaussian
normal = gaussian


__all__ = [
    # Op classes
    "ConstantOp",
    "FullOp",
    "ZerosOp", 
    "OnesOp",
    "ArangeOp",
    "UniformOp",
    "GaussianOp",
    # Public functions
    "constant",
    "full",
    "zeros",
    "ones",
    "arange",
    "uniform",
    "gaussian",
    "normal",
]
