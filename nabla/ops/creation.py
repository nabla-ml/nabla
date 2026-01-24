# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, ShapeLike, TensorType, TensorValue, ops
from max.graph.ops.constant import NestedArray, Number

from ..core import defaults
from .base import Operation


class ConstantOp(Operation):
    """Create a tensor from a constant value."""

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

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        # Constant has no differentiable inputs
        return (None, None, None)


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

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        return (None, None, None, None)


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

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        return (None, None, None)


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

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        return (None, None, None)


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

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        return (None, None, None, None, None)


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


_constant_op = ConstantOp()
_full_op = FullOp()
_zeros_op = ZerosOp()
_ones_op = OnesOp()
_arange_op = ArangeOp()
_uniform_op = UniformOp()
_gaussian_op = GaussianOp()


def constant(
    value: NestedArray | Number,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
):
    """Create a tensor from a constant value.

    Args:
        value: Scalar, array, or nested sequence.
    """
    if dtype is None:
        import numpy as np
        if isinstance(value, np.ndarray):
            try:
                dtype = DType[str(value.dtype)]
            except (KeyError, ValueError):
                pass
        elif isinstance(value, int):
            dtype = DType.int32
        elif isinstance(value, float):
            dtype = DType.float32
        elif isinstance(value, bool):
            dtype = DType.bool

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
    """Create a tensor filled with a constant value."""
    dtype, device = defaults(dtype, device)
    t = _full_op(shape, value, dtype, device)
    if traced:
        t.traced = True
    return t


def zeros(
    shape: ShapeLike,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    traced: bool = False,
):
    """Create a tensor filled with zeros."""
    dtype, device = defaults(dtype, device)
    t = _zeros_op(shape, dtype, device)
    if traced:
        t.traced = True
    return t


def ones(
    shape: ShapeLike,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    traced: bool = False,
):
    """Create a tensor filled with ones."""
    dtype, device = defaults(dtype, device)
    t = _ones_op(shape, dtype, device)
    if traced:
        t.traced = True
    return t


def arange(
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
):
    """Create a tensor with evenly spaced values."""
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
    """Create a tensor with uniform random values."""
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
    """Create a tensor with Gaussian (normal) random values."""
    dtype, device = defaults(dtype, device)
    return _gaussian_op(shape, mean, std, dtype, device)


normal = gaussian


def zeros_like(x: Any) -> Any:
    """Create a tensor of zeros with the same shape/dtype/device as x."""
    return zeros(x.shape, dtype=x.dtype, device=x.device)


def ones_like(x: Any) -> Any:
    """Create a tensor of ones with the same shape/dtype/device as x."""
    return ones(x.shape, dtype=x.dtype, device=x.device)

def full_like(x: Any, value: Number) -> Any:
    """Create a tensor of ones with the same shape/dtype/device as x."""
    return full(x.shape, value, dtype=x.dtype, device=x.device)

__all__ = [
    "ConstantOp",
    "FullOp",
    "ZerosOp",
    "OnesOp",
    "ArangeOp",
    "UniformOp",
    "GaussianOp",
    "constant",
    "full",
    "zeros",
    "ones",
    "arange",
    "uniform",
    "gaussian",
    "normal",
    "zeros_like",
    "ones_like",
    "full_like",
]
