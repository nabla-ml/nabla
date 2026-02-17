# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, ShapeLike, TensorType, ops
from max.graph.ops.constant import NestedArray, Number

from ..core import defaults
from .base import (
    CreationOperation,
    OpArgs,
    Operation,
    OpKwargs,
    OpResult,
    OpTensorValues,
)

if TYPE_CHECKING:
    from ..core import Tensor


class ConstantOp(CreationOperation):
    """Create a tensor from a constant value."""

    @property
    def name(self) -> str:
        return "constant"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        value = args[0]
        dtype = kwargs["dtype"]
        device = kwargs["device"]
        return [ops.constant(value, dtype, device)]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        return [None for _ in range(len(primals))]


class FullOp(CreationOperation):
    """Create a tensor filled with a constant value."""

    @property
    def name(self) -> str:
        return "full"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        shape = kwargs["shape"]
        value = kwargs["value"]
        dtype = kwargs["dtype"]
        device = kwargs["device"]
        const = ops.constant(value, dtype, device)
        return [ops.broadcast_to(const, shape)]


class ZerosOp(CreationOperation):
    """Create a tensor filled with zeros."""

    @property
    def name(self) -> str:
        return "zeros"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        shape = kwargs["shape"]
        dtype = kwargs["dtype"]
        device = kwargs["device"]
        const = ops.constant(0, dtype, device)
        return [ops.broadcast_to(const, shape)]


class OnesOp(CreationOperation):
    """Create a tensor filled with ones."""

    @property
    def name(self) -> str:
        return "ones"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        shape = kwargs["shape"]
        dtype = kwargs["dtype"]
        device = kwargs["device"]
        const = ops.constant(1, dtype, device)
        return [ops.broadcast_to(const, shape)]


class ArangeOp(CreationOperation):
    """Create a tensor with evenly spaced values."""

    @property
    def name(self) -> str:
        return "arange"

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for arange (1D)."""
        import math

        from ..core.sharding import spmd

        start = args[0] if len(args) > 0 else kwargs.get("start", 0)
        stop = args[1] if len(args) > 1 else kwargs.get("stop")
        step = args[2] if len(args) > 2 else kwargs.get("step", 1)
        dtype = args[3] if len(args) > 3 else kwargs.get("dtype")

        if stop is None:
            stop = start
            start = 0

        if step == 0:
            raise ValueError("arange step must be non-zero")

        length = math.ceil((stop - start) / step)
        length = max(0, int(length))

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = [(length,)] * num_shards

        device = args[4] if len(args) > 4 else kwargs.get("device")
        dtypes = [dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = list(mesh.device_refs)
            else:
                devices = [mesh.device_refs[0]] * num_shards
        else:
            devices = [device] * num_shards

        return shapes, dtypes, devices

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        start = kwargs["start"]
        stop = kwargs["stop"]
        step = kwargs["step"]
        dtype = kwargs["dtype"]
        device = kwargs["device"]
        return [ops.range(start, stop, step, dtype=dtype, device=device)]


class UniformOp(CreationOperation):
    """Create a tensor with uniform random values."""

    @property
    def name(self) -> str:
        return "uniform"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        shape = kwargs["shape"]
        low = kwargs["low"]
        high = kwargs["high"]
        dtype = kwargs["dtype"]
        device = kwargs["device"]
        tensor_type = TensorType(dtype, shape, device=DeviceRef.from_device(device))
        return [ops.random.uniform(tensor_type, range=(low, high))]


class GaussianOp(CreationOperation):
    """Create a tensor with Gaussian (normal) random values."""

    @property
    def name(self) -> str:
        return "gaussian"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        shape = kwargs["shape"]
        mean = kwargs["mean"]
        std = kwargs["std"]
        dtype = kwargs["dtype"]
        device = kwargs["device"]
        tensor_type = TensorType(dtype, shape, device=DeviceRef.from_device(device))
        return [ops.random.gaussian(tensor_type, mean=mean, std=std)]


class HannWindowOp(CreationOperation):
    """Create a 1D Hann window tensor."""

    @property
    def name(self) -> str:
        return "hann_window"

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        window_length = args[0] if len(args) > 0 else kwargs.get("window_length")
        dtype = kwargs.get("dtype", DType.float32)
        device = args[1] if len(args) > 1 else kwargs.get("device")

        from ..core.sharding import spmd

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = [(window_length,)] * num_shards
        dtypes = [dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = list(mesh.device_refs)
            else:
                devices = [mesh.device_refs[0]] * num_shards
        else:
            devices = [device] * num_shards

        return shapes, dtypes, devices

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        window_length = kwargs["window_length"]
        device = kwargs["device"]
        periodic = kwargs.get("periodic", True)
        dtype = kwargs.get("dtype", DType.float32)
        return [ops.hann_window(window_length, device, periodic=periodic, dtype=dtype)]


class TriOp(Operation):
    """Base for Triu and Tril."""

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        x = args[0]
        shapes = [
            tuple(int(d) for d in x.physical_local_shape(i))
            for i in range(x.num_shards)
        ]
        return shapes, [x.dtype] * x.num_shards, [x.device] * x.num_shards

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        k = kwargs.get("k", 0)
        return [self.__class__()([cotangents[0]], {"k": k})[0]]


class TriuOp(TriOp):
    """Upper triangular part of a matrix."""

    @property
    def name(self) -> str:
        return "triu"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        k = kwargs.get("k", 0)
        try:
            return [ops.triu(x, k)]
        except AttributeError:
            return [ops.band_part(x, -k, -1)]


class TrilOp(TriOp):
    """Lower triangular part of a matrix."""

    @property
    def name(self) -> str:
        return "tril"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        k = kwargs.get("k", 0)
        try:
            return [ops.tril(x, k)]
        except AttributeError:
            return [ops.band_part(x, -1, k)]


_constant_op = ConstantOp()
_full_op = FullOp()
_zeros_op = ZerosOp()
_ones_op = OnesOp()
_arange_op = ArangeOp()
_uniform_op = UniformOp()
_gaussian_op = GaussianOp()
_hann_window_op = HannWindowOp()
_triu_op = TriuOp()
_tril_op = TrilOp()


def constant(
    value: NestedArray | Number,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
):
    """Create a tensor from a constant value."""
    import numpy as np

    from ..core import Tensor

    def _to_numpy_dtype(dt: DType):
        try:
            return dt.to_numpy()
        except Exception:
            return np.dtype(dt.name)

    if dtype is None:
        if isinstance(value, np.ndarray):
            with contextlib.suppress(KeyError, ValueError):
                dtype = DType[str(value.dtype)]
        elif isinstance(value, int):
            dtype = DType.int32
        elif isinstance(value, float):
            dtype = DType.float32
        elif isinstance(value, bool):
            dtype = DType.bool

    scalar_like = isinstance(value, (int, float, bool, complex)) or (
        isinstance(value, np.ndarray) and value.ndim == 0
    )

    if scalar_like:
        dtype, device = defaults(dtype, device)
        arr = np.asarray(value)
        arr = arr.astype(_to_numpy_dtype(dtype), copy=False)
        t = Tensor.from_dlpack(arr)
        if t.dtype != dtype:
            t = t.to(dtype)
        if t.device != device:
            t = t.to(device)
        return t

    if isinstance(value, (list, tuple)):
        value = np.array(value)
        if dtype:
            with contextlib.suppress(Exception):
                value = value.astype(_to_numpy_dtype(dtype), copy=False)
    return Tensor.from_dlpack(value)


def full(
    shape: ShapeLike,
    value: Number,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    is_traced: bool = False,
):
    """Create a tensor filled with a constant value."""
    dtype, device = defaults(dtype, device)
    t = _full_op(
        [], {"shape": shape, "value": value, "dtype": dtype, "device": device}
    )[0]
    if is_traced:
        t.is_traced = True
    return t


def zeros(
    shape: ShapeLike,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    is_traced: bool = False,
):
    """Create a tensor filled with zeros."""
    dtype, device = defaults(dtype, device)
    t = _zeros_op([], {"shape": shape, "dtype": dtype, "device": device})[0]
    if is_traced:
        t.is_traced = True
    return t


def ones(
    shape: ShapeLike,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    is_traced: bool = False,
):
    """Create a tensor filled with ones."""
    dtype, device = defaults(dtype, device)
    t = _ones_op([], {"shape": shape, "dtype": dtype, "device": device})[0]
    if is_traced:
        t.is_traced = True
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
    return _arange_op(
        [],
        {"start": start, "stop": stop, "step": step, "dtype": dtype, "device": device},
    )[0]


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
    return _uniform_op(
        [], {"shape": shape, "low": low, "high": high, "dtype": dtype, "device": device}
    )[0]


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
    return _gaussian_op(
        [], {"shape": shape, "mean": mean, "std": std, "dtype": dtype, "device": device}
    )[0]


normal = gaussian


def hann_window(
    window_length: int,
    *,
    periodic: bool = True,
    dtype: DType | None = None,
    device: Device | None = None,
):
    """Create a 1D Hann window tensor."""
    dtype, device = defaults(dtype, device)
    return _hann_window_op(
        [],
        {
            "window_length": window_length,
            "device": device,
            "periodic": periodic,
            "dtype": dtype,
        },
    )[0]


def triu(x: Tensor, k: int = 0) -> Tensor:
    """Upper triangular part of a matrix."""
    return _triu_op([x], {"k": k})[0]


def tril(x: Tensor, k: int = 0) -> Tensor:
    """Lower triangular part of a matrix."""
    return _tril_op([x], {"k": k})[0]


def _like_helper(
    x: Tensor, create_fn: Callable[..., Tensor], *extra_args: Any
) -> Tensor:
    """Shared implementation for zeros_like/ones_like/full_like."""
    from ..core import Tensor

    if not isinstance(x, Tensor):
        return create_fn(*extra_args, x.shape, dtype=x.dtype, device=x.device)

    shape = x.physical_global_shape
    res = create_fn(*extra_args, shape, dtype=x.dtype, device=x.device)
    res.batch_dims = x.batch_dims

    if x.sharding:
        from .communication.shard import shard

        res = shard(
            res,
            x.sharding.mesh,
            x.sharding.dim_specs,
            replicated_axes=x.sharding.replicated_axes,
        )
    return res


def zeros_like(x: Tensor) -> Tensor:
    """Create a tensor of zeros with the same shape/dtype/device/sharding as x."""
    return _like_helper(x, zeros)


def ones_like(x: Tensor) -> Tensor:
    """Create a tensor of ones with the same shape/dtype/device/sharding as x."""
    return _like_helper(x, ones)


def full_like(x: Tensor, value: Number) -> Tensor:
    """Create a tensor filled with value, matching x's properties."""
    return _like_helper(x, full, value)


__all__ = [
    "constant",
    "full",
    "zeros",
    "ones",
    "arange",
    "uniform",
    "gaussian",
    "normal",
    "hann_window",
    "triu",
    "tril",
    "zeros_like",
    "ones_like",
    "full_like",
]
