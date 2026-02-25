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
    """Create a tensor from a Python scalar, list, or NumPy array.

    Scalars and 0-d arrays are wrapped with the default dtype unless
    *dtype* is specified. Multi-dimensional arrays/lists are converted
    via DLPack without copying.

    Args:
        value: Python int, float, bool, complex, list, or ``np.ndarray``.
        dtype: Target element dtype. Inferred from *value* if ``None``.
        device: Target device. Uses the current default if ``None``.

    Returns:
        A realized ``Tensor`` wrapping *value*.
    """
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
    """Return a tensor of *shape* filled with *value*.

    Args:
        shape: Output shape.
        value: Fill value (scalar).
        dtype: Element dtype. Uses the current default if ``None``.
        device: Target device. Uses the current default if ``None``.

    Returns:
        Tensor with all elements equal to *value*.
    """
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
    """Return a tensor of *shape* filled with zeros.

    Args:
        shape: Output shape.
        dtype: Element dtype. Uses the current default if ``None``.
        device: Target device. Uses the current default if ``None``.

    Returns:
        Zero-valued tensor of the given shape and dtype.
    """
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
    """Return a tensor of *shape* filled with ones.

    Args:
        shape: Output shape.
        dtype: Element dtype. Uses the current default if ``None``.
        device: Target device. Uses the current default if ``None``.

    Returns:
        One-valued tensor of the given shape and dtype.
    """
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
    """Return a 1-D tensor with evenly spaced values in ``[start, stop)``.

    When called with a single positional argument, it is treated as *stop*
    and *start* defaults to ``0``, matching NumPy / PyTorch semantics.

    Args:
        start: Start of the interval (inclusive). Default: ``0``.
        stop: End of the interval (exclusive). If ``None``, *start* is
            used as *stop* and start becomes ``0``.
        step: Spacing between values. Default: ``1``.
        dtype: Element dtype. Uses the current default if ``None``.
        device: Target device. Uses the current default if ``None``.

    Returns:
        1-D tensor of length ``ceil((stop - start) / step)``.
    """
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
    """Return a tensor of *shape* with values sampled from U(*low*, *high*).

    Args:
        shape: Output shape.
        low: Lower bound of the uniform distribution.
        high: Upper bound of the uniform distribution.
        dtype: Element dtype. Uses the current default if ``None``.
        device: Target device. Uses the current default if ``None``.

    Returns:
        Tensor with elements drawn uniformly at random from [*low*, *high*).
    """
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
    """Return a tensor of *shape* with values sampled from N(*mean*, *std*²).

    Also accessible as ``nabla.normal``.

    Args:
        shape: Output shape.
        mean: Mean of the Gaussian distribution. Default: ``0.0``.
        std: Standard deviation. Default: ``1.0``.
        dtype: Element dtype. Uses the current default if ``None``.
        device: Target device. Uses the current default if ``None``.

    Returns:
        Tensor with elements drawn from a normal distribution.
    """
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
    """Return a 1-D Hann (raised cosine) window of length *window_length*.

    The window follows the convention used in signal processing:
    ``w[n] = 0.5 * (1 - cos(2π n / N))`` where *N* is the window size.

    Args:
        window_length: Number of points in the window.
        periodic: If ``True`` (default), generates a periodic window for
            use in spectral analysis. If ``False``, generates a symmetric
            window.
        dtype: Element dtype. Defaults to ``float32``.
        device: Target device. Uses the current default if ``None``.

    Returns:
        1-D Tensor of shape ``(window_length,)``.
    """
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
    """Return the upper triangular part of a matrix (or batch of matrices).

    Elements below the *k*-th diagonal are zeroed out.

    Args:
        x: Input tensor of shape ``(*, M, N)``.
        k: Diagonal offset. ``k=0`` (default) is the main diagonal,
           ``k>0`` is above it, ``k<0`` is below.

    Returns:
        Tensor of the same shape as *x* with the lower triangle zeroed.
    """
    return _triu_op([x], {"k": k})[0]


def tril(x: Tensor, k: int = 0) -> Tensor:
    """Return the lower triangular part of a matrix (or batch of matrices).

    Elements above the *k*-th diagonal are zeroed out.

    Args:
        x: Input tensor of shape ``(*, M, N)``.
        k: Diagonal offset. ``k=0`` (default) is the main diagonal,
           ``k>0`` is above it, ``k<0`` is below.

    Returns:
        Tensor of the same shape as *x* with the upper triangle zeroed.
    """
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
    """Return a zero tensor with the same shape, dtype, device, and sharding as *x*.

    Args:
        x: Reference tensor.

    Returns:
        Tensor of zeros matching all metadata of *x*.
    """
    return _like_helper(x, zeros)


def ones_like(x: Tensor) -> Tensor:
    """Return a ones tensor with the same shape, dtype, device, and sharding as *x*.

    Args:
        x: Reference tensor.

    Returns:
        Tensor of ones matching all metadata of *x*.
    """
    return _like_helper(x, ones)


def full_like(x: Tensor, value: Number) -> Tensor:
    """Return a tensor filled with *value*, matching *x*'s shape, dtype, device, and sharding.

    Args:
        x: Reference tensor.
        value: Scalar fill value.

    Returns:
        Tensor filled with *value*, with the same metadata as *x*.
    """
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
