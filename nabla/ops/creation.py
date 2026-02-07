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

from ..core import defaults
from .base import CreationOperation, Operation


class ConstantOp(CreationOperation):
    """Create a tensor from a constant value."""

    @property
    def name(self) -> str:
        return "constant"

    def kernel(
        self,
        value: NestedArray | Number,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        return ops.constant(value, dtype, device)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        return (None, None, None)


class FullOp(CreationOperation):
    """Create a tensor filled with a constant value."""

    @property
    def name(self) -> str:
        return "full"

    def kernel(
        self,
        shape: ShapeLike,
        value: Number,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        const = ops.constant(value, dtype, device)
        return ops.broadcast_to(const, shape)


class ZerosOp(CreationOperation):
    """Create a tensor filled with zeros."""

    @property
    def name(self) -> str:
        return "zeros"

    def kernel(
        self,
        shape: ShapeLike,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        const = ops.constant(0, dtype, device)
        return ops.broadcast_to(const, shape)


class OnesOp(CreationOperation):
    """Create a tensor filled with ones."""

    @property
    def name(self) -> str:
        return "ones"

    def kernel(
        self,
        shape: ShapeLike,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        const = ops.constant(1, dtype, device)
        return ops.broadcast_to(const, shape)


class ArangeOp(CreationOperation):
    """Create a tensor with evenly spaced values."""

    @property
    def name(self) -> str:
        return "arange"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
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
                devices = [d for d in mesh.device_refs]
            else:
                devices = [mesh.device_refs[0]] * num_shards
        else:
            devices = [device] * num_shards

        return shapes, dtypes, devices

    def kernel(
        self,
        start: int,
        stop: int,
        step: int,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        return ops.range(start, stop, step, dtype=dtype, device=device)


class UniformOp(CreationOperation):
    """Create a tensor with uniform random values."""

    @property
    def name(self) -> str:
        return "uniform"

    def kernel(
        self,
        shape: ShapeLike,
        low: float,
        high: float,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        tensor_type = TensorType(dtype, shape, device=DeviceRef.from_device(device))
        return ops.random.uniform(tensor_type, range=(low, high))


class GaussianOp(CreationOperation):
    """Create a tensor with Gaussian (normal) random values."""

    @property
    def name(self) -> str:
        return "gaussian"

    def kernel(
        self,
        shape: ShapeLike,
        mean: float,
        std: float,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        tensor_type = TensorType(dtype, shape, device=DeviceRef.from_device(device))
        return ops.random.gaussian(tensor_type, mean=mean, std=std)


class HannWindowOp(CreationOperation):
    """Create a 1D Hann window tensor."""

    @property
    def name(self) -> str:
        return "hann_window"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
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
                devices = [d for d in mesh.device_refs]
            else:
                devices = [mesh.device_refs[0]] * num_shards
        else:
            devices = [device] * num_shards

        return shapes, dtypes, devices

    def kernel(
        self,
        window_length: int,
        device: Device,
        periodic: bool = True,
        dtype: DType = DType.float32,
    ) -> TensorValue:
        return ops.hann_window(window_length, device, periodic=periodic, dtype=dtype)


class TriOp(Operation):
    """Base for Triu and Tril."""

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        x = args[0]
        shapes = [
            tuple(int(d) for d in x.physical_local_shape(i))
            for i in range(x.num_shards)
        ]
        return shapes, [x.dtype] * x.num_shards, [x.device] * x.num_shards

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        k = output.op_kwargs.get("k", 0)
        return (self.__class__()(cotangent, k=k),)


class TriuOp(TriOp):
    """Upper triangular part of a matrix."""

    @property
    def name(self) -> str:
        return "triu"

    def kernel(self, x: TensorValue, *, k: int = 0) -> TensorValue:
        # Fallback to band_part if triu is missing
        # Upper triangle: keep everything from diagonal k upwards
        # lower=-1 (ignore), upper=0 (diagonal) -- wait, band_part(x, num_lower, num_upper)
        # Keeps elements where in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) &&
        #                                      (num_upper < 0 || (n-m) <= num_upper)
        # For triu(k): keep j >= i + k => j - i >= k => (i - j) <= -k.
        # This roughly maps to num_lower = -k? No, that's unintuitive.
        # XLA/TF semantics: band_part(x, num_lower, num_upper)
        # Retains: num_lower <= (i - j) <= num_upper. (assuming m=i, n=j)
        # triu(k): j >= i + k  => 0 >= i - j + k => i - j <= -k.
        # So we want (i - j) <= -k.
        #num_lower = 0 # No wait, we keep everything "above" diagonal. So i is small, j is large.
        # i-j is negative.
        # So we want lower bound on i-j? We want to discard things where i-j > -k.
        try:
             return ops.triu(x, k)
        except AttributeError:
             # Fallback: band_part(x, 0, -1) keeps upper triangle (k=0)
             # ops.band_part(input, num_lower, num_upper)
             # To Implement triu(k): num_lower = -k? No.
             # We want to KEEP if j >= i + k.
             # This means i - j <= -k.
             # band_part keeps num_lower <= i - j <= num_upper.
             # So we set num_upper = -1 (infinity), and num_lower?
             # Actually, simpler:
             # triu(k) is everything *above* the k-th diagonal.
             # This corresponds to band_part(x, 0, -1) for k=0? No wait.
             # Let's check max ops.
             return ops.band_part(x, -k, -1) # wait, if k=0, band_part(0, -1) -> main diagonal and up? 
             # If k=1 (superdiagonal), we want j >= i+1. i-j <= -1.
             # band_part(x, -1, -1) would mean -1 <= i-j. That keeps lower stuff.
             # Correct mapping for triu(x, k):
             # band_part(x, num_lower, num_upper)
             # We want (j >= i + k) => (j - i >= k).
             # band_part keeps (i - j <= num_lower) AND (j - i <= num_upper).
             # Wait, TF says: (num_lower < 0 || (m-n) <= num_lower) && (num_upper < 0 || (n-m) <= num_upper)
             
             # Let's assume `ops.band_part` follows standard XLA/TF.
             # triu(x, k): mask where j >= i + k.
             # band_part(x, -k-1??, -1)?
             # Actually easiest way is usually multiply by mask if `ops.band_part` is confusing.
             # But let's try strict substitution. 
             # triu(k=0): j >= i. i <= j. Keep upper.
             # band_part(x, 0, -1):
             #   (m-n) <= 0 => m <= n (i<=j). Correct for k=0.
             # triu(k=1): j >= i+1. i <= j-1.
             # band_part(x, -1, -1)?
             #   (m-n) <= -1 => m <= n-1 => i <= j-1. Correct.
             return ops.band_part(x, -k, -1)


class TrilOp(TriOp):
    """Lower triangular part of a matrix."""

    @property
    def name(self) -> str:
        return "tril"

    def kernel(self, x: TensorValue, *, k: int = 0) -> TensorValue:
        try:
            return ops.tril(x, k)
        except AttributeError:
             # tril(k): j <= i + k.
             # j - i <= k. (i - j) >= -k.
             # band_part(x, num_lower, num_upper)
             # We want to KEEP if j <= i + k.
             # band_part keeps (m-n <= num_lower) and (n-m <= num_upper).
             # n-m <= k.
             # So num_upper = k.
             # num_lower = -1 (infinity).
             return ops.band_part(x, -1, k)


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

    if dtype is None:
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

    if not isinstance(value, (int, float, bool, complex)) and not (
        isinstance(value, np.ndarray) and value.ndim == 0
    ):
        if isinstance(value, (list, tuple)):
            value = np.array(value)
            if dtype:
                try:
                    value = value.astype(dtype.name)
                except Exception:
                    pass
        return Tensor.from_dlpack(value)

    dtype, device = defaults(dtype, device)
    return _constant_op(value, dtype=dtype, device=device)


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
    t = _full_op(shape, value, dtype=dtype, device=device)
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
    t = _zeros_op(shape, dtype=dtype, device=device)
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
    t = _ones_op(shape, dtype=dtype, device=device)
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
    return _arange_op(start, stop, step, dtype=dtype, device=device)


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
    return _uniform_op(shape, low, high, dtype=dtype, device=device)


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
    return _gaussian_op(shape, mean, std, dtype=dtype, device=device)


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
    return _hann_window_op(window_length, device=device, periodic=periodic, dtype=dtype)


def triu(x: Any, k: int = 0) -> Any:
    """Upper triangular part of a matrix."""
    return _triu_op(x, k=k)


def tril(x: Any, k: int = 0) -> Any:
    """Lower triangular part of a matrix."""
    return _tril_op(x, k=k)


def _like_helper(x: Any, create_fn, *extra_args) -> Any:
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


def zeros_like(x: Any) -> Any:
    """Create a tensor of zeros with the same shape/dtype/device/sharding as x."""
    return _like_helper(x, zeros)


def ones_like(x: Any) -> Any:
    """Create a tensor of ones with the same shape/dtype/device/sharding as x."""
    return _like_helper(x, ones)


def full_like(x: Any, value: Number) -> Any:
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
