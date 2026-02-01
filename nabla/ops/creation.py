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

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shape for constant (scalar)."""
        from ..core.sharding import spmd

        dtype = args[1] if len(args) > 1 else kwargs.get("dtype")

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = [()] * num_shards
        device = args[2] if len(args) > 2 else kwargs.get("device")
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
        value: NestedArray | Number,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        return ops.constant(value, dtype, device)

    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Physical execution for ConstantOp.

        Creation ops don't use sharding - they create new data.
        We just call kernel once and replicate across all shards if needed.
        """
        from ..core import GRAPH
        from ..core.sharding import spmd
        from typing import Any

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        with GRAPH.graph:
            result = self.kernel(*args, **kwargs)
            shard_results = [result] * num_shards

        return (shard_results, None, mesh)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        # Constant has no differentiable inputs
        return (None, None, None)


class FullOp(Operation):
    """Create a tensor filled with a constant value."""

    @property
    def name(self) -> str:
        return "full"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for full."""
        from ..core.sharding import spmd, spec

        shape = args[0] if len(args) > 0 else kwargs.get("shape")
        dtype = args[2] if len(args) > 2 else kwargs.get("dtype")

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        if shape is None:
            raise RuntimeError("full requires a shape")

        shapes = []
        if output_sharding and mesh:
            for i in range(num_shards):
                local = spec.compute_local_shape(shape, output_sharding, device_id=i)
                shapes.append(tuple(int(d) for d in local))
        else:
            shapes = [tuple(int(d) for d in shape)] * num_shards

        device = args[3] if len(args) > 3 else kwargs.get("device")
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
        shape: ShapeLike,
        value: Number,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        const = ops.constant(value, dtype, device)
        return ops.broadcast_to(const, shape)

    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Physical execution for FullOp."""
        from ..core import GRAPH
        from ..core.sharding import spmd
        from typing import Any

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        with GRAPH.graph:
            result = self.kernel(*args, **kwargs)
            shard_results = [result] * num_shards

        return (shard_results, None, mesh)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        return (None, None, None, None)


class ZerosOp(Operation):
    """Create a tensor filled with zeros."""

    @property
    def name(self) -> str:
        return "zeros"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for zeros."""
        from ..core.sharding import spmd, spec

        shape = args[0] if len(args) > 0 else kwargs.get("shape")
        dtype = args[1] if len(args) > 1 else kwargs.get("dtype")

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        if shape is None:
            raise RuntimeError("zeros requires a shape")

        shapes = []
        if output_sharding and mesh:
            for i in range(num_shards):
                local = spec.compute_local_shape(shape, output_sharding, device_id=i)
                shapes.append(tuple(int(d) for d in local))
        else:
            shapes = [tuple(int(d) for d in shape)] * num_shards

        device = args[2] if len(args) > 2 else kwargs.get("device")
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
        shape: ShapeLike,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        const = ops.constant(0, dtype, device)
        return ops.broadcast_to(const, shape)

    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Physical execution for ZerosOp."""
        from ..core import GRAPH
        from ..core.sharding import spmd
        from typing import Any

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        with GRAPH.graph:
            result = self.kernel(*args, **kwargs)
            shard_results = [result] * num_shards

        return (shard_results, None, mesh)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        return (None, None, None)


class OnesOp(Operation):
    """Create a tensor filled with ones."""

    @property
    def name(self) -> str:
        return "ones"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for ones."""
        from ..core.sharding import spmd, spec

        shape = args[0] if len(args) > 0 else kwargs.get("shape")
        dtype = args[1] if len(args) > 1 else kwargs.get("dtype")

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        if shape is None:
            raise RuntimeError("ones requires a shape")

        shapes = []
        if output_sharding and mesh:
            for i in range(num_shards):
                local = spec.compute_local_shape(shape, output_sharding, device_id=i)
                shapes.append(tuple(int(d) for d in local))
        else:
            shapes = [tuple(int(d) for d in shape)] * num_shards

        device = args[2] if len(args) > 2 else kwargs.get("device")
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
        shape: ShapeLike,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        const = ops.constant(1, dtype, device)
        return ops.broadcast_to(const, shape)

    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Physical execution for OnesOp."""
        from ..core import GRAPH
        from ..core.sharding import spmd
        from typing import Any

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        with GRAPH.graph:
            result = self.kernel(*args, **kwargs)
            shard_results = [result] * num_shards

        return (shard_results, None, mesh)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        return (None, None, None)


class ArangeOp(Operation):
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

    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Physical execution for ArangeOp."""
        from ..core import GRAPH
        from ..core.sharding import spmd
        from typing import Any

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        with GRAPH.graph:
            result = self.kernel(*args, **kwargs)
            shard_results = [result] * num_shards

        return (shard_results, None, mesh)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        return (None, None, None, None, None)


class UniformOp(Operation):
    """Create a tensor with uniform random values."""

    @property
    def name(self) -> str:
        return "uniform"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for uniform."""
        from ..core.sharding import spmd, spec

        shape = args[0] if len(args) > 0 else kwargs.get("shape")
        dtype = args[3] if len(args) > 3 else kwargs.get("dtype")

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        if shape is None:
            raise RuntimeError("uniform requires a shape")

        shapes = []
        if output_sharding and mesh:
            for i in range(num_shards):
                local = spec.compute_local_shape(shape, output_sharding, device_id=i)
                shapes.append(tuple(int(d) for d in local))
        else:
            shapes = [tuple(int(d) for d in shape)] * num_shards

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
        shape: ShapeLike,
        low: float,
        high: float,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        tensor_type = TensorType(dtype, shape, device=DeviceRef.from_device(device))
        return ops.random.uniform(tensor_type, range=(low, high))

    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Physical execution for UniformOp.

        Random ops create independent samples on each shard.
        """
        from ..core import GRAPH
        from ..core.sharding import spmd
        from typing import Any

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        with GRAPH.graph:
            # Each shard gets independent random values
            shard_results = [self.kernel(*args, **kwargs) for _ in range(num_shards)]

        return (shard_results, None, mesh)


class GaussianOp(Operation):
    """Create a tensor with Gaussian (normal) random values."""

    @property
    def name(self) -> str:
        return "gaussian"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for gaussian."""
        from ..core.sharding import spmd, spec

        shape = args[0] if len(args) > 0 else kwargs.get("shape")
        dtype = args[3] if len(args) > 3 else kwargs.get("dtype")

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        if shape is None:
            raise RuntimeError("gaussian requires a shape")

        shapes = []
        if output_sharding and mesh:
            for i in range(num_shards):
                local = spec.compute_local_shape(shape, output_sharding, device_id=i)
                shapes.append(tuple(int(d) for d in local))
        else:
            shapes = [tuple(int(d) for d in shape)] * num_shards

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
        shape: ShapeLike,
        mean: float,
        std: float,
        dtype: DType,
        device: Device,
    ) -> TensorValue:
        tensor_type = TensorType(dtype, shape, device=DeviceRef.from_device(device))
        return ops.random.gaussian(tensor_type, mean=mean, std=std)

    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Physical execution for GaussianOp.

        Random ops create independent samples on each shard.
        """
        from ..core import GRAPH
        from ..core.sharding import spmd
        from typing import Any

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        with GRAPH.graph:
            # Each shard gets independent random values
            shard_results = [self.kernel(*args, **kwargs) for _ in range(num_shards)]

        return (shard_results, None, mesh)


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
    import numpy as np
    from ..core.tensor.api import Tensor

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

    # If it's not a simple number, convert to a realized tensor via DLPack
    # to avoid embedding large data as constants in the MAX graph.
    if not isinstance(value, (int, float, bool, complex)) and not (
        isinstance(value, np.ndarray) and value.ndim == 0
    ):
        # Convert lists/tuples to numpy array first
        if isinstance(value, (list, tuple)):
            value = np.array(value)
            if dtype:
                # Map MAX DType to numpy dtype if needed
                try:
                    # DType.float32.name -> "float32", which numpy accepts
                    value = value.astype(dtype.name)
                except Exception:
                    pass

        return Tensor.from_dlpack(value)

    dtype, device = defaults(dtype, device)
    return _constant_op(value, dtype, device)


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
    t = _full_op(shape, value, dtype, device)
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
    t = _zeros_op(shape, dtype, device)
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
    t = _ones_op(shape, dtype, device)
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
    """Create a tensor of zeros with the same shape/dtype/device/sharding as x."""
    from ..core.tensor.api import Tensor

    if not isinstance(x, Tensor):
        return zeros(x.shape, dtype=x.dtype, device=x.device)

    # Use physical global shape (handles batch_dims cases)
    # The _impl access is safe because we check isinstance(Tensor)
    shape = x._impl.physical_global_shape

    # Use global logical shape so that shard() can correctly slice it down
    # (MAX compiler optimizes zeros(global) -> shard -> zeros(local))
    result = zeros(shape, dtype=x.dtype, device=x.device)

    # Inherit batch_dims
    result.batch_dims = x.batch_dims

    # If x is sharded, shard the result the same way
    if x.sharding and x.sharding.mesh:
        from .communication import shard

        result = shard(
            result,
            x.sharding.mesh,
            x.sharding.dim_specs,
            replicated_axes=x.sharding.replicated_axes,
        )

    return result


def ones_like(x: Any) -> Any:
    """Create a tensor of ones with the same shape/dtype/device/sharding as x."""
    from ..core.tensor.api import Tensor

    if not isinstance(x, Tensor):
        return ones(x.shape, dtype=x.dtype, device=x.device)

    # Use physical global shape
    shape = x._impl.physical_global_shape

    # Use global logical shape
    result = ones(shape, dtype=x.dtype, device=x.device)

    # Inherit batch_dims
    result.batch_dims = x.batch_dims

    # If x is sharded, shard the result the same way
    if x.sharding and x.sharding.mesh:
        from .communication import shard

        result = shard(
            result,
            x.sharding.mesh,
            x.sharding.dim_specs,
            replicated_axes=x.sharding.replicated_axes,
        )

    return result


def full_like(x: Any, value: Number) -> Any:
    """Create a tensor filled with value with the same shape/dtype/device/sharding as x."""
    from ..core.tensor.api import Tensor

    if not isinstance(x, Tensor):
        return full(x.shape, value, dtype=x.dtype, device=x.device)

    # Use physical global shape
    shape = x._impl.physical_global_shape

    # Use global logical shape
    result = full(shape, value, dtype=x.dtype, device=x.device)

    # Inherit batch_dims
    result.batch_dims = x.batch_dims

    # If x is sharded, shard the result the same way
    if x.sharding and x.sharding.mesh:
        from .communication import shard

        result = shard(
            result,
            x.sharding.mesh,
            x.sharding.dim_specs,
            replicated_axes=x.sharding.replicated_axes,
        )

    return result


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
