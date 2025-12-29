# ===----------------------------------------------------------------------=== #
# Nabla 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Provides experimental tensor operations with eager execution capabilities."""

from __future__ import annotations

import asyncio
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

try:
    from rich.pretty import pretty_repr
except ImportError:
    def pretty_repr(obj, **kwargs):
        return repr(obj)

from max import driver, graph
from max.driver import CPU, Device, DLPackArray
from max.dtype import DType
from max.graph import ShapeLike, TensorType, TensorValueLike, ops
from max.graph.ops.constant import NestedArray, Number
from max.graph.value import HasTensorValue

# Import from new modules
from .context import (
    defaults,
    default_device,
    default_dtype,
    defaults_like,
    _in_running_loop,
)
from .tensor_impl import TensorImpl, get_topological_order, print_computation_graph
from .compute_graph import GRAPH, driver_tensor_type

# Import ops modules
from ..ops import binary as binary_ops
from ..ops import creation


class Tensor(DLPackArray, HasTensorValue):
    """A multi-dimensional array with eager execution and automatic compilation."""

    _impl: TensorImpl
    _real: bool = False

    def __init__(
        self,
        *,
        storage: driver.Tensor | None = None,
        value: graph.BufferValue | graph.TensorValue | None = None,
        impl: TensorImpl | None = None,
        traced: bool = False,
    ):
        if impl is not None:
            self._impl = impl
        else:
            assert storage is not None or value is not None
            self._impl = TensorImpl(storages=storage, values=value, traced=traced)
        
        self.real = self._impl.is_realized

    # ===== Properties delegating to _impl =====
    
    @property
    def _storages(self) -> list[driver.Tensor] | None:
        return self._impl._storages
    
    @_storages.setter
    def _storages(self, value: list[driver.Tensor] | None) -> None:
        self._impl._storages = value
    
    @property
    def storage(self) -> driver.Tensor | None:
        if self._impl._storages and len(self._impl._storages) > 0:
            return self._impl._storages[0]
        return None
    
    @storage.setter
    def storage(self, value: driver.Tensor | None) -> None:
        if value is None:
            self._impl._storages = None
        else:
            self._impl._storages = [value]
    
    @property
    def _value(self) -> graph.BufferValue | graph.TensorValue | None:
        if self._impl._values and len(self._impl._values) > 0:
            return self._impl._values[0]
        return None
    
    @_value.setter
    def _value(self, value: graph.BufferValue | graph.TensorValue | None) -> None:
        if value is None:
            self._impl._values = []
        else:
            self._impl._values = [value]

    @property
    def _backing_value(self) -> driver.Tensor | graph.BufferValue | graph.TensorValue:
        if self._impl._storages is not None and len(self._impl._storages) > 0:
            return self._impl._storages[0]
        assert self._impl._values and len(self._impl._values) > 0
        return self._impl._values[0]

    @property
    def traced(self) -> bool:
        return self._impl.traced
    
    @traced.setter
    def traced(self, value: bool) -> None:
        self._impl.traced = value
    
    @property
    def batch_dims(self) -> int:
        """Number of batch dimensions (prefix of physical shape, used by vmap)."""
        return self._impl.batch_dims
    
    @property
    def op_kwargs(self) -> dict[str, Any]:
        """Keyword arguments passed to the operation that created this tensor.
        
        Useful in vjp_rule to access axis, keepdims, etc.
        """
        return self._impl.op_kwargs or {}
    
    def trace(self) -> Tensor:
        """Enable tracing on this tensor for autograd."""
        self._impl.traced = True
        return self
    
    def shard(
        self,
        mesh: Any,
        dim_specs: list[Any],
        replicated_axes: set[str] | None = None,
    ) -> Tensor:
        """Annotate tensor with sharding constraint.
        
        Args:
            mesh: DeviceMesh to shard across
            dim_specs: List of DimSpec for each dimension
            replicated_axes: Axes to explicitly replicate (optional)
        
        Returns:
            self (for chaining with .trace())
        """
        from ..sharding import ShardingSpec
        self._impl.sharding = ShardingSpec(
            mesh, dim_specs, replicated_axes=replicated_axes or set()
        )
        return self
    
    def with_sharding(
        self,
        mesh: Any,
        dim_specs: list[Any],
        replicated_axes: set[str] | None = None,
    ) -> Tensor:
        """Apply sharding constraint, resharding if needed.
        
        This is the explicit way to set output sharding. If the tensor's current
        sharding differs from the target, it will be resharded automatically.
        
        Args:
            mesh: DeviceMesh to shard across
            dim_specs: List of DimSpec for each dimension
            replicated_axes: Axes to explicitly replicate (optional)
            
        Returns:
            New tensor with target sharding (resharded if necessary)
        
        Example:
            # Force output to be replicated, even if operation infers sharding
            C = (A @ B).with_sharding(mesh, [DimSpec([]), DimSpec([])])
        """
        from ..sharding import ShardingSpec
        from ..sharding.spmd import needs_reshard, reshard_tensor
        
        target = ShardingSpec(mesh, dim_specs, replicated_axes=replicated_axes or set())
        
        # If already has sharding and it differs, reshard
        if self._impl.sharding and needs_reshard(self._impl.sharding, target):
            return reshard_tensor(self, self._impl.sharding, target, mesh)
        
        # Otherwise just annotate
        self._impl.sharding = target
        return self

    # ===== Factory methods =====

    @classmethod
    def from_graph_value(cls, value: graph.Value) -> Tensor:
        if not isinstance(value, (graph.TensorValue, graph.BufferValue)):
            raise TypeError(f"{value=} must be a tensor or buffer value")
        return cls(value=value)

    @classmethod
    def from_dlpack(cls, array: DLPackArray) -> Tensor:
        if isinstance(array, Tensor):
            return array
        return Tensor(storage=driver.Tensor.from_dlpack(array))

    @classmethod
    def constant(
        cls,
        value: DLPackArray | NestedArray | Number,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        return creation.constant(value, dtype=dtype, device=device)

    @classmethod
    def full(
        cls,
        shape: ShapeLike,
        value: Number,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
        traced: bool = False,
    ) -> Tensor:
        return creation.full(shape, value, dtype=dtype, device=device, traced=traced)

    @classmethod
    def zeros(
        cls,
        shape: ShapeLike,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
        traced: bool = False,
    ) -> Tensor:
        return creation.zeros(shape, dtype=dtype, device=device, traced=traced)

    @classmethod
    def ones(
        cls,
        shape: ShapeLike,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
        traced: bool = False,
    ) -> Tensor:
        return creation.ones(shape, dtype=dtype, device=device, traced=traced)

    @classmethod
    def arange(
        cls,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        return creation.arange(start, stop, step, dtype=dtype, device=device)

    @classmethod
    def uniform(
        cls,
        shape: ShapeLike,
        low: float = 0.0,
        high: float = 1.0,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        return creation.uniform(shape, low, high, dtype=dtype, device=device)

    @classmethod
    def gaussian(
        cls,
        shape: ShapeLike,
        mean: float = 0.0,
        std: float = 1.0,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        return creation.gaussian(shape, mean, std, dtype=dtype, device=device)
    
    # Alias for gaussian
    normal = gaussian

    # ===== Properties =====

    @property
    def type(self) -> graph.TensorType:
        value = self._backing_value
        t = driver_tensor_type(value) if isinstance(value, driver.Tensor) else value.type
        return t.as_tensor() if isinstance(t, graph.BufferType) else t

    @property
    def rank(self) -> int:
        return self._backing_value.rank

    @property
    def shape(self) -> graph.Shape:
        """Returns the global logical shape of the tensor.
        
        For sharded tensors, this returns the full tensor shape (not shard shape).
        Use local_shape to get the shape of individual shards.
        """
        # Prefer global_shape for consistency (already a Shape!)
        if self._impl.global_shape is not None:
            return self._impl.global_shape
        # Fallback for unrealized tensors without cached shape
        shape = self._backing_value.shape
        return shape if isinstance(shape, graph.Shape) else graph.Shape(shape)
    
    def local_shape(self, shard_idx: int = 0) -> graph.Shape:
        """Returns the shape of a specific shard.
        
        Args:
            shard_idx: Index of the shard (default: 0)
            
        Returns:
            Shape of the specified shard. For unsharded tensors, equals shape.
        """
        local = self._impl.logical_local_shape(shard_idx)
        if local is not None:
            return graph.Shape(local)
        return self.shape  # Fallback to global

    @property
    def dtype(self) -> DType:
        return self._backing_value.dtype

    @property
    def device(self) -> Device:
        device = self._backing_value.device
        return device if isinstance(device, Device) else device.to_device()

    @property
    def driver_tensor(self) -> driver.Tensor:
        if (storage := self.storage) is None:
            raise TypeError("Can't get driver tensor for symbolic tensor")
        return storage

    @property
    def real(self) -> bool:
        return self._real

    @real.setter
    def real(self, real: bool) -> None:
        if not real and self._in_global_compute_graph:
            GRAPH.add_unrealized(self)
        self._real = real

    # ===== Graph integration =====

    def __tensorvalue__(self) -> graph.TensorValue:
        if self._value is None:
            GRAPH.add_input(self)
        if isinstance(self._value, graph.BufferValue):
            return self._value[...]
        assert isinstance(self._value, graph.TensorValue)
        return self._value

    def __buffervalue__(self) -> graph.BufferValue:
        self.real = False
        if self._value is None:
            GRAPH.add_input(self)
        if isinstance(self._value, graph.BufferValue):
            return self._value
        assert isinstance(self._value, graph.TensorValue)
        tensor = self._value
        self._value = buffer = ops.buffer_create(tensor.type.as_buffer())
        buffer[...] = tensor
        return buffer

    @property
    def _in_global_compute_graph(self) -> bool:
        from max import _core
        if self._value is None:
            return True
        mlir_value = self._value.to_mlir()
        graph_op = mlir_value.owner.parent_op
        return graph_op == _core.Operation._from_cmlir(GRAPH.graph._mlir_op)

    # ===== Realization =====

    def __await__(self):
        if self.real:
            return self
        yield from asyncio.create_task(GRAPH.evaluate(self))
        assert self.real
        return self

    @property
    async def realize(self):
        return await self

    def _sync_realize(self) -> Tensor:
        if self.real:
            return self
        if not self._in_global_compute_graph:
            raise TypeError("Can't realize symbolic tensors in graph compilation.")
        if not _in_running_loop():
            return asyncio.run(self.realize)
        
        def is_interactive() -> bool:
            import __main__ as main
            return not hasattr(main, "__file__")

        if not is_interactive():
            warnings.warn("Use of synchronous tensor method inside another event loop.")

        loop = asyncio.new_event_loop()
        with ThreadPoolExecutor() as pool:
            fut = pool.submit(loop.run_until_complete, self.realize)
        return fut.result()

    # ===== Data access =====

    def __bool__(self) -> bool:
        return bool(self.item())

    def __hash__(self):
        return id(self)

    def __dlpack__(self, stream: int | None = None):
        self._sync_realize()
        assert self.storage is not None
        return self.storage.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        self._sync_realize()
        assert self.storage is not None
        return self.storage.__dlpack_device__()

    def __rich_repr__(self):
        yield "shape", self.shape
        yield "dtype", self.dtype
        yield "device", self.device

    def __repr__(self):
        if not self._in_global_compute_graph:
            return repr(self)
        self._sync_realize()
        dt = self.driver_tensor.to(CPU())
        values = [dt[idx].item() for idx in dt._iterate_indices()]
        return f"{self.type}: [{', '.join(str(v) for v in values)}]"

    def __deepcopy__(self, memo: object) -> Tensor:
        return self

    def item(self):
        if self.num_elements() != 1:
            raise TypeError("Only single-element tensors can be converted to Python scalars")
        self._sync_realize()
        return self.driver_tensor.to(CPU()).item()
    
    def to_numpy(self):
        """Convert tensor to numpy array, gathering shards if needed."""
        import numpy as np
        self._sync_realize()
        
        # Multi-shard: gather along sharded dimension(s)
        if self._impl._storages and len(self._impl._storages) > 1:
            return self._gather_shards_to_numpy()
        
        return self.driver_tensor.to(CPU()).to_numpy()
    
    def _gather_shards_to_numpy(self):
        """Gather all shards into a single numpy array.
        
        For 2D meshes with partial sharding (e.g., only one axis sharded),
        we have duplicate shards and must select only unique ones.
        """
        import numpy as np
        
        storages = self._impl._storages
        sharding = self._impl.sharding
        mesh = sharding.mesh if sharding else None
        
        if not storages or len(storages) <= 1:
            return storages[0].to(CPU()).to_numpy() if storages else None
        
        # If tensor is replicated (no sharded dimensions), all shards are identical
        # Just return the first one
        if sharding is None or sharding.is_fully_replicated():
            return storages[0].to(CPU()).to_numpy()
        
        # Find which dimension is sharded and on which mesh axis
        sharded_dim = None
        sharded_axis = None
        for dim_idx, dim_spec in enumerate(sharding.dim_specs):
            if dim_spec.axes:
                sharded_dim = dim_idx
                sharded_axis = dim_spec.axes[0]  # Use first axis
                break
        
        if sharded_dim is None or sharded_axis is None or mesh is None:
            return storages[0].to(CPU()).to_numpy()
        
        # Get the size of the sharded mesh axis
        sharded_axis_size = mesh.get_axis_size(sharded_axis)
        
        # Select unique shards: one per coordinate along the sharded axis
        # For 2x2 mesh with "model" sharding: shards 0,1 correspond to model=0,1
        # Shards 2,3 are duplicates (same computation, different data slice)
        unique_shards = []
        seen_coords = set()
        
        for shard_idx, storage in enumerate(storages):
            # Get the coordinate on the sharded axis
            coord = mesh.get_coordinate(shard_idx, sharded_axis)
            
            if coord not in seen_coords:
                seen_coords.add(coord)
                unique_shards.append((coord, storage.to(CPU()).to_numpy()))
        
        # Sort by coordinate and concatenate
        unique_shards.sort(key=lambda x: x[0])
        shard_arrays = [arr for _, arr in unique_shards]
        
        return np.concatenate(shard_arrays, axis=sharded_dim)

    def num_elements(self) -> int:
        elts = 1
        for dim in self.shape:
            elts *= int(dim)
        return elts

    # ===== Operators using binary_ops =====

    def __add__(self, rhs: TensorValueLike) -> Tensor:
        return binary_ops.add(self, _ensure_tensor(rhs, self))

    def __radd__(self, lhs: TensorValueLike) -> Tensor:
        return binary_ops.add(_ensure_tensor(lhs, self), self)

    def __sub__(self, rhs: TensorValueLike) -> Tensor:
        return binary_ops.sub(self, _ensure_tensor(rhs, self))

    def __rsub__(self, lhs: TensorValueLike) -> Tensor:
        return binary_ops.sub(_ensure_tensor(lhs, self), self)

    def __mul__(self, rhs: TensorValueLike) -> Tensor:
        return binary_ops.mul(self, _ensure_tensor(rhs, self))

    def __rmul__(self, lhs: TensorValueLike) -> Tensor:
        return binary_ops.mul(_ensure_tensor(lhs, self), self)

    def __truediv__(self, rhs: TensorValueLike) -> Tensor:
        return binary_ops.div(self, _ensure_tensor(rhs, self))

    def __rtruediv__(self, lhs: TensorValueLike) -> Tensor:
        return binary_ops.div(_ensure_tensor(lhs, self), self)

    def __matmul__(self, rhs: TensorValueLike) -> Tensor:
        return binary_ops.matmul(self, _ensure_tensor(rhs, self))

    def __rmatmul__(self, lhs: TensorValueLike) -> Tensor:
        return binary_ops.matmul(_ensure_tensor(lhs, self), self)


def _ensure_tensor(value: TensorValueLike, like: Tensor) -> Tensor:
    """Convert scalars to Tensors matching the reference tensor's dtype/device."""
    if isinstance(value, Tensor):
        return value
    return Tensor.constant(value, dtype=like.dtype, device=like.device)


# Re-export for backwards compatibility
__all__ = [
    "Tensor",
    "TensorImpl",
    "GRAPH",
    "defaults",
    "default_device",
    "default_dtype",
    "defaults_like",
    "get_topological_order",
    "print_computation_graph",
    "driver_tensor_type",
]
