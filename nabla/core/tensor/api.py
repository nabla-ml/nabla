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
# Import from new modules
from ..common.context import (
    defaults,
    default_device,
    default_dtype,
    defaults_like,
    _in_running_loop,
)
from .impl import TensorImpl
from ..graph.engine import GRAPH, driver_tensor_type




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
    def values(self) -> list[graph.TensorValue]:
        """Get all graph values as TensorValues.
        
        Raises RuntimeError if values are empty - call hydrate() first for realized tensors.
        """
        if not self._impl._values:
            if self._impl._storages:
                raise RuntimeError(
                    "Tensor has storages but no values. Call tensor.hydrate() first "
                    "to populate values from storages."
                )
            raise RuntimeError("Tensor has no values.")
        
        # Convert BufferValues to TensorValues
        return [
            v[...] if isinstance(v, graph.BufferValue) else v 
            for v in self._impl._values
        ]
    
    def hydrate(self) -> "Tensor":
        """Populate values from storages for realized tensors.
        
        Call this before accessing values on a tensor that was realized.
        Returns self for chaining.
        """
        if not self._impl._values and self._impl._storages:
            GRAPH.add_input(self)
        return self

    @property
    def _backing_value(self) -> driver.Tensor | graph.BufferValue | graph.TensorValue:
        return self._impl.primary_value

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
        """Keyword arguments passed to the operation that created this tensor."""
        return self._impl.op_kwargs or {}
    
    def trace(self) -> Tensor:
        """Enable tracing on this tensor for autograd."""
        self._impl.traced = True
        return self

    @property
    def dual(self) -> Tensor | None:
        """Get the dual (sharded/physical) tensor associated with this tensor."""
        if self._impl.dual is not None:
            return Tensor(impl=self._impl.dual)
        return None

    @dual.setter
    def dual(self, value: Tensor | None) -> None:
        """Set the dual (sharded/physical) associated with this tensor."""
        if value is None:
            self._impl.dual = None
        else:
            self._impl.dual = value._impl
    
    def shard(
        self,
        mesh: Any,
        dim_specs: list[Any],
        replicated_axes: set[str] | None = None,
    ) -> Tensor:
        """Shard this tensor across a device mesh.
        
        This handles both initial sharding of replicated tensors AND resharding
        of already sharded tensors.
        
        If the tensor has batch_dims (e.g. inside vmap), replicated specs are
        automatically prepended for those dimensions because `dim_specs` refers
        to logical dimensions.
        """
        from ...ops import communication as comm
        return comm.reshard(self, mesh, dim_specs, replicated_axes=replicated_axes)
    
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
        """
        from ...ops import communication as comm
        return comm.reshard(self, mesh, dim_specs, replicated_axes=replicated_axes)

    def with_sharding_constraint(
        self,
        mesh: Any,
        dim_specs: list[Any],
    ) -> Tensor:
        """Apply sharding constraint for global optimization.
        
        This sets a constraint that the GlobalShardingOptimizer will try to satisfy.
        It does NOT immediately reshard the tensor.
        
        Args:
            mesh: DeviceMesh to constrain to
            dim_specs: List of DimSpec for each dimension
        
        Returns:
            Self (for chaining)
        """
        from ..sharding.spec import ShardingSpec
        
        spec = ShardingSpec(mesh, dim_specs)
        # print(f"DEBUG: Setting sharding_constraint on TensorImpl {id(self._impl)}: {spec}")
        self._impl.sharding_constraint = spec
        return self

    @property
    def sharding(self) -> Any | None:
        """Get the current sharding specification of the tensor.
        
        Returns:
            ShardingSpec object if sharded, else None.
        """
        return self._impl.sharding

    @property
    def is_sharded(self) -> bool:
        return self._impl.is_sharded

    @property
    def local_shape(self) -> graph.Shape | None:
        """Local shape of this tensor (including batch dims).
        
        For sharded tensors, this is the shape of shard 0.
        For unsharded tensors, this equals global_shape.
        """
        return self._impl.physical_shape
    
    @property
    def global_shape(self) -> graph.Shape | None:
        """Global logical shape (excludes batch dims)."""
        return self._impl.physical_global_shape

    @property
    def physical_shape(self) -> graph.Shape | None:
        """DEPRECATED: Use local_shape instead.
        
        Physical shape of shard 0 (including batch dims at prefix).
        For sharded tensors, this is the LOCAL shard shape.
        For unsharded tensors, this equals the full shape.
        """
        return self._impl.physical_shape

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
        from ...ops import creation
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
        from ...ops import creation
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
        from ...ops import creation
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
        from ...ops import creation
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
        from ...ops import creation
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
        from ...ops import creation
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
        from ...ops import creation
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
        return self._impl.ndim

    @property
    def shape(self) -> graph.Shape:
        """Returns the global logical shape of the tensor (excludes batch dims).
        
        For sharded tensors, this returns the full tensor shape (not shard shape).
        Use local_shape to get the physical shape including batch dims.
        """
        # Use _impl.global_shape which properly excludes batch dims
        gs = self._impl.global_shape
        if gs is not None:
            return gs
        # Fallback for unrealized tensors
        return graph.Shape(self._backing_value.shape)
    
    def shard_shape(self, shard_idx: int = 0) -> graph.Shape:
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
        return self._impl.dtype

    @property
    def device(self) -> Device:
        return self._impl.device

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
        if self._value is None: return True
        mlir_value = self._value.to_mlir()
        return mlir_value.owner.parent_op == _core.Operation._from_cmlir(GRAPH.graph._mlir_op)

    # ===== Realization =====

    def realize(self) -> Tensor:
        """Force immediate realization (blocking)."""
        if not self.real:
            if not self._in_global_compute_graph:
                raise TypeError("Can't realize symbolic tensors.")
            self._impl.realize()
        return self

    # ===== Reduction Operations =====
    
    def sum(self, axis: int = 0, keepdims: bool = False) -> Tensor:
        from ...ops import reduction
        return reduction.reduce_sum(self, axis=axis, keepdims=keepdims)
        
    def mean(self, axis: int = 0, keepdims: bool = False) -> Tensor:
        from ...ops import reduction
        return reduction.mean(self, axis=axis, keepdims=keepdims)

    # ===== Data access =====

    def __bool__(self) -> bool:
        return bool(self.item())

    def __hash__(self):
        return id(self)

    def __dlpack__(self, stream: int | None = None):
        return self._impl.to_dlpack(stream=stream)

    def __dlpack_device__(self):
        return self._impl.to_dlpack_device()

    def __rich_repr__(self):
        yield "shape", self.shape
        yield "dtype", self.dtype
        yield "device", self.device

    def __repr__(self):
        if not self._in_global_compute_graph:
            return super().__repr__()
        self.realize()
        dt = self.driver_tensor.to(CPU())
        values = [dt[idx].item() for idx in dt._iterate_indices()]
        return f"{self.type}: [{', '.join(str(v) for v in values)}]"

    def __deepcopy__(self, memo: object) -> Tensor:
        return self

    def item(self):
        if self.num_elements() != 1:
            raise TypeError("Only single-element tensors can be converted to Python scalars")
        return self._impl.item()
    
    def to_numpy(self):
        """Convert tensor to numpy array."""
        return self._impl.to_numpy()

    def num_elements(self) -> int:
        elts = 1
        for dim in self.shape:
            elts *= int(dim)
        return elts

    # ===== Unary Operators =====

    def __neg__(self) -> Tensor:
        from ...ops import unary as unary_ops
        return unary_ops.neg(self)
        
    def __pos__(self) -> Tensor:
        return self
        
    def __abs__(self) -> Tensor:
        from ...ops import unary as unary_ops
        return unary_ops.abs(self)
    
    def __invert__(self) -> Tensor:
        # TODO: Implement bitwise not op
        raise NotImplementedError("Bitwise NOT not yet implemented")

    # ===== Operators using binary_ops =====

    def __add__(self, rhs: TensorValueLike) -> Tensor:
        from ...ops import binary as binary_ops
        return binary_ops.add(self, _ensure_tensor(rhs, self))

    def __radd__(self, lhs: TensorValueLike) -> Tensor:
        from ...ops import binary as binary_ops
        return binary_ops.add(_ensure_tensor(lhs, self), self)

    def __sub__(self, rhs: TensorValueLike) -> Tensor:
        from ...ops import binary as binary_ops
        return binary_ops.sub(self, _ensure_tensor(rhs, self))

    def __rsub__(self, lhs: TensorValueLike) -> Tensor:
        from ...ops import binary as binary_ops
        return binary_ops.sub(_ensure_tensor(lhs, self), self)

    def __mul__(self, rhs: TensorValueLike) -> Tensor:
        from ...ops import binary as binary_ops
        return binary_ops.mul(self, _ensure_tensor(rhs, self))

    def __rmul__(self, lhs: TensorValueLike) -> Tensor:
        from ...ops import binary as binary_ops
        return binary_ops.mul(_ensure_tensor(lhs, self), self)

    def __truediv__(self, rhs: TensorValueLike) -> Tensor:
        from ...ops import binary as binary_ops
        return binary_ops.div(self, _ensure_tensor(rhs, self))

    def __rtruediv__(self, lhs: TensorValueLike) -> Tensor:
        from ...ops import binary as binary_ops
        return binary_ops.div(_ensure_tensor(lhs, self), self)

    def __matmul__(self, rhs: TensorValueLike) -> Tensor:
        from ...ops import binary as binary_ops
        return binary_ops.matmul(self, _ensure_tensor(rhs, self))

    def __rmatmul__(self, lhs: TensorValueLike) -> Tensor:
        from ...ops import binary as binary_ops
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
]
