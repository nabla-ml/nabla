# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Experimental tensor operations with eager execution."""

from __future__ import annotations

from typing import Any

try:
    from rich.pretty import pretty_repr
except ImportError:

    def pretty_repr(obj, **kwargs):
        return repr(obj)


from max import driver, graph
from max.driver import CPU, Device, DLPackArray
from max.dtype import DType
from max.graph import ShapeLike, TensorValueLike, ops
from max.graph.ops.constant import NestedArray, Number
from max.graph.value import HasTensorValue

from ..common.context import (
    default_device,
    default_dtype,
    defaults,
    defaults_like,
)
from ..graph.engine import DEBUG_LAZY_EVAL, GRAPH, driver_tensor_type
from .impl import TensorImpl


class Tensor(DLPackArray, HasTensorValue):
    """Multi-dimensional array with eager execution and automatic compilation."""

    _impl: TensorImpl
    _real: bool = False

    def __init__(
        self,
        *,
        buffers: driver.Buffer | None = None,
        value: graph.BufferValue | graph.TensorValue | None = None,
        impl: TensorImpl | None = None,
        is_traced: bool = False,
    ):
        if impl is not None:
            self._impl = impl
        else:
            assert buffers is not None or value is not None
            self._impl = TensorImpl(bufferss=buffers, values=value, is_traced=is_traced)

        if self._impl._graph_values and self._impl.graph_values_epoch == -1:
            self._impl.graph_values_epoch = GRAPH.epoch

        self.real = self._impl.is_realized

    @property
    def _buffers(self) -> list[driver.Buffer] | None:
        return self._impl._buffers

    @_buffers.setter
    def _buffers(self, value: list[driver.Buffer] | None) -> None:
        self._impl._buffers = value

    @property
    def buffers(self) -> driver.Buffer | None:
        if self._impl._buffers and len(self._impl._buffers) > 0:
            return self._impl._buffers[0]
        return None

    @buffers.setter
    def buffers(self, value: driver.Buffer | None) -> None:
        if value is None:
            self._impl._buffers = None
        else:
            self._impl._buffers = [value]

    @property
    def _value(self) -> graph.BufferValue | graph.TensorValue | None:
        if self._impl._graph_values and len(self._impl._graph_values) > 0:
            if self._impl.graph_values_epoch != GRAPH.epoch:
                if DEBUG_LAZY_EVAL:
                    print(
                        f"[LAZY DEBUG] Clearing stale _value for tensor {id(self)} "
                        f"(epoch: {self._impl.graph_values_epoch} != {GRAPH.epoch})"
                    )
                self._impl._graph_values = []
                return None
            return self._impl._graph_values[0]
        return None

    @_value.setter
    def _value(self, value: graph.BufferValue | graph.TensorValue | None) -> None:
        if value is None:
            self._impl._graph_values = []
        else:
            self._impl._graph_values = [value]
        self._impl.graph_values_epoch = GRAPH.epoch

    @property
    def _graph_values(self) -> list[graph.BufferValue | graph.TensorValue]:
        if self._impl.graph_values_epoch != GRAPH.epoch:
            if DEBUG_LAZY_EVAL:
                print(
                    f"[LAZY DEBUG] Clearing stale _graph_values for tensor {id(self)} "
                    f"(epoch: {self._impl.graph_values_epoch} != {GRAPH.epoch})"
                )
            self._impl._graph_values = []
        return self._impl._graph_values

    @_graph_values.setter
    def _graph_values(self, value: list[graph.BufferValue | graph.TensorValue]) -> None:
        self._impl._graph_values = value
        self._impl.graph_values_epoch = GRAPH.epoch

    @property
    def values(self) -> list[graph.TensorValue]:
        """Get all graph values as TensorValues; error if empty."""

        if self._impl.graph_values_epoch != GRAPH.epoch:
            if DEBUG_LAZY_EVAL:
                print(
                    f"[LAZY DEBUG] Clearing stale values for tensor {id(self)} "
                    f"(epoch: {self._impl.graph_values_epoch} != {GRAPH.epoch})"
                )
            self._impl._graph_values = []

        if not self._impl._graph_values:
            if self._impl._buffers:

                self.hydrate()

        if not self._impl._graph_values:
            print(f"ERROR: Tensor {id(self)} values check failed.")
            print(f"  Sharding: {self.sharding}")
            print(f"  Realized: {self.is_realized}")
            print(
                f"  Bufferss: {len(self._impl._buffers) if self._impl._buffers else 0}"
            )
            print(f"  Values Epoch: {self._impl.graph_values_epoch}")
            print(f"  GRAPH.epoch: {GRAPH.epoch}")
            raise RuntimeError(
                f"Tensor {id(self)} has no values (epoch={GRAPH.epoch}, impl_epoch={self._impl.graph_values_epoch})."
            )

        return [
            v[...] if isinstance(v, graph.BufferValue) else v
            for v in self._impl._graph_values
        ]

    def hydrate(self) -> Tensor:
        """Populate graph values from buffers for realized tensors.
        
        If the tensor is already registered as a graph input, uses that.
        In EAGER_MAX_GRAPH mode, adds buffer data as a constant for intermediate
        tensors accessed during eager graph building.
        """
        from ... import config as nabla_config
        
        if not self._impl._graph_values and self._impl._buffers:
            # Check if this tensor is already registered as a graph input
            if any(t is self for t in GRAPH._input_refs):
                # Already registered - just need to set up graph values
                # This shouldn't normally happen, but handle it gracefully
                GRAPH.add_input(self)
            elif any(t._impl._buffers and t._impl._buffers[0] is self._impl._buffers[0] 
                     for t in GRAPH._input_refs):
                # A different tensor object but same underlying buffer is already an input
                # Find it and copy its graph values
                for t in GRAPH._input_refs:
                    if t._impl._buffers and t._impl._buffers[0] is self._impl._buffers[0]:
                        self._impl._graph_values = t._impl._graph_values
                        self._impl.graph_values_epoch = t._impl.graph_values_epoch
                        break
            elif nabla_config.EAGER_MAX_GRAPH:
                # Only add as constant in EAGER mode (compile tracing)
                GRAPH.add_constant(self)
        return self

    @property
    def _backing_value(self) -> driver.Buffer | graph.BufferValue | graph.TensorValue:
        return self._impl.primary_value

    @property
    def is_traced(self) -> bool:
        return self._impl.is_traced

    @is_traced.setter
    def is_traced(self, value: bool) -> None:
        self._impl.is_traced = value

    @property
    def batch_dims(self) -> int:
        """Number of batch dimensions."""
        return self._impl.batch_dims

    @batch_dims.setter
    def batch_dims(self, value: int) -> None:
        self._impl.batch_dims = value

    @property
    def op_kwargs(self) -> dict[str, Any]:
        """Kwargs passed to the creating operation."""
        return self._impl.op_kwargs or {}

    def trace(self) -> Tensor:
        """Enable tracing on this tensor for autograd."""
        self._impl.is_traced = True
        return self

    @property
    def dual(self) -> Tensor | None:
        """Get the dual (sharded/physical) tensor."""
        if self._impl.dual is not None:
            return Tensor(impl=self._impl.dual)
        return None

    @dual.setter
    def dual(self, value: Tensor | None) -> None:
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
        """Shard this tensor across a device mesh, handling resharding and vmap batch dims."""
        from ...ops import communication as comm

        return comm.reshard(self, mesh, dim_specs, replicated_axes=replicated_axes)

    def with_sharding(
        self,
        mesh: Any,
        dim_specs: list[Any],
        replicated_axes: set[str] | None = None,
    ) -> Tensor:
        """Apply sharding constraint, resharding if needed."""
        from ...ops import communication as comm

        return comm.reshard(self, mesh, dim_specs, replicated_axes=replicated_axes)

    def with_sharding_constraint(
        self,
        mesh: Any,
        dim_specs: list[Any],
        replicated_axes: set[str] | None = None,
    ) -> Tensor:
        """Apply sharding constraint for global optimization; no immediate resharding."""
        from ..sharding.spec import ShardingSpec

        spec = ShardingSpec(mesh, dim_specs)
        self._impl.sharding_constraint = spec
        return self

    @property
    def sharding(self) -> Any | None:
        """Get the current sharding specification."""
        return self._impl.sharding

    @sharding.setter
    def sharding(self, value: Any) -> None:
        self._impl.sharding = value

    @property
    def is_sharded(self) -> bool:
        return self._impl.is_sharded

    @property
    def num_shards(self) -> int:
        return self._impl.num_shards

    @property
    def is_realized(self) -> bool:
        return self._impl.is_realized

    @property
    def tangent(self) -> TensorImpl | None:
        return self._impl.tangent

    @tangent.setter
    def tangent(self, value: TensorImpl | None) -> None:
        self._impl.tangent = value

    @property
    def batch_shape(self) -> graph.Shape | None:
        return self._impl.batch_shape

    def physical_local_shape(self, shard_idx: int = 0) -> graph.Shape | None:
        return self._impl.physical_local_shape(shard_idx)

    def logical_local_shape(self, shard_idx: int = 0) -> graph.Shape | None:
        return self._impl.logical_local_shape(shard_idx)

    @property
    def local_shape(self) -> graph.Shape | None:
        """Local shape of this tensor (shard 0), including batch dims."""
        return self._impl.physical_shape

    @property
    def global_shape(self) -> graph.Shape | None:
        """Global logical shape (excludes batch dims)."""
        return self._impl.physical_global_shape

    @property
    def physical_global_shape(self) -> graph.Shape | None:
        """Alias for global_shape."""
        return self._impl.physical_global_shape

    @property
    def physical_shape(self) -> graph.Shape | None:
        """DEPRECATED: Use local_shape instead."""
        return self._impl.physical_shape

    @classmethod
    def from_graph_value(cls, value: graph.Value) -> Tensor:
        if not isinstance(value, (graph.TensorValue, graph.BufferValue)):
            raise TypeError(f"{value=} must be a tensor or buffer value")
        return cls(value=value)

    @classmethod
    def _create_unsafe(cls, **kwargs: Any) -> Tensor:
        """Internal factory to create Tensor directly from TensorImpl args."""
        return cls(impl=TensorImpl(**kwargs))

    @classmethod
    def from_dlpack(cls, array: DLPackArray) -> Tensor:
        """Import tensor from DLPack-compatible array.

        The resulting tensor will be placed on the default device. If the
        source array is on a different device, it will be transferred.

        Args:
            array: DLPack-compatible array (numpy, torch, jax, etc.)

        Returns:
            Tensor on the default device
        """
        if isinstance(array, Tensor):
            return array

        # Import from DLPack
        buffer = driver.Buffer.from_dlpack(array)

        # Get default device (using defaults helper)
        from ..common.context import defaults

        _, default_dev = defaults()

        # Transfer to default device if needed
        if buffer.device != default_dev:
            buffer = buffer.to(default_dev)

        return Tensor(buffers=buffer)

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
        is_traced: bool = False,
    ) -> Tensor:
        from ...ops import creation

        return creation.full(
            shape, value, dtype=dtype, device=device, is_traced=is_traced
        )

    @classmethod
    def zeros(
        cls,
        shape: ShapeLike,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
        is_traced: bool = False,
    ) -> Tensor:
        from ...ops import creation

        return creation.zeros(shape, dtype=dtype, device=device, is_traced=is_traced)

    @classmethod
    def ones(
        cls,
        shape: ShapeLike,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
        is_traced: bool = False,
    ) -> Tensor:
        from ...ops import creation

        return creation.ones(shape, dtype=dtype, device=device, is_traced=is_traced)

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

    normal = gaussian

    @property
    def type(self) -> graph.TensorType:
        value = self._backing_value
        t = (
            driver_tensor_type(value)
            if isinstance(value, driver.Buffer)
            else value.type
        )
        return t.as_tensor() if isinstance(t, graph.BufferType) else t

    @property
    def rank(self) -> int:
        return self._impl.ndim

    @property
    def shape(self) -> graph.Shape:
        """Returns the global logical shape of the tensor (excludes batch dims)."""

        gs = self._impl.global_shape
        if gs is not None:
            return gs

        return graph.Shape(self._backing_value.shape)

    def shard_shape(self, shard_idx: int = 0) -> graph.Shape:
        """Returns the shape of a specific shard."""
        local = self._impl.logical_local_shape(shard_idx)
        if local is not None:
            return graph.Shape(local)
        return self.shape

    @property
    def dtype(self) -> DType:
        return self._impl.dtype

    @property
    def device(self) -> Device:
        return self._impl.device

    @property
    def driver_tensor(self) -> driver.Buffer:
        if (buffers := self.buffers) is None:
            raise TypeError("Can't get driver tensor for symbolic tensor")
        return buffers

    @property
    def real(self) -> bool:
        return self._impl.is_realized

    @real.setter
    def real(self, real: bool) -> None:
        if not real and self._in_global_compute_graph:
            GRAPH.add_unrealized(self._impl)
        self._real = real

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
        return mlir_value.owner.parent_op == _core.Operation._from_cmlir(
            GRAPH.graph._mlir_op
        )

    def gather(self) -> Tensor:
        """Gather shards into a single global tensor if needed (lazy)."""
        if (
            self._impl.is_sharded
            and self._impl.sharding
            and not self._impl.sharding.is_fully_replicated()
        ):
            from ...ops.communication import gather_all_axes

            gathered = gather_all_axes(self)
            return gathered
        return self

    def realize(self) -> Tensor:
        """Force immediate realization (blocking)."""
        if not self.real:
            if not self._in_global_compute_graph:
                raise TypeError("Can't realize symbolic tensors.")

            if self._impl.is_realized:
                return self
            from ..graph.engine import GRAPH

            GRAPH.evaluate(self)

        return self

    def cpu(self) -> Tensor:
        """Move tensor to CPU, gathering shards if needed.

        For sharded tensors, this first gathers all shards to a single device,
        then transfers to CPU. For unsharded tensors, it returns self if already
        on CPU, otherwise creates a new tensor on CPU.

        Returns:
            Tensor on CPU with all data gathered.
        """
        from max.driver import CPU as CPUDevice

        # If already on CPU and not sharded, return as-is
        if not self.is_sharded and str(self.device).startswith("Device(type=cpu"):
            return self

        # If sharded, gather first
        t = self.gather() if self.is_sharded else self

        # Realize to ensure we have buffers
        t.realize()

        # Create new tensor on CPU using numpy as intermediate
        data = t.to_numpy()  # This already moves to CPU
        return Tensor.from_dlpack(data)

    def sum(self, axis: int = 0, keepdims: bool = False) -> Tensor:
        from ...ops import reduction

        return reduction.reduce_sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: int = 0, keepdims: bool = False) -> Tensor:
        from ...ops import reduction

        return reduction.mean(self, axis=axis, keepdims=keepdims)

    def __bool__(self) -> bool:
        return bool(self.item())

    def __hash__(self):
        return id(self)

    def __dlpack__(self, stream: int | None = None):
        """Unified DLPack export."""
        t = self.gather()
        t.realize()
        if not t._impl._buffers:
            raise RuntimeError("Failed to realize tensor for DLPack export")
        return t._impl._buffers[0].__dlpack__(stream=stream)

    def __dlpack_device__(self):
        """Unified DLPack device export."""
        t = self.gather()
        t.realize()
        if not t._impl._buffers:
            raise RuntimeError("Failed to realize tensor for DLPack device export")
        return t._impl._buffers[0].__dlpack_device__()

    def __rich_repr__(self):
        yield "shape", self.shape
        yield "dtype", self.dtype
        yield "device", self.device

    def __repr__(self):
        """Unambiguous representation (triggers realization)."""
        self.realize()
        content = self._impl.format_metadata(include_data=True)
        if "\n" in content:
            # Multi-line: indent for premium look
            indented = "\n".join("  " + line for line in content.split("\n"))
            return f"Tensor(\n{indented}\n)"
        return f"Tensor({content})"

    def __str__(self):
        """Readable representation (triggers realization)."""
        return self.__repr__()

    def __deepcopy__(self, memo: object) -> Tensor:
        return self

    def item(self):
        if self.num_elements() != 1:
            raise TypeError(
                "Only single-element tensors can be converted to Python scalars"
            )
        """Unified item access."""
        t = self.gather()
        t.realize()
        if not t._impl._buffers:
            raise RuntimeError("Failed to realize tensor for item access")
        return t._impl._buffers[0].to(CPU()).item()

    def to_numpy(self):
        """Convert tensor to numpy array."""
        t = self.gather()
        t.realize()
        if not t._impl._buffers:
            raise RuntimeError("Failed to realize tensor for NumPy export")
        return t._impl._buffers[0].to(CPU()).to_numpy()

    numpy = to_numpy

    @staticmethod
    def to_numpy_all(*tensors: Tensor) -> tuple:
        """Convert multiple tensors to numpy arrays in a single batched compilation.

        This is more efficient than calling `.to_numpy()` on each tensor individually,
        as it combines all gather and realize operations into a single compilation.

        Args:
            *tensors: Variable number of tensors to convert.

        Returns:
            Tuple of numpy arrays, one per input tensor.

        Example:
            >>> x_np, w_np, b_np = Tensor.to_numpy_all(x_grad, w_grad, b_grad)
        """
        from ..graph.engine import GRAPH

        # First, gather all tensors (lazy operations)
        gathered = [t.gather() for t in tensors]

        # Batch realize all gathered tensors
        unrealized = [g for g in gathered if not g.real]
        if unrealized:
            if len(unrealized) > 1:
                GRAPH.evaluate(unrealized[0], *unrealized[1:])
            else:
                GRAPH.evaluate(unrealized[0])

        # Convert to numpy
        results = []
        for g in gathered:
            if not g._impl._buffers:
                raise RuntimeError("Failed to realize tensor for NumPy export")
            results.append(g._impl._buffers[0].to(CPU()).to_numpy())

        return tuple(results)

    def num_elements(self) -> int:
        elts = 1
        for dim in self.shape:
            elts *= int(dim)
        return elts

    def __neg__(self) -> Tensor:
        from ...ops import unary as unary_ops

        return unary_ops.neg(self)

    def __pos__(self) -> Tensor:
        return self

    def __abs__(self) -> Tensor:
        from ...ops import unary as unary_ops

        return unary_ops.abs(self)

    def __invert__(self) -> Tensor:

        raise NotImplementedError("Bitwise NOT not yet implemented")

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


def realize_all(*tensors: Tensor) -> tuple[Tensor, ...]:
    """Realize multiple tensors in a single batched compilation.

    This is more efficient than calling `.realize()` on each tensor individually,
    as it combines all pending computations into a single graph compilation.

    Args:
        *tensors: Variable number of tensors to realize.

    Returns:
        Tuple of realized tensors (same tensors, now with computed values).

    Example:
        >>> w = ops.shard(w_data, mesh, spec)
        >>> b = ops.shard(b_data, mesh, spec)
        >>> w, b = nb.realize_all(w, b)  # Single compilation instead of two
    """
    from ..graph.engine import GRAPH

    # Filter to only unrealized tensors
    unrealized = [t for t in tensors if isinstance(t, Tensor) and not t.real]

    if unrealized:
        if len(unrealized) > 1:
            GRAPH.evaluate(unrealized[0], *unrealized[1:])
        else:
            GRAPH.evaluate(unrealized[0])

    return tensors


__all__ = [
    "Tensor",
    "TensorImpl",
    "GRAPH",
    "defaults",
    "default_device",
    "default_dtype",
    "defaults_like",
    "realize_all",
]
