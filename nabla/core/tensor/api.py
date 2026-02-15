# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Experimental tensor operations with eager execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, Union

if TYPE_CHECKING:
    import numpy as np
    from types import EllipsisType

    from max import driver, graph
    from max.driver import Device
    from max.dtype import DType
    from max.graph import Shape, TensorValue

    from ..sharding.spec import DeviceMesh, ShardingSpec
    from ..graph.tracing import OpNode


try:
    from rich.pretty import pretty_repr
except ImportError:

    def pretty_repr(obj: Any, **kwargs: Any) -> str:
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
        impl: "TensorImpl | None" = None,
        is_traced: bool = False,
    ) -> None:
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
            elif any(
                t._impl._buffers and t._impl._buffers[0] is self._impl._buffers[0]
                for t in GRAPH._input_refs
            ):
                # A different tensor object but same underlying buffer is already an input
                # Find it and copy its graph values
                for t in GRAPH._input_refs:
                    if (
                        t._impl._buffers
                        and t._impl._buffers[0] is self._impl._buffers[0]
                    ):
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
    def requires_grad(self) -> bool:
        """Whether this tensor requires gradient computation (PyTorch style)."""
        return self._impl.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._impl.requires_grad = value
        # When requiring grad, also enable tracing so ops record the graph
        if value:
            self._impl.is_traced = True

    def requires_grad_(self, value: bool = True) -> Tensor:
        """In-place style alias for setting requires_grad (PyTorch style)."""
        self.requires_grad = value
        return self

    @property
    def is_leaf(self) -> bool:
        """True if this tensor wasn't created by an operation (gradient leaf)."""
        return self._impl.is_leaf

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

    def detach(self) -> Tensor:
        """Returns a new Tensor, detached from the current graph (PyTorch style)."""
        # In Nabla, this just returns a new Tensor sharing the same buffers/values
        # but with is_traced=False.
        impl = self._impl
        new_impl = TensorImpl(
            bufferss=impl._buffers,
            values=impl._graph_values,
            is_traced=False,
            batch_dims=impl.batch_dims,
        )
        new_impl.sharding = impl.sharding
        return Tensor(impl=new_impl)

    @property
    def grad(self) -> Tensor | None:
        """Gradient of this tensor (populated by backward())."""
        if self._impl.cotangent is not None:
            return Tensor(impl=self._impl.cotangent)
        return None

    @grad.setter
    def grad(self, value: Tensor | None) -> None:
        if value is None:
            self._impl.cotangent = None
        else:
            self._impl.cotangent = value._impl

    def backward(
        self,
        gradient: Tensor | None = None,
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> None:
        """Compute gradients of this tensor w.r.t. graph leaves (PyTorch style).

        Populates .grad on all tensors with requires_grad=True that this tensor
        depends on. All gradients are batch-realized for efficiency.

        Args:
            gradient: Gradient w.r.t. this tensor. Required for non-scalar tensors.
            retain_graph: Unused (maintained for PyTorch API compatibility).
            create_graph: If True, graph of the derivatives will be constructed,
                allowing to compute higher order derivatives.

        Example:
            >>> x = nb.Tensor([1.0, 2.0, 3.0])
            >>> x.requires_grad = True
            >>> y = (x ** 2).sum()
            >>> y.backward()
            >>> print(x.grad)  # [2.0, 4.0, 6.0]
        """
        from ..autograd.utils import backward

        backward(self, cotangents=gradient, create_graph=create_graph)

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
        mesh: "DeviceMesh",
        dim_specs: list["ShardingSpec" | str | list[str] | None],
        replicated_axes: set[str] | None = None,
    ) -> Tensor:
        """Shard this tensor across a device mesh, handling resharding and vmap batch dims."""
        from ...ops import communication as comm

        return comm.reshard(self, mesh, dim_specs, replicated_axes=replicated_axes)

    def with_sharding(
        self,
        mesh: "DeviceMesh",
        dim_specs: list["ShardingSpec" | str | list[str] | None],
        replicated_axes: set[str] | None = None,
    ) -> Tensor:
        """Apply sharding constraint, resharding if needed."""
        from ...ops import communication as comm

        return comm.reshard(self, mesh, dim_specs, replicated_axes=replicated_axes)

    def with_sharding_constraint(
        self,
        mesh: "DeviceMesh",
        dim_specs: list[Any],
        replicated_axes: set[str] | None = None,
    ) -> Tensor:
        """Apply sharding constraint for global optimization; no immediate resharding."""
        from ..sharding.spec import ShardingSpec

        spec = ShardingSpec(mesh, dim_specs)
        self._impl.sharding_constraint = spec
        return self

    @property
    def sharding(self) -> "ShardingSpec | None":
        """Get the current sharding specification."""
        return self._impl.sharding

    @sharding.setter
    def sharding(self, value: "ShardingSpec | None") -> None:
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

    def physical_local_shape_ints(self, shard_idx: int = 0) -> tuple[int, ...] | None:
        """Int-tuple shape for a specific shard (avoids creating Shape/Dim objects)."""
        return self._impl.physical_local_shape_ints(shard_idx)

    @property
    def physical_global_shape_ints(self) -> tuple[int, ...] | None:
        """Fast global physical shape as int tuple (no Shape/Dim allocation)."""
        return self._impl.physical_global_shape_ints

    @property
    def global_shape_ints(self) -> tuple[int, ...] | None:
        """Fast global logical shape as int tuple (no Shape/Dim allocation)."""
        return self._impl.global_shape_ints

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

    def new_zeros(
        self,
        shape: ShapeLike,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        """Create a new tensor of zeros with same device/dtype as self by default."""
        return Tensor.zeros(
            shape, dtype=dtype or self.dtype, device=device or self.device
        )

    def new_ones(
        self,
        shape: ShapeLike,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        """Create a new tensor of ones with same device/dtype as self by default."""
        return Tensor.ones(
            shape, dtype=dtype or self.dtype, device=device or self.device
        )

    def new_full(
        self,
        shape: ShapeLike,
        fill_value: Number,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        """Create a new tensor filled with value with same device/dtype as self by default."""
        return Tensor.full(
            shape, fill_value, dtype=dtype or self.dtype, device=device or self.device
        )

    def new_empty(
        self,
        shape: ShapeLike,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        """Create a new uninitialized tensor (defaults to zeros in Nabla)."""
        return self.new_zeros(shape, dtype=dtype, device=device)

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
    def ndim(self) -> int:
        """Number of dimensions (PyTorch style)."""
        return self._impl.ndim

    def dim(self) -> int:
        """Alias for rank (PyTorch style)."""
        return self.rank

    @property
    def shape(self) -> graph.Shape:
        """Returns the global logical shape of the tensor (excludes batch dims)."""

        gs = self._impl.global_shape
        if gs is not None:
            return gs

        return graph.Shape(self._backing_value.shape)

    def size(self, dim: int | None = None) -> graph.Shape | int:
        """Returns the shape or size of a specific dimension (PyTorch style)."""
        if dim is not None:
            return self.shape[dim]
        return self.shape

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

    def cuda(self, device: int | str = 0) -> Tensor:
        """Move tensor to GPU (shortcut for PyTorch users)."""
        target = f"gpu:{device}" if isinstance(device, int) else device
        if not target.startswith("gpu"):
            target = f"gpu:{target}"
        return self.to(target)

    if TYPE_CHECKING:
        # Reduction/unary methods are generated at module load.
        def sum(
            self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
        ) -> Tensor: ...
        def mean(
            self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
        ) -> Tensor: ...
        def max(
            self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
        ) -> Tensor: ...
        def min(
            self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
        ) -> Tensor: ...
        def argmax(self, axis: int | None = None, keepdims: bool = False) -> Tensor: ...
        def argmin(self, axis: int | None = None, keepdims: bool = False) -> Tensor: ...
        def cumsum(self, axis: int) -> Tensor: ...
        def abs(self) -> Tensor: ...
        def exp(self) -> Tensor: ...
        def log(self) -> Tensor: ...
        def relu(self) -> Tensor: ...
        def sigmoid(self) -> Tensor: ...
        def softmax(self, axis: int = -1) -> Tensor: ...
        def logsoftmax(self, axis: int = -1) -> Tensor: ...
        def sqrt(self) -> Tensor: ...
        def tanh(self) -> Tensor: ...
        def acos(self) -> Tensor: ...
        def atanh(self) -> Tensor: ...
        def cos(self) -> Tensor: ...
        def erf(self) -> Tensor: ...
        def floor(self) -> Tensor: ...
        def is_inf(self) -> Tensor: ...
        def is_nan(self) -> Tensor: ...
        def log1p(self) -> Tensor: ...
        def rsqrt(self) -> Tensor: ...
        def silu(self) -> Tensor: ...
        def sin(self) -> Tensor: ...
        def trunc(self) -> Tensor: ...
        def gelu(self, approximate: str | bool = "none") -> Tensor: ...
        def round(self) -> Tensor: ...
        def cast(self, dtype: DType) -> Tensor: ...
        def __neg__(self) -> Tensor: ...
        def __abs__(self) -> Tensor: ...
        def __add__(self, rhs: TensorValueLike) -> Tensor: ...
        def __radd__(self, lhs: TensorValueLike) -> Tensor: ...
        def __sub__(self, rhs: TensorValueLike) -> Tensor: ...
        def __rsub__(self, lhs: TensorValueLike) -> Tensor: ...
        def __mul__(self, rhs: TensorValueLike) -> Tensor: ...
        def __rmul__(self, lhs: TensorValueLike) -> Tensor: ...
        def __truediv__(self, rhs: TensorValueLike) -> Tensor: ...
        def __rtruediv__(self, lhs: TensorValueLike) -> Tensor: ...
        def __matmul__(self, rhs: TensorValueLike) -> Tensor: ...
        def __rmatmul__(self, lhs: TensorValueLike) -> Tensor: ...
        def __pow__(self, rhs: TensorValueLike) -> Tensor: ...
        def __rpow__(self, lhs: TensorValueLike) -> Tensor: ...
        def __mod__(self, rhs: TensorValueLike) -> Tensor: ...
        def __rmod__(self, lhs: TensorValueLike) -> Tensor: ...
        def __eq__(self, rhs: TensorValueLike) -> Tensor: ...
        def __ne__(self, rhs: TensorValueLike) -> Tensor: ...
        def __lt__(self, rhs: TensorValueLike) -> Tensor: ...
        def __le__(self, rhs: TensorValueLike) -> Tensor: ...
        def __gt__(self, rhs: TensorValueLike) -> Tensor: ...
        def __ge__(self, rhs: TensorValueLike) -> Tensor: ...
        def __and__(self, rhs: TensorValueLike) -> Tensor: ...
        def __or__(self, rhs: TensorValueLike) -> Tensor: ...
        def __xor__(self, rhs: TensorValueLike) -> Tensor: ...

    def type_as(self, other: Tensor) -> Tensor:
        """Cast this tensor to the same dtype as `other`."""
        return self.cast(other.dtype)

    # --- View & Shape Operations ---

    def reshape(self, shape: ShapeLike) -> Tensor:
        from ...ops import view

        return view.reshape(self, shape)

    def view(self, *shape: int | ShapeLike) -> Tensor:
        """Alias for reshape() (PyTorch style)."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple, graph.Shape)):
            return self.reshape(shape[0])
        return self.reshape(list(shape))

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> Tensor:
        from ...ops import view

        return view.squeeze(self, axis=axis)

    def unsqueeze(self, axis: int) -> Tensor:
        from ...ops import view

        return view.unsqueeze(self, axis=axis)

    def swap_axes(self, axis1: int, axis2: int) -> Tensor:
        from ...ops import view

        return view.swap_axes(self, axis1=axis1, axis2=axis2)

    def transpose(self, axis1: int, axis2: int) -> Tensor:
        from ...ops import view

        return view.swap_axes(self, axis1=axis1, axis2=axis2)

    def permute(self, *order: int) -> Tensor:
        from ...ops import view

        if len(order) == 1 and isinstance(order[0], (tuple, list)):
            order = tuple(order[0])
        return view.permute(self, order=order)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        from ...ops import view

        return view.flatten(self, start_dim=start_dim, end_dim=end_dim)

    def broadcast_to(self, shape: ShapeLike) -> Tensor:
        from ...ops import view

        return view.broadcast_to(self, shape)

    def expand(self, *shape: int) -> Tensor:
        """Alias for broadcast_to (PyTorch style)."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            return self.broadcast_to(shape[0])
        return self.broadcast_to(list(shape))

    def flip(self, axis: int | tuple[int, ...]) -> Tensor:
        from ...ops import view

        return view.flip(self, axis=axis)

    @property
    def T(self) -> Tensor:
        """Transpose last two dimensions (PyTorch style)."""
        if self.rank < 2:
            return self
        return self.transpose(-1, -2)

    # --- Communication & Management ---

    def to(self, target: Device | str | DType) -> Tensor:
        """Move tensor to a device or cast to a dtype.

        Args:
            target: Target Device object, device string (e.g. 'cpu', 'gpu:0'), or DType.
        """
        if isinstance(target, DType):
            return self.cast(target)

        from ...ops import communication as comm

        return comm.to_device(self, target)

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

    def item(self) -> float | int | bool:
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

    def to_numpy(self) -> "np.ndarray":
        """Convert tensor to numpy array."""
        t = self.gather()
        t.realize()
        if not t._impl._buffers:
            raise RuntimeError("Failed to realize tensor for NumPy export")
        return t._impl._buffers[0].to(CPU()).to_numpy()

    numpy = to_numpy

    def tolist(self) -> list[Any]:
        """Convert tensor to a Python list (PyTorch style)."""
        return self.to_numpy().tolist()

    @staticmethod
    def to_numpy_all(*tensors: Tensor) -> tuple["np.ndarray", ...]:
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

    def numel(self) -> int:
        """Alias for num_elements() (PyTorch style)."""
        return self.num_elements()

    def __pos__(self) -> Tensor:
        return self

    def __invert__(self) -> Tensor:
        raise NotImplementedError("Bitwise NOT not yet implemented")

    # Numeric/comparison/logical dunders are generated at module load.

    # --- Indexing ---

    @property
    def at(self) -> "_TensorAtAccessor":
        """JAX-like functional indexed update accessor.

        Usage:
            x2 = x.at[idx].set(v)
            x3 = x.at[idx].add(v)
        """
        return _TensorAtAccessor(self)

    def __getitem__(self, key: int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...]) -> Tensor:
        """Basic slicing and integer indexing."""
        from ...ops import view

        start, size, squeeze_axes = _parse_basic_index(self, key)

        res = view.slice_tensor(self, start=tuple(start), size=tuple(size))
        if squeeze_axes:
            res = view.squeeze(res, axis=tuple(squeeze_axes))
        return res

    def __setitem__(
        self,
        key: int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...] | "Tensor",
        value: TensorValueLike,
    ) -> None:
        """PyTorch-like indexed update that mutates this Tensor object binding.

        The underlying update is represented via Nabla operations so history is kept.
        """
        updated = _apply_indexed_update(
            self,
            key,
            value,
            mode="set",
            prefer_inplace_buffers=True,
        )
        self._impl = updated._impl
        self.real = self._impl.is_realized


class _TensorAtAccessor:
    """Intermediate accessor for JAX-like x.at[idx].set/add."""

    def __init__(self, tensor: Tensor):
        self._tensor = tensor

    def __getitem__(
        self,
        key: int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...] | Tensor,
    ) -> "_TensorAtIndexer":
        return _TensorAtIndexer(self._tensor, key)


class _TensorAtIndexer:
    """Bound updater created by x.at[idx]."""

    def __init__(
        self,
        tensor: Tensor,
        key: int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...] | Tensor,
    ):
        self._tensor = tensor
        self._key = key

    def set(self, value: TensorValueLike) -> Tensor:
        return _apply_indexed_update(self._tensor, self._key, value, mode="set")

    def add(self, value: TensorValueLike) -> Tensor:
        return _apply_indexed_update(self._tensor, self._key, value, mode="add")


def _parse_basic_index(
    x: Tensor,
    key: int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...],
) -> tuple[list[int], list[int], list[int]]:
    shape = x.shape

    if not isinstance(key, tuple):
        key = (key,)

    if Ellipsis in key:
        if sum(1 for k in key if k is Ellipsis) > 1:
            raise ValueError("An index can only have a single ellipsis ('...')")

        ellipsis_idx = key.index(Ellipsis)
        num_expanded = len(shape) - (len(key) - 1)
        key = (
            key[:ellipsis_idx]
            + (slice(None),) * max(0, num_expanded)
            + key[ellipsis_idx + 1 :]
        )

    start: list[int] = []
    size: list[int] = []
    squeeze_axes: list[int] = []

    for i, k in enumerate(key):
        if i >= len(shape):
            break
        dim_size = int(shape[i])
        if isinstance(k, int):
            idx = k + dim_size if k < 0 else k
            start.append(idx)
            size.append(1)
            squeeze_axes.append(i)
        elif isinstance(k, slice):
            s_start = k.start if k.start is not None else 0
            if s_start < 0:
                s_start += dim_size
            s_stop = k.stop if k.stop is not None else dim_size
            if s_stop < 0:
                s_stop += dim_size
            s_step = k.step if k.step is not None else 1
            if s_step != 1:
                raise NotImplementedError("Slicing with step != 1 is not yet supported.")

            s_start = max(0, min(dim_size, s_start))
            s_stop = max(0, min(dim_size, s_stop))

            start.append(s_start)
            size.append(max(0, s_stop - s_start))
        else:
            raise TypeError(f"Invalid index type: {type(k)}")

    while len(start) < len(shape):
        idx = len(start)
        start.append(0)
        size.append(int(shape[idx]))

    return start, size, squeeze_axes


def _broadcast_updates_for_scatter(x: Tensor, indices: Tensor, updates: Tensor) -> Tensor:
    """Broadcast scalar updates for axis=0 scatter convenience."""
    if updates.shape == tuple(indices.shape):
        return updates
    if updates.numel() == 1:
        from ...ops import view

        target_shape = tuple(int(d) for d in indices.shape) + tuple(
            int(d) for d in x.shape[1:]
        )
        return view.broadcast_to(updates, target_shape)
    return updates


def _broadcast_updates_for_slice(size: tuple[int, ...], updates: Tensor) -> Tensor:
    target_shape = tuple(int(d) for d in size)
    if updates.shape == target_shape:
        return updates
    from ...ops import view

    return view.broadcast_to(updates, target_shape)


def _apply_indexed_update(
    x: Tensor,
    key: int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...] | Tensor,
    value: TensorValueLike,
    *,
    mode: str,
    prefer_inplace_buffers: bool = False,
) -> Tensor:
    from ...ops import view

    if isinstance(key, Tensor):
        indices = key
        updates = _ensure_tensor(value, x)
        updates = _broadcast_updates_for_scatter(x, indices, updates)

        if mode == "set":
            return view.scatter(x, indices, updates, axis=0)
        if mode == "add":
            gathered = view.gather(x, indices, axis=0)
            new_updates = gathered + updates
            return view.scatter(x, indices, new_updates, axis=0)
        raise ValueError(f"Unsupported indexed update mode: {mode}")

    start, size, _ = _parse_basic_index(x, key)
    updates = _ensure_tensor(value, x)
    updates = _broadcast_updates_for_slice(tuple(size), updates)

    if mode == "set":
        if prefer_inplace_buffers:
            return view.slice_update_inplace(
                x,
                updates,
                start=tuple(start),
                size=tuple(size),
            )
        return view.slice_update(x, updates, start=tuple(start), size=tuple(size))
    if mode == "add":
        current = view.slice_tensor(x, start=tuple(start), size=tuple(size))
        return view.slice_update(
            x,
            current + updates,
            start=tuple(start),
            size=tuple(size),
        )
    raise ValueError(f"Unsupported indexed update mode: {mode}")


def _ensure_tensor(value: TensorValueLike, like: Tensor) -> Tensor:
    """Convert scalars to Tensors matching the reference tensor's dtype/device."""
    if isinstance(value, Tensor):
        return value
    return Tensor.constant(value, dtype=like.dtype, device=like.device)


def _set_tensor_method(name: str, method: Any) -> None:
    method.__name__ = name
    method.__qualname__ = f"Tensor.{name}"
    setattr(Tensor, name, method)


def _make_reduce_axis_keepdims(op_name: str):
    def _method(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ):
        from ...ops import reduction

        op = getattr(reduction, op_name)
        return op(self, axis=axis, keepdims=keepdims)

    return _method


def _make_reduce_axis_optional(op_name: str):
    def _method(self, axis: int | None = None, keepdims: bool = False):
        from ...ops import reduction

        op = getattr(reduction, op_name)
        return op(self, axis=axis, keepdims=keepdims)

    return _method


def _make_reduce_axis_required(op_name: str):
    def _method(self, axis: int):
        from ...ops import reduction

        op = getattr(reduction, op_name)
        return op(self, axis=axis)

    return _method


def _make_unary_noargs(op_name: str):
    def _method(self):
        from ...ops import unary

        op = getattr(unary, op_name)
        return op(self)

    return _method


def _make_unary_axis(op_name: str, *, default_axis: int):
    def _method(self, axis: int = default_axis):
        from ...ops import unary

        op = getattr(unary, op_name)
        return op(self, axis=axis)

    return _method


def _make_unary_gelu():
    def _method(self, approximate: str | bool = "none"):
        from ...ops import unary

        return unary.gelu(self, approximate=approximate)

    return _method


def _make_unary_cast():
    def _method(self, dtype: DType):
        from ...ops import unary

        return unary.cast(self, dtype=dtype)

    return _method


def _make_binary_method(op_name: str, *, reverse: bool = False):
    def _method(self, other: TensorValueLike):
        from ...ops import binary as binary_ops

        op = getattr(binary_ops, op_name)
        if reverse:
            return op(_ensure_tensor(other, self), self)
        return op(self, _ensure_tensor(other, self))

    return _method


def _make_comparison_method(op_name: str):
    def _method(self, other: TensorValueLike):
        from ...ops import comparison as comp_ops

        op = getattr(comp_ops, op_name)
        return op(self, _ensure_tensor(other, self))

    return _method


def _bind_tensor_generated_methods() -> None:
    for name, op_name in {
        "sum": "reduce_sum",
        "mean": "mean",
        "max": "reduce_max",
        "min": "reduce_min",
    }.items():
        _set_tensor_method(name, _make_reduce_axis_keepdims(op_name))

    for name, op_name in {"argmax": "argmax", "argmin": "argmin"}.items():
        _set_tensor_method(name, _make_reduce_axis_optional(op_name))

    _set_tensor_method("cumsum", _make_reduce_axis_required("cumsum"))

    for name in [
        "abs",
        "exp",
        "log",
        "relu",
        "sigmoid",
        "sqrt",
        "tanh",
        "acos",
        "atanh",
        "cos",
        "erf",
        "floor",
        "is_inf",
        "is_nan",
        "log1p",
        "rsqrt",
        "silu",
        "sin",
        "trunc",
        "round",
    ]:
        _set_tensor_method(name, _make_unary_noargs(name))

    _set_tensor_method("__neg__", _make_unary_noargs("neg"))
    _set_tensor_method("__abs__", _make_unary_noargs("abs"))

    _set_tensor_method("softmax", _make_unary_axis("softmax", default_axis=-1))
    _set_tensor_method("logsoftmax", _make_unary_axis("logsoftmax", default_axis=-1))
    _set_tensor_method("gelu", _make_unary_gelu())
    _set_tensor_method("cast", _make_unary_cast())

    for name, op_name, reverse in [
        ("__add__", "add", False),
        ("__radd__", "add", True),
        ("__sub__", "sub", False),
        ("__rsub__", "sub", True),
        ("__mul__", "mul", False),
        ("__rmul__", "mul", True),
        ("__truediv__", "div", False),
        ("__rtruediv__", "div", True),
        ("__matmul__", "matmul", False),
        ("__rmatmul__", "matmul", True),
        ("__pow__", "pow", False),
        ("__rpow__", "pow", True),
        ("__mod__", "mod", False),
        ("__rmod__", "mod", True),
    ]:
        _set_tensor_method(name, _make_binary_method(op_name, reverse=reverse))

    for name, op_name in {
        "__eq__": "equal",
        "__ne__": "not_equal",
        "__lt__": "less",
        "__le__": "less_equal",
        "__gt__": "greater",
        "__ge__": "greater_equal",
        "__and__": "logical_and",
        "__or__": "logical_or",
        "__xor__": "logical_xor",
    }.items():
        _set_tensor_method(name, _make_comparison_method(op_name))


_bind_tensor_generated_methods()


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
