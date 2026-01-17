# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""TensorImpl: Complete computation graph node containing all tensor internals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max import driver, graph
from max.driver import Device, CPU
from max.dtype import DType

if TYPE_CHECKING:
    from ...ops import Operation
    from ..sharding import ShardingSpec
    from ..graph.tracing import OutputRefs
    
    # Cyclic import prevention
    from ..sharding.spec import compute_global_shape


class TensorImpl:
    """Complete computation graph node containing all tensor internals.
    
    TensorImpl holds both the actual data storage and the autograd graph
    structure. When `traced=True`, parent references are accessible via
    the shared OutputRefs instance.
    
    Attributes:
        _values: List of graph values for lazy execution (one per shard).
        _storages: List of realized driver.Tensors (one per shard), or None if unrealized.
        sharding: Sharding specification (None = unsharded, single shard).
        traced: Whether this node is part of a traced computation graph.
        tangent: Tangent TensorImpl for JVP (forward-mode autodiff).
        cotangent: Cotangent TensorImpl for VJP (reverse-mode autodiff).
        batch_dims: Number of batch dimensions (always prefix of physical shape).
        output_refs: Shared OutputRefs instance (holds op, op_args, op_kwargs).
        output_index: Position among sibling outputs (0-indexed).
    
    Properties:
        parents: List of parent TensorImpls (derived from output_refs.op_args).
        op: The operation that created this tensor (from output_refs).
        op_name: Name of the operation (from output_refs).
        dual: Reference to the dual (sharded) TensorImpl for replay/transformations.
    """
    __slots__ = ('_values', '_storages', 'sharding', 'sharding_constraint', 'traced',
                 'tangent', 'cotangent', 'dual', 'batch_dims', 'output_refs', 'output_index', 
                 '__weakref__')
    
    _values: list[graph.BufferValue | graph.TensorValue]
    _storages: list[driver.Tensor] | None
    sharding: object | None
    sharding_constraint: object | None
    traced: bool
    tangent: TensorImpl | None
    cotangent: TensorImpl | None
    dual: TensorImpl | None
    batch_dims: int
    output_refs: OutputRefs | None
    output_index: int
    output_index: int
    
    def __init__(
        self,
        storages: driver.Tensor | list[driver.Tensor] | None = None,
        values: graph.BufferValue | graph.TensorValue | list[graph.BufferValue | graph.TensorValue] | None = None,
        traced: bool = False,
        batch_dims: int = 0,
        sharding_constraint: ShardingSpec | None = None,
    ):
        self._values = values if isinstance(values, list) else ([values] if values else [])
        self._storages = storages if isinstance(storages, list) else ([storages] if storages else None)
        
        self.sharding = None  # Always start unsharded
        self.sharding_constraint = sharding_constraint
        self.traced = traced
        self.tangent = None
        self.cotangent = None
        self.dual = None
        self.batch_dims = batch_dims
        self.output_refs = None
        self.output_index = 0

    def _validate_sharding(self) -> None:
        """Validate consistency of shards and sharding spec."""
        n_vals = len(self._values)
        n_stores = len(self._storages) if self._storages else 0
        
        if n_vals > 0 and n_stores > 0 and n_vals != n_stores:
            raise ValueError(f"Shard count mismatch: {n_vals} values vs {n_stores} storages")
        
        if self.sharding is None:
            if n_vals > 1: raise ValueError(f"Multiple values ({n_vals}) without sharding spec")
            if n_stores > 1: raise ValueError(f"Multiple storages ({n_stores}) without sharding spec")
    
    @property
    def is_realized(self) -> bool:
        return self._storages is not None and len(self._storages) > 0
    
    @property
    def num_shards(self) -> int:
        if self._storages is not None: return len(self._storages)
        return len(self._values) if self._values else 1
    
    @property
    def is_sharded(self) -> bool:
        return self.sharding is not None
    
    @property
    def op(self) -> Operation | None:
        return self.output_refs.op if self.output_refs else None
    
    @property
    def op_kwargs(self) -> dict[str, Any] | None:
        return self.output_refs.op_kwargs if self.output_refs else None

    @property
    def op_name(self) -> str | None:
        if not self.output_refs: return None
        op = self.output_refs.op
        return getattr(op, 'name', getattr(op, '__name__', None))
    
    @property
    def parents(self) -> list[TensorImpl]:
        """Get parent TensorImpls from op_args."""
        if self.output_refs is None: return []
        from ..common import pytree
        return [arg for arg in pytree.tree_leaves(self.output_refs.op_args) if isinstance(arg, TensorImpl)]
    
    @property
    def is_leaf(self) -> bool:
        return len(self.parents) == 0
    
    def physical_local_shape(self, shard_idx: int = 0) -> graph.Shape | None:
        """Storage shape for a specific shard (includes batch dims)."""
        if self._storages and shard_idx < len(self._storages):
            return graph.Shape(self._storages[shard_idx].shape)
        if self._values and shard_idx < len(self._values):
            return self._values[shard_idx].type.shape
        return None

    def logical_local_shape(self, shard_idx: int = 0) -> graph.Shape | None:
        """Logical shape for a specific shard (excludes batch dims)."""
        physical = self.physical_local_shape(shard_idx)
        if physical is None or self.batch_dims == 0:
            return physical
        return graph.Shape(physical[self.batch_dims:])
    
    @property
    def physical_shape(self) -> graph.Shape | None:
        """Storage shape of shard 0 (includes batch dims)."""
        return self.physical_local_shape(0)
    
    @property
    def logical_shape(self) -> graph.Shape | None:
        """Logical local shape of shard 0 (excludes batch dims)."""
        return self.logical_local_shape(0)
    
    @property
    def global_shape(self) -> graph.Shape | None:
        """Global logical shape (excludes batch dims)."""
        phys = self.physical_global_shape
        if phys is None: return None
        
        if self.batch_dims > 0:
            return graph.Shape(phys[self.batch_dims:])
        return phys
    
    @property
    def physical_global_shape(self) -> graph.Shape | None:
        """Global physical shape (includes batch dims)."""
        # If not sharded, physical global = physical local
        if not self.sharding:
            return self.local_shape
            
        # Call into centralized logic in spec.py
        # Pass all shard shapes (if available) to handle uneven sharding correctly
        local = self.physical_shape
        if local is None: return None
        
        shard_shapes = [tuple(int(d) for d in v.type.shape) for v in self._values] if self._values else None

        from ..sharding.spec import compute_global_shape
        global_ints = compute_global_shape(tuple(int(d) for d in local), self.sharding, shard_shapes=shard_shapes)
        return graph.Shape(global_ints)
    
    @property
    def global_shape_ints(self) -> tuple[int, ...] | None:
        shape = self.global_shape
        return tuple(int(d) for d in shape) if shape is not None else None
    
    @property
    def ndim(self) -> int | None:
        shape = self.global_shape
        return len(shape) if shape is not None else None
    
    @property
    def batch_shape(self) -> graph.Shape | None:
        physical = self.local_shape
        if physical is None or self.batch_dims == 0: return None
        return graph.Shape(physical[:self.batch_dims])
    
    @property
    def local_shape(self) -> graph.Shape | None:
        return self.physical_local_shape(0)

    def __repr__(self) -> str:
        shards_str = f", shards={self.num_shards}" if self.is_sharded else ""
        return f"TensorImpl(op={self.op_name}, traced={self.traced}, parents={len(self.parents)}, batch_dims={self.batch_dims}{shards_str})"
    

    
    def get_unrealized_shape(self) -> graph.Shape:
        if not self._values: raise RuntimeError("Internal error: _values missing")
        return self._values[0].type.shape
    
    def get_unrealized_dtype(self) -> DType:
        if not self._values: raise RuntimeError("Internal error: _values missing")
        return self._values[0].type.dtype
    
    def get_realized_shape(self) -> graph.Shape:
        if self._values: return self._values[0].type.shape
        if self._storages: return graph.Shape(self._storages[0].shape)
        if self.sharding:
             # Try to compute if we have sharding but no values? Rare edge case.
             pass
        raise RuntimeError("No shape source available")
    
    def get_realized_dtype(self) -> DType:
        if self._values: return self._values[0].type.dtype
        if self._storages: return self._storages[0].dtype
        raise RuntimeError("No dtype source available")

    @property
    def primary_value(self) -> driver.Tensor | graph.BufferValue | graph.TensorValue:
        if self._storages: return self._storages[0]
        if self._values: return self._values[0]
        raise RuntimeError("Tensor has no storage and no values")

    @property
    def dtype(self) -> DType:
        try:
            return self.primary_value.dtype
        except RuntimeError:
            raise

    @property
    def device(self) -> Device:
        try:
            device = self.primary_value.device
            return device if isinstance(device, Device) else device.to_device()
        except RuntimeError:
            raise

    # === Unified Logic Methods ===

    def gather(self) -> TensorImpl:
        """Gather shards into a single global tensor if needed (lazy)."""
        if self.is_sharded and self.sharding and not self.sharding.is_fully_replicated():
            from ...ops.communication import gather_all_axes
            from .api import Tensor
            # Wrapper needed for op
            gathered = gather_all_axes(Tensor(impl=self))
            return gathered._impl
        return self

    def realize(self) -> None:
        """Trigger computation internally."""
        if self.is_realized:
            return
        from ..graph.engine import GRAPH
        from .api import Tensor
        # Wrapper needed for graph engine
        GRAPH.evaluate(Tensor(impl=self))

    def to_dlpack(self, stream: int | None = None):
        """Unified DLPack export."""
        t = self.gather()
        t.realize()
        if not t._storages:
             raise RuntimeError("Failed to realize tensor for DLPack export")
        return t._storages[0].__dlpack__(stream=stream)
        
    def to_dlpack_device(self):
        """Unified DLPack device export."""
        t = self.gather()
        t.realize()
        if not t._storages:
             raise RuntimeError("Failed to realize tensor for DLPack device export")
        return t._storages[0].__dlpack_device__()

    def to_numpy(self):
        """Unified NumPy export."""
        t = self.gather()
        t.realize()
        if not t._storages:
             raise RuntimeError("Failed to realize tensor for NumPy export")
        return t._storages[0].to(CPU()).to_numpy()
    
    def item(self):
        """Unified item access."""
        t = self.gather()
        t.realize()
        if not t._storages:
             raise RuntimeError("Failed to realize tensor for item access")
        return t._storages[0].to(CPU()).item()
