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

"""TensorImpl: Complete computation graph node containing all tensor internals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max import driver, graph
from max.driver import Device
from max.dtype import DType

if TYPE_CHECKING:
    from .ops import Operation
    from .sharding import ShardingSpec
    from .tracing import OutputRefs


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
    """
    __slots__ = ('_values', '_storages', 'sharding', 'traced','tangent', 'cotangent', 'batch_dims', 'output_refs', 'output_index', 'cached_shape', 'cached_dtype', 'cached_device', '__weakref__')
    
    _values: list[graph.BufferValue | graph.TensorValue]
    _storages: list[driver.Tensor] | None
    sharding: object | None
    traced: bool
    tangent: TensorImpl | None   # For JVP (forward-mode autodiff)
    cotangent: TensorImpl | None # For VJP (reverse-mode autodiff)
    batch_dims: int
    output_refs: OutputRefs | None   # Shared OutputRefs instance (set by Operation.__call__)
    output_index: int            # Position among sibling outputs (0-indexed)
    # Cached metadata for sharding (survives graph consumption)
    cached_shape: tuple[int, ...] | None
    cached_dtype: DType | None
    cached_device: Device | None
    
    def __init__(
        self,
        storages: driver.Tensor | list[driver.Tensor] | None = None,
        values: graph.BufferValue | graph.TensorValue | list[graph.BufferValue | graph.TensorValue] | None = None,
        traced: bool = False,
        batch_dims: int = 0,
        sharding: ShardingSpec | None = None,
    ):
        # Normalize values to list
        if values is None:
            self._values = []
        elif isinstance(values, list):
            self._values = values
        else:
            self._values = [values]
        
        # Normalize storages to list (or None if not realized)
        if storages is None:
            self._storages = None
        elif isinstance(storages, list):
            self._storages = storages if storages else None
        else:
            self._storages = [storages]
        
        self.sharding = sharding
        self.traced = traced
        self.tangent = None    # Populated during JVP
        self.cotangent = None  # Populated during VJP
        self.batch_dims = batch_dims
        
        # Tracing: multi-output operation tracking
        self.output_refs = None   # Shared OutputRefs instance (set by Operation.__call__)
        self.output_index = 0     # Position among siblings
        
        # Initialize cached metadata (populated during operation execution)
        self.cached_shape = None
        self.cached_dtype = None
        self.cached_device = None
        
        # Validate sharding consistency (basic checks for now)
        self._validate_sharding()
    
    def _validate_sharding(self) -> None:
        """Validate that shards are consistent with sharding specification.
        
        For now, just checks basic invariants. More detailed validation
        (e.g., shard shapes matching sharding spec) will be added later.
        """
        num_values = len(self._values)
        num_storages = len(self._storages) if self._storages else 0
        
        # If we have both values and storages, counts should match
        if num_values > 0 and num_storages > 0 and num_values != num_storages:
            raise ValueError(
                f"Mismatched shard counts: {num_values} values vs {num_storages} storages"
            )
        
        # If sharding is None, we should have at most 1 shard
        if self.sharding is None:
            if num_values > 1:
                raise ValueError(
                    f"Multiple values ({num_values}) provided without sharding specification"
                )
            if num_storages > 1:
                raise ValueError(
                    f"Multiple storages ({num_storages}) provided without sharding specification"
                )
    
    @property
    def is_realized(self) -> bool:
        """Whether this tensor has realized storage (has been computed)."""
        return self._storages is not None and len(self._storages) > 0
    
    @property
    def num_shards(self) -> int:
        """Number of shards (1 for unsharded tensors)."""
        if self._storages is not None:
            return len(self._storages)
        return len(self._values) if self._values else 1
    
    @property
    def is_sharded(self) -> bool:
        """Whether this tensor is sharded across multiple devices."""
        return self.sharding is not None
    
    @property
    def op(self) -> Operation | None:
        """Get the operation that produced this tensor."""
        if self.output_refs is None:
            return None
        return self.output_refs.op
    
    @property
    def op_name(self) -> str | None:
        """Get the operation name."""
        if self.output_refs is None:
            return None
        op = self.output_refs.op
        if hasattr(op, 'name'):
            return op.name
        return getattr(op, '__name__', None)
    
    @property
    def parents(self) -> list[TensorImpl]:
        """Get parent TensorImpls (derived from op_args).
        
        Returns an empty list if not traced or op_args is None.
        op_args now stores TensorImpl refs directly (not Tensor wrappers).
        """
        if self.output_refs is None:
            return []
        
        # Flatten the pytree of op_args and extract TensorImpls
        from . import pytree
        return [
            arg for arg in pytree.tree_leaves(self.output_refs.op_args)
            if isinstance(arg, TensorImpl)
        ]
    
    @property
    def is_leaf(self) -> bool:
        """True if this node has no parents (is an input/leaf tensor)."""
        return len(self.parents) == 0
    
    def physical_local_shape(self, shard_idx: int = 0) -> tuple[int, ...] | None:
        """Actual storage shape for a specific shard (including batch dims at prefix).
        
        Args:
            shard_idx: Index of the shard to get shape for (default: 0).
            
        Returns:
            Shape tuple or None if not available.
        """
        if self._storages is not None and shard_idx < len(self._storages):
            return tuple(self._storages[shard_idx].shape)
        if self._values and shard_idx < len(self._values):
            return tuple(self._values[shard_idx].type.shape)
        return None

    def logical_local_shape(self, shard_idx: int = 0) -> tuple[int, ...] | None:
        """Logical shape for a specific shard (excluding batch dims prefix).
        
        Args:
            shard_idx: Index of the shard to get shape for (default: 0).
            
        Returns:
            Shape tuple or None if not available.
        """
        physical = self.physical_local_shape(shard_idx)
        if physical is None:
            return None
        return physical[self.batch_dims:]
    
    # Convenience properties for the common unsharded case
    @property
    def physical_shape(self) -> tuple[int, ...] | None:
        """Actual storage shape of shard 0 (including batch dims at prefix)."""
        return self.physical_local_shape(0)
    
    @property
    def logical_shape(self) -> tuple[int, ...] | None:
        """Logical local shape of shard 0 (excluding batch dims prefix).
        
        For sharded tensors, this returns the LOCAL shard shape.
        Use global_shape for the full tensor shape.
        """
        return self.logical_local_shape(0)
    
    @property
    def global_shape(self) -> tuple[int, ...] | None:
        """Global logical shape of the tensor (excludes batch dims).
        
        This is the shape of the full tensor before sharding.
        For unsharded tensors, this equals logical_shape.
        """
        if self.cached_shape is not None:
            # Exclude batch dims from cached shape
            return tuple(int(d) for d in self.cached_shape[self.batch_dims:])
        # Fallback: unsharded case, use local shape
        return self.logical_shape
    
    @property
    def ndim(self) -> int | None:
        """Number of logical dimensions (based on global shape)."""
        shape = self.global_shape
        return len(shape) if shape is not None else None
    
    @property
    def batch_shape(self) -> tuple[int, ...] | None:
        """Shape of batch dimensions (first batch_dims axes of physical shape)."""
        physical = self.physical_shape
        if physical is None:
            return None
        return physical[:self.batch_dims]
    
    def __repr__(self) -> str:
        shards_str = f", shards={self.num_shards}" if self.is_sharded else ""
        return f"TensorImpl(op={self.op_name}, traced={self.traced}, parents={len(self.parents)}, batch_dims={self.batch_dims}{shards_str})"
    
    def cache_metadata(self, value: graph.TensorValue | graph.BufferValue) -> None:
        """Cache shape/dtype/device from a TensorValue (survives graph consumption).
        
        This metadata is used by the sharding compiler after the MAX graph
        has been consumed by evaluation.
        
        Args:
            value: The TensorValue or BufferValue to extract metadata from.
        """
        tensor_type = value.type.as_tensor() if hasattr(value.type, 'as_tensor') else value.type
        self.cached_shape = tuple(tensor_type.shape)
        self.cached_dtype = tensor_type.dtype
        device = tensor_type.device
        self.cached_device = device.to_device() if hasattr(device, 'to_device') else device


def get_topological_order(impl: TensorImpl) -> list[TensorImpl]:
    """Get TensorImpls in topological order (dependencies first)."""
    order: list[TensorImpl] = []
    visited: set[int] = set()
    
    def dfs(node: TensorImpl) -> None:
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)
        for parent in node.parents:
            dfs(parent)
        order.append(node)
    
    dfs(impl)
    return order


def print_computation_graph(impl: TensorImpl) -> None:
    """Print the computation graph for debugging."""
    order = get_topological_order(impl)
    print(f"Computation graph ({len(order)} nodes):")
    for i, node in enumerate(order):
        op_name = node.op_name or "leaf"
        parent_count = len(node.parents)
        traced_str = "traced" if node.traced else "untraced"
        print(f"  {i}: {op_name} ({traced_str}, {parent_count} parents)")
