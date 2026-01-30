# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""TensorImpl: Complete computation graph node containing all tensor internals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max import driver, graph
from max.driver import Device
from max.dtype import DType

if TYPE_CHECKING:
    from ...ops import Operation
    from ..graph.tracing import OutputRefs
    from ..sharding import ShardingSpec


class TensorImpl:
    """Graph node containing tensor data and autograd structure.

    Attributes:
        _values: Graph values for lazy execution (one per shard).
        _storages: Realized driver.Tensors (one per shard).
        sharding: Sharding specification.
        traced: Whether this node is traced.
        tangent: Tangent for JVP.
        cotangent: Cotangent for VJP.
        batch_dims: Number of batch dimensions.
        output_refs: Provenance for autograd.
    """

    __slots__ = (
        "_values",
        "_storages",
        "sharding",
        "sharding_constraint",
        "traced",
        "tangent",
        "cotangent",
        "dual",
        "batch_dims",
        "output_refs",
        "output_index",
        "values_epoch",
        "__weakref__",
    )

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
        values: (
            graph.BufferValue
            | graph.TensorValue
            | list[graph.BufferValue | graph.TensorValue]
            | None
        ) = None,
        traced: bool = False,
        batch_dims: int = 0,
        sharding_constraint: ShardingSpec | None = None,
    ):
        self._values = (
            values if isinstance(values, list) else ([values] if values else [])
        )
        self._storages = (
            storages
            if isinstance(storages, list)
            else ([storages] if storages else None)
        )

        self.sharding = None
        self.sharding_constraint = sharding_constraint
        self.traced = traced
        self.tangent = None
        self.cotangent = None
        self.dual = None
        self.batch_dims = batch_dims
        self.output_refs = None
        self.output_index = 0
        self.values_epoch = -1

    def _validate_sharding(self) -> None:
        """Validate consistency of shards and sharding spec."""
        n_vals = len(self._values)
        n_stores = len(self._storages) if self._storages else 0

        if n_vals > 0 and n_stores > 0 and n_vals != n_stores:
            raise ValueError(
                f"Shard count mismatch: {n_vals} values vs {n_stores} storages"
            )

        if self.sharding is None:
            if n_vals > 1:
                raise ValueError(f"Multiple values ({n_vals}) without sharding spec")
            if n_stores > 1:
                raise ValueError(
                    f"Multiple storages ({n_stores}) without sharding spec"
                )

    @property
    def is_realized(self) -> bool:
        return self._storages is not None and len(self._storages) > 0

    @property
    def num_shards(self) -> int:
        if self._storages is not None:
            return len(self._storages)
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
        if not self.output_refs:
            return None
        op = self.output_refs.op
        return getattr(op, "name", getattr(op, "__name__", None))

    @property
    def parents(self) -> list[TensorImpl]:
        """Get parent TensorImpls."""
        if self.output_refs is None:
            return []
        from ..common import pytree

        return [
            arg
            for arg in pytree.tree_leaves(self.output_refs.op_args)
            if isinstance(arg, TensorImpl)
        ]

    @property
    def is_leaf(self) -> bool:
        return len(self.parents) == 0

    def _get_valid_values(self):
        from ..graph.engine import GRAPH

        if self.values_epoch != GRAPH.epoch:
            return []
        return self._values

    def physical_local_shape(self, shard_idx: int = 0) -> graph.Shape | None:
        """Storage shape for a specific shard (includes batch dims)."""
        if self._storages and shard_idx < len(self._storages):
            return graph.Shape(self._storages[shard_idx].shape)

        values = self._get_valid_values()
        if values and shard_idx < len(values):
            return values[shard_idx].type.shape
        return None

    def logical_local_shape(self, shard_idx: int = 0) -> graph.Shape | None:
        """Logical shape for a specific shard (excludes batch dims)."""
        physical = self.physical_local_shape(shard_idx)
        if physical is None or self.batch_dims == 0:
            return physical
        return graph.Shape(physical[self.batch_dims :])

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
        if phys is None:
            return None

        if self.batch_dims > 0:
            return graph.Shape(phys[self.batch_dims :])
        return phys

    @property
    def physical_global_shape(self) -> graph.Shape | None:
        """Global physical shape (includes batch dims)."""
        local = self.physical_shape
        if local is None:
            raise RuntimeError(
                f"Cannot determine physical shape for tensor (sharding={self.sharding}). "
                "No valid values or storage available in current epoch."
            )

        # Unsharded case: local shape IS the physical global shape
        if not self.sharding:
            return local

        # Sharded case: reconstruct from local chunks + spec
        values = self._get_valid_values()
        shard_shapes = (
            [tuple(int(d) for d in v.type.shape) for v in values] if values else None
        )

        if shard_shapes is None and self._storages:
            shard_shapes = [tuple(int(d) for d in s.shape) for s in self._storages]

        from ..sharding.spec import compute_global_shape

        global_ints = compute_global_shape(
            tuple(int(d) for d in local), self.sharding, shard_shapes=shard_shapes
        )
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
        if physical is None or self.batch_dims == 0:
            return None
        return graph.Shape(physical[: self.batch_dims])

    @property
    def local_shape(self) -> graph.Shape | None:
        return self.physical_local_shape(0)

    def __repr__(self) -> str:
        shards_str = f", shards={self.num_shards}" if self.is_sharded else ""
        return f"TensorImpl(op={self.op_name}, traced={self.traced}, parents={len(self.parents)}, batch_dims={self.batch_dims}{shards_str})"

    def get_unrealized_shape(self) -> graph.Shape:
        values = self._get_valid_values()
        if not values:
            raise RuntimeError("Internal error: _values missing")
        return values[0].type.shape

    def get_unrealized_dtype(self) -> DType:
        values = self._get_valid_values()
        if not values:
            raise RuntimeError("Internal error: _values missing")
        return values[0].type.dtype

    def get_realized_shape(self) -> graph.Shape:
        values = self._get_valid_values()
        if values:
            return values[0].type.shape
        if self._storages:
            return graph.Shape(self._storages[0].shape)
        if self.sharding:

            pass
        raise RuntimeError("No shape source available")

    def get_realized_dtype(self) -> DType:
        values = self._get_valid_values()
        if values:
            return values[0].type.dtype
        if self._storages:
            return self._storages[0].dtype
        raise RuntimeError("No dtype source available")

    @property
    def primary_value(self) -> driver.Tensor | graph.BufferValue | graph.TensorValue:
        if self._storages:
            return self._storages[0]

        values = self._get_valid_values()
        if values:
            return values[0]
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
