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

# Formatting constants for rich representation
RESET = "\033[0m"
C_VAR = "\033[96m"
C_KEYWORD = "\033[1m"
C_BATCH = "\033[90m"

if TYPE_CHECKING:
    from ...ops import Operation
    from ..graph.tracing import OpNode
    from ..sharding import ShardingSpec

# Module-level cache for hot-path imports (avoids per-call deferred imports)
_GRAPH = None

def _get_graph():
    global _GRAPH
    from ..graph.engine import GRAPH
    _GRAPH = GRAPH


class TensorImpl:
    """Graph node containing tensor data and autograd structure."""

    __slots__ = (
        "_graph_values",
        "_buffers",
        "sharding",
        "sharding_constraint",
        "is_traced",
        "requires_grad",
        "tangent",
        "cotangent",
        "dual",
        "batch_dims",
        "output_refs",
        "output_index",
        "graph_values_epoch",
        "_physical_shapes",
        "_shard_dtypes",
        "_shard_devices",
        "__weakref__",
    )

    _graph_values: list[graph.BufferValue | graph.TensorValue]
    _buffers: list[driver.Buffer] | None
    sharding: object | None
    sharding_constraint: object | None
    is_traced: bool
    requires_grad: bool
    tangent: TensorImpl | None
    cotangent: TensorImpl | None
    dual: TensorImpl | None
    batch_dims: int
    output_refs: OpNode | None
    output_index: int
    _physical_shapes: list[tuple[int, ...]] | None
    _shard_dtypes: list[DType] | None
    _shard_devices: list[Device] | None

    def __init__(
        self,
        bufferss: driver.Buffer | list[driver.Buffer] | None = None,
        values: (
            graph.BufferValue
            | graph.TensorValue
            | list[graph.BufferValue | graph.TensorValue]
            | None
        ) = None,
        is_traced: bool = False,
        batch_dims: int = 0,
        sharding_constraint: ShardingSpec | None = None,
        physical_shapes: list[tuple[int, ...]] | None = None,
        shard_dtypes: list[DType] | None = None,
        shard_devices: list[Device] | None = None,
    ):
        self._graph_values = (
            values
            if isinstance(values, list)
            else ([values] if values is not None else [])
        )
        self._buffers = (
            bufferss
            if isinstance(bufferss, list)
            else ([bufferss] if bufferss else None)
        )

        self.sharding = None
        self.sharding_constraint = sharding_constraint
        self.is_traced = is_traced
        self.requires_grad = False  # Separate from is_traced; marks gradient leaves
        self.tangent = None
        self.cotangent = None
        self.dual = None
        self.batch_dims = batch_dims
        self.output_refs = None
        self.output_index = 0
        self.graph_values_epoch = -1
        self._physical_shapes = physical_shapes
        self._shard_dtypes = shard_dtypes
        self._shard_devices = shard_devices

    def _validate_sharding(self) -> None:
        """Validate consistency of shards and sharding spec."""
        n_vals = len(self._graph_values)
        n_stores = len(self._buffers) if self._buffers else 0

        if n_vals > 0 and n_stores > 0 and n_vals != n_stores:
            raise ValueError(
                f"Shard count mismatch: {n_vals} values vs {n_stores} bufferss"
            )

        if self.sharding is None:
            if n_vals > 1:
                raise ValueError(f"Multiple values ({n_vals}) without sharding spec")
            if n_stores > 1:
                raise ValueError(
                    f"Multiple bufferss ({n_stores}) without sharding spec"
                )

    @property
    def is_realized(self) -> bool:
        return self._buffers is not None and len(self._buffers) > 0

    @property
    def num_shards(self) -> int:
        if self._buffers is not None:
            return len(self._buffers)
        if self._graph_values:
            return len(self._graph_values)
        if self._physical_shapes:
            return len(self._physical_shapes)
        return 1

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

    def _get_valid_graph_values(self):
        global _GRAPH
        if _GRAPH is None:
            _get_graph()

        if self.graph_values_epoch != _GRAPH.epoch:
            return []
        return self._graph_values

    def physical_local_shape(self, shard_idx: int = 0) -> graph.Shape | None:
        """Buffers shape for a specific shard (includes batch dims)."""
        if self._buffers and shard_idx < len(self._buffers):
            return graph.Shape(self._buffers[shard_idx].shape)

        values = self._get_valid_graph_values()
        if values and shard_idx < len(values):
            return values[shard_idx].type.shape

        if self._physical_shapes and shard_idx < len(self._physical_shapes):
            return graph.Shape(self._physical_shapes[shard_idx])

        return None

    def physical_local_shape_ints(self, shard_idx: int = 0) -> tuple[int, ...] | None:
        """Int-tuple shape for a specific shard (avoids creating Shape/Dim objects)."""
        if self._buffers and shard_idx < len(self._buffers):
            return tuple(self._buffers[shard_idx].shape)

        values = self._get_valid_graph_values()
        if values and shard_idx < len(values):
            return tuple(int(d) for d in values[shard_idx].type.shape)

        if self._physical_shapes and shard_idx < len(self._physical_shapes):
            return self._physical_shapes[shard_idx]

        return None

    def logical_local_shape(self, shard_idx: int = 0) -> graph.Shape | None:
        """Logical shape for a specific shard (excludes batch dims)."""
        physical = self.physical_local_shape(shard_idx)
        if physical is None or self.batch_dims == 0:
            return physical
        return graph.Shape(physical[self.batch_dims :])

    @property
    def physical_shape(self) -> graph.Shape | None:
        """Buffers shape of shard 0 (includes batch dims)."""
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
                "No valid values or buffers available in current epoch."
            )

        # Unsharded case: local shape IS the physical global shape
        if not self.sharding:
            return local

        # Sharded case: reconstruct from local chunks + spec
        values = self._get_valid_graph_values()
        shard_shapes = (
            [tuple(int(d) for d in v.type.shape) for v in values] if values else None
        )

        if shard_shapes is None and self._buffers:
            shard_shapes = [tuple(int(d) for d in s.shape) for s in self._buffers]

        if shard_shapes is None and self._physical_shapes:
            shard_shapes = self._physical_shapes

        from ..sharding.spec import compute_global_shape

        global_ints = compute_global_shape(
            tuple(int(d) for d in local), self.sharding, shard_shapes=shard_shapes
        )
        return graph.Shape(global_ints)

    @property
    def physical_global_shape_ints(self) -> tuple[int, ...] | None:
        """Fast global physical shape as ints (no Shape/Dim allocation)."""
        # Unsharded: physical global = physical local shard 0
        if not self.sharding:
            return self.physical_local_shape_ints(0)
        # Sharded: fall back to the full path
        shape = self.physical_global_shape
        return tuple(int(d) for d in shape) if shape is not None else None

    @property
    def global_shape_ints(self) -> tuple[int, ...] | None:
        """Fast global logical shape as ints (no Shape/Dim allocation)."""
        # Fast path: unsharded tensor — physical_local_shape_ints is the global shape
        if not self.sharding:
            phys = self.physical_local_shape_ints(0)
            if phys is None:
                return None
            if self.batch_dims > 0:
                return phys[self.batch_dims:]
            return phys
        # Sharded: fall back to the full path
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

    @staticmethod
    def _format_type(impl: TensorImpl) -> str:
        """Get a concise string representation of the dtype."""
        try:
            dtype_obj = impl.dtype
            dtype_str = str(dtype_obj)
        except Exception:
            # Fallback for unrealized/partially initialized tensors
            try:
                if impl._shard_dtypes:
                    dtype_str = str(impl._shard_dtypes[0])
                elif impl._buffers:
                    dtype_str = str(impl._buffers[0].dtype)
                else:
                    vals = impl._get_valid_graph_values()
                    if vals:
                        dtype_str = str(vals[0].type.dtype)
                    else:
                        dtype_str = "unknown"
            except Exception:
                dtype_str = "unknown"

        return (
            dtype_str.lower()
            .replace("dtype.", "")
            .replace("float", "f")
            .replace("int", "i")
        )

    @staticmethod
    def _format_shape_part(shape: tuple | list | None, batch_dims: int = 0) -> str:
        """Format a shape tuple with batch dims colored."""
        if shape is None:
            return "[?]"

        clean = [int(d) if hasattr(d, "__int__") else str(d) for d in shape]

        if batch_dims > 0 and batch_dims <= len(clean):
            batch_part = clean[:batch_dims]
            logical_part = clean[batch_dims:]

            b_str = str(batch_part).replace("[", "").replace("]", "")
            l_str = str(logical_part).replace("[", "").replace("]", "")

            if logical_part:
                return f"[{C_BATCH}{b_str}{RESET} | {l_str}]"
            else:
                return f"[{C_BATCH}{b_str}{RESET}]"

        return str(clean).replace(" ", "")

    @staticmethod
    def _format_spec_factors(sharding: Any) -> str:
        """Format sharding factors: (<dp, tp>)"""
        if not sharding:
            return ""

        all_partial_axes = set()
        if hasattr(sharding, "partial_sum_axes"):
            all_partial_axes.update(sharding.partial_sum_axes)

        factors = []
        if hasattr(sharding, "dim_specs"):
            for dim in sharding.dim_specs:
                if getattr(dim, "partial", False):
                    all_partial_axes.update(dim.axes)

                if not dim.axes:
                    factors.append("*")
                else:
                    factors.append(", ".join(dim.axes))

        partial_sum_str = ""
        if all_partial_axes:
            ordered_partial = sorted(list(all_partial_axes))
            axes_joined = ", ".join(f"'{a}'" for a in ordered_partial)
            partial_sum_str = f" | partial={{{axes_joined}}}"

        return f"(<{', '.join(factors)}>{partial_sum_str})"

    def format_metadata(self, include_data: bool = False) -> str:
        """Format complete tensor metadata: dtype[global](factors)(local=[local])"""
        dtype = self._format_type(self)
        batch_dims = getattr(self, "batch_dims", 0)

        # Try to get shapes without triggering errors
        local_shape = None
        try:
            local_shape = self.physical_local_shape(0)
        except Exception:
            pass

        local_str = self._format_shape_part(local_shape, batch_dims)
        global_str = "[?]"
        factors_str = ""

        if self.sharding:
            factors_str = self._format_spec_factors(self.sharding)
            try:
                if local_shape is not None:
                    from ..sharding.spec import compute_global_shape

                    shard_shapes = self._physical_shapes
                    if not shard_shapes and self._buffers:
                        shard_shapes = [s.shape for s in self._buffers]
                    elif not shard_shapes:
                        vals = self._get_valid_graph_values()
                        if vals:
                            shard_shapes = [v.type.shape for v in vals]

                    g_shape = compute_global_shape(
                        tuple(local_shape), self.sharding, shard_shapes=shard_shapes
                    )
                    global_str = self._format_shape_part(g_shape, batch_dims)
            except Exception:
                pass
        else:
            global_str = local_str

        # Add " (traced)" if is_traced is true
        traced_str = " (traced)" if self.is_traced else ""

        header = ""
        if self.sharding:
            header = f"{dtype}{global_str}{factors_str}(local={local_str}){traced_str}"
        else:
            header = f"{dtype}{global_str}{traced_str}"

        if not include_data:
            return header

        try:
            from max.driver import CPU
            import numpy as np

            if self._buffers:
                # Reduced view settings: max 6 elements per dim (3 at start, 3 at end), no row wrapping
                print_opts = {
                    "edgeitems": 3,
                    "threshold": 6,
                    "max_line_width": 1000,  # Corrected from linewidth
                    "precision": 4,
                    "suppress_small": True,
                }

                if self.is_sharded:
                    shard_reprs = []
                    for i, buf in enumerate(self._buffers):
                        arr = buf.to(CPU()).to_numpy()
                        arr_str = np.array2string(arr, **print_opts)
                        prefix = f"shard({i}): "
                        lines = arr_str.split("\n")
                        first_line = prefix + lines[0]
                        indent = " " * len(prefix)
                        other_lines = [indent + line for line in lines[1:]]
                        shard_reprs.append("\n".join([first_line] + other_lines))
                    
                    shards_block = "\n".join(shard_reprs)
                    indented_shards = "\n".join("  " + line for line in shards_block.split("\n"))
                    data_str = f"[\n{indented_shards}\n]"
                else:
                    arr = self._buffers[0].to(CPU()).to_numpy()
                    data_str = np.array2string(arr, **print_opts)
            else:
                data_str = "[unrealized]"
        except Exception as e:
            data_str = f"[data unavailable: {e}]"

        return f"{data_str} : {header}"

    def __repr__(self) -> str:
        return self.format_metadata(include_data=True)

    def __str__(self) -> str:
        return self.format_metadata(include_data=True)

    def get_unrealized_shape(self) -> graph.Shape:
        values = self._get_valid_graph_values()
        if not values:
            raise RuntimeError("Internal error: _graph_values missing")
        return values[0].type.shape

    def get_unrealized_dtype(self) -> DType:
        values = self._get_valid_graph_values()
        if not values:
            raise RuntimeError("Internal error: _graph_values missing")
        return values[0].type.dtype

    def get_realized_shape(self) -> graph.Shape:
        values = self._get_valid_graph_values()
        if values:
            return values[0].type.shape
        if self._buffers:
            return graph.Shape(self._buffers[0].shape)
        if self.sharding:

            pass
        raise RuntimeError("No shape source available")

    def get_realized_dtype(self) -> DType:
        values = self._get_valid_graph_values()
        if values:
            return values[0].type.dtype
        if self._buffers:
            return self._buffers[0].dtype
        raise RuntimeError("No dtype source available")

    @property
    def primary_value(self) -> driver.Buffer | graph.BufferValue | graph.TensorValue:
        if self._buffers:
            return self._buffers[0]

        values = self._get_valid_graph_values()
        if values:
            return values[0]
        raise RuntimeError("Tensor has no buffers and no values")

    @property
    def dtype(self) -> DType:
        if self._shard_dtypes:
            return self._shard_dtypes[0]
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
            # Promise tensor - use stored shard_devices
            if self._shard_devices:
                return self._shard_devices[0]
            raise
