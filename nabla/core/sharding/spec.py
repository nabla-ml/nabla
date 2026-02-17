# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from max.graph import DeviceRef, Value


def parse_sub_axis(axis_name: str) -> tuple[str, int, int] | None:
    """Parse 'axis:(pre_size)size' -> (parent, pre_size, size), or None if not a sub-axis."""
    if ":" not in axis_name:
        return None
    match = re.match(r"^(\w+):\((\d+)\)(\d+)$", axis_name)
    if not match:
        raise ValueError(f"Invalid sub-axis format: {axis_name}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def validate_sub_axes_non_overlapping(axes: list[str]) -> None:
    """Validate that sub-axes of the same parent don't overlap."""
    parent_ranges: dict[str, list[tuple[int, int, str]]] = {}

    for axis in axes:
        parsed = parse_sub_axis(axis)
        if parsed is None:
            continue
        parent, pre_size, size = parsed

        start = pre_size
        end = pre_size * size

        if parent not in parent_ranges:
            parent_ranges[parent] = []

        for existing_start, existing_end, existing_axis in parent_ranges[parent]:
            if not (end <= existing_start or start >= existing_end):
                raise ValueError(
                    f"Sub-axes overlap: '{axis}' and '{existing_axis}' "
                    f"(ranges [{start}, {end}) and [{existing_start}, {existing_end}))"
                )

        parent_ranges[parent].append((start, end, axis))


def check_sub_axes_maximality(axes: list[str]) -> list[str]:
    """Return warnings for adjacent sub-axes that could be merged."""
    warnings = []
    parent_sub_axes: dict[str, list[tuple[int, int, str]]] = {}

    for axis in axes:
        parsed = parse_sub_axis(axis)
        if parsed is None:
            continue
        parent, pre_size, size = parsed
        if parent not in parent_sub_axes:
            parent_sub_axes[parent] = []
        parent_sub_axes[parent].append((pre_size, size, axis))

    for parent, subs in parent_sub_axes.items():
        subs_sorted = sorted(subs, key=lambda x: x[0])
        for i in range(len(subs_sorted) - 1):
            pre1, size1, ax1 = subs_sorted[i]
            pre2, size2, ax2 = subs_sorted[i + 1]
            if pre1 * size1 == pre2:
                merged_size = size1 * size2
                warnings.append(
                    f"Adjacent sub-axes '{ax1}' and '{ax2}' could be merged "
                    f"into '{parent}:({pre1}){merged_size}'"
                )

    return warnings


class DeviceMesh:
    """Logical multi-dimensional view of devices: @name = <["axis1"=size1, ...]>.

    Attributes:
        name: Name of the mesh.
        shape: Shape of the mesh (e.g., (2, 4)).
        axis_names: Names for each axis (e.g., ("x", "y")).
        devices: Logical device IDs.
        device_refs: Physical device references.
    """

    name: str
    shape: tuple[int, ...]
    axis_names: tuple[str, ...]
    bandwidth: float
    devices: list[int]
    device_refs: list["DeviceRef"]
    axis_lookup: dict[str, int]
    phys_strides: list[int]
    _signal_buffers: dict[tuple[int, tuple["DeviceRef", ...]], list["Value"]]

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        axis_names: tuple[str, ...],
        devices: list[int] | None = None,
        device_refs: list["DeviceRef"] | None = None,
        bandwidth: float = 1.0,
    ) -> None:
        self.name = name
        self.shape = shape
        self.axis_names = axis_names
        self.bandwidth = bandwidth

        total_devices = int(np.prod(shape))
        if devices is None:
            devices = list(range(total_devices))
        self.devices = devices

        if total_devices != len(devices):
            raise ValueError(
                f"Mesh shape {shape} requires {total_devices} devices, "
                f"but got {len(devices)}"
            )

        if device_refs is None:
            from max.driver import Accelerator, accelerator_count
            from max.graph import DeviceRef

            # Try to use actual GPUs if available, otherwise simulate on CPU
            gpu_count = accelerator_count()
            if gpu_count >= total_devices:
                # Use real GPUs - create DeviceRefs for actual GPU devices
                device_refs = [
                    DeviceRef.from_device(Accelerator(i)) for i in range(total_devices)
                ]
            else:
                # Simulation mode - all on CPU
                device_refs = [DeviceRef.CPU() for _ in range(total_devices)]
        self.device_refs = device_refs

        self.axis_lookup = {name: i for i, name in enumerate(axis_names)}

        self.phys_strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            self.phys_strides[i] = shape[i + 1] * self.phys_strides[i + 1]

        self._signal_buffers = {}  # Cache: (buffer_size, device_hash) -> list[Value]

    def get_signal_buffers(
        self, buffer_size: int = 65536, use_cache: bool = False
    ) -> list["Value"]:
        """Create signal buffers for collective operations.

        Args:
            buffer_size: Size of the signal buffer in bytes.
            use_cache: Whether to reuse cached buffers.

        Notes:
            MAX graph values are region-scoped SSA values. Reusing cached
            buffers across graph regions can lead to verifier failures such as
            UNKNOWN SSA VALUE. This method therefore defaults to creating fresh
            buffers for correctness.

        Returns:
            list: List of MAX buffer values, one per participating device.
        """
        from max.dtype import DType
        from max.graph import ops
        from max.graph.type import BufferType

        # Cache key is mesh-specific (device refs are part of the key).
        cache_key = (buffer_size, tuple(self.device_refs))
        if use_cache and cache_key in self._signal_buffers:
            return self._signal_buffers[cache_key]

        buffers = [
            ops.buffer_create(BufferType(DType.uint8, (buffer_size,), dev))
            for dev in self.device_refs
        ]
        if use_cache:
            self._signal_buffers[cache_key] = buffers
        return buffers

    @property
    def is_distributed(self) -> bool:
        """Check if mesh has unique device refs (true distributed vs simulated)."""
        return self.device_refs is not None and len(set(self.device_refs)) == len(
            self.device_refs
        )

    def __repr__(self) -> str:
        axes_str = ", ".join(
            f'"{n}"={s}' for n, s in zip(self.axis_names, self.shape, strict=False)
        )
        return f"@{self.name} = <[{axes_str}]>"

    def get_axis_size(self, axis_name: str) -> int:
        """Get size of an axis. For sub-axes 'x:(m)k', returns k."""
        parsed = parse_sub_axis(axis_name)
        if parsed:
            return parsed[2]

        if axis_name not in self.axis_lookup:
            raise ValueError(
                f"Unknown axis: {axis_name} (available: {self.axis_names})"
            )
        return self.shape[self.axis_lookup[axis_name]]

    def get_coordinate(self, device_id: int, axis_name: str) -> int:
        """Get coordinate of device along axis. Handles sub-axes 'x:(m)k'."""

        parsed = parse_sub_axis(axis_name)
        if parsed:
            parent_name, pre_size, size = parsed
            parent_coord = self.get_coordinate(device_id, parent_name)
            parent_total = self.get_axis_size(parent_name)

            if pre_size * size == 0:
                raise ValueError(
                    f"Invalid sub-axis sizes: pre_size={pre_size}, size={size}"
                )
            if parent_total % (pre_size * size) != 0:
                raise ValueError(
                    f"Sub-axis {axis_name} invalid: parent size {parent_total} "
                    f"not divisible by pre_size*size = {pre_size * size}"
                )

            post_size = parent_total // (pre_size * size)
            return (parent_coord // post_size) % size

        if axis_name not in self.axis_lookup:
            raise ValueError(f"Unknown axis: {axis_name}")

        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not in mesh {self.name}")

        flat_idx = self.devices.index(device_id)

        coords = []
        rem = flat_idx
        for stride in self.phys_strides:
            coords.append(rem // stride)
            rem %= stride

        return coords[self.axis_lookup[axis_name]]

    def get_devices_on_axis_slice(self, axis_name: str, coordinate: int) -> list[int]:
        """Get all device IDs that have the given coordinate on the specified axis."""
        return [
            d for d in self.devices if self.get_coordinate(d, axis_name) == coordinate
        ]

    def get_axis_indices(self, device_id: int) -> dict[str, int]:
        """Get all axis coordinates for a given device."""
        return {name: self.get_coordinate(device_id, name) for name in self.axis_names}


@dataclass
class DimSpec:
    """Per-dimension sharding specification."""

    axes: list[str] = field(default_factory=list)
    is_open: bool = False
    priority: int = 0
    partial: bool = False

    def __post_init__(self) -> None:
        if not self.axes and not self.is_open and self.priority != 0:
            raise ValueError(
                f"Empty closed dimension {{}} cannot have non-zero priority (got p{self.priority})"
            )

    def __repr__(self) -> str:
        """Format: {axes} or {axes, ?} with optional p<N> suffix."""
        if not self.axes:
            marker = "?" if self.is_open else ""
            prio_str = (
                f"p{self.priority}" if self.is_open and self.priority != 0 else ""
            )
            return f"{{{marker}}}{prio_str}"

        axes_str = ", ".join(f"'{a}'" for a in self.axes)
        open_marker = ", ?" if self.is_open else ""
        if not self.is_open:
            open_marker = ""

        prio_str = f"p{self.priority}" if self.priority != 0 else ""
        return f"{{{axes_str}{open_marker}}}{prio_str}"

    def is_replicated(self) -> bool:
        """True if fully replicated (no axes)."""
        return len(self.axes) == 0

    def get_total_shards(self, mesh: "DeviceMesh") -> int:
        """Total shards for this dimension."""
        total = 1
        for axis in self.axes:
            total *= mesh.get_axis_size(axis)
        return total

    def clone(self) -> "DimSpec":
        """Create a deep copy of this DimSpec."""
        return DimSpec(
            axes=list(self.axes),
            is_open=self.is_open,
            priority=self.priority,
            partial=self.partial,
        )

    @staticmethod
    def from_raw(raw: Any) -> "DimSpec":
        """Convert raw input (None, str, tuple, DimSpec) to DimSpec."""
        if isinstance(raw, DimSpec):
            return raw
        if raw is None:
            return DimSpec([])
        if isinstance(raw, str):
            return DimSpec([raw])
        if isinstance(raw, (tuple, list)):
            return DimSpec([str(x) for x in raw])
        raise ValueError(
            f"Invalid dimension spec input: {raw!r}. Expected None, str, tuple/list of str, or DimSpec."
        )


@dataclass
class ShardingSpec:
    """Complete tensor sharding: sharding<@mesh, [dim_shardings], replicated={axes}>."""

    mesh: DeviceMesh
    dim_specs: list[DimSpec] = field(default_factory=list)
    replicated_axes: set[str] = field(default_factory=set)
    partial_sum_axes: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Validate: no duplicate axes, no explicit-replicated axes in dims."""

        if hasattr(self, "dim_specs") and self.dim_specs:
            self.dim_specs = [DimSpec.from_raw(d) for d in self.dim_specs]

        used_axes = set()
        all_axes = []

        for dim_idx, dim in enumerate(self.dim_specs):
            for axis in dim.axes:
                if axis in self.replicated_axes:
                    raise ValueError(
                        f"Axis '{axis}' is explicitly replicated but assigned to dimension {dim_idx}."
                    )

                all_axes.append(axis)

                if axis in used_axes:
                    raise ValueError(f"Axis '{axis}' used multiple times in sharding.")
                used_axes.add(axis)

        for axis in self.partial_sum_axes:
            if axis in used_axes:
                raise ValueError(
                    f"Axis '{axis}' is both a dimension axis and a partial sum axis."
                )
            if axis in self.replicated_axes:
                raise ValueError(
                    f"Axis '{axis}' is both explicitly replicated and a partial sum axis."
                )
            used_axes.add(axis)

        all_axes.extend(list(self.partial_sum_axes))
        validate_sub_axes_non_overlapping(all_axes)

        check_sub_axes_maximality(all_axes)

    def __repr__(self) -> str:
        """String representation following Shardy spec grammar."""
        dims_str = ", ".join(str(d) for d in self.dim_specs)
        rep_str = ""
        if self.replicated_axes:
            ordered_rep = self._order_replicated_axes(self.replicated_axes)
            rep_str = ", replicated={" + ", ".join(f"'{a}'" for a in ordered_rep) + "}"

        all_partial_axes = set(self.partial_sum_axes)
        for dim in self.dim_specs:
            if dim.partial:
                all_partial_axes.update(dim.axes)

        partial_str = ""
        if all_partial_axes:
            ordered_partial = self._order_replicated_axes(all_partial_axes)
            partial_str = (
                ", partial={" + ", ".join(f"'{a}'" for a in ordered_partial) + "}"
            )

        return f"sharding<@{self.mesh.name}, [{dims_str}]{rep_str}{partial_str}>"

    def _order_replicated_axes(self, axes_set: set[str]) -> list[str]:
        """Order axes: mesh order, sub-axes by pre-size."""
        full_axes = []
        sub_axes_by_parent: dict[str, list[tuple[str, int, int]]] = {}

        for ax in axes_set:
            parsed = parse_sub_axis(ax)
            if parsed is None:
                full_axes.append(ax)
            else:
                parent, pre_size, size = parsed
                if parent not in sub_axes_by_parent:
                    sub_axes_by_parent[parent] = []
                sub_axes_by_parent[parent].append((ax, pre_size, size))

        result = []

        for ax_name in self.mesh.axis_names:
            if ax_name in full_axes:
                result.append(ax_name)

            if ax_name in sub_axes_by_parent:
                sorted_subs = sorted(sub_axes_by_parent[ax_name], key=lambda x: x[1])
                result.extend(ax_str for ax_str, _, _ in sorted_subs)

        return result

    def get_implicitly_replicated_axes(self) -> set[str]:
        """Get axes not used in sharding or explicitly replicated."""
        used = set()
        for dim in self.dim_specs:
            used.update(dim.axes)
        used.update(self.replicated_axes)
        used.update(self.partial_sum_axes)

        implicit = set()
        for ax_name in self.mesh.axis_names:
            if ax_name not in used:
                implicit.add(ax_name)
        return implicit

    def is_fully_replicated(self) -> bool:
        """True if tensor is fully replicated."""
        return (
            all(dim.is_replicated() for dim in self.dim_specs)
            and not self.partial_sum_axes
            and not any(dim.partial for dim in self.dim_specs)
        )

    @property
    def total_shards(self) -> int:
        """Total number of shards across all dimensions."""
        total = 1
        for dim_spec in self.dim_specs:
            for axis in dim_spec.axes:
                total *= self.mesh.get_axis_size(axis)
        return total

    def get_max_priority(self) -> int:
        """Get the maximum (lowest urgency) priority used in this spec."""
        return max((d.priority for d in self.dim_specs), default=0)

    def get_min_priority(self) -> int:
        """Get the minimum (highest urgency) priority used in this spec."""
        return min((d.priority for d in self.dim_specs), default=0)

    def clone(self) -> "ShardingSpec":
        """Create a deep copy of this ShardingSpec."""
        return ShardingSpec(
            mesh=self.mesh,
            dim_specs=[d.clone() for d in self.dim_specs],
            replicated_axes=set(self.replicated_axes),
            partial_sum_axes=set(self.partial_sum_axes),
        )


def compute_local_shape(
    global_shape: tuple[int, ...],
    sharding: ShardingSpec,
    device_id: int,
) -> tuple[int, ...]:
    """Compute the local shard shape for a device."""
    if len(global_shape) != len(sharding.dim_specs):
        raise ValueError(
            f"Rank mismatch: global_shape has rank {len(global_shape)}, "
            f"but sharding spec has {len(sharding.dim_specs)} dim specs."
        )

    local_shape = []
    for dim_idx in range(len(global_shape)):
        dim_spec = sharding.dim_specs[dim_idx]
        global_len = int(global_shape[dim_idx])

        if not dim_spec.axes:
            local_shape.append(global_len)
            continue

        total_shards = 1
        my_shard_index = 0

        for axis_name in dim_spec.axes:
            size = sharding.mesh.get_axis_size(axis_name)
            coord = sharding.mesh.get_coordinate(device_id, axis_name)
            my_shard_index = (my_shard_index * size) + coord
            total_shards *= size

        chunk_size = math.ceil(global_len / total_shards)
        start = my_shard_index * chunk_size
        theoretical_end = start + chunk_size
        real_end = min(theoretical_end, global_len)

        length = max(0, real_end - start)
        local_shape.append(length)

    return tuple(local_shape)


def get_num_shards(sharding: ShardingSpec) -> int:
    """Get the total number of shards for this sharding spec."""
    return len(sharding.mesh.devices)


def compute_global_shape(
    local_shape: tuple[int, ...],
    sharding: ShardingSpec | None,
    shard_shapes: list[tuple[int, ...]] | None = None,
) -> tuple[int, ...]:
    """Compute global shape from local shape and sharding spec.

    This function handles two modes:
    1. Aggregation (shard_shapes provided): Sums actual shards along sharded axes.
       This is the 'source of truth' for uneven sharding in simulation.
    2. Prediction (only local_shape provided): Uses multiplication.
    """
    if not sharding or not local_shape:
        return local_shape

    if shard_shapes and len(shard_shapes) > 1 and sharding.mesh:
        mesh = sharding.mesh
        rank = len(local_shape)
        global_shape = []

        for i in range(rank):
            dim_spec = sharding.dim_specs[i] if i < len(sharding.dim_specs) else None
            if not dim_spec or not dim_spec.axes or dim_spec.partial:
                global_shape.append(int(local_shape[i]))
            else:
                total_shards = dim_spec.get_total_shards(mesh)
                num_total_devices = len(shard_shapes)

                if num_total_devices % total_shards != 0:
                    num_replicas = 1
                else:
                    num_replicas = num_total_devices // total_shards

                sum_local = sum(int(s_shape[i]) for s_shape in shard_shapes)

                global_shape.append(sum_local // num_replicas)

        return tuple(global_shape)

    result = [int(d) for d in local_shape]
    for i, dim_spec in enumerate(sharding.dim_specs[: len(result)]):
        if dim_spec.axes and not dim_spec.partial:
            result[i] *= dim_spec.get_total_shards(sharding.mesh)

    return tuple(result)


def needs_reshard(from_spec: ShardingSpec | None, to_spec: ShardingSpec | None) -> bool:
    """Check if specs differ requiring resharding."""
    if (from_spec is None) != (to_spec is None):
        return True
    if from_spec is None:
        return False
    if len(from_spec.dim_specs) != len(to_spec.dim_specs):
        return True
    return any(
        f.axes != t.axes or f.partial != t.partial
        for f, t in zip(from_spec.dim_specs, to_spec.dim_specs, strict=False)
    ) or (from_spec.partial_sum_axes != to_spec.partial_sum_axes)


class PartitionSpec(tuple):
    """JAX-compatible PartitionSpec."""

    def __new__(cls, *args):
        return super().__new__(cls, args)


P = PartitionSpec
