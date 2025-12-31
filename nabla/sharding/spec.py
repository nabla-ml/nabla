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

"""
Sharding Core: Physical and Representation Layers
===============================================

This module implements the fundamental data structures for the Sharding system:
1.  **Physical Layer**: DeviceMesh for logical device organization.
2.  **Representation Layer**: ShardingSpec and DimSpec for tensor sharding.

This file contains state definitions only and does not contain propagation algorithms.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# --- Helper Functions for Axis Parsing ---

def parse_sub_axis(axis_name: str) -> Optional[Tuple[str, int, int]]:
    """Parse 'axis:(pre_size)size' -> (parent, pre_size, size), or None if not a sub-axis."""
    if ":" not in axis_name:
        return None
    match = re.match(r"^(\w+):\((\d+)\)(\d+)$", axis_name)
    if not match:
        raise ValueError(f"Invalid sub-axis format: {axis_name}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def validate_sub_axes_non_overlapping(axes: List[str]) -> None:
    """Validate that sub-axes of the same parent don't overlap."""
    parent_ranges: Dict[str, List[Tuple[int, int, str]]] = {}
    
    for axis in axes:
        parsed = parse_sub_axis(axis)
        if parsed is None:
            continue
        parent, pre_size, size = parsed
        # Range covered: [pre_size, pre_size * size)
        start = pre_size
        end = pre_size * size
        
        if parent not in parent_ranges:
            parent_ranges[parent] = []
        
        # Check overlap with existing ranges
        for existing_start, existing_end, existing_axis in parent_ranges[parent]:
            # Ranges overlap if not (end <= existing_start or start >= existing_end)
            if not (end <= existing_start or start >= existing_end):
                raise ValueError(
                    f"Sub-axes overlap: '{axis}' and '{existing_axis}' "
                    f"(ranges [{start}, {end}) and [{existing_start}, {existing_end}))"
                )
        
        parent_ranges[parent].append((start, end, axis))


def check_sub_axes_maximality(axes: List[str]) -> List[str]:
    """Return warnings for adjacent sub-axes that could be merged."""
    warnings = []
    parent_sub_axes: Dict[str, List[Tuple[int, int, str]]] = {}
    
    for axis in axes:
        parsed = parse_sub_axis(axis)
        if parsed is None:
            continue
        parent, pre_size, size = parsed
        if parent not in parent_sub_axes:
            parent_sub_axes[parent] = []
        parent_sub_axes[parent].append((pre_size, size, axis))
    
    for parent, subs in parent_sub_axes.items():
        # Sort by pre_size (major to minor order)
        subs_sorted = sorted(subs, key=lambda x: x[0])
        for i in range(len(subs_sorted) - 1):
            pre1, size1, ax1 = subs_sorted[i]
            pre2, size2, ax2 = subs_sorted[i + 1]
            # Adjacent if pre1 * size1 == pre2
            if pre1 * size1 == pre2:
                merged_size = size1 * size2
                warnings.append(
                    f"Adjacent sub-axes '{ax1}' and '{ax2}' could be merged "
                    f"into '{parent}:({pre1}){merged_size}'"
                )
    
    return warnings


# --- Physical Layer: Device Mesh ---

class DeviceMesh:
    """Logical multi-dimensional view of devices: @name = <["axis1"=size1, ...]>.
    
    Args:
        name: Name of the mesh
        shape: Shape of the mesh (e.g., (2, 4) for 2x4 grid)
        axis_names: Names for each axis (e.g., ("x", "y"))
        devices: Logical device IDs (default: sequential 0..N-1)
        device_refs: Physical device references (default: all CPU)
    """
    
    def __init__(self, name: str, shape: Tuple[int, ...], axis_names: Tuple[str, ...], 
                 devices: List[int] = None, device_refs: List = None):
        self.name = name
        self.shape = shape
        self.axis_names = axis_names
        
        # Default to sequential device IDs if not specified
        total_devices = int(np.prod(shape))
        if devices is None:
            devices = list(range(total_devices))
        self.devices = devices
        
        if total_devices != len(devices):
            raise ValueError(
                f"Mesh shape {shape} requires {total_devices} devices, "
                f"but got {len(devices)}"
            )
        
        # Physical device references (for distributed execution)
        if device_refs is None:
            from max.graph import DeviceRef
            device_refs = [DeviceRef.CPU() for _ in range(total_devices)]
        self.device_refs = device_refs
        
        # Lookup table for axis name -> index
        self.axis_lookup = {name: i for i, name in enumerate(axis_names)}
        
        # Compute strides for coordinate calculation (row-major order)
        self.phys_strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            self.phys_strides[i] = shape[i+1] * self.phys_strides[i+1]
    
    def __repr__(self) -> str:
        axes_str = ", ".join(f'"{n}"={s}' for n, s in zip(self.axis_names, self.shape))
        return f"@{self.name} = <[{axes_str}]>"

    def get_axis_size(self, axis_name: str) -> int:
        """Get size of an axis. For sub-axes 'x:(m)k', returns k."""
        parsed = parse_sub_axis(axis_name)
        if parsed:
            return parsed[2]  # Return the size k
        
        if axis_name not in self.axis_lookup:
            raise ValueError(f"Unknown axis: {axis_name} (available: {self.axis_names})")
        return self.shape[self.axis_lookup[axis_name]]

    def get_coordinate(self, device_id: int, axis_name: str) -> int:
        """Get coordinate of device along axis. Handles sub-axes 'x:(m)k'."""
        # Handle Sub-Axes
        parsed = parse_sub_axis(axis_name)
        if parsed:
            parent_name, pre_size, size = parsed
            parent_coord = self.get_coordinate(device_id, parent_name)
            parent_total = self.get_axis_size(parent_name)
            
            if pre_size * size == 0:
                raise ValueError(f"Invalid sub-axis sizes: pre_size={pre_size}, size={size}")
            if parent_total % (pre_size * size) != 0:
                raise ValueError(
                    f"Sub-axis {axis_name} invalid: parent size {parent_total} "
                    f"not divisible by pre_size*size = {pre_size * size}"
                )
            
            post_size = parent_total // (pre_size * size)
            return (parent_coord // post_size) % size

        # Handle Full Axes
        if axis_name not in self.axis_lookup:
            raise ValueError(f"Unknown axis: {axis_name}")

        if device_id not in self.devices: 
            raise ValueError(f"Device {device_id} not in mesh {self.name}")
            
        flat_idx = self.devices.index(device_id)
        
        # Convert flat index to n-dimensional coordinates
        coords = []
        rem = flat_idx
        for stride in self.phys_strides:
            coords.append(rem // stride)
            rem %= stride
            
        return coords[self.axis_lookup[axis_name]]
    
    def get_devices_on_axis_slice(self, axis_name: str, coordinate: int) -> List[int]:
        """Get all device IDs that have the given coordinate on the specified axis."""
        return [d for d in self.devices if self.get_coordinate(d, axis_name) == coordinate]


# --- Representation Layer: Sharding Specification ---

@dataclass
class DimSpec:
    """
    Per-dimension sharding specification.
    
    - axes: Axis names that shard this dim (major to minor order)
    - is_open: If True, can be further sharded; if False (closed), sharding is fixed
    - priority: Lower = stronger (0 is default/strongest)
    
    Examples: {"x"} (closed), {"x", ?} (open), {"x"}p1 (priority 1)
    """
    axes: List[str] = field(default_factory=list)
    is_open: bool = False  # Open vs Closed dimension (from Shardy spec)
    priority: int = 0      # 0=User/default (Strong), 1+=Lower priority
    
    def __post_init__(self):
        # Empty closed dims can't have non-zero priority (no effect per spec)
        if not self.axes and not self.is_open and self.priority != 0:
            raise ValueError(
                f"Empty closed dimension {{}} cannot have non-zero priority (got p{self.priority})"
            )
    
    def __repr__(self) -> str:
        """Format: {axes} or {axes, ?} with optional p<N> suffix."""
        if not self.axes:
            marker = "?" if self.is_open else ""
            # Per spec: empty closed dim shouldn't show priority (and can't have non-zero)
            prio_str = f"p{self.priority}" if self.is_open and self.priority != 0 else ""
            return f"{{{marker}}}{prio_str}"
        
        axes_str = ", ".join(f'"{a}"' for a in self.axes)
        open_marker = ", ?" if self.is_open else ""
        # Priority 0 is default, only show non-zero priorities per spec convention
        prio_str = f"p{self.priority}" if self.priority != 0 else ""
        return f"{{{axes_str}{open_marker}}}{prio_str}"
    
    def is_replicated(self) -> bool:
        """Returns True if this dimension is fully replicated (no sharding axes)."""
        return len(self.axes) == 0
    
    def get_total_shards(self, mesh: 'DeviceMesh') -> int:
        """Calculate total number of shards for this dimension."""
        total = 1
        for axis in self.axes:
            total *= mesh.get_axis_size(axis)
        return total


@dataclass
class ShardingSpec:
    """
    Complete tensor sharding: sharding<@mesh, [dim_shardings], replicated={axes}>.
    
    - dim_specs: Per-dimension specs (must match tensor rank)
    - replicated_axes: Explicitly replicated axes (can't shard any dimension)
    """
    mesh: DeviceMesh
    dim_specs: List[DimSpec]
    replicated_axes: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Validate: no duplicate axes, no explicit-replicated axes in dims."""
        used_axes = set()
        all_axes = []
        
        for dim_idx, dim in enumerate(self.dim_specs):
            for axis in dim.axes:
                # Check: axis not in explicitly replicated set
                if axis in self.replicated_axes:
                    raise ValueError(
                        f"Axis '{axis}' is explicitly replicated but assigned to dimension {dim_idx}."
                    )
                
                # Collect for duplicate/overlap checking
                all_axes.append(axis)
                
                # Simple duplicate check (exact match)
                if axis in used_axes: 
                    raise ValueError(f"Axis '{axis}' used multiple times in sharding.")
                used_axes.add(axis)
        
        # Check sub-axis overlap invariant
        validate_sub_axes_non_overlapping(all_axes)
        
        # Warn about non-maximal sub-axes (could be merged)
        # Note: In a production environment, this might log a warning
        check_sub_axes_maximality(all_axes)
    
    def __repr__(self) -> str:
        """
        String representation following Shardy spec grammar.
        
        From spec:
            sharding<@mesh_name, dim_shardings, replicated=replicated_axes>
        
        Note: replicated={} is omitted when empty per spec convention.
        """
        dims_str = ", ".join(str(d) for d in self.dim_specs)
        rep_str = ""
        if self.replicated_axes:
            # Order replicated axes by mesh order (per Shardy spec)
            ordered_rep = self._order_replicated_axes()
            rep_str = ", replicated={" + ", ".join(f'"{a}"' for a in ordered_rep) + "}"
        return f"sharding<@{self.mesh.name}, [{dims_str}]{rep_str}>"
    
    def _order_replicated_axes(self) -> List[str]:
        """Order replicated axes: mesh order, sub-axes by pre-size."""
        full_axes = []
        sub_axes_by_parent: Dict[str, List[Tuple[str, int, int]]] = {}  # parent -> [(axis_str, pre, size)]
        
        for ax in self.replicated_axes:
            parsed = parse_sub_axis(ax)
            if parsed is None:
                full_axes.append(ax)
            else:
                parent, pre_size, size = parsed
                if parent not in sub_axes_by_parent:
                    sub_axes_by_parent[parent] = []
                sub_axes_by_parent[parent].append((ax, pre_size, size))
        
        result = []
        
        # Process in mesh axis order
        for ax_name in self.mesh.axis_names:
            # Add full axis if present
            if ax_name in full_axes:
                result.append(ax_name)
            
            # Add sub-axes for this parent, sorted by pre-size (ascending)
            if ax_name in sub_axes_by_parent:
                sorted_subs = sorted(sub_axes_by_parent[ax_name], key=lambda x: x[1])
                result.extend(ax_str for ax_str, _, _ in sorted_subs)
        
        return result
    
    def get_implicitly_replicated_axes(self) -> Set[str]:
        """
        Get axes that are implicitly replicated (not used, not explicitly replicated).
        
        From spec:
            "All axes that are not used to shard a dimension are implicitly replicated."
        """
        used = set()
        for dim in self.dim_specs:
            used.update(dim.axes)
        used.update(self.replicated_axes)
        
        implicit = set()
        for ax_name in self.mesh.axis_names:
            if ax_name not in used:
                implicit.add(ax_name)
        return implicit
    
    def is_fully_replicated(self) -> bool:
        """Returns True if tensor is fully replicated (no dimension sharding)."""
        return all(dim.is_replicated() for dim in self.dim_specs)
    
    def get_max_priority(self) -> int:
        """Get the maximum (lowest urgency) priority used in this spec."""
        return max((d.priority for d in self.dim_specs), default=0)
    
    def get_min_priority(self) -> int:
        """Get the minimum (highest urgency) priority used in this spec."""
        return min((d.priority for d in self.dim_specs), default=0)


def compute_local_shape(
    global_shape: Tuple[int, ...],
    sharding: ShardingSpec,
    device_id: int,
) -> Tuple[int, ...]:
    """Compute the local shard shape for a device.
    
    Args:
        global_shape: The global tensor shape
        sharding: The sharding specification
        device_id: The device ID to compute shape for
        
    Returns:
        The local shape on that device
        
    Example:
        >>> mesh = DeviceMesh("m", (2,), ("x",))
        >>> spec = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
        >>> compute_local_shape((8, 4), spec, device_id=0)
        (4, 4)  # First half of dim 0
    """
    local_shape = []
    for dim_idx in range(len(global_shape)):
        if dim_idx >= len(sharding.dim_specs):
            # Implicitly replicated if dim spec missing
            local_shape.append(global_shape[dim_idx])
            continue

        dim_spec = sharding.dim_specs[dim_idx]
        global_len = global_shape[dim_idx]
        
        # Fully replicated dimension
        if not dim_spec.axes:
            local_shape.append(global_len)
            continue

        # Calculate shard index from major-to-minor axis coordinates
        total_shards = 1
        my_shard_index = 0
        
        for axis_name in dim_spec.axes:
            size = sharding.mesh.get_axis_size(axis_name)
            coord = sharding.mesh.get_coordinate(device_id, axis_name)
            my_shard_index = (my_shard_index * size) + coord
            total_shards *= size
        
        # Compute chunk boundaries (ceiling division for uneven splits)
        chunk_size = math.ceil(global_len / total_shards)
        start = my_shard_index * chunk_size
        theoretical_end = start + chunk_size
        real_end = min(theoretical_end, global_len)
        
        # Handle padding
        length = max(0, real_end - start)
        # In a real implementation we might need to handle padding
        local_shape.append(length)

    return tuple(local_shape)


def get_num_shards(sharding: ShardingSpec) -> int:
    """Get the total number of shards for this sharding spec.
    
    This equals the total number of devices in the mesh.
    
    Args:
        sharding: The sharding specification
        
    Returns:
        Number of shards (devices)
    """
    return len(sharding.mesh.devices)
