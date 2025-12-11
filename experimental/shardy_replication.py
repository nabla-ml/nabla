"""
Shardy Sharding Representation and Propagation Implementation
=============================================================

This module implements the Shardy sharding representation and propagation system,
based on the XLA/Shardy specification. It provides:

1. **Physical Layer**: DeviceMesh for logical device organization
2. **Representation Layer**: ShardingSpec with DimSpec for tensor sharding
3. **Execution Layer**: DistributedTensor for actual data distribution
4. **Propagation Layer**: Factor-based sharding propagation with conflict resolution
5. **Graph Layer**: Operations and ShardingPass for whole-program propagation

Key Concepts from Shardy Spec:
- Logical mesh: Multi-dimensional view of devices with named axes
- Dimension sharding: Per-dimension specification with axes, open/closed status, priority
- Sub-axes: Splitting mesh axes for finer-grained sharding control
- Factor-based propagation: Abstract dimension relationships via einsum-like factors
- Conflict resolution hierarchy: User priorities > Op priorities > Aggressive > Basic

References:
- Sharding Representation: https://openxla.org/shardy/sharding_representation
- Sharding Propagation: https://openxla.org/shardy/propagation
"""

import math
import re
import unittest
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Type aliases for clarity
FactorName = str
AxisName = str


class OpPriority(IntEnum):
    """Operation-based priority for propagation ordering (lower = higher priority)."""
    PASSTHROUGH = 0    # Element-wise, reshape
    CONTRACTION = 1    # Matmul, dot, einsum
    REDUCTION = 2      # Reduce, scan
    COMMUNICATION = 3  # Collectives


class PropagationStrategy(IntEnum):
    """Conflict resolution strategy (BASIC = no conflicts, AGGRESSIVE = resolve conflicts)."""
    BASIC = 0
    AGGRESSIVE = 1

# --- Physical Layer: Device Mesh ---

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


class DeviceMesh:
    """Logical multi-dimensional view of devices: @name = <["axis1"=size1, ...]>."""
    
    def __init__(self, name: str, shape: Tuple[int, ...], axis_names: Tuple[str, ...], 
                 devices: List[int] = None):
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
        warnings = check_sub_axes_maximality(all_axes)
        for w in warnings:
            # In production, this could be a proper warning/log
            pass  # print(f"Warning: {w}")
    
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

# --- Execution Layer: Distributed Tensor ---

class DistributedTensor:
    """A tensor distributed across devices according to a ShardingSpec."""
    
    def __init__(self, global_shape: Tuple[int, ...], spec: ShardingSpec, dtype_size: int = 4):
        self.global_shape = global_shape
        self.spec = spec
        self.mesh = spec.mesh
        self.dtype_size = dtype_size  # Bytes per element (4 for float32)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.global_shape

    def get_local_shape(self, device_id: int) -> Tuple[int, ...]:
        """Get the local tensor shape on a specific device."""
        local_shape = []
        for dim_idx in range(len(self.global_shape)):
            start, end, padding = self.get_local_interval(dim_idx, device_id)
            local_shape.append(end - start + padding)
        return tuple(local_shape)

    def get_local_interval(self, dim_idx: int, device_id: int) -> Tuple[int, int, int]:
        """Return (start, end, padding) for this dimension on the given device."""
        # Handle dimensions beyond spec (implicit replication for scalars, etc.)
        if dim_idx >= len(self.spec.dim_specs):
            return 0, self.global_shape[dim_idx], 0

        dim_spec = self.spec.dim_specs[dim_idx]
        global_len = self.global_shape[dim_idx]
        
        # Fully replicated dimension
        if not dim_spec.axes:
            return 0, global_len, 0

        # Calculate shard index from major-to-minor axis coordinates
        total_shards = 1
        my_shard_index = 0
        
        for axis_name in dim_spec.axes:
            size = self.mesh.get_axis_size(axis_name)
            coord = self.mesh.get_coordinate(device_id, axis_name)
            my_shard_index = (my_shard_index * size) + coord
            total_shards *= size
        
        # Compute chunk boundaries (ceiling division for uneven splits)
        chunk_size = math.ceil(global_len / total_shards)
        start = my_shard_index * chunk_size
        theoretical_end = start + chunk_size
        real_end = min(theoretical_end, global_len)
        
        # Handle padding for uneven sharding
        padding = 0
        if start >= global_len:
            # This shard is entirely padding
            return 0, 0, chunk_size
        if theoretical_end > global_len:
            # Partial padding needed
            padding = theoretical_end - global_len

        return start, real_end, padding

    def get_byte_size(self) -> int:
        """Get total byte size of the global tensor."""
        return int(np.prod(self.global_shape)) * self.dtype_size
    
    def get_local_byte_size(self, device_id: int) -> int:
        """Get byte size of the local shard on a device."""
        local_shape = self.get_local_shape(device_id)
        return int(np.prod(local_shape)) * self.dtype_size

# --- Propagation Layer ---

# =============================================================================
# FACTOR SHARDING: Intermediate Representation for Propagation
# =============================================================================
# 
# The key insight from Shardy is to propagate in "factor space" rather than
# directly between dimensions. This provides:
#   1. A clean abstraction that decouples op-specific logic from propagation
#   2. Natural handling of compound factors (e.g., reshape splitting dims)
#   3. Clear conflict resolution semantics
#
# The propagation algorithm has 3 phases:
#   Phase 1 (COLLECT): Project DimSharding -> FactorSharding
#   Phase 2 (RESOLVE): Merge/resolve conflicts in factor space  
#   Phase 3 (UPDATE):  Project FactorSharding -> DimSharding

@dataclass
class FactorSharding:
    """
    Sharding state for a single factor during propagation.
    
    This is the intermediate representation between tensor dimensions and the
    propagation algorithm. It captures:
    - Which mesh axes shard this factor
    - The priority level (lower = stronger constraint)
    - Whether the factor is "open" (can accept more sharding) or "closed" (fixed)
    
    Semantics (from Shardy spec):
    - Empty + Open: "I have no opinion, tell me what sharding to use" (receptive)
    - Empty + Closed: "I explicitly want replication, don't shard me" (assertive)
    - Non-empty: "I want this specific sharding" (assertive)
    
    Attributes:
        axes: Mesh axes assigned to this factor (major to minor order)
        priority: Priority level (0 = strongest/user-specified, higher = weaker)
        is_open: If True, can accept additional sharding; if False, fixed
        is_explicit_replication: True if empty+closed (explicit replication constraint)
    """
    axes: List[str] = field(default_factory=list)
    priority: int = 999  # Default: weakest priority (unspecified)
    is_open: bool = True
    
    @property
    def is_explicit_replication(self) -> bool:
        """True if this represents an explicit replication constraint (empty + closed)."""
        return not self.axes and not self.is_open
    
    @property
    def is_receptive(self) -> bool:
        """True if this factor can receive sharding from others (empty + open)."""
        return not self.axes and self.is_open
    
    @property
    def has_sharding(self) -> bool:
        """True if this factor has actual sharding axes."""
        return bool(self.axes)
    
    def copy(self) -> 'FactorSharding':
        """Create a deep copy of this factor sharding."""
        return FactorSharding(
            axes=list(self.axes),
            priority=self.priority,
            is_open=self.is_open
        )
    
    def __repr__(self) -> str:
        axes_str = ",".join(self.axes) if self.axes else "∅"
        status = "open" if self.is_open else ("repl" if self.is_explicit_replication else "closed")
        return f"FactorSharding({axes_str}, p{self.priority}, {status})"


@dataclass
class FactorShardingState:
    """
    Complete factor sharding state for an operation during propagation.
    
    This holds the sharding state for all factors in an OpShardingRule,
    serving as the working space for the propagation algorithm.
    
    Usage:
        state = FactorShardingState()
        state.get_or_create("m")  # Get/create factor "m"
        state.merge("m", new_axes, new_priority, new_is_open, mesh)  # Merge info
    """
    factors: Dict[str, FactorSharding] = field(default_factory=dict)
    
    def get_or_create(self, factor_name: str) -> FactorSharding:
        """Get existing factor or create a new receptive one."""
        if factor_name not in self.factors:
            self.factors[factor_name] = FactorSharding()
        return self.factors[factor_name]
    
    def get(self, factor_name: str) -> Optional[FactorSharding]:
        """Get factor if it exists, else None."""
        return self.factors.get(factor_name)
    
    def merge(
        self,
        factor_name: str,
        new_axes: List[str],
        new_priority: int,
        new_is_open: bool,
        mesh: 'DeviceMesh',
        strategy: 'PropagationStrategy' = None,
    ) -> None:
        """
        Merge new sharding information into a factor.
        
        Implements the Shardy conflict resolution semantics:
        1. Empty+open (receptive) doesn't override anything
        2. Explicit replication (empty+closed) with higher priority wins
        3. Lower priority number = stronger constraint
        4. Same priority: use strategy (BASIC=common prefix, AGGRESSIVE=max parallelism)
        
        Args:
            factor_name: Name of the factor to merge into
            new_axes: Axes from the new dimension
            new_priority: Priority of the new dimension
            new_is_open: Open/closed status of the new dimension
            mesh: Device mesh for parallelism calculations
            strategy: Conflict resolution strategy (default: BASIC)
        """
        if strategy is None:
            strategy = PropagationStrategy.BASIC
            
        factor = self.get_or_create(factor_name)
        
        has_new = bool(new_axes)
        has_existing = factor.has_sharding
        new_is_receptive = not has_new and new_is_open
        new_is_explicit_repl = not has_new and not new_is_open
        
        # Case 1: New is receptive (empty+open) -> don't change anything
        # Receptive means "I have no opinion", so we don't override
        if new_is_receptive:
            return
        
        # Case 2: Existing is explicit replication with equal/stronger priority
        # Explicit replication is assertive, so it blocks weaker constraints
        if factor.is_explicit_replication and new_priority >= factor.priority:
            return
        
        # Case 3: New is explicit replication with stronger priority
        # This overrides whatever was there before
        if new_is_explicit_repl and new_priority < factor.priority:
            factor.axes = []
            factor.priority = new_priority
            factor.is_open = False
            return
        
        # Case 4: New is explicit replication with equal priority
        # Explicit replication at same priority also wins (conservative)
        if new_is_explicit_repl and new_priority == factor.priority:
            factor.axes = []
            factor.is_open = False
            return
        
        # Case 5: New has axes, existing is receptive -> contribute axes
        if has_new and not has_existing and factor.is_open:
            factor.axes = list(new_axes)
            factor.priority = new_priority
            factor.is_open = new_is_open
            return
        
        # Case 6: Both have axes -> priority-based resolution
        if has_new and has_existing:
            if new_priority < factor.priority:
                # New has stronger priority -> override
                factor.axes = list(new_axes)
                factor.priority = new_priority
                factor.is_open = new_is_open
            elif new_priority == factor.priority:
                # Same priority -> strategy-based resolution
                if strategy == PropagationStrategy.AGGRESSIVE:
                    # Pick sharding with more parallelism
                    old_par = self._get_parallelism(factor.axes, mesh)
                    new_par = self._get_parallelism(new_axes, mesh)
                    if new_par > old_par:
                        factor.axes = list(new_axes)
                        factor.is_open = new_is_open
                else:
                    # BASIC: take longest common prefix (conservative)
                    common = self._longest_common_prefix(factor.axes, new_axes)
                    factor.axes = common
                    factor.is_open = factor.is_open or new_is_open
            # else: new_priority > factor.priority -> keep existing (stronger)
            return
        
        # Case 7: New has axes but weaker priority -> only update receptive factors
        if has_new and new_priority > factor.priority:
            # Keep existing (it has stronger priority)
            return
        
        # Case 8: New has axes, factor was receptive -> contribute
        if has_new:
            factor.axes = list(new_axes)
            factor.priority = new_priority
            factor.is_open = new_is_open
    
    @staticmethod
    def _longest_common_prefix(list1: List[str], list2: List[str]) -> List[str]:
        """Find longest common prefix of two axis lists."""
        common = []
        for x, y in zip(list1, list2):
            if x == y:
                common.append(x)
            else:
                break
        return common
    
    @staticmethod
    def _get_parallelism(axes: List[str], mesh: 'DeviceMesh') -> int:
        """Calculate total parallelism (product of axis sizes)."""
        if not axes:
            return 1
        total = 1
        for ax in axes:
            total *= mesh.get_axis_size(ax)
        return total
    
    def __repr__(self) -> str:
        lines = ["FactorShardingState:"]
        for name, fs in sorted(self.factors.items()):
            lines.append(f"  {name}: {fs}")
        return "\n".join(lines)


@dataclass
class OpShardingRule:
    """
    Einsum-like factor mapping defining how shardings propagate through an operation.
    
    This is the core abstraction that allows Shardy to handle any operation uniformly.
    Each tensor dimension maps to one or more "factors", and factors are the unit of
    propagation. Dimensions that share a factor must have compatible shardings.
    
    Example for C = A @ B (matmul):
        A: (m, k), B: (k, n) -> C: (m, n)
        
        input_mappings = [{0: ["m"], 1: ["k"]}, {0: ["k"], 1: ["n"]}]
        output_mappings = [{0: ["m"], 1: ["n"]}]
        factor_sizes = {"m": 4, "k": 8, "n": 16}
        
        Key insight: factor "k" appears in both A and B (contraction dimension)
        but NOT in C. This naturally encodes that k's sharding doesn't propagate
        to the output (it's reduced over).
    
    Compound Factors (for reshape):
        When a dimension maps to multiple factors, it represents a "split":
        input_mappings = [{0: ["f1", "f2"]}]  # dim 0 -> factors f1, f2
        output_mappings = [{0: ["f1"], 1: ["f2"]}]  # f1 -> dim 0, f2 -> dim 1
        
        If dim 0 is sharded on axis "x" of size 8, and f1=2, f2=4, then:
        - f1 gets sub-axis x:(1)2 (major portion)
        - f2 gets sub-axis x:(2)4 (minor portion)
    
    Attributes:
        input_mappings: Per-input-tensor mapping from dim_idx -> factor names
        output_mappings: Per-output-tensor mapping from dim_idx -> factor names
        factor_sizes: Size of each factor (for sub-axis decomposition)
    """
    input_mappings: List[Dict[int, List[str]]]  
    output_mappings: List[Dict[int, List[str]]] 
    factor_sizes: Dict[str, int]
    
    def get_all_factors(self) -> Set[str]:
        """Get all factor names used in this rule."""
        factors = set()
        for mapping in self.input_mappings + self.output_mappings:
            for factor_list in mapping.values():
                factors.update(factor_list)
        return factors
    
    def get_factor_tensors(self, factor_name: str) -> List[Tuple[str, int, int]]:
        """
        Get all (tensor_type, tensor_idx, dim_idx) tuples where a factor appears.
        
        Useful for understanding which dimensions are connected by a factor.
        """
        results = []
        for t_idx, mapping in enumerate(self.input_mappings):
            for dim_idx, factors in mapping.items():
                if factor_name in factors:
                    results.append(("input", t_idx, dim_idx))
        for t_idx, mapping in enumerate(self.output_mappings):
            for dim_idx, factors in mapping.items():
                if factor_name in factors:
                    results.append(("output", t_idx, dim_idx))
        return results
    
    def to_einsum_notation(self) -> str:
        """
        Convert rule to human-readable einsum-like notation.
        
        Example output: "(m, k), (k, n) -> (m, n)"
        """
        def mapping_to_str(mapping: Dict[int, List[str]]) -> str:
            if not mapping:
                return "()"
            sorted_dims = sorted(mapping.keys())
            parts = []
            for d in sorted_dims:
                factors = mapping[d]
                if len(factors) == 1:
                    parts.append(factors[0])
                else:
                    parts.append(f"({','.join(factors)})")
            return f"({', '.join(parts)})"
        
        inputs = ", ".join(mapping_to_str(m) for m in self.input_mappings)
        outputs = ", ".join(mapping_to_str(m) for m in self.output_mappings)
        return f"{inputs} -> {outputs}"


@dataclass
class OpShardingRuleTemplate:
    """
    Shape-agnostic sharding rule template.
    
    Factor sizes are NOT stored here — they're inferred from tensor shapes
    at instantiation time via `instantiate()`.
    
    Example for matmul:
        template = OpShardingRuleTemplate(
            input_mappings=[{0: ["m"], 1: ["k"]}, {0: ["k"], 1: ["n"]}],
            output_mappings=[{0: ["m"], 1: ["n"]}]
        )
        rule = template.instantiate([(4, 8), (8, 16)], [(4, 16)])
        # Infers: factor_sizes = {"m": 4, "k": 8, "n": 16}
    """
    input_mappings: List[Dict[int, List[str]]]
    output_mappings: List[Dict[int, List[str]]]
    
    def get_all_factors(self) -> Set[str]:
        """Get all factor names used in this template."""
        factors = set()
        for mapping in self.input_mappings + self.output_mappings:
            for factor_list in mapping.values():
                factors.update(factor_list)
        return factors
    
    def to_einsum_notation(self) -> str:
        """Convert to human-readable einsum-like notation."""
        def mapping_to_str(mapping: Dict[int, List[str]]) -> str:
            if not mapping:
                return "()"
            sorted_dims = sorted(mapping.keys())
            parts = []
            for d in sorted_dims:
                factors = mapping[d]
                if len(factors) == 1:
                    parts.append(factors[0])
                else:
                    parts.append(f"({','.join(factors)})")
            return f"({', '.join(parts)})"
        
        inputs = ", ".join(mapping_to_str(m) for m in self.input_mappings)
        outputs = ", ".join(mapping_to_str(m) for m in self.output_mappings)
        return f"{inputs} -> {outputs}"
    
    def instantiate(
        self,
        input_shapes: List[Tuple[int, ...]],
        output_shapes: List[Tuple[int, ...]]
    ) -> OpShardingRule:
        """
        Instantiate template with concrete shapes to infer factor sizes.
        
        Algorithm:
        1. First pass: infer sizes from single-factor dimensions (unambiguous)
        2. Second pass: verify compound factors multiply correctly, infer remaining
        """
        factor_sizes: Dict[str, int] = {}
        
        all_mappings_and_shapes = list(zip(self.input_mappings, input_shapes)) + \
                                   list(zip(self.output_mappings, output_shapes))
        
        # Pass 1: Infer from single-factor dimensions
        for mapping, shape in all_mappings_and_shapes:
            for dim_idx, factors in mapping.items():
                if dim_idx >= len(shape):
                    continue
                dim_size = shape[dim_idx]
                
                if len(factors) == 1:
                    f = factors[0]
                    if f in factor_sizes:
                        if factor_sizes[f] != dim_size:
                            raise ValueError(
                                f"Inconsistent size for factor '{f}': "
                                f"got {dim_size}, expected {factor_sizes[f]}"
                            )
                    else:
                        factor_sizes[f] = dim_size
        
        # Pass 2: Verify compound factors and infer remaining unknowns
        for mapping, shape in all_mappings_and_shapes:
            for dim_idx, factors in mapping.items():
                if dim_idx >= len(shape) or len(factors) <= 1:
                    continue
                dim_size = shape[dim_idx]
                
                known_product = 1
                unknown_factors = []
                
                for f in factors:
                    if f in factor_sizes:
                        known_product *= factor_sizes[f]
                    else:
                        unknown_factors.append(f)
                
                if not unknown_factors:
                    if known_product != dim_size:
                        raise ValueError(
                            f"Factor product {known_product} != dim size {dim_size} "
                            f"for factors {factors}"
                        )
                elif len(unknown_factors) == 1:
                    if dim_size % known_product != 0:
                        raise ValueError(
                            f"Cannot infer factor '{unknown_factors[0]}': "
                            f"dim_size {dim_size} not divisible by {known_product}"
                        )
                    factor_sizes[unknown_factors[0]] = dim_size // known_product
                else:
                    raise ValueError(
                        f"Cannot infer multiple unknown factors {unknown_factors} "
                        f"for dimension of size {dim_size}"
                    )
        
        return OpShardingRule(self.input_mappings, self.output_mappings, factor_sizes)


# =============================================================================
# PROPAGATION ALGORITHM
# =============================================================================
#
# The propagation algorithm has 3 phases (from Shardy spec):
#   Phase 1 (COLLECT): Project DimSharding -> FactorSharding
#   Phase 2 (RESOLVE): Merge/resolve conflicts in factor space  
#   Phase 3 (UPDATE):  Project FactorSharding -> DimSharding


def _expand_axes_for_factors(
    axes: List[str],
    factors: List[str],
    factor_sizes: Dict[str, int],
    mesh: 'DeviceMesh'
) -> List[str]:
    """
    Expand axes into sub-axes when one axis covers multiple factors.
    
    When a dimension is mapped to compound factors (e.g., dim -> [f1, f2]) and
    is sharded on a single axis whose size equals the product of factor sizes,
    we decompose the axis into sub-axes for each factor.
    
    Example:
        axes = ["x"]  (size 8)
        factors = ["f1", "f2"]  (sizes 2, 4)
        Result: ["x:(1)2", "x:(2)4"]
        
    The sub-axis notation "x:(pre)size" means:
        - parent axis is "x"
        - pre_size is the product of all major factors (1 for first)
        - size is the factor size
        
    Args:
        axes: Original axes from dimension sharding
        factors: Factor names this dimension maps to
        factor_sizes: Size of each factor
        mesh: Device mesh for axis size lookup
        
    Returns:
        Expanded axis list with sub-axes where appropriate
    """
    if not axes or not factors:
        return axes
    
    expanded = []
    curr_ax_idx = 0
    
    while curr_ax_idx < len(axes) and len(expanded) < len(factors):
        ax = axes[curr_ax_idx]
        ax_size = mesh.get_axis_size(ax)
        
        # Look ahead at remaining factors
        remaining_factors = factors[len(expanded):]
        if not remaining_factors:
            break
        
        # Check if next N factors multiply to exactly axis size
        cum_prod = 1
        sub_factors = []
        found_split = False
        
        for f in remaining_factors:
            f_size = factor_sizes.get(f, 1)
            cum_prod *= f_size
            sub_factors.append((f, f_size))
            
            if cum_prod == ax_size and len(sub_factors) > 1:
                # Match found! Decompose into sub-axes
                # Pre-size starts at 1 (major) and accumulates
                pre_size = 1
                for _, f_size in sub_factors:
                    # Notation: axis:(pre_size)size
                    expanded.append(f"{ax}:({pre_size}){f_size}")
                    pre_size *= f_size
                
                curr_ax_idx += 1
                found_split = True
                break
            
            if cum_prod > ax_size:
                # Exceeded axis size, no split possible here
                break
        
        if not found_split:
            expanded.append(ax)
            curr_ax_idx += 1
    
    return expanded


def _collect_to_factors(
    specs: List[ShardingSpec],
    mappings: List[Dict[int, List[str]]],
    rule: OpShardingRule,
    mesh: 'DeviceMesh',
    state: FactorShardingState,
    strategy: PropagationStrategy,
    max_priority: Optional[int],
) -> None:
    """
    Phase 1: Project dimension shardings to factor shardings (COLLECT).
    
    For each tensor dimension, extract its sharding and merge it into the
    corresponding factor(s) in the state.
    
    From Shardy spec: "Project DimSharding to FactorSharding"
    """
    for t_idx, spec in enumerate(specs):
        if t_idx >= len(mappings):
            continue
        mapping = mappings[t_idx]
        
        for dim_idx, factors in mapping.items():
            if dim_idx >= len(spec.dim_specs):
                continue
            dim_spec = spec.dim_specs[dim_idx]
            
            # Priority filtering: skip if priority > max_priority
            if max_priority is not None and dim_spec.priority > max_priority:
                continue
            
            # Handle sub-axis decomposition for compound factors
            expanded_axes = _expand_axes_for_factors(
                dim_spec.axes, factors, rule.factor_sizes, mesh
            )
            
            # Distribute axes to factors
            available_axes = list(expanded_axes)
            
            for f in factors:
                # Get axis for this factor (if any)
                axes_for_f = []
                if available_axes:
                    proposed_axis = available_axes.pop(0)
                    # Filter out explicitly replicated axes
                    if proposed_axis not in spec.replicated_axes:
                        axes_for_f = [proposed_axis]
                
                # Merge into factor state
                state.merge(
                    f,
                    axes_for_f,
                    dim_spec.priority,
                    dim_spec.is_open,
                    mesh,
                    strategy
                )


def _update_from_factors(
    specs: List[ShardingSpec],
    mappings: List[Dict[int, List[str]]],
    state: FactorShardingState,
) -> bool:
    """
    Phase 3: Project factor shardings back to dimension shardings (UPDATE).
    
    For each tensor dimension, gather sharding from its factor(s) and
    update if the priority hierarchy allows.
    
    From Shardy spec: "Project the updated FactorSharding to get the updated DimSharding"
    
    Returns True if any spec was modified.
    """
    did_change = False
    
    for t_idx, spec in enumerate(specs):
        if t_idx >= len(mappings):
            continue
        mapping = mappings[t_idx]
        new_dim_specs = []
        spec_dirty = False
        
        for dim_idx, current_dim in enumerate(spec.dim_specs):
            factors = mapping.get(dim_idx, [])
            
            if not factors:
                new_dim_specs.append(current_dim)
                continue
            
            # Gather sharding from all factors for this dimension
            proposed_axes = []
            proposed_prio = 999
            proposed_open = current_dim.is_open
            has_factor_info = False
            
            for f in factors:
                f_state = state.get(f)
                if f_state is not None:
                    # Filter out axes that are explicitly replicated on target
                    valid_axes = [ax for ax in f_state.axes 
                                 if ax not in spec.replicated_axes]
                    proposed_axes.extend(valid_axes)
                    proposed_prio = min(proposed_prio, f_state.priority)
                    has_factor_info = True
            
            if not has_factor_info:
                new_dim_specs.append(current_dim)
                continue
            
            # Determine if update should happen based on priority hierarchy
            should_update = _should_update_dim(
                current_dim, proposed_axes, proposed_prio
            )
            
            if should_update:
                new_dim_specs.append(DimSpec(
                    axes=proposed_axes,
                    is_open=proposed_open,
                    priority=proposed_prio
                ))
                spec_dirty = True
            else:
                new_dim_specs.append(current_dim)
        
        if spec_dirty:
            spec.dim_specs = new_dim_specs
            did_change = True
    
    return did_change


def _should_update_dim(
    current: DimSpec,
    proposed_axes: List[str],
    proposed_priority: int,
) -> bool:
    """
    Determine if a dimension should be updated based on Shardy semantics.
    
    From Shardy spec:
        "priorities determine in which order per-dimension sharding constraints 
        will be propagated" and "propagation won't override user defined shardings 
        with lower priority (>i)"
    
    Key semantics:
        - Lower priority number = higher urgency/strength (0 is strongest)
        - Open dimensions can be extended; closed dimensions are fixed
        - Empty open dimensions accept any propagation
        
    Args:
        current: Current dimension spec
        proposed_axes: Axes proposed by factor propagation
        proposed_priority: Priority from factor propagation
        
    Returns:
        True if dimension should be updated
    """
    # Higher priority (lower number) ALWAYS wins
    if proposed_priority < current.priority:
        return True
    
    # Same priority: open dimensions can be further sharded
    if proposed_priority == current.priority:
        if current.is_open:
            # Empty open dimension accepts any sharding
            if not current.axes and proposed_axes:
                return True
            # Open dimensions can accept longer shardings if compatible
            if len(proposed_axes) > len(current.axes):
                if proposed_axes[:len(current.axes)] == current.axes:
                    return True
    
    # Lower priority (higher number) can fill empty open dimensions
    if proposed_priority > current.priority:
        if current.is_open and not current.axes and proposed_axes:
            return True
    
    return False


def propagate_sharding(
    rule: OpShardingRule,
    input_specs: List[ShardingSpec],
    output_specs: List[ShardingSpec],
    strategy: PropagationStrategy = PropagationStrategy.BASIC,
    max_priority: Optional[int] = None,
) -> bool:
    """
    Propagate shardings between inputs/outputs using the factor-based algorithm.
    
    This implements the 3-phase Shardy propagation algorithm:
        Phase 1 (COLLECT): Project DimSharding -> FactorSharding
        Phase 2 (RESOLVE): Conflicts resolved during merge in Phase 1
        Phase 3 (UPDATE):  Project FactorSharding -> DimSharding
    
    Returns True if any spec was modified (for fixed-point iteration).
    
    Args:
        rule: The operation's sharding rule defining factor mappings
        input_specs: Sharding specs for input tensors
        output_specs: Sharding specs for output tensors  
        strategy: BASIC (conservative common prefix) or AGGRESSIVE (max parallelism)
        max_priority: If set, only propagate shardings with priority <= max_priority
    
    Example:
        # Matmul: C = A @ B
        rule = OpShardingRule(
            [{0: ["m"], 1: ["k"]}, {0: ["k"], 1: ["n"]}],  # A, B
            [{0: ["m"], 1: ["n"]}],  # C
            {"m": 4, "k": 8, "n": 16}
        )
        # If A is sharded on ["x"] for dim 0, factor "m" gets ["x"],
        # and C's dim 0 (also mapped to "m") receives ["x"].
    """
    if not input_specs and not output_specs:
        return False
    
    mesh = input_specs[0].mesh if input_specs else output_specs[0].mesh
    
    # Phase 1: Collect dimension shardings into factor space
    state = FactorShardingState()
    
    _collect_to_factors(
        input_specs, rule.input_mappings, rule, mesh, state, strategy, max_priority
    )
    _collect_to_factors(
        output_specs, rule.output_mappings, rule, mesh, state, strategy, max_priority
    )
    
    # Phase 2: Conflict resolution happens during merge (in _collect_to_factors)
    # No additional step needed here - resolution is eager
    
    # Phase 3: Project factors back to dimensions
    changed = False
    
    if _update_from_factors(input_specs, rule.input_mappings, state):
        changed = True
    if _update_from_factors(output_specs, rule.output_mappings, state):
        changed = True
    
    return changed

# --- Graph & Compiler Layer ---

class GraphTensor(DistributedTensor):
    """A named tensor in the computation graph with associated sharding."""
    
    def __init__(self, name: str, shape: Tuple[int, ...], mesh: DeviceMesh, 
                 initial_spec: ShardingSpec = None):
        if initial_spec is None:
            # Default: fully open, unspecified sharding
            # From Shardy spec: "no sharding attribute on a tensor is equivalent 
            # to a fully open tensor sharding"
            initial_spec = ShardingSpec(
                mesh, 
                [DimSpec([], is_open=True) for _ in shape]
            )
        super().__init__(shape, initial_spec)
        self.name = name
    
    def __repr__(self) -> str:
        return f"GraphTensor('{self.name}', shape={self.shape}, spec={self.spec})"


@dataclass
class Operation:
    """An operation connecting input/output GraphTensors through an OpShardingRule."""
    name: str
    rule: OpShardingRule
    inputs: List[GraphTensor]
    outputs: List[GraphTensor]
    op_priority: int = OpPriority.CONTRACTION  # Default to medium priority
    
    def __repr__(self) -> str:
        in_names = [t.name for t in self.inputs]
        out_names = [t.name for t in self.outputs]
        return f"Operation('{self.name}', inputs={in_names}, outputs={out_names})"


@dataclass
class DataFlowEdge:
    """
    Bridge between sources and targets that must share the same sharding.
    
    Used for control flow ops (while, case, etc.) where multiple tensors
    must be identically sharded.
    """
    sources: List[GraphTensor]
    targets: List[GraphTensor]
    owner: Optional[GraphTensor] = None
    
    def __post_init__(self):
        if self.owner is None and self.targets:
            self.owner = self.targets[0]
    
    def to_identity_rule(self) -> OpShardingRule:
        """Convert to identity sharding rule (all dims map to same factors)."""
        if not self.sources:
            return OpShardingRule([], [], {})
        
        # Use the first source's shape to define factors
        ref_shape = self.sources[0].shape
        factor_names = [f"df{i}" for i in range(len(ref_shape))]
        factor_sizes = {f"df{i}": ref_shape[i] for i in range(len(ref_shape))}
        
        # All tensors map identically: dim_i -> factor_i
        def make_mapping(tensor: GraphTensor) -> Dict[int, List[str]]:
            return {i: [factor_names[i]] for i in range(len(tensor.shape))}
        
        input_mappings = [make_mapping(t) for t in self.sources]
        output_mappings = [make_mapping(t) for t in self.targets]
        
        return OpShardingRule(input_mappings, output_mappings, factor_sizes)


class ShardingPass:
    """
    Compiler pass that propagates shardings to fixed point.
    
    Iterates through operations and data flow edges, propagating shardings
    until no changes occur or max_iterations reached.
    
    From Shardy spec, conflict resolution hierarchy:
      1. User priorities: propagate priority <=i before priority >i
      2. Op priorities: pass-through > contraction > reduction > communication
      3. Aggressive: resolve conflicts by picking higher parallelism
      4. Basic: conservative, only propagate compatible axes
    """
    
    def __init__(self, ops: List[Operation], 
                 data_flow_edges: List[DataFlowEdge] = None,
                 max_iterations: int = 20,
                 use_op_priority: bool = False,
                 use_priority_iteration: bool = False,
                 strategy: PropagationStrategy = PropagationStrategy.BASIC):
        self.ops = ops
        self.data_flow_edges = data_flow_edges or []
        self.max_iterations = max_iterations
        self.use_op_priority = use_op_priority
        self.use_priority_iteration = use_priority_iteration
        self.strategy = strategy
        self.iteration_count = 0

    def _get_sorted_ops(self) -> List[Operation]:
        """Sort ops by priority if enabled (passthrough > contraction > reduction)."""
        if not self.use_op_priority:
            return self.ops
        return sorted(self.ops, key=lambda op: op.op_priority)
    
    def _get_user_priority_levels(self) -> List[int]:
        """Get all user priority levels used across tensors (sorted)."""
        priorities = set()
        for op in self.ops:
            for t in op.inputs + op.outputs:
                for dim in t.spec.dim_specs:
                    priorities.add(dim.priority)
        return sorted(priorities)

    def run_pass(self) -> int:
        """
        Run propagation to fixed point. Returns iteration count.
        
        If use_priority_iteration is True, implements the Shardy spec's
        priority-based iteration: "at iteration i we propagate all dimension
        shardings that have priority <=i and ignore all others."
        
        Raises RuntimeError if not converged within max_iterations.
        """
        self.iteration_count = 0
        sorted_ops = self._get_sorted_ops()
        
        if self.use_priority_iteration:
            return self._run_priority_iteration(sorted_ops)
        else:
            return self._run_simple_iteration(sorted_ops)
    
    def _run_simple_iteration(self, sorted_ops: List[Operation]) -> int:
        """Simple fixed-point iteration without priority levels."""
        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            changed = self._propagate_once(sorted_ops, max_priority=None)
            if not changed:
                return self.iteration_count
        
        raise RuntimeError(
            f"Sharding propagation did not converge after {self.max_iterations} iterations"
        )
    
    def _run_priority_iteration(self, sorted_ops: List[Operation]) -> int:
        """
        Priority-based iteration per Shardy spec.
        
        For each priority level i (from 0 to max), propagate to fixed point
        only considering shardings with priority <= i. This ensures high-priority
        shardings fully propagate before lower-priority ones can interfere.
        """
        priority_levels = self._get_user_priority_levels()
        
        for current_max_priority in priority_levels:
            # Propagate to fixed point at this priority level
            level_iterations = 0
            max_level_iterations = self.max_iterations
            
            while level_iterations < max_level_iterations:
                self.iteration_count += 1
                level_iterations += 1
                
                changed = self._propagate_once(sorted_ops, max_priority=current_max_priority)
                if not changed:
                    break
            
            if level_iterations >= max_level_iterations:
                raise RuntimeError(
                    f"Sharding propagation did not converge at priority level "
                    f"{current_max_priority} after {max_level_iterations} iterations"
                )
        
        return self.iteration_count
    
    def _propagate_once(self, sorted_ops: List[Operation], max_priority: Optional[int]) -> bool:
        """Single propagation pass through all ops and edges."""
        changed = False
        
        # Propagate through regular operations
        for op in sorted_ops:
            in_specs = [t.spec for t in op.inputs]
            out_specs = [t.spec for t in op.outputs]
            
            if propagate_sharding(op.rule, in_specs, out_specs, self.strategy, max_priority):
                changed = True
        
        # Propagate through data flow edges (control flow ops)
        for edge in self.data_flow_edges:
            rule = edge.to_identity_rule()
            source_specs = [t.spec for t in edge.sources]
            target_specs = [t.spec for t in edge.targets]
            
            if propagate_sharding(rule, source_specs, target_specs, self.strategy, max_priority):
                changed = True
        
        return changed
    
    def get_sharding_summary(self) -> Dict[str, str]:
        """Get a summary of tensor shardings after propagation."""
        summary = {}
        seen = set()
        
        for op in self.ops:
            for t in op.inputs + op.outputs:
                if t.name not in seen:
                    summary[t.name] = str(t.spec)
                    seen.add(t.name)
        
        return summary
    
    def get_propagation_table(self) -> str:
        """
        Generate a visualization table of factor shardings (for debugging).
        
        Similar to the table visualization from the Shardy spec.
        """
        lines = ["Sharding Propagation Summary", "=" * 40]
        for op in self.ops:
            lines.append(f"\nOperation: {op.name}")
            lines.append(f"  Rule: {op.rule.to_einsum_notation()}")
            lines.append(f"  Inputs:")
            for t in op.inputs:
                lines.append(f"    {t.name}: {t.spec}")
            lines.append(f"  Outputs:")
            for t in op.outputs:
                lines.append(f"    {t.name}: {t.spec}")
        return "\n".join(lines)

# --- Tests ---

class TestShardingCompiler(unittest.TestCase):
    def setUp(self):
        # 8 devices: [0..7]
        self.devices = list(range(8))
        self.mesh = DeviceMesh("cluster", (2, 4), ("x", "y"), self.devices)

    # --- Group 1: Physical & Execution ---
    def test_01_mesh_coords(self):
        # x=2, y=4. Device 5 is index 5.
        # 5 // 4 = 1 (x), 5 % 4 = 1 (y)
        self.assertEqual(self.mesh.get_coordinate(5, "x"), 1)
        self.assertEqual(self.mesh.get_coordinate(5, "y"), 1)

    def test_02_uneven_slicing(self):
        mesh_1d = DeviceMesh("1d", (2,), ("x",), [0, 1])
        spec = ShardingSpec(mesh_1d, [DimSpec(["x"])])
        dt = DistributedTensor((7,), spec)
        # Dev 0: [0, 4) -> size 4
        # Dev 1: [4, 7) -> size 3 + 1 padding
        s1, e1, p1 = dt.get_local_interval(0, 1)
        self.assertEqual((s1, e1, p1), (4, 7, 1))

    def test_03_axis_size(self):
        y_size = self.mesh.get_axis_size("y")
        self.assertEqual(y_size, 4)

    # --- Group 2: Safety Invariants ---
    def test_04_invariant_conflict(self):
        with self.assertRaisesRegex(ValueError, "multiple times"):
            ShardingSpec(self.mesh, [DimSpec(["x", "x"])])

    def test_05_explicit_replication_conflict(self):
        # Cannot use "x" in dim if "x" is replicated
        with self.assertRaisesRegex(ValueError, "explicitly replicated"):
            ShardingSpec(self.mesh, [DimSpec(["x"])], replicated_axes={"x"})

    # --- Group 3: Basic Propagation ---
    def test_06_forward_prop(self):
        """Test forward propagation from input to output."""
        rule = OpShardingRule([{0:["m"]}], [{0:["m"]}], {"m": 2})
        in_a = ShardingSpec(self.mesh, [DimSpec(["x"], priority=1)])
        # Open dimension can accept propagation from higher priority source
        out_b = ShardingSpec(self.mesh, [DimSpec([], is_open=True)]) 
        propagate_sharding(rule, [in_a], [out_b])
        self.assertEqual(out_b.dim_specs[0].axes, ["x"])

    def test_07_backward_prop(self):
        """Test backward propagation from output to input."""
        rule = OpShardingRule([{0:["m"]}], [{0:["m"]}], {"m": 2})
        # Open dimension can accept propagation from higher priority source
        in_a = ShardingSpec(self.mesh, [DimSpec([], is_open=True)])
        out_b = ShardingSpec(self.mesh, [DimSpec(["x"], priority=1)])
        propagate_sharding(rule, [in_a], [out_b])
        self.assertEqual(in_a.dim_specs[0].axes, ["x"])

    # --- Group 4: Complex Ops ---
    def test_08_reshape_split(self):
        """Test reshape that splits one dimension into two.
        
        From Shardy spec compound factors:
            (i,j,k) -> ((ij), k) for reshape tensor<2x4x32xf32> -> tensor<8x32xf32>
        """
        # Input: Dim0 -> [f1, f2]. Output: Dim0->f1, Dim1->f2.
        rule = OpShardingRule(
            input_mappings=[{0: ["f1", "f2"]}],
            output_mappings=[{0: ["f1"], 1: ["f2"]}],
            factor_sizes={"f1": 2, "f2": 4} # x=2, y=4
        )
        in_a = ShardingSpec(self.mesh, [DimSpec(["x", "y"], priority=1)])
        # Open dimensions accept propagation
        out_b = ShardingSpec(self.mesh, [DimSpec([], is_open=True), DimSpec([], is_open=True)])
        propagate_sharding(rule, [in_a], [out_b])
        # f1 gets x, f2 gets y
        self.assertEqual(out_b.dim_specs[0].axes, ["x"])
        self.assertEqual(out_b.dim_specs[1].axes, ["y"])

    def test_09_transpose(self):
        """Test transpose swaps dimension shardings."""
        rule = OpShardingRule([{0:["m"], 1:["n"]}], [{0:["n"], 1:["m"]}], {"m":2, "n":4})
        in_a = ShardingSpec(self.mesh, [DimSpec(["x"]), DimSpec(["y"])])
        out_b = ShardingSpec(self.mesh, [DimSpec([], is_open=True), DimSpec([], is_open=True)])
        propagate_sharding(rule, [in_a], [out_b])
        self.assertEqual(out_b.dim_specs[0].axes, ["y"])
        self.assertEqual(out_b.dim_specs[1].axes, ["x"])

    def test_10_reduce(self):
        """Test reduce operation drops one factor."""
        rule = OpShardingRule([{0:["m"], 1:["n"]}], [{0:["m"]}], {"m":2, "n":4})
        in_a = ShardingSpec(self.mesh, [DimSpec(["x"]), DimSpec(["y"])])
        out_b = ShardingSpec(self.mesh, [DimSpec([], is_open=True)])
        propagate_sharding(rule, [in_a], [out_b])
        self.assertEqual(out_b.dim_specs[0].axes, ["x"])

    # --- Group 5: Conflict Resolution ---
    def test_11_priority_overwrite(self):
        rule = OpShardingRule([{0:["m"]}], [{0:["m"]}], {"m": 2})
        in_a = ShardingSpec(self.mesh, [DimSpec(["x"], priority=0)])
        out_b = ShardingSpec(self.mesh, [DimSpec(["y"], priority=1)])
        propagate_sharding(rule, [in_a], [out_b])
        # Priority 0 wins
        self.assertEqual(out_b.dim_specs[0].axes, ["x"])

    def test_12_priority_intersection(self):
        rule = OpShardingRule([{0:["m"]}], [{0:["m"]}], {"m": 2})
        in_a = ShardingSpec(self.mesh, [DimSpec(["x", "y"], priority=1)])
        out_b = ShardingSpec(self.mesh, [DimSpec(["x"], priority=1)])
        propagate_sharding(rule, [in_a], [out_b])
        # Intersection of [x,y] and [x] is [x]
        self.assertEqual(out_b.dim_specs[0].axes, ["x"])

    def test_13_implicit_replication_intersection(self):
        rule = OpShardingRule([{0:["m"]}], [{0:["m"]}], {"m": 2})
        in_a = ShardingSpec(self.mesh, [DimSpec([], priority=0)]) # Strong empty
        out_b = ShardingSpec(self.mesh, [DimSpec(["x"], priority=1)]) 
        propagate_sharding(rule, [in_a], [out_b])
        # Stronger priority 0 forces replication
        self.assertEqual(out_b.dim_specs[0].axes, [])

    def test_14_matmul_broadcast_conflict(self):
        """Test conflict resolution when bias forces replication.
        
        Matmul with bias: C = A @ B + bias
        If bias is replicated with high priority, factor 'n' must be replicated.
        """
        rule = OpShardingRule(
             [{0:["m"], 1:["n"]}, {0:["n"]}], 
             [{0:["m"], 1:["n"]}], 
             {"m":2, "n":4}
        )
        in_a = ShardingSpec(self.mesh, [DimSpec(["x"]), DimSpec(["y"])])
        # Bias has priority 0 on replicated (empty closed = strong replication constraint)
        in_bias = ShardingSpec(self.mesh, [DimSpec([], priority=0)]) 
        # Output is open to accept propagation
        out_c = ShardingSpec(self.mesh, [DimSpec([], is_open=True), DimSpec([], is_open=True)])
        
        propagate_sharding(rule, [in_a, in_bias], [out_c])
        # Bias forces factor 'n' to be replicated.
        # Factor 'm' comes from A (x).
        self.assertEqual(out_c.dim_specs[0].axes, ["x"]) 
        self.assertEqual(out_c.dim_specs[1].axes, []) # n is replicated

    # --- Group 6: Edge Cases ---
    def test_15_single_element_dim(self):
        mesh_1d = DeviceMesh("1d", (4,), ("x",), [0,1,2,3])
        # Sharding a dim of size 1 on axis of size 4 -> Padded
        spec = ShardingSpec(mesh_1d, [DimSpec(["x"])])
        dt = DistributedTensor((1,), spec)
        # Dev 0 gets it, Dev 1,2,3 get nothing
        s, e, p = dt.get_local_interval(0, 1)
        self.assertEqual(e - s, 0) # Empty on dev 1

    def test_17_explicit_replication_enforcement(self):
        """Test that explicit replication blocks axis propagation.
        
        When an axis is in replicated_axes, it cannot be used to shard any dimension
        of that tensor, even if propagation would otherwise assign it.
        """
        rule = OpShardingRule([{0:["m"]}], [{0:["m"]}], {"m":2})
        in_a = ShardingSpec(self.mesh, [DimSpec(["x"], priority=1)])
        # Output explicitly replicates on "x" - use empty open dim (not empty closed with priority)
        out_b = ShardingSpec(self.mesh, [DimSpec([], is_open=True)], replicated_axes={"x"})
        
        propagate_sharding(rule, [in_a], [out_b])
        # Should NOT propagate x to out_b because x is explicitly replicated
        self.assertEqual(out_b.dim_specs[0].axes, [])

    # --- Group 7: Sub-Axis Logic (REVISED) ---
    def test_18_subaxis_auto_decomposition_strings(self):
        # Mesh X (size 8). Factors f1(2), f2(4).
        mesh_x = DeviceMesh("x8", (8,), ("x",), list(range(8)))
        t_in = GraphTensor("In", (8,), mesh_x, ShardingSpec(mesh_x, [DimSpec(["x"], priority=0)]))
        t_out = GraphTensor("Out", (2, 4), mesh_x)
        
        rule = OpShardingRule(
            input_mappings=[{0: ["f1", "f2"]}],
            output_mappings=[{0: ["f1"], 1: ["f2"]}],
            factor_sizes={"f1": 2, "f2": 4}
        )
        propagate_sharding(rule, [t_in.spec], [t_out.spec])
        
        ax0 = t_out.spec.dim_specs[0].axes[0]
        ax1 = t_out.spec.dim_specs[1].axes[0]
        
        # f1 is Major. Pre-size = 1. Size = 2. -> x:(1)2
        self.assertEqual(ax0, "x:(1)2")
        # f2 is Minor. Pre-size = 2. Size = 4. -> x:(2)4
        self.assertEqual(ax1, "x:(2)4")

    def test_19_subaxis_coordinate_math(self):
        # NEW: Verify the "x:(m)k" math on specific indices
        mesh_x = DeviceMesh("x8", (8,), ("x",), list(range(8)))
        
        # Case A: x:(1)2 (Major splits, size 2). Post-size = 8/(1*2) = 4.
        # Indices [0..3] -> 0, [4..7] -> 1.
        # Formula: (coord // 4) % 2
        self.assertEqual(mesh_x.get_coordinate(0, "x:(1)2"), 0)
        self.assertEqual(mesh_x.get_coordinate(3, "x:(1)2"), 0)
        self.assertEqual(mesh_x.get_coordinate(4, "x:(1)2"), 1)
        self.assertEqual(mesh_x.get_coordinate(7, "x:(1)2"), 1)
        
        # Case B: x:(2)4 (Minor splits, size 4). Post-size = 8/(2*4) = 1.
        # Indices 0->0, 1->1 ... 4->0, 5->1 ...
        # Formula: (coord // 1) % 4
        self.assertEqual(mesh_x.get_coordinate(0, "x:(2)4"), 0)
        self.assertEqual(mesh_x.get_coordinate(4, "x:(2)4"), 0) # Wraps
        self.assertEqual(mesh_x.get_coordinate(5, "x:(2)4"), 1)

    def test_20_three_way_split(self):
        # NEW: Split 8 -> 2x2x2
        mesh_x = DeviceMesh("x8", (8,), ("x",), list(range(8)))
        t_in = GraphTensor("In", (8,), mesh_x, ShardingSpec(mesh_x, [DimSpec(["x"], priority=0)]))
        t_out = GraphTensor("Out", (2, 2, 2), mesh_x)
        
        rule = OpShardingRule(
            input_mappings=[{0: ["f1", "f2", "f3"]}],
            output_mappings=[{0: ["f1"], 1: ["f2"], 2: ["f3"]}],
            factor_sizes={"f1": 2, "f2": 2, "f3": 2}
        )
        propagate_sharding(rule, [t_in.spec], [t_out.spec])
        
        # f1: Pre=1, Size=2 -> x:(1)2
        # f2: Pre=2, Size=2 -> x:(2)2
        # f3: Pre=4, Size=2 -> x:(4)2
        self.assertEqual(t_out.spec.dim_specs[0].axes[0], "x:(1)2")
        self.assertEqual(t_out.spec.dim_specs[1].axes[0], "x:(2)2")
        self.assertEqual(t_out.spec.dim_specs[2].axes[0], "x:(4)2")

    def test_21_partial_split_usage(self):
        # NEW: Split 8 -> 2x4. Only use the '4' part.
        mesh_x = DeviceMesh("x8", (8,), ("x",), list(range(8)))
        t_in = GraphTensor("In", (8,), mesh_x, ShardingSpec(mesh_x, [DimSpec(["x"], priority=0)]))
        t_out = GraphTensor("Out", (4,), mesh_x) # Dim size 4
        
        # Map input to [f1, f2]. Map output to [f2] (f1 dropped/reduced)
        rule = OpShardingRule(
            input_mappings=[{0: ["f1", "f2"]}],
            output_mappings=[{0: ["f2"]}],
            factor_sizes={"f1": 2, "f2": 4}
        )
        propagate_sharding(rule, [t_in.spec], [t_out.spec])
        
        # Output should use f2 -> x:(2)4
        self.assertEqual(t_out.spec.dim_specs[0].axes[0], "x:(2)4")

    # --- Group 8: Data Flow Edges ---
    def test_22_data_flow_edge_identity(self):
        """Test DataFlowEdge for control flow ops like while/case."""
        mesh_1d = DeviceMesh("1d", (4,), ("x",), [0, 1, 2, 3])
        
        # Simulate while loop: x_i -> body_arg_i, return_value_i -> y_i
        x_input = GraphTensor("x_input", (8, 4), mesh_1d, 
                              ShardingSpec(mesh_1d, [DimSpec(["x"], priority=0), DimSpec([])]))
        body_arg = GraphTensor("body_arg", (8, 4), mesh_1d)  # Target: should get same sharding
        
        edge = DataFlowEdge(sources=[x_input], targets=[body_arg])
        rule = edge.to_identity_rule()
        
        propagate_sharding(rule, [x_input.spec], [body_arg.spec])
        
        # body_arg should receive the sharding from x_input
        self.assertEqual(body_arg.spec.dim_specs[0].axes, ["x"])
        self.assertEqual(body_arg.spec.dim_specs[1].axes, [])

    def test_23_data_flow_edge_bidirectional(self):
        """Test that data flow edges propagate both ways."""
        mesh_1d = DeviceMesh("1d", (2,), ("x",), [0, 1])
        
        source = GraphTensor("source", (4,), mesh_1d)  # No initial sharding
        target = GraphTensor("target", (4,), mesh_1d, 
                             ShardingSpec(mesh_1d, [DimSpec(["x"], priority=0)]))
        
        edge = DataFlowEdge(sources=[source], targets=[target])
        rule = edge.to_identity_rule()
        
        # Propagate backwards: target -> source
        propagate_sharding(rule, [source.spec], [target.spec])
        
        self.assertEqual(source.spec.dim_specs[0].axes, ["x"])

    # --- Group 9: Open/Closed Dimension Semantics ---
    def test_24_open_dimension_extension(self):
        """Open dimensions can be further sharded.
        
        From Shardy spec:
            "An open dimension is open for propagation to further shard it along 
            additional axes, i.e. the specified dimension sharding doesn't have to 
            be the final sharding of that dimension."
        
        Open dimensions with lower priority accept shardings from higher priority sources.
        """
        rule = OpShardingRule([{0: ["m"]}], [{0: ["m"]}], {"m": 2})
        
        # Input has priority 1, output is open and can accept propagation
        in_a = ShardingSpec(self.mesh, [DimSpec(["x"], is_open=True, priority=1)])
        out_b = ShardingSpec(self.mesh, [DimSpec([], is_open=True)])  # Default priority 0
        
        propagate_sharding(rule, [in_a], [out_b])
        
        # Open dimension accepts the sharding from input
        self.assertEqual(out_b.dim_specs[0].axes, ["x"])

    def test_25_closed_dimension_no_extension(self):
        """Closed dimensions cannot be extended at same priority."""
        rule = OpShardingRule([{0: ["m"]}], [{0: ["m"]}], {"m": 2})
        
        # Both have priority 1, but output is closed with existing sharding
        in_a = ShardingSpec(self.mesh, [DimSpec(["x", "y"], is_open=False, priority=1)])
        out_b = ShardingSpec(self.mesh, [DimSpec(["x"], is_open=False, priority=1)])
        
        propagate_sharding(rule, [in_a], [out_b])
        
        # Closed dimension should NOT be extended (stays at ["x"])
        self.assertEqual(out_b.dim_specs[0].axes, ["x"])

    def test_26_closed_dimension_priority_override(self):
        """Closed dimensions CAN be overridden by higher priority."""
        rule = OpShardingRule([{0: ["m"]}], [{0: ["m"]}], {"m": 2})
        
        # Input has higher priority (0), output has lower (1)
        in_a = ShardingSpec(self.mesh, [DimSpec(["y"], is_open=False, priority=0)])
        out_b = ShardingSpec(self.mesh, [DimSpec(["x"], is_open=False, priority=1)])
        
        propagate_sharding(rule, [in_a], [out_b])
        
        # Higher priority DOES override, even for closed dimensions
        self.assertEqual(out_b.dim_specs[0].axes, ["y"])

    # --- Group 10: ShardingPass with Operation Priorities ---
    def test_27_sharding_pass_basic(self):
        """Test basic ShardingPass fixed-point iteration."""
        mesh_1d = DeviceMesh("1d", (2,), ("x",), [0, 1])
        
        t_a = GraphTensor("A", (4,), mesh_1d, 
                          ShardingSpec(mesh_1d, [DimSpec(["x"], priority=0)]))
        t_b = GraphTensor("B", (4,), mesh_1d)
        t_c = GraphTensor("C", (4,), mesh_1d)
        
        # A -> B -> C chain
        op1 = Operation("op1", 
                        OpShardingRule([{0: ["m"]}], [{0: ["m"]}], {"m": 2}),
                        [t_a], [t_b])
        op2 = Operation("op2",
                        OpShardingRule([{0: ["m"]}], [{0: ["m"]}], {"m": 2}),
                        [t_b], [t_c])
        
        spass = ShardingPass([op1, op2])
        iterations = spass.run_pass()
        
        # Should propagate A's sharding through B to C
        self.assertEqual(t_b.spec.dim_specs[0].axes, ["x"])
        self.assertEqual(t_c.spec.dim_specs[0].axes, ["x"])
        self.assertLessEqual(iterations, 3)  # Should converge quickly

    def test_28_sharding_pass_with_data_flow_edges(self):
        """Test ShardingPass with data flow edges for control flow."""
        mesh_1d = DeviceMesh("1d", (2,), ("x",), [0, 1])
        
        loop_input = GraphTensor("loop_input", (4,), mesh_1d,
                                 ShardingSpec(mesh_1d, [DimSpec(["x"], priority=0)]))
        body_arg = GraphTensor("body_arg", (4,), mesh_1d)
        
        # No regular ops, just data flow edge
        edge = DataFlowEdge(sources=[loop_input], targets=[body_arg])
        
        spass = ShardingPass([], data_flow_edges=[edge])
        spass.run_pass()
        
        self.assertEqual(body_arg.spec.dim_specs[0].axes, ["x"])

    # --- Group 11: Multiple Meshes & Implicit Replication ---
    def test_29_implicit_replication_detection(self):
        """Test detection of implicitly replicated axes."""
        mesh = DeviceMesh("3d", (2, 4, 2), ("x", "y", "z"), list(range(16)))
        
        # Only shard on x, y and z are implicitly replicated
        spec = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
        
        implicit = spec.get_implicitly_replicated_axes()
        
        self.assertIn("y", implicit)
        self.assertIn("z", implicit)
        self.assertNotIn("x", implicit)

    def test_30_fully_replicated_check(self):
        """Test is_fully_replicated helper."""
        mesh_1d = DeviceMesh("1d", (4,), ("x",), [0, 1, 2, 3])
        
        replicated_spec = ShardingSpec(mesh_1d, [DimSpec([]), DimSpec([])])
        sharded_spec = ShardingSpec(mesh_1d, [DimSpec(["x"]), DimSpec([])])
        
        self.assertTrue(replicated_spec.is_fully_replicated())
        self.assertFalse(sharded_spec.is_fully_replicated())

    # --- Group 12: Edge Cases from Shardy Spec ---
    def test_31_subaxis_overlap_validation(self):
        """Sub-axes must not overlap (spec invariant)."""
        # x:(1)4 covers range [1, 4), x:(2)4 covers range [2, 8)
        # They overlap at [2, 4)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_sub_axes_non_overlapping(["x:(1)4", "x:(2)4"])

    def test_32_subaxis_maximality_warning(self):
        """Adjacent sub-axes should be flagged as mergeable."""
        # x:(1)2 and x:(2)4 are adjacent (end of first = start of second)
        # and could be merged into x:(1)8
        warnings = check_sub_axes_maximality(["x:(1)2", "x:(2)4"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("could be merged", warnings[0])

    def test_33_mesh_repr_format(self):
        """Test mesh string representation matches spec grammar."""
        mesh = DeviceMesh("cluster", (2, 4), ("x", "y"), list(range(8)))
        repr_str = repr(mesh)
        
        self.assertIn('@cluster', repr_str)
        self.assertIn('"x"=2', repr_str)
        self.assertIn('"y"=4', repr_str)

    def test_34_sharding_spec_repr_format(self):
        """Test sharding spec string representation matches spec grammar."""
        mesh = DeviceMesh("m", (2, 4), ("x", "y"), list(range(8)))
        spec = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([], is_open=True)])
        repr_str = repr(spec)
        
        self.assertIn('sharding<@m', repr_str)
        self.assertIn('"x"', repr_str)

    # --- Group 13: Priority Syntax and Representation Tests ---
    def test_35_priority_repr_syntax(self):
        """Test DimSpec priority syntax matches Shardy spec: {axes}p<N>."""
        # Priority 0 is default, should not show 'p0'
        dim_p0 = DimSpec(["x"], priority=0)
        self.assertNotIn("p", repr(dim_p0))
        self.assertEqual(repr(dim_p0), '{"x"}')
        
        # Priority 1 should show 'p1'
        dim_p1 = DimSpec(["x"], priority=1)
        self.assertIn("p1", repr(dim_p1))
        self.assertEqual(repr(dim_p1), '{"x"}p1')
        
        # Priority 2 should show 'p2'
        dim_p2 = DimSpec(["x", "y"], priority=2)
        self.assertIn("p2", repr(dim_p2))
        self.assertEqual(repr(dim_p2), '{"x", "y"}p2')
        
        # Open dimension with priority
        dim_open_p1 = DimSpec(["x"], is_open=True, priority=1)
        self.assertIn("?", repr(dim_open_p1))
        self.assertIn("p1", repr(dim_open_p1))
        self.assertEqual(repr(dim_open_p1), '{"x", ?}p1')

    def test_36_empty_dim_repr(self):
        """Test empty dimension representations."""
        # Empty closed - should be '{}'
        empty_closed = DimSpec([])
        self.assertEqual(repr(empty_closed), '{}')
        
        # Empty open - should be '{?}'
        empty_open = DimSpec([], is_open=True)
        self.assertEqual(repr(empty_open), '{?}')
        
        # Empty open with priority (shows p<N>)
        empty_open_p1 = DimSpec([], is_open=True, priority=1)
        self.assertEqual(repr(empty_open_p1), '{?}p1')
        
        # Empty closed with non-zero priority should raise error (Shardy spec invariant)
        with self.assertRaisesRegex(ValueError, "cannot have non-zero priority"):
            DimSpec([], is_open=False, priority=1)

    def test_37_replicated_axes_ordering(self):
        """Test that replicated axes are ordered per spec: mesh order, sub-axes by pre-size."""
        mesh = DeviceMesh("m", (8, 4), ("x", "y"), list(range(32)))
        
        # Create spec with replicated axes in wrong order
        spec = ShardingSpec(
            mesh, 
            [DimSpec([])], 
            replicated_axes={"y", "x:(4)2", "x:(1)2"}  # Out of order
        )
        
        # Check the ordering in repr
        ordered = spec._order_replicated_axes()
        
        # Should be: x sub-axes (sorted by pre-size), then y
        # x:(1)2 comes before x:(4)2 (pre-size 1 < 4)
        # y comes after x (mesh axis order)
        self.assertEqual(ordered[0], "x:(1)2")
        self.assertEqual(ordered[1], "x:(4)2")
        self.assertEqual(ordered[2], "y")

    def test_38_propagation_strategy_enum(self):
        """Test that PropagationStrategy enum exists and has correct values."""
        self.assertEqual(PropagationStrategy.BASIC, 0)
        self.assertEqual(PropagationStrategy.AGGRESSIVE, 1)

    # --- Group 14: Additional Coverage Tests ---
    def test_39_multi_factor_with_replication(self):
        """Test propagation with multiple factors and explicit replication."""
        mesh = DeviceMesh("m", (2, 4), ("x", "y"), list(range(8)))
        
        rule = OpShardingRule(
            [{0: ["m"], 1: ["n"]}], 
            [{0: ["m"], 1: ["n"]}], 
            {"m": 2, "n": 4}
        )
        
        # Input sharded on both dims with high priority
        in_a = ShardingSpec(mesh, [DimSpec(["x"], priority=0), DimSpec(["y"], priority=0)])
        # Output has "y" explicitly replicated - should not receive "y" sharding on dim 1
        # Using is_open=True for dimensions that can accept propagation
        out_b = ShardingSpec(mesh, [DimSpec([], is_open=True), DimSpec([], is_open=True)], replicated_axes={"y"})
        
        propagate_sharding(rule, [in_a], [out_b])
        
        self.assertEqual(out_b.dim_specs[0].axes, ["x"])  # m -> x propagates
        self.assertEqual(out_b.dim_specs[1].axes, [])     # n -> y blocked by explicit replication

    def test_40_dim_spec_get_total_shards(self):
        """Test DimSpec.get_total_shards calculation."""
        mesh = DeviceMesh("m", (2, 4), ("x", "y"), list(range(8)))
        
        # Sharded on x only: 2 shards
        dim_x = DimSpec(["x"])
        self.assertEqual(dim_x.get_total_shards(mesh), 2)
        
        # Sharded on x then y: 2 * 4 = 8 shards
        dim_xy = DimSpec(["x", "y"])
        self.assertEqual(dim_xy.get_total_shards(mesh), 8)
        
        # No sharding: 1 shard (replicated)
        dim_none = DimSpec([])
        self.assertEqual(dim_none.get_total_shards(mesh), 1)

    def test_41_distributed_tensor_byte_size(self):
        """Test byte size calculations for DistributedTensor."""
        mesh = DeviceMesh("m", (2,), ("x",), [0, 1])
        spec = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
        
        # Global shape (4, 8), dtype_size=4 (float32)
        dt = DistributedTensor((4, 8), spec, dtype_size=4)
        
        # Global: 4 * 8 * 4 = 128 bytes
        self.assertEqual(dt.get_byte_size(), 128)
        
        # Local on device 0: (2, 8) -> 2 * 8 * 4 = 64 bytes
        self.assertEqual(dt.get_local_byte_size(0), 64)

    def test_42_op_sharding_rule_get_all_factors(self):
        """Test OpShardingRule.get_all_factors."""
        rule = OpShardingRule(
            [{0: ["m", "k"]}, {0: ["k"], 1: ["n"]}],
            [{0: ["m"], 1: ["n"]}],
            {"m": 2, "n": 4, "k": 8}
        )
        
        all_factors = rule.get_all_factors()
        self.assertEqual(all_factors, {"m", "n", "k"})

    def test_43_op_sharding_rule_to_einsum(self):
        """Test OpShardingRule.to_einsum_notation for matmul."""
        rule = OpShardingRule(
            [{0: ["m"], 1: ["k"]}, {0: ["k"], 1: ["n"]}],
            [{0: ["m"], 1: ["n"]}],
            {"m": 4, "k": 8, "n": 16}
        )
        einsum = rule.to_einsum_notation()
        # Should produce something like "(m, k), (k, n) -> (m, n)"
        self.assertIn("m", einsum)
        self.assertIn("k", einsum)
        self.assertIn("n", einsum)
        self.assertIn("->", einsum)

    def test_44_sharding_spec_priority_helpers(self):
        """Test ShardingSpec priority helper methods."""
        mesh = DeviceMesh("m", (2, 4), ("x", "y"), list(range(8)))
        spec = ShardingSpec(mesh, [
            DimSpec(["x"], priority=0),
            DimSpec(["y"], priority=2),
            DimSpec([], is_open=True)  # priority 0 default
        ])
        
        self.assertEqual(spec.get_min_priority(), 0)
        self.assertEqual(spec.get_max_priority(), 2)

    def test_45_sharding_pass_get_propagation_table(self):
        """Test ShardingPass debug output."""
        mesh_1d = DeviceMesh("1d", (2,), ("x",), [0, 1])
        
        t_a = GraphTensor("A", (4,), mesh_1d, 
                          ShardingSpec(mesh_1d, [DimSpec(["x"], priority=0)]))
        t_b = GraphTensor("B", (4,), mesh_1d)
        
        op = Operation("identity", 
                       OpShardingRule([{0: ["m"]}], [{0: ["m"]}], {"m": 2}),
                       [t_a], [t_b])
        
        spass = ShardingPass([op])
        spass.run_pass()
        
        table = spass.get_propagation_table()
        self.assertIn("identity", table)
        self.assertIn("A", table)
        self.assertIn("B", table)

    def test_46_empty_open_vs_closed_semantics(self):
        """Test that empty open and empty closed have different propagation semantics.
        
        From Shardy spec:
        - Empty open dimension: "can be further sharded during propagation"
        - Empty closed dimension: "sharding is fixed and can't be changed"
        """
        mesh = DeviceMesh("m", (2,), ("x",), [0, 1])
        rule = OpShardingRule([{0: ["m"]}], [{0: ["m"]}], {"m": 2})
        
        # Test 1: Empty OPEN accepts propagation
        in_open = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        out_with_sharding = ShardingSpec(mesh, [DimSpec(["x"], priority=1)])
        propagate_sharding(rule, [in_open], [out_with_sharding])
        # Empty open should receive the sharding
        self.assertEqual(in_open.dim_specs[0].axes, ["x"])
        
        # Test 2: Empty CLOSED with higher priority blocks propagation
        in_closed = ShardingSpec(mesh, [DimSpec([], is_open=False, priority=0)])
        out_with_sharding2 = ShardingSpec(mesh, [DimSpec(["x"], priority=1)])
        propagate_sharding(rule, [in_closed], [out_with_sharding2])
        # Empty closed p0 should force replication on the output
        self.assertEqual(out_with_sharding2.dim_specs[0].axes, [])

    def test_47_divisibility_uneven_sharding(self):
        """Test dimension sharding with non-divisible sizes (requires padding).
        
        From Shardy spec:
        "It's possible for a dimension of size d to be sharded along axes whose
        product of sizes is n, such that d is not divisible by n."
        """
        mesh = DeviceMesh("m", (3,), ("x",), [0, 1, 2])
        spec = ShardingSpec(mesh, [DimSpec(["x"])])
        
        # 7 elements sharded across 3 devices
        dt = DistributedTensor((7,), spec)
        
        # ceil(7/3) = 3 elements per device
        # Device 0: [0, 3), Device 1: [3, 6), Device 2: [6, 7) + 2 padding
        s0, e0, p0 = dt.get_local_interval(0, 0)
        self.assertEqual((s0, e0, p0), (0, 3, 0))
        
        s1, e1, p1 = dt.get_local_interval(0, 1)
        self.assertEqual((s1, e1, p1), (3, 6, 0))
        
        s2, e2, p2 = dt.get_local_interval(0, 2)
        self.assertEqual((s2, e2, p2), (6, 7, 2))  # 1 real element + 2 padding

    # --- Group 15: Aggressive Strategy Tests ---
    def test_48_aggressive_picks_higher_parallelism(self):
        """Test AGGRESSIVE strategy picks sharding with more parallelism.
        
        From Shardy spec: "The aggressive strategy resolves conflicts. Higher
        aggressiveness can reduce the memory footprint at the cost of potential
        communication."
        """
        mesh = DeviceMesh("m", (2, 4), ("x", "y"), list(range(8)))
        rule = OpShardingRule([{0: ["m"]}], [{0: ["m"]}], {"m": 2})
        
        # Both have priority 1, but different parallelism
        # x has size 2, y has size 4
        in_a = ShardingSpec(mesh, [DimSpec(["x"], priority=1)])  # parallelism=2
        out_b = ShardingSpec(mesh, [DimSpec(["y"], priority=1)])  # parallelism=4
        
        # BASIC: common prefix = [] (different axes)
        propagate_sharding(rule, [in_a], [out_b], PropagationStrategy.BASIC)
        # Both keep their shardings (common prefix is empty, no propagation to change)
        
        # Reset for AGGRESSIVE test
        in_a2 = ShardingSpec(mesh, [DimSpec(["x"], priority=1)])
        out_b2 = ShardingSpec(mesh, [DimSpec(["y"], priority=1)])
        
        # AGGRESSIVE: should keep "y" (higher parallelism=4 > 2)
        propagate_sharding(rule, [in_a2], [out_b2], PropagationStrategy.AGGRESSIVE)
        # The factor state will have the higher parallelism winner
        # Since out_b2 has higher parallelism, it should influence the factor
        self.assertEqual(out_b2.dim_specs[0].axes, ["y"])

    def test_49_aggressive_vs_basic_conflict(self):
        """Test difference between BASIC and AGGRESSIVE on same-priority conflicting shardings.
        
        Key insight: BASIC takes longest common prefix, AGGRESSIVE picks higher parallelism.
        """
        mesh = DeviceMesh("m", (2, 4), ("x", "y"), list(range(8)))
        rule = OpShardingRule([{0: ["m"]}, {0: ["m"]}], [{0: ["m"]}], {"m": 2})
        
        # Two inputs with DIFFERENT shardings at same priority (actual conflict)
        # x has size 2, y has size 4
        in_a = ShardingSpec(mesh, [DimSpec(["x"], priority=1)])  # parallelism=2
        in_b = ShardingSpec(mesh, [DimSpec(["y"], priority=1)])  # parallelism=4
        out_c = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        # BASIC: common prefix of ["x"] and ["y"] = [] (different axes)
        propagate_sharding(rule, [in_a, in_b], [out_c], PropagationStrategy.BASIC)
        self.assertEqual(out_c.dim_specs[0].axes, [])  # Conservative: no common prefix
        
        # Reset for AGGRESSIVE test
        in_a2 = ShardingSpec(mesh, [DimSpec(["x"], priority=1)])
        in_b2 = ShardingSpec(mesh, [DimSpec(["y"], priority=1)])
        out_c2 = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        # AGGRESSIVE: pick ["y"] (parallelism=4 > 2)
        propagate_sharding(rule, [in_a2, in_b2], [out_c2], PropagationStrategy.AGGRESSIVE)
        self.assertEqual(out_c2.dim_specs[0].axes, ["y"])  # Higher parallelism wins

    def test_50_aggressive_same_parallelism_keeps_first(self):
        """Test AGGRESSIVE with equal parallelism keeps the first encountered."""
        mesh = DeviceMesh("m", (4, 4), ("x", "y"), list(range(16)))
        rule = OpShardingRule([{0: ["m"]}, {0: ["m"]}], [{0: ["m"]}], {"m": 4})
        
        # Both have parallelism=4 at same priority
        in_a = ShardingSpec(mesh, [DimSpec(["x"], priority=1)])  # parallelism=4
        in_b = ShardingSpec(mesh, [DimSpec(["y"], priority=1)])  # parallelism=4
        out_c = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        propagate_sharding(rule, [in_a, in_b], [out_c], PropagationStrategy.AGGRESSIVE)
        # First encountered (x) should be kept when parallelism is equal
        self.assertEqual(out_c.dim_specs[0].axes, ["x"])

    # --- Group 16: Priority Iteration Tests ---
    def test_51_priority_iteration_basic(self):
        """Test priority-iterative propagation processes priorities in order.
        
        From Shardy spec: "at iteration i we propagate all dimension shardings
        that have priority <=i and ignore all others."
        """
        mesh = DeviceMesh("m", (2, 4), ("x", "y"), list(range(8)))
        
        # Simple chain: A -> B
        # A has p0 sharding, B has p1 sharding on different dims
        t_a = GraphTensor("A", (4, 8), mesh, 
                          ShardingSpec(mesh, [DimSpec(["x"], priority=0), DimSpec([], is_open=True)]))
        t_b = GraphTensor("B", (4, 8), mesh,
                          ShardingSpec(mesh, [DimSpec([], is_open=True), DimSpec([], is_open=True)]))
        
        rule = OpShardingRule(
            [{0: ["m"], 1: ["n"]}],
            [{0: ["m"], 1: ["n"]}],
            {"m": 4, "n": 8}
        )
        
        op1 = Operation("op1", rule, [t_a], [t_b])
        
        # With priority iteration
        spass = ShardingPass([op1], use_priority_iteration=True)
        spass.run_pass()
        
        # A's p0 sharding on dim0 should propagate to B
        self.assertEqual(t_b.spec.dim_specs[0].axes, ["x"])

    def test_52_priority_iteration_respects_order(self):
        """Test that higher priority shardings propagate before lower ones."""
        mesh = DeviceMesh("m", (2, 4), ("x", "y"), list(range(8)))
        
        # Tensor A wants dim0 sharded on x (priority 0)
        # Tensor B wants dim0 sharded on y (priority 1)  
        # With priority iteration, p0 should win and propagate first
        t_a = GraphTensor("A", (8,), mesh,
                          ShardingSpec(mesh, [DimSpec(["x"], priority=0)]))
        t_b = GraphTensor("B", (8,), mesh,
                          ShardingSpec(mesh, [DimSpec(["y"], priority=1)]))
        t_c = GraphTensor("C", (8,), mesh,
                          ShardingSpec(mesh, [DimSpec([], is_open=True)]))
        
        # A -> C, B -> C (both contribute to C)
        rule = OpShardingRule([{0: ["m"]}], [{0: ["m"]}], {"m": 8})
        op1 = Operation("op1", rule, [t_a], [t_c])
        op2 = Operation("op2", rule, [t_b], [t_c])
        
        spass = ShardingPass([op1, op2], use_priority_iteration=True)
        spass.run_pass()
        
        # C should have x (from p0) not y (from p1)
        self.assertEqual(t_c.spec.dim_specs[0].axes, ["x"])

    def test_53_priority_iteration_vs_simple(self):
        """Compare priority iteration with simple iteration."""
        mesh = DeviceMesh("m", (2,), ("x",), [0, 1])
        
        def make_tensors():
            t_a = GraphTensor("A", (4,), mesh,
                              ShardingSpec(mesh, [DimSpec(["x"], priority=0)]))
            t_b = GraphTensor("B", (4,), mesh,
                              ShardingSpec(mesh, [DimSpec([], is_open=True)]))
            return t_a, t_b
        
        rule = OpShardingRule([{0: ["m"]}], [{0: ["m"]}], {"m": 2})
        
        # Simple iteration
        t_a1, t_b1 = make_tensors()
        op1 = Operation("op", rule, [t_a1], [t_b1])
        spass1 = ShardingPass([op1], use_priority_iteration=False)
        spass1.run_pass()
        
        # Priority iteration
        t_a2, t_b2 = make_tensors()
        op2 = Operation("op", rule, [t_a2], [t_b2])
        spass2 = ShardingPass([op2], use_priority_iteration=True)
        spass2.run_pass()
        
        # Both should give same result for this simple case
        self.assertEqual(t_b1.spec.dim_specs[0].axes, t_b2.spec.dim_specs[0].axes)
        self.assertEqual(t_b1.spec.dim_specs[0].axes, ["x"])

    def test_54_max_priority_filtering(self):
        """Test that max_priority parameter filters propagation correctly."""
        mesh = DeviceMesh("m", (2, 4), ("x", "y"), list(range(8)))
        rule = OpShardingRule([{0: ["m"], 1: ["n"]}], [{0: ["m"], 1: ["n"]}], {"m": 2, "n": 4})
        
        # Input has p0 on dim0, p2 on dim1
        in_a = ShardingSpec(mesh, [DimSpec(["x"], priority=0), DimSpec(["y"], priority=2)])
        out_b = ShardingSpec(mesh, [DimSpec([], is_open=True), DimSpec([], is_open=True)])
        
        # Only propagate priority <= 0
        propagate_sharding(rule, [in_a], [out_b], max_priority=0)
        
        # Only dim0 (p0) should propagate
        self.assertEqual(out_b.dim_specs[0].axes, ["x"])
        self.assertEqual(out_b.dim_specs[1].axes, [])  # p2 filtered out

    def test_55_sharding_pass_with_aggressive_strategy(self):
        """Test ShardingPass uses strategy correctly."""
        mesh = DeviceMesh("m", (2, 4), ("x", "y"), list(range(8)))
        
        # Process y first (higher parallelism=4) then x (parallelism=2)
        # By making y the first input, aggressive should keep it
        t_b = GraphTensor("B", (8,), mesh,
                          ShardingSpec(mesh, [DimSpec(["y"], priority=1)]))  # parallelism=4
        t_a = GraphTensor("A", (8,), mesh,
                          ShardingSpec(mesh, [DimSpec(["x"], priority=1)]))  # parallelism=2
        t_c = GraphTensor("C", (8,), mesh,
                          ShardingSpec(mesh, [DimSpec([], is_open=True)]))
        
        rule = OpShardingRule([{0: ["m"]}], [{0: ["m"]}], {"m": 8})
        # Process B (y, higher parallelism) first
        op1 = Operation("op1", rule, [t_b], [t_c])
        op2 = Operation("op2", rule, [t_a], [t_c])
        
        spass = ShardingPass([op1, op2], strategy=PropagationStrategy.AGGRESSIVE)
        spass.run_pass()
        
        # AGGRESSIVE should keep y (higher parallelism, processed first)
        self.assertEqual(t_c.spec.dim_specs[0].axes, ["y"])

    # --- Group 17: OpShardingRuleTemplate Tests ---
    def test_56_template_instantiate_matmul(self):
        """Test template instantiation infers factor sizes from shapes."""
        template = OpShardingRuleTemplate(
            input_mappings=[{0: ["m"], 1: ["k"]}, {0: ["k"], 1: ["n"]}],
            output_mappings=[{0: ["m"], 1: ["n"]}]
        )
        
        rule = template.instantiate(
            input_shapes=[(4, 8), (8, 16)],
            output_shapes=[(4, 16)]
        )
        
        self.assertEqual(rule.factor_sizes["m"], 4)
        self.assertEqual(rule.factor_sizes["k"], 8)
        self.assertEqual(rule.factor_sizes["n"], 16)

    def test_57_template_instantiate_reshape_split(self):
        """Test template for reshape (24,) -> (2, 3, 4) infers sizes from output."""
        template = OpShardingRuleTemplate(
            input_mappings=[{0: ["f1", "f2", "f3"]}],
            output_mappings=[{0: ["f1"], 1: ["f2"], 2: ["f3"]}]
        )
        
        rule = template.instantiate(
            input_shapes=[(24,)],
            output_shapes=[(2, 3, 4)]
        )
        
        # Sizes inferred from single-factor output dims
        self.assertEqual(rule.factor_sizes["f1"], 2)
        self.assertEqual(rule.factor_sizes["f2"], 3)
        self.assertEqual(rule.factor_sizes["f3"], 4)
        
        # Verify product matches input
        self.assertEqual(2 * 3 * 4, 24)

    def test_58_template_instantiate_inconsistent_sizes(self):
        """Test template raises error on inconsistent factor sizes."""
        template = OpShardingRuleTemplate(
            input_mappings=[{0: ["m"], 1: ["k"]}, {0: ["k"], 1: ["n"]}],
            output_mappings=[{0: ["m"], 1: ["n"]}]
        )
        
        # k appears twice with different sizes: 8 vs 10
        with self.assertRaisesRegex(ValueError, "Inconsistent size"):
            template.instantiate(
                input_shapes=[(4, 8), (10, 16)],  # k=8 vs k=10!
                output_shapes=[(4, 16)]
            )

    def test_59_template_instantiate_wrong_product(self):
        """Test template raises error when compound factors don't multiply correctly."""
        template = OpShardingRuleTemplate(
            input_mappings=[{0: ["f1", "f2"]}],
            output_mappings=[{0: ["f1"], 1: ["f2"]}]
        )
        
        # Input is 8, but output is 2*5=10
        with self.assertRaisesRegex(ValueError, "Factor product.*!=.*dim size"):
            template.instantiate(
                input_shapes=[(8,)],
                output_shapes=[(2, 5)]  # 2*5=10 != 8
            )

    def test_60_template_with_propagation(self):
        """Test template-instantiated rule works with propagation."""
        mesh = DeviceMesh("m", (2, 4), ("x", "y"), list(range(8)))
        
        # Create template and instantiate with shapes
        template = OpShardingRuleTemplate(
            input_mappings=[{0: ["m"], 1: ["n"]}],
            output_mappings=[{0: ["n"], 1: ["m"]}]  # Transpose
        )
        rule = template.instantiate(
            input_shapes=[(2, 4)],
            output_shapes=[(4, 2)]
        )
        
        in_a = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec(["y"])])
        out_b = ShardingSpec(mesh, [DimSpec([], is_open=True), DimSpec([], is_open=True)])
        
        propagate_sharding(rule, [in_a], [out_b])
        
        # Transpose swaps shardings
        self.assertEqual(out_b.dim_specs[0].axes, ["y"])
        self.assertEqual(out_b.dim_specs[1].axes, ["x"])

    def test_61_template_einsum_notation(self):
        """Test template string representation."""
        template = OpShardingRuleTemplate(
            input_mappings=[{0: ["m"], 1: ["k"]}, {0: ["k"], 1: ["n"]}],
            output_mappings=[{0: ["m"], 1: ["n"]}]
        )
        notation = template.to_einsum_notation()
        self.assertIn("m", notation)
        self.assertIn("k", notation)
        self.assertIn("n", notation)
        self.assertIn("->", notation)

    # --- Group 18: Irregular Op Templates ---
    def test_62_gather_template_basic(self):
        """Test gather template correctly handles indexed dimension."""
        mesh = DeviceMesh("m", (2, 4), ("x", "y"), list(range(8)))
        
        # gather(data[4,8,16], indices[2,3], axis=1)
        # data: [d0, d1, d2], indices: [i0, i1]
        # output: [d0, i0, i1, d2] (d1 replaced by i0, i1)
        template = gather_template(data_rank=3, indices_rank=2, axis=1)
        rule = template.instantiate(
            input_shapes=[(4, 8, 16), (2, 3)],
            output_shapes=[(4, 2, 3, 16)]
        )
        
        # Data sharded on d0 (batch) and d1 (indexed dim)
        data_spec = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec(["y"]), DimSpec([])])
        indices_spec = ShardingSpec(mesh, [DimSpec([], is_open=True), DimSpec([], is_open=True)])
        output_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),  # d0
            DimSpec([], is_open=True),  # i0
            DimSpec([], is_open=True),  # i1
            DimSpec([], is_open=True),  # d2
        ])
        
        propagate_sharding(rule, [data_spec, indices_spec], [output_spec])
        
        # d0 (x) should propagate to output[0]
        self.assertEqual(output_spec.dim_specs[0].axes, ["x"])
        # d1 (y) should NOT propagate - it's the indexed dimension (only in data)
        # indices dims i0, i1 are unsharded, so output[1], output[2] stay empty
        self.assertEqual(output_spec.dim_specs[1].axes, [])
        self.assertEqual(output_spec.dim_specs[2].axes, [])
        # d2 is replicated in data, stays replicated
        self.assertEqual(output_spec.dim_specs[3].axes, [])

    def test_63_gather_preserves_non_indexed_sharding(self):
        """Gather preserves sharding on non-indexed dimensions."""
        mesh = DeviceMesh("m", (2, 4), ("x", "y"), list(range(8)))
        
        # Simple gather along axis 0
        template = gather_template(data_rank=2, indices_rank=1, axis=0)
        rule = template.instantiate(
            input_shapes=[(8, 16), (4,)],
            output_shapes=[(4, 16)]
        )
        
        # Data: dim1 sharded on y
        data_spec = ShardingSpec(mesh, [DimSpec([]), DimSpec(["y"])])
        indices_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        output_spec = ShardingSpec(mesh, [DimSpec([], is_open=True), DimSpec([], is_open=True)])
        
        propagate_sharding(rule, [data_spec, indices_spec], [output_spec])
        
        # dim1's sharding (y) should propagate to output
        self.assertEqual(output_spec.dim_specs[1].axes, ["y"])

    def test_64_attention_template_head_sharding(self):
        """Test attention template supports tensor parallelism on heads."""
        mesh = DeviceMesh("m", (2, 4), ("dp", "tp"), list(range(8)))
        
        # Attention: Q, K, V all (batch, heads, seq, dim)
        template = attention_template(batch_dims=1, has_head_dim=True)
        rule = template.instantiate(
            input_shapes=[(2, 8, 64, 64), (2, 8, 64, 64), (2, 8, 64, 64)],  # Q, K, V
            output_shapes=[(2, 8, 64, 64)]
        )
        
        # Shard batch on dp, heads on tp
        q_spec = ShardingSpec(mesh, [
            DimSpec(["dp"]),   # batch
            DimSpec(["tp"]),   # heads
            DimSpec([]),       # seq_q
            DimSpec([]),       # head_dim
        ])
        k_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),  # batch
            DimSpec([], is_open=True),  # heads  
            DimSpec([]),                 # seq_kv
            DimSpec([]),                 # head_dim
        ])
        v_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),
            DimSpec([], is_open=True),
            DimSpec([]),
            DimSpec([]),
        ])
        out_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),
            DimSpec([], is_open=True),
            DimSpec([], is_open=True),
            DimSpec([], is_open=True),
        ])
        
        propagate_sharding(rule, [q_spec, k_spec, v_spec], [out_spec])
        
        # Batch (dp) and heads (tp) should propagate
        self.assertEqual(out_spec.dim_specs[0].axes, ["dp"])  # batch
        self.assertEqual(out_spec.dim_specs[1].axes, ["tp"])  # heads
        # seq_q from Q propagates to output (same factor)
        # head_dim is shared but unsharded
        
    def test_65_embedding_template_basic(self):
        """Test embedding lookup template."""
        mesh = DeviceMesh("m", (2, 4), ("x", "y"), list(range(8)))
        
        template = embedding_template(vocab_sharded=False)
        rule = template.instantiate(
            input_shapes=[(10000, 512), (4, 32)],  # embedding, indices
            output_shapes=[(4, 32, 512)]
        )
        
        # Embedding: embed_dim sharded on y
        embed_spec = ShardingSpec(mesh, [DimSpec([]), DimSpec(["y"])])
        # Indices: batch sharded on x
        indices_spec = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
        output_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),
            DimSpec([], is_open=True),
            DimSpec([], is_open=True),
        ])
        
        propagate_sharding(rule, [embed_spec, indices_spec], [output_spec])
        
        # batch (x) and embed_dim (y) should propagate
        self.assertEqual(output_spec.dim_specs[0].axes, ["x"])  # batch
        self.assertEqual(output_spec.dim_specs[2].axes, ["y"])  # embed_dim

    def test_66_factor_sharding_state_debug(self):
        """Test FactorShardingState provides useful debug output."""
        mesh = DeviceMesh("m", (2,), ("x",), [0, 1])
        state = FactorShardingState()
        
        # Add some factors
        state.merge("m", ["x"], 0, False, mesh)
        state.merge("n", [], 1, True, mesh)  # Receptive
        state.merge("k", [], 0, False, mesh)  # Explicit replication
        
        repr_str = repr(state)
        self.assertIn("FactorShardingState", repr_str)
        self.assertIn("m:", repr_str)

    def test_67_factor_sharding_properties(self):
        """Test FactorSharding property methods."""
        # Empty open (receptive)
        f1 = FactorSharding([], 999, True)
        self.assertTrue(f1.is_receptive)
        self.assertFalse(f1.is_explicit_replication)
        self.assertFalse(f1.has_sharding)
        
        # Empty closed (explicit replication)
        f2 = FactorSharding([], 0, False)
        self.assertFalse(f2.is_receptive)
        self.assertTrue(f2.is_explicit_replication)
        self.assertFalse(f2.has_sharding)
        
        # Has sharding
        f3 = FactorSharding(["x"], 0, False)
        self.assertFalse(f3.is_receptive)
        self.assertFalse(f3.is_explicit_replication)
        self.assertTrue(f3.has_sharding)

    def test_68_op_sharding_rule_get_factor_tensors(self):
        """Test OpShardingRule.get_factor_tensors helper."""
        rule = OpShardingRule(
            [{0: ["m"], 1: ["k"]}, {0: ["k"], 1: ["n"]}],  # A, B
            [{0: ["m"], 1: ["n"]}],  # C
            {"m": 4, "k": 8, "n": 16}
        )
        
        # Factor "m" appears in A[0] and C[0]
        m_locs = rule.get_factor_tensors("m")
        self.assertEqual(len(m_locs), 2)
        self.assertIn(("input", 0, 0), m_locs)
        self.assertIn(("output", 0, 0), m_locs)
        
        # Factor "k" appears in A[1] and B[0] only (contraction)
        k_locs = rule.get_factor_tensors("k")
        self.assertEqual(len(k_locs), 2)
        self.assertIn(("input", 0, 1), k_locs)
        self.assertIn(("input", 1, 0), k_locs)
        # "k" should NOT be in output
        self.assertNotIn(("output", 0, 0), k_locs)
        self.assertNotIn(("output", 0, 1), k_locs)

if __name__ == '__main__':
    unittest.main(argv=['ignore'], exit=False)


# =============================================================================
# SHARDING RULE TEMPLATES FOR COMMON OPS
# =============================================================================
# This section provides factory functions for building OpShardingRuleTemplates
# (shape-agnostic) and convenience functions for OpShardingRules (with sizes).
#
# Key patterns:
#   1. Elementwise: All dims map to same factors (1:1 correspondence)
#   2. Contraction: Shared dims (k) are "contracted out", others preserved
#   3. Reduction: One or more dims disappear from output
#   4. View: Dims may be split, merged, or permuted

# --- Shape-Agnostic Templates ---

def matmul_template(batch_dims: int = 0) -> OpShardingRuleTemplate:
    """Template for matmul: (...batch, m, k) @ (...batch, k, n) -> (...batch, m, n)."""
    batch_factors = [f"b{i}" for i in range(batch_dims)]
    
    a_mapping = {i: [batch_factors[i]] for i in range(batch_dims)}
    a_mapping[batch_dims] = ["m"]
    a_mapping[batch_dims + 1] = ["k"]
    
    b_mapping = {i: [batch_factors[i]] for i in range(batch_dims)}
    b_mapping[batch_dims] = ["k"]
    b_mapping[batch_dims + 1] = ["n"]
    
    c_mapping = {i: [batch_factors[i]] for i in range(batch_dims)}
    c_mapping[batch_dims] = ["m"]
    c_mapping[batch_dims + 1] = ["n"]
    
    return OpShardingRuleTemplate([a_mapping, b_mapping], [c_mapping])


def elementwise_template(rank: int, prefix: str = "d") -> OpShardingRuleTemplate:
    """Template for elementwise ops: (d0, d1, ...) op (d0, d1, ...) -> (d0, d1, ...)."""
    factors = [f"{prefix}{i}" for i in range(rank)]
    mapping = {i: [factors[i]] for i in range(rank)}
    return OpShardingRuleTemplate([mapping, mapping], [mapping])


def unary_template(rank: int, prefix: str = "d") -> OpShardingRuleTemplate:
    """Template for unary ops: (d0, d1, ...) -> (d0, d1, ...)."""
    factors = [f"{prefix}{i}" for i in range(rank)]
    mapping = {i: [factors[i]] for i in range(rank)}
    return OpShardingRuleTemplate([mapping], [mapping])


def transpose_template(rank: int, perm: List[int]) -> OpShardingRuleTemplate:
    """Template for transpose: (d0, d1, d2) -> (d2, d0, d1) for perm=[2,0,1]."""
    factors = [f"d{i}" for i in range(rank)]
    in_mapping = {i: [factors[i]] for i in range(rank)}
    out_mapping = {i: [factors[perm[i]]] for i in range(rank)}
    return OpShardingRuleTemplate([in_mapping], [out_mapping])


def reduce_template(rank: int, reduce_dims: List[int], keepdims: bool = False) -> OpShardingRuleTemplate:
    """Template for reduction ops: (d0, d1, d2) -> (d0, d2) if reducing dim 1."""
    factors = [f"d{i}" for i in range(rank)]
    reduce_set = set(reduce_dims)
    
    in_mapping = {i: [factors[i]] for i in range(rank)}
    
    out_mapping = {}
    out_idx = 0
    for i in range(rank):
        if i in reduce_set:
            if keepdims:
                out_mapping[out_idx] = []
                out_idx += 1
        else:
            out_mapping[out_idx] = [factors[i]]
            out_idx += 1
    
    return OpShardingRuleTemplate([in_mapping], [out_mapping])


def reshape_template(
    in_to_factors: Dict[int, List[str]],
    out_to_factors: Dict[int, List[str]]
) -> OpShardingRuleTemplate:
    """
    Template for reshape with explicit factor assignment.
    
    Example: reshape (8,) -> (2, 4)
        in_to_factors = {0: ["f1", "f2"]}
        out_to_factors = {0: ["f1"], 1: ["f2"]}
    """
    return OpShardingRuleTemplate([in_to_factors], [out_to_factors])


# --- Convenience Functions (with sizes, backward compatible) ---

def make_elementwise_rule(rank: int, prefix: str = "d") -> OpShardingRule:
    """Create sharding rule for elementwise/binary ops (add, mul, etc.).
    
    Pattern: (d0, d1, ...) op (d0, d1, ...) -> (d0, d1, ...)
    All dimensions map to the same factors.
    """
    factors = [f"{prefix}{i}" for i in range(rank)]
    mapping = {i: [factors[i]] for i in range(rank)}
    sizes = {f: 1 for f in factors}  # Sizes filled in at runtime
    return OpShardingRule([mapping, mapping], [mapping], sizes)


def make_unary_rule(rank: int, prefix: str = "d") -> OpShardingRule:
    """Create sharding rule for unary ops (exp, sin, neg, etc.).
    
    Pattern: (d0, d1, ...) -> (d0, d1, ...)
    """
    factors = [f"{prefix}{i}" for i in range(rank)]
    mapping = {i: [factors[i]] for i in range(rank)}
    sizes = {f: 1 for f in factors}
    return OpShardingRule([mapping], [mapping], sizes)


def make_matmul_rule(
    m: int, k: int, n: int,
    batch_dims: int = 0
) -> OpShardingRule:
    """Create sharding rule for matmul: (..., m, k) @ (..., k, n) -> (..., m, n).
    
    Args:
        m, k, n: Core matrix dimensions
        batch_dims: Number of leading batch dimensions (broadcasted)
    
    Note: For batched matmul, batch dims are shared across all tensors.
    The contraction dim 'k' appears in both inputs but NOT in output.
    """
    # Batch factors: b0, b1, ...
    batch_factors = [f"b{i}" for i in range(batch_dims)]
    
    # Build mappings
    # A: (...batch, m, k)
    a_mapping = {i: [batch_factors[i]] for i in range(batch_dims)}
    a_mapping[batch_dims] = ["m"]
    a_mapping[batch_dims + 1] = ["k"]
    
    # B: (...batch, k, n)
    b_mapping = {i: [batch_factors[i]] for i in range(batch_dims)}
    b_mapping[batch_dims] = ["k"]
    b_mapping[batch_dims + 1] = ["n"]
    
    # C: (...batch, m, n)
    c_mapping = {i: [batch_factors[i]] for i in range(batch_dims)}
    c_mapping[batch_dims] = ["m"]
    c_mapping[batch_dims + 1] = ["n"]
    
    sizes = {"m": m, "k": k, "n": n}
    sizes.update({f"b{i}": 1 for i in range(batch_dims)})  # Batch sizes filled at runtime
    
    return OpShardingRule([a_mapping, b_mapping], [c_mapping], sizes)


def make_reduce_rule(
    rank: int,
    reduce_dims: List[int],
    keepdims: bool = False,
    prefix: str = "d"
) -> OpShardingRule:
    """Create sharding rule for reduction ops (sum, mean, max, etc.).
    
    Args:
        rank: Input tensor rank
        reduce_dims: Dimensions to reduce over
        keepdims: If True, reduced dims become size 1 (still present)
    
    Pattern: (d0, d1, d2) -> (d0, d2) if reducing dim 1 with keepdims=False
    """
    factors = [f"{prefix}{i}" for i in range(rank)]
    reduce_set = set(reduce_dims)
    
    # Input mapping: all dims
    in_mapping = {i: [factors[i]] for i in range(rank)}
    
    # Output mapping: skip reduced dims (unless keepdims)
    out_mapping = {}
    out_idx = 0
    for i in range(rank):
        if i in reduce_set:
            if keepdims:
                # Reduced dim kept but no sharding factor (replicated)
                out_mapping[out_idx] = []
                out_idx += 1
        else:
            out_mapping[out_idx] = [factors[i]]
            out_idx += 1
    
    sizes = {f: 1 for f in factors}
    return OpShardingRule([in_mapping], [out_mapping], sizes)


def make_transpose_rule(rank: int, perm: List[int]) -> OpShardingRule:
    """Create sharding rule for transpose/permute.
    
    Pattern: (d0, d1, d2) -> (d2, d0, d1) for perm=[2,0,1]
    """
    factors = [f"d{i}" for i in range(rank)]
    
    in_mapping = {i: [factors[i]] for i in range(rank)}
    out_mapping = {i: [factors[perm[i]]] for i in range(rank)}
    
    sizes = {f: 1 for f in factors}
    return OpShardingRule([in_mapping], [out_mapping], sizes)


def make_reshape_rule(
    in_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
    factor_mapping: Dict[int, List[int]]
) -> OpShardingRule:
    """Create sharding rule for reshape (manual factor assignment).
    
    Args:
        in_shape: Input shape
        out_shape: Output shape  
        factor_mapping: Maps input dim -> list of output dims it maps to
                       (for splits) or multiple input dims map to one output
                       (for merges)
    
    Example - reshape (8,) -> (2, 4):
        factor_mapping = {0: [0, 1]}  # Input dim 0 splits into output dims 0,1
        Creates factors f0, f1 where input[0] -> [f0, f1], output[0] -> f0, output[1] -> f1
    """
    # This is complex - just a sketch. Real impl needs careful factor assignment.
    # Key insight: when dim splits, use compound factors; when dims merge, share factor.
    
    factors = []
    in_mapping: Dict[int, List[str]] = {}
    out_mapping: Dict[int, List[str]] = {}
    sizes = {}
    
    factor_idx = 0
    for in_dim, out_dims in factor_mapping.items():
        if len(out_dims) == 1:
            # 1:1 mapping
            f = f"f{factor_idx}"
            factors.append(f)
            in_mapping.setdefault(in_dim, []).append(f)
            out_mapping.setdefault(out_dims[0], []).append(f)
            sizes[f] = in_shape[in_dim]
            factor_idx += 1
        else:
            # Split: one input dim -> multiple output dims
            for out_dim in out_dims:
                f = f"f{factor_idx}"
                factors.append(f)
                in_mapping.setdefault(in_dim, []).append(f)
                out_mapping.setdefault(out_dim, []).append(f)
                sizes[f] = out_shape[out_dim]
                factor_idx += 1
    
    return OpShardingRule([in_mapping], [out_mapping], sizes)


def make_broadcast_rule(
    in_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...]
) -> OpShardingRule:
    """Create sharding rule for broadcast.
    
    Broadcast adds/expands dims from right alignment.
    New dims (size 1 -> N) are replicated.
    """
    in_rank = len(in_shape)
    out_rank = len(out_shape)
    offset = out_rank - in_rank
    
    factors = [f"d{i}" for i in range(out_rank)]
    
    # Input maps to rightmost factors
    in_mapping = {i: [factors[offset + i]] for i in range(in_rank)}
    
    # Output: all factors, but new dims are empty (replicated)
    out_mapping = {}
    for i in range(out_rank):
        if i < offset:
            out_mapping[i] = []  # New dim, replicated
        else:
            in_dim = i - offset
            if in_shape[in_dim] == 1 and out_shape[i] > 1:
                out_mapping[i] = []  # Broadcast dim, replicated
            else:
                out_mapping[i] = [factors[i]]
    
    sizes = {f"d{i}": out_shape[i] for i in range(out_rank)}
    return OpShardingRule([in_mapping], [out_mapping], sizes)


def make_concat_rule(num_inputs: int, rank: int, axis: int) -> OpShardingRule:
    """Create sharding rule for concatenation along axis.
    
    All inputs must have same sharding on non-concat dims.
    Concat dim sharding is tricky - typically requires same sharding or
    results in replicated concat dim.
    """
    factors = [f"d{i}" for i in range(rank)]
    
    # All inputs share same factors except concat axis
    base_mapping = {i: [factors[i]] for i in range(rank) if i != axis}
    base_mapping[axis] = []  # Concat axis typically replicated for simplicity
    
    input_mappings = [dict(base_mapping) for _ in range(num_inputs)]
    output_mapping = dict(base_mapping)
    
    sizes = {f: 1 for f in factors}
    return OpShardingRule(input_mappings, [output_mapping], sizes)


def make_slice_rule(rank: int, sliced_dims: Set[int]) -> OpShardingRule:
    """Create sharding rule for slice/index.
    
    Sliced dims may have reduced size or be eliminated.
    Non-sliced dims preserve sharding.
    """
    factors = [f"d{i}" for i in range(rank)]
    
    in_mapping = {i: [factors[i]] for i in range(rank)}
    # Output: sliced dims become empty (conservatively replicated)
    out_mapping = {
        i: ([] if i in sliced_dims else [factors[i]])
        for i in range(rank)
    }
    
    sizes = {f: 1 for f in factors}
    return OpShardingRule([in_mapping], [out_mapping], sizes)


# =============================================================================
# IRREGULAR OPS: Gather, Scatter, Attention
# =============================================================================
# 
# These operations have non-trivial sharding semantics that the factor system
# handles elegantly by using INDEPENDENT factors for dimensions that don't
# directly correspond.
#
# Key insight: A factor that appears in only ONE tensor represents a dimension
# that is INDEPENDENT of other tensors. This naturally encodes:
#   - Gather: indexed dimension is independent of output
#   - Scatter: similar to gather but reversed
#   - Attention: Q and K sequence dims are independent


def gather_template(data_rank: int, indices_rank: int, axis: int) -> OpShardingRuleTemplate:
    """
    Template for gather/index_select: output = data[indices] along axis.
    
    data: (d0, d1, ..., d_axis, ..., d_n)
    indices: (i0, i1, ..., i_m)
    output: (d0, ..., d_{axis-1}, i0, ..., i_m, d_{axis+1}, ..., d_n)
    
    Key insight: The indexed dimension (d_axis) is INDEPENDENT of indices.
    Factor d_axis only appears in data, not in indices or output.
    This means sharding on d_axis doesn't propagate - exactly right!
    
    Example: gather(data[4,8,16], indices[2,3], axis=1)
        data factors:    [d0, d1, d2]           # d1 is indexed dim
        indices factors: [i0, i1]
        output factors:  [d0, i0, i1, d2]       # d1 gone, replaced by i0,i1
        
    If data is sharded on d1, that sharding stays local to data
    (requires gather communication). Other dims propagate normally.
    """
    # Factors for data dims (d_axis will only appear in data)
    data_factors = [f"d{i}" for i in range(data_rank)]
    
    # Factors for indices dims
    indices_factors = [f"i{i}" for i in range(indices_rank)]
    
    # Data mapping: all dims including the indexed one
    data_mapping = {i: [data_factors[i]] for i in range(data_rank)}
    
    # Indices mapping: all dims
    indices_mapping = {i: [indices_factors[i]] for i in range(indices_rank)}
    
    # Output mapping: replace data[axis] with all indices dims
    out_mapping = {}
    out_idx = 0
    for i in range(data_rank):
        if i == axis:
            # Insert all indices dims here
            for j in range(indices_rank):
                out_mapping[out_idx] = [indices_factors[j]]
                out_idx += 1
        else:
            out_mapping[out_idx] = [data_factors[i]]
            out_idx += 1
    
    return OpShardingRuleTemplate([data_mapping, indices_mapping], [out_mapping])


def scatter_template(
    data_rank: int, 
    indices_rank: int, 
    updates_rank: int,
    axis: int
) -> OpShardingRuleTemplate:
    """
    Template for scatter/index_put: output = scatter(data, indices, updates, axis).
    
    data: (d0, d1, ..., d_axis, ..., d_n)     # Base tensor
    indices: (i0, i1, ..., i_m)                # Where to scatter
    updates: (d0, ..., i0, ..., i_m, ..., d_n) # Values to scatter
    output: (d0, d1, ..., d_axis, ..., d_n)    # Same shape as data
    
    Key constraint: data and output MUST have identical sharding (in-place semantics).
    The axis dimension sharding is independent of indices.
    
    This is the inverse of gather - updates has the "gathered" shape.
    """
    data_factors = [f"d{i}" for i in range(data_rank)]
    indices_factors = [f"i{i}" for i in range(indices_rank)]
    
    # Data mapping
    data_mapping = {i: [data_factors[i]] for i in range(data_rank)}
    
    # Indices mapping
    indices_mapping = {i: [indices_factors[i]] for i in range(indices_rank)}
    
    # Updates has same structure as gather output
    updates_mapping = {}
    u_idx = 0
    for i in range(data_rank):
        if i == axis:
            for j in range(indices_rank):
                updates_mapping[u_idx] = [indices_factors[j]]
                u_idx += 1
        else:
            updates_mapping[u_idx] = [data_factors[i]]
            u_idx += 1
    
    # Output same as data (identity mapping)
    output_mapping = dict(data_mapping)
    
    return OpShardingRuleTemplate(
        [data_mapping, indices_mapping, updates_mapping],
        [output_mapping]
    )


def attention_template(
    batch_dims: int = 1,
    has_head_dim: bool = True
) -> OpShardingRuleTemplate:
    """
    Template for attention: softmax(Q @ K^T / sqrt(d)) @ V.
    
    Standard shapes (with batch_dims=1, has_head_dim=True):
        Q: (batch, heads, seq_q, head_dim)
        K: (batch, heads, seq_kv, head_dim)
        V: (batch, heads, seq_kv, head_dim)
        Output: (batch, heads, seq_q, head_dim)
    
    Key insights:
        1. batch and heads can be freely sharded (data parallel + tensor parallel)
        2. seq_q and seq_kv are INDEPENDENT (different positions in Q vs K/V)
        3. head_dim is a CONTRACTION dimension (Q @ K^T contracts over it)
        
    The factor setup:
        - "b{i}" for batch dims: shared across Q, K, V, output
        - "h" for heads: shared across all (tensor parallel)
        - "sq" for seq_q: only in Q and output
        - "skv" for seq_kv: only in K and V (NOT in output after attention)
        - "d" for head_dim: in Q, K, V but typically replicated (contraction)
        
    Note: This is simplified - real attention may have different mask shapes,
    multi-query attention, etc. Extend as needed.
    """
    batch_factors = [f"b{i}" for i in range(batch_dims)]
    
    # Q: (batch..., heads, seq_q, head_dim)
    q_mapping = {i: [batch_factors[i]] for i in range(batch_dims)}
    if has_head_dim:
        q_mapping[batch_dims] = ["h"]      # heads
        q_mapping[batch_dims + 1] = ["sq"] # seq_q
        q_mapping[batch_dims + 2] = ["d"]  # head_dim
    else:
        q_mapping[batch_dims] = ["sq"]
        q_mapping[batch_dims + 1] = ["d"]
    
    # K: (batch..., heads, seq_kv, head_dim)
    k_mapping = {i: [batch_factors[i]] for i in range(batch_dims)}
    if has_head_dim:
        k_mapping[batch_dims] = ["h"]       # heads
        k_mapping[batch_dims + 1] = ["skv"] # seq_kv
        k_mapping[batch_dims + 2] = ["d"]   # head_dim
    else:
        k_mapping[batch_dims] = ["skv"]
        k_mapping[batch_dims + 1] = ["d"]
    
    # V: (batch..., heads, seq_kv, head_dim)
    v_mapping = dict(k_mapping)  # Same as K
    
    # Output: (batch..., heads, seq_q, head_dim)
    out_mapping = dict(q_mapping)  # Same as Q
    
    return OpShardingRuleTemplate([q_mapping, k_mapping, v_mapping], [out_mapping])


def conv2d_template(
    batch_dim: bool = True,
    groups: int = 1
) -> OpShardingRuleTemplate:
    """
    Template for 2D convolution: output = conv2d(input, weight).
    
    Standard shapes:
        input: (N, C_in, H, W)
        weight: (C_out, C_in/groups, kH, kW)
        output: (N, C_out, H_out, W_out)
    
    Key insights:
        1. N (batch) freely shardable
        2. C_out freely shardable (each output channel independent)
        3. C_in is a contraction dimension (summed over)
        4. H, W spatial dims: stencil pattern (tricky for sharding)
        
    For simplicity, we treat spatial dims as independent factors that
    don't propagate sharding (would require halo exchange).
    
    Factor setup:
        - "n": batch, shared between input and output
        - "ci": input channels, only in input and weight (contraction)
        - "co": output channels, in weight and output
        - "h", "w": spatial (replicated for simplicity)
        - "kh", "kw": kernel spatial (only in weight)
    """
    # Input: (N, C_in, H, W)
    input_mapping = {
        0: ["n"],   # batch
        1: ["ci"],  # input channels (contraction)
        2: [],      # H - replicated (stencil)
        3: [],      # W - replicated (stencil)
    }
    
    # Weight: (C_out, C_in/groups, kH, kW)
    weight_mapping = {
        0: ["co"],  # output channels
        1: ["ci"],  # input channels (contraction)
        2: [],      # kH - replicated
        3: [],      # kW - replicated
    }
    
    # Output: (N, C_out, H_out, W_out)
    output_mapping = {
        0: ["n"],   # batch
        1: ["co"],  # output channels
        2: [],      # H_out - replicated
        3: [],      # W_out - replicated
    }
    
    return OpShardingRuleTemplate([input_mapping, weight_mapping], [output_mapping])


def embedding_template(vocab_sharded: bool = False) -> OpShardingRuleTemplate:
    """
    Template for embedding lookup: output = embedding[indices].
    
    Shapes:
        embedding: (vocab_size, embed_dim)
        indices: (batch, seq_len)
        output: (batch, seq_len, embed_dim)
    
    Key insights:
        1. vocab_size is the "indexed" dimension (like gather axis)
        2. embed_dim can be freely sharded (tensor parallel)
        3. batch and seq_len from indices appear in output
        
    If vocab_sharded=True, this indicates the vocab dimension CAN be sharded
    (requires all-gather during lookup). Default is replicated vocab.
    """
    # Embedding: (vocab_size, embed_dim)
    if vocab_sharded:
        embed_mapping = {0: ["v"], 1: ["e"]}
    else:
        embed_mapping = {0: [], 1: ["e"]}  # vocab replicated
    
    # Indices: (batch, seq_len)
    indices_mapping = {0: ["b"], 1: ["s"]}
    
    # Output: (batch, seq_len, embed_dim)
    output_mapping = {0: ["b"], 1: ["s"], 2: ["e"]}
    
    return OpShardingRuleTemplate([embed_mapping, indices_mapping], [output_mapping])


def softmax_template(rank: int, axis: int) -> OpShardingRuleTemplate:
    """
    Template for softmax along an axis.
    
    Softmax is special: the reduction axis requires communication if sharded.
    Other axes can be freely sharded.
    
    Pattern: softmax(x, axis) normalizes along axis
    
    We mark the softmax axis as having an empty factor (no propagation)
    to indicate it should typically be replicated.
    """
    factors = [f"d{i}" for i in range(rank)]
    
    in_mapping = {}
    out_mapping = {}
    
    for i in range(rank):
        if i == axis:
            # Softmax axis: mark as empty (communication required if sharded)
            in_mapping[i] = []
            out_mapping[i] = []
        else:
            in_mapping[i] = [factors[i]]
            out_mapping[i] = [factors[i]]
    
    return OpShardingRuleTemplate([in_mapping], [out_mapping])


# --- Quick Reference Table ---
# 
# Op Type       | Example Einsum    | Key Insight
# --------------|-------------------|------------------------------------------
# Elementwise   | ij,ij->ij        | All factors shared 1:1
# Unary         | ij->ij           | Identity factor mapping
# Matmul        | mk,kn->mn        | k is contracted (appears in inputs only)
# BatchMatmul   | bmk,bkn->bmn     | b is batch (shared), k contracted
# Reduce        | ijk->ij          | k disappears from output
# Transpose     | ij->ji           | Factors permuted
# Reshape       | (ij)->i,j        | Compound factors for splits
# Broadcast     | i->(b,i)         | New dims have no factor (replicated)
# Concat        | (i,j),(i,j)->..  | Concat axis typically replicated
# Slice         | ij[s:e]->...     | Sliced dims conservatively replicated
# Gather        | d[i]->...        | Indexed dim independent (only in data)
# Scatter       | scatter(d,i,u)   | Inverse of gather, same independence
# Attention     | QK^T/sqrt(d)@V   | seq_q and seq_kv independent
# Conv2d        | input*weight     | Spatial dims are stencil (replicated)
# Embedding     | emb[idx]         | Vocab indexed, embed_dim shardable
# Softmax       | exp(x)/sum       | Reduction axis needs communication