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
Sharding Propagation: Algorithms and Templates
==============================================

This module implements the logic for propagating sharding constraints.
It provides:
1.  **Rule System**: OpShardingRule and templates for defining operation semantics.
2.  **Propagation Algorithm**: Core logic to propagate specs through a rule.
3.  **Templates**: Factory functions for common operation rules (matmul, etc.).
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple, Union

from .spec import DeviceMesh, DimSpec, ShardingSpec

# --- Enums ---

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


# --- Factor Logic: Intermediate Representation ---

@dataclass
class FactorSharding:
    """
    Sharding state for a single factor during propagation.
    
    Attributes:
        axes: Mesh axes assigned to this factor (major to minor order)
        priority: Priority level (0 = strongest/user-specified, higher = weaker)
        is_open: If True, can accept additional sharding; if False, fixed
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
    Holds the sharding state for all factors in an OpShardingRule.
    """
    factors: Dict[str, FactorSharding] = field(default_factory=dict)
    
    def get_or_create(self, factor_name: str) -> FactorSharding:
        if factor_name not in self.factors:
            self.factors[factor_name] = FactorSharding()
        return self.factors[factor_name]
    
    def get(self, factor_name: str) -> Optional[FactorSharding]:
        return self.factors.get(factor_name)
    
    def merge(
        self,
        factor_name: str,
        new_axes: List[str],
        new_priority: int,
        new_is_open: bool,
        mesh: DeviceMesh,
        strategy: PropagationStrategy = None,
    ) -> None:
        """
        Merge new sharding information into a factor using Shardy conflict resolution semantics.
        """
        if strategy is None:
            strategy = PropagationStrategy.BASIC
            
        factor = self.get_or_create(factor_name)
        
        has_new = bool(new_axes)
        has_existing = factor.has_sharding
        new_is_receptive = not has_new and new_is_open
        new_is_explicit_repl = not has_new and not new_is_open
        
        # Case 1: New is receptive -> don't change anything
        if new_is_receptive:
            return
        
        # Case 2: Existing is explicit replication with equal/stronger priority
        if factor.is_explicit_replication and new_priority >= factor.priority:
            return
        
        # Case 3: New is explicit replication with stronger priority
        if new_is_explicit_repl and new_priority < factor.priority:
            factor.axes = []
            factor.priority = new_priority
            factor.is_open = False
            return
        
        # Case 4: New is explicit replication with equal priority
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
            return
        
        # Case 7: New has axes but weaker priority -> only update receptive factors
        if has_new and new_priority > factor.priority:
            return
        
        # Case 8: New has axes, factor was receptive -> contribute
        if has_new:
            factor.axes = list(new_axes)
            factor.priority = new_priority
            factor.is_open = new_is_open
    
    @staticmethod
    def _longest_common_prefix(list1: List[str], list2: List[str]) -> List[str]:
        common = []
        for x, y in zip(list1, list2):
            if x == y:
                common.append(x)
            else:
                break
        return common
    
    @staticmethod
    def _get_parallelism(axes: List[str], mesh: DeviceMesh) -> int:
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


# --- Rule System ---

@dataclass
class OpShardingRule:
    """
    Einsum-like factor mapping defining how shardings propagate through an operation.
    """
    input_mappings: List[Dict[int, List[str]]]  
    output_mappings: List[Dict[int, List[str]]] 
    factor_sizes: Dict[str, int]
    
    def get_all_factors(self) -> Set[str]:
        factors = set()
        for mapping in self.input_mappings + self.output_mappings:
            for factor_list in mapping.values():
                factors.update(factor_list)
        return factors
    
    def get_factor_tensors(self, factor_name: str) -> List[Tuple[str, int, int]]:
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
    
    def get_contracting_factors(self) -> Set[str]:
        """Return factors that appear only in inputs (not in outputs).
        
        These are "contracting" factors (like k in matmul A[m,k] @ B[k,n] -> C[m,n]).
        When sharded, operations on contracting factors produce partial results
        that require AllReduce to combine.
        """
        input_factors = set()
        for mapping in self.input_mappings:
            for factors in mapping.values():
                input_factors.update(factors)
        
        output_factors = set()
        for mapping in self.output_mappings:
            for factors in mapping.values():
                output_factors.update(factors)
        
        return input_factors - output_factors
    
    def to_einsum_notation(self) -> str:
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
    """Shape-agnostic sharding rule template."""
    input_mappings: List[Dict[int, List[str]]]
    output_mappings: List[Dict[int, List[str]]]
    
    def instantiate(
        self,
        input_shapes: List[Tuple[int, ...]],
        output_shapes: List[Tuple[int, ...]]
    ) -> OpShardingRule:
        """Instantiate template with concrete shapes to infer factor sizes."""
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
    
    def to_einsum_notation(self) -> str:
        # Re-use logic for consistency
        return OpShardingRule(self.input_mappings, self.output_mappings, {}).to_einsum_notation()


# --- Propagation Algorithm ---

def _expand_axes_for_factors(
    axes: List[str],
    factors: List[str],
    factor_sizes: Dict[str, int],
    mesh: DeviceMesh
) -> List[str]:
    """Expand axes into sub-axes when one axis covers multiple factors."""
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
                pre_size = 1
                for _, f_size in sub_factors:
                    expanded.append(f"{ax}:({pre_size}){f_size}")
                    pre_size *= f_size
                
                curr_ax_idx += 1
                found_split = True
                break
            
            if cum_prod > ax_size:
                break
        
        if not found_split:
            expanded.append(ax)
            curr_ax_idx += 1
    
    return expanded


def _collect_to_factors(
    specs: List[ShardingSpec],
    mappings: List[Dict[int, List[str]]],
    rule: OpShardingRule,
    mesh: DeviceMesh,
    state: FactorShardingState,
    strategy: PropagationStrategy,
    max_priority: Optional[int],
) -> None:
    """Phase 1: Project dimension shardings to factor shardings (COLLECT)."""
    for t_idx, spec in enumerate(specs):
        if t_idx >= len(mappings):
            continue
        mapping = mappings[t_idx]
        
        for dim_idx, factors in mapping.items():
            if dim_idx >= len(spec.dim_specs):
                continue
            dim_spec = spec.dim_specs[dim_idx]
            
            if max_priority is not None and dim_spec.priority > max_priority:
                continue
            
            expanded_axes = _expand_axes_for_factors(
                dim_spec.axes, factors, rule.factor_sizes, mesh
            )
            
            available_axes = list(expanded_axes)
            
            for f in factors:
                axes_for_f = []
                if available_axes:
                    proposed_axis = available_axes.pop(0)
                    if proposed_axis not in spec.replicated_axes:
                        axes_for_f = [proposed_axis]
                
                state.merge(
                    f,
                    axes_for_f,
                    dim_spec.priority,
                    dim_spec.is_open,
                    mesh,
                    strategy
                )


def _should_update_dim(
    current: DimSpec,
    proposed_axes: List[str],
    proposed_priority: int,
) -> bool:
    """Determine if a dimension should be updated based on Shardy semantics.
    
    Key cases:
    - Stronger priority always wins
    - Equal priority: 
      - If current is open, can receive new sharding
      - If proposed is the "common prefix" from conflict resolution,
        we MUST update to enforce consistency (even for closed dims)
    """
    # Case 1: Stronger priority always wins
    if proposed_priority < current.priority:
        return True
    
    # Case 2: Equal priority
    if proposed_priority == current.priority:
        # Open dims can receive/extend sharding
        if current.is_open:
            if not current.axes and proposed_axes:
                return True
            if len(proposed_axes) > len(current.axes):
                if proposed_axes[:len(current.axes)] == current.axes:
                    return True
        
        # CRITICAL: If current has axes but proposed is empty or shorter,
        # this means factor resolution resulted in "common prefix" (conflict).
        # We MUST update to enforce compatibility across all tensors.
        if current.axes and (not proposed_axes or len(proposed_axes) < len(current.axes)):
            # Only enforce if proposed_axes is a prefix of current.axes
            if not proposed_axes or current.axes[:len(proposed_axes)] == proposed_axes:
                return True
    
    # Case 3: Weaker priority - only update empty open dims
    if proposed_priority > current.priority:
        if current.is_open and not current.axes and proposed_axes:
            return True
    
    return False




def _update_from_factors(
    specs: List[ShardingSpec],
    mappings: List[Dict[int, List[str]]],
    state: FactorShardingState,
) -> bool:
    """Phase 3: Project factor shardings back to dimension shardings (UPDATE)."""
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


def propagate_sharding(
    rule: OpShardingRule,
    input_specs: List[ShardingSpec],
    output_specs: List[ShardingSpec],
    strategy: PropagationStrategy = PropagationStrategy.BASIC,
    max_priority: Optional[int] = None,
) -> bool:
    """
    Propagate shardings between inputs/outputs using the factor-based algorithm.
    Returns True if any spec was modified.
    """
    if not input_specs and not output_specs:
        return False
    
    mesh = input_specs[0].mesh if input_specs else output_specs[0].mesh
    
    # Phase 1: Collect
    state = FactorShardingState()
    _collect_to_factors(input_specs, rule.input_mappings, rule, mesh, state, strategy, max_priority)
    _collect_to_factors(output_specs, rule.output_mappings, rule, mesh, state, strategy, max_priority)
    
    # Phase 3: Update (Phase 2 Resolve is implicit in Collect/Merge)
    changed = False
    if _update_from_factors(input_specs, rule.input_mappings, state):
        changed = True
    if _update_from_factors(output_specs, rule.output_mappings, state):
        changed = True
    
    return changed


# --- Template Factories ---

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


def broadcast_matmul_template(
    a_rank: int, b_rank: int, out_rank: int
) -> OpShardingRuleTemplate:
    """Template for broadcast matmul with different input ranks.
    
    Handles cases like:
    - (batch, m, k) @ (k, n) -> (batch, m, n)  # weights don't have batch
    - (m, k) @ (batch, k, n) -> (batch, m, n)  # activations don't have batch
    
    The batch dims come from whichever input has them (broadcast semantics).
    """
    out_batch_dims = out_rank - 2
    a_batch_dims = a_rank - 2
    b_batch_dims = b_rank - 2
    
    batch_factors = [f"b{i}" for i in range(out_batch_dims)]
    
    # Input A mapping: only include batch factors if it has batch dims
    a_mapping = {}
    if a_batch_dims > 0:
        for i in range(a_batch_dims):
            a_mapping[i] = [batch_factors[i]]
    a_mapping[a_rank - 2] = ["m"]  # second-to-last = m
    a_mapping[a_rank - 1] = ["k"]  # last = k
    
    # Input B mapping: only include batch factors if it has batch dims
    b_mapping = {}
    if b_batch_dims > 0:
        for i in range(b_batch_dims):
            b_mapping[i] = [batch_factors[i]]
    b_mapping[b_rank - 2] = ["k"]  # second-to-last = k
    b_mapping[b_rank - 1] = ["n"]  # last = n
    
    # Output mapping: always has full batch dims
    c_mapping = {i: [batch_factors[i]] for i in range(out_batch_dims)}
    c_mapping[out_batch_dims] = ["m"]
    c_mapping[out_batch_dims + 1] = ["n"]
    
    return OpShardingRuleTemplate([a_mapping, b_mapping], [c_mapping])

def elementwise_template(rank: int, prefix: str = "d") -> OpShardingRuleTemplate:
    factors = [f"{prefix}{i}" for i in range(rank)]
    mapping = {i: [factors[i]] for i in range(rank)}
    return OpShardingRuleTemplate([mapping, mapping], [mapping])

def unary_template(rank: int, prefix: str = "d") -> OpShardingRuleTemplate:
    factors = [f"{prefix}{i}" for i in range(rank)]
    mapping = {i: [factors[i]] for i in range(rank)}
    return OpShardingRuleTemplate([mapping], [mapping])

def broadcast_template(in_rank: int, out_rank: int) -> OpShardingRuleTemplate:
    """Template for broadcast operation: input dims align to output SUFFIX.
    
    For broadcast (4,) -> (4,4) with numpy semantics:
    - Input dim0 maps to output dim1 (last dim)
    - Output dim0 is NEW (gets a new factor, replicated)
    
    Rule: (j) -> (i, j) where i is new, j preserves sharding from input.
    
    Also handles dimension expansion where input dim has size 1 but output has larger size.
    In that case, the expanded dimension gets a NEW factor since it's being replicated.
    """
    # New dimensions get new factors (will be replicated since input doesn't have them)
    new_dims = out_rank - in_rank
    
    # For same-rank broadcast, we need to mark size-1 dimensions as NEW factors
    # since their values are being replicated
    out_factors = []
    in_factors = [f"d{i}" for i in range(in_rank)]
    
    # Output factors: first new_dims are brand new, rest align with input suffix
    for i in range(new_dims):
        out_factors.append(f"new{i}")  # New dimensions from rank expansion
    for i in range(in_rank):
        out_factors.append(f"d{i}")  # Align with input dimensions
    
    in_mapping = {i: [in_factors[i]] for i in range(in_rank)}
    out_mapping = {i: [out_factors[i]] for i in range(out_rank)}
    
    return OpShardingRuleTemplate([in_mapping], [out_mapping])


def broadcast_with_shapes_template(in_shape: tuple, out_shape: tuple) -> OpShardingRuleTemplate:
    """Template for broadcast with known shapes.
    
    This handles both rank expansion and dimension expansion (size 1 -> N).
    Dimensions that are replicated (1 -> N) get new factors.
    Dimensions that match get shared factors.
    """
    in_rank = len(in_shape)
    out_rank = len(out_shape)
    new_dims = out_rank - in_rank
    
    # Align input to output suffix
    in_factors = []
    out_factors = []
    factor_idx = 0
    
    # First new_dims of output are new factors
    for i in range(new_dims):
        out_factors.append(f"new{i}")
    
    # Remaining output dims align with input
    for i in range(in_rank):
        in_size = in_shape[i]
        out_idx = new_dims + i
        out_size = out_shape[out_idx] if out_idx < len(out_shape) else 0
        
        if in_size == 1 and out_size > 1:
            # Dimension expansion: size 1 -> N means replication
            # Input dim gets its own factor, output gets a NEW factor
            in_factors.append(f"in{i}")
            out_factors.append(f"expand{i}")
        else:
            # Same size: shared factor
            in_factors.append(f"d{i}")
            out_factors.append(f"d{i}")
    
    in_mapping = {i: [in_factors[i]] for i in range(in_rank)}
    out_mapping = {i: [out_factors[i]] for i in range(out_rank)}
    
    return OpShardingRuleTemplate([in_mapping], [out_mapping])



def unsqueeze_template(in_rank: int, axis: int) -> OpShardingRuleTemplate:
    """Template for unsqueeze: insert a new dimension at axis.
    
    For unsqueeze (4,) axis=0 -> (1,4):
    - New dim at axis=0 gets new factor (replicated)
    - Input dim0 -> output dim1 (shifted by 1)
    
    Rule: (d0) -> (new, d0)
    """
    out_rank = in_rank + 1
    # Normalize axis
    if axis < 0:
        axis = out_rank + axis
    
    # Input factors
    in_factors = [f"d{i}" for i in range(in_rank)]
    
    # Output factors: insert "new" at axis position
    out_factors = in_factors[:axis] + [] + in_factors[axis:]  # Copy input factors
    
    # Build output by inserting new factor at axis
    out_factors_with_new = []
    in_idx = 0
    for i in range(out_rank):
        if i == axis:
            out_factors_with_new.append("new")  # New dimension
        else:
            out_factors_with_new.append(in_factors[in_idx])
            in_idx += 1
    
    in_mapping = {i: [in_factors[i]] for i in range(in_rank)}
    out_mapping = {i: [out_factors_with_new[i]] for i in range(out_rank)}
    
    return OpShardingRuleTemplate([in_mapping], [out_mapping])



def transpose_template(rank: int, perm: List[int]) -> OpShardingRuleTemplate:
    factors = [f"d{i}" for i in range(rank)]
    in_mapping = {i: [factors[i]] for i in range(rank)}
    out_mapping = {i: [factors[perm[i]]] for i in range(rank)}
    return OpShardingRuleTemplate([in_mapping], [out_mapping])

def squeeze_template(in_rank: int, axis: int) -> OpShardingRuleTemplate:
    """Template for squeeze: remove dimension at axis (must be size 1).
    
    For squeeze (1,4) axis=0 -> (4,):
    - Dim at axis=0 is removed (was size 1, no sharding)
    - Input dim1 -> output dim0 (shifted down)
    
    Rule: (removed, d0) -> (d0)
    """
    out_rank = in_rank - 1
    # Normalize axis
    if axis < 0:
        axis = in_rank + axis
    
    # Input factors: the squeezed dimension gets a dummy factor
    in_factors = []
    out_idx = 0
    for i in range(in_rank):
        if i == axis:
            in_factors.append("squeezed")  # Will be removed
        else:
            in_factors.append(f"d{out_idx}")
            out_idx += 1
    
    # Output factors: all dims except the squeezed one
    out_factors = [f"d{i}" for i in range(out_rank)]
    
    in_mapping = {i: [in_factors[i]] for i in range(in_rank)}
    out_mapping = {i: [out_factors[i]] for i in range(out_rank)}
    
    return OpShardingRuleTemplate([in_mapping], [out_mapping])

def swap_axes_template(rank: int, axis1: int, axis2: int) -> OpShardingRuleTemplate:
    """Template for swap_axes: swap two dimensions.
    
    This is equivalent to transpose with a permutation that swaps axis1 and axis2.
    """
    # Normalize axes
    if axis1 < 0:
        axis1 = rank + axis1
    if axis2 < 0:
        axis2 = rank + axis2
    
    # Create permutation that swaps axis1 and axis2
    perm = list(range(rank))
    perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
    
    return transpose_template(rank, perm)

def reduce_template(rank: int, reduce_dims: List[int], keepdims: bool = False) -> OpShardingRuleTemplate:
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

def reshape_template(in_shape: Tuple[int, ...], out_shape: Tuple[int, ...]) -> OpShardingRuleTemplate:
    """Create sharding rule template for reshape with compound factors.
    
    This automatically determines factor mappings for arbitrary reshapes by:
    1. Finding which dimensions merge/split
    2. Creating atomic factors from the shape with HIGHER rank (more granular)
    3. Mapping compound factors for the shape with LOWER rank (less granular)
    
    Examples:
        reshape_template((2, 4), (8,))     -> {0: ["d0"], 1: ["d1"]} -> {0: ["d0", "d1"]}
        reshape_template((8,), (2, 4))     -> {0: ["d0", "d1"]} -> {0: ["d0"], 1: ["d1"]}
    """
    import math
    
    in_size = math.prod(in_shape)
    out_size = math.prod(out_shape)
    
    if in_size != out_size:
        raise ValueError(f"Reshape size mismatch: {in_size} != {out_size}")
    
    in_rank = len(in_shape)
    out_rank = len(out_shape)
    
    # Strategy: Use the shape with more dimensions as the source of atomic factors
    # This assumes that the higher-rank shape creates the granularity boundary
    
    if in_rank >= out_rank:
        # Input has more/equal dims: Input dims are ATOMIC factors
        # Output dims are COMPOUND of input factors
        factors = [f"d{i}" for i in range(in_rank)]
        in_mapping = {i: [factors[i]] for i in range(in_rank)}
        
        out_mapping = {}
        factor_idx = 0
        current_prod = 1
        current_factors = []
        
        # Iterate over output dims and try to form them from input factors
        for out_dim_idx in range(out_rank):
            target_size = out_shape[out_dim_idx]
            
            # Consume input factors until we match target size
            while factor_idx < in_rank:
                f_size = in_shape[factor_idx]
                current_factors.append(factors[factor_idx])
                current_prod *= f_size
                factor_idx += 1
                
                if current_prod == target_size:
                    # Match found!
                    out_mapping[out_dim_idx] = list(current_factors)
                    current_factors = []
                    current_prod = 1
                    break
                elif current_prod > target_size:
                    # Input granularity is coarser than output required
                    # This happens if input dim splits into multiple output dims
                    # BUT we assumed in_rank >= out_rank, so this usually implies simple merge
                    # OR mixed split/merge.
                    # Fallback for now: Assign remaining factors to this output dim?
                    # Strict mapping requires splitting factors, which this template doesn't do yet.
                    # We'll assign current factors and move on, relying on runtime checks.
                    out_mapping[out_dim_idx] = list(current_factors)
                    current_factors = []
                    current_prod = 1
                    break
            
            if out_dim_idx not in out_mapping and current_factors:
                 # Flush remaining if we exhausted factors but didn't exact match (shouldn't happen if shapes align)
                 out_mapping[out_dim_idx] = list(current_factors)

    else:
        # Output has more dims: Output dims are ATOMIC factors
        # Input dims are COMPOUND of output factors
        factors = [f"d{i}" for i in range(out_rank)]
        out_mapping = {i: [factors[i]] for i in range(out_rank)}
        
        in_mapping = {}
        factor_idx = 0
        current_prod = 1
        current_factors = []
        
        # Iterate over input dims and try to form them from output factors
        for in_dim_idx in range(in_rank):
            target_size = in_shape[in_dim_idx]
            
            # Consume output factors until we match target size
            while factor_idx < out_rank:
                f_size = out_shape[factor_idx]
                current_factors.append(factors[factor_idx])
                current_prod *= f_size
                factor_idx += 1
                
                if current_prod == target_size:
                    # Match found!
                    in_mapping[in_dim_idx] = list(current_factors)
                    current_factors = []
                    current_prod = 1
                    break
                elif current_prod > target_size:
                    in_mapping[in_dim_idx] = list(current_factors)
                    current_factors = []
                    current_prod = 1
                    break

            if in_dim_idx not in in_mapping and current_factors:
                 in_mapping[in_dim_idx] = list(current_factors)
    
    return OpShardingRuleTemplate([in_mapping], [out_mapping])

def gather_template(data_rank: int, indices_rank: int, axis: int) -> OpShardingRuleTemplate:
    data_factors = [f"d{i}" for i in range(data_rank)]
    indices_factors = [f"i{i}" for i in range(indices_rank)]
    
    data_mapping = {i: [data_factors[i]] for i in range(data_rank)}
    indices_mapping = {i: [indices_factors[i]] for i in range(indices_rank)}
    
    out_mapping = {}
    out_idx = 0
    for i in range(data_rank):
        if i == axis:
            for j in range(indices_rank):
                out_mapping[out_idx] = [indices_factors[j]]
                out_idx += 1
        else:
            out_mapping[out_idx] = [data_factors[i]]
            out_idx += 1
    
    return OpShardingRuleTemplate([data_mapping, indices_mapping], [out_mapping])

def attention_template(batch_dims: int = 1, has_head_dim: bool = True) -> OpShardingRuleTemplate:
    batch_factors = [f"b{i}" for i in range(batch_dims)]
    
    q_mapping = {i: [batch_factors[i]] for i in range(batch_dims)}
    if has_head_dim:
        q_mapping[batch_dims] = ["h"]
        q_mapping[batch_dims + 1] = ["sq"]
        q_mapping[batch_dims + 2] = ["d"]
    else:
        q_mapping[batch_dims] = ["sq"]
        q_mapping[batch_dims + 1] = ["d"]
    
    k_mapping = {i: [batch_factors[i]] for i in range(batch_dims)}
    if has_head_dim:
        k_mapping[batch_dims] = ["h"]
        k_mapping[batch_dims + 1] = ["skv"]
        k_mapping[batch_dims + 2] = ["d"]
    else:
        k_mapping[batch_dims] = ["skv"]
        k_mapping[batch_dims + 1] = ["d"]
        
    v_mapping = dict(k_mapping)
    out_mapping = dict(q_mapping)
    
    return OpShardingRuleTemplate([q_mapping, k_mapping, v_mapping], [out_mapping])

def embedding_template(vocab_sharded: bool = False) -> OpShardingRuleTemplate:
    if vocab_sharded:
        embed_mapping = {0: ["v"], 1: ["e"]}
    else:
        embed_mapping = {0: [], 1: ["e"]}
    indices_mapping = {0: ["b"], 1: ["s"]}
    output_mapping = {0: ["b"], 1: ["s"], 2: ["e"]}
    return OpShardingRuleTemplate([embed_mapping, indices_mapping], [output_mapping])


# ============================================================================
# Hierarchical Propagation Pass (XLA Shardy-style)
# ============================================================================

def run_hierarchical_propagation_pass(
    operations_with_rules,
    max_user_priority: int = 10,
    max_iterations: int = 100,
) -> int:
    """Run hierarchical sharding propagation following XLA Shardy's nested loop structure.
    
    This implements the complete propagation hierarchy:
    1. User priorities (p0, p1, p2, ...)
    2. Operation priorities (PASSTHROUGH, CONTRACTION, REDUCTION, COMMUNICATION)
    3. Propagation strategies (AGGRESSIVE, BASIC)
    
    Args:
        operations_with_rules: List of (op, rule, input_specs, output_specs) tuples
        max_user_priority: Maximum user priority to propagate (default: 10)
        max_iterations: Max iterations per nested loop to prevent infinite loops
        
    Returns:
        Total number of changes made across all iterations
    
    Example:
        # operations_with_rules = [
        #     (matmul_op, matmul_rule, [a_spec, b_spec], [c_spec]),
        #     (add_op, add_rule, [c_spec, d_spec], [e_spec]),
        # ]
        # changes = run_hierarchical_propagation_pass(operations_with_rules)
    """
    total_changes = 0
    
    # Outer loop: User priorities (p0, p1, p2, ...)
    for user_priority in range(max_user_priority + 1):
        
        # Middle loop: Operation priorities (PASSTHROUGH → TRANSFORM)
        for op_priority in [OpPriority.PASSTHROUGH, OpPriority.CONTRACTION, 
                            OpPriority.REDUCTION, OpPriority.COMMUNICATION]:
            
            # Inner loop: Propagation strategies (AGGRESSIVE → BASIC)
            for strategy in [PropagationStrategy.AGGRESSIVE, PropagationStrategy.BASIC]:
                
                # Fixed-point iteration until no changes
                iteration = 0
                while iteration < max_iterations:
                    changed_this_iter = False
                    
                    # Propagate through all operations
                    for op, rule, input_specs, output_specs in operations_with_rules:
                        # Filter by operation priority
                        # (Note: operations should have op_priority attribute)
                        op_prio = getattr(op, 'op_priority', OpPriority.CONTRACTION)
                        if op_prio != op_priority:
                            continue
                        
                        # Propagate with current user priority and strategy
                        changed = propagate_sharding(
                            rule, input_specs, output_specs,
                            strategy=strategy,
                            max_priority=user_priority
                        )
                        
                        if changed:
                            changed_this_iter = True
                            total_changes += 1
                    
                    # Fixed point reached?
                    if not changed_this_iter:
                        break
                    
                    iteration += 1
                    
                    if iteration >= max_iterations:
                        import warnings
                        warnings.warn(
                            f"Propagation did not converge after {max_iterations} iterations "
                            f"at user_priority={user_priority}, op_priority={op_priority}, "
                            f"strategy={strategy}"
                        )
                        break
    
    return total_changes

