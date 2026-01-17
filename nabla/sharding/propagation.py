# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

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
        partial: If True, factor holds partial sums
    """
    axes: List[str] = field(default_factory=list)
    priority: int = 999  # Default: weakest priority (unspecified)
    is_open: bool = True
    partial: bool = False
    
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
            is_open=self.is_open,
            partial=self.partial
        )
    
    def __repr__(self) -> str:
        axes_str = ",".join(self.axes) if self.axes else "∅"
        status = "open" if self.is_open else ("repl" if self.is_explicit_replication else "closed")
        partial_str = "!" if self.partial else ""
        return f"FactorSharding({axes_str}, p{self.priority}, {status}{partial_str})"


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
        new_partial: bool,
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
        
        # Merging logic: 
        # 1. Partial is OR'd: if any contributor is partial, result is partial
        factor.partial = factor.partial or new_partial
        
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
        """Convert to einsum-like notation string.
        
        Outputs space-separated format: "m k, k n -> m n"
        - Single factors: just the factor name
        - Multiple factors on one dim: grouped with parentheses
        - Empty mapping: "1" (for reduced/broadcast dimensions)
        """
        def mapping_to_str(mapping: Dict[int, List[str]]) -> str:
            if not mapping:
                return "1"
            sorted_dims = sorted(mapping.keys())
            parts = []
            for d in sorted_dims:
                factors = mapping[d]
                if not factors:
                    parts.append("1")
                elif len(factors) == 1:
                    parts.append(factors[0])
                else:
                    # Multiple factors on one dimension - keep parentheses for grouping
                    parts.append(f"({' '.join(factors)})")
            return " ".join(parts)
        
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
        output_shapes: Optional[List[Tuple[int, ...]]] = None
    ) -> OpShardingRule:
        """Instantiate template with concrete shapes to infer factor sizes."""
        factor_sizes: Dict[str, int] = {}
        
        # Combine mappings with shapes if available
        pairs = list(zip(self.input_mappings, input_shapes))
        if output_shapes:
            pairs.extend(zip(self.output_mappings, output_shapes))
        
        # Pass 1: Infer from single-factor dimensions
        for mapping, shape in pairs:
            for dim_idx, factors in mapping.items():
                if dim_idx >= len(shape):
                    continue
                dim_size = shape[dim_idx]
                
                if len(factors) == 1:
                    f = factors[0]
                    if f in factor_sizes:
                        if factor_sizes[f] != dim_size:
                            pass
                    else:
                        factor_sizes[f] = dim_size
        
        # Pass 2: Verify compound factors and infer remaining unknowns
        for mapping, shape in pairs:
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
                    pass # Validation skipped for simplicity
                elif len(unknown_factors) == 1:
                    if dim_size % known_product == 0:
                        factor_sizes[unknown_factors[0]] = dim_size // known_product
        
        return OpShardingRule(self.input_mappings, self.output_mappings, factor_sizes)
        
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
        return OpShardingRule(self.input_mappings, self.output_mappings, {}).to_einsum_notation()

    @classmethod
    def parse(cls, equation: str, input_shapes: Optional[List[Tuple[int, ...]]] = None) -> "OpShardingRuleTemplate":
        """Create template from einsum string (e.g. 'mk,kn->mn').
        
        Supports '...' for broadcasting batch dimensions.
        """
        lhs, rhs = equation.split('->')
        input_strs = [s.strip() for s in lhs.split(',')]
        output_strs = [s.strip() for s in rhs.split(',')]
        
        input_mappings = []
        output_mappings = []
        
        def parse_factors(s: str, shape: Optional[Tuple[int, ...]] = None, batch_rank: int = 0) -> Dict[int, List[str]]:
            mapping = {}
            parts = s.split()
            idx = 0
            for part in parts:
                if part == '...':
                    if shape is not None:
                         explicit_count = len(parts) - 1
                         batch_rank = len(shape) - explicit_count
                    
                    if batch_rank < 0:
                         raise ValueError(f"Batch rank negative/invalid for spec '{s}'")
                    
                    for b in range(batch_rank):
                        mapping[idx] = [f"b{b}"]
                        idx += 1
                elif part == '1':
                    mapping[idx] = []
                    idx += 1
                else:
                    mapping[idx] = [part]
                    idx += 1
            return mapping

        # Pass 1: Inputs and deduce batch_rank
        batch_rank = 0
        for i, s in enumerate(input_strs):
            shape = input_shapes[i] if input_shapes else None
            # If shape available, we can verify/deduce batch_rank from '...'
            if '...' in s and shape:
                explicit = len(s.split()) - 1
                br = len(shape) - explicit
                if batch_rank == 0: batch_rank = br
                elif br != batch_rank:
                    batch_rank = max(batch_rank, br)

            input_mappings.append(parse_factors(s, shape, batch_rank))
            
        # Pass 2: Outputs
        for s in output_strs:
            output_mappings.append(parse_factors(s, None, batch_rank))
                 
        return cls(input_mappings, output_mappings)


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
                
                # CRITICAL FIX: Skip contributions from empty closed dimensions.
                # An empty closed dim (e.g., h's dim1 with {}) doesn't mean "force replicate this factor".
                # It means "this tensor doesn't use this factor on this dimension".
                # Other tensors (like W's dim0 for factor k) should still be able to contribute sharding.
                if not axes_for_f and not dim_spec.is_open:
                    # Empty + closed = this dim has no sharding info to contribute
                    continue
                
                state.merge(
                    f,
                    axes_for_f,
                    dim_spec.priority,
                    dim_spec.is_open,
                    dim_spec.partial,
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
        
        if current.axes and (not proposed_axes or len(proposed_axes) < len(current.axes)):
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
    """Phase 3: Project factor shardings back to dimension shardings (UPDATE).
    
    IMPORTANT: Tracks axes used across ALL dimensions of a tensor to prevent
    the same axis from being assigned to multiple dimensions (which is invalid).
    If a conflict arises, later dimensions get the axis stripped (replicated).
    """
    did_change = False
    
    for t_idx, spec in enumerate(specs):
        if t_idx >= len(mappings):
            continue
        mapping = mappings[t_idx]
        new_dim_specs = []
        spec_dirty = False
        
        # Track axes already assigned to dimensions within THIS tensor
        used_axes_in_tensor: Set[str] = set()
        
        for dim_idx, current_dim in enumerate(spec.dim_specs):
            factors = mapping.get(dim_idx, [])
            
            if not factors:
                # Keep current axes as used
                used_axes_in_tensor.update(current_dim.axes)
                new_dim_specs.append(current_dim)
                continue
            
            proposed_axes = []
            proposed_prio = 999
            proposed_open = current_dim.is_open
            proposed_partial = False
            has_factor_info = False
            
            for f in factors:
                f_state = state.get(f)
                if f_state is not None:
                    valid_axes = [ax for ax in f_state.axes 
                                 if ax not in used_axes_in_tensor]
                    
                    proposed_axes.extend(valid_axes)
                    proposed_prio = min(proposed_prio, f_state.priority)
                    
                    proposed_partial = proposed_partial or f_state.partial
                    has_factor_info = True
            
            if not has_factor_info:
                used_axes_in_tensor.update(current_dim.axes)
                new_dim_specs.append(current_dim)
                continue
            
            should_update = _should_update_dim(
                current_dim, proposed_axes, proposed_prio
            ) or (proposed_partial != current_dim.partial)
            
            if should_update:
                # Mark these axes as used BEFORE adding the dim spec
                used_axes_in_tensor.update(proposed_axes)
                new_dim_specs.append(DimSpec(
                    axes=proposed_axes,
                    is_open=current_dim.is_open,
                    priority=proposed_prio,
                    partial=proposed_partial
                ))
                spec_dirty = True
            else:
                used_axes_in_tensor.update(current_dim.axes)
                new_dim_specs.append(current_dim)
        
        if spec_dirty:
            spec.dim_specs = new_dim_specs
            did_change = True
    
    return did_change


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
    if _update_from_factors(output_specs, rule.output_mappings, state):
        changed = True
    if _update_from_factors(input_specs, rule.input_mappings, state):
        changed = True
    
    return changed


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
                        op_prio = getattr(op, 'op_priority', OpPriority.CONTRACTION)
                        if op_prio != op_priority:
                            continue
                        
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

