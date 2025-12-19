"""
Shardy Propagation: Algorithms and Graph Logic
==============================================

This module implements the logic for propagating sharding constraints through a
computation graph. It includes:

1.  **Factor Logic**: Intermediate representations (FactorSharding) for propagation.
2.  **Rule System**: OpShardingRule and templates for defining operation semantics.
3.  **Propagation Algorithm**: The 3-phase collect/resolve/update algorithm.
4.  **Graph Layer**: Operation and ShardingPass definitions.
5.  **Templates**: Factory functions for common operation rules (matmul, etc.).
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple, Union

from core import DeviceMesh, DimSpec, ShardingSpec, GraphTensor

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
    """Determine if a dimension should be updated based on Shardy semantics."""
    if proposed_priority < current.priority:
        return True
    
    if proposed_priority == current.priority:
        if current.is_open:
            if not current.axes and proposed_axes:
                return True
            if len(proposed_axes) > len(current.axes):
                if proposed_axes[:len(current.axes)] == current.axes:
                    return True
    
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


# --- Graph Layer ---

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
    """Bridge between sources and targets that must share the same sharding (e.g., control flow)."""
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
        
        ref_shape = self.sources[0].shape
        factor_names = [f"df{i}" for i in range(len(ref_shape))]
        factor_sizes = {f"df{i}": ref_shape[i] for i in range(len(ref_shape))}
        
        def make_mapping(tensor: GraphTensor) -> Dict[int, List[str]]:
            return {i: [factor_names[i]] for i in range(len(tensor.shape))}
        
        input_mappings = [make_mapping(t) for t in self.sources]
        output_mappings = [make_mapping(t) for t in self.targets]
        
        return OpShardingRule(input_mappings, output_mappings, factor_sizes)


class ShardingPass:
    """Compiler pass that propagates shardings to fixed point."""
    
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
        if not self.use_op_priority:
            return self.ops
        return sorted(self.ops, key=lambda op: op.op_priority)
    
    def _get_user_priority_levels(self) -> List[int]:
        priorities = set()
        for op in self.ops:
            for t in op.inputs + op.outputs:
                for dim in t.spec.dim_specs:
                    priorities.add(dim.priority)
        return sorted(priorities)

    def run_pass(self) -> int:
        self.iteration_count = 0
        sorted_ops = self._get_sorted_ops()
        
        if self.use_priority_iteration:
            return self._run_priority_iteration(sorted_ops)
        else:
            return self._run_simple_iteration(sorted_ops)
    
    def _run_simple_iteration(self, sorted_ops: List[Operation]) -> int:
        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            changed = self._propagate_once(sorted_ops, max_priority=None)
            if not changed:
                return self.iteration_count
        
        raise RuntimeError(
            f"Sharding propagation did not converge after {self.max_iterations} iterations"
        )
    
    def _run_priority_iteration(self, sorted_ops: List[Operation]) -> int:
        priority_levels = self._get_user_priority_levels()
        
        for current_max_priority in priority_levels:
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
        changed = False
        
        for op in sorted_ops:
            in_specs = [t.spec for t in op.inputs]
            out_specs = [t.spec for t in op.outputs]
            
            if propagate_sharding(op.rule, in_specs, out_specs, self.strategy, max_priority):
                changed = True
        
        for edge in self.data_flow_edges:
            rule = edge.to_identity_rule()
            source_specs = [t.spec for t in edge.sources]
            target_specs = [t.spec for t in edge.targets]
            
            if propagate_sharding(rule, source_specs, target_specs, self.strategy, max_priority):
                changed = True
        
        return changed
    
    def get_sharding_summary(self) -> Dict[str, str]:
        summary = {}
        seen = set()
        
        for op in self.ops:
            for t in op.inputs + op.outputs:
                if t.name not in seen:
                    summary[t.name] = str(t.spec)
                    seen.add(t.name)
        return summary
    
    def get_propagation_table(self) -> str:
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

def elementwise_template(rank: int, prefix: str = "d") -> OpShardingRuleTemplate:
    factors = [f"{prefix}{i}" for i in range(rank)]
    mapping = {i: [factors[i]] for i in range(rank)}
    return OpShardingRuleTemplate([mapping, mapping], [mapping])

def unary_template(rank: int, prefix: str = "d") -> OpShardingRuleTemplate:
    factors = [f"{prefix}{i}" for i in range(rank)]
    mapping = {i: [factors[i]] for i in range(rank)}
    return OpShardingRuleTemplate([mapping], [mapping])

def transpose_template(rank: int, perm: List[int]) -> OpShardingRuleTemplate:
    factors = [f"d{i}" for i in range(rank)]
    in_mapping = {i: [factors[i]] for i in range(rank)}
    out_mapping = {i: [factors[perm[i]]] for i in range(rank)}
    return OpShardingRuleTemplate([in_mapping], [out_mapping])

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