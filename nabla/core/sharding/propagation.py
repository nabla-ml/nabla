# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass, field
from enum import IntEnum

from .spec import DeviceMesh, DimSpec, ShardingSpec


class OpPriority(IntEnum):
    """Operation-based priority for propagation ordering (lower = higher priority)."""

    PASSTHROUGH = 0
    CONTRACTION = 1
    REDUCTION = 2
    COMMUNICATION = 3


class PropagationStrategy(IntEnum):
    """Conflict resolution strategy (BASIC = no conflicts, AGGRESSIVE = resolve conflicts)."""

    BASIC = 0
    AGGRESSIVE = 1


@dataclass
class FactorSharding:
    """Sharding state for a single factor during propagation.

    Attributes:
        axes: Assigned mesh axes (major-to-minor).
        priority: 0=Strongest, 999=Weakest.
        is_open: If True, can accept more sharding.
        partial: If True, factor holds partial sums.
    """

    axes: list[str] = field(default_factory=list)
    priority: int = 999
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

    def copy(self) -> "FactorSharding":
        return FactorSharding(
            axes=list(self.axes),
            priority=self.priority,
            is_open=self.is_open,
            partial=self.partial,
        )

    def __repr__(self) -> str:
        axes_str = ",".join(self.axes) if self.axes else "∅"
        status = (
            "open"
            if self.is_open
            else ("repl" if self.is_explicit_replication else "closed")
        )
        partial_str = "!" if self.partial else ""
        return f"FactorSharding({axes_str}, p{self.priority}, {status}{partial_str})"


@dataclass
class FactorShardingState:
    """
    Complete factor sharding state for an operation during propagation.
    Holds the sharding state for all factors in an OpShardingRule.
    """

    factors: dict[str, FactorSharding] = field(default_factory=dict)

    def get_or_create(self, factor_name: str) -> FactorSharding:
        if factor_name not in self.factors:
            self.factors[factor_name] = FactorSharding()
        return self.factors[factor_name]

    def get(self, factor_name: str) -> FactorSharding | None:
        return self.factors.get(factor_name)

    def merge(
        self,
        factor_name: str,
        new_axes: list[str],
        new_priority: int,
        new_is_open: bool,
        new_partial: bool,
        mesh: DeviceMesh,
        strategy: PropagationStrategy = None,
    ) -> None:
        """Merge new sharding information with Shardy conflict resolution."""
        if strategy is None:
            strategy = PropagationStrategy.BASIC

        factor = self.get_or_create(factor_name)

        has_new = bool(new_axes)
        has_existing = factor.has_sharding
        new_is_receptive = not has_new and new_is_open
        new_is_explicit_repl = not has_new and not new_is_open

        factor.partial = factor.partial or new_partial

        if new_is_receptive:
            return

        if factor.is_explicit_replication and new_priority >= factor.priority:
            return

        if new_is_explicit_repl and new_priority < factor.priority:
            factor.axes = []
            factor.priority = new_priority
            factor.is_open = False
            return

        if new_is_explicit_repl and new_priority == factor.priority:
            factor.axes = []
            factor.is_open = False
            return

        if has_new and not has_existing and factor.is_open:
            factor.axes = list(new_axes)
            factor.priority = new_priority
            factor.is_open = new_is_open
            return

        if has_new and has_existing:
            if new_priority < factor.priority:

                factor.axes = list(new_axes)
                factor.priority = new_priority
                factor.is_open = new_is_open
            elif new_priority == factor.priority:

                if strategy == PropagationStrategy.AGGRESSIVE:

                    old_par = self._get_parallelism(factor.axes, mesh)
                    new_par = self._get_parallelism(new_axes, mesh)
                    if new_par > old_par:
                        factor.axes = list(new_axes)
                        factor.is_open = new_is_open
                else:
                    # If one is a prefix of the other, take the longer one (alignment)
                    if len(new_axes) > len(factor.axes) and new_axes[:len(factor.axes)] == factor.axes:
                        factor.axes = list(new_axes)
                    elif len(factor.axes) >= len(new_axes) and factor.axes[:len(new_axes)] == new_axes:
                        # factor.axes already contains new_axes as prefix, keep it
                        pass
                    else:
                        # Real conflict, take common prefix
                        common = self._longest_common_prefix(factor.axes, new_axes)
                        factor.axes = common
                        # If we lost axes, it might become open if either was open
                        factor.is_open = factor.is_open or new_is_open
            return

        if has_new and not has_existing:
             # Replication matches anything if it's a prefix, but sharding is more specific.
             # If factor was already replicated (closed or open), and new is sharded.
             if new_priority <= factor.priority:
                 factor.axes = list(new_axes)
                 factor.priority = new_priority
                 factor.is_open = new_is_open
             return

        if has_new and new_priority > factor.priority:
            return

        if has_new:
            factor.axes = list(new_axes)
            factor.priority = new_priority
            factor.is_open = new_is_open

    @staticmethod
    def _longest_common_prefix(list1: list[str], list2: list[str]) -> list[str]:
        common = []
        for x, y in zip(list1, list2, strict=False):
            if x == y:
                common.append(x)
            else:
                break
        return common

    @staticmethod
    def _get_parallelism(axes: list[str], mesh: DeviceMesh) -> int:
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
    """Einsum-like factor mapping for sharding propagation."""

    input_mappings: list[dict[int, list[str]]]
    output_mappings: list[dict[int, list[str]]]
    factor_sizes: dict[str, int]

    def get_all_factors(self) -> set[str]:
        factors = set()
        for mapping in self.input_mappings + self.output_mappings:
            for factor_list in mapping.values():
                factors.update(factor_list)
        return factors

    def get_factor_tensors(self, factor_name: str) -> list[tuple[str, int, int]]:
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

    def get_contracting_factors(self) -> set[str]:
        """Return factors that appear only in inputs (contracting)."""
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
        """Convert to einsum-like string "m k, k n -> m n"."""

        def mapping_to_str(mapping: dict[int, list[str]]) -> str:
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

                    parts.append(f"({' '.join(factors)})")
            return " ".join(parts)

        inputs = ", ".join(mapping_to_str(m) for m in self.input_mappings)
        outputs = ", ".join(mapping_to_str(m) for m in self.output_mappings)
        return f"{inputs} -> {outputs}"


@dataclass
class OpShardingRuleTemplate:
    """Shape-agnostic sharding rule template."""

    input_mappings: list[dict[int, list[str]]]
    output_mappings: list[dict[int, list[str]]]

    def instantiate(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]] | None = None,
    ) -> OpShardingRule:
        """Instantiate template with concrete shapes to infer factor sizes."""
        factor_sizes: dict[str, int] = {}

        pairs = list(zip(self.input_mappings, input_shapes, strict=False))
        if output_shapes:
            pairs.extend(zip(self.output_mappings, output_shapes, strict=False))

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
                    pass
                elif len(unknown_factors) == 1:
                    if dim_size % known_product == 0:
                        factor_sizes[unknown_factors[0]] = dim_size // known_product

        return OpShardingRule(self.input_mappings, self.output_mappings, factor_sizes)

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
        return OpShardingRule(
            self.input_mappings, self.output_mappings, {}
        ).to_einsum_notation()

    @classmethod
    def parse(
        cls, equation: str, input_shapes: list[tuple[int, ...]] | None = None
    ) -> "OpShardingRuleTemplate":
        """Create template from einsum string (e.g. 'mk,kn->mn')."""
        lhs, rhs = equation.split("->")
        input_strs = [s.strip() for s in lhs.split(",")]
        output_strs = [s.strip() for s in rhs.split(",")]

        input_mappings = []
        output_mappings = []

        def parse_factors(
            s: str, shape: tuple[int, ...] | None = None, batch_rank: int = 0
        ) -> dict[int, list[str]]:
            mapping = {}
            parts = s.split()
            idx = 0
            for part in parts:
                if part == "...":
                    if shape is not None:
                        explicit_count = len(parts) - 1
                        batch_rank = len(shape) - explicit_count

                    if batch_rank < 0:
                        raise ValueError(f"Batch rank negative/invalid for spec '{s}'")

                    for b in range(batch_rank):
                        mapping[idx] = [f"b{b}"]
                        idx += 1
                elif part == "1":
                    mapping[idx] = []
                    idx += 1
                else:
                    mapping[idx] = [part]
                    idx += 1
            return mapping

        batch_rank = 0
        for i, s in enumerate(input_strs):
            shape = input_shapes[i] if input_shapes else None

            if "..." in s and shape:
                explicit = len(s.split()) - 1
                br = len(shape) - explicit
                if batch_rank == 0:
                    batch_rank = br
                elif br != batch_rank:
                    batch_rank = max(batch_rank, br)

            input_mappings.append(parse_factors(s, shape, batch_rank))

        for s in output_strs:
            output_mappings.append(parse_factors(s, None, batch_rank))

        return cls(input_mappings, output_mappings)


def _expand_axes_for_factors(
    axes: list[str], factors: list[str], factor_sizes: dict[str, int], mesh: DeviceMesh
) -> list[str]:
    """Expand axes into sub-axes when one axis covers multiple factors."""
    if not axes or not factors:
        return axes

    expanded = []
    curr_ax_idx = 0

    while curr_ax_idx < len(axes) and len(expanded) < len(factors):
        ax = axes[curr_ax_idx]
        ax_size = mesh.get_axis_size(ax)

        remaining_factors = factors[len(expanded) :]
        if not remaining_factors:
            break

        cum_prod = 1
        sub_factors = []
        found_split = False

        for f in remaining_factors:
            f_size = factor_sizes.get(f, 1)
            cum_prod *= f_size
            sub_factors.append((f, f_size))

            if cum_prod == ax_size and len(sub_factors) > 1:

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
    specs: list[ShardingSpec],
    mappings: list[dict[int, list[str]]],
    rule: OpShardingRule,
    mesh: DeviceMesh,
    state: FactorShardingState,
    strategy: PropagationStrategy,
    max_priority: int | None,
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

                if not axes_for_f and not dim_spec.is_open:
                    # Explicit replication: merge it to ensure priority/closed status is seen
                    pass

                state.merge(
                    f,
                    axes_for_f,
                    dim_spec.priority,
                    dim_spec.is_open,
                    dim_spec.partial,
                    mesh,
                    strategy,
                )


def _should_update_dim(
    current: DimSpec,
    proposed_axes: list[str],
    proposed_priority: int,
) -> bool:
    """Determine if a dimension should be updated based on conflicts."""

    if proposed_priority < current.priority:
        return True

    if proposed_priority == current.priority:
        # If current is empty but proposed has sharding, adopt it (upgrade from replication)
        if not current.axes and proposed_axes:
            return True

        if current.is_open:
            if not current.axes and proposed_axes:
                return True
            if len(proposed_axes) > len(current.axes):
                if proposed_axes[: len(current.axes)] == current.axes:
                    return True

        if current.axes and (
            not proposed_axes or len(proposed_axes) < len(current.axes)
        ):
            if not proposed_axes or current.axes[: len(proposed_axes)] == proposed_axes:
                return True

    if proposed_priority > current.priority:
        if current.is_open and not current.axes and proposed_axes:
            return True

    return False


def _update_from_factors(
    specs: list[ShardingSpec],
    mappings: list[dict[int, list[str]]],
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

        used_axes_in_tensor: set[str] = set()

        for dim_idx, current_dim in enumerate(spec.dim_specs):
            factors = mapping.get(dim_idx, [])

            if not factors:

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
                    valid_axes = [
                        ax for ax in f_state.axes if ax not in used_axes_in_tensor
                    ]

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

                used_axes_in_tensor.update(proposed_axes)
                new_dim_specs.append(
                    DimSpec(
                        axes=proposed_axes,
                        is_open=current_dim.is_open,
                        priority=proposed_prio,
                        partial=proposed_partial,
                    )
                )
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
    input_specs: list[ShardingSpec],
    output_specs: list[ShardingSpec],
    strategy: PropagationStrategy = PropagationStrategy.BASIC,
    max_priority: int | None = None,
) -> bool:
    """Propagate shardings between inputs/outputs. Returns True if changed."""
    if not input_specs and not output_specs:
        return False

    mesh = input_specs[0].mesh if input_specs else output_specs[0].mesh

    state = FactorShardingState()
    _collect_to_factors(
        input_specs, rule.input_mappings, rule, mesh, state, strategy, max_priority
    )
    _collect_to_factors(
        output_specs, rule.output_mappings, rule, mesh, state, strategy, max_priority
    )

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
    """Run hierarchical sharding propagation (User -> Op Priority -> Strategy)."""
    total_changes = 0

    for user_priority in range(max_user_priority + 1):

        for op_priority in [
            OpPriority.PASSTHROUGH,
            OpPriority.CONTRACTION,
            OpPriority.REDUCTION,
            OpPriority.COMMUNICATION,
        ]:

            for strategy in [PropagationStrategy.AGGRESSIVE, PropagationStrategy.BASIC]:

                iteration = 0
                while iteration < max_iterations:
                    changed_this_iter = False

                    for op, rule, input_specs, output_specs in operations_with_rules:
                        op_prio = getattr(op, "op_priority", OpPriority.CONTRACTION)
                        if op_prio != op_priority:
                            continue

                        changed = propagate_sharding(
                            rule,
                            input_specs,
                            output_specs,
                            strategy=strategy,
                            max_priority=user_priority,
                        )

                        if changed:
                            changed_this_iter = True
                            total_changes += 1

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
