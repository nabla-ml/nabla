# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Sharding propagation: per-op sharding inference from einsum-like rules.

The key abstractions are:

- ``OpShardingRuleTemplate``: a declarative, einsum-like specification of how
  an op's dimensions relate (e.g. ``"... m k, ... k n -> ... m n"``).  Ops
  define one via their ``sharding_rule()`` method.

- ``OpShardingRule``: a template instantiated with concrete shapes, giving
  factor sizes.

- ``infer_from_rule()``: given input ``ShardingSpec``s and a rule, produces
  the output ``ShardingSpec`` (including ``partial_sum_axes``) and the set of
  axes that need AllReduce.  This is the single entry point used by
  ``spmd.infer_output_sharding``.

The factor representation is kept because it is the natural input for future
graph-level auto-sharding (ALPA-style strategy enumeration / ILP).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .spec import DimSpec, ShardingSpec

if TYPE_CHECKING:
    from .spec import DeviceMesh


# ---------------------------------------------------------------------------
#  OpShardingRule  /  OpShardingRuleTemplate  (kept unchanged)
# ---------------------------------------------------------------------------


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
                    if f not in factor_sizes:
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

                if len(unknown_factors) == 1 and dim_size % known_product == 0:
                    factor_sizes[unknown_factors[0]] = dim_size // known_product

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


# ---------------------------------------------------------------------------
#  infer_from_rule  —  the simple, eager-mode sharding inference
# ---------------------------------------------------------------------------


def infer_from_rule(
    rule: OpShardingRule,
    input_specs: list[ShardingSpec],
    mesh: "DeviceMesh",
) -> tuple[ShardingSpec, set[str]]:
    """Infer output sharding from input shardings and an op's factor rule.

    This is the single entry point replacing the old ``propagate_sharding``
    + ``ghost_axes`` + ``save/restore`` machinery.  It answers a simple
    question: given input shardings and a factor rule, what is the output
    sharding (including ``partial_sum_axes``) and which axes need AllReduce?

    Algorithm:
        1. Build a factor→axes map from all input dimensions.
        2. For each output dimension, look up its factors and copy across the
           sharding axes.
        3. Identify *contracting* factors (in inputs but NOT in any output).
           If a contracting factor is sharded, the corresponding axis becomes
           a candidate for ``partial_sum_axes`` (deferred reduction).
        4. Axis conflicts (same axis on two different output dims) are resolved
           by giving it to the first dim that claims it.

    Returns:
        (output_spec, contraction_partial_axes)
        - ``output_spec`` has dimensional sharding + ``partial_sum_axes`` from
          contracting factors.
        - ``contraction_partial_axes`` is the set of axes that became partial
          due to contracting over a sharded dimension.
    """
    # --- Step 1: collect factor → axes from inputs --------------------------
    factor_axes: dict[str, list[str]] = {}

    for t_idx, spec in enumerate(input_specs):
        if t_idx >= len(rule.input_mappings):
            continue
        mapping = rule.input_mappings[t_idx]
        for dim_idx, factors in mapping.items():
            if dim_idx >= len(spec.dim_specs):
                continue
            dim_spec = spec.dim_specs[dim_idx]
            if not dim_spec.axes:
                continue
            for f in factors:
                if f not in factor_axes:
                    factor_axes[f] = list(dim_spec.axes)
                else:
                    # Conflict: two inputs disagree on a factor's sharding.
                    # Keep the existing one (first-come wins — simple heuristic).
                    pass

    # --- Step 2: build output dim specs from factors ------------------------
    contracting = rule.get_contracting_factors()
    output_mapping = rule.output_mappings[0] if rule.output_mappings else {}
    output_rank = max(output_mapping.keys(), default=-1) + 1

    used_axes: set[str] = set()
    output_dim_specs: list[DimSpec] = []

    for dim_idx in range(output_rank):
        factors = output_mapping.get(dim_idx, [])
        dim_axes: list[str] = []
        for f in factors:
            for ax in factor_axes.get(f, []):
                if ax not in used_axes:
                    dim_axes.append(ax)
                    used_axes.add(ax)
        output_dim_specs.append(DimSpec(axes=dim_axes, is_open=False))

    # --- Step 3: identify contraction-produced partial axes -----------------
    contraction_partial_axes: set[str] = set()
    for f in contracting:
        for ax in factor_axes.get(f, []):
            contraction_partial_axes.add(ax)

    # --- Step 4: build output spec ------------------------------------------
    output_spec = ShardingSpec(
        mesh,
        output_dim_specs,
        partial_sum_axes=set(contraction_partial_axes),
    )

    return output_spec, contraction_partial_axes


# ---------------------------------------------------------------------------
#  Backward-compat aliases (kept so __init__.py and other imports don't break)
# ---------------------------------------------------------------------------

# Legacy: kept as no-op stubs so existing imports don't fail at import time.
# These should be removed once downstream code is fully migrated.
def propagate_sharding(rule, input_specs, output_specs, **_kw):
    """Legacy stub — use ``infer_from_rule`` instead."""
    if not input_specs:
        return False
    mesh = input_specs[0].mesh
    out_spec, _ = infer_from_rule(rule, input_specs, mesh)
    if output_specs:
        target = output_specs[0]
        target.dim_specs = out_spec.dim_specs
        target.partial_sum_axes = out_spec.partial_sum_axes
    return True
