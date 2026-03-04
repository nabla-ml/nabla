# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from max.graph import TensorValue

    from ..tensor.api import Tensor
    from .propagation import OpShardingRule
    from .spec import DeviceMesh, ShardingSpec


def get_mesh_from_args(args: tuple[Any, ...]) -> DeviceMesh | None:
    """Extract DeviceMesh from first tensor with sharding spec."""
    from ..tensor import Tensor

    # Fast path: most common case is simple tuple of tensors
    # Check top-level args first without tree traversal
    for a in args:
        if isinstance(a, Tensor) and a.sharding:
            return a.sharding.mesh

    # Only do tree traversal if we haven't found mesh and args might be nested
    for a in args:
        if isinstance(a, (list, tuple, dict)):
            from .. import pytree

            for leaf in pytree.tree_leaves((a,)):
                if isinstance(leaf, Tensor) and leaf.sharding:
                    return leaf.sharding.mesh
    return None


def ensure_specs(args: tuple[Any, ...], mesh: DeviceMesh | None) -> tuple[Any, ...]:
    """Ensure all tensors have explicit sharding specs (replicated if default)."""
    return args


def reshard_inputs(
    args: tuple[Any, ...],
    required_specs: list[ShardingSpec | None],
    mesh: DeviceMesh | None,
) -> tuple[Any, ...]:
    """Pre-operation resharding: align inputs to required propagation specs."""
    if mesh is None or not required_specs:
        return args

    from .. import pytree
    from ..tensor import Tensor

    leaves = [a for a in pytree.tree_leaves(args) if isinstance(a, Tensor)]

    if len(leaves) != len(required_specs):
        return args

    tensor_to_spec = {
        id(t): spec for t, spec in zip(leaves, required_specs, strict=False)
    }

    def reshard_if_needed(x):
        if not isinstance(x, Tensor):
            return x
        required = tensor_to_spec.get(id(x))
        if required is None:
            return x

        current = x.sharding

        if current is None or current.is_fully_replicated():
            if required is not None and not required.is_fully_replicated():
                from ...ops.communication import shard as shard_fn

                return shard_fn(x, mesh, required.dim_specs)
            return x

        from .spec import needs_reshard

        if not needs_reshard(current, required):
            return x

        from ...ops.communication.reshard import reshard_tensor

        return reshard_tensor(x, current, required, mesh)

    return pytree.tree_map(reshard_if_needed, args)


def infer_output_sharding(
    op: Any,
    args: tuple[Any, ...],
    mesh: DeviceMesh | None,
    kwargs: dict[str, Any] | None = None,
) -> tuple[ShardingSpec | None, list[ShardingSpec | None], set[str]]:
    """Infer output/input shardings via factor propagation.

    Returns:
        (output_sharding, input_shardings, needs_allreduce)
    """
    if mesh is None:
        return None, [], False

    if hasattr(op, "infer_sharding_spec"):
        return op.infer_sharding_spec(args, mesh, kwargs)

    from .. import pytree
    from ..tensor import Tensor
    from .propagation import propagate_sharding
    from .spec import DimSpec, ShardingSpec

    leaves = [a for a in pytree.tree_leaves(args) if isinstance(a, Tensor)]

    if not leaves:
        return None, [], False

    def dim_to_int(d):
        """Safely convert Dim object to int."""
        try:
            return int(d)
        except (TypeError, ValueError):
            return 1

    input_specs = []
    input_shapes = []
    for t in leaves:
        spec = t.sharding

        shape = t.physical_global_shape
        if shape is None and (t.sharding is None or t.sharding.is_fully_replicated()):
            shape = t.physical_shape

        if shape is None:
            batch_dims = t.batch_dims
            if batch_dims > 0 and t.batch_shape is not None:
                batch_ints = tuple(dim_to_int(d) for d in t.batch_shape)
                logical_ints = tuple(dim_to_int(d) for d in t.shape)
                phys_shape_tuple = batch_ints + logical_ints
            else:
                phys_shape_tuple = tuple(dim_to_int(d) for d in t.shape)
        else:
            phys_shape_tuple = tuple(dim_to_int(d) for d in shape)

        if spec is None:
            rank = len(phys_shape_tuple)
            spec = ShardingSpec(mesh, [DimSpec([], is_open=True) for _ in range(rank)])

        input_specs.append(spec.clone())
        input_shapes.append(phys_shape_tuple)

    if not any(
        spec.dim_specs and any(d.axes for d in spec.dim_specs) for spec in input_specs
    ):
        input_partial_axes = set()
        for spec in input_specs:
            input_partial_axes.update(spec.partial_sum_axes)

        if not input_partial_axes:
            return None, input_specs, False

        # Only linear/distributive ops may defer reduction on Partial tensors.
        # Non-linear ops (relu, exp, softmax, ...) MUST reduce first because
        # f(P0+P1) ≠ f(P0)+f(P1).
        if not op.allows_partial_passthrough:
            output_rank = op.infer_output_rank(input_shapes, **(kwargs or {}))
            output_spec = ShardingSpec(
                mesh,
                [DimSpec([], is_open=False) for _ in range(output_rank)],
            )
            return output_spec, input_specs, input_partial_axes

        output_rank = op.infer_output_rank(input_shapes, **(kwargs or {}))
        output_spec = ShardingSpec(
            mesh,
            [DimSpec([], is_open=False) for _ in range(output_rank)],
            partial_sum_axes=input_partial_axes,
        )
        return output_spec, input_specs, set()

    try:
        output_rank = op.infer_output_rank(input_shapes, **(kwargs or {}))
    except (NotImplementedError, AttributeError):
        output_rank = len(input_shapes[0]) if input_shapes else 0

    rule = None
    try:
        rule = op.sharding_rule(input_shapes, None, **(kwargs or {}))
    except (NotImplementedError, AttributeError):
        rule = None

    if rule is None:
        for spec in input_specs:
            if (
                any(d.axes for d in spec.dim_specs)
                and len(spec.dim_specs) == output_rank
            ):
                cloned_spec = spec.clone()
                for dim_spec in cloned_spec.dim_specs:
                    dim_spec.is_open = False
                return cloned_spec, input_specs, False
        return None, input_specs, False

    output_spec = ShardingSpec(
        mesh, [DimSpec([], is_open=True) for _ in range(output_rank)]
    )

    # Save dims on multi-input contracting factors before propagation
    # resolves conflicts by clearing them (e.g. matmul k-dim: sharded
    # vs replicated → propagation picks replicated, erasing the info).
    saved_dims = _save_multi_input_contracting_dims(rule, input_specs)

    propagate_sharding(rule, input_specs, [output_spec])

    # Restore dims that propagation cleared — prevents reshard_inputs
    # from inserting all_gather; the op produces Partial output instead.
    restored_axes = _restore_cleared_contracting_dims(input_specs, saved_dims)

    input_partial_axes = set()
    for spec in input_specs:
        if spec:
            input_partial_axes.update(spec.partial_sum_axes)
    output_spec.partial_sum_axes.update(input_partial_axes)

    reduce_axes, ghost_axes = _check_contracting_factors_sharded(
        rule, input_specs, output_spec
    )

    # Restored axes become ghost (Partial output) instead of triggering
    # immediate AllReduce. Non-restored reduce_axes (e.g. reduce_sum on
    # a sharded dim) fire AllReduce normally.
    for ax in restored_axes:
        if ax in reduce_axes:
            reduce_axes.discard(ax)
            ghost_axes.add(ax)

    for ax in ghost_axes:
        sharded_in_dim = False
        for dim in output_spec.dim_specs:
            if ax in dim.axes:
                dim.partial = True
                sharded_in_dim = True

        if not sharded_in_dim:
            output_spec.partial_sum_axes.add(ax)

    used_in_dims = set()
    for dim in output_spec.dim_specs:
        for ax in dim.axes:
            used_in_dims.add(ax)

    for ax in list(output_spec.partial_sum_axes):
        if ax in used_in_dims:
            output_spec.partial_sum_axes.remove(ax)
            for dim in output_spec.dim_specs:
                if ax in dim.axes:
                    dim.partial = True

    for dim in output_spec.dim_specs:
        dim.is_open = False

    return output_spec, input_specs, reduce_axes


def _save_multi_input_contracting_dims(
    rule: OpShardingRule,
    input_specs: list[ShardingSpec],
) -> dict[tuple[int, int], list[str]]:
    """Snapshot sharding on dims that carry multi-input contracting factors.

    A "multi-input" contracting factor appears in 2+ input tensors (e.g. "k"
    in matmul: A[m,k] @ B[k,n]).  ``propagate_sharding`` may clear their
    sharding when resolving conflicts, but we need the original info to
    produce Partial output instead of inserting all_gather.

    Single-input contracting factors (e.g. reduced dims in reduce_sum) are
    left alone — they keep the default all_gather + local reduce behaviour.
    """
    contracting = rule.get_contracting_factors()
    if not contracting:
        return {}

    # Which contracting factors appear in 2+ input tensors?
    multi = set()
    for f in contracting:
        count = sum(
            1
            for m in rule.input_mappings
            if any(f in facs for facs in m.values())
        )
        if count >= 2:
            multi.add(f)
    if not multi:
        return {}

    saved: dict[tuple[int, int], list[str]] = {}
    for t_idx, spec in enumerate(input_specs):
        if t_idx >= len(rule.input_mappings):
            continue
        for dim_idx, factors in rule.input_mappings[t_idx].items():
            if dim_idx < len(spec.dim_specs):
                ds = spec.dim_specs[dim_idx]
                if ds.axes and any(f in multi for f in factors):
                    saved[(t_idx, dim_idx)] = list(ds.axes)
    return saved


def _restore_cleared_contracting_dims(
    input_specs: list[ShardingSpec],
    saved: dict[tuple[int, int], list[str]],
) -> set[str]:
    """Restore dims that ``propagate_sharding`` cleared due to conflicts.

    Returns the set of restored mesh axis names.
    """
    restored: set[str] = set()
    for (t_idx, dim_idx), orig_axes in saved.items():
        dim = input_specs[t_idx].dim_specs[dim_idx]
        if not dim.axes:  # propagation cleared it
            dim.axes = orig_axes
            dim.is_open = False
            restored.update(orig_axes)
    return restored


def _check_contracting_factors_sharded(
    rule: OpShardingRule,
    input_specs: list[ShardingSpec],
    output_spec: ShardingSpec | None,
) -> tuple[set[str], set[str]]:
    """Check if contracting factors need AllReduce (sharded input, not in output)."""
    contracting_factors = rule.get_contracting_factors()
    if not contracting_factors:
        return set(), set()

    reduce_axes = set()
    ghost_axes = set()

    preserved_axes = set()
    if output_spec:
        for ds in output_spec.dim_specs:
            preserved_axes.update(ds.axes)

    axis_partial_map: dict[str, bool] = {}
    contracted_axes = set()

    for input_idx, spec in enumerate(input_specs):
        if not spec or input_idx >= len(rule.input_mappings):
            continue

        mapping = rule.input_mappings[input_idx]

        for dim_idx, current_dim in enumerate(spec.dim_specs):
            factors = mapping.get(dim_idx, [])
            for f in factors:
                if f in contracting_factors:
                    for ax in current_dim.axes:
                        if ax not in preserved_axes:
                            contracted_axes.add(ax)
                            axis_partial_map[ax] = (
                                axis_partial_map.get(ax, False) or current_dim.partial
                            )

    for ax in contracted_axes:
        if axis_partial_map[ax]:
            ghost_axes.add(ax)
        else:
            reduce_axes.add(ax)

    return reduce_axes, ghost_axes


def create_replicated_spec(mesh: DeviceMesh, rank: int) -> ShardingSpec:
    """Create a fully replicated sharding spec."""
    from .spec import DimSpec, ShardingSpec

    return ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)])


def get_shard_args(
    args: Any,
    shard_idx: int,
    per_input_shardings: list[ShardingSpec | None],
    g: Any,
    Tensor: type,
    pytree: Any,
) -> Any:
    """Get per-shard TensorValues, slicing each input according to its sharding."""

    input_idx = [0]

    def extract(x):
        if not isinstance(x, Tensor):
            return x

        _this_sharding = None
        if per_input_shardings and input_idx[0] < len(per_input_shardings):
            _this_sharding = per_input_shardings[input_idx[0]]
        input_idx[0] += 1

        from ..graph.engine import GRAPH

        if x._impl.graph_values_epoch != GRAPH.epoch:
            x._impl._graph_values = []

        x.hydrate()
        vals = x.values

        if shard_idx < len(vals):
            return vals[shard_idx]
        return vals[0] if vals else x.__tensorvalue__()

    return pytree.tree_map(extract, args)


def create_sharded_output(
    results: list[TensorValue],
    sharding: ShardingSpec | None,
    is_traced: bool,
    batch_dims: int,
    mesh: DeviceMesh | None = None,
    physical_shapes: list[tuple[int, ...]] | None = None,
    shard_dtypes: list[Any] | None = None,
    shard_devices: list[Any] | None = None,
) -> Tensor:
    """Build sharded Tensor from per-shard TensorValues."""

    from ..tensor import Tensor

    # Create default replicated sharding if mesh provided but no spec
    if (
        sharding is None
        and mesh is not None
        and (len(results) > 1 or (not results and len(mesh.devices) > 1))
    ):
        from .spec import DimSpec, ShardingSpec

        first_shape = (
            results[0].type.shape
            if results
            else (physical_shapes[0] if physical_shapes else None)
        )
        if first_shape is not None:
            rank = len(first_shape)
            sharding = ShardingSpec(
                mesh, [DimSpec([], is_open=True) for _ in range(rank)]
            )

    output = Tensor._create_unsafe(
        values=results,
        is_traced=is_traced,
        batch_dims=batch_dims,
        physical_shapes=physical_shapes,
        shard_dtypes=shard_dtypes,
        shard_devices=shard_devices,
    )
    output.sharding = sharding

    return output


def execute_on_shards(
    op_fn: Any,
    args: Any,
    kwargs: dict[str, Any],
    mesh: DeviceMesh | None,
    input_shardings: list[ShardingSpec | None] | None = None,
    op: Any | None = None,
) -> list[Any]:
    """Execute op_fn on each shard, handling input slicing and kwarg transformation."""
    from max import graph as g

    from .. import pytree
    from ..tensor import Tensor

    # Pre-fetch input shardings if not provided
    if mesh is not None and input_shardings is None:
        leaves = [a for a in pytree.tree_leaves(args) if isinstance(a, Tensor)]
        input_shardings = [t.sharding for t in leaves]

    # Infer output sharding if needed for keyword transformation
    output_sharding = None
    if op is not None and mesh is not None and hasattr(op, "_transform_shard_kwargs"):
        output_sharding, _, _ = infer_output_sharding(op, args, mesh, kwargs)

    if mesh is None:
        # Local execution (0-th shard/unsharded view)
        shard_args = get_shard_args(args, 0, input_shardings, g, Tensor, pytree)
        local_kwargs = kwargs
        if (
            op is not None
            and output_sharding is not None
            and hasattr(op, "_transform_shard_kwargs")
        ):
            local_kwargs = op._transform_shard_kwargs(
                kwargs, output_sharding, 0, shard_args
            )
        result = op_fn(
            list(shard_args) if not isinstance(shard_args, list) else shard_args,
            local_kwargs,
        )
        # kernel returns list[TensorValue]; for single-output we need the single value for shard_results
        # But execute_on_shards returns list-of-shard-results. Each shard result should be
        # a single TensorValue (single output) or a list/tuple of TensorValue (multi output).
        # Since kernel now always returns list, we unwrap single-element lists for backward compat
        # with the packaging layer.
        if isinstance(result, list) and len(result) == 1:
            return [result[0]]
        return [result]

    results = []
    num_shards = len(mesh.devices)

    for i in range(num_shards):
        # Slice args for this shard
        shard_args = get_shard_args(args, i, input_shardings, g, Tensor, pytree)

        # Transform kwargs if op provides a hook
        local_kwargs = kwargs
        if (
            op is not None
            and output_sharding is not None
            and hasattr(op, "_transform_shard_kwargs")
        ):
            local_kwargs = op._transform_shard_kwargs(kwargs, output_sharding, i, args)

        # Execute with unified kernel signature
        result = op_fn(
            list(shard_args) if not isinstance(shard_args, list) else shard_args,
            local_kwargs,
        )
        if isinstance(result, list) and len(result) == 1:
            results.append(result[0])
        else:
            results.append(result)

    return results
