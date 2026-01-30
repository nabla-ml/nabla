# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Execution utilities for operations.

This module contains purely functional helpers used during operation execution,
separated from the core Operation base class to keep base.py clean and focused
on the abstract interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.sharding.spec import DeviceMesh


def apply_auto_reduction(
    op: Any, output: Any, mesh: DeviceMesh, reduce_axes: set[str]
) -> Any:
    """Apply automatic collective reductions on sharded tensors.

    This is called during operation execution when the sharding rule indicates
    that certain mesh axes need to be reduced across (e.g., for SPMD matmul).

    Args:
        op: The operation instance (needed for collective_reduce_type)
        output: Output tensor(s) from the operation
        mesh: Device mesh for sharding
        reduce_axes: Set of mesh axis names to reduce across

    Returns:
        Output with reductions applied
    """
    from ..core import GRAPH, Tensor, pytree
    from ..core.sharding import spmd
    from ..core.sharding.spec import DimSpec, ShardingSpec
    from .communication import all_reduce_op

    def apply_grouped_all_reduce(t):
        if not isinstance(t, Tensor):
            return t

        t.hydrate()
        if not t._values:
            return t

        with GRAPH.graph:
            reduced_values = all_reduce_op.simulate_grouped_execution(
                t.values, mesh, reduce_axes, reduce_op=op.collective_reduce_type
            )

        current_spec = t.sharding
        if current_spec:
            new_dim_specs = []
            for ds in current_spec.dim_specs:
                new_axes = sorted(list(set(ds.axes) - reduce_axes))
                new_dim_specs.append(DimSpec(new_axes))
            new_spec = ShardingSpec(mesh, new_dim_specs)
        else:
            rank = len(t.shape)
            new_spec = ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)])

        reduced_tensor = spmd.create_sharded_output(
            reduced_values, new_spec, t.traced, t.batch_dims, mesh
        )

        trace_kwargs = {"mesh": mesh, "reduce_axes": list(reduce_axes)}
        all_reduce_op._setup_output_refs(
            reduced_tensor, (t,), trace_kwargs
        )

        return reduced_tensor

    return pytree.tree_map(apply_grouped_all_reduce, output)


def apply_jvp(op: Any, args: tuple, output: Any) -> None:
    """Apply JVP (forward-mode autodiff) by propagating tangents through the operation.

    This is called during operation execution when any input has tangents attached.
    It computes the JVP using the operation's jvp_rule and attaches the result
    tangent to the output.

    Args:
        op: The operation instance (needed for jvp_rule)
        args: Input arguments (may have tangents attached)
        output: Output tensor(s) to attach tangents to
    """
    from ..core import Tensor, pytree

    tangents = pytree.tree_map(
        lambda x: (
            Tensor(impl=x.tangent) if isinstance(x, Tensor) and x.tangent else None
        ),
        args,
    )
    output_tangent = op.jvp_rule(args, tangents, output)
    if output_tangent is not None:
        pytree.tree_map(
            lambda o, t: (
                setattr(o._impl, "tangent", t._impl)
                if isinstance(o, Tensor) and isinstance(t, Tensor)
                else None
            ),
            output,
            output_tangent,
        )
