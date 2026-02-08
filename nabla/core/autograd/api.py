# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .utils import backward_on_trace
from ..graph.tracing import trace
from ..common import pytree
from ..tensor.api import Tensor
from ...transforms.utils import select_argnums

if TYPE_CHECKING:
    from .utils import GradsMap
    from ..graph.tracing import Trace


def grad(
    fun: Callable,
    argnums: int | tuple[int, ...] = 0,
    create_graph: bool = False,
    realize: bool = True,
) -> Callable:
    """Return a function computing the gradient of *fun* (must return a scalar)."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Trace the function execution
        t: "Trace" = trace(fun, *args, **kwargs)

        output = t.outputs

        from ...ops.creation import ones_like
        from ..tensor.api import Tensor

        # If output is a single Tensor, create ones_like
        if isinstance(output, Tensor):
            cotangent: Tensor = ones_like(output)
        else:
            raise TypeError(f"grad: output must be a Tensor, got {type(output)}")

        grads_map: "GradsMap" = backward_on_trace(t, cotangent, create_graph=create_graph)
        input_leaves = pytree.tree_leaves(args)
        grad_leaves: list[Tensor | None] = []
        for inp in input_leaves:
            if isinstance(inp, Tensor) and inp in grads_map:
                grad_leaves.append(grads_map[inp])
            elif isinstance(inp, Tensor):
                grad_leaves.append(None)  # Or zeros?
            else:
                grad_leaves.append(None)

        # Group and evaluate all gradients at once to optimize compilation
        if realize:
            non_none_grads = [
                g for g in grad_leaves if isinstance(g, Tensor) and not g.real
            ]
            if non_none_grads:
                from ..graph.engine import GRAPH

                if len(non_none_grads) > 1:
                    GRAPH.evaluate(non_none_grads[0], *non_none_grads[1:])
                else:
                    GRAPH.evaluate(non_none_grads[0])

        # Unflatten and select by argnums
        grads_struct = pytree.tree_unflatten(pytree.tree_structure(args), grad_leaves)
        return select_argnums(grads_struct, argnums)

    return wrapper


def value_and_grad(
    fun: Callable,
    argnums: int | tuple[int, ...] = 0,
    create_graph: bool = False,
    realize: bool = True,
) -> Callable:
    """Return a function computing ``(value, grad)`` of *fun*."""

    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        t: "Trace" = trace(fun, *args, **kwargs)
        output = t.outputs

        from ...ops.creation import ones_like
        from ..tensor.api import Tensor

        if isinstance(output, Tensor):
            cotangent: Tensor = ones_like(output)
        else:
            raise TypeError(f"grad: output must be a Tensor, got {type(output)}")

        grads_map: "GradsMap" = backward_on_trace(t, cotangent, create_graph=create_graph)

        input_leaves = pytree.tree_leaves(args)
        grad_leaves: list[Tensor | None] = []
        for inp in input_leaves:
            if isinstance(inp, Tensor) and inp in grads_map:
                grad_leaves.append(grads_map[inp])
            else:
                grad_leaves.append(None)

        # Group and evaluate everything at once: primal output + all gradients
        if realize:
            all_targets: list[Tensor] = (
                [output] if isinstance(output, Tensor) and not output.real else []
            )
            all_targets.extend(
                [g for g in grad_leaves if isinstance(g, Tensor) and not g.real]
            )

            if all_targets:
                from ... import config as nabla_config
                from ..graph.engine import GRAPH

                # In EAGER_MAX_GRAPH mode (compile tracing), ops have already built
                # their graph values. Calling evaluate() would reset the graph.
                if not nabla_config.EAGER_MAX_GRAPH:
                    if len(all_targets) > 1:
                        GRAPH.evaluate(all_targets[0], *all_targets[1:])
                    else:
                        GRAPH.evaluate(all_targets[0])

        grads_struct = pytree.tree_unflatten(pytree.tree_structure(args), grad_leaves)
        return output, select_argnums(grads_struct, argnums)

    return wrapper
