# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..core.common import pytree
from ..core.graph.tracing import trace
from ..core.tensor.api import Tensor
from ..core.autograd.backward import backward_on_trace

if TYPE_CHECKING:
    from ..graph.tracing import Trace
    from .utils import GradsMap


def grad(
    fun: Callable,
    argnums: int | tuple[int, ...] = 0,
    create_graph: bool = True,
    realize: bool = True,
) -> Callable:
    """Return a function computing the gradient of *fun* (must return a scalar).

    *create_graph* defaults to ``True`` so that gradients are always
    differentiable, enabling higher-order compositions like
    ``jacrev(grad(f))`` or ``jacfwd(grad(f))`` out of the box.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from ..ops.creation import ones_like
        from ..core.tensor.api import Tensor

        t: Trace = trace(fun, *args, **kwargs)
        output = t.outputs

        if isinstance(output, Tensor):
            cotangent: Tensor = ones_like(output)
        else:
            raise TypeError(f"grad: output must be a Tensor, got {type(output)}")

        grads_map: GradsMap = backward_on_trace(
            t, cotangent, create_graph=create_graph
        )
        input_leaves = pytree.tree_leaves(args)
        grad_leaves: list[Tensor | None] = []
        for inp in input_leaves:
            if isinstance(inp, Tensor) and inp in grads_map:
                grad_leaves.append(grads_map[inp])
            elif isinstance(inp, Tensor):
                grad_leaves.append(None)
            else:
                grad_leaves.append(None)

        # Realize eagerly only when create_graph is off (leaf-level usage).
        if realize and not create_graph:
            non_none_grads = [
                g for g in grad_leaves if isinstance(g, Tensor) and not g.real
            ]
            if non_none_grads:
                from ..core.graph.engine import GRAPH

                if len(non_none_grads) > 1:
                    GRAPH.evaluate(non_none_grads[0], *non_none_grads[1:])
                else:
                    GRAPH.evaluate(non_none_grads[0])

        grads_struct = pytree.tree_unflatten(pytree.tree_structure(args), grad_leaves)
        from .utils import select_argnums

        return select_argnums(grads_struct, argnums)

    return wrapper


def value_and_grad(
    fun: Callable,
    argnums: int | tuple[int, ...] = 0,
    create_graph: bool = True,
    realize: bool = True,
) -> Callable:
    """Return a function computing ``(value, grad)`` of *fun*.

    See :func:`grad` for *create_graph* semantics.
    """

    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        from ..ops.creation import ones_like
        from ..core.tensor.api import Tensor

        t: Trace = trace(fun, *args, **kwargs)
        output = t.outputs

        if isinstance(output, Tensor):
            cotangent: Tensor = ones_like(output)
        else:
            raise TypeError(f"grad: output must be a Tensor, got {type(output)}")

        grads_map: GradsMap = backward_on_trace(
            t, cotangent, create_graph=create_graph
        )

        input_leaves = pytree.tree_leaves(args)
        grad_leaves: list[Tensor | None] = []
        for inp in input_leaves:
            if isinstance(inp, Tensor) and inp in grads_map:
                grad_leaves.append(grads_map[inp])
            else:
                grad_leaves.append(None)

        # Realize eagerly only when create_graph is off.
        if realize and not create_graph:
            all_targets: list[Tensor] = (
                [output] if isinstance(output, Tensor) and not output.real else []
            )
            all_targets.extend(
                [g for g in grad_leaves if isinstance(g, Tensor) and not g.real]
            )
            if all_targets:
                from .. import config as nabla_config
                from ..core.graph.engine import GRAPH

                if not nabla_config.EAGER_MAX_GRAPH:
                    if len(all_targets) > 1:
                        GRAPH.evaluate(all_targets[0], *all_targets[1:])
                    else:
                        GRAPH.evaluate(all_targets[0])

        grads_struct = pytree.tree_unflatten(pytree.tree_structure(args), grad_leaves)
        from .utils import select_argnums

        return output, select_argnums(grads_struct, argnums)

    return wrapper
