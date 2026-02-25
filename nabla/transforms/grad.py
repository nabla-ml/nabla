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
    """Return a function that computes the gradient of *fun*.

    *fun* must return a scalar tensor. The returned callable accepts the same
    arguments as *fun* and returns the gradient with respect to the inputs
    specified by *argnums*.

    Args:
        fun: Scalar-valued function to differentiate.
        argnums: Index or tuple of indices of positional arguments to
            differentiate with respect to. Default: ``0`` (first argument).
        create_graph: If ``True`` (default), the gradient is itself
            differentiable, enabling higher-order derivatives such as
            ``jacrev(grad(f))``.
        realize: If ``True`` and *create_graph* is ``False``, eagerly
            materialise the gradient tensors before returning.

    Returns:
        A callable with the same signature as *fun* that returns the gradient
        (or a tuple of gradients when *argnums* is a tuple).

    Example::

        f = lambda x: nabla.reduce_sum(x ** 2)
        df = nabla.grad(f)
        gradient = df(x)  # shape == x.shape
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
    """Return a function that evaluates *fun* and its gradient simultaneously.

    More efficient than calling *fun* and :func:`grad` separately because
    the forward pass is shared.

    Args:
        fun: Scalar-valued function to differentiate.
        argnums: Index or tuple of indices of positional arguments to
            differentiate with respect to. Default: ``0``.
        create_graph: If ``True`` (default), the gradient is differentiable.
        realize: If ``True`` and *create_graph* is ``False``, eagerly
            materialise outputs before returning.

    Returns:
        A callable with the same signature as *fun* that returns
        ``(value, gradient)`` where *value* is the scalar output of *fun*
        and *gradient* is its gradient.
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
