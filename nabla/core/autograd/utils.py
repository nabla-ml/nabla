# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Core backpropagation utilities for Trace-based automatic differentiation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..graph.tracing import Trace, OpNode
    from ..tensor.impl import TensorImpl

from ..tensor.api import Tensor


def _unwrap_single(tree: Any) -> Any:
    """Unwrap a single-leaf tree if it's a list/tuple of size 1."""
    if isinstance(tree, (list, tuple)) and len(tree) == 1:
        return tree[0]
    return tree


def _reduce_to_shape(cot_tensor: Tensor, target_shape: tuple[int, ...]) -> Tensor:
    """Reduce cotangent tensor to match target primal shape (handling broadcasting)."""
    cot_shape = tuple(int(d) for d in cot_tensor.shape)

    # 1. Reduce leading added dimensions (rank mismatch)
    if len(cot_shape) > len(target_shape):
        diff = len(cot_shape) - len(target_shape)
        from ...ops.reduction import reduce_sum

        cot_tensor = reduce_sum(cot_tensor, axis=list(range(diff)))
        cot_shape = tuple(int(d) for d in cot_tensor.shape)

    # 2. Reduce broadcasted internal dimensions (size 1 in target, size > 1 in cot)
    reduce_axes = [
        i
        for i, (c_d, a_d) in enumerate(zip(cot_shape, target_shape))
        if a_d == 1 and c_d > 1
    ]
    if reduce_axes:
        from ...ops.reduction import reduce_sum

        cot_tensor = reduce_sum(cot_tensor, axis=reduce_axes, keepdims=True)

    return cot_tensor


def _accumulate_cotangent(
    cotangent_map: dict[int, TensorImpl],
    target_impl: TensorImpl,
    cot_tensor: Tensor,
):
    """Accumulates a cotangent tensor into the global cotangent map for a specific target.

    Handles sharding resolution (partial sums, resharding) and addition.
    """
    from ...core.sharding.spec import needs_reshard
    from ...ops.binary import add
    from ...ops.communication import all_reduce, reshard

    target_id = id(target_impl)

    # 1. Resolve Partial Sums if target is not sharded or doesn't share partial axes
    if cot_tensor.sharding and cot_tensor.sharding.partial_sum_axes:
        target_partials = (
            target_impl.sharding.partial_sum_axes if target_impl.sharding else set()
        )
        if not target_partials:
            cot_tensor = all_reduce(
                cot_tensor, reduce_axes=list(cot_tensor.sharding.partial_sum_axes)
            )

    # 2. Ensure sharding matches target (crucial for DP/PP boundaries)
    if target_impl.sharding and needs_reshard(
        cot_tensor.sharding, target_impl.sharding
    ):
        cot_tensor = reshard(
            cot_tensor,
            target_impl.sharding.mesh,
            target_impl.sharding.dim_specs,
            replicated_axes=target_impl.sharding.replicated_axes,
        )

    # 3. Add to existing or initialize
    if target_id in cotangent_map:
        existing = Tensor(impl=cotangent_map[target_id])
        accumulated = add(existing, cot_tensor)
        cotangent_map[target_id] = accumulated._impl
    else:
        cotangent_map[target_id] = cot_tensor._impl


class BackwardEngine:
    """Stateful engine for backpropagation on a captured Trace."""

    def __init__(self, trace: Trace, cotangents: Any, create_graph: bool = False):
        from ..common import pytree

        self.trace = trace
        self.create_graph = create_graph
        self.cotangent_map: dict[int, TensorImpl] = {}
        self._original_flags: dict[int, bool] = {}

        # Initialize cotangent map from trace outputs
        output_leaves = [
            t._impl for t in pytree.tree_leaves(trace.outputs) if isinstance(t, Tensor)
        ]
        cot_leaves = [
            t._impl for t in pytree.tree_leaves(cotangents) if isinstance(t, Tensor)
        ]

        if len(cot_leaves) != len(output_leaves):
            raise ValueError(
                f"Number of cotangents ({len(cot_leaves)}) must match "
                f"number of outputs ({len(output_leaves)})"
            )

        for out_impl, cot_impl in zip(output_leaves, cot_leaves, strict=True):
            self.cotangent_map[id(out_impl)] = cot_impl

    def _set_trace_state(self, tree: Any):
        """Suppress tracing for backward ops if create_graph=False."""
        from ..common import pytree

        for x in pytree.tree_leaves(tree):
            if isinstance(x, Tensor):
                self._original_flags[id(x)] = x.is_traced
                if not self.create_graph:
                    x.is_traced = False

    def _restore_trace_state(self, tree: Any):
        """Restore original tracing flags."""
        from ..common import pytree

        for x in pytree.tree_leaves(tree):
            if isinstance(x, Tensor) and id(x) in self._original_flags:
                x.is_traced = self._original_flags[id(x)]

    def run(self) -> dict[Tensor, Tensor]:
        """Execute backward pass."""
        for node in reversed(self.trace.nodes):
            self._process_node(node)
        return self._finalize()

    def _process_node(self, node: OpNode):
        from ..common import pytree


        from ..tensor.impl import TensorImpl

        alive_outputs = node.get_alive_outputs()
        op = node.op

        # 1. Skip if op has no VJP or no output has a cotangent
        if not hasattr(op, "vjp_rule"):
            return

        if not any(
            o is not None and id(o) in self.cotangent_map for o in alive_outputs
        ):
            return

        # 2. Prepare structural arguments for VJP rule
        def wrap(x):
            if isinstance(x, (Tensor, TensorImpl)):
                impl = x._impl if isinstance(x, Tensor) else x
                return Tensor(impl=impl)
            return x

        vjp_primals = pytree.tree_map(wrap, node.op_args)

        output_tensors = [
            Tensor(impl=o) if o is not None else None for o in alive_outputs
        ]
        vjp_outputs = pytree.tree_unflatten(node.tree_def, output_tensors)

        cot_tensors = []
        for o in alive_outputs:
            if o is not None:
                if id(o) in self.cotangent_map:
                    cot_tensors.append(Tensor(impl=self.cotangent_map[id(o)]))
                else:
                    from ...ops.creation import zeros_like

                    cot_tensors.append(zeros_like(Tensor(impl=o)))
            else:
                cot_tensors.append(None)
        vjp_cotangents = pytree.tree_unflatten(node.tree_def, cot_tensors)

        # 3. Invoke VJP Rule
        unwrapped_primals = _unwrap_single(vjp_primals)
        unwrapped_outputs = _unwrap_single(vjp_outputs)
        unwrapped_cotangents = _unwrap_single(vjp_cotangents)

        vjp_inputs = [unwrapped_primals, unwrapped_outputs, unwrapped_cotangents]
        self._set_trace_state(vjp_inputs)

        try:
            input_cotangents = op.vjp_rule(
                unwrapped_primals, unwrapped_cotangents, unwrapped_outputs
            )
        except Exception as e:
            self._restore_trace_state(vjp_inputs)
            op_name = getattr(op, "name", str(op))
            raise RuntimeError(f"VJP rule failed for operation '{op_name}': {e}") from e
        finally:
            self._restore_trace_state(vjp_inputs)

        # 4. Align and Accumulate
        arg_leaves, _ = pytree.tree_flatten_full(node.op_args)
        cot_leaves, _ = pytree.tree_flatten_full(input_cotangents)

        # Workaround for single-arg ops that return unwrapped cotangents or (None, cot)
        if len(arg_leaves) == 1 and len(cot_leaves) > 1 and cot_leaves[0] is None:
            cot_leaves = [c for c in cot_leaves if c is not None]

        if len(cot_leaves) != len(arg_leaves):
            if len(arg_leaves) == 1 and not isinstance(
                input_cotangents, (list, tuple, dict)
            ):
                cot_leaves = [input_cotangents]

        for arg_impl, cot_result in zip(arg_leaves, cot_leaves, strict=False):
            if arg_impl is None or cot_result is None:
                continue

            if not isinstance(arg_impl, (Tensor, TensorImpl)):
                continue

            arg_impl_real = arg_impl._impl if isinstance(arg_impl, Tensor) else arg_impl
            cot_tensor = (
                Tensor(impl=cot_result)
                if isinstance(cot_result, TensorImpl)
                else cot_result
            )
            if not isinstance(cot_tensor, Tensor):
                continue

            # Shape alignment (Broadcasting)
            target_shape = tuple(int(d) for d in Tensor(impl=arg_impl_real).shape)
            cot_tensor = _reduce_to_shape(cot_tensor, target_shape)

            # Sharding and Addition
            _accumulate_cotangent(self.cotangent_map, arg_impl_real, cot_tensor)

    def _finalize(self) -> dict[Tensor, Tensor]:
        """Convert accumulated cotangents to input gradients."""
        from ..common import pytree

        gradients = {}
        input_leaves = [
            t for t in pytree.tree_leaves(self.trace.inputs) if isinstance(t, Tensor)
        ]

        for inp in input_leaves:
            inp_id = id(inp._impl)
            if inp_id in self.cotangent_map:
                cot_impl = self.cotangent_map[inp_id]
                grad = Tensor(impl=cot_impl)

                # Double check for un-reduced partials at inputs
                if grad.sharding and grad.sharding.partial_sum_axes:
                    from ...ops.communication import all_reduce

                    grad = all_reduce(
                        grad, reduce_axes=list(grad.sharding.partial_sum_axes)
                    )

                # Ensure it matches input sharding (for Pipeline/ZeRO)
                from ...core.sharding.spec import needs_reshard

                if inp.sharding and needs_reshard(grad.sharding, inp.sharding):
                    from ...ops.communication import reshard

                    grad = reshard(
                        grad,
                        inp.sharding.mesh,
                        inp.sharding.dim_specs,
                        replicated_axes=inp.sharding.replicated_axes,
                    )
                gradients[inp] = grad
            else:
                from ...ops.creation import zeros_like

                gradients[inp] = zeros_like(inp)

        return gradients


def backward_on_trace(
    trace: Trace,
    cotangents: Any,
    *,
    create_graph: bool = False,
    checkpoint_policy: str = "none",
) -> dict[Tensor, Tensor]:
    """Pure-function backpropagation on a Trace."""

    if not trace._computed:
        trace.compute()

    # In Eager MAX Graph mode, we MUST rehydrate the graph values because
    # VJP operations will immediately try to build their graph nodes, requiring
    # all inputs (primals) to have valid graph values in the CURRENT epoch.
    from ...config import EAGER_MAX_GRAPH
    if EAGER_MAX_GRAPH:
        trace.refresh_graph_values()

    engine = BackwardEngine(trace, cotangents, create_graph=create_graph)
    return engine.run()


__all__ = ["backward_on_trace"]
