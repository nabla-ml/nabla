# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Core forward-mode AD utilities for Trace-based automatic differentiation.

Mirrors backward_on_trace / BackwardEngine but walks the trace FORWARD,
propagating tangent vectors through jvp_rule instead of cotangents through
vjp_rule.

Design:
  1. Trace the function (capture OpNode graph) — already done by caller.
  2. Rehydrate graph values if EAGER_MAX_GRAPH (same as backward).
  3. Walk nodes in topological (forward) order.
  4. For each node, look up tangents for its inputs, call op.jvp_rule,
     and store tangents for its outputs.
  5. Return tangent map keyed by output TensorImpl.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..graph.tracing import OpNode, Trace
    from ..tensor.impl import TensorImpl

from ..tensor.api import Tensor

TangentMap = dict["TensorImpl", "TensorImpl"]


class ForwardEngine:
    """Stateful engine for forward-mode AD on a captured Trace.

    Mirrors BackwardEngine but processes nodes in forward (topological) order.
    """

    trace: Trace
    create_graph: bool
    tangent_map: TangentMap
    _original_flags: dict[int, bool]

    def __init__(
        self,
        trace: Trace,
        tangents: Any,
        *,
        create_graph: bool = True,
    ) -> None:
        from ..common import pytree

        self.trace = trace
        self.create_graph = create_graph
        self.tangent_map: TangentMap = {}
        self._original_flags: dict[int, bool] = {}

        # Initialize tangent map from trace inputs.
        input_leaves = [
            t._impl for t in pytree.tree_leaves(trace.inputs) if isinstance(t, Tensor)
        ]
        tangent_leaves = [
            t._impl for t in pytree.tree_leaves(tangents) if isinstance(t, Tensor)
        ]

        if len(tangent_leaves) != len(input_leaves):
            raise ValueError(
                f"Number of tangents ({len(tangent_leaves)}) must match "
                f"number of inputs ({len(input_leaves)})"
            )

        for inp_impl, tan_impl in zip(input_leaves, tangent_leaves, strict=True):
            self.tangent_map[inp_impl] = tan_impl

    def _set_trace_state(self, tree: Any) -> None:
        """Suppress tracing for forward-AD ops if create_graph=False."""
        from ..common import pytree

        for x in pytree.tree_leaves(tree):
            if isinstance(x, Tensor):
                self._original_flags[id(x)] = x.is_traced
                if not self.create_graph:
                    x.is_traced = False

    def _restore_trace_state(self, tree: Any) -> None:
        """Restore original tracing flags."""
        from ..common import pytree

        for x in pytree.tree_leaves(tree):
            if isinstance(x, Tensor) and id(x) in self._original_flags:
                x.is_traced = self._original_flags[id(x)]

    def run(self) -> TangentMap:
        """Execute forward-mode AD pass (topological order)."""
        for node in self.trace.nodes:
            self._process_node(node)
        return self.tangent_map

    def _process_node(self, node: OpNode) -> None:
        """Apply jvp_rule for a single OpNode."""
        from ..tensor.impl import TensorImpl

        op = node.op

        # 1. Build primals and tangents from the node's stored arguments.
        def wrap(x: Any) -> Any:
            if isinstance(x, (Tensor, TensorImpl)):
                impl = x._impl if isinstance(x, Tensor) else x
                return Tensor(impl=impl)
            return x

        jvp_primals: list[Any] = [wrap(a) for a in node.op_args]

        # Build tangent list: look up each primal in tangent_map.
        # Track whether *any* input carries a non-zero tangent.
        # Use None as placeholder for zero tangents — resolved to zeros_like
        # ONLY after the has_active_tangent check, so that skipped nodes
        # never create orphan zero graph ops (which can confuse the MAX
        # compiler's MOToMGP pass).
        jvp_tangents: list[Any] = []
        has_active_tangent = False
        for a in node.op_args:
            if isinstance(a, (Tensor, TensorImpl)):
                impl = a._impl if isinstance(a, Tensor) else a
                if impl in self.tangent_map:
                    jvp_tangents.append(Tensor(impl=self.tangent_map[impl]))
                    has_active_tangent = True
                else:
                    jvp_tangents.append(None)  # placeholder
            else:
                jvp_tangents.append(None)

        # Skip nodes with no tensor inputs that carry tangents (creation ops,
        # constants, etc.)  Their outputs have zero tangent by construction —
        # the extraction code will produce zeros_like for missing entries.
        if not has_active_tangent:
            return

        # Resolve None tangent placeholders → zeros_like (centrally, so
        # individual JVP rules never see None tangents).
        from ...ops.creation import zeros_like

        jvp_tangents = [
            t if t is not None else zeros_like(p)
            for t, p in zip(jvp_tangents, jvp_primals)
        ]

        # 2. Build output list from alive outputs.
        alive_outputs = node.get_alive_outputs()
        jvp_outputs: list[Tensor | None] = [
            Tensor(impl=o) if o is not None else None for o in alive_outputs
        ]

        # 3. Get original (logical, unadapted) kwargs — same as VJP.
        jvp_kwargs = node.op_kwargs or {}

        # 4. Invoke jvp_rule.
        # Set trace state on all involved tensors to control graph building.
        all_tensors = [jvp_primals, jvp_tangents, jvp_outputs]
        self._set_trace_state(all_tensors)

        try:
            output_tangents = op.jvp_rule(
                jvp_primals, jvp_tangents, jvp_outputs, jvp_kwargs
            )
        except NotImplementedError:
            # Op does not implement jvp_rule — treat output tangents as zero.
            self._restore_trace_state(all_tensors)
            return
        except Exception as e:
            self._restore_trace_state(all_tensors)
            op_name = getattr(op, "name", str(op))
            raise RuntimeError(
                f"JVP rule failed for operation '{op_name}': {e}"
            ) from e
        finally:
            self._restore_trace_state(all_tensors)

        # 5. Store output tangents in the tangent map.
        if output_tangents is not None:
            for out_impl, tan in zip(alive_outputs, output_tangents, strict=False):
                if out_impl is not None and tan is not None:
                    tan_impl = tan._impl if isinstance(tan, Tensor) else tan
                    if isinstance(tan_impl, TensorImpl):
                        self.tangent_map[out_impl] = tan_impl


def forward_on_trace(
    trace: Trace,
    tangents: Any,
    *,
    create_graph: bool = True,
) -> TangentMap:
    """Pure-function forward-mode AD on a Trace.

    Analogous to backward_on_trace but propagates tangents forward.

    Args:
        trace: Captured computation trace.
        tangents: Tangent vectors for trace inputs (same pytree structure).
        create_graph: If True, tangent ops are traced for higher-order AD.

    Returns:
        TangentMap: dict mapping output TensorImpl → tangent TensorImpl.
    """
    if not trace._computed:
        trace.compute()

    # In Eager MAX Graph mode, rehydrate graph values so jvp_rule ops
    # can build their graph nodes with valid inputs in the current epoch.
    from ...config import EAGER_MAX_GRAPH

    if EAGER_MAX_GRAPH:
        trace.refresh_graph_values()

    engine = ForwardEngine(trace, tangents, create_graph=create_graph)
    return engine.run()


__all__ = ["forward_on_trace", "ForwardEngine"]
