# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Core backpropagation utilities for Trace-based automatic differentiation."""

from __future__ import annotations

import os

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..graph.tracing import OpNode, Trace
    from ..tensor.impl import TensorImpl

GradsMap = dict["Tensor", "Tensor"]
CotangentMap = dict["TensorImpl", "TensorImpl"]

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
        for i, (c_d, a_d) in enumerate(zip(cot_shape, target_shape, strict=False))
        if a_d == 1 and c_d > 1
    ]
    if reduce_axes:
        from ...ops.reduction import reduce_sum

        cot_tensor = reduce_sum(cot_tensor, axis=reduce_axes, keepdims=True)

    return cot_tensor


def _accumulate_cotangent(
    cotangent_map: CotangentMap,
    target_impl: TensorImpl,
    cot_tensor: Tensor,
) -> None:
    """Accumulates a cotangent tensor into the global cotangent map for a specific target.

    Handles sharding resolution (partial sums, resharding) and addition.
    """
    from ...core.sharding.spec import needs_reshard
    from ...ops.binary import add
    from ...ops.communication import all_reduce, reshard

    target_key = target_impl

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
    if target_key in cotangent_map:
        existing = Tensor(impl=cotangent_map[target_key])
        try:
            accumulated = add(existing, cot_tensor)
        except Exception:
            if os.environ.get("NABLA_DEBUG_COT_ACCUM", "0") in {
                "1",
                "true",
                "TRUE",
                "True",
            }:
                target_tensor = Tensor(impl=target_impl)
                target_phys = (
                    target_tensor.physical_global_shape or target_tensor.local_shape
                )
                existing_phys = existing.physical_global_shape or existing.local_shape
                cot_phys = cot_tensor.physical_global_shape or cot_tensor.local_shape
                print(
                    "[NABLA_DEBUG_COT_ACCUM] add failed: "
                    f"target shape={tuple(int(d) for d in target_tensor.shape)} "
                    f"batch_dims={target_tensor.batch_dims} "
                    f"phys={tuple(int(d) for d in target_phys)} | "
                    f"existing shape={tuple(int(d) for d in existing.shape)} "
                    f"batch_dims={existing.batch_dims} "
                    f"phys={tuple(int(d) for d in existing_phys)} | "
                    f"incoming shape={tuple(int(d) for d in cot_tensor.shape)} "
                    f"batch_dims={cot_tensor.batch_dims} "
                    f"phys={tuple(int(d) for d in cot_phys)}"
                )
            raise
        cotangent_map[target_key] = accumulated._impl
    else:
        cotangent_map[target_key] = cot_tensor._impl


class BackwardEngine:
    """Stateful engine for backpropagation on a captured Trace."""

    trace: Trace
    create_graph: bool
    cotangent_map: CotangentMap
    _original_flags: dict[int, bool]

    def __init__(
        self, trace: Trace, cotangents: Any, create_graph: bool = False
    ) -> None:
        from ..common import pytree

        self.trace = trace
        self.create_graph = create_graph
        self.cotangent_map: dict[TensorImpl, TensorImpl] = {}
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
            self.cotangent_map[out_impl] = cot_impl

    def _set_trace_state(self, tree: Any):
        """Suppress tracing for backward ops if create_graph=False."""
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

    def run(self) -> GradsMap:
        """Execute backward pass."""
        for node in reversed(self.trace.nodes):
            self._process_node(node)
        return self._finalize()

    def _process_node(self, node: OpNode) -> None:
        from ..tensor.impl import TensorImpl

        alive_outputs = node.get_alive_outputs()
        op = node.op

        if not hasattr(op, "vjp_rule"):
            return

        active_indices = [
            i
            for i, o in enumerate(alive_outputs)
            if o is not None and o in self.cotangent_map
        ]
        if not active_indices:
            return

        # 2. Prepare flat list arguments for VJP rule
        def wrap(x):
            if isinstance(x, (Tensor, TensorImpl)):
                impl = x._impl if isinstance(x, Tensor) else x
                return Tensor(impl=impl)
            return x

        # Flat list of primal Tensors (preserving non-Tensor args as-is)
        vjp_primals = [wrap(a) for a in node.op_args]

        # Flat list of output Tensors
        vjp_outputs = [Tensor(impl=o) if o is not None else None for o in alive_outputs]

        # Flat list of cotangent Tensors (matching outputs)
        vjp_cotangents = []
        for o in alive_outputs:
            if o is not None:
                if o in self.cotangent_map:
                    vjp_cotangents.append(Tensor(impl=self.cotangent_map[o]))
                else:
                    from ...ops.creation import zeros_like

                    vjp_cotangents.append(zeros_like(Tensor(impl=o)))
            else:
                vjp_cotangents.append(None)

        # Get kwargs from the OpNode (stored during forward pass)
        vjp_kwargs = node.op_kwargs or {}

        # 3. Invoke VJP Rule (unified: flat lists + kwargs)
        vjp_inputs = [vjp_primals, vjp_outputs, vjp_cotangents]
        self._set_trace_state(vjp_inputs)

        try:
            input_cotangents = op.vjp_rule(
                vjp_primals, vjp_cotangents, vjp_outputs, vjp_kwargs
            )
        except Exception as e:
            self._restore_trace_state(vjp_inputs)
            op_name = getattr(op, "name", str(op))
            raise RuntimeError(f"VJP rule failed for operation '{op_name}': {e}") from e
        finally:
            self._restore_trace_state(vjp_inputs)

        # 4. Align and Accumulate — input_cotangents is flat list matching primals
        arg_leaves = list(node.op_args)

        for arg_impl, cot_result in zip(arg_leaves, input_cotangents, strict=False):
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

            if os.environ.get("NABLA_DEBUG_COT_FLOW", "0") in {
                "1",
                "true",
                "TRUE",
                "True",
            }:
                target_tensor = Tensor(impl=arg_impl_real)
                target_phys = target_tensor.physical_global_shape or target_tensor.local_shape
                cot_phys = cot_tensor.physical_global_shape or cot_tensor.local_shape
                existing = self.cotangent_map.get(arg_impl_real)
                if existing is None:
                    print(
                        "[NABLA_DEBUG_COT_FLOW] init "
                        f"op={getattr(op, 'name', type(op).__name__)} "
                        f"target_shape={tuple(int(d) for d in target_tensor.shape)} "
                        f"target_batch_dims={target_tensor.batch_dims} "
                        f"target_phys={tuple(int(d) for d in target_phys)} "
                        f"cot_shape={tuple(int(d) for d in cot_tensor.shape)} "
                        f"cot_batch_dims={cot_tensor.batch_dims} "
                        f"cot_phys={tuple(int(d) for d in cot_phys)}"
                    )
                else:
                    ex_t = Tensor(impl=existing)
                    ex_phys = ex_t.physical_global_shape or ex_t.local_shape
                    print(
                        "[NABLA_DEBUG_COT_FLOW] add "
                        f"op={getattr(op, 'name', type(op).__name__)} "
                        f"target_shape={tuple(int(d) for d in target_tensor.shape)} "
                        f"target_batch_dims={target_tensor.batch_dims} "
                        f"target_phys={tuple(int(d) for d in target_phys)} "
                        f"existing_shape={tuple(int(d) for d in ex_t.shape)} "
                        f"existing_batch_dims={ex_t.batch_dims} "
                        f"existing_phys={tuple(int(d) for d in ex_phys)} "
                        f"incoming_shape={tuple(int(d) for d in cot_tensor.shape)} "
                        f"incoming_batch_dims={cot_tensor.batch_dims} "
                        f"incoming_phys={tuple(int(d) for d in cot_phys)}"
                    )

            # Sharding and Addition
            _accumulate_cotangent(self.cotangent_map, arg_impl_real, cot_tensor)

    def _finalize(self) -> GradsMap:
        """Convert accumulated cotangents to input gradients."""
        from ..common import pytree

        gradients = {}
        input_leaves = [
            t for t in pytree.tree_leaves(self.trace.inputs) if isinstance(t, Tensor)
        ]

        for inp in input_leaves:
            if inp._impl in self.cotangent_map:
                cot_impl = self.cotangent_map[inp._impl]
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
) -> GradsMap:
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


def backward(
    outputs: Any,
    cotangents: Any = None,
    *,
    create_graph: bool = False,
) -> None:
    """PyTorch-style backward pass that populates .grad on requires_grad tensors.

    This function:
    1. Builds a Trace from outputs using compute_for_backward()
       - Traverses through all OpNodes back to true leaves.
       - Collects all tensors with requires_grad=True as gradient leaves.
    2. Runs VJP on the trace.
    3. Populates .grad attributes on the collected gradient leaves.
    4. Batch-realizes all gradients for efficiency.
    """
    from ..common import pytree
    from ..graph.tracing import Trace

    # 1. Handle default cotangents
    if cotangents is None:
        output_leaves = [
            t for t in pytree.tree_leaves(outputs) if isinstance(t, Tensor)
        ]
        if len(output_leaves) == 1:
            out = output_leaves[0]
            if len(out.shape) == 0 or (len(out.shape) == 1 and out.shape[0] == 1):
                from ...ops.creation import ones_like

                cotangents = ones_like(out)
            else:
                raise ValueError(
                    "backward() requires cotangents for non-scalar outputs. "
                    f"Output has shape {out.shape}."
                )
        else:
            raise ValueError(
                "backward() requires explicit cotangents for multiple outputs."
            )

    # 2. Build Trace from outputs and collect gradient leaves
    trace = Trace(inputs=(), outputs=outputs)
    trace.compute_for_backward()

    if not trace.nodes:
        # If no nodes, just check if any output itself requires grad
        output_leaves = [
            t._impl for t in pytree.tree_leaves(outputs) if isinstance(t, Tensor)
        ]
        cot_leaves = [
            t._impl for t in pytree.tree_leaves(cotangents) if isinstance(t, Tensor)
        ]
        for out_impl, cot_impl in zip(output_leaves, cot_leaves, strict=False):
            if out_impl.requires_grad:
                out_impl.cotangent = cot_impl
        return

    # 3. Refresh graph values (Crucial for Eager Max Graph)
    from ...config import EAGER_MAX_GRAPH

    if EAGER_MAX_GRAPH:
        trace.refresh_graph_values()

    # 4. Run Backward Engine
    engine = BackwardEngine(trace, cotangents, create_graph=create_graph)
    engine.run()

    # 5. Populate .grad on gradient leaves and collect for batch realization
    grad_tensors_to_realize = []

    for impl in trace.gradient_leaves:
        if impl in engine.cotangent_map:
            cot_impl = engine.cotangent_map[impl]
            impl.cotangent = cot_impl
            grad_tensors_to_realize.append(Tensor(impl=cot_impl))

    # 6. Batch-realize all gradients for efficiency
    if grad_tensors_to_realize:
        from ..tensor.api import realize_all

        realize_all(*grad_tensors_to_realize)


__all__ = ["backward_on_trace", "backward"]
