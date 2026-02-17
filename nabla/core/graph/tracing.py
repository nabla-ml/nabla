# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Tracing infrastructure and visualization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..common import pytree
from ..common.pytree import PyTreeDef, tree_leaves
from ..tensor.impl import TensorImpl

if TYPE_CHECKING:
    from max.graph.graph import Shape

    from ...ops import Operation
    from ..tensor.impl import TensorImpl


@dataclass(frozen=True)
class OpNode:
    """Lightweight container for multi-output operation siblings.

    Attributes:
        _refs: Strong references to output TensorImpls.
        tree_def: PyTreeDef for output structure.
        op: Producing Operation.
        op_args: Original input arguments.
        op_kwargs: Original logical keyword arguments.
        _op_hash: Operation hash for cache key computation.
    """

    _refs: tuple[TensorImpl | None, ...]
    tree_def: PyTreeDef
    op: Operation
    op_args: tuple[Any, ...]
    op_kwargs: dict[str, Any] | None
    _op_hash: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        """Validate that refs and tree_def are consistent."""
        if len(self._refs) != self.tree_def.num_leaves:
            raise ValueError(
                f"OpNode: ref count {len(self._refs)} doesn't match "
                f"tree_def leaves {self.tree_def.num_leaves}"
            )

    def get_alive_outputs(self) -> list[TensorImpl | None]:
        """Get output TensorImpls."""
        return list(self._refs)

    @property
    def num_outputs(self) -> int:
        """Number of outputs (including potentially dead ones)."""
        return len(self._refs)

    def __repr__(self) -> str:
        alive = len(self._refs)
        op_name = self.op.name if hasattr(self.op, "name") else str(self.op)
        return f"OpNode(op={op_name}, outputs={self.num_outputs}, alive={alive})"


COMM_OPS = {
    "shard",
    "all_gather",
    "all_reduce",
    "reduce_scatter",
    "gather_all_axes",
    "reshard",
    "ppermute",
    "all_to_all",
    "pmean",
}


RESET = "\033[0m"
C_VAR = "\033[96m"
C_KEYWORD = "\033[1m"
C_BATCH = "\033[90m"


class Trace:
    """Represents a captured computation subgraph.

    Attributes:
        inputs: Input pytree structure.
        outputs: Output pytree structure.
        nodes: Topological list of OpNode.
    """

    def __init__(self, inputs: Any, outputs: Any) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self._computed = False
        self.nodes: list[OpNode] = []
        self.gradient_leaves: list[TensorImpl] = []

        from ..tensor.api import Tensor

        self._input_tensor_ids = {
            id(t._impl) for t in tree_leaves(inputs) if isinstance(t, Tensor)
        }

    def _compute_topology(self, stop_at_inputs: bool, collect_grad: bool) -> None:
        """Shared DFS topology computation for compute() and compute_for_backward()."""
        if self._computed:
            return

        visited: set[int] = set()
        nodes: list[OpNode] = []
        grad_leaves: list[TensorImpl] | None = [] if collect_grad else None

        from ..tensor.api import Tensor

        output_leaves = [
            t._impl for t in tree_leaves(self.outputs) if isinstance(t, Tensor)
        ]

        def dfs(refs: OpNode) -> None:
            refs_id = id(refs)
            if refs_id in visited:
                return

            arg_leaves = []
            for arg in tree_leaves(refs.op_args):
                if isinstance(arg, Tensor):
                    arg_leaves.append(arg._impl)
                elif isinstance(arg, TensorImpl):
                    arg_leaves.append(arg)

            for arg in arg_leaves:
                if (
                    collect_grad
                    and grad_leaves is not None
                    and arg.requires_grad
                    and arg not in grad_leaves
                ):
                    grad_leaves.append(arg)
                if stop_at_inputs and id(arg) in self._input_tensor_ids:
                    continue
                if arg.output_refs is not None:
                    dfs(arg.output_refs)

            visited.add(refs_id)
            nodes.append(refs)

        if collect_grad:
            for leaf in output_leaves:
                if (
                    grad_leaves is not None
                    and leaf.requires_grad
                    and leaf not in grad_leaves
                ):
                    grad_leaves.append(leaf)
                if leaf.output_refs is not None:
                    dfs(leaf.output_refs)
        else:
            root_refs: list[OpNode] = []
            for leaf in output_leaves:
                if leaf.output_refs is not None:
                    root_refs.append(leaf.output_refs)
            for root in root_refs:
                dfs(root)

        self.nodes = nodes
        if collect_grad:
            self.gradient_leaves = grad_leaves or []
        self._computed = True

    def compute(self) -> None:
        """Compute subgraph topology."""
        self._compute_topology(stop_at_inputs=True, collect_grad=False)

    def compute_for_backward(self) -> None:
        """Compute subgraph topology for backward pass.

        Unlike compute(), this method:
        1. Traverses the ENTIRE graph back to leaves (ignores self._input_tensor_ids).
        2. Collects all TensorImpls encountered that have requires_grad=True.
        3. Only stops at true leaves (no output_refs).
        """
        self._compute_topology(stop_at_inputs=False, collect_grad=True)

    def refresh_graph_values(self) -> None:
        """Rehydrate all TensorImpl._graph_values by replaying operations.

        Identifies all leaf tensors in the trace (including captured constants),
        ensures they are realized, and then replays all operations in
        topological order to re-establish graph values for the current epoch.
        """
        if not self._computed:
            self.compute()

        from ..tensor.api import Tensor
        from .engine import GRAPH

        # 1. Collect all leaf TensorImpls in the trace (stable order).
        leaf_impls = []
        seen_leaf_impls: set[int] = set()

        def add_leaf(impl: TensorImpl) -> None:
            iid = id(impl)
            if iid in seen_leaf_impls:
                return
            seen_leaf_impls.add(iid)
            leaf_impls.append(impl)

        for ref in self.nodes:
            for arg in tree_leaves(ref.op_args):
                if isinstance(arg, (Tensor, TensorImpl)):
                    impl = arg._impl if isinstance(arg, Tensor) else arg
                    if not impl.output_refs:
                        add_leaf(impl)

        # Always include user-provided inputs as leaves.
        for inp in tree_leaves(self.inputs):
            if isinstance(inp, (Tensor, TensorImpl)):
                add_leaf(inp._impl if isinstance(inp, Tensor) else inp)

        # 2. Realize all unrealized leaves together to reduce compilation cycles.
        leaf_tensors = [Tensor(impl=impl) for impl in leaf_impls]
        unrealized_leaves = [t for t in leaf_tensors if not t.real]

        if unrealized_leaves:
            # Evaluate all found unrealized leaves in a single graph
            if len(unrealized_leaves) > 1:
                GRAPH.evaluate(unrealized_leaves[0], *unrealized_leaves[1:])
            else:
                GRAPH.evaluate(unrealized_leaves[0])

        for t in leaf_tensors:
            if t.real:
                GRAPH.add_input(t)

        # 4. Iterate through nodes and call kernel_all to recompute intermediates.
        for _, output_refs in enumerate(self.nodes):
            alive_outputs = output_refs.get_alive_outputs()
            if not any(out is not None for out in alive_outputs):
                continue

            op = output_refs.op

            def to_tensor(x):
                if isinstance(x, TensorImpl):
                    return Tensor(impl=x)
                return x

            op_args = pytree.tree_map(to_tensor, output_refs.op_args)
            op_kwargs = output_refs.op_kwargs or {}

            # Collect metadata for current batch_dims awareness.
            max_batch_dims = 0
            for arg in tree_leaves(op_args):
                if isinstance(arg, Tensor) and arg.batch_dims > max_batch_dims:
                    max_batch_dims = arg.batch_dims

            # Use adapt_kwargs dynamically during rehydration as batch_dims might change
            if hasattr(op, "adapt_kwargs"):
                _adapted_kwargs = op.adapt_kwargs(op_args, op_kwargs, max_batch_dims)
            else:
                _adapted_kwargs = op_kwargs

            # Determine sharding context from one of the alive outputs.
            mesh = None
            output_sharding = None
            first_out = next((o for o in alive_outputs if o is not None), None)
            if first_out and first_out.sharding:
                output_sharding = first_out.sharding
                mesh = first_out.sharding.mesh

            # === New Physical Execution Path ===
            if hasattr(op, "execute"):
                try:
                    # execute returns a PhysicalResult (or tuple/list of TensorValues)
                    # It handles its own auto-reduction and internal logic.
                    # We pass the original arguments and kwargs.
                    with GRAPH.graph:
                        raw_result = op.execute(op_args, op_kwargs)

                    # If it returns a named tuple or custom object, we might need to extract values.
                    # For now, we assume it returns the raw value structure matching the output.
                    # If PhysicalResult (namedtuple) is used, extracting .shard_graph_values is needed.
                    # Let's assume for now it returns the values directly as the plan implies "raw values".
                    # However, the plan mentioned PhysicalResult(shard_graph_values, ...).
                    # Let's be flexible: check if it has 'shard_graph_values' attr.
                    if isinstance(raw_result, tuple) and len(raw_result) == 3:
                        # Standard Tuple Return: (values, sharding, mesh)
                        shard_graph_values, r_sharding, r_mesh = raw_result

                        from ..sharding import spmd

                        output_tensor_struct = spmd.create_sharded_output(
                            shard_graph_values,
                            r_sharding,
                            is_traced=True,  # Always true in rehydration
                            batch_dims=max_batch_dims,
                            mesh=r_mesh,
                        )
                    elif hasattr(raw_result, "shard_graph_values"):
                        # Legacy/Object Return (PhysicalResult)
                        r_sharding = (
                            raw_result.output_sharding
                            if hasattr(raw_result, "output_sharding")
                            else output_sharding
                        )
                        r_mesh = (
                            raw_result.mesh if hasattr(raw_result, "mesh") else mesh
                        )

                        from ..sharding import spmd

                        output_tensor_struct = spmd.create_sharded_output(
                            raw_result.shard_graph_values,
                            r_sharding,
                            is_traced=True,  # Always true in rehydration
                            batch_dims=max_batch_dims,
                            mesh=r_mesh,
                        )
                    else:
                        # Direct Return (if strict contract not yet fully enforced or simpler op)
                        output_tensor_struct = raw_result

                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    print(f"ERROR: execute failed for {op.name}: {e}")
                    output_tensor_struct = None
            else:
                # All operations should now have execute implemented
                raise NotImplementedError(
                    f"Operation '{op.name}' missing execute. "
                    f"All operations must implement execute for trace rehydration."
                )

            if output_tensor_struct is None:
                # print(f"DEBUG: kernel_all returned None for {op.name}")
                continue

            # Extract produced TensorImpls
            produced_leaves = pytree.tree_leaves(
                output_tensor_struct, is_leaf=pytree.is_tensor
            )

            produced_impls = []
            for leaf in produced_leaves:
                if isinstance(leaf, Tensor):
                    produced_impls.append(leaf._impl)
                elif isinstance(leaf, TensorImpl):
                    produced_impls.append(leaf)
                else:
                    produced_impls.append(None)

            # Map produced values back to the alive outputs in the trace.
            for ref, produced_impl in zip(
                output_refs._refs, produced_impls, strict=False
            ):
                out_impl = ref
                if out_impl is not None and produced_impl is not None:
                    out_impl._graph_values = produced_impl._graph_values
                    out_impl.graph_values_epoch = GRAPH.epoch

    def __str__(self) -> str:
        """Pretty-print the trace."""
        if not self._computed:
            self.compute()
        return GraphPrinter(self).to_string()

    def __repr__(self) -> str:
        from ..tensor.api import Tensor

        n_inputs = len([t for t in tree_leaves(self.inputs) if isinstance(t, Tensor)])
        n_outputs = len([t for t in tree_leaves(self.outputs) if isinstance(t, Tensor)])
        n_nodes = len(self.nodes) if self._computed else "?"
        return f"Trace(inputs={n_inputs}, outputs={n_outputs}, nodes={n_nodes})"


def trace(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Trace:
    """Trace a function's computation graph."""

    from ..tensor.api import Tensor

    flat_args = tree_leaves(args)
    flat_kwargs = tree_leaves(kwargs)
    all_inputs = flat_args + flat_kwargs
    input_tensors = [t for t in all_inputs if isinstance(t, Tensor)]

    if not input_tensors:
        raise ValueError("trace: No input tensors found. Pass at least one Tensor.")

    original_is_traced: dict[int, bool] = {}
    for t in input_tensors:
        original_is_traced[id(t)] = t.is_traced
        t.is_traced = True

    try:
        outputs = fn(*args, **kwargs)

        traced_obj = Trace(args, outputs)
        traced_obj.compute()
        # print("\n--- TRACE CAPTURED ---")
        # print(traced_obj)
        # print("----------------------\n")
        return traced_obj

    finally:
        for t in input_tensors:
            if id(t) in original_is_traced:
                t.is_traced = original_is_traced[id(t)]


class GraphPrinter:
    """Visualizes a Trace."""

    def __init__(self, trace: Trace):
        self.trace = trace
        self.var_names: dict[int, str] = {}
        self.name_counters: dict[str, int] = {}

    def _get_next_name(self, prefix: str = "v") -> str:
        if prefix not in self.name_counters:
            self.name_counters[prefix] = 0
        self.name_counters[prefix] += 1
        return f"%{prefix}{self.name_counters[prefix]}"

    def _format_type(self, node: TensorImpl) -> str:
        from ..tensor.impl import TensorImpl

        return TensorImpl._format_type(node)

    def _format_shape_part(
        self, shape: tuple[int, ...] | list[int] | Shape, batch_dims: int = 0
    ) -> str:
        """Format a shape tuple with batch dims colors: [2, 3 | 4, 5]"""
        from ..tensor.impl import TensorImpl

        return TensorImpl._format_shape_part(shape, batch_dims)

    def _format_spec_factors(self, sharding: Any) -> str:
        """Format sharding factors: (<dp, tp>)"""
        from ..tensor.impl import TensorImpl

        return TensorImpl._format_spec_factors(sharding)

    def _format_full_info(self, node: TensorImpl) -> str:
        """Format: dtype[global](factors)(local=[local])"""
        return node.format_metadata(include_data=False)

    def _format_mesh_def(self, mesh: Any) -> str:
        """Format mesh definition: @mesh(shape=(2,2), devices=[...], axes=(dp, tp))"""
        if not mesh:
            return ""

        shape_str = str(tuple(mesh.shape)).replace(" ", "")
        axes_str = f"({', '.join(mesh.axis_names)})"

        devs = mesh.devices
        if len(devs) > 8:
            dev_str = f"[{devs[0]}..{devs[-1]}]"
        else:
            dev_str = str(devs).replace(" ", "")

        return f"@{mesh.name}(shape={shape_str}, devices={dev_str}, axes={axes_str})"

    def _format_kwargs(self, kwargs: dict | None) -> str:
        if not kwargs:
            return ""
        parts = []

        for k in sorted(kwargs.keys()):
            v = kwargs[k]

            if k in ("shard_idx",):
                continue

            if k == "mesh" and hasattr(v, "name"):
                parts.append(f"mesh=@{v.name}")

            elif k == "dim_specs" and isinstance(v, (list, tuple)):
                factors = []
                all_p_axes = set()
                for dim in v:
                    if getattr(dim, "partial", False):
                        all_p_axes.update(dim.axes)

                    if hasattr(dim, "axes"):
                        if not dim.axes:
                            factors.append("*")
                        else:
                            factors.append(", ".join(dim.axes))
                    else:
                        factors.append(str(dim))

                p_str = ""
                if all_p_axes:
                    ordered_p = sorted(all_p_axes)
                    axes_joined = ", ".join(f"'{a}'" for a in ordered_p)
                    p_str = f" | partial={{{axes_joined}}}"
                parts.append(f"spec=<{', '.join(factors)}>{p_str}")

            elif k == "spec":
                continue
            elif isinstance(v, (list, tuple)):
                v_clean = tuple(int(x) if hasattr(x, "__int__") else x for x in v)
                v_str = str(v_clean).replace(" ", "")
                parts.append(f"{k}={v_str}")
            elif hasattr(v, "name"):
                parts.append(f"{k}=@{v.name}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)

    def to_string(self) -> str:
        if not self.trace._computed:
            self.trace.compute()

        from ..tensor.api import Tensor

        lines = []

        meshes_used: dict[str, Any] = {}

        # 1) Input/output tensor shardings
        tensor_impls = [
            t._impl for t in tree_leaves(self.trace.inputs) if isinstance(t, Tensor)
        ] + [t._impl for t in tree_leaves(self.trace.outputs) if isinstance(t, Tensor)]
        for impl in tensor_impls:
            if hasattr(impl, "sharding") and impl.sharding and impl.sharding.mesh:
                m = impl.sharding.mesh
                meshes_used[m.name] = m

        # 2) Meshes referenced by traced operations (e.g. mesh=@mesh_8 in kwargs)
        for refs in self.trace.nodes:
            kwargs = refs.op_kwargs or {}
            mesh = kwargs.get("mesh")
            if mesh is not None and hasattr(mesh, "name"):
                meshes_used[mesh.name] = mesh

            for out in refs.get_alive_outputs():
                if out is not None and out.sharding and out.sharding.mesh:
                    m = out.sharding.mesh
                    meshes_used[m.name] = m

        if meshes_used:
            for mesh_name in sorted(meshes_used.keys()):
                lines.append(self._format_mesh_def(meshes_used[mesh_name]))

        input_leaves = [
            t._impl for t in tree_leaves(self.trace.inputs) if isinstance(t, Tensor)
        ]
        input_vars = []
        for node in input_leaves:
            if id(node) not in self.var_names:
                name = self._get_next_name("a")
                self.var_names[id(node)] = name

            desc = f"{C_VAR}{self.var_names[id(node)]}{RESET}: {self._format_full_info(node)}"
            input_vars.append(desc)

        if len(input_vars) > 1:
            lines.append(f"{C_KEYWORD}fn{RESET}(")
            for i, var in enumerate(input_vars):
                comma = "," if i < len(input_vars) - 1 else ""
                lines.append(f"    {var}{comma}")
            lines.append(") {")
        else:
            lines.append(f"{C_KEYWORD}fn{RESET}({', '.join(input_vars)}) {{ ")

        current_block_mesh = None
        block_lines = []

        def flush_block():
            nonlocal current_block_mesh, block_lines
            if current_block_mesh and block_lines:
                mesh_def = self._format_mesh_def(current_block_mesh)
                lines.append(f"  {C_KEYWORD}spmd{RESET} {mesh_def} {{")
                lines.extend(block_lines)
                lines.append("  }")
            elif block_lines:
                lines.extend(block_lines)

            current_block_mesh = None
            block_lines = []

        for refs in self.trace.nodes:
            op_name = (
                refs.op.name.lower()
                if hasattr(refs.op, "name")
                else str(type(refs.op).__name__)
            )

            arg_names = []

            def collect_arg_names(x, arg_names=arg_names):
                if isinstance(x, Tensor):
                    arg_names.append(
                        f"{C_VAR}{self.var_names.get(id(x._impl), '?')}{RESET}"
                    )
                elif isinstance(x, TensorImpl):
                    arg_names.append(f"{C_VAR}{self.var_names.get(id(x), '?')}{RESET}")
                return x

            pytree.tree_map(collect_arg_names, refs.op_args)

            args_str = ", ".join(arg_names)
            kwargs_str = self._format_kwargs(refs.op_kwargs)

            if not args_str and op_name in ("shard", "reshard"):
                args_str = f"{C_VAR}const{RESET}"

            call_args = ", ".join(filter(None, [args_str, kwargs_str]))

            outputs = refs.get_alive_outputs()
            valid_outputs = [o for o in outputs if o is not None]

            if not valid_outputs:
                continue

            lhs_parts = []
            is_sharded = False
            first_valid = valid_outputs[0]
            if first_valid.sharding and first_valid.sharding.mesh:
                is_sharded = True

            mesh = (
                first_valid.sharding.mesh
                if is_sharded and first_valid.sharding
                else None
            )

            for out in outputs:
                if out is None:
                    lhs_parts.append("_")
                    continue

                name = self._get_next_name()
                self.var_names[id(out)] = name
                info = self._format_full_info(out)
                lhs_parts.append(f"{C_VAR}{name}{RESET}: {info}")

            lhs_str = ", ".join(lhs_parts)
            line = f"    {lhs_str} = {op_name}({call_args})"

            is_comm = op_name in COMM_OPS

            if is_comm:
                flush_block()
                lines.append(line.replace("    ", "  "))
            elif is_sharded:
                if current_block_mesh and mesh and current_block_mesh.name != mesh.name:
                    flush_block()

                current_block_mesh = mesh
                block_lines.append(line)
            else:
                flush_block()
                lines.append(line.replace("    ", "  "))

        flush_block()

        output_leaves = [
            t._impl for t in tree_leaves(self.trace.outputs) if isinstance(t, Tensor)
        ]
        output_vars = [
            f"{C_VAR}{self.var_names.get(id(n), '?')}{RESET}" for n in output_leaves
        ]
        output_sig = ", ".join(output_vars) if output_vars else "()"

        if len(output_vars) == 1:
            lines.append(f"  {C_KEYWORD}return{RESET} {output_sig}")
        else:
            lines.append(f"  {C_KEYWORD}return{RESET} ({output_sig})")

        lines.append("}")
        return "\n".join(lines)


__all__ = ["Trace", "trace", "OpNode"]
