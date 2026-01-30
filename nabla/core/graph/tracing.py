# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Tracing infrastructure and visualization."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..common import pytree
from ..common.pytree import PyTreeDef, tree_leaves
from ..sharding.spec import compute_global_shape
from ..tensor.impl import TensorImpl

if TYPE_CHECKING:
    from ...ops import Operation
    from ..tensor.impl import TensorImpl


@dataclass(frozen=True)
class OutputRefs:
    """Lightweight container for multi-output operation siblings.

    Attributes:
        _refs: Weak references to output TensorImpls.
        tree_def: PyTreeDef for output structure.
        op: Producing Operation.
        op_args: Original input arguments.
        op_kwargs: Original logical keyword arguments.
        physical_kwargs: Adapted physical keyword arguments.
    """

    _refs: tuple[weakref.ref, ...]
    tree_def: PyTreeDef
    op: Operation
    op_args: tuple[Any, ...]
    op_kwargs: dict[str, Any] | None
    physical_kwargs: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate that refs and tree_def are consistent."""
        if len(self._refs) != self.tree_def.num_leaves:
            raise ValueError(
                f"OutputRefs: ref count {len(self._refs)} doesn't match "
                f"tree_def leaves {self.tree_def.num_leaves}"
            )

    def get_alive_outputs(self) -> list[TensorImpl | None]:
        """Get output TensorImpls (None if GC'd)."""
        return [ref() for ref in self._refs]

    @property
    def num_outputs(self) -> int:
        """Number of outputs (including potentially dead ones)."""
        return len(self._refs)

    def __repr__(self) -> str:
        alive = sum(1 for ref in self._refs if ref() is not None)
        op_name = self.op.name if hasattr(self.op, "name") else str(self.op)
        return f"OutputRefs(op={op_name}, outputs={self.num_outputs}, alive={alive})"


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
        nodes: Topological list of OutputRefs.
    """

    def __init__(self, inputs: Any, outputs: Any):
        self.inputs = inputs
        self.outputs = outputs
        self._computed = False
        self.nodes: list[OutputRefs] = []

        from ..tensor.api import Tensor

        self._input_tensor_ids = {
            id(t._impl) for t in tree_leaves(inputs) if isinstance(t, Tensor)
        }

    def compute(self) -> None:
        """Compute subgraph topology."""
        if self._computed:
            return

        visited: set[int] = set()
        nodes: list[OutputRefs] = []

        from ..tensor.api import Tensor

        output_leaves = [
            t._impl for t in tree_leaves(self.outputs) if isinstance(t, Tensor)
        ]

        root_refs: list[OutputRefs] = []
        for leaf in output_leaves:
            if leaf.output_refs is not None:
                root_refs.append(leaf.output_refs)

        def dfs(refs: OutputRefs) -> None:
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

                if id(arg) in self._input_tensor_ids:
                    continue

                if arg.output_refs:
                    dfs(arg.output_refs)

            visited.add(refs_id)
            nodes.append(refs)

        for root in root_refs:
            dfs(root)

        self.nodes = nodes
        self._computed = True

    def rehydrate(self) -> None:
        """Rehydrate all TensorImpl._values by replaying operations.

        Identifies all leaf tensors in the trace (including captured constants),
        ensures they are realized, and then replays all operations in
        topological order to re-establish graph values for the current epoch.
        """
        if not self._computed:
            self.compute()

        from ..tensor.api import Tensor
        from .engine import GRAPH

        # 1. Collect all leaf TensorImpls in the trace.
        leaf_impls = set()
        for ref in self.nodes:
            for arg in tree_leaves(ref.op_args):
                if isinstance(arg, (Tensor, TensorImpl)):
                    impl = arg._impl if isinstance(arg, Tensor) else arg
                    if not impl.output_refs:
                        leaf_impls.add(impl)

        # Always include user-provided inputs as leaves.
        for inp in tree_leaves(self.inputs):
            if isinstance(inp, (Tensor, TensorImpl)):
                leaf_impls.add(inp._impl if isinstance(inp, Tensor) else inp)

        # 2. Realize all leaves together.
        leaf_tensors = [Tensor(impl=impl) for impl in leaf_impls]
        if leaf_tensors:
            GRAPH.evaluate(leaf_tensors[0], *leaf_tensors[1:])

        # 3. Add all leaves to the current graph to get fresh values.
        for t in leaf_tensors:
            GRAPH.add_input(t)
            if hasattr(t, "_values") and t._values:
                with GRAPH.graph:
                    t._impl._values = [v[...] for v in t._values]
                    t._impl._values = [v[...] for v in t._values]
                    t._impl.values_epoch = GRAPH.epoch

        # 4. Iterate through nodes and call maxpr_all to recompute intermediates.
        for i, output_refs in enumerate(self.nodes):

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
                if isinstance(arg, Tensor):
                    if arg.batch_dims > max_batch_dims:
                        max_batch_dims = arg.batch_dims

            # Use adapt_kwargs dynamically during rehydration as batch_dims might change
            if hasattr(op, "adapt_kwargs"):
                adapted_kwargs = op.adapt_kwargs(op_args, op_kwargs, max_batch_dims)
            else:
                adapted_kwargs = op_kwargs

            # Determine sharding context from one of the alive outputs.
            mesh = None
            output_sharding = None
            first_out = next((o for o in alive_outputs if o is not None), None)
            if first_out and first_out.sharding:
                output_sharding = first_out.sharding
                mesh = first_out.sharding.mesh

            # === New Physical Execution Path ===
            if hasattr(op, "physical_execute"):
                try:
                    # physical_execute returns a PhysicalResult (or tuple/list of TensorValues)
                    # It handles its own auto-reduction and internal logic.
                    # We pass the original arguments and kwargs.
                    with GRAPH.graph:
                        raw_result = op.physical_execute(op_args, op_kwargs)

                    # If it returns a named tuple or custom object, we might need to extract values.
                    # For now, we assume it returns the raw value structure matching the output.
                    # If PhysicalResult (namedtuple) is used, extracting .shard_values is needed.
                    # Let's assume for now it returns the values directly as the plan implies "raw values".
                    # However, the plan mentioned PhysicalResult(shard_values, ...).
                    # Let's be flexible: check if it has 'shard_values' attr.
                    if isinstance(raw_result, tuple) and len(raw_result) == 3:
                        # Standard Tuple Return: (values, sharding, mesh)
                        shard_values, r_sharding, r_mesh = raw_result

                        from ..sharding import spmd

                        output_tensor_struct = spmd.create_sharded_output(
                            shard_values,
                            r_sharding,
                            traced=True,  # Always true in rehydration
                            batch_dims=max_batch_dims,
                            mesh=r_mesh,
                        )
                    elif hasattr(raw_result, "shard_values"):
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
                            raw_result.shard_values,
                            r_sharding,
                            traced=True,  # Always true in rehydration
                            batch_dims=max_batch_dims,
                            mesh=r_mesh,
                        )
                    else:
                        # Direct Return (if strict contract not yet fully enforced or simpler op)
                        output_tensor_struct = raw_result

                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    print(f"ERROR: physical_execute failed for {op.name}: {e}")
                    output_tensor_struct = None
            else:
                # === Legacy Path (DEPRECATED - Kept for reference) ===
                # All operations should now have physical_execute implemented.
                # If you see this error, the operation needs to be refactored.
                raise NotImplementedError(
                    f"Operation '{op.name}' missing physical_execute. "
                    f"All operations must implement physical_execute for trace rehydration."
                )
                # try:
                #     output_tensor_struct = op.maxpr_all(
                #         op_args,
                #         adapted_kwargs,
                #         output_sharding,
                #         mesh,
                #         any_traced=False,
                #         max_batch_dims=max_batch_dims,
                #         original_kwargs=op_kwargs,
                #     )
                # except Exception as e:
                #     print(f"ERROR: maxpr_all failed for {op.name}: {e}")
                #     output_tensor_struct = None

            if output_tensor_struct is None:
                # print(f"DEBUG: maxpr_all returned None for {op.name}")
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
                out_impl = ref()
                if out_impl is not None and produced_impl is not None:
                    out_impl._values = produced_impl._values
                    out_impl.values_epoch = GRAPH.epoch

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

    original_traced: dict[int, bool] = {}
    for t in input_tensors:
        original_traced[id(t)] = t.traced
        t.traced = True

    try:

        outputs = fn(*args, **kwargs)

        traced = Trace(args, outputs)
        traced.compute()
        return traced
    finally:

        for t in input_tensors:
            if id(t) in original_traced:
                t.traced = original_traced[id(t)]


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
        try:
            dtype = "?"
            if hasattr(node, "dtype") and node.dtype:
                dtype = str(node.dtype)
            elif hasattr(node, "_storages") and node._storages:
                dtype = str(node._storages[0].dtype)
            elif hasattr(node, "_values") and node._values:
                dtype = str(node._values[0].type.dtype)
            return (
                dtype.lower()
                .replace("dtype.", "")
                .replace("float", "f")
                .replace("int", "i")
            )
        except Exception:
            return "?"

    def _format_shape_part(self, shape: tuple | list, batch_dims: int = 0) -> str:
        """Format a shape tuple with batch dims colors: [2, 3 | 4, 5]"""
        if shape is None:
            return "[?]"

        clean = [int(d) if hasattr(d, "__int__") else str(d) for d in shape]

        if batch_dims > 0 and batch_dims <= len(clean):
            batch_part = clean[:batch_dims]
            logical_part = clean[batch_dims:]

            b_str = str(batch_part).replace("[", "").replace("]", "")
            l_str = str(logical_part).replace("[", "").replace("]", "")

            if logical_part:
                return f"[{C_BATCH}{b_str}{RESET} | {l_str}]"
            else:
                return f"[{C_BATCH}{b_str}{RESET}]"

        return str(clean).replace(" ", "")

    def _format_spec_factors(self, sharding: Any) -> str:
        """Format sharding factors: (<dp, tp>)"""
        if not sharding:
            return ""

        all_partial_axes = set()
        if hasattr(sharding, "partial_sum_axes"):
            all_partial_axes.update(sharding.partial_sum_axes)

        factors = []
        for dim in sharding.dim_specs:
            if dim.partial:
                all_partial_axes.update(dim.axes)

            if not dim.axes:
                factors.append("*")
            else:
                factors.append(", ".join(dim.axes))

        partial_sum_str = ""
        if all_partial_axes:
            ordered_partial = sorted(list(all_partial_axes))
            axes_joined = ", ".join(f"'{a}'" for a in ordered_partial)
            partial_sum_str = f" | partial={{{axes_joined}}}"

        return f"(<{', '.join(factors)}>{partial_sum_str})"

    def _format_full_info(self, node: TensorImpl) -> str:
        """Format: dtype[global](factors)(local=[local])"""
        dtype = self._format_type(node)
        batch_dims = getattr(node, "batch_dims", 0)

        local_shape = node.physical_local_shape(0)
        local_str = self._format_shape_part(local_shape, batch_dims)

        global_str = "[?]"
        factors_str = ""

        if node.sharding:
            factors_str = self._format_spec_factors(node.sharding)
            try:

                if local_shape is not None:
                    g_shape = compute_global_shape(tuple(local_shape), node.sharding)
                    global_str = self._format_shape_part(g_shape, batch_dims)
            except Exception:
                pass
        else:

            global_str = local_str

        if node.sharding:
            return f"{dtype}{global_str}{factors_str}(local={local_str})"
        else:
            return f"{dtype}{global_str}"

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
                    ordered_p = sorted(list(all_p_axes))
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
        all_nodes = [
            t._impl for t in tree_leaves(self.trace.inputs) if isinstance(t, Tensor)
        ] + list(self.trace.nodes)
        for node in all_nodes:
            if hasattr(node, "sharding") and node.sharding and node.sharding.mesh:
                m = node.sharding.mesh
                if m.name not in meshes_used:
                    meshes_used[m.name] = m

        fn_mesh_str = ""
        if len(meshes_used) == 1:
            fn_mesh_str = f" {self._format_mesh_def(list(meshes_used.values())[0])}"

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
            lines.append(f"){fn_mesh_str} {{")
        else:
            lines.append(
                f"{C_KEYWORD}fn{RESET}({', '.join(input_vars)}){fn_mesh_str} {{ "
            )

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

            def collect_arg_names(x):
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


__all__ = ["Trace", "trace", "OutputRefs"]
