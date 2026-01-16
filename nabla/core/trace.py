# ===----------------------------------------------------------------------=== #
# Nabla 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Core Trace primitive for capturing and manipulating computation subgraphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .pytree import tree_leaves
from . import pytree
from .tensor import Tensor
from .tensor_impl import TensorImpl
from ..sharding.spmd import compute_global_shape

if TYPE_CHECKING:
    from .tensor_impl import TensorImpl

# Communication operations that define context boundaries for printing
COMM_OPS = {"shard", "all_gather", "all_reduce", "reduce_scatter", "gather_all_axes", "reshard", "ppermute", "all_to_all", "pmean"}

# ANSI color codes for terminal output
# minimalist, readable color scheme:
# - Variables: Cyan
# - Keywords: Bold
# - Batch dimensions: Dark gray
RESET = "\033[0m"
C_VAR = "\033[96m"        # Cyan - variables (%v1, %a1)
C_KEYWORD = "\033[1m"     # Bold - fn, spmd, return
C_BATCH = "\033[90m"      # Dark gray - batch dimensions


class Trace:
    """Represents a captured computation subgraph.
    
    A Trace allows viewing and manipulating the computation graph between
    a set of input tensors (boundary) and output tensors.
    
    Use the `trace()` function to create traces - it handles all the setup.
    
    Attributes:
        inputs: The original input pytree structure.
        outputs: The original output pytree structure.
        nodes: Topologically sorted list of unique TensorImpls in the subgraph.
        
    Example:
        >>> t = trace(my_function, input_tensor)
        >>> print(t)  # Pretty-printed computation graph
    """
    
    def __init__(self, inputs: Any, outputs: Any):
        self.inputs = inputs
        self.outputs = outputs
        self._computed = False
        self.nodes: list[OutputRefs] = []
        
        # Flatten inputs to establish the boundary
        # We track input TENSORS, not refs, because inputs might not have output_refs (leaves)
        self._input_tensor_ids = {
            id(t._impl) for t in tree_leaves(inputs) if isinstance(t, Tensor)
        }
        
    def compute(self) -> None:
        """Compute the topological ordering of the subgraph (list of OutputRefs)."""
        if self._computed:
            return
            
        visited: set[int] = set()
        nodes: list[OutputRefs] = []
        
        # Get all output tensors
        output_leaves = [
            t._impl for t in tree_leaves(self.outputs) if isinstance(t, Tensor)
        ]
        
        # Identify "root" operations (those producing the outputs)
        root_refs: list[OutputRefs] = []
        for leaf in output_leaves:
            if leaf.output_refs is not None:
                root_refs.append(leaf.output_refs)
            # If leaf has no output_refs, it's a constant or input leaf, so no op to trace.
        
        def dfs(refs: OutputRefs) -> None:
            refs_id = id(refs)
            if refs_id in visited:
                return
            
            # Recurse into parents (arguments to this op)
            # Iterate through args to find tensor inputs
            arg_leaves = []
            for arg in tree_leaves(refs.op_args):
                if isinstance(arg, Tensor):
                    arg_leaves.append(arg._impl)
                elif isinstance(arg, TensorImpl):
                     arg_leaves.append(arg)
            
            for arg in arg_leaves:
                # If arg is an input to the trace, stop.
                if id(arg) in self._input_tensor_ids:
                    continue
                
                # If arg was produced by an op, recurse
                if arg.output_refs:
                    dfs(arg.output_refs)
            
            visited.add(refs_id)
            nodes.append(refs)
            
        for root in root_refs:
            dfs(root)
            
        self.nodes = nodes
        self._computed = True
        
    def __str__(self) -> str:
        """Pretty-print the trace using the GraphPrinter."""
        if not self._computed:
            self.compute()
        return GraphPrinter(self).to_string()
    
    def __repr__(self) -> str:
        n_inputs = len([t for t in tree_leaves(self.inputs) if isinstance(t, Tensor)])
        n_outputs = len([t for t in tree_leaves(self.outputs) if isinstance(t, Tensor)])
        n_nodes = len(self.nodes) if self._computed else "?"
        return f"Trace(inputs={n_inputs}, outputs={n_outputs}, nodes={n_nodes})"


def trace(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Trace:
    """Trace a function's computation graph.
    
    This is the main entry point for debugging and inspecting computation graphs.
    It marks input tensors as traced, executes the function, and captures the
    resulting computation graph.
    
    Args:
        fn: The function to trace.
        *args: Positional arguments (pytrees) passed to the function.
        **kwargs: Keyword arguments (pytrees) passed to the function.
        
    Returns:
        A Trace object with the computed graph nodes.
    """
    # Collect all input tensor leaves
    flat_args = tree_leaves(args)
    flat_kwargs = tree_leaves(kwargs)
    all_inputs = flat_args + flat_kwargs
    input_tensors = [t for t in all_inputs if isinstance(t, Tensor)]
    
    if not input_tensors:
        raise ValueError("trace: No input tensors found. Pass at least one Tensor.")
    
    # Save original traced state and mark as traced
    original_traced: dict[int, bool] = {}
    for t in input_tensors:
        original_traced[id(t)] = t.traced
        t._impl.traced = True
    
    try:
        # Execute function
        outputs = fn(*args, **kwargs)
        
        # Build trace from inputs to outputs
        traced = Trace(args, outputs)
        traced.compute()
        return traced
    finally:
        # Restore original traced state
        for t in input_tensors:
            if id(t) in original_traced:
                t._impl.traced = original_traced[id(t)]



class GraphPrinter:
    """Visualizes a Trace with a block-based, rich format."""

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
            if hasattr(node, "cached_dtype") and node.cached_dtype:
                dtype = str(node.cached_dtype)
            elif hasattr(node, "_storages") and node._storages:
                dtype = str(node._storages[0].dtype)
            elif hasattr(node, "_values") and node._values:
                dtype = str(node._values[0].type.dtype)
            return dtype.lower().replace("dtype.", "").replace("float", "f").replace("int", "i")
        except Exception:
            return "?"

    def _format_shape_part(self, shape: tuple | list, batch_dims: int = 0) -> str:
        """Format a shape tuple with batch dims colors: [2, 3 | 4, 5]"""
        if shape is None:
            return "[?]"
        
        clean = [int(d) if hasattr(d, '__int__') else str(d) for d in shape]
        
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
        if hasattr(sharding, 'partial_sum_axes'):
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
        batch_dims = getattr(node, 'batch_dims', 0)
        
        # Local Shape
        local_shape = node.physical_local_shape(0)
        local_str = self._format_shape_part(local_shape, batch_dims)
        
        # Global Shape & Factors
        global_str = "[?]"
        factors_str = ""
        
        if node.sharding:
            factors_str = self._format_spec_factors(node.sharding)
            try:
                # Infer global shape from local shape and sharding spec
                if local_shape is not None:
                     g_shape = compute_global_shape(tuple(local_shape), node.sharding)
                     global_str = self._format_shape_part(g_shape, batch_dims)
            except Exception:
                pass
        else:
             # Unsharded: global = local
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
        if not kwargs: return ""
        parts = []
        # Sort keys for deterministic output
        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            # Skip internal tracking fields only
            if k in ('shard_idx',): continue
            
            # Format mesh references nicely
            if k == 'mesh' and hasattr(v, 'name'):
                parts.append(f"mesh=@{v.name}")
            # Format dim_specs as sharding notation
            elif k == 'dim_specs' and isinstance(v, (list, tuple)):
                factors = []
                all_p_axes = set()
                for dim in v:
                    if getattr(dim, "partial", False):
                        all_p_axes.update(dim.axes)
                    
                    if hasattr(dim, 'axes'):
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
            # Skip redundant spec kwarg (already shown in output type)
            elif k == 'spec':
                continue
            elif isinstance(v, (list, tuple)):
                v_clean = tuple(int(x) if hasattr(x, '__int__') else x for x in v)
                v_str = str(v_clean).replace(" ", "")
                parts.append(f"{k}={v_str}")
            elif hasattr(v, 'name'):
                parts.append(f"{k}=@{v.name}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)

    def to_string(self) -> str:
        if not self.trace._computed:
            self.trace.compute()
        
        lines = []
        
        # 0. Collect Meshes
        meshes_used: dict[str, Any] = {}
        all_nodes = [t._impl for t in tree_leaves(self.trace.inputs) if isinstance(t, Tensor)] + list(self.trace.nodes)
        for node in all_nodes:
            if hasattr(node, 'sharding') and node.sharding and node.sharding.mesh:
                m = node.sharding.mesh
                if m.name not in meshes_used:
                    meshes_used[m.name] = m

        # Get mesh string for function header (if exactly one mesh is used)
        fn_mesh_str = ""
        if len(meshes_used) == 1:
            fn_mesh_str = f" {self._format_mesh_def(list(meshes_used.values())[0])}"
        
        # 1. Inputs
        input_leaves = [t._impl for t in tree_leaves(self.trace.inputs) if isinstance(t, Tensor)]
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
             lines.append(f"{C_KEYWORD}fn{RESET}({', '.join(input_vars)}){fn_mesh_str} {{")

        # 2. Body - Group into SPMD blocks
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
            # refs is OutputRefs
            op_name = refs.op.name.lower() if hasattr(refs.op, 'name') else str(type(refs.op).__name__)
            
            # Helper to find parent variable names
            arg_names = []
            def collect_arg_names(x):
                if isinstance(x, Tensor):
                     arg_names.append(f"{C_VAR}{self.var_names.get(id(x._impl), '?')}{RESET}")
                elif isinstance(x, TensorImpl):
                     arg_names.append(f"{C_VAR}{self.var_names.get(id(x), '?')}{RESET}")
                return x
            pytree.tree_map(collect_arg_names, refs.op_args)
            
            args_str = ", ".join(arg_names)
            kwargs_str = self._format_kwargs(refs.op_kwargs)
            
             # Fallback for communication ops on constants
            if not args_str and op_name in ('shard', 'reshard'):
                args_str = f"{C_VAR}const{RESET}"

            call_args = ", ".join(filter(None, [args_str, kwargs_str]))
            
            # Outputs
            outputs = refs.get_alive_outputs()
            valid_outputs = [o for o in outputs if o is not None]
            
            # Skip if no outputs are alive (dead code)
            if not valid_outputs:
                continue

            # Assign names and format info
            lhs_parts = []
            is_sharded = False
            first_valid = valid_outputs[0]
            if first_valid.sharding and first_valid.sharding.mesh:
                is_sharded = True
            
            # Determine mesh (use first valid output)
            mesh = first_valid.sharding.mesh if is_sharded and first_valid.sharding else None

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

        # 3. Return
        output_leaves = [t._impl for t in tree_leaves(self.trace.outputs) if isinstance(t, Tensor)]
        output_vars = [f"{C_VAR}{self.var_names.get(id(n), '?')}{RESET}" for n in output_leaves]
        output_sig = ", ".join(output_vars) if output_vars else "()"
        
        if len(output_vars) == 1:
            lines.append(f"  {C_KEYWORD}return{RESET} {output_sig}")
        else:
            lines.append(f"  {C_KEYWORD}return{RESET} ({output_sig})")
            
        lines.append("}")
        return "\n".join(lines)


__all__ = ["Trace", "trace"]
