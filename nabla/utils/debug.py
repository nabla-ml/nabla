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

"""Debugging utilities for visualizing computation graphs and sharding."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ..core import pytree
from ..core.tensor import Tensor
from ..core.trace import Trace

if TYPE_CHECKING:
    from ..core.tensor_impl import TensorImpl

# Communication operations that define context boundaries
COMM_OPS = {"shard", "all_gather", "all_reduce", "reduce_scatter", "gather_all_axes", "reshard"}

# ANSI color codes for terminal output
LIGHT_PURPLE = "\033[94m"  # Batch dims
PURPLE = "\033[95m"        # Regular dims/type
CYAN = "\033[96m"          # Mesh/sharding
RESET = "\033[0m"


def capture_trace(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Trace:
    """Execute a function and capture its computation graph as a Trace.

    Args:
        fn: The function to trace.
        *args: Positional arguments (pytrees) passed to the function.
        **kwargs: Keyword arguments (pytrees) passed to the function.

    Returns:
        A Trace object containing the captured graph.
        
    Note:
        This marks input tensors as traced IN-PLACE for the duration of the call.
        The traced flag is restored after tracing completes.
    """
    flat_args = pytree.tree_leaves(args)
    flat_kwargs = pytree.tree_leaves(kwargs)
    all_inputs = flat_args + flat_kwargs
    
    original_traced: dict[int, bool] = {}
    for arg in all_inputs:
        if isinstance(arg, Tensor):
            original_traced[id(arg)] = arg.traced
            arg._impl.traced = True

    try:
        outputs = fn(*args, **kwargs)
        trace = Trace(args, outputs)
        return trace
    finally:
        for arg in all_inputs:
            if isinstance(arg, Tensor) and id(arg) in original_traced:
                arg._impl.traced = original_traced[id(arg)]


def xpr(current_node: Any) -> str:
    """Pretty-print the computation graph leading to the given node(s).

    Args:
        current_node: A Tensor or Pytree of Tensors.

    Returns:
        String representation of the computation graph.
    """
    leaves = [t for t in pytree.tree_leaves(current_node) if isinstance(t, Tensor)]
    
    if not leaves:
        return "xpr: No tensors found to trace."

    trace = Trace(inputs=[], outputs=current_node)
from nabla.sharding.spmd import compute_global_shape

class GraphPrinter:
    """Visualizes a Trace with a block-based, rich format.
    
    Format:
    # Meshes:
    #   mesh = [shape=(2,2), devices=[...], axes=(dp, tp)]
    
    fn(
        a: f32[4,4](<dp, tp>)(local=[2,2]),
        ...
    ) {
        v1: f32[4,4](<*, tp>)(local=[4,2]) = all_gather(a, axis=0)
        
        spmd @mesh(shape=(2,2), devices=[...], axes=(dp, tp)) {
            v5: f32[4,4](<*, *>)(local=[4,4]) = add(v2, v4)
        }
    }
    """

    def __init__(self, trace: Trace):
        self.trace = trace
        self.var_names: dict[int, str] = {}
        self.name_counters: dict[str, int] = {}
    
    def _get_next_name(self, prefix: str = "v") -> str:
        if prefix not in self.name_counters:
            self.name_counters[prefix] = 0
        self.name_counters[prefix] += 1
        return f"{prefix}{self.name_counters[prefix]}"

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
                return f"[{LIGHT_PURPLE}{b_str}{RESET} | {l_str}]"
            else:
                return f"[{LIGHT_PURPLE}{b_str}{RESET}]"
        
        return str(clean).replace(" ", "")

    def _format_spec_factors(self, sharding: Any) -> str:
        """Format sharding factors: (<dp, tp>)"""
        if not sharding:
            return ""
        
        factors = []
        for dim in sharding.dim_specs:
            if not dim.axes:
                factors.append("*")
            else:
                factors.append(", ".join(dim.axes))
        
        return f"(<{', '.join(factors)}>)"

    def _format_full_info(self, node: TensorImpl) -> str:
        """Format: dtype[global](factors)(local=[local])"""
        dtype = self._format_type(node)
        batch_dims = getattr(node, 'batch_dims', 0)
        
        # Local Shape
        local_shape = node.physical_shape
        local_str = self._format_shape_part(local_shape, batch_dims)
        
        # Global Shape & Factors
        global_str = "[?]"
        factors_str = ""
        
        if node.sharding:
            factors_str = self._format_spec_factors(node.sharding)
            try:
                # Infer global shape
                # Note: node.physical_shape is already a tuple/Shape from TensorImpl
                if local_shape is not None:
                     # Calculate global shape tuple
                     # compute_global_shape returns a tuple
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
        
        # Format devices concisely if possible
        devs = mesh.devices
        if len(devs) > 8:
            dev_str = f"[{devs[0]}..{devs[-1]}]"
        else:
            dev_str = str(devs).replace(" ", "")
            
        return f"@{mesh.name}(shape={shape_str}, devices={dev_str}, axes={axes_str})"

    def _format_kwargs(self, kwargs: dict | None) -> str:
        if not kwargs: return ""
        parts = []
        for k, v in kwargs.items():
            if k in ('shard_idx', 'mesh', 'dim_specs', 'spec'): continue
            if isinstance(v, (list, tuple)):
                v_clean = tuple(int(x) if hasattr(x, '__int__') else x for x in v)
                v_str = str(v_clean).replace(" ", "")
            elif hasattr(v, 'name'): v_str = f"@{v.name}"
            else: v_str = str(v)
            parts.append(f"{k}={v_str}")
        return ", ".join(parts)

    def to_string(self) -> str:
        if not self.trace._computed:
            self.trace.compute()
        
        lines = []
        
        # 0. Collect Meshes
        meshes_used: dict[str, Any] = {}
        all_nodes = [t._impl for t in pytree.tree_leaves(self.trace.inputs) if isinstance(t, Tensor)] + list(self.trace.nodes)
        for node in all_nodes:
            if hasattr(node, 'sharding') and node.sharding and node.sharding.mesh:
                m = node.sharding.mesh
                if m.name not in meshes_used:
                    meshes_used[m.name] = m

        if meshes_used:
            for m in meshes_used.values():
                lines.append(f"{self._format_mesh_def(m)}")
        
        # 1. Inputs
        input_leaves = [t._impl for t in pytree.tree_leaves(self.trace.inputs) if isinstance(t, Tensor)]
        input_vars = []
        for node in input_leaves:
            if id(node) not in self.var_names:
                name = self._get_next_name("a")
                self.var_names[id(node)] = name
            
            desc = f"{self.var_names[id(node)]}: {self._format_full_info(node)}"
            input_vars.append(desc)
        
        if len(input_vars) > 1:
            lines.append("fn(")
            for i, var in enumerate(input_vars):
                comma = "," if i < len(input_vars) - 1 else ""
                lines.append(f"    {var}{comma}")
            lines.append(") {")
        else:
             lines.append(f"fn({', '.join(input_vars)}) {{")

        # 2. Body - Group into SPMD blocks
        # We iterate linearly. If we see a Computation Op with sharding, it starts/continues a block.
        # Comm ops or unsharded ops break blocks.
        
        current_block_mesh = None
        block_lines = []
        
        def flush_block():
            nonlocal current_block_mesh, block_lines
            if current_block_mesh and block_lines:
                mesh_def = self._format_mesh_def(current_block_mesh)
                lines.append(f"  {mesh_def} spmd {{")
                lines.extend(block_lines)
                lines.append("  }")
            elif block_lines: # Unsharded block? usually just print lines
                lines.extend(block_lines)
            
            current_block_mesh = None
            block_lines = []

        for node in self.trace.nodes:
            if id(node) in self.var_names: continue
            
            name = self._get_next_name()
            self.var_names[id(node)] = name
            
            # Format Op
            op_name = node.op_name.lower() if node.op_name else "unknown"
            
            # Args
            arg_names = [self.var_names.get(id(p), "?") for p in node.parents]
            args_str = ", ".join(arg_names)
            kwargs_str = self._format_kwargs(node.op_kwargs)
            call_args = ", ".join(filter(None, [args_str, kwargs_str]))
            
            # Line
            info_str = self._format_full_info(node)
            line = f"    {name}: {info_str} = {op_name}({call_args})"
            
            is_sharded = node.sharding and node.sharding.mesh
            is_comm = op_name in COMM_OPS
            
            if is_comm:
                # Comm ops break blocks and are printed standalone
                flush_block()
                # Indent less for standalone
                lines.append(line.replace("    ", "  "))
            elif is_sharded:
                # Start or continue ID-based block check? 
                # Ideally check if mesh matches.
                mesh = node.sharding.mesh
                if current_block_mesh and current_block_mesh.name != mesh.name:
                    flush_block()
                
                current_block_mesh = mesh
                block_lines.append(line)
            else:
                 # Unsharded op - treat as standalone for now or generic block?
                 # User requested spmd blocks specifically for sharded.
                 flush_block()
                 lines.append(line.replace("    ", "  "))
        
        flush_block() # Flush remaining

        # 3. Return
        output_leaves = [t._impl for t in pytree.tree_leaves(self.trace.outputs) if isinstance(t, Tensor)]
        output_vars = [self.var_names.get(id(n), "?") for n in output_leaves]
        output_sig = ", ".join(output_vars) if output_vars else "()"
        
        if len(output_vars) == 1:
            lines.append(f"  return {output_sig}")
        else:
            lines.append(f"  return ({output_sig})")
            
        lines.append("}")
        return "\n".join(lines)
