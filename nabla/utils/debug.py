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
    return str(trace)


class GraphPrinter:
    """Visualizes a Trace with Option C block-based formatting.
    
    Features:
    - Block-based context visualization (@CPU, @mesh)
    - Communication ops as block boundaries
    - Full kwargs display
    - Input sharding annotations
    - No empty blocks
    """

    def __init__(self, trace: Trace):
        self.trace = trace
        self.var_names: dict[int, str] = {}
        self.name_counter = 0
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"

    def _get_next_name(self) -> str:
        if self.name_counter < len(self.alphabet):
            name = self.alphabet[self.name_counter]
        else:
            idx = self.name_counter - len(self.alphabet)
            first = idx // len(self.alphabet)
            second = idx % len(self.alphabet)
            name = self.alphabet[first] + self.alphabet[second]
        self.name_counter += 1
        return name

    def _format_type(self, impl: TensorImpl, use_color: bool = True) -> str:
        """Format: f32[**batch_dims**|logical_dims] with markers for batch dims.
        
        Uses physical_shape and splits at batch_dims position.
        """
        try:
            dtype = "?"
            if impl.cached_dtype:
                dtype = str(impl.cached_dtype)
            elif impl._storages:
                dtype = str(impl._storages[0].dtype)
            elif impl._values:
                dtype = str(impl._values[0].type.dtype)
            dtype = dtype.lower().replace("dtype.", "").replace("float", "f").replace("int", "i")
            
            # Use physical shape (includes batch dims as prefix)
            shape = impl.physical_shape
            batch_dims = impl.batch_dims or 0
            
            if shape is None:
                return f"{dtype}[?]"
            
            # Convert to list of ints
            dims = [int(d) if hasattr(d, '__int__') else str(d) for d in shape]
            
            if batch_dims > 0 and use_color:
                # Split into batch and regular dims
                batch_part = dims[:batch_dims]
                regular_part = dims[batch_dims:]
                
                batch_str = ", ".join(str(d) for d in batch_part)
                regular_str = ", ".join(str(d) for d in regular_part)
                
                # Color: batch dims in light purple, separator, regular in default
                if regular_part:
                    shape_str = f"{LIGHT_PURPLE}{batch_str}{RESET}|{regular_str}"
                else:
                    shape_str = f"{LIGHT_PURPLE}{batch_str}{RESET}"
                return f"{dtype}[{shape_str}]"
            else:
                shape_str = ", ".join(str(d) for d in dims)
                return f"{dtype}[{shape_str}]"
        except Exception:
            return "?[?]"

    def _format_input_type(self, impl: TensorImpl) -> str:
        """Format input with sharding: f32[4, 4]@mesh[spec] or f32[4, 4]"""
        base = self._format_type(impl)
        if impl.sharding and impl.sharding.mesh:
            spec_str = self._format_sharding_spec_compact(impl.sharding)
            return f"{base}@{impl.sharding.mesh.name}{spec_str}"
        return base

    def _format_output_type(self, impl: TensorImpl) -> str:
        """Format output with sharding annotation if different from CPU.
        
        Shows sharding spec for visibility into propagation.
        """
        base = self._format_type(impl)
        if impl.sharding and impl.sharding.mesh:
            spec_str = self._format_sharding_spec_compact(impl.sharding)
            return f"{base}@{spec_str}"
        return base

    def _format_sharding_spec_compact(self, sharding: Any) -> str:
        """Compact spec: [{\"x\"}, {}]"""
        if not sharding:
            return ""
        dims = ", ".join(str(d) for d in sharding.dim_specs)
        return f"[{dims}]"

    def _format_kwargs(self, kwargs: dict | None) -> str:
        """Format kwargs as key=value pairs."""
        if not kwargs:
            return ""
        parts = []
        for k, v in kwargs.items():
            # Skip internal kwargs
            if k in ('shard_idx', 'mesh', 'dim_specs'):
                continue
            if isinstance(v, (list, tuple)):
                v_clean = tuple(int(x) if hasattr(x, '__int__') else x for x in v)
                v_str = str(v_clean).replace(" ", "")
            elif hasattr(v, 'name'):  # For objects with name attr like DeviceMesh
                v_str = f"@{v.name}"
            else:
                v_str = str(v)
            parts.append(f"{k}={v_str}")
        return " ".join(parts)

    def _format_mesh(self, sharding: Any) -> str:
        """Format mesh info: @name shape=(2,4) axes=("x","y") devices=[0,1,2,3]"""
        if not sharding or not sharding.mesh:
            return ""
        mesh = sharding.mesh
        axes_str = str(mesh.axis_names).replace(" ", "")
        return f"@{mesh.name} shape={mesh.shape} axes={axes_str} devices={mesh.devices}"

    def _is_comm_op(self, op_name: str | None) -> bool:
        return op_name in COMM_OPS if op_name else False

    def _get_context_name(self, node: TensorImpl) -> str:
        """Get context name for a node (@CPU or @mesh_name)."""
        if node.sharding and node.sharding.mesh:
            return f"@{node.sharding.mesh.name}"
        return "@CPU"

    def to_string(self) -> str:
        if not self.trace._computed:
            self.trace.compute()
        
        # 1. Name and format inputs
        input_leaves = [t._impl for t in pytree.tree_leaves(self.trace.inputs) if isinstance(t, Tensor)]
        input_vars = []
        for node in input_leaves:
            if id(node) not in self.var_names:
                name = self._get_next_name()
                self.var_names[id(node)] = name
            input_vars.append(f"{self.var_names[id(node)]}:{self._format_input_type(node)}")
        
        # 2. Pre-scan to determine first context
        first_context = "@CPU"
        for node in self.trace.nodes:
            if id(node) not in self.var_names:
                first_context = self._get_context_name(node)
                break
        
        # 3. Build output with blocks
        lines = []
        current_context = first_context
        block_ops: list[str] = []
        indent = "    "
        
        for node in self.trace.nodes:
            if id(node) in self.var_names:
                continue
            
            op_name = node.op_name or "unknown"
            is_comm = self._is_comm_op(op_name)
            
            if is_comm:
                # Flush current block if non-empty
                if block_ops:
                    lines.append(f"  {{ {current_context}")
                    lines.extend(block_ops)
                    lines.append("  }")
                    lines.append("")  # Empty line after block
                    block_ops = []
                
                # Assign name
                name = self._get_next_name()
                self.var_names[id(node)] = name
                
                # Format comm op
                kwargs_str = self._format_kwargs(node.op_kwargs)
                mesh_str = self._format_mesh(node.sharding)
                spec_str = self._format_sharding_spec_compact(node.sharding) if node.sharding else ""
                
                arg_names = [self.var_names.get(id(p), "?") for p in node.parents]
                args_str = ", ".join(arg_names) if arg_names else ""
                
                comm_line = f"  {op_name.upper()}({args_str}"
                if mesh_str:
                    comm_line += f", {mesh_str}"
                if spec_str:
                    comm_line += f", spec={spec_str}"
                if kwargs_str:
                    comm_line += f", {kwargs_str}"
                comm_line += f") -> {name}:{self._format_type(node)}"
                
                lines.append(comm_line)
                lines.append("")  # Empty line after comm op
                
                # Update context for next block
                current_context = self._get_context_name(node)
            else:
                # Regular op: accumulate
                name = self._get_next_name()
                self.var_names[id(node)] = name
                
                arg_names = [self.var_names.get(id(p), "?") for p in node.parents]
                args_str = " ".join(arg_names)
                kwargs_str = self._format_kwargs(node.op_kwargs)
                
                if args_str and kwargs_str:
                    full_args = f"{args_str} {kwargs_str}"
                elif kwargs_str:
                    full_args = kwargs_str
                else:
                    full_args = args_str
                
                line = f"{indent}{name}:{self._format_output_type(node)} = {op_name} {full_args}"
                block_ops.append(line)
        
        # 4. Close final block with output
        output_leaves = [t._impl for t in pytree.tree_leaves(self.trace.outputs) if isinstance(t, Tensor)]
        output_vars = [self.var_names.get(id(n), "?") for n in output_leaves]
        output_sig = ", ".join(output_vars) if output_vars else "()"
        
        if block_ops:
            lines.append(f"  {{ {current_context}")
            lines.extend(block_ops)
            lines.append(f"  }}")
        
        # 5. Build final output with proper structure
        result_lines = []
        
        # Header with inputs
        if input_vars:
            result_lines.append(f"fn({', '.join(input_vars)}) {{")
        else:
            result_lines.append("fn() {")
        
        # Body
        result_lines.extend(lines)
        
        # Return statement
        if len(output_vars) == 1:
            result_lines.append(f"  return {output_sig}")
        else:
            result_lines.append(f"  return ({output_sig})")
        
        # Closing brace
        result_lines.append("}")
        
        return "\n".join(result_lines)
