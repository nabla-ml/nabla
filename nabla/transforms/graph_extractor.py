# ===----------------------------------------------------------------------=== #
# Nabla 2026
# ===----------------------------------------------------------------------=== #

"""Graph Extractor: Converts Logical Trace to Unsharded Graph JSON."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.tensor import Tensor
from ..core.tensor_impl import TensorImpl
from ..core.trace import Trace
from ..ops.operation import Operation

class ShardingGraphExtractor:
    """Extracts a JSON graph representation from a logical trace for sharding optimization."""

    def __init__(self, trace: Trace, in_specs: Dict[int, Any], out_specs: Optional[Dict[int, Any]] = None, debug: bool = False):
        self.trace = trace
        self.in_specs = in_specs
        self.out_specs = out_specs or {}
        self.debug = debug
        
        self.tensors: List[Dict[str, Any]] = []
        self.nodes: List[Dict[str, Any]] = []
        self.tensor_id_map: Dict[int, int] = {}  # id(TensorImpl) -> json_id
        self.id_to_tensor_impl: Dict[int, TensorImpl] = {} # json_id -> TensorImpl
        self.next_tensor_id = 0

    def _get_or_create_tensor_id(self, tensor_impl: TensorImpl) -> int:
        if id(tensor_impl) not in self.tensor_id_map:
            json_id = self.next_tensor_id
            self.next_tensor_id += 1
            self.tensor_id_map[id(tensor_impl)] = json_id
            self.id_to_tensor_impl[json_id] = tensor_impl
            
            # Extract tensor metadata
            shape = tensor_impl.global_shape
            if shape is None:
                shape = tensor_impl.physical_shape
            
            shape_tuple = tuple(int(d) for d in shape) if shape else ()
            
            # Determine if this tensor has a fixed sharding constraint
            fixed_sharding = None
            if tensor_impl.sharding_constraint:
                spec = tensor_impl.sharding_constraint
                # Basic serialization of spec (we might need more detail later)
                fixed_sharding = {
                    "dims": [d.axes if d.axes else None for d in spec.dim_specs],
                    "replicated": list(spec.replicated_axes)
                }

            self.tensors.append({
                "id": json_id,
                "shape": shape_tuple,
                "dtype": str(tensor_impl.cached_dtype) if tensor_impl.cached_dtype else "float32",
                "size_bytes": 0,  # Placeholder
                "fixed_sharding": fixed_sharding
            })
            
        return self.tensor_id_map[id(tensor_impl)]

    def extract(self) -> str:
        """Run extraction and return JSON string."""
        if not self.trace._computed:
            self.trace.compute()

        # 1. Register Inputs
        # Flatten input args to match in_specs indices
        from ..core import pytree
        flat_args, _ = pytree.tree_flatten(self.trace.inputs)
        for i, val in enumerate(flat_args):
            if isinstance(val, Tensor):
                tid = self._get_or_create_tensor_id(val._impl)
                # Apply input constraint if provided
                if i in self.in_specs:
                    spec = self.in_specs[i]
                    if spec:
                        # Update the tensor definition with this constraint
                        # Note: trace inputs might already have constraints on them, 
                        # but in_specs passed to shard_map are the "boundary conditions".
                        # We merge or overwrite.
                        self.tensors[tid]["fixed_sharding"] = {
                            "dims": [d.axes if d.axes else None for d in spec.dim_specs],
                            "replicated": list(spec.replicated_axes)
                        }

        # 2. Process Nodes
        for i, refs in enumerate(self.trace.nodes):
            op: Operation = refs.op
            
            # Inputs
            input_ids = []
            input_shapes = []
            
            def collect_inputs(x):
                if isinstance(x, Tensor):
                    input_ids.append(self._get_or_create_tensor_id(x._impl))
                    input_shapes.append(tuple(int(d) for d in x.shape))
                elif isinstance(x, TensorImpl):
                    input_ids.append(self._get_or_create_tensor_id(x))
                    input_shapes.append(tuple(int(d) for d in x.global_shape or x.physical_shape))
            
            pytree.tree_map(collect_inputs, refs.op_args)
            if refs.op_kwargs:
                pytree.tree_map(collect_inputs, refs.op_kwargs)

            # Outputs
            output_ids = []
            output_shapes = []
            outputs = refs.get_alive_outputs()
            valid_outputs = [o for o in outputs if o is not None]
            
            for out in valid_outputs:
                output_ids.append(self._get_or_create_tensor_id(out))
                output_shapes.append(tuple(int(d) for d in out.global_shape or out.physical_shape))

            # Sharding Rule & Cost
            rule_info = None
            try:
                # Instantiate rule to get factors
                # Note: This might fail if rule generation is complex or args missing.
                # We assume simple rules for now.
                rule = op.sharding_rule(input_shapes, output_shapes, **(refs.op_kwargs or {}))
                if rule:
                    rule_info = {
                        "equation": rule.to_einsum_notation(),
                        "factor_sizes": rule.factor_sizes
                    }
            except Exception as e:
                # Log the error in debug mode
                if self.debug:
                    print(f"[GraphExtractor] WARNING: Failed to get sharding_rule for node {i} [{op.name}]: {e}")

            cost = op.cost_model(input_shapes, output_shapes)

            self.nodes.append({
                "id": i,
                "op_name": op.name,
                "inputs": input_ids,
                "outputs": output_ids,
                "sharding_rule": rule_info,
                "compute_stats": {
                    "flops": cost
                }
            })
            
        # 3. Register Outputs (Constraints)
        flat_outs, _ = pytree.tree_flatten(self.trace.outputs)
        for i, val in enumerate(flat_outs):
             if i in self.out_specs and isinstance(val, Tensor):
                 tid = self._get_or_create_tensor_id(val._impl)
                 spec = self.out_specs[i]
                 # We need to enforce this constraint on the tensor produced by the last node
                 self.tensors[tid]["fixed_sharding"] = {
                    "dims": [d.axes if d.axes else None for d in spec.dim_specs],
                    "replicated": list(spec.replicated_axes)
                 }

        # Build final JSON
        graph_data = {
            "meta": {
                # Placeholder for mesh info - passed separately to solver usually, 
                # but good to have if we knew it here from in_specs[0].mesh
                "mesh": {} 
            },
            "tensors": self.tensors,
            "nodes": self.nodes
        }
        
        json_output = json.dumps(graph_data, indent=2)
        
        if self.debug:
            print("\n[AutoSharding] Extracted Graph JSON:")
            print(json_output)
            print("-" * 50)
            
        return json_output
