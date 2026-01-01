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

"""ComputeGraph: Manages lazy evaluation and graph compilation."""

from __future__ import annotations

import gc
import sys
import weakref
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Iterable

from max import _core, driver, graph, mlir
from max._core.dialects import builtin, kgen, mo
from max.graph import Value, ops
from max.graph.graph import _location

if TYPE_CHECKING:
    from .tensor import Tensor
    from .sharding import DeviceMesh
    from .tensor_impl import TensorImpl

from .context import _session

# =============================================================================
# 1. Global State & Constants
# =============================================================================

_GRAPH_EPOCH: int = 0
_SEED: ContextVar[Tensor] = ContextVar("_SEED")

import os

# Debug flag for lazy evaluation - prints graph outputs before compilation
# Set NABLA_DEBUG=1 to enable
DEBUG_LAZY_EVAL: bool = os.getenv("NABLA_DEBUG", "0") == "1"

def seed() -> Tensor:
    """Returns the global random seed tensor, initializing if necessary."""
    from .tensor import Tensor
    if (s := _SEED.get(None)) is None:
        s = driver.Tensor(ops.random.SeedType)
        s[0] = 0
        _SEED.set(Tensor(storage=s))
    return _SEED.get()

def driver_tensor_type(t: driver.Tensor) -> graph.TensorType:
    """Converts a driver tensor to a TensorType."""
    return graph.TensorType(t.dtype, t.shape, graph.DeviceRef.from_device(t.device))

# =============================================================================
# 2. Graph Algorithms (Static Helpers)
# =============================================================================

def _collect_execution_graph(
    unrealized: Iterable[Tensor],
) -> tuple[list[TensorImpl], list[TensorImpl]]:
    """Topologically sorts the subgraph required to compute 'unrealized'."""
    op_impls: list[TensorImpl] = []
    leaf_impls: list[TensorImpl] = []
    visited: set[int] = set()

    def _visit(impl: TensorImpl) -> None:
        if id(impl) in visited:
            return
        visited.add(id(impl))

        if impl.is_realized or not impl.parents:
            leaf_impls.append(impl)
            return

        for parent in impl.parents:
            _visit(parent)
        op_impls.append(impl)

    for t in unrealized:
        _visit(t._impl)
    return op_impls, leaf_impls


def _needs_sharded_compilation(
    unrealized: Iterable[Tensor],
) -> tuple[bool, list[TensorImpl], list[TensorImpl]]:
    """Determines if the graph requires sharded compilation strategies."""
    op_impls, leaf_impls = _collect_execution_graph(unrealized)
    
    if not op_impls:
        return False, [], []

    # Check for sharding annotations
    all_impls = op_impls + leaf_impls
    if not any(impl.sharding is not None for impl in all_impls):
        return False, [], []

    # Verify tracing
    if untraced := [impl for impl in op_impls if not impl.traced]:
        raise ValueError(
            f"Sharded compilation requires traced tensors. "
            f"Found {len(untraced)} untraced operation(s)."
        )

    return True, op_impls, leaf_impls


def _remove_unused_arguments(g: graph.Graph) -> None:
    """Optimizes the graph by removing input arguments that are never used."""
    op = _core.Operation._from_cmlir(g._mlir_op)
    assert isinstance(op, mo.GraphOp)

    block = op.regions[0].front
    for i, inp in reversed(list(enumerate(g.inputs))):
        if not inp._mlir_value.num_uses:
            block.erase_argument(i)

    g.inputs = [Value.from_mlir(arg) for arg in block.arguments]

    with g:
        op.function_type = builtin.FunctionType(
            [inp.type.to_mlir() for inp in g.inputs],
            op.function_type.results,
        )
        op.signature = kgen.FuncTypeGeneratorType([], op.function_type)
        op.discardable_attributes["argument_names"] = builtin.ArrayAttr(
            [builtin.StringAttr(f"input{i}") for i in range(len(g.inputs))]
        )

# =============================================================================
# 3. The Compute Engine
# =============================================================================


class ComputeGraph:
    """Manages the DAG of operations, lazy evaluation, and compilation."""

    graph: graph.Graph
    sources: dict[_core.Value[Any], driver.Tensor]  # Maps buffer values to their storages
    unrealized: weakref.WeakValueDictionary[int, Tensor]
    epoch: int

    # --- Lifecycle ---

    def __init__(self, context: mlir.Context | None = None, seed: int = 0):
        global _GRAPH_EPOCH
        _GRAPH_EPOCH += 1
        self.epoch = _GRAPH_EPOCH
        self._reset(context, seed)

    def _reset(self, context: mlir.Context | None, seed: int) -> None:
        """Resets the internal graph state for a new compilation cycle."""
        self.context = context or mlir.Context()
        self.sources = {}
        self.unrealized = weakref.WeakValueDictionary()
        self.graph = graph.Graph("main", input_types=[], context=self.context)
        with self.graph:
            ops.random.set_seed(seed)

    # --- Public API ---

    def add_input(self, tensor: Tensor) -> None:
        """Registers a realized tensor's storages as graph inputs.
        
        Handles both single-value and multi-shard tensors uniformly.
        All storages become graph inputs, and corresponding TensorValues
        are stored back in the tensor impl.
        """
        storages = tensor._impl._storages
        if not storages:
            raise TypeError("Only realized tensors may be graph inputs.")

        op = _core.Operation._from_cmlir(self.graph._mlir_op)
        assert isinstance(op, mo.GraphOp)
        block = op.regions[0].front
        
        tensor_values = []
        for storage in storages:
            with self.graph:
                tensor_type = graph.TensorType(
                    storage.dtype, storage.shape,
                    graph.DeviceRef.from_device(storage.device)
                )
                typ = tensor_type.as_buffer().to_mlir()
                
                # Update MLIR function signature
                inputs = op.function_type.inputs
                op.function_type = builtin.FunctionType([*inputs, typ])
                
                buffer_val = graph.BufferValue.from_mlir(
                    block.add_argument(typ, _location())
                )
                tensor_values.append(buffer_val)
            
            # Register source for execution
            self.sources[buffer_val._mlir_value] = storage
        
        # Store values back - single value or list
        if len(tensor_values) == 1:
            tensor._value = tensor_values[0]
        else:
            # For multi-shard: store all values, convert buffers to tensor values
            with self.graph:
                tensor._impl._values = [bv[...] for bv in tensor_values]

    def add_unrealized(self, tensor: Tensor) -> None:
        """Registers a tensor as pending computation."""
        self.unrealized[id(tensor)] = tensor

    async def evaluate(
        self, 
        tensor: Tensor, 
        *extra_outputs: Any, 
        return_model: bool = False,
    ) -> Any:
        """Main entry point: Evaluates specific tensors and their dependencies."""
        from .pytree import tree_leaves
        from .tensor import Tensor

        sys.last_value = None
        sys.last_traceback = None
        gc.collect()

        # Gather all targets, preserving order of explicitly requested outputs
        seen: set[int] = set()
        targets: list[Tensor] = []
        
        def add_target(t: Tensor) -> None:
            if id(t) not in seen:
                seen.add(id(t))
                targets.append(t)
        
        # Add explicit outputs first (in order)
        add_target(tensor)
        for out in extra_outputs:
            if isinstance(out, Tensor):
                add_target(out)
            else:
                for leaf in tree_leaves(out):
                    if isinstance(leaf, Tensor):
                        add_target(leaf)
        
        # Add any other unrealized tensors
        for t in self.unrealized.values():
            add_target(t)
        
        # Select Strategy
        needs_sharding, ops, leaves = _needs_sharded_compilation(targets)

        if needs_sharding:
            await self._evaluate_sharded(targets, ops, leaves)
            return None
            
        return await self._evaluate_normal(targets, return_model=return_model)

    # --- Execution Strategies ---

    async def _evaluate_normal(self, unrealized: list[Tensor], return_model: bool) -> Any:
        """Standard compilation path for single-device execution.
        
        Handles both regular tensors (single value) and sharded tensors (multiple values).
        """
        
        if DEBUG_LAZY_EVAL:
            print("=" * 70)
            print(f"[LAZY EVAL] Epoch {self.epoch} - Setting {len(unrealized)} output(s):")
            for i, t in enumerate(unrealized):
                op_name = t._impl.op_name if t._impl.output_refs else "<leaf>"
                num_shards = len(t._impl._values) if t._impl._values else 1
                print(f"  [{i}] id={id(t)} op={op_name} shards={num_shards}")
            print("-" * 70)
        
        # Collect all output values - for sharded tensors, collect ALL shard values
        # Also build index mapping for result storage
        all_values = []
        value_map = []  # List of (tensor, shard_idx) or (tensor, None) for single-value
        
        with self.graph:
            for t in unrealized:
                values = t._impl._values
                if values and len(values) > 1:
                    # Sharded tensor: output all shards
                    for shard_idx, val in enumerate(values):
                        all_values.append(val)
                        value_map.append((t, shard_idx))
                else:
                    # Single-value tensor
                    all_values.append(graph.TensorValue(t))
                    value_map.append((t, None))
            
            self.graph.output(ops.random._peek_seed(), *all_values)
        
        if DEBUG_LAZY_EVAL:
            print("[LAZY EVAL] MAX Graph MLIR:")
            print(self.graph._module.operation)
            print("=" * 70)
        
        return await self._compile_and_execute_with_map(unrealized, value_map, return_model)

    async def _evaluate_sharded(
        self, 
        unrealized: list[Tensor], 
        op_impls: list[TensorImpl], 
        leaf_impls: list[TensorImpl]
    ) -> None:
        """Sharded compilation path with graph transformation.
        
        1. Runs sharding propagation (annotates all tensors)
        2. Builds sharded TensorValues by calling maxpr for each shard (same graph)
        3. Compiles and executes normally
        """
        from ..sharding.transform import transform_trace_for_sharding
        from ..sharding.spec import ShardingSpec
        
        # Find mesh from any sharded tensor
        mesh = None
        all_impls = op_impls + leaf_impls
        for impl in all_impls:
            if isinstance(impl.sharding, ShardingSpec):
                mesh = impl.sharding.mesh
                break
        
        if mesh is None:
            raise ValueError("No mesh found in sharded tensors")
        
        # Create a FRESH graph for sharded execution.
        # We do not want to mix the Logical Graph (which has dead ops) with the Sharded Graph.
        # Since the Partitioner "replays" the trace (executes maxpr), it will populate this new graph.
        sharded_graph = graph.Graph("sharded_main", input_types=[], context=self.context)
        
        # Transform: propagate + build sharded values in the NEW graph
        root_impls = [t._impl for t in unrealized]
        
        with sharded_graph:
             ops.random.set_seed(0) # TODO: Propagate actual seed
             # This populates sharded_graph with the detailed SPMD ops
             transform_trace_for_sharding(root_impls, mesh)
             
             # Set Outputs
             # Collect all shard outputs (which are now in sharded_graph)
             all_outputs: list = []
             for t in unrealized:
                 all_outputs.extend(t._impl._values)
             sharded_graph.output(ops.random._peek_seed(), *all_outputs)
        
        if DEBUG_LAZY_EVAL:
             print("[LAZY EVAL] Sharding: Graph Built. Inputs:", len(sharded_graph.inputs))
             # op = _core.Operation._from_cmlir(sharded_graph._mlir_op)
             # print(op)

        # Compile and execute
        # sharded_graph should have NO inputs (everything constant folded or replayed)
        # Verify?
        if len(sharded_graph.inputs) > 0:
             # Are these real inputs? 
             # partitioner currently doesn't create inputs, only constants.
             # If we support logical inputs later, we'd need to bind them here.
             pass

        try:
             model = _session().load(sharded_graph)
             seed_val, *results = model()
        except BaseException as e:
             if DEBUG_LAZY_EVAL:
                 print("[LAZY EVAL ERROR] Sharded Compilation Failed:")
                 print(sharded_graph._module.operation)
             raise RuntimeError(f"Sharded compilation failed: {e}") from e
        
        # Store results back into TensorImpls
        result_idx = 0
        for t in unrealized:
            num_shards = len(t._impl._values)
            shard_results = []
            for _ in range(num_shards):
                shard_results.append(results[result_idx])
                result_idx += 1
            t._impl._storages = shard_results
            t._impl._values = []
            t.real = True
        
        self._finalize_evaluation(seed_value=seed_val.item())

    # --- Low-Level Execution Mechanics ---

    async def _compile_and_execute(self, unrealized: list[Tensor], return_model: bool) -> Any:
        """Compiles the MLIR graph, executes it, and stores results."""
        # 1. Optimizations
        module = _core.Operation._from_cmlir(self.graph._module.operation)
        _core.lower(module, [builtin.passes.RemoveDeadValues()])
        _remove_unused_arguments(self.graph)
        
        # 2. Execution
        inputs = [self.sources[inp._mlir_value] for inp in self.graph.inputs]
        try:
            model = _session().load(self.graph)
            seed_val, *results = model(*inputs)
        except BaseException as e:
            self.graph._erase_output_if_present()
            if DEBUG_LAZY_EVAL:
                print("\n[LAZY EVAL ERROR] Failed to compile/execute. Graph state:")
                print(self.graph._module.operation)
                print("=" * 70)
            raise RuntimeError(f"Failed to compile/execute graph: {e}") from e

        # 3. Storage & Cleanup
        self._store_results(unrealized, results)
        result = (model, inputs) if return_model else None
        self._finalize_evaluation(seed_value=seed_val.item())
        return result

    async def _compile_and_execute_with_map(
        self, 
        unrealized: list[Tensor], 
        value_map: list[tuple[Tensor, int | None]], 
        return_model: bool
    ) -> Any:
        """Compiles and executes with sharded tensor support.
        
        Uses value_map to correctly store results back into tensors,
        handling both single-value and multi-shard tensors.
        """
        # 1. Optimizations
        module = _core.Operation._from_cmlir(self.graph._module.operation)
        _core.lower(module, [builtin.passes.RemoveDeadValues()])
        _remove_unused_arguments(self.graph)
        
        # 2. Execution
        inputs = [self.sources[inp._mlir_value] for inp in self.graph.inputs]
        try:
            model = _session().load(self.graph)
            seed_val, *results = model(*inputs)
        except BaseException as e:
            self.graph._erase_output_if_present()
            if DEBUG_LAZY_EVAL:
                print("\n[LAZY EVAL ERROR] Failed to compile/execute. Graph state:")
                print(self.graph._module.operation)
                print("=" * 70)
            raise RuntimeError(f"Failed to compile/execute graph: {e}") from e

        # 3. Storage using value_map - group results by tensor
        tensor_results: dict[int, list] = {}  # id(tensor) -> list of results
        for (tensor, shard_idx), result in zip(value_map, results, strict=True):
            tid = id(tensor)
            if tid not in tensor_results:
                tensor_results[tid] = []
            tensor_results[tid].append((shard_idx, result))
        
        # Store results into tensors
        for t in unrealized:
            tid = id(t)
            if tid in tensor_results:
                shard_results = tensor_results[tid]
                if len(shard_results) > 1:
                    # Multiple shards - sort by shard_idx and store all
                    shard_results.sort(key=lambda x: x[0] if x[0] is not None else 0)
                    t._impl._storages = [r for _, r in shard_results]
                    t._impl._values = []
                else:
                    # Single result
                    _, storage = shard_results[0]
                    t.storage = storage
                    t._value = None
                t.real = True
        
        result = (model, inputs) if return_model else None
        self._finalize_evaluation(seed_value=seed_val.item())
        return result


    def _store_results(self, unrealized: list[Tensor], results: list) -> None:
        """Populates tensors with execution results."""
        for t, storage in zip(unrealized, results, strict=True):
            if isinstance(storage, list):
                t._impl._storages = storage
                t._impl._values = []
            else:
                t.storage = storage
                t._value = None
            t.real = True

    def _finalize_evaluation(self, seed_value: int) -> None:
        """Prepares the graph for the next epoch."""
        self._reset(self.graph._context, seed_value)

# =============================================================================
# 4. High-Level Tools & Instance
# =============================================================================

def compile_with_sharding(outputs: list[Tensor], mesh: DeviceMesh):
    """Standalone tool to compile a sharded MAX graph."""
    from .tensor_impl import get_topological_order
    
    # Validation logic...
    for t in outputs:
        if not t.traced or t._impl.cached_shape is None:
            raise ValueError("Invalid tensor for sharding (must be traced/cached).")

    # Topological sort...
    all_impls = []
    seen = set()
    for t in outputs:
        for impl in get_topological_order(t._impl):
            if id(impl) not in seen:
                all_impls.append(impl)
                seen.add(id(impl))

    raise NotImplementedError(
        "Sharding compiler not yet implemented. "
        f"Graph has {len(all_impls)} operations ready for sharded compilation."
    )


# Global Singleton
GRAPH = ComputeGraph()