# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
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
    from ..tensor.api import Tensor
    from ...sharding import DeviceMesh
    from ..tensor.impl import TensorImpl

from ..common.context import _session

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
    from ..tensor.api import Tensor
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

    def evaluate(
        self, 
        tensor: Tensor, 
        *extra_outputs: Any, 
        return_model: bool = False,
    ) -> Any:
        """Main entry point: Evaluates specific tensors and their dependencies."""
        from ..common.pytree import tree_leaves
        from ..tensor.api import Tensor

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
        return self._evaluate_normal(targets, return_model=return_model)

    # --- Execution Strategies ---

    def _evaluate_normal(self, unrealized: list[Tensor], return_model: bool) -> Any:
        """Standard compilation path handling both single-device and eager-sharded execution.
        
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
        
        return self._compile_and_execute_with_map(unrealized, value_map, return_model)

    # --- Low-Level Execution Mechanics ---

    def _compile_and_execute_with_map(
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
        try:
            module = _core.Operation._from_cmlir(self.graph._module.operation)
            _core.lower(module, [builtin.passes.RemoveDeadValues()])
            _remove_unused_arguments(self.graph)
        except Exception as e:
            if DEBUG_LAZY_EVAL:
                print(f"[LAZY EVAL ERROR] Optimization failed: {e}")
            raise

        # 2. Execution
        try:
            inputs = [self.sources[inp._mlir_value] for inp in self.graph.inputs]
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
        # Force new context creation to avoid pollution/segfaults
        self._reset(None, seed_value)

# Global Singleton
GRAPH = ComputeGraph()