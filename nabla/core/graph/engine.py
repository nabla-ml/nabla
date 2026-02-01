# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Graph: Manages lazy evaluation and graph compilation."""

from __future__ import annotations

import gc
import sys
import weakref
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from max import _core, driver, graph, mlir
from max._core.dialects import builtin, kgen, mo
from max.graph import Value, ops
from max.graph.graph import _location

if TYPE_CHECKING:
    from ..tensor.api import Tensor

from ..common.context import _session

_GRAPH_EPOCH: int = 0
_SEED: ContextVar[Tensor] = ContextVar("_SEED")
_GRAPH_CACHE: dict[tuple, Any] = {}  # cache_key -> compiled model

import os

DEBUG_LAZY_EVAL: bool = os.environ.get("NABLA_DEBUG", "0") == "1"


def seed() -> Tensor:
    """Returns the global random seed tensor."""
    from ..tensor.api import Tensor

    if (s := _SEED.get(None)) is None:
        s = driver.Tensor(ops.random.SeedType)
        s[0] = 0
        _SEED.set(Tensor(storage=s))
    return _SEED.get()


def driver_tensor_type(t: driver.Tensor) -> graph.TensorType:
    """Converts a driver tensor to a TensorType."""
    return graph.TensorType(t.dtype, t.shape, graph.DeviceRef.from_device(t.device))


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


class ComputeGraph:
    """Manages the DAG of operations, lazy evaluation, and compilation."""

    graph: graph.Graph
    sources: dict[_core.Value[Any], driver.Tensor]
    unrealized: weakref.WeakValueDictionary[int, TensorImpl]
    epoch: int
    _input_refs: list[Tensor]
    _skip_finalize: bool

    def __init__(self, context: mlir.Context | None = None, seed: int = 0):
        global _GRAPH_EPOCH
        _GRAPH_EPOCH += 1
        self.epoch = _GRAPH_EPOCH
        self._input_refs = []
        self._skip_finalize = False
        self._reset(context, seed)

    def _reset(self, context: mlir.Context | None, seed: int) -> None:
        """Resets the internal graph state."""
        self.context = context or mlir.Context()
        self.sources = {}
        self.unrealized = weakref.WeakValueDictionary()
        self.graph = graph.Graph("main", input_types=[], context=self.context)
        self._input_refs = []
        self._skip_finalize = False
        with self.graph:
            ops.random.set_seed(seed)

    def clear_all(self) -> None:
        """Clears tracing state for fresh start."""
        global _GRAPH_EPOCH
        _GRAPH_EPOCH += 1
        self.epoch = _GRAPH_EPOCH
        self._reset(None, 0)
        # gc.collect()  # Removed: too expensive for hot paths

    def add_input(self, tensor: Tensor) -> None:
        """Registers a realized tensor's storages as graph inputs."""
        if any(t is tensor for t in self._input_refs):
            return

        storages = tensor._impl._buffers
        if not storages:
            raise TypeError("Only realized tensors may be graph inputs.")

        self._input_refs.append(tensor)

        op = _core.Operation._from_cmlir(self.graph._mlir_op)
        assert isinstance(op, mo.GraphOp)
        block = op.regions[0].front

        tensor_graph_values = []
        for storage in storages:
            with self.graph:
                tensor_type = graph.TensorType(
                    storage.dtype,
                    storage.shape,
                    graph.DeviceRef.from_device(storage.device),
                )
                typ = tensor_type.as_buffer().to_mlir()

                inputs = op.function_type.inputs
                op.function_type = builtin.FunctionType([*inputs, typ])

                buffer_val = graph.BufferValue.from_mlir(
                    block.add_argument(typ, _location())
                )
                tensor_graph_values.append(buffer_val)

            self.sources[buffer_val._mlir_value] = storage

        if len(tensor_graph_values) == 1:
            tensor._value = tensor_graph_values[0]
            with self.graph:
                tensor._impl._graph_values = [tensor_graph_values[0][...]]
                tensor._impl.graph_values_epoch = self.epoch
        else:
            with self.graph:
                tensor._impl._graph_values = [bv[...] for bv in tensor_graph_values]
                tensor._impl.graph_values_epoch = self.epoch

    def add_unrealized(self, impl: TensorImpl) -> None:
        """Registers a tensor implementation as pending computation."""
        self.unrealized[id(impl)] = impl

    def evaluate(
        self,
        tensor: Tensor,
        *extra_outputs: Any,
        return_model: bool = False,
    ) -> Any:
        """Main entry point: Evaluates specific tensors and their dependencies."""
        from ..common.pytree import tree_leaves
        from ..common import pytree
        from ..tensor.api import Tensor
        from ..tensor.impl import TensorImpl

        sys.last_value = None
        sys.last_traceback = None
        # gc.collect()  # Removed: too expensive for hot paths

        # Collect target tensors
        seen: set[int] = set()
        targets: list[Tensor] = []

        def add_target(t: Tensor) -> None:
            if id(t._impl) not in seen:
                seen.add(id(t._impl))
                targets.append(t)

        add_target(tensor)
        for out in extra_outputs:
            if isinstance(out, Tensor):
                add_target(out)
            else:
                for leaf in tree_leaves(out):
                    if isinstance(leaf, Tensor):
                        add_target(leaf)

        # Skip if only evaluating leaf inputs (nothing to compute)
        self._skip_finalize = all(t._impl.output_refs is None for t in targets)
        if self._skip_finalize:
            return None

        # --- COMPUTE CACHE KEY ---
        # We sort targets to ensure deterministic cache keys regardless of registration order.
        def get_tensor_key(t: Tensor):
            if t._impl.output_refs is not None and t._impl.output_refs._op_hash is not None:
                # Unrealized: (sorting_bucket=0, op_hash, output_index)
                return (0, t._impl.output_refs._op_hash, t._impl.output_index)
            # Realized: (sorting_bucket=1, dtype, shape, sharding)
            from ..ops.base import _make_hashable
            sharding_key = _make_hashable(t.sharding) if t.sharding else None
            return (1, str(t.dtype), tuple(int(d) for d in t.shape), sharding_key)

        targets.sort(key=lambda t: str(get_tensor_key(t)))
        
        op_hashes = [get_tensor_key(t) for t in targets]
        cache_key = tuple(op_hashes) if op_hashes else None
        
        if DEBUG_LAZY_EVAL:
            print(f"\n[CACHE] Key hash: {hash(cache_key)} | Cache size: {len(_GRAPH_CACHE)}")

        # === CHECK CACHE ===
        if cache_key is not None:
            entry = _GRAPH_CACHE.get(cache_key)
            if entry is not None:
                cached_model, kept_indices = entry
                if DEBUG_LAZY_EVAL:
                    print(f"[CACHE] HIT! key_hash={hash(cache_key)}")
                
                # Gather ALL candidate buffers from the trace in the order they would be added.
                # Since we don't have a fresh graph yet, we simulate the input ordering.
                all_candidate_tensors = self._get_input_tensors_ordered(targets)
                all_buffers = []
                for impl in all_candidate_tensors:
                    all_buffers.extend(impl._buffers)
                
                # Filter to only the ones recorded during MISS
                inputs = [all_buffers[i] for i in kept_indices]
                
                if DEBUG_LAZY_EVAL:
                    print(f"[CACHE] inputs: {[(tuple(inp.shape), str(inp.dtype), id(inp)) for inp in inputs]}")
                
                seed_val, *results = cached_model(*inputs)
                
                # Store results to targets
                result_idx = 0
                for t in targets:
                    n_shards = t.num_shards
                    t_results = results[result_idx : result_idx + n_shards]
                    
                    if n_shards > 1:
                        t._impl._buffers = list(t_results)
                    else:
                        t.storage = t_results[0]
                    
                    t._value = None
                    t.real = True
                    t._impl._graph_values = []
                    result_idx += n_shards
                    # Remove from unrealized since it's now real
                    self.unrealized.pop(id(t), None)
                
                self._finalize_evaluation(seed_value=seed_val.item())
                self._cleanup_trace(targets)
                return (cached_model, inputs) if return_model else None

        # === CACHE MISS - Build and compile graph ===
        if DEBUG_LAZY_EVAL:
            print(f"[CACHE] MISS - storing. key_hash={hash(cache_key)}")

        # Bump epoch and create fresh MAX graph
        global _GRAPH_EPOCH
        _GRAPH_EPOCH += 1
        self.epoch = _GRAPH_EPOCH

        self.graph = graph.Graph("main", input_types=[], context=self.context)
        self.sources = {}
        self._input_refs = []
        with self.graph:
            ops.random.set_seed(0)

        # Replay trace to build MAX graph
        self._replay_trace_to_build_graph(targets)
        
        # Build graph outputs
        all_graph_values = []
        value_map = []

        with self.graph:
            for t in targets:
                if t._impl.graph_values_epoch != self.epoch:
                    t._impl._graph_values = []

                if not t._impl._graph_values and t._impl.is_realized:
                    self.add_input(t)

                values = t._impl._graph_values
                if not values:
                    raise RuntimeError(f"Tensor {id(t)} has no graph values")

                if len(values) > 1:
                    for shard_idx, val in enumerate(values):
                        all_graph_values.append(val)
                        value_map.append((t, shard_idx))
                else:
                    all_graph_values.append(values[0])
                    value_map.append((t, None))

            seed_out = ops.random._peek_seed()
            self.graph.output(seed_out, *all_graph_values)

        # Optimize and compile
        module = _core.Operation._from_cmlir(self.graph._module.operation)
        _core.lower(module, [builtin.passes.RemoveDeadValues()])
        _remove_unused_arguments(self.graph)

        inputs: list[driver.Tensor] = []
        for inp in self.graph.inputs:
            storage = self.sources.get(inp._mlir_value)
            if storage is None:
                raise RuntimeError("Missing storage for graph input")
            inputs.append(storage)

        model = _session().load(self.graph)
        seed_val, *results = model(*inputs)
        
        # Store results
        tensor_results: dict[int, list] = {}
        for (t, shard_idx), result in zip(value_map, results, strict=True):
            tid = id(t)
            if tid not in tensor_results:
                tensor_results[tid] = []
            tensor_results[tid].append((shard_idx, result))

        for t in targets:
            tid = id(t)
            if tid in tensor_results:
                shard_results = tensor_results[tid]
                if len(shard_results) > 1:
                    shard_results.sort(key=lambda x: x[0] if x[0] is not None else 0)
                    t._impl._buffers = [r for _, r in shard_results]
                    t._impl._graph_values = []
                else:
                    _, storage = shard_results[0]
                    t.storage = storage
                    t._value = None
                t.real = True
                # Remove from unrealized since it's now real
                self.unrealized.pop(id(t._impl), None)

        # Cache the model
        if cache_key is not None:
            # Identify which buffers in the full trace order were actually added as graph inputs
            all_candidate_tensors = self._get_input_tensors_ordered(targets)
            all_candidate_buffers = []
            for impl in all_candidate_tensors:
                all_candidate_buffers.extend(impl._buffers)
                
            used_storages = [self.sources.get(inp._mlir_value) for inp in self.graph.inputs]
            
            kept_indices = []
            for storage in used_storages:
                found = False
                for i, s in enumerate(all_candidate_buffers):
                    if s is storage:
                        kept_indices.append(i)
                        found = True
                        break
                if not found:
                    raise RuntimeError("Could not map graph input back to trace")

            _GRAPH_CACHE[cache_key] = (model, kept_indices)

        self._finalize_evaluation(seed_value=seed_val.item())
        self._cleanup_trace(targets)
        return (model, inputs) if return_model else None

    def _cleanup_trace(self, targets: list[Tensor]) -> None:
        """Clean up trace references to prevent unbounded memory growth.
        
        Once a tensor is realized, we can clear internal graph values but
        we must preserve output_refs for tensors that may be used as inputs
        to future operations. We only clear output_refs on the targets themselves.
        """
        from ..common import pytree
        from ..tensor.impl import TensorImpl

        visited: set[int] = set()
        target_impl_ids = {id(t._impl) for t in targets}
        
        def clean(impl: TensorImpl) -> None:
            if id(impl) in visited:
                return
            visited.add(id(impl))
            impl._graph_values = []
            impl.graph_values_epoch = -1
            
            if impl.output_refs:
                for arg in pytree.tree_leaves(impl.output_refs.op_args):
                    if isinstance(arg, TensorImpl):
                        clean(arg)
                        
        # Traverse and clean graph values
        for t in targets:
            clean(t._impl)
            
        # Only clear output_refs on the targets themselves (not their inputs)
        # This breaks the chain for realized outputs while preserving 
        # reusability of inputs
        for t in targets:
            t._impl.output_refs = None
            t._impl.output_index = None

    def _replay_trace_to_build_graph(self, targets: list[Tensor]) -> None:
        """Walk OpNode DAG and execute operations to build MAX graph."""
        from ..common import pytree
        from ..tensor.api import Tensor
        from ..tensor.impl import TensorImpl

        # Collect OpNodes in topological order
        visited: set[int] = set()
        opnodes_topo: list[Any] = []

        def dfs(opnode) -> None:
            if id(opnode) in visited:
                return
            for arg in pytree.tree_leaves(opnode.op_args):
                if isinstance(arg, TensorImpl) and not arg.is_realized and arg.output_refs:
                    dfs(arg.output_refs)
            visited.add(id(opnode))
            opnodes_topo.append(opnode)

        for t in targets:
            if t._impl.output_refs:
                dfs(t._impl.output_refs)

        # Execute each OpNode
        for opnode in opnodes_topo:
            # Skip if outputs already valid
            if all(ref.graph_values_epoch == self.epoch and ref._graph_values
                   for ref in opnode._refs if ref is not None):
                continue

            # Ensure inputs have graph values
            for arg in pytree.tree_leaves(opnode.op_args):
                if isinstance(arg, TensorImpl):
                    if arg.graph_values_epoch != self.epoch or not arg._graph_values:
                        if arg.is_realized:
                            self.add_input(Tensor(impl=arg))

            # Execute operation
            def to_tensor(x):
                return Tensor(impl=x) if isinstance(x, TensorImpl) else x

            op_args = pytree.tree_map(to_tensor, opnode.op_args)
            
            with self.graph:
                raw_result = opnode.op.execute(op_args, opnode.op_kwargs or {})

            # Extract graph values
            if isinstance(raw_result, tuple) and len(raw_result) == 3:
                shard_graph_values, _, _ = raw_result
            elif hasattr(raw_result, "shard_graph_values"):
                shard_graph_values = raw_result.shard_graph_values
            else:
                shard_graph_values = raw_result

            # Store graph values to output refs
            if isinstance(shard_graph_values, (list, tuple)) and \
               not isinstance(shard_graph_values[0] if shard_graph_values else None, (list, tuple, dict)):
                if len(opnode._refs) == 1 and opnode._refs[0] is not None:
                    opnode._refs[0]._graph_values = shard_graph_values
                    opnode._refs[0].graph_values_epoch = self.epoch
            else:
                if isinstance(shard_graph_values[0] if shard_graph_values else None, (list, tuple)):
                    unzipped = list(zip(*shard_graph_values)) if shard_graph_values else []
                    for i, ref in enumerate(opnode._refs):
                        if ref is not None and i < len(unzipped):
                            ref._graph_values = list(unzipped[i])
                            ref.graph_values_epoch = self.epoch

    def _finalize_evaluation(self, seed_value: int) -> None:
        """Prepares the graph for the next epoch."""
        global _GRAPH_EPOCH
        _GRAPH_EPOCH += 1
        self.epoch = _GRAPH_EPOCH
        self._reset(None, seed_value)

    def _get_input_tensors_ordered(self, targets: list[Tensor]) -> list[TensorImpl]:
        """Returns realized TensorImpls in the canonical order they would be added to a graph."""
        from ..common import pytree
        from ..tensor.impl import TensorImpl

        visited_nodes: set[int] = set()
        visited_impls: set[int] = set()
        ordered_inputs: list[TensorImpl] = []

        # 1. Topological walk of OpNodes
        opnodes_topo: list[Any] = []
        def dfs(opnode) -> None:
            if id(opnode) in visited_nodes:
                return
            for arg in pytree.tree_leaves(opnode.op_args):
                if isinstance(arg, TensorImpl) and not arg.is_realized and arg.output_refs:
                    dfs(arg.output_refs)
            visited_nodes.add(id(opnode))
            opnodes_topo.append(opnode)

        for t in targets:
            if t._impl.output_refs:
                dfs(t._impl.output_refs)

        # 2. Simulate add_input calls in execution order
        # Important: this must match EXACTLY the order in _replay_trace_to_build_graph
        for opnode in opnodes_topo:
            for arg in pytree.tree_leaves(opnode.op_args):
                if isinstance(arg, TensorImpl):
                    if id(arg) not in visited_impls:
                        if arg.is_realized:
                            ordered_inputs.append(arg)
                        visited_impls.add(id(arg))

        # 3. Add any targets that are themselves realized leaves
        for t in targets:
            if t._impl.is_realized and id(t._impl) not in visited_impls:
                ordered_inputs.append(t._impl)
                visited_impls.add(id(t._impl))

        return ordered_inputs


GRAPH = ComputeGraph()
