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
# Cache: key -> (model, num_outputs_per_target)
# num_outputs_per_target tells us how many results each target tensor has
_GRAPH_CACHE: dict[tuple, tuple[Any, list[int]]] = {}

import os

DEBUG_LAZY_EVAL: bool = False  # Disabled to reduce noise


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
    unrealized: weakref.WeakValueDictionary[int, Tensor]
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
        """Clears tracing state for fresh start (useful for testing).
        
        NOTE: Does NOT clear _GRAPH_CACHE - that's the JIT cache for performance!
        Incrementing epoch effectively invalidates stale references without
        losing compiled graphs.
        """
        global _GRAPH_EPOCH
        _GRAPH_EPOCH += 1
        self.epoch = _GRAPH_EPOCH
        self._reset(None, 0)
        gc.collect()

    def add_input(self, tensor: Tensor) -> None:
        """Registers a realized tensor's storages as graph inputs."""
        if any(t is tensor for t in self._input_refs):
            return

        storages = tensor._impl._buffers
        if not storages:
            raise TypeError("Only realized tensors may be graph inputs.")

        # Track input refs for deterministic input ordering
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

        # Bump epoch and create fresh MAX graph (without resetting ComputeGraph state)
        global _GRAPH_EPOCH
        _GRAPH_EPOCH += 1
        self.epoch = _GRAPH_EPOCH
        
        self.graph = graph.Graph("main", input_types=[], context=self.context)
        self.sources = {}
        self._input_refs = []
        self._skip_finalize = False
        with self.graph:
            ops.random.set_seed(0)

        sys.last_value = None
        sys.last_traceback = None
        gc.collect()

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

        # Skip finalize when only evaluating leaf inputs (used by rehydration)
        self._skip_finalize = all(t._impl.output_refs is None for t in targets)

        # === EARLY CACHE CHECK ===
        # Compute cache key and gather inputs BEFORE building graph
        if not self._skip_finalize:
            cache_key, input_tensors = self._compute_cache_key_and_inputs(targets)
            
            print(f"\n[EARLY CACHE CHECK] Cache key (first 200): {str(cache_key)[:200]}")
            print(f"[EARLY CACHE CHECK] Cache key HASH: {hash(cache_key)}")
            print(f"[EARLY CACHE CHECK] Cache has {len(_GRAPH_CACHE)} entries")
            if _GRAPH_CACHE:
                for i, k in enumerate(_GRAPH_CACHE.keys()):
                    print(f"[EARLY CACHE CHECK]   Entry {i} hash: {hash(k)}")
            print(f"[EARLY CACHE CHECK] Input tensors: {len(input_tensors)}")
            
            cached = _GRAPH_CACHE.get(cache_key)
            
            if cached is not None:
                # CACHE HIT - skip all graph building!
                model, num_outputs_per_target = cached
                print(f"\n{'='*80}")
                print(f"[CACHE] ✓✓✓ EARLY CACHE HIT ✓✓✓ - Skipping graph building!")
                print(f"{'='*80}\n")
                
                # Build inputs from the gathered input tensors
                inputs = []
                for t in input_tensors:
                    for buf in t._impl._buffers:
                        inputs.append(buf)
                
                # Execute cached model
                seed_val, *results = model(*inputs)
                
                # Store results back to tensors using num_outputs_per_target
                result_idx = 0
                for i, t in enumerate(targets):
                    if t._impl._buffers:
                        # Already realized - skip
                        continue
                    
                    num_outputs = num_outputs_per_target[i] if i < len(num_outputs_per_target) else 1
                    
                    if num_outputs > 1:
                        t._impl._buffers = results[result_idx:result_idx + num_outputs]
                        result_idx += num_outputs
                    else:
                        t.storage = results[result_idx]
                        t._value = None
                        result_idx += 1
                    t.real = True
                    t._impl._graph_values = []
                
                # Finalize
                if not self._skip_finalize:
                    self._finalize_evaluation(seed_value=seed_val.item())
                self._skip_finalize = False
                
                # Clean up
                self._cleanup_trace_after_evaluation(targets)
                return (model, inputs) if return_model else None
        
        # === CACHE MISS - Build graph as usual ===
        # Walk OpNode DAG and execute operations to populate _graph_values
        if not self._skip_finalize:
            self._replay_trace_to_build_graph(targets)

        result = self._evaluate_normal(targets, return_model=return_model)

        # Clean up trace after evaluation to prevent infinite growth
        self._cleanup_trace_after_evaluation(targets)

        return result
    
    def _compute_cache_key_and_inputs(self, targets: list["Tensor"]) -> tuple[tuple, list["Tensor"]]:
        """Compute cache key from OpNode hashes and gather realized input tensors.
        
        Returns:
            (cache_key, input_tensors): The cache key tuple and ordered list of realized input tensors
        """
        from ..common import pytree
        from ..tensor.impl import TensorImpl
        
        # Cache key is simply the sorted hashes of target OpNodes
        # Each OpNode's _op_hash already recursively contains all dependency hashes!
        op_hashes = []
        for t in targets:
            if t._impl.output_refs is not None and t._impl.output_refs._op_hash is not None:
                op_hashes.append(t._impl.output_refs._op_hash)
        
        cache_key = tuple(sorted(op_hashes))
        
        # Gather realized input tensors by walking the DAG
        visited_opnode_ids: set[int] = set()
        realized_inputs: list["Tensor"] = []
        seen_input_ids: set[int] = set()
        
        def walk_opnode(opnode) -> None:
            opnode_id = id(opnode)
            if opnode_id in visited_opnode_ids:
                return
            
            # Visit dependencies first (DFS) to get inputs in consistent order
            for arg in pytree.tree_leaves(opnode.op_args):
                if isinstance(arg, TensorImpl):
                    if arg.output_refs:
                        walk_opnode(arg.output_refs)
                    elif arg._buffers:  # Realized input tensor
                        impl_id = id(arg)
                        if impl_id not in seen_input_ids:
                            seen_input_ids.add(impl_id)
                            from ..tensor.api import Tensor
                            realized_inputs.append(Tensor(impl=arg))
            
            visited_opnode_ids.add(opnode_id)
        
        # Start from target OpNodes
        for target in targets:
            if target._impl.output_refs:
                walk_opnode(target._impl.output_refs)
        
        return cache_key, realized_inputs

    def _cleanup_trace_after_evaluation(self, targets: list[Tensor]) -> None:
        """Clean up trace references to allow garbage collection of OpNodes."""
        from ..common import pytree
        from ..tensor.api import Tensor
        from ..tensor.impl import TensorImpl

        if DEBUG_LAZY_EVAL:
            print("[LAZY EVAL] Cleaning up trace after evaluation...")

        # Collect all TensorImpls reachable from targets
        visited_impl_ids: set[int] = set()
        all_impls: list[TensorImpl] = []

        def collect_impls(impl: TensorImpl) -> None:
            impl_id = id(impl)
            if impl_id in visited_impl_ids:
                return
            visited_impl_ids.add(impl_id)
            all_impls.append(impl)

            # Follow OpNode dependencies
            if impl.output_refs:
                for arg in pytree.tree_leaves(impl.output_refs.op_args):
                    if isinstance(arg, TensorImpl):
                        collect_impls(arg)

        for target in targets:
            collect_impls(target._impl)

        # Clear graph values (they're in MAX graph now or in _buffers, not needed in trace)
        # NOTE: We keep output_refs because they're needed for backward pass and transforms.
        # The cache works because get_tensor_hash() now checks if tensor is realized FIRST,
        # and uses stable metadata for realized tensors regardless of their output_refs history.
        for impl in all_impls:
            impl._graph_values = []
            impl.graph_values_epoch = -1

        if DEBUG_LAZY_EVAL:
            print(f"[LAZY EVAL] Cleaned up {len(all_impls)} TensorImpls (cleared graph values)")
    def _replay_trace_to_build_graph(self, targets: list[Tensor]) -> None:
        """Walk OpNode DAG and execute operations to build MAX graph."""
        from ..common import pytree
        from ..tensor.api import Tensor
        from ..tensor.impl import TensorImpl

        if DEBUG_LAZY_EVAL:
            print("[LAZY EVAL] Replaying trace to build MAX graph...")

        # Collect all OpNodes in topological order via DFS
        visited_opnode_ids: set[int] = set()
        opnodes_topo: list[Any] = []  # OpNode list

        def dfs_opnode(opnode) -> None:
            opnode_id = id(opnode)
            if opnode_id in visited_opnode_ids:
                return
            
            # Visit dependencies first (inputs)
            for arg in pytree.tree_leaves(opnode.op_args):
                if isinstance(arg, TensorImpl) and arg.output_refs:
                    dfs_opnode(arg.output_refs)
            
            visited_opnode_ids.add(opnode_id)
            opnodes_topo.append(opnode)

        # Start DFS from target OpNodes
        for target in targets:
            if target._impl.output_refs:
                dfs_opnode(target._impl.output_refs)

        if DEBUG_LAZY_EVAL:
            print(f"[LAZY EVAL] Found {len(opnodes_topo)} operations to replay")

        # Execute each OpNode in topological order
        for opnode in opnodes_topo:
            # Check if outputs already have valid graph values
            outputs_valid = all(
                ref.graph_values_epoch == self.epoch and ref._graph_values
                for ref in opnode._refs if ref is not None
            )
            
            if outputs_valid:
                if DEBUG_LAZY_EVAL:
                    print(f"[LAZY EVAL] Skipping {opnode.op.name} (already valid)")
                continue

            # Ensure all inputs have graph values
            def ensure_graph_values(x):
                if isinstance(x, TensorImpl):
                    if x.graph_values_epoch != self.epoch or not x._graph_values:
                        if x.is_realized:
                            # Realized leaf tensor - add as input
                            tensor_wrapper = Tensor(impl=x)
                            self.add_input(tensor_wrapper)
                        elif x.output_refs is None:
                            raise RuntimeError(
                                f"Tensor {id(x)} has no graph values and no output_refs to compute them"
                            )
                        # else: will be computed by earlier OpNode in topo order

            for arg in pytree.tree_leaves(opnode.op_args):
                ensure_graph_values(arg)

            # Convert TensorImpl args back to Tensor wrappers for execute()
            def to_tensor(x):
                if isinstance(x, TensorImpl):
                    return Tensor(impl=x)
                return x

            op_args = pytree.tree_map(to_tensor, opnode.op_args)
            op_kwargs = opnode.op_kwargs or {}

            if DEBUG_LAZY_EVAL:
                print(f"[LAZY EVAL] Executing {opnode.op.name}")

            # Execute operation to get graph values
            with self.graph:
                raw_result = opnode.op.execute(op_args, op_kwargs)

            # Extract shard_graph_values from result
            if isinstance(raw_result, tuple) and len(raw_result) == 3:
                shard_graph_values, output_sharding, res_mesh = raw_result
            elif hasattr(raw_result, "shard_graph_values"):
                shard_graph_values = raw_result.shard_graph_values
                output_sharding = getattr(raw_result, "output_sharding", None)
                res_mesh = getattr(raw_result, "mesh", None)
            else:
                shard_graph_values = raw_result

            # Populate graph values for output tensors
            if isinstance(shard_graph_values, (list, tuple)) and not isinstance(shard_graph_values[0] if shard_graph_values else None, (list, tuple, dict)):
                # Single output case
                if len(opnode._refs) == 1 and opnode._refs[0] is not None:
                    opnode._refs[0]._graph_values = shard_graph_values
                    opnode._refs[0].graph_values_epoch = self.epoch
            else:
                # Multi-output case - structured outputs
                if isinstance(shard_graph_values[0] if shard_graph_values else None, (list, tuple)):
                    # Unzip shard results
                    unzipped = list(zip(*shard_graph_values)) if shard_graph_values else []
                    for i, ref in enumerate(opnode._refs):
                        if ref is not None and i < len(unzipped):
                            ref._graph_values = list(unzipped[i])
                            ref.graph_values_epoch = self.epoch
                elif isinstance(shard_graph_values[0] if shard_graph_values else None, dict):
                    # Dict outputs - map by key
                    keys = shard_graph_values[0].keys() if shard_graph_values else []
                    for ref in opnode._refs:
                        if ref is not None:
                            # Find matching key - this requires storing key info in OpNode
                            # For now, assume single dict output
                            pass

    def _evaluate_normal(self, unrealized: list[Tensor], return_model: bool) -> Any:
        """Standard compilation path handling both single-device and eager-sharded execution."""

        if DEBUG_LAZY_EVAL:
            print("=" * 70)
            print(
                f"[LAZY EVAL] Epoch {self.epoch} - Setting {len(unrealized)} output(s):"
            )
            for i, t in enumerate(unrealized):
                op_name = t._impl.op_name if t._impl.output_refs else "<leaf>"
                num_shards = len(t._impl._graph_values) if t._impl._graph_values else 1
                print(f"  [{i}] id={id(t)} op={op_name} shards={num_shards}")
            print("-" * 70)

        all_graph_values = []
        value_map = []

        with self.graph:
            for i, t in enumerate(unrealized):
                if t._impl.graph_values_epoch != self.epoch:
                    if DEBUG_LAZY_EVAL:
                        print(
                            f"[LAZY DEBUG] Clearing stale values for target tensor {id(t)} "
                            f"(epoch: {t._impl.graph_values_epoch} != {self.epoch})"
                        )
                    t._impl._graph_values = []

                if not t._impl._graph_values and t._impl.is_realized:
                    self.add_input(t)

                values = t._impl._graph_values
                if not values:
                    print(f"FAILED TENSOR {id(t)} info:")
                    print(f"  sharding: {t.sharding}")
                    print(f"  realized: {t.is_realized}")
                    print(f"  epoch: {t._impl.graph_values_epoch}")
                    print(f"  graph epoch: {self.epoch}")
                    if t._impl.output_refs:
                        print(f"  op: {t._impl.op_name}")
                    raise RuntimeError(
                        f"Attempting to evaluate tensor {id(t)} with no values/storage"
                    )

                if values and len(values) > 1:
                    for shard_idx, val in enumerate(values):
                        all_graph_values.append(val)
                        value_map.append((t, shard_idx))
                else:
                    all_graph_values.append(values[0])
                    value_map.append((t, None))

            seed = ops.random._peek_seed()
            self.graph.output(seed, *all_graph_values)

        if DEBUG_LAZY_EVAL:
            print("[LAZY EVAL] MAX Graph:")
            print(self.graph)
            print("=" * 70)

        return self._compile_and_execute_with_map(unrealized, value_map, return_model)

    def _compile_and_execute_with_map(
        self,
        unrealized: list[Tensor],
        value_map: list[tuple[Tensor, int | None]],
        return_model: bool,
    ) -> Any:
        """Compiles and executes with sharded tensor support."""

        try:
            module = _core.Operation._from_cmlir(self.graph._module.operation)
            _core.lower(module, [builtin.passes.RemoveDeadValues()])
            _remove_unused_arguments(self.graph)
        except Exception as e:
            if DEBUG_LAZY_EVAL:
                print(f"[LAZY EVAL ERROR] Optimization failed: {e}")
            raise

        # Build input buffers in graph argument order for deterministic caching
        inputs: list[driver.Tensor] = []
        for inp in self.graph.inputs:
            storage = self.sources.get(inp._mlir_value)
            if storage is None:
                raise RuntimeError("Missing storage for graph input")
            inputs.append(storage)

        # Build cache key from sorted hashes of all unrealized OpNodes
        # Note: This path is only reached on cache MISS (early check happens in evaluate())
        print(f"\n{'='*80}")
        print(f"[CACHE] CACHE MISS - Compiling new model for {len(unrealized)} unrealized tensors")
        
        hashes_to_cache = []
        for i, t in enumerate(unrealized):
            if t._impl.output_refs is not None and t._impl.output_refs._op_hash is not None:
                op_name = t._impl.output_refs.op.name if hasattr(t._impl.output_refs.op, 'name') else 'unknown'
                hash_val = t._impl.output_refs._op_hash
                hashes_to_cache.append(hash_val)
                print(f"[CACHE]   Tensor {i}: {op_name}")
                print(f"[CACHE]     Hash (first 200 chars): {str(hash_val)[:200]}")
        
        cache_key = tuple(sorted(hashes_to_cache))
        
        print(f"[CACHE] Cache key (first 300 chars): {str(cache_key)[:300]}")
        print(f"[CACHE] Cache key HASH: {hash(cache_key)}")
        print(f"{'='*80}\n")
        
        # Compile and execute
        try:
            if DEBUG_LAZY_EVAL:
                print(f"[LAZY EVAL] Compiling new model...")
            model = _session().load(self.graph)
            seed_val, *results = model(*inputs)
            
            # Count how many outputs each target tensor has
            output_counts: dict[int, int] = {}
            for (tensor, shard_idx) in value_map:
                tid = id(tensor)
                output_counts[tid] = output_counts.get(tid, 0) + 1
            
            # Store num_outputs_per_target in target order
            num_outputs_per_target = [output_counts.get(id(t), 1) for t in unrealized]
            
            # Cache both the model and output structure
            _GRAPH_CACHE[cache_key] = (model, num_outputs_per_target)
            
            print(f"[CACHE] ✓ Model compiled and cached (cache now has {len(_GRAPH_CACHE)} entries)")
        except BaseException as e:
            self.graph._erase_output_if_present()
            if DEBUG_LAZY_EVAL:
                print("\n[LAZY EVAL ERROR] Failed to compile/execute. Graph state:")
                print(self.graph._module.operation)
                print("=" * 70)
            raise RuntimeError(f"Failed to compile/execute graph: {e}") from e

        tensor_results: dict[int, list] = {}
        for (tensor, shard_idx), result in zip(value_map, results, strict=True):
            tid = id(tensor)
            if tid not in tensor_results:
                tensor_results[tid] = []
            tensor_results[tid].append((shard_idx, result))

        for t in unrealized:
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

        result = (model, inputs) if return_model else None
        if not self._skip_finalize:
            self._finalize_evaluation(seed_value=seed_val.item())
        self._skip_finalize = False
        return result

    def _store_results(self, unrealized: list[Tensor], results: list) -> None:
        """Populates tensors with execution results."""
        for t, storage in zip(unrealized, results, strict=True):
            if isinstance(storage, list):
                t._impl._buffers = storage
                t._impl._graph_values = []
            else:
                t.storage = storage
                t._value = None
            t.real = True

    def _finalize_evaluation(self, seed_value: int) -> None:
        """Prepares the graph for the next epoch."""
        global _GRAPH_EPOCH
        _GRAPH_EPOCH += 1
        self.epoch = _GRAPH_EPOCH

        self._reset(None, seed_value)


GRAPH = ComputeGraph()
