# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Graph: Manages lazy evaluation and graph compilation."""

from __future__ import annotations

import sys
import weakref
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from max import _core, driver, graph, mlir
from max._core.dialects import builtin, kgen, mo
from max.graph import Value, ops
from max.graph.graph import _location

if TYPE_CHECKING:
    from max.graph.model import CompiledModel

    from ..tensor.api import Tensor
    from ..tensor.impl import TensorImpl
    from .tracing import OpNode


from ..common.context import _session

_GRAPH_EPOCH: int = 0
_SEED: ContextVar[Tensor | None] = ContextVar("_SEED", default=None)
_GRAPH_CACHE: dict[
    tuple[Any, ...], tuple[CompiledModel, list[int]]
] = {}  # cache_key -> (compiled model, kept_input_indices)

import os

DEBUG_LAZY_EVAL: bool = os.environ.get("NABLA_DEBUG", "0") == "1"


def _debug_eval(msg: str) -> None:
    if not DEBUG_LAZY_EVAL:
        return
    import time

    ts = time.strftime("%H:%M:%S")
    print(f"[NABLA_DEBUG {ts}] {msg}", flush=True)


def seed() -> Tensor:
    """Returns the global random seed tensor."""
    from ..tensor.api import Tensor

    if (s := _SEED.get(None)) is None:
        s = driver.Buffer(ops.random.SeedType)
        s[0] = 0
        _SEED.set(Tensor(buffers=s))
    return _SEED.get()


def driver_tensor_type(t: driver.Buffer) -> graph.TensorType:
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
    sources: dict[_core.Value[Any], driver.Buffer]
    unrealized: weakref.WeakValueDictionary[int, TensorImpl]
    epoch: int
    _input_refs: list[Tensor]
    _skip_finalize: bool
    _debug_input_add_attempts: int
    _debug_input_add_registered: int
    _debug_input_add_by_reason: dict[str, int]
    _debug_input_add_buffers_by_reason: dict[str, int]
    _debug_constant_add_count: int
    _debug_constant_add_buffers: int

    def __init__(self, context: mlir.Context | None = None, seed: int = 0):
        global _GRAPH_EPOCH
        _GRAPH_EPOCH += 1
        self.epoch = _GRAPH_EPOCH
        self._input_refs = []
        self._skip_finalize = False
        self._reset(context, seed)

    def _reset(
        self,
        context: mlir.Context | None,
        seed: int,
        input_types: list[graph.TensorType] | None = None,
    ) -> None:
        """Resets the internal graph state.

        Args:
            context: MLIR context to use (creates new one if None)
            seed: Random seed for the graph
            input_types: Optional list of TensorTypes for graph inputs (for symbolic dims)
        """
        self.context = context or mlir.Context()
        self.sources = {}
        self.unrealized = weakref.WeakValueDictionary()
        self.graph = graph.Graph(
            "main", input_types=input_types or [], context=self.context
        )
        self._input_refs = []
        self._skip_finalize = False
        self._debug_input_add_attempts = 0
        self._debug_input_add_registered = 0
        self._debug_input_add_by_reason = {}
        self._debug_input_add_buffers_by_reason = {}
        self._debug_constant_add_count = 0
        self._debug_constant_add_buffers = 0
        with self.graph:
            ops.random.set_seed(seed)

    def clear_all(self) -> None:
        """Clears tracing state and compiled model cache for fresh start."""
        global _GRAPH_EPOCH, _GRAPH_CACHE
        _GRAPH_EPOCH += 1
        self.epoch = _GRAPH_EPOCH
        _GRAPH_CACHE.clear()
        self._reset(None, 0)

        # gc.collect()  # Removed: too expensive for hot paths

    def add_input(
        self,
        tensor: Tensor,
        shape: tuple[int, ...] | None = None,
        *,
        reason: str = "unknown",
    ) -> None:
        """Registers a realized tensor's bufferss as graph inputs."""
        self._debug_input_add_attempts += 1
        self._debug_input_add_by_reason[reason] = (
            self._debug_input_add_by_reason.get(reason, 0) + 1
        )

        impl = tensor._impl
        bufferss = impl._buffers
        if not bufferss:
            raise TypeError("Only realized tensors may be graph inputs.")

        # Check if this impl's buffer is already registered (use buffer identity, not tensor)
        buf = bufferss[0]
        for ref_tensor in self._input_refs:
            ref_buffers = ref_tensor._impl._buffers
            if ref_buffers and ref_buffers[0] is buf:
                # Same buffer already registered - copy graph values to this impl
                with self.graph:
                    impl._graph_values = [
                        gv[...] for gv in ref_tensor._impl._graph_values
                    ]
                    impl.graph_values_epoch = self.epoch
                return

        self._input_refs.append(tensor)
        self._debug_input_add_registered += 1
        self._debug_input_add_buffers_by_reason[reason] = (
            self._debug_input_add_buffers_by_reason.get(reason, 0) + len(bufferss)
        )

        op = _core.Operation._from_cmlir(self.graph._mlir_op)
        assert isinstance(op, mo.GraphOp)
        block = op.regions[0].front

        tensor_graph_values = []
        for buffers in bufferss:
            with self.graph:
                # Use provided shape (e.g. symbolic) or fall back to physical buffer shape
                input_shape = shape if shape is not None else buffers.shape

                tensor_type = graph.TensorType(
                    buffers.dtype,
                    input_shape,
                    graph.DeviceRef.from_device(buffers.device),
                )
                typ = tensor_type.as_buffer().to_mlir()

                inputs = op.function_type.inputs
                op.function_type = builtin.FunctionType([*inputs, typ])

                buffer_val = graph.BufferValue.from_mlir(
                    block.add_argument(typ, _location())
                )
                tensor_graph_values.append(buffer_val)

            self.sources[buffer_val._mlir_value] = buffers

        # Initialize graph values and load them (essential for use in ops)
        if len(tensor_graph_values) == 1:
            tensor._value = tensor_graph_values[0]
            with self.graph:
                impl._graph_values = [tensor_graph_values[0][...]]
                impl.graph_values_epoch = self.epoch
        else:
            with self.graph:
                impl._graph_values = [bv[...] for bv in tensor_graph_values]
                impl.graph_values_epoch = self.epoch

    def add_constant(self, tensor: Tensor, *, reason: str = "unknown") -> None:
        """Adds a realized tensor's data as a constant in the graph (not an input).

        Use this for intermediate tensors that are accessed during eager graph building
        but shouldn't be function arguments.
        """
        impl = tensor._impl
        buffers = impl._buffers
        if not buffers:
            raise TypeError("Only realized tensors may be added as constants.")

        self._debug_constant_add_count += 1
        self._debug_constant_add_buffers += len(buffers)

        with self.graph:
            const_values = []
            for buf in buffers:
                # Convert buffer to numpy and create a graph constant
                np_data = buf.to_numpy()
                const_val = ops.constant(np_data)
                const_values.append(const_val)

            impl._graph_values = const_values
            impl.graph_values_epoch = self.epoch

    def add_unrealized(self, impl: TensorImpl) -> None:
        """Registers a tensor implementation as pending computation."""
        self.unrealized[id(impl)] = impl

    def evaluate(
        self,
        tensor: Tensor,
        *extra_outputs: Any,
        return_model: bool = False,
    ) -> tuple[CompiledModel, list[driver.Buffer]] | None:
        """Main entry point: Evaluates specific tensors and their dependencies."""

        from ..common.pytree import tree_leaves
        from ..tensor.api import Tensor

        sys.last_value = None
        sys.last_traceback = None
        # gc.collect()  # Removed: too expensive for hot paths

        _debug_eval("evaluate(): begin")
        self._debug_input_add_attempts = 0
        self._debug_input_add_registered = 0
        self._debug_input_add_by_reason = {}
        self._debug_input_add_buffers_by_reason = {}
        self._debug_constant_add_count = 0
        self._debug_constant_add_buffers = 0

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
            _debug_eval(
                "evaluate(): skip finalize (all targets are already leaf inputs)"
            )
            return None

        _debug_eval(f"evaluate(): collected {len(targets)} targets")

        # --- COMPUTE CACHE KEY ---
        # We sort targets to ensure deterministic cache keys regardless of registration order.
        def get_tensor_key(t: Tensor):
            if (
                t._impl.output_refs is not None
                and t._impl.output_refs._op_hash is not None
            ):
                # Unrealized: (sorting_bucket=0, op_hash, output_index)
                return (0, t._impl.output_refs._op_hash, t._impl.output_index)
            # Realized: (sorting_bucket=1, dtype, shape, sharding)
            from ...ops.base import _make_hashable

            sharding_key = _make_hashable(t.sharding) if t.sharding else None
            return (1, str(t.dtype), tuple(int(d) for d in t.shape), sharding_key)

        # Pre-compute keys once in stable target order.
        # NOTE:
        #   Sorting via hash(target_key) can be very expensive for large nested
        #   op-hash tuples and can dominate evaluate() time before compilation.
        #   We keep the natural target order (already deterministic for callers)
        #   and avoid per-target hashing here.
        if DEBUG_LAZY_EVAL:
            import time

            prep_start = time.perf_counter()
            _debug_eval("evaluate(): computing target keys")

        target_keys = [get_tensor_key(t) for t in targets]

        if DEBUG_LAZY_EVAL:
            prep_time = time.perf_counter() - prep_start
            _debug_eval(
                f"evaluate(): target keys ready | targets={len(targets)} | key_prep={prep_time:.4f}s"
            )

        def _fingerprint_obj(
            obj: Any, memo: dict[int, int], str_memo: dict[str, int]
        ) -> int:
            """Compute a compact stable fingerprint for nested hashable structures.

            This avoids repeatedly hashing very large nested tuples (e.g. op hashes)
            during dict lookup.
            """
            if isinstance(obj, (int, bool, float, type(None))):
                return hash(obj)
            if isinstance(obj, str):
                if obj in str_memo:
                    return str_memo[obj]
                h = hash(obj)
                str_memo[obj] = h
                return h

            obj_id = id(obj)
            if obj_id in memo:
                return memo[obj_id]

            if isinstance(obj, tuple):
                h = 1469598103934665603
                for item in obj:
                    h ^= _fingerprint_obj(item, memo, str_memo)
                    h *= 1099511628211
                    h &= (1 << 63) - 1
                memo[obj_id] = h
                return h

            if isinstance(obj, list):
                h = 1099511628211
                for item in obj:
                    h ^= _fingerprint_obj(item, memo, str_memo)
                    h *= 1469598103934665603
                    h &= (1 << 63) - 1
                memo[obj_id] = h
                return h

            if isinstance(obj, dict):
                items = tuple(sorted(obj.items(), key=lambda kv: str(kv[0])))
                h = 7809847782465536322
                for k, v in items:
                    h ^= _fingerprint_obj(k, memo, str_memo)
                    h ^= _fingerprint_obj(v, memo, str_memo)
                    h *= 6364136223846793005
                    h &= (1 << 63) - 1
                memo[obj_id] = h
                return h

            return hash(str(obj))

        if DEBUG_LAZY_EVAL:
            import time

            fp_start = time.perf_counter()

        memo: dict[int, int] = {}
        str_memo: dict[str, int] = {}
        per_target_fps = [_fingerprint_obj(k, memo, str_memo) for k in target_keys]
        cache_key = tuple(sorted(per_target_fps)) if per_target_fps else None

        if DEBUG_LAZY_EVAL:
            fp_time = time.perf_counter() - fp_start
            _debug_eval(
                "cache: fingerprint key ready "
                f"in {fp_time:.4f}s | entries={len(per_target_fps)}"
            )

        if DEBUG_LAZY_EVAL:
            _debug_eval(
                f"cache: key_ready entries={len(target_keys)} cache_size={len(_GRAPH_CACHE)}"
            )

        def _buf_signature(buf: driver.Buffer) -> tuple[str, tuple[int, ...], str]:
            return (str(buf.dtype), tuple(int(d) for d in buf.shape), str(buf.device))

        def _remap_inputs_by_signature(
            signatures: list[tuple[str, tuple[int, ...], str]],
            candidates: list[driver.Buffer],
        ) -> list[driver.Buffer] | None:
            used = [False] * len(candidates)
            remapped: list[driver.Buffer] = []
            for sig in signatures:
                found_idx = None
                for i, candidate in enumerate(candidates):
                    if used[i]:
                        continue
                    if _buf_signature(candidate) == sig:
                        found_idx = i
                        break
                if found_idx is None:
                    return None
                used[found_idx] = True
                remapped.append(candidates[found_idx])
            return remapped

        # === CHECK CACHE ===
        if cache_key is not None:
            if DEBUG_LAZY_EVAL:
                import time

                cache_lookup_start = time.perf_counter()
                _debug_eval("cache: lookup start")
            entry = _GRAPH_CACHE.get(cache_key)
            if DEBUG_LAZY_EVAL:
                cache_lookup_time = time.perf_counter() - cache_lookup_start
                _debug_eval(f"cache: lookup done in {cache_lookup_time:.4f}s")
            if entry is not None:
                if len(entry) == 2:
                    cached_model, kept_indices = entry
                    input_signatures = None
                else:
                    cached_model, kept_indices, input_signatures = entry
                if DEBUG_LAZY_EVAL:
                    _debug_eval("cache: HIT")

                # Gather ALL candidate buffers from the trace in the order they would be added.
                # Since we don't have a fresh graph yet, we simulate the input ordering.
                all_candidate_tensors = self._get_input_tensors_ordered(targets)
                all_buffers = []
                for impl in all_candidate_tensors:
                    all_buffers.extend(impl._buffers)

                remapped_inputs = None
                if any(i < 0 or i >= len(all_buffers) for i in kept_indices):
                    if input_signatures is not None:
                        remapped_inputs = _remap_inputs_by_signature(
                            input_signatures, all_buffers
                        )
                    if DEBUG_LAZY_EVAL:
                        if remapped_inputs is None:
                            _debug_eval(
                                "[CACHE] STALE INPUT MAP - invalidating "
                                f"(max_index={max(kept_indices) if kept_indices else -1}, buffers={len(all_buffers)})"
                            )
                        else:
                            _debug_eval("[CACHE] REMAP mode=signature")
                    if remapped_inputs is None:
                        _GRAPH_CACHE.pop(cache_key, None)
                else:
                    remapped_inputs = [all_buffers[i] for i in kept_indices]

                if remapped_inputs is not None:
                    inputs = remapped_inputs

                    if DEBUG_LAZY_EVAL:
                        preview = [
                            (tuple(inp.shape), str(inp.dtype)) for inp in inputs[:5]
                        ]
                        _debug_eval(
                            f"[CACHE] inputs: count={len(inputs)} preview={preview}"
                        )

                    _debug_eval("cache: executing cached model")
                    seed_val, *results = cached_model(*inputs)

                    result_idx = 0
                    for t in targets:
                        n_shards = t.num_shards
                        t_results = results[result_idx : result_idx + n_shards]

                        if n_shards > 1:
                            t._impl._buffers = list(t_results)
                        else:
                            t.buffers = t_results[0]

                        t._value = None
                        t.real = True
                        t._impl._graph_values = []
                        result_idx += n_shards
                        self.unrealized.pop(id(t), None)

                    self._finalize_evaluation(seed_value=seed_val.item())
                    self._cleanup_trace(targets)
                    _debug_eval("cache: cached execution complete")
                    return (cached_model, inputs) if return_model else None

        # === CACHE MISS - Build and compile graph ===
        if DEBUG_LAZY_EVAL:
            _debug_eval("cache: MISS")

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
        _debug_eval("miss: replaying trace to build graph")
        self._replay_trace_to_build_graph(targets)

        # Build graph outputs
        _debug_eval("miss: building graph outputs")
        all_graph_values = []
        value_map = []

        with self.graph:
            for t in targets:
                if t._impl.graph_values_epoch != self.epoch:
                    t._impl._graph_values = []

                if not t._impl._graph_values and t._impl.is_realized:
                    self.add_input(t, reason="target_realized")

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
        _debug_eval("miss: lowering graph")
        _core.lower(module, [builtin.passes.RemoveDeadValues()])
        _debug_eval("miss: removing unused arguments")
        _remove_unused_arguments(self.graph)

        inputs: list[driver.Buffer] = []
        for inp in self.graph.inputs:
            buffers = self.sources.get(inp._mlir_value)
            if buffers is None:
                raise RuntimeError("Missing buffers for graph input")
            inputs.append(buffers)

        if DEBUG_LAZY_EVAL:
            from collections import Counter

            shape_counter: Counter[tuple[int, ...]] = Counter(
                tuple(int(d) for d in b.shape) for b in inputs
            )
            scalar_inputs = shape_counter.get((), 0)
            top_shapes = shape_counter.most_common(6)
            _debug_eval(
                "inputs: "
                f"graph_inputs={len(inputs)} unique_tensors={len(self._input_refs)} "
                f"add_attempts={self._debug_input_add_attempts} "
                f"registered_tensors={self._debug_input_add_registered} "
                f"scalar_inputs={scalar_inputs} "
                f"constants={self._debug_constant_add_count} "
                f"constant_buffers={self._debug_constant_add_buffers}"
            )
            _debug_eval(
                f"inputs: add_by_reason={self._debug_input_add_by_reason} "
                f"buffers_by_reason={self._debug_input_add_buffers_by_reason}"
            )
            _debug_eval(f"inputs: top_shapes={top_shapes}")

        if DEBUG_LAZY_EVAL:
            import time

            _debug_eval(f"miss: compiling graph with {len(self.graph.inputs)} inputs")
            start_comp = time.perf_counter()

        model = _session().load(self.graph)

        if DEBUG_LAZY_EVAL:
            comp_time = time.perf_counter() - start_comp
            _debug_eval(f"miss: compilation finished in {comp_time:.2f}s")
            start_exec = time.perf_counter()

        seed_val, *results = model(*inputs)

        if DEBUG_LAZY_EVAL:
            exec_time = time.perf_counter() - start_exec
            _debug_eval(f"miss: execution finished in {exec_time:.2f}s")

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
                    _, buffers = shard_results[0]
                    t.buffers = buffers
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

            used_bufferss = [
                self.sources.get(inp._mlir_value) for inp in self.graph.inputs
            ]

            kept_indices = []
            for buffers in used_bufferss:
                found = False
                for i, s in enumerate(all_candidate_buffers):
                    if s is buffers:
                        kept_indices.append(i)
                        found = True
                        break
                if not found:
                    raise RuntimeError("Could not map graph input back to trace")

            input_signatures = [
                _buf_signature(b) for b in used_bufferss if b is not None
            ]
            _GRAPH_CACHE[cache_key] = (model, kept_indices, input_signatures)

        self._finalize_evaluation(seed_value=seed_val.item())
        self._cleanup_trace(targets)
        _debug_eval("miss: evaluation complete")
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
        _target_impl_ids = {id(t._impl) for t in targets}

        def clean(impl: TensorImpl) -> None:
            if id(impl) in visited:
                return
            visited.add(id(impl))
            impl._graph_values = []
            impl.graph_values_epoch = -1

            # Thorough cleanup: if this branch isn't being traced for autograd,
            # clear the output_refs link to allow the OpNode and its inputs
            # to be garbage collected immediately.
            # IMPORTANT: Only clear output_refs if the tensor is now realized,
            # otherwise we lose the trace for unrealized tensors that may be
            # used in future computations.
            if impl.output_refs and not impl.is_traced and impl.is_realized:
                refs = impl.output_refs
                impl.output_refs = None
                impl.output_index = None

                for arg in pytree.tree_leaves(refs.op_args):
                    if isinstance(arg, TensorImpl):
                        clean(arg)

        # Traverse and clean
        for t in targets:
            clean(t._impl)

    @staticmethod
    def _topo_sort_opnodes(targets: list[Tensor]) -> list[OpNode]:
        """Topologically sort OpNodes reachable from *targets*."""
        from ..common import pytree
        from ..tensor.impl import TensorImpl

        visited: set[int] = set()
        result: list = []

        def dfs(opnode) -> None:
            if id(opnode) in visited:
                return
            for arg in pytree.tree_leaves(opnode.op_args):
                if (
                    isinstance(arg, TensorImpl)
                    and not arg.is_realized
                    and arg.output_refs
                ):
                    dfs(arg.output_refs)
            visited.add(id(opnode))
            result.append(opnode)

        for t in targets:
            if t._impl.output_refs:
                dfs(t._impl.output_refs)
        return result

    def _replay_trace_to_build_graph(self, targets: list[Tensor]) -> None:
        """Walk OpNode DAG and execute operations to build MAX graph."""
        import time

        from ..common import pytree
        from ..tensor.api import Tensor
        from ..tensor.impl import TensorImpl

        topo_start = time.perf_counter()
        opnodes_topo = self._topo_sort_opnodes(targets)
        topo_time = time.perf_counter() - topo_start
        _debug_eval(
            f"replay: topo-sort complete | opnodes={len(opnodes_topo)} | time={topo_time:.4f}s"
        )

        replay_start = time.perf_counter()
        executed_ops = 0
        skipped_ops = 0

        # Execute each OpNode
        for opnode in opnodes_topo:
            # Skip if outputs already valid
            if all(
                ref.graph_values_epoch == self.epoch and ref._graph_values
                for ref in opnode._refs
                if ref is not None
            ):
                skipped_ops += 1
                continue

            # Ensure inputs have graph values
            for arg in pytree.tree_leaves(opnode.op_args):
                if (
                    isinstance(arg, TensorImpl)
                    and (arg.graph_values_epoch != self.epoch or not arg._graph_values)
                    and arg.is_realized
                ):
                    self.add_input(Tensor(impl=arg), reason="replay_realized_arg")

            # Execute operation
            def to_tensor(x):
                return Tensor(impl=x) if isinstance(x, TensorImpl) else x

            op_args = pytree.tree_map(to_tensor, opnode.op_args)

            with self.graph:
                raw_result = opnode.op.execute(op_args, opnode.op_kwargs or {})
            executed_ops += 1

            # Extract graph values
            if isinstance(raw_result, tuple) and len(raw_result) == 3:
                shard_graph_values, _, _ = raw_result
            elif hasattr(raw_result, "shard_graph_values"):
                shard_graph_values = raw_result.shard_graph_values
            else:
                shard_graph_values = raw_result

            # Store graph values to output refs
            if isinstance(shard_graph_values, (list, tuple)) and not isinstance(
                shard_graph_values[0] if shard_graph_values else None,
                (list, tuple, dict),
            ):
                if len(opnode._refs) == 1 and opnode._refs[0] is not None:
                    opnode._refs[0]._graph_values = shard_graph_values
                    opnode._refs[0].graph_values_epoch = self.epoch
            else:
                if isinstance(
                    shard_graph_values[0] if shard_graph_values else None, (list, tuple)
                ):
                    unzipped = (
                        list(zip(*shard_graph_values, strict=False))
                        if shard_graph_values
                        else []
                    )
                    for i, ref in enumerate(opnode._refs):
                        if ref is not None and i < len(unzipped):
                            ref._graph_values = list(unzipped[i])
                            ref.graph_values_epoch = self.epoch

        replay_time = time.perf_counter() - replay_start
        _debug_eval(
            "replay: execute complete "
            f"| executed={executed_ops} skipped={skipped_ops} "
            f"| time={replay_time:.4f}s"
        )

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

        visited_impls: set[int] = set()
        ordered_inputs: list[TensorImpl] = []

        opnodes_topo = self._topo_sort_opnodes(targets)
        # Important: this must match EXACTLY the order in _replay_trace_to_build_graph
        for opnode in opnodes_topo:
            for arg in pytree.tree_leaves(opnode.op_args):
                if isinstance(arg, TensorImpl) and id(arg) not in visited_impls:
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
