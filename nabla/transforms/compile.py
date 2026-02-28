# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Compile transform: JIT compilation with caching for nabla functions.

Traces a function to build a MAX graph, then caches the compiled model.
Subsequent calls with matching signatures use the cached version.

Key mechanisms:
- EAGER_MAX_GRAPH=True: ops build graph immediately during trace
- Dynamic dimensions: use SymbolicDim for batch-independent compilation
- Smart caching: deduplicates inputs via buffer identity checks
"""

from __future__ import annotations

import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from max.graph import ops
from max.graph.dim import SymbolicDim

from ..core import GRAPH, tree_flatten, tree_unflatten
from ..core.tensor.api import Tensor
from ..ops.creation import full

if TYPE_CHECKING:
    from max.graph.model import CompiledModel

    from ..core.common.pytree import PyTreeDef
    from ..core.sharding.spec import ShardingSpec

T = TypeVar("T")


@dataclass
class CompilationStats:
    """Runtime statistics for a :class:`CompiledFunction`.

    Attributes:
        hits: Number of cache hit executions (fast path).
        misses: Number of cache misses that triggered recompilation.
        fallbacks: Number of calls that fell back to eager execution.
        total_compile_time_ms: Cumulative compilation time in milliseconds.
        total_cached_exec_time_ms: Cumulative execution time for cache hits.
        cache_size: Current number of entries in the LRU cache.
    """

    hits: int = 0
    misses: int = 0
    fallbacks: int = 0
    total_compile_time_ms: float = 0.0
    total_cached_exec_time_ms: float = 0.0
    cache_size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses + self.fallbacks
        return (self.hits / total * 100) if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"CompilationStats(hits={self.hits}, misses={self.misses}, "
            f"fallbacks={self.fallbacks}, hit_rate={self.hit_rate:.1f}%)"
        )


@dataclass(frozen=True)
class _CacheKey:
    """Cache key: (tensor signatures, static values, pytree structure)."""

    tensor_sigs: tuple[tuple[tuple[int | str, ...], str, Any], ...]
    static_vals: tuple[Any, ...]
    treedef: PyTreeDef


@dataclass
class _CachedModel:
    """Cached compilation result."""

    model: CompiledModel
    output_treedef: PyTreeDef
    output_tensor_mask: list[bool]
    output_static_values: list[Any]
    output_shard_counts: list[int]
    output_shardings: list[ShardingSpec | None]
    # Fast path metadata for cache hits
    tensor_indices: list[int]  # Which positions in flat args are tensors
    input_treedef: PyTreeDef  # Cached input tree structure
    input_plan: list[
        tuple[str, Any]
    ]  # [("arg", arg_buf_pos) | ("captured", driver.Buffer)]


class CompiledFunction(Generic[T]):
    """A JIT-compiled function with signature-based LRU caching.

    On each call, the argument signatures (shapes, dtypes, pytree structure)
    are hashed and looked up in the cache. On a cache hit the pre-compiled
    MAX graph model is executed directly; on a miss the function is traced
    and compiled before execution.

    Attributes:
        stats: A :class:`CompilationStats` instance tracking hits, misses,
            fallbacks, and compile time.
    """

    def __init__(
        self,
        fn: Callable[..., T],
        *,
        fullgraph: bool = False,
        max_cache_size: int = 64,
        dynamic_dims: dict[int, dict[int, str]] | None = None,
    ):
        self.__wrapped__ = fn
        self.__name__ = getattr(fn, "__name__", repr(fn))
        self.__doc__ = fn.__doc__

        self.fullgraph = fullgraph
        self.max_cache_size = max_cache_size
        self.dynamic_dims = dynamic_dims or {}
        self._cache: OrderedDict[_CacheKey, _CachedModel] = OrderedDict()
        self._stats = CompilationStats()

    @property
    def stats(self) -> CompilationStats:
        self._stats.cache_size = len(self._cache)
        return self._stats

    def clear_cache(self) -> None:
        self._cache.clear()

    def _normalize_optimizer_state_for_compile(self, value: Any) -> Any:
        """Convert optimizer step scalars to tensor leaves for compiled calls.

        Functional optimizer states often store "step" as a Python scalar.
        In compiled mode this scalar becomes part of static cache keys and can
        trigger retracing every iteration as it changes. For optimizer-like
        dicts, convert scalar step to a 0-D tensor so it becomes a runtime
        tensor input instead of static metadata.
        """
        if isinstance(value, dict):
            normalized = {
                k: self._normalize_optimizer_state_for_compile(v)
                for k, v in value.items()
            }
            has_optimizer_state = (
                ("m" in normalized and "v" in normalized)
                or "momentum_buffers" in normalized
            )
            if (
                has_optimizer_state
                and "step" in normalized
                and isinstance(normalized["step"], (int, float))
            ):
                normalized["step"] = full((), float(normalized["step"]))
            return normalized

        if isinstance(value, tuple):
            return tuple(self._normalize_optimizer_state_for_compile(v) for v in value)

        if isinstance(value, list):
            return [self._normalize_optimizer_state_for_compile(v) for v in value]

        return value

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Main entry: check cache, trace if miss, execute."""
        args = tuple(self._normalize_optimizer_state_for_compile(arg) for arg in args)
        kwargs = {
            k: self._normalize_optimizer_state_for_compile(v)
            for k, v in kwargs.items()
        }
        flat, treedef = tree_flatten((args, kwargs))

        # Fast path: reuse structure from last cached call
        if self._cache:
            last_cached = next(reversed(self._cache.values()))
            if treedef == last_cached.input_treedef:
                key = self._build_cache_key(flat, last_cached.tensor_indices, treedef)
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._execute_cached_fast(self._cache[key], flat)

        # Slow path: discover tensor positions
        tensor_indices = [i for i, x in enumerate(flat) if isinstance(x, Tensor)]

        # Realize unrealized tensors before caching
        unrealized = [flat[i] for i in tensor_indices if not flat[i].is_realized]
        if unrealized:
            GRAPH.evaluate(*unrealized)

        # Build key and check cache
        key = self._build_cache_key(flat, tensor_indices, treedef)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._execute_cached(key, flat, tensor_indices)

        # Cache miss - trace and compile
        return self._trace_and_compile(args, kwargs, flat, tensor_indices, key, treedef)

    def _build_cache_key(
        self, flat: list[Any], tensor_indices: list[int], treedef: PyTreeDef
    ) -> _CacheKey:
        """Build cache key from tensor specs and static values."""
        tensor_sigs = []
        static_vals = []

        tensor_counter = 0
        for i, x in enumerate(flat):
            if i in tensor_indices:
                sig = self._tensor_signature(x, tensor_counter)
                tensor_sigs.append(sig)
                tensor_counter += 1
            else:
                static_vals.append(self._make_hashable_static(x))

        return _CacheKey(tuple(tensor_sigs), tuple(static_vals), treedef)

    def _make_hashable_static(self, value: Any) -> Any:
        """Convert static cache-key values to hashable form."""
        if isinstance(value, (str, bytes, int, float, bool, type(None))):
            return value

        if isinstance(value, tuple):
            return tuple(self._make_hashable_static(v) for v in value)

        if isinstance(value, list):
            return tuple(self._make_hashable_static(v) for v in value)

        if isinstance(value, dict):
            return tuple(
                sorted(
                    (self._make_hashable_static(k), self._make_hashable_static(v))
                    for k, v in value.items()
                )
            )

        if isinstance(value, (set, frozenset)):
            return frozenset(self._make_hashable_static(v) for v in value)

        try:
            hash(value)
            return value
        except TypeError:
            return (type(value).__qualname__, repr(value))

    def _tensor_signature(
        self, tensor: Tensor, arg_idx: int
    ) -> tuple[tuple[int | str, ...], str, Any]:
        """Build tensor signature for cache key, applying dynamic_dims."""
        # Apply symbolic markers to dynamic dimensions
        shape = list(tensor.shape)
        if arg_idx in self.dynamic_dims:
            for dim_idx, sym_name in self.dynamic_dims[arg_idx].items():
                if dim_idx < len(shape):
                    shape[dim_idx] = f"${sym_name}"

        # Extract sharding key
        sharding_key = (
            self._extract_sharding_key(tensor.sharding) if tensor.sharding else None
        )
        return (tuple(shape), str(tensor.dtype), sharding_key)

    def _extract_sharding_key(self, sharding: ShardingSpec | None) -> Any:
        """Extract hashable sharding info."""
        mesh = getattr(sharding, "mesh", None)
        mesh_key = None
        if mesh is not None:
            mesh_key = (
                getattr(mesh, "name", None),
                tuple(mesh.shape) if hasattr(mesh, "shape") and mesh.shape else (),
                (
                    tuple(mesh.axis_names)
                    if hasattr(mesh, "axis_names") and mesh.axis_names
                    else ()
                ),
            )

        dim_specs = getattr(sharding, "dim_specs", [])
        dim_specs_key = (
            tuple(
                (tuple(ds.axes) if ds.axes else (), bool(ds.partial))
                for ds in dim_specs
            )
            if dim_specs
            else ()
        )

        return (mesh_key, dim_specs_key)

    def _build_input_types(self, flat: list[Any], tensor_indices: list[int]) -> list:
        """Build TensorType list for graph with symbolic dims where specified."""
        from max.graph import DeviceRef, TensorType

        input_types = []
        for arg_idx, flat_idx in enumerate(tensor_indices):
            tensor = flat[flat_idx]
            buf = tensor._impl._buffers[0]

            # Build shape with SymbolicDim for dynamic dimensions
            shape = list(tensor.shape)
            if arg_idx in self.dynamic_dims:
                shape = [
                    (
                        SymbolicDim(self.dynamic_dims[arg_idx][i])
                        if i in self.dynamic_dims[arg_idx]
                        else int(d)
                    )
                    for i, d in enumerate(shape)
                ]

            input_types.append(
                TensorType(buf.dtype, shape, DeviceRef.from_device(buf.device))
            )

        return input_types

    def _trace_and_compile(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        flat: list[Any],
        tensor_indices: list[int],
        key: _CacheKey,
        treedef: PyTreeDef,
    ) -> T:
        """Trace function and compile to MAX graph."""
        from ..core.graph import engine as _graph_engine
        from ..config import (
            _EAGER_MAX_GRAPH,
            _VERIFY_EAGER_SHAPES,
            _TRACING,
        )

        # Validate: no sharded + dynamic dims yet
        if self.dynamic_dims and any(flat[i].is_sharded for i in tensor_indices):
            raise NotImplementedError(
                "Compilation of sharded tensors with dynamic dimensions is not yet supported."
            )

        t0 = time.perf_counter()

        # Use ContextVar tokens for safe, exception-proof save/restore
        token_eager = _EAGER_MAX_GRAPH.set(True)  # Ops build graph during trace
        token_verify = _VERIFY_EAGER_SHAPES.set(False)  # Skip shape checks
        token_tracing = _TRACING.set(True)  # Suppress realization during trace

        try:
            # Start a fresh epoch so reused tensors cannot leak stale graph values
            # from a previously reset graph region.
            new_epoch = _graph_engine._GRAPH_EPOCH.get() + 1
            _graph_engine._GRAPH_EPOCH.set(new_epoch)
            GRAPH.epoch = new_epoch

            # Prepare graph with input types
            input_types = (
                self._build_input_types(flat, tensor_indices)
                if self.dynamic_dims
                else None
            )
            GRAPH._reset(GRAPH.context, 0, input_types=input_types)

            # Register inputs
            self._register_inputs(flat, tensor_indices, input_types)

            # Trace: run function to build graph
            result = self.__wrapped__(*args, **kwargs)

            # Extract outputs and compile
            return self._finalize_compilation(
                result, key, treedef, flat, tensor_indices, t0
            )

        finally:
            _EAGER_MAX_GRAPH.reset(token_eager)
            _VERIFY_EAGER_SHAPES.reset(token_verify)
            _TRACING.reset(token_tracing)

    def _register_inputs(
        self, flat: list[Any], tensor_indices: list[int], input_types: list | None
    ) -> None:
        """Register input tensors with the graph."""
        if input_types:
            # Symbolic dims: link existing graph inputs to buffers
            for arg_idx, flat_idx in enumerate(tensor_indices):
                tensor = flat[flat_idx]
                impl = tensor._impl
                buf = impl._buffers[0]

                graph_input = GRAPH.graph.inputs[arg_idx]
                GRAPH.sources[graph_input._mlir_value] = buf
                GRAPH._input_refs.append(tensor)

                with GRAPH.graph:
                    impl._graph_values = [graph_input[...]]
                    impl.graph_values_epoch = GRAPH.epoch
        else:
            # Standard path: add_input creates graph nodes
            for flat_idx in tensor_indices:
                GRAPH.add_input(flat[flat_idx])

    def _finalize_compilation(
        self,
        result: T,
        key: _CacheKey,
        treedef: PyTreeDef,
        flat: list[Any],
        tensor_indices: list[int],
        t0: float,
    ) -> T:
        """Finalize graph, compile, execute, and cache."""
        from ..core.common.context import _session

        # Extract outputs
        flat_out, out_treedef = tree_flatten(result)
        output_mask = [isinstance(x, Tensor) for x in flat_out]
        output_tensors = [x for x in flat_out if isinstance(x, Tensor)]
        static_outputs = [x for x in flat_out if not isinstance(x, Tensor)]

        if not output_tensors:
            self._stats.misses += 1
            return result

        # Set graph outputs
        with GRAPH.graph:
            ops.random.set_seed(0)

            all_graph_values = []
            for tensor in output_tensors:
                if tensor._impl.graph_values_epoch != GRAPH.epoch:
                    tensor._impl._graph_values = []

                if not tensor._impl._graph_values:
                    if not tensor.is_realized:
                        GRAPH.evaluate(tensor)
                    GRAPH.add_input(tensor)  # Pass-through

                if not tensor._impl._graph_values:
                    raise RuntimeError(f"Output tensor {id(tensor)} has no graph values")

                all_graph_values.extend(tensor._impl._graph_values)

            seed_out = ops.random._peek_seed()
            GRAPH.graph.output(seed_out, *all_graph_values)

        # Compile and execute
        model = _session().load(GRAPH.graph)
        graph_inputs = [GRAPH.sources[inp._mlir_value] for inp in GRAPH.graph.inputs]

        # Build input replay plan for cached execution.
        # Graph inputs can include both call-time args and captured tensors (closures).
        arg_buffers: list[Any] = []
        for i in tensor_indices:
            arg_buffers.extend(flat[i]._impl._buffers)

        positions_by_id: dict[int, deque[int]] = defaultdict(deque)
        for pos, buf in enumerate(arg_buffers):
            positions_by_id[id(buf)].append(pos)

        input_plan: list[tuple[str, Any]] = []
        for buf in graph_inputs:
            q = positions_by_id.get(id(buf))
            if q and len(q) > 0:
                input_plan.append(("arg", q.popleft()))
            else:
                input_plan.append(("captured", buf))

        inputs = graph_inputs
        seed_val, *result_buffers = model(*inputs)

        # Assign result buffers to output tensors
        buf_idx = 0
        output_shard_counts, output_shardings = [], []

        for tensor in output_tensors:
            n_shards = len(tensor._impl._graph_values) or 1
            output_shard_counts.append(n_shards)
            output_shardings.append(tensor.sharding)

            if n_shards == 1:
                tensor.buffers = result_buffers[buf_idx]
            else:
                tensor._impl._buffers = result_buffers[buf_idx : buf_idx + n_shards]
            buf_idx += n_shards

        # Cache the model
        cached = _CachedModel(
            model=model,
            output_treedef=out_treedef,
            output_tensor_mask=output_mask,
            output_static_values=static_outputs,
            output_shard_counts=output_shard_counts,
            output_shardings=output_shardings,
            tensor_indices=tensor_indices,
            input_treedef=treedef,
            input_plan=input_plan,
        )
        self._add_to_cache(key, cached, t0)
        return result

    def _add_to_cache(self, key: _CacheKey, cached: _CachedModel, t0: float) -> None:
        """Add to cache with LRU eviction."""
        self._cache[key] = cached
        if len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)
        self._stats.total_compile_time_ms += (time.perf_counter() - t0) * 1000
        self._stats.misses += 1

    def _execute_cached_fast(self, cached: _CachedModel, flat: list[Any]) -> T:
        """Fast execution using cached tensor_indices."""
        return self._execute_cached_impl(
            cached, flat, tensor_indices=cached.tensor_indices
        )

    def _execute_cached(
        self, key: _CacheKey, flat: list[Any], tensor_indices: list[int]
    ) -> T:
        """Execute cached model with given tensor indices."""
        return self._execute_cached_impl(
            self._cache[key], flat, tensor_indices=tensor_indices
        )

    def _execute_cached_impl(
        self, cached: _CachedModel, flat: list[Any], tensor_indices: list[int]
    ) -> T:
        """Core execution logic for cached models."""
        t0 = time.perf_counter()

        # Gather argument buffers in canonical arg order
        arg_buffers: list[Any] = []
        for i in tensor_indices:
            arg_buffers.extend(flat[i]._impl._buffers)

        # Rebuild model inputs using stored input plan (args + captured closures)
        inputs: list[Any] = []
        for src_kind, src_value in cached.input_plan:
            if src_kind == "arg":
                inputs.append(arg_buffers[int(src_value)])
            else:
                inputs.append(src_value)

        # Execute model
        _, *outputs = cached.model(*inputs)

        # Reconstruct outputs
        all_leaves = []
        out_buf_iter = iter(outputs)
        shard_counts_iter = iter(cached.output_shard_counts)
        shardings_iter = iter(cached.output_shardings)
        static_iter = iter(cached.output_static_values)

        for is_tensor in cached.output_tensor_mask:
            if is_tensor:
                n_shards = next(shard_counts_iter)
                sharding = next(shardings_iter)
                shard_bufs = [next(out_buf_iter) for _ in range(n_shards)]

                t = (
                    Tensor(buffers=shard_bufs[0])
                    if n_shards == 1
                    else Tensor._create_unsafe(bufferss=shard_bufs)
                )
                t.sharding = sharding
                all_leaves.append(t)
            else:
                all_leaves.append(next(static_iter))

        self._stats.total_cached_exec_time_ms += (time.perf_counter() - t0) * 1000
        self._stats.hits += 1
        return tree_unflatten(cached.output_treedef, all_leaves)

    def __repr__(self) -> str:
        return f"<CompiledFunction {self.__name__} cache_size={len(self._cache)}>"


def compile(
    fn: Callable[..., T] | None = None,
    *,
    fullgraph: bool = False,
    max_cache_size: int = 64,
    dynamic_dims: dict[int, dict[int, str]] | None = None,
) -> CompiledFunction[T] | Callable[..., CompiledFunction[Any]]:
    """Compile a function for cached graph execution.

    Args:
        fn: Function to compile. If None, returns a decorator.
        fullgraph: If True, error on side effects. If False, fall back to eager.
        max_cache_size: Maximum cached compilations (LRU eviction).
        dynamic_dims: Mark dimensions as symbolic. Format: {arg_idx: {dim_idx: "name"}}
            E.g., {0: {0: "batch"}} makes arg 0, dim 0 dynamic.

    Returns:
        CompiledFunction wrapping the original function.

    Examples:
        @compile
        def fn(x): return x * x

        @compile(dynamic_dims={0: {0: "batch"}})
        def batched(x, W): return x @ W  # One compilation, any batch size
    """
    if fn is None:
        return lambda f: compile(
            f,
            fullgraph=fullgraph,
            max_cache_size=max_cache_size,
            dynamic_dims=dynamic_dims,
        )
    return CompiledFunction(
        fn,
        fullgraph=fullgraph,
        max_cache_size=max_cache_size,
        dynamic_dims=dynamic_dims,
    )


__all__ = ["compile", "CompiledFunction", "CompilationStats"]
