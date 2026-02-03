# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Compile transform: JIT compilation with caching for nabla functions.

The compile decorator traces a function, builds a MAX graph, and caches
the compiled model for subsequent calls with matching signatures.

Key Design Principles:
1. Build cache key from tensor specs (shape, dtype, sharding) with dynamic dim support
2. Set EAGER_MAX_GRAPH=True during tracing so ops build the graph immediately  
3. Use GRAPH.add_input with optional symbolic shapes for dynamic dimensions
4. Rely on existing hydrate/add_input deduplication to avoid duplicate graph inputs
5. After tracing, compile and execute; cache the model for future calls
"""

from __future__ import annotations

import time
import warnings
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from max.graph import ops
from max.graph.dim import SymbolicDim

from ..core import GRAPH, tree_flatten, tree_unflatten
from ..core.tensor import Tensor

T = TypeVar("T")


@dataclass
class CompilationStats:
    """Compilation statistics."""

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

    tensor_sigs: tuple[Any, ...]
    static_vals: tuple[Any, ...]
    treedef: Any


@dataclass
class _CachedModel:
    """Cached compilation result."""

    model: Any
    output_treedef: Any
    output_tensor_mask: list[bool]
    output_static_values: list[Any]
    output_shard_counts: list[int]
    output_shardings: list[Any]


class CompiledFunction(Generic[T]):
    """A compiled function with caching."""

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

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Main entry point: check cache, trace if miss, execute cached model."""
        
        # Step 1: Flatten inputs and identify tensors
        flat, treedef = tree_flatten((args, kwargs))
        tensor_indices = [i for i, x in enumerate(flat) if isinstance(x, Tensor)]
        
        # Step 2: Ensure all input tensors are realized
        unrealized = [flat[i] for i in tensor_indices if not flat[i].is_realized]
        if unrealized:
            GRAPH.evaluate(*unrealized)

        # Step 3: Build cache key (shape + dtype + sharding, respecting dynamic_dims)
        key = self._build_cache_key(flat, tensor_indices, treedef)

        # Step 4: Check cache
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._execute_cached(key, flat, tensor_indices)

        # Step 5: Cache miss - trace and compile
        return self._trace_and_compile(args, kwargs, flat, tensor_indices, key)

    def _build_cache_key(
        self, flat: list[Any], tensor_indices: list[int], treedef: Any
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
                static_vals.append(x)
        
        return _CacheKey(tuple(tensor_sigs), tuple(static_vals), treedef)

    def _tensor_signature(self, tensor: Tensor, arg_idx: int) -> tuple:
        """Build tensor signature for cache key, applying dynamic_dims."""
        # Build shape with symbolic markers for dynamic dims
        shape = list(tensor.shape)
        if arg_idx in self.dynamic_dims:
            dim_map = self.dynamic_dims[arg_idx]
            for dim_idx, sym_name in dim_map.items():
                if dim_idx < len(shape):
                    shape[dim_idx] = f"${sym_name}"
        
        # Include sharding info for correctness
        sharding_key = None
        if tensor.sharding is not None:
            sharding = tensor.sharding
            mesh = sharding.mesh if hasattr(sharding, 'mesh') else None
            mesh_key = None
            if mesh is not None:
                mesh_key = (
                    getattr(mesh, 'name', None),
                    tuple(mesh.shape) if hasattr(mesh, 'shape') and mesh.shape else (),
                    tuple(mesh.axis_names) if hasattr(mesh, 'axis_names') and mesh.axis_names else (),
                )
            dim_specs = sharding.dim_specs if hasattr(sharding, 'dim_specs') else []
            dim_specs_key = tuple(
                (tuple(ds.axes) if ds.axes else (), bool(ds.partial))
                for ds in dim_specs
            ) if dim_specs else ()
            sharding_key = (mesh_key, dim_specs_key)
        
        return (tuple(shape), str(tensor.dtype), sharding_key)

    def _build_symbolic_shape(self, tensor: Tensor, arg_idx: int) -> list | None:
        """Build shape for MAX graph with SymbolicDim if dynamic_dims specified."""
        if arg_idx not in self.dynamic_dims:
            return None  # Use default shape
        
        dims = list(tensor.shape)
        dim_map = self.dynamic_dims[arg_idx]
        result = []
        for i, d in enumerate(dims):
            if i in dim_map:
                result.append(SymbolicDim(dim_map[i]))
            else:
                result.append(int(d))
        return result

    def _build_input_types(self, flat: list[Any], tensor_indices: list[int]) -> list:
        """Build TensorType list for graph creation with proper symbolic dims."""
        from max.graph import TensorType, DeviceRef
        
        input_types = []
        for arg_idx, flat_idx in enumerate(tensor_indices):
            tensor = flat[flat_idx]
            
            # Get shape (with symbolic dims if specified)
            sym_shape = self._build_symbolic_shape(tensor, arg_idx)
            shape = sym_shape if sym_shape is not None else list(tensor.shape)
            
            # Get dtype and device from buffer
            buf = tensor._impl._buffers[0]
            dtype = buf.dtype
            device_ref = DeviceRef.from_device(buf.device)
            
            input_types.append(TensorType(dtype, shape, device_ref))
        
        return input_types

    def _trace_and_compile(
        self,
        args: tuple,
        kwargs: dict,
        flat: list[Any],
        tensor_indices: list[int],
        key: _CacheKey,
    ) -> T:
        """Trace the function and compile it."""
        from .. import config as nabla_config
        from ..core.common.context import _session

        # Safety: reject sharded + dynamic for now
        if self.dynamic_dims:
            for i in tensor_indices:
                if flat[i].is_sharded:
                    raise NotImplementedError(
                        "Compilation of sharded tensors with dynamic dimensions is not yet supported."
                    )

        # Save original config
        orig_eager = nabla_config.EAGER_MAX_GRAPH
        orig_verify = nabla_config.VERIFY_EAGER_SHAPES
        t0 = time.perf_counter()

        try:
            # Set eager mode for tracing (ops build graph immediately)
            nabla_config.EAGER_MAX_GRAPH = True
            nabla_config.VERIFY_EAGER_SHAPES = False

            # Build input types upfront (critical for symbolic dims!)
            input_types = self._build_input_types(flat, tensor_indices) if self.dynamic_dims else None
            
            # Reset graph with proper input types
            GRAPH._reset(GRAPH.context, 0, input_types=input_types)

            # Register input tensors - link buffers to graph inputs
            if input_types:
                # When input_types are provided, graph already has the inputs
                # We just need to set up the tensor graph values and source mapping
                for arg_idx, flat_idx in enumerate(tensor_indices):
                    tensor = flat[flat_idx]
                    impl = tensor._impl
                    buf = impl._buffers[0]
                    
                    graph_input = GRAPH.graph.inputs[arg_idx]
                    GRAPH.sources[graph_input._mlir_value] = buf
                    GRAPH._input_refs.append(tensor)
                    
                    # Set the tensor's graph values
                    with GRAPH.graph:
                        impl._graph_values = [graph_input[...]]
                        impl.graph_values_epoch = GRAPH.epoch
            else:
                # Standard path without symbolic dims
                for arg_idx, flat_idx in enumerate(tensor_indices):
                    tensor = flat[flat_idx]
                    GRAPH.add_input(tensor)

            # Execute the function (ops will build graph eagerly)
            result = self.__wrapped__(*args, **kwargs)

            # Flatten outputs
            flat_out, out_treedef = tree_flatten(result)
            output_mask = [isinstance(x, Tensor) for x in flat_out]
            output_tensors = [x for x in flat_out if isinstance(x, Tensor)]
            static_outputs = [x for x in flat_out if not isinstance(x, Tensor)]

            if not output_tensors:
                # No tensor outputs - just return (rare case)
                self._stats.misses += 1
                return result

            # Finalize graph outputs
            with GRAPH.graph:
                ops.random.set_seed(0)
                
                all_graph_values = []
                for tensor in output_tensors:
                    if not tensor._impl._graph_values:
                        # Output is a pass-through of input (no ops applied)
                        GRAPH.add_input(tensor)
                    all_graph_values.extend(tensor._impl._graph_values)
                
                seed_out = ops.random._peek_seed()
                GRAPH.graph.output(seed_out, *all_graph_values)

            # Compile the graph
            model = _session().load(GRAPH.graph)

            # Execute immediately to get result buffers
            inputs = [GRAPH.sources[inp._mlir_value] for inp in GRAPH.graph.inputs]
            seed_val, *result_buffers = model(*inputs)

            # Assign buffers to output tensors
            buf_idx = 0
            output_shard_counts = []
            output_shardings = []
            
            for tensor in output_tensors:
                n_shards = len(tensor._impl._graph_values) or 1
                output_shard_counts.append(n_shards)
                output_shardings.append(tensor.sharding)
                
                if n_shards == 1:
                    tensor.buffers = result_buffers[buf_idx]
                else:
                    tensor._impl._buffers = result_buffers[buf_idx : buf_idx + n_shards]
                buf_idx += n_shards

            # Cache the compiled model
            cached = _CachedModel(
                model=model,
                output_treedef=out_treedef,
                output_tensor_mask=output_mask,
                output_static_values=static_outputs,
                output_shard_counts=output_shard_counts,
                output_shardings=output_shardings,
            )
            self._add_to_cache(key, cached, t0)

            return result

        finally:
            # Restore original config
            nabla_config.EAGER_MAX_GRAPH = orig_eager
            nabla_config.VERIFY_EAGER_SHAPES = orig_verify

    def _add_to_cache(self, key: _CacheKey, cached: _CachedModel, t0: float) -> None:
        """Add to cache with LRU eviction."""
        self._cache[key] = cached
        if len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)
        self._stats.total_compile_time_ms += (time.perf_counter() - t0) * 1000
        self._stats.misses += 1

    def _execute_cached(
        self, key: _CacheKey, flat: list[Any], tensor_indices: list[int]
    ) -> T:
        """Execute a cached compiled model."""
        t0 = time.perf_counter()
        cached = self._cache[key]

        # Gather input buffers in order
        inputs = []
        for i in tensor_indices:
            tensor = flat[i]
            inputs.extend(tensor._impl._buffers)

        # Run the cached model
        _, *outputs = cached.model(*inputs)

        # Reconstruct output structure
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

                if n_shards == 1:
                    t = Tensor(buffers=shard_bufs[0])
                else:
                    t = Tensor._create_unsafe(bufferss=shard_bufs)
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
) -> Callable[..., T] | CompiledFunction[T]:
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
