# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import asyncio
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from max import graph
from max.graph import DeviceRef, TensorType
from max.graph.dim import AlgebraicDim, StaticDim, SymbolicDim

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

    tensor_sigs: tuple[tuple[tuple, Any], ...]
    static_vals: tuple[Any, ...]
    treedef: Any


@dataclass
class _CachedModel:
    """Cached compilation result."""

    model: Any
    input_order: list[int]
    output_treedef: Any
    output_tensor_mask: list[bool]
    output_static_graph_values: list[Any]


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

        flat, treedef = tree_flatten((args, kwargs))
        tensor_idx, tensor_sigs, static_vals = self._extract_inputs(flat)
        key = _CacheKey(tuple(tensor_sigs), tuple(static_vals), treedef)

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._execute_cached(key, flat, tensor_idx)

        return self._trace_and_compile(args, kwargs, flat, treedef, tensor_idx, key)

    def _extract_inputs(
        self, flat: list[Any]
    ) -> tuple[list[int], list[tuple[tuple, Any]], list[Any]]:
        """Extract tensor indices, signatures, and static values from flat args."""
        tensor_idx: list[int] = []
        tensor_sigs: list[tuple[tuple, Any]] = []
        static_vals: list[Any] = []

        arg_counter = 0
        for i, arg in enumerate(flat):
            if isinstance(arg, Tensor):
                arg.realize()
                tensor_idx.append(i)

                shape = self._build_signature_shape(arg, arg_counter)
                tensor_sigs.append((shape, arg.dtype))
                arg_counter += 1
            else:
                static_vals.append(arg)

        return tensor_idx, tensor_sigs, static_vals

    def _build_signature_shape(self, tensor: Tensor, arg_idx: int) -> tuple:
        """Build cache key shape signature, applying dynamic_dims if specified."""
        dims = list(tensor.buffers.shape)

        if arg_idx not in self.dynamic_dims:
            return tuple(dims)

        dim_map = self.dynamic_dims[arg_idx]
        shape_list = [
            SymbolicDim(dim_map[i]) if i in dim_map else StaticDim(dims[i])
            for i in range(len(dims))
        ]
        return _shape_signature(graph.Shape(shape_list))

    def _build_input_shape(self, tensor: Tensor, arg_idx: int) -> graph.Shape | tuple:
        """Build input type shape for Graph construction."""
        dims = list(tensor.buffers.shape)

        if arg_idx not in self.dynamic_dims:
            return tuple(dims)

        dim_map = self.dynamic_dims[arg_idx]
        return graph.Shape(
            [
                SymbolicDim(dim_map[i]) if i in dim_map else StaticDim(dims[i])
                for i in range(len(dims))
            ]
        )

    def _trace_and_compile(
        self,
        args: tuple,
        kwargs: dict,
        flat: list[Any],
        treedef: Any,
        tensor_idx: list[int],
        key: _CacheKey,
    ) -> T:
        """Trace function with proxies, compile graph, cache result."""
        t0 = time.perf_counter()

        proxy_flat = self._setup_graph_and_proxies(flat, tensor_idx)
        proxy_args, proxy_kwargs = tree_unflatten(treedef, proxy_flat)

        epoch_before = GRAPH.epoch
        result = self.__wrapped__(*proxy_args, **proxy_kwargs)

        if GRAPH.epoch != epoch_before:
            return self._handle_side_effects(args, kwargs)

        flat_out, out_treedef = tree_flatten(result)
        tensors_out, output_mask, static_out = self._extract_outputs(flat_out)

        if not tensors_out:
            return result

        try:
            model_result = asyncio.run(
                GRAPH.evaluate(tensors_out[0], *tensors_out[1:], return_model=True)
            )

            if model_result is None:
                self._stats.fallbacks += 1
                return self.__wrapped__(*args, **kwargs)

            model, _ = model_result
            cached = _CachedModel(
                model=model,
                input_order=list(range(len(tensor_idx))),
                output_treedef=out_treedef,
                output_tensor_mask=output_mask,
                output_static_graph_values=static_out,
            )

            self._add_to_cache(key, cached, t0)
            return self._run_cached(cached, flat, tensor_idx)

        except Exception as e:
            self._stats.fallbacks += 1
            warnings.warn(f"Compilation failed: {e}. Falling back.", stacklevel=2)
            return self.__wrapped__(*args, **kwargs)

    def _setup_graph_and_proxies(
        self, flat: list[Any], tensor_idx: list[int]
    ) -> list[Any]:
        """Create fresh Graph with symbolic inputs and return proxy-substituted flat list."""

        input_types = [
            TensorType(
                dtype=flat[i].dtype,
                shape=self._build_input_shape(flat[i], idx),
                device=DeviceRef.from_device(flat[i].device),
            )
            for idx, i in enumerate(tensor_idx)
        ]

        GRAPH._reset(GRAPH.context, 0)
        GRAPH.graph = graph.Graph(
            "main", input_types=input_types, context=GRAPH.context
        )

        proxy_flat = list(flat)
        with GRAPH.graph:
            from max.graph import ops

            ops.random.set_seed(0)

            for i, input_sym in zip(tensor_idx, GRAPH.graph.inputs, strict=False):
                proxy = Tensor(value=input_sym)
                proxy_flat[i] = proxy
                GRAPH.sources[input_sym._mlir_value] = flat[i]

        return proxy_flat

    def _extract_outputs(
        self, flat_out: list[Any]
    ) -> tuple[list[Tensor], list[bool], list[Any]]:
        """Extract tensors, mask, and static values from flattened outputs."""
        tensors: list[Tensor] = []
        mask: list[bool] = []
        statics: list[Any] = []

        for leaf in flat_out:
            if isinstance(leaf, Tensor):
                mask.append(True)
                tensors.append(leaf)
            else:
                mask.append(False)
                statics.append(leaf)

        return tensors, mask, statics

    def _handle_side_effects(self, args: tuple, kwargs: dict) -> T:
        """Handle side effects during tracing."""
        self._stats.fallbacks += 1
        if self.fullgraph:
            raise RuntimeError(
                f"'{self.__name__}' has side effects. Not allowed with fullgraph=True."
            )
        warnings.warn(f"{self.__name__} has side effects. Falling back.", stacklevel=2)
        return self.__wrapped__(*args, **kwargs)

    def _add_to_cache(self, key: _CacheKey, cached: _CachedModel, t0: float) -> None:
        """Add to cache with LRU eviction."""
        self._cache[key] = cached
        if len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)
        self._stats.total_compile_time_ms += (time.perf_counter() - t0) * 1000
        self._stats.misses += 1

    def _execute_cached(
        self, key: _CacheKey, flat: list[Any], tensor_idx: list[int]
    ) -> T:
        """Execute cached model and track stats."""
        t0 = time.perf_counter()
        result = self._run_cached(self._cache[key], flat, tensor_idx)
        self._stats.total_cached_exec_time_ms += (time.perf_counter() - t0) * 1000
        self._stats.hits += 1
        return result

    def _run_cached(
        self, cached: _CachedModel, flat: list[Any], tensor_idx: list[int]
    ) -> Any:
        """Execute cached model with realized tensors."""
        inputs = [flat[tensor_idx[i]].driver_tensor for i in cached.input_order]
        _, *outputs = cached.model(*inputs)

        all_leaves: list[Any] = []
        tensor_iter = iter(Tensor(buffers=o) for o in outputs)
        static_iter = iter(cached.output_static_graph_values)

        for is_tensor in cached.output_tensor_mask:
            all_leaves.append(next(tensor_iter) if is_tensor else next(static_iter))

        return tree_unflatten(cached.output_treedef, all_leaves)

    def __repr__(self) -> str:
        return f"<CompiledFunction {self.__name__} cache_size={len(self._cache)}>"


def _shape_signature(shape: graph.Shape) -> tuple:
    """Convert Shape to hashable tuple: StaticDim→int, SymbolicDim→'$name'."""
    result = []
    for d in shape:
        if isinstance(d, StaticDim):
            result.append(d.dim)
        elif isinstance(d, SymbolicDim):
            result.append(f"${d.name}")
        elif isinstance(d, AlgebraicDim):
            result.append(f"${d}")
        else:
            try:
                result.append(int(d))
            except (TypeError, ValueError):
                result.append(str(d))
    return tuple(result)


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
