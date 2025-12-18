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

"""Compile transform for the eager module (PyTorch-style torch.compile).

Caches compiled MAX models keyed by input signatures (shapes, dtypes)
and static argument values. Avoids Python re-tracing on cache hits.

Usage:
    @compile
    def mlp(x, w1, w2):
        return (x @ w1).relu() @ w2
    
    # First call: traces + compiles
    mlp(x, w1, w2)
    
    # Subsequent calls: cache hit if shapes/dtypes match
    mlp(x2, w1, w2)  # Reuses compiled model
    
    # Check compilation stats
    print(mlp.stats)
"""

from __future__ import annotations

import asyncio
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Generic, TypeVar

from .compute_graph import GRAPH
from .pytree import tree_flatten, tree_unflatten, tree_leaves
from .tensor import Tensor

T = TypeVar("T")


# =============================================================================
# Compilation Statistics
# =============================================================================

@dataclass
class CompilationStats:
    """Statistics for a compiled function.
    
    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses (compilations).
        fallbacks: Number of fallbacks due to side effects.
        total_compile_time_ms: Total time spent compiling (ms).
        total_cached_exec_time_ms: Total time in cached execution (ms).
        cache_size: Current number of cached compilations.
    """
    hits: int = 0
    misses: int = 0
    fallbacks: int = 0
    total_compile_time_ms: float = 0.0
    total_cached_exec_time_ms: float = 0.0
    cache_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage."""
        total = self.hits + self.misses + self.fallbacks
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def avg_compile_time_ms(self) -> float:
        """Average compilation time in milliseconds."""
        return self.total_compile_time_ms / self.misses if self.misses > 0 else 0.0
    
    @property
    def avg_cached_exec_time_ms(self) -> float:
        """Average cached execution time in milliseconds."""
        return self.total_cached_exec_time_ms / self.hits if self.hits > 0 else 0.0
    
    def __repr__(self) -> str:
        return (
            f"CompilationStats(\n"
            f"  hits={self.hits}, misses={self.misses}, fallbacks={self.fallbacks},\n"
            f"  hit_rate={self.hit_rate:.1f}%,\n"
            f"  avg_compile_time={self.avg_compile_time_ms:.2f}ms,\n"
            f"  avg_cached_exec={self.avg_cached_exec_time_ms:.2f}ms,\n"
            f"  cache_size={self.cache_size}\n"
            f")"
        )


# =============================================================================
# Cache Key and Cached Model
# =============================================================================

@dataclass(frozen=True)
class _CacheKey:
    """Cache key for compiled function (shapes/dtypes + static args + treedef)."""
    tensor_sigs: tuple[tuple[tuple, Any], ...]  # (shape_sig, dtype) for each Tensor
    static_vals: tuple[Any, ...]                # Non-Tensor values
    treedef: Any                                 # Pytree structure


def _shape_signature(shape: graph.Shape) -> tuple:
    """Create hashable signature from Shape, handling symbolic dimensions.
    
    Returns tuple where each element is:
    - StaticDim → int value
    - SymbolicDim → "$name" string
    - AlgebraicDim → "$expression" string
    
    This enables cache sharing for functions with same symbolic dimensions.
    
    Args:
        shape: MAX Shape object
        
    Returns:
        Tuple suitable for use in cache keys
        
    Examples:
        Shape([StaticDim(10), StaticDim(20)]) → (10, 20)
        Shape([SymbolicDim("batch"), StaticDim(128)]) → ("$batch", 128)
    """
    from max.graph.dim import StaticDim, SymbolicDim, AlgebraicDim
    
    sig = []
    for d in shape:
        if isinstance(d, StaticDim):
            sig.append(d.dim)
        elif isinstance(d, SymbolicDim):
            sig.append(f"${d.name}")
        elif isinstance(d, AlgebraicDim):
            sig.append(f"${str(d)}")
        else:
            # Fallback for other Dim types
            try:
                sig.append(int(d))
            except (TypeError, ValueError):
                sig.append(str(d))
    return tuple(sig)


@dataclass
class _CachedModel:
    """Cached compilation result.
    
    Attributes:
        model: Compiled MAX model.
        input_order: Indices mapping flat args to model inputs.
        output_treedef: Pytree structure of the outputs.
        output_tensor_mask: Boolean mask - True for Tensor leaves, False for static.
        output_static_values: Cached static output values (non-Tensors).
    """
    model: Any
    input_order: list[int]
    output_treedef: Any
    output_tensor_mask: list[bool]  # Which output leaves are Tensors
    output_static_values: list[Any]  # Cached non-Tensor output values


# =============================================================================
# Compiled Function Class
# =============================================================================

class CompiledFunction(Generic[T]):
    """A compiled function with caching and statistics.
    
    Wraps the original function with JIT compilation logic. Provides
    access to compilation statistics and cache management.
    
    Attributes:
        __wrapped__: The original unwrapped function.
        stats: Compilation statistics.
        fullgraph: If True, errors on side effects instead of fallback.
    """
    
    def __init__(
        self, 
        fn: Callable[..., T],
        *,
        fullgraph: bool = False,
        max_cache_size: int = 64,
    ):
        """Initialize compiled function.
        
        Args:
            fn: Function to compile.
            fullgraph: If True, raise error on side effects instead of fallback.
            max_cache_size: Maximum number of cached compilations (LRU eviction).
        """
        self.__wrapped__ = fn
        self.__name__ = getattr(fn, '__name__', repr(fn))
        self.__doc__ = fn.__doc__
        
        self.fullgraph = fullgraph
        self.max_cache_size = max_cache_size
        self._cache: OrderedDict[_CacheKey, _CachedModel] = OrderedDict()
        self._stats = CompilationStats()
    
    @property
    def stats(self) -> CompilationStats:
        """Get compilation statistics."""
        self._stats.cache_size = len(self._cache)
        return self._stats
    
    def clear_cache(self) -> None:
        """Clear the compilation cache."""
        self._cache.clear()
        self._stats.cache_size = 0
    
    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Execute the compiled function."""
        # 1. Flatten inputs to extract Tensors and static values
        flat, treedef = tree_flatten((args, kwargs))
        
        # 2. Separate Tensors (dynamic) from static values
        tensor_idx: list[int] = []
        tensor_sigs: list[tuple[tuple[int, ...], Any]] = []
        static_vals: list[Any] = []
        
        for i, arg in enumerate(flat):
            if isinstance(arg, Tensor):
                arg._sync_realize()  # Ensure realized for shape/dtype
                tensor_idx.append(i)
                # Use shape signature to handle symbolic dims
                shape = arg.shape  # graph.Shape
                sig = _shape_signature(shape)
                tensor_sigs.append((sig, arg.dtype))
            else:
                static_vals.append(arg)
        
        key = _CacheKey(tuple(tensor_sigs), tuple(static_vals), treedef)
        
        # 3. Cache hit → execute cached model directly
        if key in self._cache:
            # Move to end for LRU
            self._cache.move_to_end(key)
            
            t0 = time.perf_counter()
            result = self._run_cached(self._cache[key], flat)
            self._stats.total_cached_exec_time_ms += (time.perf_counter() - t0) * 1000
            self._stats.hits += 1
            return result
        
        # 4. Cache miss → trace, compile, cache
        t0 = time.perf_counter()
        epoch_before = GRAPH.epoch
        result = self.__wrapped__(*args, **kwargs)
        
        # Check for side effects (intermediate evaluations)
        if GRAPH.epoch != epoch_before:
            self._stats.fallbacks += 1
            
            if self.fullgraph:
                raise RuntimeError(
                    f"Compiled function '{self.__name__}' has side effects that "
                    "trigger graph evaluation (e.g., .item() call). "
                    "This is not allowed with fullgraph=True. "
                    "Either remove the side effect or set fullgraph=False."
                )
            else:
                warnings.warn(
                    f"{self.__name__} has side effects that trigger evaluation. "
                    "Cannot cache compiled graph. Falling back to dynamic execution.",
                    stacklevel=2
                )
                return result
        
        # Extract output structure
        flat_out, out_treedef = tree_flatten(result)
        
        # Identify which outputs are Tensors vs static
        output_tensor_mask: list[bool] = []
        output_static_values: list[Any] = []
        tensors_out: list[Tensor] = []
        
        for leaf in flat_out:
            if isinstance(leaf, Tensor):
                output_tensor_mask.append(True)
                tensors_out.append(leaf)
            else:
                output_tensor_mask.append(False)
                output_static_values.append(leaf)
        
        if not tensors_out:
            # No tensors to compile, just return result
            return result
        
        # Compile and cache
        try:
            # Evaluate (triggers compilation) and capture model
            model_result = asyncio.run(
                GRAPH.evaluate(tensors_out[0], *tensors_out[1:], return_model=True)
            )
            
            if model_result is None:
                # Fallback (e.g., sharded compilation)
                self._stats.fallbacks += 1
                return result
            
            model, ordered_inputs = model_result
            
            # Map ordered inputs back to flat arg indices
            input_order = _compute_input_order(flat, tensor_idx, ordered_inputs)
            
            cached = _CachedModel(
                model=model,
                input_order=input_order,
                output_treedef=out_treedef,
                output_tensor_mask=output_tensor_mask,
                output_static_values=output_static_values,
            )
            
            # Add to cache with LRU eviction
            self._cache[key] = cached
            if len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)  # Remove oldest
            
            compile_time = (time.perf_counter() - t0) * 1000
            self._stats.total_compile_time_ms += compile_time
            self._stats.misses += 1
            
        except Exception as e:
            self._stats.fallbacks += 1
            warnings.warn(
                f"Compilation failed: {e}. Falling back to eager execution.",
                stacklevel=2
            )
        
        # First call: outputs are already realized from evaluate()
        return result
    
    def _run_cached(self, cached: _CachedModel, flat: list[Any]) -> Any:
        """Execute cached model and reconstruct output pytree."""
        # Get driver tensors in the order expected by the model
        inputs = [flat[i].driver_tensor for i in cached.input_order]
        
        # Execute model (returns seed + outputs)
        _, *outputs = cached.model(*inputs)
        
        # Reconstruct full output with tensors and static values
        all_leaves: list[Any] = []
        tensor_iter = iter(Tensor(storage=o) for o in outputs)
        static_iter = iter(cached.output_static_values)
        
        for is_tensor in cached.output_tensor_mask:
            if is_tensor:
                all_leaves.append(next(tensor_iter))
            else:
                all_leaves.append(next(static_iter))
        
        return tree_unflatten(cached.output_treedef, all_leaves)
    
    def __repr__(self) -> str:
        return f"<CompiledFunction {self.__name__} cache_size={len(self._cache)}>"


# =============================================================================
# Main compile decorator
# =============================================================================

def compile(
    fn: Callable[..., T] | None = None,
    *,
    fullgraph: bool = False,
    max_cache_size: int = 64,
) -> Callable[..., T] | CompiledFunction[T]:
    """Compile a function for cached graph execution.
    
    Similar to torch.compile: on first call with a given input signature,
    traces through the function to build a MAX graph, compiles it, and
    caches the result. Subsequent calls with matching signatures reuse
    the cached model without re-tracing Python code.
    
    Args:
        fn: Function to compile. If None, returns a decorator.
        fullgraph: If True, raise error on side effects (like PyTorch's
            fullgraph=True). If False (default), fall back to eager execution.
        max_cache_size: Maximum cached compilations before LRU eviction.
    
    Returns:
        CompiledFunction wrapping the original function.
    
    Notes:
        - Tensor args: Cached by (shape, dtype). Values can vary.
        - Non-Tensor args: Cached by value. Different values = recompile.
        - Side effects (e.g., `if x.item() > 0`) prevent caching.
        - Access stats via: `compiled_fn.stats`
        - Clear cache via: `compiled_fn.clear_cache()`
    
    Examples:
        >>> @compile
        ... def square(x):
        ...     return x * x
        >>> square(Tensor.ones((3, 3)))  # Traces + compiles
        >>> square(Tensor.zeros((3, 3)))  # Cache hit! Same shape/dtype
        >>> print(square.stats)
        
        >>> @compile(fullgraph=True)
        ... def strict_fn(x):
        ...     return x * 2
        >>> # Will raise error if any side effects occur
    """
    if fn is None:
        return lambda f: compile(f, fullgraph=fullgraph, max_cache_size=max_cache_size)
    
    return CompiledFunction(fn, fullgraph=fullgraph, max_cache_size=max_cache_size)


# =============================================================================
# Helper functions
# =============================================================================

def _compute_input_order(
    flat: list[Any],
    tensor_idx: list[int],
    ordered_inputs: list[Tensor],
) -> list[int]:
    """Map graph inputs to flat arg indices.
    
    The ordered_inputs are the Tensors that became graph sources,
    in the order they were added to the graph. We need to match
    each back to its position in the flattened arguments.
    """
    input_order: list[int] = []
    for inp_tensor in ordered_inputs:
        # Find this tensor in flat args
        for i in tensor_idx:
            if flat[i] is inp_tensor:
                input_order.append(i)
                break
        else:
            # Tensor not found in inputs - shouldn't happen
            # Fall back to assuming order matches tensor_idx
            pass
    
    # If we couldn't match all, fall back to tensor_idx order
    if len(input_order) != len(ordered_inputs):
        return tensor_idx[:len(ordered_inputs)]
    
    return input_order


__all__ = ["compile", "CompiledFunction", "CompilationStats"]