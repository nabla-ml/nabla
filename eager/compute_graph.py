# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
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
from typing import TYPE_CHECKING, Any

from max import _core, driver, graph, mlir
from max._core.dialects import builtin, kgen, mo
from max.graph import Value, ops
from max.graph.graph import _location

if TYPE_CHECKING:
    from .tensor import Tensor

from .context import _session
from .tensor_impl import TensorImpl


def driver_tensor_type(t: driver.Tensor) -> graph.TensorType:
    """Converts a driver tensor to a TensorType."""
    return graph.TensorType(t.dtype, t.shape, graph.DeviceRef.from_device(t.device))


_SEED: ContextVar[Tensor] = ContextVar("_SEED")


def seed() -> Tensor:
    from .tensor import Tensor
    if (s := _SEED.get(None)) is None:
        s = driver.Tensor(ops.random.SeedType)
        s[0] = 0
        _SEED.set(Tensor(storage=s))
    return _SEED.get()


# =============================================================================
# Sharding Detection Helpers
# =============================================================================

def _collect_execution_graph(
    unrealized: list[Tensor],
) -> tuple[list[TensorImpl], list[TensorImpl]]:
    """Collect the execution subgraph (stop at realized tensors).
    
    Walks the parents graph from unrealized tensors, stopping at:
    - Realized tensors (have _storages) - these are inputs
    - Root tensors (no parents) - constants or initial inputs
    
    Args:
        unrealized: List of tensors pending evaluation
        
    Returns:
        (op_impls, leaf_impls) where:
        - op_impls: TensorImpls with operations, in topological order
        - leaf_impls: Input tensors (realized or no parents)
    """
    op_impls: list[TensorImpl] = []
    leaf_impls: list[TensorImpl] = []
    visited: set[int] = set()
    
    def dfs(impl: TensorImpl) -> None:
        impl_id = id(impl)
        if impl_id in visited:
            return
        visited.add(impl_id)
        
        # STOP at realized tensors - they're inputs to this execution
        if impl.is_realized:
            leaf_impls.append(impl)
            return
        
        # STOP at "root" tensors (no parents) - constants or untraced inputs
        if not impl.parents:
            leaf_impls.append(impl)
            return
        
        # Traverse parents first (for topological order)
        for parent in impl.parents:
            dfs(parent)
        
        # Add this operation after its dependencies
        op_impls.append(impl)
    
    for t in unrealized:
        dfs(t._impl)
    
    return op_impls, leaf_impls


def _needs_sharded_compilation(
    unrealized: list[Tensor],
) -> tuple[bool, list[TensorImpl], list[TensorImpl]]:
    """Check if these tensors need sharded compilation.
    
    Sharded compilation is needed when:
    1. ANY tensor in the execution graph has a sharding annotation
    2. ALL operation tensors are traced (so we can rebuild the graph)
    
    Args:
        unrealized: List of tensors pending evaluation
        
    Returns:
        (needs_sharding, op_impls, leaf_impls) where:
        - needs_sharding: True if sharded compilation should be used
        - op_impls: Operation TensorImpls in topological order
        - leaf_impls: Input TensorImpls (realized or roots)
        
    Raises:
        ValueError: If sharding is requested but tensors are untraced
    """
    op_impls, leaf_impls = _collect_execution_graph(unrealized)
    
    # If no operations to compile, no sharding needed
    if not op_impls:
        return False, [], []
    
    # Check for sharding annotations on ANY impl (operations or inputs)
    all_impls = op_impls + leaf_impls
    any_sharded = any(impl.sharding is not None for impl in all_impls)
    
    if not any_sharded:
        return False, [], []
    
    # Sharding requested - verify ALL operation impls are traced
    # (we need the parents graph to rebuild with sharding)
    untraced_ops = [impl for impl in op_impls if not impl.traced]
    if untraced_ops:
        raise ValueError(
            f"Sharded compilation requires traced tensors. "
            f"Found {len(untraced_ops)} untraced operation(s) in graph with sharding annotations. "
            f"Use traced=True when creating tensors or call tensor.trace()."
        )
    
    return True, op_impls, leaf_impls


class ComputeGraph:
    """Computation graph for managing tensor operations.

    This class manages the directed acyclic graph (DAG) of tensor operations
    for lazy evaluation and optimization. It tracks both realized tensors
    (with concrete data in memory) and unrealized tensors (pending computations).
    """

    graph: graph.Graph
    sources: dict[_core.Value[Any], Tensor]
    unrealized: weakref.WeakValueDictionary[int, Tensor]

    def __init__(self, context: mlir.Context | None = None, seed: int = 0):
        self.context = context or mlir.Context()
        self.sources = {}
        self.unrealized = weakref.WeakValueDictionary()
        self.graph = graph.Graph("main", input_types=[], context=self.context)

        with self.graph:
            ops.random.set_seed(seed)

    async def evaluate(self, tensor: Tensor) -> None:
        """Evaluates and realizes the specified tensor.
        
        Automatically detects if sharding annotations are present and uses
        the sharded compilation path when needed.
        """
        sys.last_value = None
        sys.last_traceback = None
        gc.collect()

        unrealized = list(self.unrealized.values())
        
        # Check if we need sharded compilation
        needs_sharding, op_impls, leaf_impls = _needs_sharded_compilation(unrealized)
        
        if needs_sharding:
            # Sharded compilation path
            await self._evaluate_sharded(unrealized, op_impls, leaf_impls)
        else:
            # Normal compilation path (existing behavior)
            await self._evaluate_normal(unrealized)
    
    async def _evaluate_normal(self, unrealized: list[Tensor]) -> None:
        """Normal (non-sharded) evaluation path."""
        with self.graph:
            self.graph.output(
                ops.random._peek_seed(), *map(graph.TensorValue, unrealized)
            )
        
        module: builtin.ModuleOp = _core.Operation._from_cmlir(
            self.graph._module.operation
        )
        _core.lower(module, [builtin.passes.RemoveDeadValues()])
        _remove_unused_arguments(self.graph)
        inputs = [
            self.sources[input._mlir_value] for input in self.graph.inputs
        ]

        try:
            model = _session().load(self.graph)
            seed_val, *results = model(*(input.driver_tensor for input in inputs))
            assert isinstance(seed_val, driver.Tensor)
        except BaseException as e:
            self.graph._erase_output_if_present()
            raise RuntimeError(
                "Failed to compile and execute graph!"
            ) from e

        for t, storage in zip(unrealized, results, strict=True):
            assert isinstance(storage, driver.Tensor)
            t.storage = storage
            t.real = True
            t._value = None

        for t in self.sources.values():
            t._value = None

        ComputeGraph.__init__(
            self, context=self.graph._context, seed=seed_val.item()
        )
    
    async def _evaluate_sharded(
        self,
        unrealized: list[Tensor],
        op_impls: list[TensorImpl],
        leaf_impls: list[TensorImpl],
    ) -> None:
        """Sharded evaluation path.
        
        Currently implements a DUMMY version to test multi-shard infrastructure:
        - Creates zeros for each shard based on ShardingSpec
        - Populates _storages with a list of driver.Tensors
        - Tests that multi-shard TensorImpls work with existing code
        
        TODO: Implement real sharded compilation:
        1. Run shardy propagation on op_impls + leaf_impls
        2. Build new MAX graph with sharded operations
        3. Compile and execute
        
        Args:
            unrealized: Tensors to evaluate
            op_impls: Operation TensorImpls in topological order
            leaf_impls: Input TensorImpls (realized or roots)
        """
        from .sharding import ShardingSpec, compute_local_shape, get_num_shards
        from max.dtype import DType
        import numpy as np
        
        # Erase any pending output on the graph (we won't use it)
        self.graph._erase_output_if_present()
        
        # For each unrealized tensor, create dummy shards
        for t in unrealized:
            impl = t._impl
            sharding = impl.sharding
            
            # Get global shape from cached metadata (or from _values if available)
            if impl.cached_shape is not None:
                # Convert Dim objects to int
                global_shape = tuple(int(d) for d in impl.cached_shape)
            elif impl._values and len(impl._values) > 0:
                global_shape = tuple(int(d) for d in impl._values[0].type.shape)
            else:
                raise RuntimeError(f"Cannot determine shape for sharded tensor")
            
            # Get dtype
            if impl.cached_dtype is not None:
                dtype = impl.cached_dtype
            elif impl._values and len(impl._values) > 0:
                dtype = impl._values[0].type.dtype
            else:
                dtype = DType.float32
            
            if sharding is not None and isinstance(sharding, ShardingSpec):
                # Create shards based on ShardingSpec
                num_devices = get_num_shards(sharding)
                
                # Create one driver.Tensor per device (shard)
                storages: list[driver.Tensor] = []
                for device_id in range(num_devices):
                    local_shape = compute_local_shape(global_shape, sharding, device_id)
                    # Create zeros with the local shape
                    np_zeros = np.zeros(local_shape, dtype=np.float32)
                    storage = driver.Tensor.from_numpy(np_zeros)
                    storages.append(storage)
                
                # Populate _storages as a list of shards
                impl._storages = storages
            else:
                # No sharding - create single shard with global shape
                np_zeros = np.zeros(global_shape, dtype=np.float32)
                impl._storages = [driver.Tensor.from_numpy(np_zeros)]
            
            # Mark as realized
            t.real = True
            impl._values = []  # Clear symbolic values
        
        # Clear source values
        for t in self.sources.values():
            t._value = None
        
        # Re-initialize the graph for next evaluation
        ComputeGraph.__init__(self, context=self.graph._context, seed=0)

    def add_source(self, tensor: Tensor) -> None:
        if tensor.storage is None:
            raise TypeError("Only realized tensors may be graph sources.")

        op = _core.Operation._from_cmlir(self.graph._mlir_op)
        assert isinstance(op, mo.GraphOp)
        block = op.regions[0].front
        with self.graph:
            type = driver_tensor_type(tensor.storage).as_buffer().to_mlir()
            inputs = op.function_type.inputs
            op.function_type = builtin.FunctionType([*inputs, type])
            tensor._value = graph.BufferValue.from_mlir(
                block.add_argument(type, _location())
            )
        self.sources[tensor._value._mlir_value] = tensor

    def add_unrealized(self, tensor: Tensor) -> None:
        self.unrealized[id(tensor)] = tensor


def _remove_unused_arguments(g: graph.Graph) -> None:
    """Remove unused arguments from the graph."""
    op = _core.Operation._from_cmlir(g._mlir_op)
    assert isinstance(op, mo.GraphOp)

    block = op.regions[0].front
    for i, input in reversed(list(enumerate(g.inputs))):
        if not input._mlir_value.num_uses:
            block.erase_argument(i)

    g.inputs = [Value.from_mlir(arg) for arg in block.arguments]

    with g:
        op.function_type = builtin.FunctionType(
            [input.type.to_mlir() for input in g.inputs],
            op.function_type.results,
        )
        op.signature = kgen.FuncTypeGeneratorType([], op.function_type)
        op.discardable_attributes["argument_names"] = builtin.ArrayAttr(
            [builtin.StringAttr(f"input{i}") for i in range(len(g.inputs))]
        )



# Global compute graph singleton
GRAPH = ComputeGraph()


def compile_with_sharding(
    outputs: list[Tensor], 
    mesh: "driver.DeviceMesh"  # Using forward reference or Any if needed, but lets try to be specific if possible or just Any. Actually shardoing types are in sharding.py.
):
    """Compile a sharded MAX graph by replaying ops with sharding awareness.
    
    This function walks the TensorImpl.parents graph, runs sharding propagation
    to fill in missing annotations, then rebuilds a fresh MAX graph with
    sharding-aware operations (including necessary collectives).
    
    Args:
        outputs: Output tensors of the computation (must be traced)
        mesh: Device mesh defining the logical device arrangement
        
    Returns:
        A compiled model ready for execution on the mesh
        
    Raises:
        ValueError: If any tensor is not traced or missing cached metadata
    """
    from .tensor_impl import get_topological_order
    from .sharding import DeviceMesh
    
    # Validate: all outputs must be traced with cached metadata
    for t in outputs:
        if not t.traced:
            raise ValueError(
                "Sharded compilation requires traced tensors. "
                "Use tensor.trace() or create with traced=True."
            )
        if t._impl.cached_shape is None:
            raise ValueError(
                "Missing cached metadata for sharding. "
                "This tensor may not have been created via an Operation."
            )
    
    # 1. Collect all impls in topological order
    all_impls = []
    seen = set()
    for t in outputs:
        for impl in get_topological_order(t._impl):
            if id(impl) not in seen:
                all_impls.append(impl)
                seen.add(id(impl))
    
    # 2. TODO: Run sharding propagation
    # from .sharding_propagation import propagate_sharding
    # propagate_sharding_over_graph(all_impls, mesh)
    
    # 3. TODO: Rebuild graph with sharding awareness
    # return build_sharded_graph(all_impls, mesh)
    
    raise NotImplementedError(
        "Sharding compiler not yet implemented. "
        f"Graph has {len(all_impls)} operations ready for sharded compilation."
    )
