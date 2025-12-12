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
        """Evaluates and realizes the specified tensor."""
        sys.last_value = None
        sys.last_traceback = None
        gc.collect()

        unrealized = list(self.unrealized.values())
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
