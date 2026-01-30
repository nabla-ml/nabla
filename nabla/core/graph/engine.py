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

import os

DEBUG_LAZY_EVAL: bool = os.getenv("NABLA_DEBUG", "0") == "1"


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

    def __init__(self, context: mlir.Context | None = None, seed: int = 0):
        global _GRAPH_EPOCH
        _GRAPH_EPOCH += 1
        self.epoch = _GRAPH_EPOCH
        self._reset(context, seed)

    def _reset(self, context: mlir.Context | None, seed: int) -> None:
        """Resets the internal graph state."""
        self.context = context or mlir.Context()
        self.sources = {}
        self.unrealized = weakref.WeakValueDictionary()
        self.graph = graph.Graph("main", input_types=[], context=self.context)
        with self.graph:
            ops.random.set_seed(seed)

    def add_input(self, tensor: Tensor) -> None:
        """Registers a realized tensor's storages as graph inputs."""
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
                tensor_values.append(buffer_val)

            self.sources[buffer_val._mlir_value] = storage

        if len(tensor_values) == 1:
            tensor._value = tensor_values[0]
        else:

            with self.graph:
                tensor._impl._values = [bv[...] for bv in tensor_values]
                tensor._impl.values_epoch = self.epoch

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
        
        import sys as _sys
        _sys.stderr.write(f"\n[ENGINE.evaluate] Starting, tensor shape={tensor.shape}\n")
        _sys.stderr.flush()

        sys.last_value = None
        sys.last_traceback = None
        gc.collect()
        
        _sys.stderr.write(f"[ENGINE.evaluate] After gc\n")
        _sys.stderr.flush()

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

        for t in self.unrealized.values():
            add_target(t)
            
        _sys.stderr.write(f"[ENGINE.evaluate] targets={len(targets)}, calling _evaluate_normal\n")
        _sys.stderr.flush()

        return self._evaluate_normal(targets, return_model=return_model)

    def _evaluate_normal(self, unrealized: list[Tensor], return_model: bool) -> Any:
        """Standard compilation path handling both single-device and eager-sharded execution."""
        import sys as _sys
        _sys.stderr.write(f"[_evaluate_normal] Starting with {len(unrealized)} tensors\n")
        _sys.stderr.flush()

        if DEBUG_LAZY_EVAL:
            print("=" * 70)
            print(
                f"[LAZY EVAL] Epoch {self.epoch} - Setting {len(unrealized)} output(s):"
            )
            for i, t in enumerate(unrealized):
                op_name = t._impl.op_name if t._impl.output_refs else "<leaf>"
                num_shards = len(t._impl._values) if t._impl._values else 1
                print(f"  [{i}] id={id(t)} op={op_name} shards={num_shards}")
            print("-" * 70)

        all_values = []
        value_map = []

        _sys.stderr.write(f"[_evaluate_normal] About to enter graph context\n")
        _sys.stderr.flush()
        
        with self.graph:
            _sys.stderr.write(f"[_evaluate_normal] Inside graph context\n")
            _sys.stderr.flush()
            
            for i, t in enumerate(unrealized):
                _sys.stderr.write(f"[_evaluate_normal] Processing tensor {i}, epoch check: {t._impl.values_epoch} vs {self.epoch}\n")
                _sys.stderr.flush()

                if t._impl.values_epoch != self.epoch:
                    _sys.stderr.write(f"[_evaluate_normal] Clearing stale values\n")
                    _sys.stderr.flush()
                    if DEBUG_LAZY_EVAL:
                        print(
                            f"[LAZY DEBUG] Clearing stale values for target tensor {id(t)} "
                            f"(epoch: {t._impl.values_epoch} != {self.epoch})"
                        )
                    t._impl._values = []

                _sys.stderr.write(f"[_evaluate_normal] Checking if need to add_input\n")
                _sys.stderr.flush()
                
                if not t._impl._values and t._impl.is_realized:
                    _sys.stderr.write(f"[_evaluate_normal] Adding as input\n")
                    _sys.stderr.flush()
                    self.add_input(t)

                _sys.stderr.write(f"[_evaluate_normal] Getting values\n")
                _sys.stderr.flush()
                
                values = t._impl._values
                _sys.stderr.write(f"[_evaluate_normal] Got {len(values) if values else 0} values\n")
                _sys.stderr.flush()
                if not values:
                    print(f"FAILED TENSOR {id(t)} info:")
                    print(f"  sharding: {t.sharding}")
                    print(f"  realized: {t.is_realized}")
                    print(f"  epoch: {t._impl.values_epoch}")
                    print(f"  graph epoch: {self.epoch}")
                    if t._impl.output_refs:
                        print(f"  op: {t._impl.op_name}")
                    raise RuntimeError(
                        f"Attempting to evaluate tensor {id(t)} with no values/storage"
                    )

                _sys.stderr.write(f"[_evaluate_normal] Checking multi-shard\n")
                _sys.stderr.flush()
                
                if values and len(values) > 1:
                    _sys.stderr.write(f"[_evaluate_normal] Multi-shard path\n")
                    _sys.stderr.flush()
                    for shard_idx, val in enumerate(values):
                        all_values.append(val)
                        value_map.append((t, shard_idx))
                else:
                    _sys.stderr.write(f"[_evaluate_normal] Single value path, appending values[0]\n")
                    _sys.stderr.flush()
                    _sys.stderr.write(f"[_evaluate_normal] values[0] type: {type(values[0])}\n")
                    _sys.stderr.flush()
                    all_values.append(values[0])
                    _sys.stderr.write(f"[_evaluate_normal] Appended to all_values\n")
                    _sys.stderr.flush()
                    value_map.append((t, None))
                    _sys.stderr.write(f"[_evaluate_normal] Appended to value_map\n")
                    _sys.stderr.flush()

            _sys.stderr.write(f"[_evaluate_normal] Loop done, calling graph.output with {len(all_values)} values\n")
            _sys.stderr.flush()
            _sys.stderr.write(f"[_evaluate_normal] all_values[0]: {all_values[0] if all_values else 'empty'}\n")
            _sys.stderr.flush()
            seed = ops.random._peek_seed()
            _sys.stderr.write(f"[_evaluate_normal] Got seed: {seed}\n")
            _sys.stderr.flush()
            self.graph.output(seed, *all_values)
            _sys.stderr.write(f"[_evaluate_normal] graph.output done\n")
            _sys.stderr.write(f"[_evaluate_normal] Graph after output: {self.graph}\n")
            _sys.stderr.flush()
        
        _sys.stderr.write(f"[_evaluate_normal] Exited graph context\n")
        _sys.stderr.flush()

        if DEBUG_LAZY_EVAL:
            print("[LAZY EVAL] MAX Graph:")
            print(self.graph)
            print("=" * 70)

        _sys.stderr.write(f"[_evaluate_normal] Calling _compile_and_execute_with_map\n")
        _sys.stderr.flush()
        
        return self._compile_and_execute_with_map(unrealized, value_map, return_model)

    def _compile_and_execute_with_map(
        self,
        unrealized: list[Tensor],
        value_map: list[tuple[Tensor, int | None]],
        return_model: bool,
    ) -> Any:
        """Compiles and executes with sharded tensor support."""
        import sys as _sys
        _sys.stderr.write(f"[_compile_and_execute_with_map] Starting\n")
        _sys.stderr.flush()

        try:
            _sys.stderr.write(f"[_compile_and_execute_with_map] About to lower\n")
            _sys.stderr.flush()
            module = _core.Operation._from_cmlir(self.graph._module.operation)
            _sys.stderr.write(f"[_compile_and_execute_with_map] Got module\n")
            _sys.stderr.flush()
            _core.lower(module, [builtin.passes.RemoveDeadValues()])
            _sys.stderr.write(f"[_compile_and_execute_with_map] Lowered\n")
            _sys.stderr.flush()
            _remove_unused_arguments(self.graph)
            _sys.stderr.write(f"[_compile_and_execute_with_map] Removed unused args\n")
            _sys.stderr.flush()
        except Exception as e:
            if DEBUG_LAZY_EVAL:
                print(f"[LAZY EVAL ERROR] Optimization failed: {e}")
            raise

        try:
            _sys.stderr.write(f"[_compile_and_execute_with_map] Building inputs\n")
            _sys.stderr.flush()
            inputs = [self.sources[inp._mlir_value] for inp in self.graph.inputs]
            _sys.stderr.write(f"[_compile_and_execute_with_map] Got {len(inputs)} inputs, {len(self.graph.inputs)} graph inputs\n")
            _sys.stderr.flush()
            _sys.stderr.write(f"[_compile_and_execute_with_map] Graph before load:\n{self.graph}\n")
            _sys.stderr.flush()
            _sys.stderr.write(f"[_compile_and_execute_with_map] Loading model...\n")
            _sys.stderr.flush()
            model = _session().load(self.graph)
            _sys.stderr.write(f"[_compile_and_execute_with_map] Model loaded, executing\n")
            _sys.stderr.flush()
            seed_val, *results = model(*inputs)
            _sys.stderr.write(f"[_compile_and_execute_with_map] Model executed\n")
            _sys.stderr.flush()
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
                    t._impl._storages = [r for _, r in shard_results]
                    t._impl._values = []
                else:

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
        global _GRAPH_EPOCH
        _GRAPH_EPOCH += 1
        self.epoch = _GRAPH_EPOCH

        self._reset(None, seed_value)


GRAPH = ComputeGraph()
