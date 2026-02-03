# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Execution utilities for operations.

This module contains purely functional helpers used during operation execution,
separated from the core Operation base class to keep base.py clean and focused
on the abstract interface.
"""

from __future__ import annotations
from typing import Any, Union
from pathlib import Path


def ensure_tensor(x: Any) -> Any:
    """Convert scalar or array-like to Tensor."""
    from ..core import Tensor

    if isinstance(x, Tensor):
        return x

    return Tensor.constant(x)


# Cache for _make_hashable to avoid recomputing for same objects
_HASHABLE_CACHE: dict[int, Any] = {}
_HASHABLE_CACHE_SIZE = 1024  # Limit cache size to avoid memory bloat


def _clear_hashable_cache() -> None:
    """Clear the hashable cache. Call periodically to avoid memory bloat."""
    _HASHABLE_CACHE.clear()


def _make_hashable(obj: Any) -> Any:
    """Convert objects to stable, hashable keys for graph caching.

    Optimized with caching for Tensor objects since they're frequently hashed.
    """
    from ..core import Tensor

    if isinstance(obj, Tensor):
        # Check cache first - use id() for Tensor objects
        obj_id = id(obj)
        cached = _HASHABLE_CACHE.get(obj_id)
        if cached is not None:
            return cached

        sharding = obj.sharding
        sharding_key = None
        if sharding is not None:
            mesh = sharding.mesh  # Direct attribute access, avoid getattr
            mesh_key = None
            if mesh is not None:
                mesh_key = (
                    mesh.name if hasattr(mesh, "name") else None,
                    tuple(mesh.shape) if hasattr(mesh, "shape") and mesh.shape else (),
                    (
                        tuple(mesh.axis_names)
                        if hasattr(mesh, "axis_names") and mesh.axis_names
                        else ()
                    ),
                )

            dim_specs_list = sharding.dim_specs
            if dim_specs_list:
                dim_specs = tuple(
                    (tuple(ds.axes) if ds.axes else (), bool(ds.partial))
                    for ds in dim_specs_list
                )
            else:
                dim_specs = ()

            replicated = sharding.replicated_axes
            partial_sum = sharding.partial_sum_axes
            sharding_key = (
                mesh_key,
                dim_specs,
                tuple(sorted(replicated)) if replicated else (),
                tuple(sorted(partial_sum)) if partial_sum else (),
            )

        impl = obj._impl
        physical_shapes = impl._physical_shapes
        shard_dtypes = impl._shard_dtypes
        shard_devices = impl._shard_devices

        result = (
            "tensor",
            str(obj.dtype),
            tuple(int(d) for d in obj.shape),
            sharding_key,
            tuple(physical_shapes) if physical_shapes else None,
            tuple(shard_dtypes) if shard_dtypes else None,
            tuple(str(d) for d in shard_devices) if shard_devices else None,
        )

        # Cache result (with size limit)
        if len(_HASHABLE_CACHE) < _HASHABLE_CACHE_SIZE:
            _HASHABLE_CACHE[obj_id] = result

        return result

    # Fast path for primitives (most common case for kwargs)
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(x) for x in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    return str(obj)


def _get_tensor_hash(x: Any) -> Any:
    """Get hash from tensor - shape/dtype/sharding for realized, OpNode hash+index for unrealized.

    Optimized module-level function to avoid closure overhead in hot path.
    """
    from ..core import Tensor

    if isinstance(x, Tensor):
        impl = x._impl
        buffers = impl._buffers
        output_refs = impl.output_refs

        # Fast path: get sharding key only if sharding exists
        sharding = x.sharding
        sharding_key = _make_hashable(sharding) if sharding else None

        if buffers:
            # Realized tensor - hash based on shape/dtype/sharding (data-independent)
            shape_tuple = tuple(int(d) for d in x.shape)
            return ("realized", str(x.dtype), shape_tuple, sharding_key)
        elif output_refs is not None and output_refs._op_hash is not None:
            # Unrealized tensor - use its OpNode's hash AND its output index
            return (output_refs._op_hash, impl.output_index, sharding_key)
        else:
            # Leaf tensor without buffers (shouldn't happen in normal flow)
            shape_tuple = tuple(int(d) for d in x.shape)
            return ("leaf", str(x.dtype), shape_tuple, sharding_key)
    return _make_hashable(x)


def validate_physical_metadata(
    op: Any,
    shard_graph_values: list[Any],
    output_physical_shapes: list[Any] | None,
    output_shard_dtypes: list[Any] | None,
    output_shard_devices: list[Any] | None,
) -> None:
    """Check if inferred physical metadata matches actual values from MAX backend."""
    if output_physical_shapes is None:
        return

    from max import graph as g

    def to_dev_ref(d):
        from max.graph import DeviceRef
        from max.driver import Device as DriverDevice

        if isinstance(d, DeviceRef):
            return d
        if isinstance(d, DriverDevice):
            return DeviceRef.from_device(d)
        if isinstance(d, int):
            # Assume CPU for simulation if int provided
            return DeviceRef.CPU()
        return d

    first_shard = shard_graph_values[0] if shard_graph_values else None
    if isinstance(first_shard, (list, tuple)):
        # Multi-output: metadata should be list of lists
        # e.g., output_physical_shapes: [[(shape0_s0), ...], [(shape1_s0), ...]]
        unzipped = list(zip(*shard_graph_values))
        for i, (
            out_inferred_shapes,
            out_inferred_dtypes,
            out_inferred_devices,
            out_shards,
        ) in enumerate(
            zip(
                output_physical_shapes,
                output_shard_dtypes,
                output_shard_devices,
                unzipped,
            )
        ):
            for j, (
                inferred_shape,
                inferred_dtype,
                inferred_device,
                actual_value,
            ) in enumerate(
                zip(
                    out_inferred_shapes,
                    out_inferred_dtypes,
                    out_inferred_devices,
                    out_shards,
                )
            ):
                if isinstance(actual_value, (g.TensorValue, g.BufferValue)):
                    actual_shape = tuple(int(d) for d in actual_value.type.shape)
                    actual_dtype = actual_value.type.dtype
                    actual_device = actual_value.device
                    if inferred_shape != actual_shape:
                        raise RuntimeError(
                            f"Shape Mismatch in {op.name} (Multi-output) Phase 4:\n"
                            f"  Output Index: {i}, Shard Index: {j}\n"
                            f"  Inferred: {inferred_shape}, Actual: {actual_shape}"
                        )
                    if inferred_dtype != actual_dtype:
                        raise RuntimeError(
                            f"DType Mismatch in {op.name} (Multi-output) Phase 4:\n"
                            f"  Output Index: {i}, Shard Index: {j}\n"
                            f"  Inferred: {inferred_dtype}, Actual: {actual_dtype}"
                        )
                    if to_dev_ref(inferred_device) != to_dev_ref(actual_device):
                        raise RuntimeError(
                            f"Device Mismatch in {op.name} (Multi-output) Phase 4:\n"
                            f"  Output Index: {i}, Shard Index: {j}\n"
                            f"  Inferred: {inferred_device}, Actual: {actual_device}"
                        )
    else:
        # Single output
        for i, (
            inferred_shape,
            inferred_dtype,
            inferred_device,
            actual_value,
        ) in enumerate(
            zip(
                output_physical_shapes,
                output_shard_dtypes,
                output_shard_devices,
                shard_graph_values,
            )
        ):
            if isinstance(actual_value, (g.TensorValue, g.BufferValue)):
                actual_shape = tuple(int(d) for d in actual_value.type.shape)
                actual_dtype = actual_value.type.dtype
                actual_device = actual_value.device
                if inferred_shape != actual_shape:
                    raise RuntimeError(
                        f"Shape Mismatch in {op.name} Phase 4:\n"
                        f"  Shard Index: {i}\n"
                        f"  Inferred: {inferred_shape}, Actual: {actual_shape}"
                    )
                if inferred_dtype != actual_dtype:
                    raise RuntimeError(
                        f"DType Mismatch in {op.name} Phase 4:\n"
                        f"  Shard Index: {i}\n"
                        f"  Inferred: {inferred_dtype}, Actual: {actual_dtype}"
                    )
                if to_dev_ref(inferred_device) != to_dev_ref(actual_device):
                    raise RuntimeError(
                        f"Device Mismatch in {op.name} Phase 4:\n"
                        f"  Shard Index: {i}\n"
                        f"  Inferred: {inferred_device}, Actual: {actual_device}"
                    )


def apply_jvp(op: Any, args: tuple, output: Any) -> None:
    """Apply JVP (forward-mode autodiff) by propagating tangents through the operation.

    This is called during operation execution when any input has tangents attached.
    It computes the JVP using the operation's jvp_rule and attaches the result
    tangent to the output.

    Args:
        op: The operation instance (needed for jvp_rule)
        args: Input arguments (may have tangents attached)
        output: Output tensor(s) to attach tangents to
    """
    from ..core import Tensor, pytree

    tangents = pytree.tree_map(
        lambda x: (
            Tensor(impl=x.tangent) if isinstance(x, Tensor) and x.tangent else None
        ),
        args,
    )
    output_tangent = op.jvp_rule(args, tangents, output)
    if output_tangent is not None:
        pytree.tree_map(
            lambda o, t: (
                setattr(o._impl, "tangent", t._impl)
                if isinstance(o, Tensor) and isinstance(t, Tensor)
                else None
            ),
            output,
            output_tangent,
        )


def collect_metadata(args: tuple) -> tuple[int, bool, bool, bool]:
    """Analyze arguments to collect metadata needed for execution adaptation."""
    from ..core import Tensor

    max_batch_dims = 0
    any_traced = False
    any_sharded = False
    any_has_tangent = False

    # Fast path: most ops have simple tensor args (not nested)
    # Use stack-based iteration to avoid tree_map overhead
    stack = list(args)
    while stack:
        x = stack.pop()
        if isinstance(x, Tensor):
            if x.batch_dims > max_batch_dims:
                max_batch_dims = x.batch_dims
            if x.is_traced:
                any_traced = True
            if x.is_sharded:
                any_sharded = True
            if x.tangent is not None:
                any_has_tangent = True
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
        elif isinstance(x, dict):
            stack.extend(x.values())
    return max_batch_dims, any_traced, any_sharded, any_has_tangent


def adapt_and_reshard(
    op: Any, args: tuple, kwargs: dict, any_sharded: bool, max_batch_dims: int
) -> tuple[tuple, dict, Any, Any, Any]:
    """Perform logical adaptation and input resharding."""
    from ..core.sharding import spmd

    mesh = spmd.get_mesh_from_args(args) if any_sharded else None
    # Also check kwargs for mesh (e.g., shard/reshard ops pass mesh as kwarg)
    if mesh is None and kwargs.get("mesh") is not None:
        mesh = kwargs["mesh"]

    adapted_kwargs = op.adapt_kwargs(args, kwargs, max_batch_dims)
    args_with_specs = spmd.ensure_specs(args, mesh)
    predicted_output_spec, input_shardings, reduce_axes = spmd.infer_output_sharding(
        op, args_with_specs, mesh, adapted_kwargs or {}
    )

    # Perform the data movement (Logical Adaptation)
    resharded_args = spmd.reshard_inputs(args_with_specs, input_shardings, mesh)

    return resharded_args, adapted_kwargs, predicted_output_spec, mesh, reduce_axes


def compute_structural_hash(
    op_name: str, resharded_args: tuple, adapted_kwargs: dict
) -> tuple:
    """Compute structural hash for operation caching."""
    # Compute hash AFTER resharding - use resharded_args for cache key
    arg_hashes = tuple(_get_tensor_hash(x) for x in resharded_args)
    kwarg_hashes = tuple(
        sorted((k, _get_tensor_hash(v)) for k, v in (adapted_kwargs or {}).items())
    )
    return (op_name, arg_hashes, kwarg_hashes)


def eager_execute(
    op: Any, resharded_args: tuple, kwargs: dict, adapted_kwargs: dict
) -> Any:
    """Execute operation eagerly if enabled by configuration."""
    from ..config import EAGER_MAX_GRAPH

    if not EAGER_MAX_GRAPH:
        return None

    from ..core import GRAPH, Tensor, pytree

    # PRE-EXECUTION: Ensure all inputs are valid in the current graph context.
    seen_impls = set()
    inputs_to_hydrate = []
    inputs_to_replay = []

    def check_input_validity(x):
        if isinstance(x, Tensor):
            impl = x._impl
            if id(impl) in seen_impls:
                return
            seen_impls.add(id(impl))

            if impl.graph_values_epoch != GRAPH.epoch:
                if impl.is_realized:
                    inputs_to_hydrate.append(x)
                elif impl.output_refs:
                    inputs_to_replay.append(x)

    pytree.tree_map(check_input_validity, resharded_args)
    pytree.tree_map(check_input_validity, adapted_kwargs)

    # 1. Bring realized tensors into current graph
    for t in inputs_to_hydrate:
        GRAPH.add_input(t)

    # 2. Replay trace for unrealized tensors (stale promises)
    if inputs_to_replay:
        GRAPH._replay_trace_to_build_graph(inputs_to_replay)

    # Build graph immediately
    try:
        # Note: We pass original kwargs, as some ops (like ReduceOperation)
        # adapt them internally during execute().
        return op.execute(resharded_args, kwargs)
    except Exception as e:
        raise RuntimeError(f"Eager graph building failed for {op.name}: {e}") from e


def verify_eager_shapes(
    op: Any, execution_results: Any, output_physical_shapes: Any
) -> None:
    """Verify that eager execution produced the expected physical shapes."""
    from ..config import VERIFY_EAGER_SHAPES

    if not VERIFY_EAGER_SHAPES or execution_results is None:
        return

    shard_vals = execution_results[0]

    def verify_shape_fn(val, expected_shape, shard_idx):
        if val is None:
            return
        actual = tuple(int(d) for d in val.type.shape)
        expected = (
            tuple(int(d) for d in expected_shape)
            if expected_shape is not None
            else None
        )
        if actual != expected:
            raise RuntimeError(
                f"Shape mismatch in {op.name} (Shard {shard_idx}): "
                f"Execution produced {actual}, but compute_physical_shape predicted {expected}. "
                f"This indicates a bug in {op.__class__.__name__}.compute_physical_shape."
            )

    if (
        isinstance(output_physical_shapes, list)
        and output_physical_shapes
        and isinstance(output_physical_shapes[0], list)
    ):
        # Multi-output
        for shard_idx, vals_tuple in enumerate(shard_vals):
            if len(vals_tuple) != len(output_physical_shapes):
                raise RuntimeError(
                    f"Output count mismatch for {op.name} (Shard {shard_idx}): "
                    f"Expected {len(output_physical_shapes)} outputs, got {len(vals_tuple)}."
                )
            for out_idx, val in enumerate(vals_tuple):
                verify_shape_fn(
                    val, output_physical_shapes[out_idx][shard_idx], shard_idx
                )
    else:
        # Single output
        for shard_idx, val in enumerate(shard_vals):
            verify_shape_fn(val, output_physical_shapes[shard_idx], shard_idx)


def package_outputs(
    op: Any,
    execution_results: Any,
    output_physical_shapes: Any,
    output_shard_dtypes: Any,
    output_shard_devices: Any,
    predicted_output_spec: Any,
    mesh: Any,
    any_traced: bool,
    max_batch_dims: int,
) -> Any:
    """Create resulting Tensor(s) with appropriate metadata and graph references."""
    from ..core import GRAPH
    from ..core.sharding import spmd
    from ..config import EAGER_MAX_GRAPH

    # Detect multi-output mode
    is_multi_output = False
    if output_physical_shapes is not None:
        is_multi_output = (
            isinstance(output_physical_shapes, list)
            and output_physical_shapes
            and isinstance(output_physical_shapes[0], list)
        )
    elif isinstance(predicted_output_spec, (list, tuple)):
        is_multi_output = True
    elif (
        execution_results
        and execution_results[0]
        and isinstance(execution_results[0][0], (list, tuple))
    ):
        is_multi_output = True

    if is_multi_output:
        # 1. Determine number of outputs
        if output_physical_shapes is not None:
            num_outputs = len(output_physical_shapes)
        elif isinstance(predicted_output_spec, (list, tuple)):
            num_outputs = len(predicted_output_spec)
        else:
            num_outputs = len(execution_results[0][0])

        # 2. Transpose shard-major results to output-major if available
        if EAGER_MAX_GRAPH and execution_results:
            transposed_results = list(zip(*execution_results[0]))
        else:
            transposed_results = [[] for _ in range(num_outputs)]

        outputs = []
        for i in range(num_outputs):
            out_sharding = (
                predicted_output_spec[i]
                if isinstance(predicted_output_spec, (list, tuple))
                else predicted_output_spec
            )

            # Use provided metadata if available, otherwise pass None and let Tensor infer from values
            out_shapes = output_physical_shapes[i] if output_physical_shapes else None
            out_dtypes = (
                output_shard_dtypes[i]
                if isinstance(output_shard_dtypes, list)
                and len(output_shard_dtypes) > i
                else output_shard_dtypes
            )
            out_devices = (
                output_shard_devices[i]
                if isinstance(output_shard_devices, list)
                and len(output_shard_devices) > i
                else output_shard_devices
            )

            o = spmd.create_sharded_output(
                list(transposed_results[i]) if EAGER_MAX_GRAPH else [],
                out_sharding,
                any_traced,
                max_batch_dims,
                mesh=mesh,
                physical_shapes=out_shapes,
                shard_dtypes=out_dtypes,
                shard_devices=out_devices,
            )

            if EAGER_MAX_GRAPH:
                o._impl.graph_values_epoch = GRAPH.epoch
            else:
                o._impl.graph_values_epoch = -1
                GRAPH.add_unrealized(o._impl)
            outputs.append(o)

        container_type = getattr(op, "output_container_type", tuple)
        return container_type(outputs)
    else:
        output = spmd.create_sharded_output(
            execution_results[0] if EAGER_MAX_GRAPH and execution_results else [],
            predicted_output_spec,
            any_traced,
            max_batch_dims,
            mesh=mesh,
            physical_shapes=output_physical_shapes,
            shard_dtypes=output_shard_dtypes,
            shard_devices=output_shard_devices,
        )

        if EAGER_MAX_GRAPH:
            output._impl.graph_values_epoch = GRAPH.epoch
        else:
            output._impl.graph_values_epoch = -1
            GRAPH.add_unrealized(output._impl)
        return output


def apply_auto_reduction(op: Any, output: Any, mesh: Any, reduce_axes: Any) -> Any:
    """Apply automatic reductions (all-reduce) if required by sharding propagation."""
    if not (reduce_axes and mesh):
        return output

    from .communication.all_reduce import all_reduce
    from ..core import pytree, Tensor

    def _apply_fn(t):
        if isinstance(t, Tensor):
            return all_reduce(
                t,
                mesh=mesh,
                reduce_axes=reduce_axes,
                reduce_op=op.collective_reduce_type,
            )
        return t

    return pytree.tree_map(_apply_fn, output)


def call_custom_kernel(
    func_name: str,
    kernel_path: Union[str, Path, list[Union[str, Path]]],
    values: Union[TensorValue, list[TensorValue]],
    out_types: Union[Any, list[Any]],
    device: None | DeviceRef = None,
    **kwargs: Any,
) -> Union[TensorValue, list[TensorValue]]:
    """Helper to invoke a custom Mojo kernel, handling library loading automatically.

    Args:
        func_name: The name of the registered Mojo kernel (e.g. @register("name")).
        kernel_path: Path(s) to the kernel source file or directory.
        values: Input TensorValue(s).
        out_types: Expected output type(s).
        device: Device to run on (default: CPU).
        **kwargs: Additional arguments passed to ops.custom.

    Returns:
        Result TensorValue(s).
    """
    from max.graph import DeviceRef, TensorValue, ops
    from ..core import GRAPH

    if device is None:
        device = DeviceRef.CPU()

    if not isinstance(kernel_path, list):
        kernel_path = [kernel_path]

    if isinstance(values, TensorValue):
        values_list = [values]
    else:
        values_list = values
    unwrap_result = False
    if not isinstance(out_types, list):
        out_types_list = [out_types]
        unwrap_result = True
    else:
        out_types_list = out_types

    resolved_paths: list[Path] = []
    for p in kernel_path:
        path_obj = Path(p).resolve()
        resolved_paths.append(path_obj)
    GRAPH.graph._kernel_library.load_paths(GRAPH.graph._context, resolved_paths)

    results = ops.custom(
        name=func_name,
        device=device,
        values=values_list,
        out_types=out_types_list,
        **kwargs,
    )

    if results:
        op_instance = results[0].to_mlir().owner
        GRAPH.graph._kernel_library.verify_custom_op(op_instance)

    if unwrap_result:
        return results[0]
    return results
