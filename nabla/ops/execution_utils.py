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
