# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Device transfer operations for moving tensors between devices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.driver import Device
from max.graph import TensorValue

from ..base import Operation

if TYPE_CHECKING:
    from ...core import Tensor


class TransferOp(Operation):
    """Transfer tensor to a different device.

    This is a differentiable operation (identity on forward pass).
    Gradients flow through device transfers.

    Args:
        target_device: Device to transfer to
    """

    def __init__(self, target_device: Device):
        self.target_device = target_device

    @property
    def name(self) -> str:
        return f"transfer_to_{self.target_device}"

    def kernel(self, args: list, kwargs: dict) -> list:
        """Transfer tensor to target device using MAX's transfer_to operation."""
        from max.graph import ops as graph_ops

        x = args[0]
        # Use MAX's built-in transfer_to operation
        return [graph_ops.transfer_to(x, self.target_device)]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Physical shape matches input shape, but on target device."""
        x = args[0]

        # Get input shape
        if hasattr(x, "physical_local_shape"):
            local_shape = x.physical_local_shape(0)
            if local_shape is not None:
                shapes = [tuple(int(d) for d in local_shape)]
            else:
                shapes = [tuple(int(d) for d in x.shape)]
        else:
            shapes = [tuple(int(d) for d in x.shape)]

        # Same dtype as input
        dtypes = [x.dtype]

        # Output is on target device
        devices = [self.target_device]

        return shapes, dtypes, devices

    def execute(self, args: list, kwargs: dict) -> Any:
        """Execute device transfer.

        Returns:
            tuple: (shard_results, output_sharding, mesh)
        """
        from ...core import GRAPH
        from ...core.sharding import spmd

        x = args[0]

        # For device transfer, we don't use mesh-based execution
        # We simply execute the kernel and let the physical shape
        # computation place the result on the target device

        with GRAPH.graph:
            # Execute kernel (identity operation)
            result = self.kernel([x.value if hasattr(x, "value") else x], kwargs)[0]

        # No sharding for single-device transfer
        output_sharding = None
        mesh = None

        return ([result], output_sharding, mesh)

    def vjp_rule(
        self, primals: list, cotangents: list, outputs: list, kwargs: dict
    ) -> list:
        """VJP for device transfer: identity (gradient stays on output device).

        The gradient is already on the correct device (same as output),
        so we just return it as-is. If the input was on a different device,
        we need to transfer the gradient back.
        """
        from ...core import Tensor

        x = primals[0]

        # If input and output are on different devices, transfer gradient back
        if isinstance(x, Tensor) and isinstance(cotangents[0], Tensor):
            if x.device != cotangents[0].device:
                # Transfer gradient back to input's device
                return [to_device(cotangents[0], x.device)]

        # Otherwise gradient is already on correct device
        return [cotangents[0]]

    def infer_sharding_spec(self, args, mesh, kwargs):
        """Device transfer doesn't change sharding."""
        # No sharding changes for device transfer
        return None, [None] * len(args), False


def to_device(
    x: Tensor, device: Device | Any = None, *, sharding: Any = None
) -> Tensor:
    """Transfer tensor to specified device or sharding.

    Like JAX's device_put, this function supports both:
    - Single device transfer: `to_device(x, CPU())`
    - Multi-device sharding: `to_device(x, sharding=my_sharding_spec)`

    This operation is differentiable - gradients flow through the transfer.
    On the forward pass, data is moved/distributed.
    On the backward pass, gradients are transferred back to the input's layout.

    Args:
        x: Input tensor
        device: Target device (Device object). If None and sharding is None,
                returns x as-is if already on a device, otherwise moves to default device.
        sharding: Target ShardingSpec for multi-device distribution (mutually exclusive with device)

    Returns:
        Tensor on the target device or with target sharding

    Examples:
        >>> from max.driver import CPU, Accelerator
        >>> # Single device transfer
        >>> x = zeros((10,), device=CPU())
        >>> y = to_device(x, Accelerator())
        >>> assert y.device == Accelerator()

        >>> # Multi-device sharding (like JAX)
        >>> mesh = DeviceMesh(...)
        >>> spec = ShardingSpec(mesh, [DimSpec.SHARD, DimSpec.REPLICATE])
        >>> y = to_device(x, sharding=spec)  # Distributes across mesh

        >>> # Default device behavior (like JAX with device=None)
        >>> y = to_device(x)  # No-op if already on device, else to default

        >>> # Gradients flow through
        >>> def f(x):
        ...     y = to_device(x, Accelerator())
        ...     return y.sum()
        >>> grad = vjp(f)(x)
        >>> assert grad.device == x.device  # Gradient back on CPU

    Note:
        When using sharding, the operation becomes a communication op
        (similar to shard()) and distributes the tensor across multiple devices.
    """
    from ...core import Tensor

    if not isinstance(x, Tensor):
        raise TypeError(f"Expected Tensor, got {type(x)}")

    # Validate parameters
    if device is not None and sharding is not None:
        raise ValueError(
            "Cannot specify both 'device' and 'sharding' - they are mutually exclusive"
        )

    # Case 1: Multi-device sharding (like JAX's device_put with Sharding)
    if sharding is not None:
        from ..communication.shard import shard
        from ...core.sharding.spec import ShardingSpec

        if not isinstance(sharding, ShardingSpec):
            raise TypeError(
                f"Expected ShardingSpec for sharding parameter, got {type(sharding)}"
            )

        # Use existing shard operation for multi-device distribution
        return shard(
            x,
            sharding.mesh,
            sharding.dim_specs,
            replicated_axes=sharding.replicated_axes,
        )

    # Case 2: device=None - JAX behavior (no-op if on device, else default)
    if device is None:
        # If already on any device, return as-is
        if x.device is not None:
            return x
        # Otherwise transfer to default device
        from ...core.common.context import defaults

        _, device = defaults()

    # Case 3: Single device transfer
    # No-op if already on target device
    if x.device == device:
        return x

    # Create transfer operation
    op = TransferOp(device)
    return op([x], {})[0]


def cpu(x: Tensor) -> Tensor:
    """Transfer tensor to CPU.

    Convenience function for `to_device(x, CPU())`.

    Args:
        x: Input tensor

    Returns:
        Tensor on CPU
    """
    from max.driver import CPU

    return to_device(x, CPU())


def gpu(x: Tensor) -> Tensor:
    """Transfer tensor to GPU/Accelerator.

    Convenience function for `to_device(x, Accelerator())`.

    Args:
        x: Input tensor

    Returns:
        Tensor on GPU/Accelerator

    Note:
        Requires an accelerator to be available. Use `accelerator_count()` to check.
    """
    from max.driver import Accelerator

    return to_device(x, Accelerator())


def accelerator(x: Tensor, device_id: int = 0) -> Tensor:
    """Transfer tensor to specific accelerator device.

    Args:
        x: Input tensor
        device_id: Accelerator device ID (default: 0)

    Returns:
        Tensor on specified accelerator
    """
    from max.driver import Accelerator

    return to_device(x, Accelerator(device_id))


transfer_to = to_device

__all__ = [
    "TransferOp",
    "to_device",
    "transfer_to",
    "cpu",
    "gpu",
    "accelerator",
]
