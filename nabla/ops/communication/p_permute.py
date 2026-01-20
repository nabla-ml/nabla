# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING

from max.graph import TensorValue, ops

from .base import CollectiveOperation

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh


class PPermuteOp(CollectiveOperation):
    """Point-to-point permutation collective.

    Each device sends its value to exactly one other device according to a
    permutation table.
    """

    @property
    def name(self) -> str:
        return "ppermute"

    def maxpr(
        self,
        shard_values: list[TensorValue],
        permutation: list[tuple],
        mesh: DeviceMesh = None,
    ) -> list[TensorValue]:
        """Permute values between devices according to permutation."""
        num_devices = len(shard_values)

        dest_to_src = {}
        for src, dst in permutation:
            if dst in dest_to_src:
                raise ValueError(
                    f"Destination {dst} appears multiple times in permutation"
                )
            dest_to_src[dst] = src

        results = []

        for dst in range(num_devices):
            if dst in dest_to_src:
                src = dest_to_src[dst]
                val = shard_values[src]

                if mesh and mesh.is_distributed:
                    val = ops.transfer_to(val, mesh.device_refs[dst])

                results.append(val)
            else:

                template = shard_values[0]
                zero_val = ops.constant(0, template.type.dtype, template.type.device)
                zero_val = ops.broadcast_to(zero_val, template.type.shape)
                results.append(zero_val)

        return results


ppermute_op = PPermuteOp()


def ppermute(sharded_tensor, permutation: list[tuple]):
    """Point-to-point permutation collective."""
    return ppermute_op(sharded_tensor, permutation=permutation)
