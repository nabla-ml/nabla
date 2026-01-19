# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List

from max.graph import TensorValue, ops

from .base import CollectiveOperation

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh


class AllToAllOp(CollectiveOperation):
    """All-to-all collective (distributed transpose).
    
    Each device splits its tensor along split_axis, sends parts to other devices,
    receives from all, and concatenates along concat_axis.
    """
    
    @property
    def name(self) -> str:
        return "all_to_all"
    
    def maxpr(
        self,
        shard_values: List[TensorValue],
        split_axis: int,
        concat_axis: int,
        mesh: "DeviceMesh" = None,
        tiled: bool = True,
    ) -> List[TensorValue]:
        """All-to-all: distributed transpose of tensor blocks."""
        num_devices = len(shard_values)
        
        if num_devices <= 1:
            return shard_values
        
        # 1. Each device splits its tensor into num_devices chunks
        chunks_per_device = []
        for val in shard_values:
            shape = val.type.shape
            axis_size = int(shape[split_axis])
            chunk_size = axis_size // num_devices
            
            if axis_size % num_devices != 0:
                raise ValueError(
                    f"Split axis size {axis_size} not divisible by {num_devices} devices"
                )
            
            chunks = []
            for i in range(num_devices):
                slices = [slice(None)] * len(shape)
                slices[split_axis] = slice(i * chunk_size, (i + 1) * chunk_size)
                chunks.append(val[tuple(slices)])
            
            chunks_per_device.append(chunks)
        
        # 2. Transpose: device j collects chunk[i][j] from each device i
        received_per_device = []
        for dst in range(num_devices):
            received = []
            for src in range(num_devices):
                chunk = chunks_per_device[src][dst]
                
                # DISTRIBUTED: Transfer to destination device
                if mesh and mesh.is_distributed:
                    chunk = ops.transfer_to(chunk, mesh.device_refs[dst])
                
                received.append(chunk)
            received_per_device.append(received)
        
        # 3. Each device concatenates (or stacks) received chunks
        results = []
        for dst in range(num_devices):
            if tiled:
                concatenated = ops.concat(received_per_device[dst], axis=concat_axis)
            else:
                concatenated = ops.stack(received_per_device[dst], axis=concat_axis)
            results.append(concatenated)
        
        return results


# Singleton instance
all_to_all_op = AllToAllOp()

# Public API
def all_to_all(sharded_tensor, split_axis: int, concat_axis: int, tiled: bool = True):
    """All-to-all collective (distributed transpose)."""
    return all_to_all_op(sharded_tensor, split_axis=split_axis, concat_axis=concat_axis, tiled=tiled)
