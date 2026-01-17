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


class PPermuteOp(CollectiveOperation):
    """Point-to-point permutation collective.
    
    Each device sends its value to exactly one other device according to a
    permutation table. This is useful for ring-based algorithms, pipeline
    parallelism, and halo exchange.
    
    Example:
        # Ring shift: device 0→1→2→3→0
        perm = [(0, 1), (1, 2), (2, 3), (3, 0)]
        y = ppermute(x, perm)
    """
    
    @property
    def name(self) -> str:
        return "ppermute"
    
    def maxpr(
        self,
        shard_values: List[TensorValue],
        permutation: List[tuple],
        mesh: "DeviceMesh" = None,
    ) -> List[TensorValue]:
        """Permute values between devices according to permutation.
        
        Args:
            shard_values: List of TensorValues, one per device
            permutation: List of (source_idx, dest_idx) pairs
            mesh: Device mesh for distributed execution
            
        Returns:
            List of TensorValues after permutation (zeros for missing dests)
        """
        num_devices = len(shard_values)
        
        # Build reverse map: dest -> src
        dest_to_src = {}
        for src, dst in permutation:
            if dst in dest_to_src:
                raise ValueError(f"Destination {dst} appears multiple times in permutation")
            dest_to_src[dst] = src
        
        # Create result array
        results = []
        
        for dst in range(num_devices):
            if dst in dest_to_src:
                src = dest_to_src[dst]
                val = shard_values[src]
                
                # DISTRIBUTED: Transfer to destination device
                if mesh and mesh.is_distributed:
                    val = ops.transfer_to(val, mesh.device_refs[dst])
                
                results.append(val)
            else:
                # No sender for this destination - return zeros
                template = shard_values[0]
                zero_val = ops.constant(0, template.type.dtype, template.type.device)
                zero_val = ops.broadcast_to(zero_val, template.type.shape)
                results.append(zero_val)
        
        return results


# Singleton instance
ppermute_op = PPermuteOp()

# Public API
def ppermute(sharded_tensor, permutation: List[tuple]):
    """Point-to-point permutation collective.
    
    Each device sends its value to exactly one other device according to
    a permutation table. Useful for ring-based algorithms, pipeline
    parallelism, and halo exchange.
    
    Args:
        sharded_tensor: Tensor with multiple shards
        permutation: List of (source_idx, dest_idx) pairs specifying
                     which device sends to which. Destinations without
                     senders receive zeros.
        
    Returns:
        Tensor with permuted values
    """
    return ppermute_op(sharded_tensor, permutation=permutation)
