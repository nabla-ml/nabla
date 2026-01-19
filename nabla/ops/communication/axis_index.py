# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List

from max.graph import TensorValue, ops

from ..base import Operation

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh


class AxisIndexOp(Operation):
    """Return the device's position along a mesh axis.
    
    This is essential for shard_map-style programming where each device
    needs to know its position for conditional logic.
    
    Example:
        idx = axis_index('i')  # Returns 0, 1, 2, 3 on 4 devices
    """
    
    @property
    def name(self) -> str:
        return "axis_index"
    
    def maxpr(
        self,
        mesh: "DeviceMesh",
        axis_name: str,
        shard_idx: int,
    ) -> TensorValue:
        """Return this device's index along the specified axis.
        
        Args:
            mesh: Device mesh
            axis_name: Name of axis to get index for
            shard_idx: Current shard index
            
        Returns:
            Scalar TensorValue with the axis index
        """
        coord = mesh.get_coordinate(shard_idx, axis_name)
        return ops.constant(coord, mesh.device_refs[shard_idx].dtype if hasattr(mesh.device_refs[shard_idx], 'dtype') else None)
    
    def __call__(self, mesh: "DeviceMesh", axis_name: str):
        """Get axis indices for all devices.
        
        Args:
            mesh: Device mesh
            axis_name: Name of axis to get indices for
            
        Returns:
            Tensor (sharded/distributed) containing the index for each device.
        """
        from ...core import Tensor
        from ...core import GRAPH
        from ...core.sharding.spec import ShardingSpec, DimSpec
        from max.dtype import DType
        # No DeviceRef in max.graph, it's typically in max.driver or inferred. Using int here for simplicity.
        # But we need to specify device placement if using ops.constant
        
        results = []
        with GRAPH.graph:
            for shard_idx in range(len(mesh.devices)):
                coord = mesh.get_coordinate(shard_idx, axis_name)
                # Use device ref from mesh, or default to CPU
                device = mesh.device_refs[shard_idx] if mesh.device_refs else None
                # Constant needs explicit device if distributed?
                # MAX ops.constant usually takes (value, dtype, device)
                val = ops.constant(coord, DType.int32, device)
                # Reshape to (1,) to match sharded 1D tensor logic
                val = ops.reshape(val, (1,))
                results.append(val)
        
        # Result is a 1D tensor [0, 1, 2...] sharded on axis_name
        # Shape: (axis_size,)
        spec = ShardingSpec(mesh, [DimSpec([axis_name])])
        
        output = Tensor._create_unsafe(
            values=results,
            traced=False,
            batch_dims=0,
            sharding=spec
        )
        return output


# Singleton instance
axis_index_op = AxisIndexOp()

# Public API
def axis_index(mesh: "DeviceMesh", axis_name: str):
    """Return each device's position along a mesh axis.
    
    Essential for shard_map-style programming where devices need to
    know their position for conditional logic.
    
    Args:
        mesh: Device mesh
        axis_name: Name of axis to get indices for
        
    Returns:
        List of scalar TensorValues, one per device, containing 0, 1, 2, ...
    """
    return axis_index_op(mesh, axis_name)
