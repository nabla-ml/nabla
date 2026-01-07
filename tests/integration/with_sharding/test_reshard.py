
import unittest
import numpy as np
from nabla.core.tensor import Tensor
from nabla.sharding.spec import DeviceMesh, ShardingSpec, DimSpec
from nabla.ops.communication import reshard, shard
from nabla.core.compute_graph import GRAPH
import asyncio

def get_shard_numpy(tensor, idx):
    """Helper to get numpy data from a potentially realized sharded tensor."""
    if tensor._impl.is_realized:
        storage = tensor._impl._storages[idx]
        # driver.Tensor usually has to_numpy or __array__
        # Assuming to_numpy() based on typical Mojo/Max bindings
        if hasattr(storage, 'to_numpy'):
            return storage.to_numpy()
        return np.array(storage)
    elif tensor._impl._values:
       # Can't get numpy from symbolic value easily without running graph.
       raise ValueError("Tensor must be realized to check values in test.")
    else:
       raise ValueError("Tensor has no values or storages!")

class TestResharding(unittest.TestCase):
    def setUp(self):
        # 4 devices: 2x2 mesh
        self.mesh_2x2 = DeviceMesh("square", (2, 2), ("x", "y"))
        # 4 devices: 1D mesh
        self.mesh_4 = DeviceMesh("flat", (4,), ("d",))
        # Global tensor 4x4
        self.data_np = np.arange(16).reshape(4, 4).astype(np.float32)
        self.tensor = Tensor.from_dlpack(self.data_np)

    def test_reshard_identity(self):
        """Test resharding to identical spec returns same tensor (logic check)."""
        spec = ShardingSpec(self.mesh_2x2, [DimSpec(["x"]), DimSpec(["y"])])
        
        # We need to trace creation to allow reshard op to graph it
        sharded = shard(self.tensor, self.mesh_2x2, spec.dim_specs)
        
        resharded = reshard(sharded, self.mesh_2x2, spec.dim_specs)
        
        # Realize both to compare values
        asyncio.run(sharded.realize)
        asyncio.run(resharded.realize)
        
        # Compare shards manually
        num_shards = len(sharded._impl._storages)
        for i in range(num_shards):
             v1 = get_shard_numpy(resharded, i)
             v2 = get_shard_numpy(sharded, i)
             np.testing.assert_allclose(v1, v2)

    def test_reshard_replicate_to_shard(self):
        """Test Replicated -> Sharded (Simulates ShardOp)."""
        replicated_spec = ShardingSpec(self.mesh_2x2, [DimSpec([]), DimSpec([])])
        target_spec = ShardingSpec(self.mesh_2x2, [DimSpec(["x"]), DimSpec(["y"])])
        
        rep = shard(self.tensor, self.mesh_2x2, replicated_spec.dim_specs)
        
        resharded = reshard(rep, target_spec.mesh, target_spec.dim_specs)
        asyncio.run(resharded.realize)
        
        # Verify specific shard content
        # Top-Left (0,0) -> Index 0 in flat list [0,1,2,3]
        # Mesh 2x2: (0,0), (0,1), (1,0), (1,1)
        # Device 0 is (0,0).
        shard_0 = get_shard_numpy(resharded, 0)
        expected_0 = self.data_np[0:2, 0:2]
        
        np.testing.assert_allclose(shard_0, expected_0)

    def test_reshard_shard_to_replicate(self):
        """Test Sharded -> Replicated (Simulates AllGather)."""
        source_spec = ShardingSpec(self.mesh_2x2, [DimSpec(["x"]), DimSpec(["y"])])
        target_spec = ShardingSpec(self.mesh_2x2, [DimSpec([]), DimSpec([])])
        
        sharded = shard(self.tensor, self.mesh_2x2, source_spec.dim_specs)
        resharded = reshard(sharded, target_spec.mesh, target_spec.dim_specs)
        asyncio.run(resharded.realize)
        
        # Every shard should now be the full tensor
        shard_0 = get_shard_numpy(resharded, 0)
        np.testing.assert_allclose(shard_0, self.data_np)

    def test_reshard_axis_swap(self):
        """Test AllToAll: Swap axes."""
        source_spec = ShardingSpec(self.mesh_2x2, [DimSpec(["x"]), DimSpec(["y"])])
        target_spec = ShardingSpec(self.mesh_2x2, [DimSpec(["y"]), DimSpec(["x"])])
        
        sharded = shard(self.tensor, self.mesh_2x2, source_spec.dim_specs)
        resharded = reshard(sharded, target_spec.mesh, target_spec.dim_specs)
        asyncio.run(resharded.realize)
        
        # Device 1: x=0, y=1.
        # Source (x, y): d0 on x(0), d1 on y(1) -> d0[0:2], d1[2:4] (Top-Right)
        # Target (y, x): d0 on y(1), d1 on x(0) -> d0[2:4], d1[0:2] (Bottom-Left)
        
        shard_1 = get_shard_numpy(resharded, 1) # Target slice: [2:4, 0:2]
        expected_slice = self.data_np[2:4, 0:2]
        
        np.testing.assert_allclose(shard_1, expected_slice)

if __name__ == "__main__":
    unittest.main()
