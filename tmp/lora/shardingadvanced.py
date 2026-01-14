import itertools
import math
import numpy as np

# --- Step 1: Core Placement Logic (Our proven functions) ---

def get_device_placements(tensor_shape, mesh_shape, sharding_spec):
    all_devices = itertools.product(*(range(s) for s in mesh_shape))
    return {
        device: _get_slices_for_device(device, tensor_shape, mesh_shape, sharding_spec)
        for device in all_devices
    }

def _get_slices_for_device(device_coord, tensor_shape, mesh_shape, sharding_spec):
    return tuple(
        slice(None) if spec is None else _get_partitioned_slice(
            dim_size, spec, device_coord, mesh_shape
        )
        for dim_size, spec in zip(tensor_shape, sharding_spec)
    )

def _get_partitioned_slice(dim_size, spec, device_coord, mesh_shape):
    mesh_axes = (spec,) if isinstance(spec, int) else spec
    num_shards = math.prod(mesh_shape[axis] for axis in mesh_axes)
    if dim_size % num_shards != 0:
        raise ValueError("Dimension cannot be sharded evenly.")
    shard_size = dim_size // num_shards
    linear_index = 0
    stride = 1
    for axis in reversed(mesh_axes):
        linear_index += device_coord[axis] * stride
        stride *= mesh_shape[axis]
    start = linear_index * shard_size
    return slice(start, start + shard_size)

# --- Step 2: The Data Structures ---

class Mesh:
    """Represents the device grid."""
    def __init__(self, mesh_shape):
        self.shape = mesh_shape
        self.size = math.prod(mesh_shape)
        # In a real system, this would hold actual device objects.
        # Here we just generate the coordinates for iteration.
        self.device_coords = list(itertools.product(*(range(s) for s in mesh_shape)))

class ShardedTensor:
    """Represents a single logical tensor physically distributed across a mesh."""
    def __init__(self, global_shape, dtype, mesh, sharding_spec):
        self.global_shape = global_shape
        self.dtype = dtype
        self.mesh = mesh
        self.sharding_spec = sharding_spec
        self.placements = get_device_placements(global_shape, mesh.shape, sharding_spec)
        self._shards = {} # This will map device_coord -> numpy array (the shard)

    def local_shape(self, device_coord):
        """Calculates the shape of a shard on a specific device."""
        slices = self.placements[device_coord]
        return tuple(
            self.global_shape[i] if s.start is None else s.stop - s.start
            for i, s in enumerate(slices)
        )

    def get_shard(self, device_coord):
        """Returns the actual data shard for a given device."""
        return self._shards.get(device_coord)

    def to_global_tensor(self):
        """Materializes the full tensor by assembling all shards. For verification."""
        global_array = np.zeros(self.global_shape, dtype=self.dtype)
        for device_coord, slices in self.placements.items():
            shard = self.get_shard(device_coord)
            if shard is not None:
                global_array[slices] = shard
        return global_array

    def __repr__(self):
        return (f"ShardedTensor(global_shape={self.global_shape}, "
                f"mesh={self.mesh.shape}, spec={self.sharding_spec})")

# --- Step 3: A Distributed Operation ---

def add(A: ShardedTensor, B: ShardedTensor) -> ShardedTensor:
    """Simulates a distributed element-wise addition."""
    # 1. Verify compatibility of the inputs
    if A.global_shape != B.global_shape or A.mesh.shape != B.mesh.shape or A.sharding_spec != B.sharding_spec:
        raise ValueError("ShardedTensors must have matching metadata for element-wise add.")

    # 2. Create the output tensor "shell"
    C = ShardedTensor(A.global_shape, A.dtype, A.mesh, A.sharding_spec)

    # 3. SIMULATED: Dispatch local computations to each device in parallel
    print("\n--- Running Simulated Distributed Add ---")
    for device_coord in A.mesh.device_coords:
        shard_A = A.get_shard(device_coord)
        shard_B = B.get_shard(device_coord)
        
        # This is the actual computation happening locally on each device's data
        result_shard = shard_A + shard_B
        
        # Store the resulting shard in the output tensor
        C._shards[device_coord] = result_shard
        print(f"  Device {device_coord}: Computed {shard_A.shape} + {shard_B.shape} -> {result_shard.shape}")
    print("--- Distributed Add Complete ---\n")

    return C

# --- Step 4: The Full Working Test ---

if __name__ == "__main__":
    # A. Define the distributed environment and sharding strategy
    mesh = Mesh((2, 2))
    global_shape = (8, 4)
    sharding_spec = (0, 1) # Shard dim 0 on mesh axis 0, dim 1 on mesh axis 1

    print("="*60)
    print("Test Scenario: Adding two ShardedTensors")
    print(f"Global Shape: {global_shape}, Mesh: {mesh.shape}, Sharding: {sharding_spec}")
    print("="*60)

    # B. Create Tensor A, filled with ones
    print("Initializing Tensor A...")
    Tensor_A = ShardedTensor(global_shape, np.int32, mesh, sharding_spec)
    for device in mesh.device_coords:
        Tensor_A._shards[device] = np.ones(Tensor_A.local_shape(device), dtype=np.int32)
    
    # C. Create Tensor B, filled with a range of numbers
    print("Initializing Tensor B...")
    Tensor_B = ShardedTensor(global_shape, np.int32, mesh, sharding_spec)
    # Create a global tensor first to easily slice it for our shards
    global_B_source = np.arange(np.prod(global_shape), dtype=np.int32).reshape(global_shape)
    for device in mesh.device_coords:
        slices = Tensor_B.placements[device]
        Tensor_B._shards[device] = global_B_source[slices]
        
    # Show the state of a single shard for inspection
    dev_to_inspect = (1, 0)
    print(f"\nInspecting shards on Device {dev_to_inspect}:")
    print(f"  Shard A:\n{Tensor_A.get_shard(dev_to_inspect)}")
    print(f"  Shard B:\n{Tensor_B.get_shard(dev_to_inspect)}")

    # D. Execute the distributed operation
    Tensor_C = add(Tensor_A, Tensor_B)

    # E. VERIFICATION: Materialize all tensors and check the result
    print("--- Verification ---")
    global_A = Tensor_A.to_global_tensor()
    global_B = Tensor_B.to_global_tensor()
    global_C = Tensor_C.to_global_tensor()

    print("Global Tensor A (reconstructed):\n", global_A)
    print("\nGlobal Tensor B (reconstructed):\n", global_B)
    print("\nGlobal Tensor C (reconstructed from distributed result):\n", global_C)

    # The ultimate proof
    expected_C = global_A + global_B
    is_correct = np.array_equal(global_C, expected_C)

    print("\n" + "="*60)
    if is_correct:
        print("✅ SUCCESS: The reconstructed result of the distributed add matches the expected result.")
    else:
        print("❌ FAILURE: The distributed computation produced an incorrect result.")
    print("="*60)