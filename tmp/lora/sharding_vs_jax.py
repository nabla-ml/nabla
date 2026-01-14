import itertools
import math
import os
import numpy as np

# --- Our Sharding Logic (Unchanged) ---

def get_device_placements(tensor_shape, mesh_shape, sharding_spec):
    if len(tensor_shape) != len(sharding_spec):
        raise ValueError("tensor_shape and sharding_spec must have the same length.")
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
        raise ValueError(f"Tensor dimension (size {dim_size}) cannot be evenly sharded across {num_shards} devices.")
    shard_size = dim_size // num_shards
    linear_index = 0
    stride = 1
    for axis in reversed(mesh_axes):
        linear_index += device_coord[axis] * stride
        stride *= mesh_shape[axis]
    start = linear_index * shard_size
    return slice(start, start + shard_size)

# --- JAX Validation Logic (Corrected) ---

# This must be set BEFORE jax is imported to ensure enough virtual devices are created
# We find the biggest mesh we need (8 devices for a 2x2x2 mesh) and request that many.
MAX_DEVICES_NEEDED = 8
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={MAX_DEVICES_NEEDED}'

try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec, NamedSharding
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    
def get_jax_placements(tensor_shape, mesh_shape, sharding_spec):
    """
    Uses JAX's actual sharding mechanism to determine device placements.
    """
    num_devices_required = math.prod(mesh_shape)
    
    # --- THIS IS THE CORRECTED SECTION ---
    # The old `jax.experimental.mesh_utils` is deprecated.
    # The modern way is to get the flat list of devices and reshape it.
    available_devices = jax.devices()
    if len(available_devices) < num_devices_required:
        raise RuntimeError(f"Not enough JAX devices available. Need {num_devices_required}, have {len(available_devices)}")
        
    # Take the first N devices and reshape them into our desired mesh topology
    devices = np.array(available_devices[:num_devices_required]).reshape(mesh_shape)
    # --- END OF CORRECTION ---

    axis_names = ('x', 'y', 'z', 'w')[:len(mesh_shape)]
    mesh = Mesh(devices, axis_names=axis_names)

    def to_jax_spec(spec):
        if spec is None: return None
        if isinstance(spec, int): return axis_names[spec]
        return tuple(axis_names[axis] for axis in spec)

    jax_partition_spec = PartitionSpec(*(to_jax_spec(s) for s in sharding_spec))
    sharding = NamedSharding(mesh, jax_partition_spec)
    dummy_array = jnp.empty(tensor_shape)
    sharded_array = jax.device_put(dummy_array, sharding)

    jax_placements = {}
    device_id_map = {device.id: coord for coord, device in np.ndenumerate(devices)}

    for shard in sharded_array.addressable_shards:
        device_coord = device_id_map[shard.device.id]
        jax_placements[device_coord] = shard.index
        
    return jax_placements

# --- --- --- --- --- ---
#      MAIN VALIDATION
# --- --- --- --- --- ---
if __name__ == "__main__":
    if not JAX_AVAILABLE:
        print("JAX is not installed. Skipping validation.")
        print("Please run: pip install jax numpy")
    else:
        jax.config.update("jax_enable_x64", True)

        test_suite = [
            ((4096, 1024), (4,), (0, None)),
            ((4096, 1024), (4,), (None, 0)),
            ((1024, 2048, 512), (4, 2), (0, 1, None)),
            ((1024, 2048, 512), (4, 2), ((0, 1), None, None)),
            ((1024, 2048, 512), (4, 2), ((1, 0), None, None)),
            ((128, 64, 32, 16), (2, 2, 2), (0, 1, 2, None)),
            ((128, 64, 32, 16), (2, 2, 2), (None, (0, 1, 2), None, None)),
            ((128, 64, 32, 16), (2, 2, 2), ((1, 2), None, 0, None)),
        ]

        print("=" * 60)
        print("### Running Validation Against JAX Sharding Logic ###")
        print("=" * 60)

        all_passed = True
        for i, (t_shape, m_shape, s_spec) in enumerate(test_suite):
            print(f"\n--- Test Case {i+1} ---")
            print(f"Config: T={t_shape}, M={m_shape}, S={s_spec}")
            
            try:
                our_placements = get_device_placements(t_shape, m_shape, s_spec)
                jax_placements = get_jax_placements(t_shape, m_shape, s_spec)
                assert our_placements == jax_placements
                print("✅ PASSED: Output matches JAX.")
            except Exception as e:
                print(f"❌ FAILED: {e}")
                all_passed = False
        
        print("\n" + "=" * 60)
        if all_passed:
            print("🎉 All test cases passed validation against JAX! The implementation is correct.")
        else:
            print("🔥 One or more test cases failed validation.")