import os
# This flag tells JAX to create 8 virtual CPU devices.
# It MUST be set before jax is imported.
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
import jax.numpy as jnp
# We need numpy to reshape the device list
import numpy as np
from jax.sharding import Mesh, PartitionSpec, NamedSharding

# This check will pass.
if len(jax.devices()) < 8:
  raise Exception(f"Device simulation failed. Only found {len(jax.devices())} devices.")

print(f"Successfully simulated {len(jax.devices())} devices!")

# --- THE FIX IS HERE ---
# 1. Get the flat list of 8 devices.
devices = jax.devices()
print(devices)

# 2. Reshape this flat list into a 2x4 NumPy array.
#    This 2D array now matches the 2D mesh ('x', 'y') we want to create.
device_array = np.array(devices).reshape((2, 4))
# 3. Create the mesh from the reshaped 2D device array.
mesh = Mesh(device_array, ('x', 'y'))

# --- The rest of the code works as before ---

# Create a sample array of shape (16, 32).
my_array = jnp.arange(16 * 32, dtype=jnp.float32).reshape(16, 32)

# P is a convenient alias for PartitionSpec.
P = PartitionSpec

# Shard the first array axis by 'x' and the second by 'y'.
meshmap = P(None, "x")
sharding = NamedSharding(mesh, meshmap)
sharded_array_3 = jax.device_put(my_array, sharding)

print(f"\n--- Scenario 3: Sharding both dimensions {meshmap} ---")

# Visualize how the data is laid out across the virtual devices.
jax.debug.visualize_array_sharding(sharded_array_3)