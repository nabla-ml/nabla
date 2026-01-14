import os
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec, NamedSharding

# This flag tells JAX to create 8 virtual CPU devices.
# It MUST be set before jax is imported.
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

# This check will pass.
if len(jax.devices()) < 8:
  raise Exception(f"Device simulation failed. Only found {len(jax.devices())} devices.")

print(f"Successfully simulated {len(jax.devices())} devices!")
print(jax.devices())

# 1. Get the flat list of 8 devices.
devices = jax.devices()

# 2. Reshape this flat list into a 2x4 NumPy array.
#    This 2D array now matches the 2D mesh ('x', 'y') we want to create.
device_array = np.array(devices).reshape((2, 4))

# 3. Create the mesh from the reshaped 2D device array.
mesh = Mesh(device_array, ('x', 'y'))

# Create a sample array of shape (8, 8).
x = jnp.arange(8 * 8, dtype=jnp.float32).reshape(8, 8)

# P is a convenient alias for PartitionSpec.
P = PartitionSpec

# --- Dot Product Visualization ---

# Shard the first input array (lhs) along the 'x' axis.
y = jax.device_put(x, NamedSharding(mesh, P('x', None)))

# Shard the second input array (rhs) along the 'y' axis.
z = jax.device_put(x, NamedSharding(mesh, P(None, 'x')))

print("\n--- Visualizing Dot Product Sharding ---")

print('\nLHS Sharding (y):')
jax.debug.visualize_array_sharding(y)

print('\nRHS Sharding (z):')
jax.debug.visualize_array_sharding(z)

# Perform the dot product.
# JAX will automatically determine the output sharding.
w = jnp.dot(y, z)

print('\nOutput Sharding (w = y @ z):')
jax.debug.visualize_array_sharding(w)