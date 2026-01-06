# Here is how JAX supposedly handles vmapping over sharded fucntions!

To show how JAX handles this, we will create a 2D mesh:data axis: Used for the vmap (batch) dimension.model axis: Used for sharding internal weights (model parallelism).In this example, JAX "reasons" that since the input is sharded on data, the intermediate products of the vmap should also stay sharded on data to avoid communication.JAX Example: vmap + 2D Mesh ShardingPythonimport jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.pjit import with_sharding_constraint
from jax.experimental import mesh_utils

# 1. Setup a 2D Mesh (2 devices for data, 4 for model = 8 devices total)
# If you have fewer devices, JAX will map this to your available hardware.
devices = mesh_utils.create_device_mesh((2, 4))
mesh = Mesh(devices, axis_names=('data', 'model'))

def sharded_func(x, weights):
    """
    Inside this function, we assume x is a single 'example' from vmap.
    We will shard the weights across the 'model' axis.
    """
    # Force the weights to be sharded across the 'model' axis
    # even though this is inside a vmap!
    sharded_weights = with_sharding_constraint(weights, P('model', None))
    
    # Simple computation
    return jnp.dot(x, sharded_weights)

# 2. Prepare Data
# Let's say batch size 16, feature dim 128, output dim 256
key = jax.random.PRNGKey(0)
batch_input = jax.random.normal(key, (16, 128))
weights = jax.random.normal(key, (128, 256))

# 3. Use vmap with spmd_axis_name
# This is the 'magic' part: we tell vmap that its mapping axis 
# corresponds to the 'data' axis of our physical mesh.
vmapped_sharded_func = jax.vmap(
    sharded_func, 
    in_axes=(0, None), 
    spmd_axis_name='data'
)

# 4. Final JIT with global sharding constraints
@jax.jit
def run_parallel(x, w):
    # We tell JAX the input x is sharded across 'data'
    x = with_sharding_constraint(x, P('data', None))
    return vmapped_sharded_func(x, w)

# Execute
with mesh:
    result = run_parallel(batch_input, weights)

print(f"Result shape: {result.shape}")
print(f"Result sharding: {result.sharding}")
How JAX "Reasons" About ThisInside the run_parallel function, JAX's compiler performs Sharding Propagation:The Input (x): You told JAX x is sharded on the data axis.The vmap: Because you used spmd_axis_name='data', JAX knows the $0^{th}$ dimension of the batch corresponds exactly to the data partitioning of the devices.The Intermediates: Inside sharded_func, when you do jnp.dot(x, weights), JAX sees:x is distributed across the data mesh axis.weights is distributed across the model mesh axis.The Result: JAX "reasons" that the output should be sharded across both (data, model).Key Takeaways for your use case:The Batch Dim is not just replicated: By using spmd_axis_name, you effectively turn vmap into a collective operation that knows about the hardware mesh.Intermediate Efficiency: Because the compiler knows the batch dim is on the data axis, it won't try to gather the whole batch onto one device to perform the dot product. It performs the math locally on each shard.Nested Sharding: You can explicitly shard internal tensors using with_sharding_constraint inside the vmap, and JAX will respect those axes alongside the batch axis.

# The BIG quesiton: How can we have sth similar in NABLA?