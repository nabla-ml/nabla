import jax
import jax.numpy as jnp
import time

def mlp(args):
    h1 = jax.nn.relu(args[0] @ args[1] + args[2])
    h2 = jax.nn.relu(h1 @ args[3] + args[4])
    h3 = jax.nn.relu(h2 @ args[5] + args[6])
    h4 = jax.nn.relu(h3 @ args[7] + args[8])
    h5 = jax.nn.relu(h4 @ args[9] + args[10])
    h6 = jax.nn.relu(h5 @ args[11] + args[12])
    h7 = jax.nn.relu(h6 @ args[13] + args[14])
    h8 = jax.nn.relu(h7 @ args[15] + args[16])
    output = h8 @ args[17] + args[18]
    return output

def test_jit():
    print("Running JAX JIT benchmark...")
    key = jax.random.PRNGKey(0)
    
    # Initialize params
    params_shapes = [
        (512, 2048), (2048,),
        (2048, 4096), (4096,),
        (4096, 4096), (4096,),
        (4096, 4096), (4096,),
        (4096, 2048), (2048,),
        (2048, 1024), (1024,),
        (1024, 512), (512,),
        (512, 256), (256,),
        (256, 10), (10,)
    ]
    
    params = []
    for i, shape in enumerate(params_shapes):
        key, subkey = jax.random.split(key)
        # Mojo uses 0.0 mean, 0.02 and 0.01 stds
        std = 0.02 if i % 2 == 0 else 0.01
        params.append(jax.random.normal(subkey, shape) * std)
        
    mlp_jit = jax.jit(mlp)
    
    for it in range(10):
        t_iter_start = time.perf_counter()
        
        key, subkey = jax.random.split(key)
        input_data = jax.random.normal(subkey, (4, 512))
        
        args = [input_data] + params
        output = mlp_jit(args)
        
        # Ensure the computation is finished
        output.block_until_ready()
        
        t_iter_end = time.perf_counter()
        iter_time_ms = (t_iter_end - t_iter_start) * 1000
        
        if it % 1 == 0:
             print(f"Iteration {it} | Time: {iter_time_ms:.4f} ms")
             if it == 0:
                 print("First iteration output sample:", output[0, :5])

if __name__ == "__main__":
    test_jit()
