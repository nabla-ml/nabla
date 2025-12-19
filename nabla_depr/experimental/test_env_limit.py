import jax
import jax.numpy as jnp
from PPO.cartpole_swingup import PureJaxCartPoleSwingUp

def test_env():
    env = PureJaxCartPoleSwingUp()
    key = jax.random.PRNGKey(0)
    
    # Reset
    state = env.reset(key)
    # Force state to be stable: x=0, theta=pi (hanging down), velocities=0
    state = jnp.array([0.0, 0.0, jnp.pi, 0.0])
    
    print("Starting test with stable state (hanging down)...")
    
    for i in range(1000):
        # Action = 0
        action = jnp.array([0.0])
        state, reward, done = env.step(state, action)
        
        if done:
            print(f"Terminated at step {i+1}")
            return
            
    print("Survived 1000 steps without termination.")

if __name__ == "__main__":
    test_env()
