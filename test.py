# verify_jax.py
import jax
try:
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
except Exception as e:
    print(f"An error occurred: {e}")