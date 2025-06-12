#!/usr/bin/env python3
"""Debug SJIT NaN issue by comparing JIT vs SJIT step by step."""

import numpy as np

import nabla as nb


# Simple AdamW function that matches the style used in the training
def adamw_step_simple(
    param,
    grad,
    m,
    v,
    step,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.01,
):
    """Simplified AdamW step for debugging."""
    # Update moments
    new_m = beta1 * m + (1.0 - beta1) * grad
    new_v = beta2 * v + (1.0 - beta2) * (grad * grad)

    # JAX-style fused parameter update
    new_param = param * (1.0 - weight_decay * lr) - lr * (
        new_m / (1.0 - beta1**step)
    ) / (((new_v / (1.0 - beta2**step)) ** 0.5) + eps)

    return new_param, new_m, new_v


@nb.jit
def adamw_jit(param, grad, m, v, step, lr=0.001):
    """JIT version."""
    return adamw_step_simple(param, grad, m, v, step, lr)


@nb.sjit
def adamw_sjit(param, grad, m, v, step, lr=0.001):
    """SJIT version."""
    return adamw_step_simple(param, grad, m, v, step, lr)


def debug_nan_issue():
    """Debug the NaN issue step by step."""
    print("=== Debugging SJIT NaN Issue ===")

    # Initialize test data
    param_np = np.array([[1.0, 2.0]], dtype=np.float32)
    grad_np = np.array([[0.1, 0.2]], dtype=np.float32)
    m_np = np.zeros_like(param_np)
    v_np = np.zeros_like(param_np)

    param = nb.Array.from_numpy(param_np)
    grad = nb.Array.from_numpy(grad_np)
    m = nb.Array.from_numpy(m_np)
    v = nb.Array.from_numpy(v_np)

    print(f"Initial param: {param.to_numpy()}")
    print(f"Initial grad: {grad.to_numpy()}")
    print(f"Initial m: {m.to_numpy()}")
    print(f"Initial v: {v.to_numpy()}")

    # Test JIT version with more iterations (like training loop)
    print("\n=== JIT Version (100 steps) ===")
    param_jit, m_jit, v_jit = param, m, v
    for step in range(1, 101):
        param_jit, m_jit, v_jit = adamw_jit(param_jit, grad, m_jit, v_jit, step)

        # Check for NaN values
        if np.any(np.isnan(param_jit.to_numpy())):
            print(f"  ❌ JIT NaN detected in param at step {step}!")
            print(f"    param: {param_jit.to_numpy()}")
            print(f"    m: {m_jit.to_numpy()}")
            print(f"    v: {v_jit.to_numpy()}")
            break
        if np.any(np.isnan(m_jit.to_numpy())):
            print(f"  ❌ JIT NaN detected in m at step {step}!")
            break
        if np.any(np.isnan(v_jit.to_numpy())):
            print(f"  ❌ JIT NaN detected in v at step {step}!")
            break

        # Print every 10 steps
        if step % 10 == 0:
            print(
                f"Step {step}: param={param_jit.to_numpy().flatten()[:2]}, m={m_jit.to_numpy().flatten()[:2]}"
            )

    if step == 100:
        print("✅ JIT completed 100 steps without NaN")
        print(f"Final param: {param_jit.to_numpy()}")

    # Test SJIT version with more iterations
    print("\n=== SJIT Version (100 steps) ===")
    param_sjit, m_sjit, v_sjit = param, m, v
    for step in range(1, 101):
        param_sjit, m_sjit, v_sjit = adamw_sjit(param_sjit, grad, m_sjit, v_sjit, step)

        # Check for NaN values
        if np.any(np.isnan(param_sjit.to_numpy())):
            print(f"  ❌ SJIT NaN detected in param at step {step}!")
            print(f"    param: {param_sjit.to_numpy()}")
            print(f"    m: {m_sjit.to_numpy()}")
            print(f"    v: {v_sjit.to_numpy()}")
            break
        if np.any(np.isnan(m_sjit.to_numpy())):
            print(f"  ❌ SJIT NaN detected in m at step {step}!")
            break
        if np.any(np.isnan(v_sjit.to_numpy())):
            print(f"  ❌ SJIT NaN detected in v at step {step}!")
            break

        # Print every 10 steps
        if step % 10 == 0:
            print(
                f"Step {step}: param={param_sjit.to_numpy().flatten()[:2]}, m={m_sjit.to_numpy().flatten()[:2]}"
            )

    if step == 100:
        print("✅ SJIT completed 100 steps without NaN")
        print(f"Final param: {param_sjit.to_numpy()}")


def debug_multiple_calls():
    """Test multiple independent calls to see if caching is the issue."""
    print("\n=== Testing Multiple Independent Calls ===")

    # Create fresh data for each call
    def create_fresh_data():
        param_np = np.array([[1.0, 2.0]], dtype=np.float32)
        grad_np = np.array([[0.1, 0.2]], dtype=np.float32)
        m_np = np.array([[0.01, 0.02]], dtype=np.float32)  # Non-zero momentum
        v_np = np.array([[0.001, 0.002]], dtype=np.float32)  # Non-zero velocity

        return (
            nb.Array.from_numpy(param_np),
            nb.Array.from_numpy(grad_np),
            nb.Array.from_numpy(m_np),
            nb.Array.from_numpy(v_np),
        )

    # Test SJIT with different step values in separate calls
    print("SJIT separate calls with different steps:")
    for step_val in [1, 2, 10, 50, 100]:
        param, grad, m, v = create_fresh_data()
        try:
            param_new, m_new, v_new = adamw_sjit(param, grad, m, v, step_val)
            print(f"  Step {step_val}: param={param_new.to_numpy().flatten()[:2]}")

            # Check for NaN
            if np.any(np.isnan(param_new.to_numpy())):
                print(f"  ❌ SJIT NaN at step {step_val}!")
                print(f"    step value used: {step_val}")
                print(f"    param: {param_new.to_numpy()}")
                break
        except Exception as e:
            print(f"  ❌ SJIT Error at step {step_val}: {e}")
            break


if __name__ == "__main__":
    debug_nan_issue()
