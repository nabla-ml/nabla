#!/usr/bin/env python3
"""Test to isolate the difference between jit and sjit"""

import nabla as nb


# Simple test function
def simple_function(x, y):
    """Simple function for testing"""
    return x * y + nb.sin(x)


# Test with value_and_grad
def test_function_for_grad(inputs):
    """Function that takes a list of inputs for value_and_grad"""
    x, y = inputs
    return x * y + nb.sin(x)


def test_adamw_step(
    param, grad, m, v, step, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8
):
    """Simple AdamW step for one parameter"""
    new_m = beta1 * m + (1.0 - beta1) * grad
    new_v = beta2 * v + (1.0 - beta2) * (grad * grad)

    # Bias correction
    m_hat = new_m / (1.0 - beta1**step)
    v_hat = new_v / (1.0 - beta2**step)

    new_param = param - lr * m_hat / (nb.sqrt(v_hat) + eps)

    return new_param, new_m, new_v


def test_gradient_computation():
    """Test basic gradient computation with jit vs sjit"""
    print("=== Testing Gradient Computation ===")

    # Create test inputs
    x = nb.array([2.0])
    y = nb.array([3.0])

    # Test regular function
    result_regular = simple_function(x, y)
    print(f"Regular function result: {result_regular.to_numpy()}")

    # Test jit function
    jit_func = nb.jit(simple_function)
    result_jit = jit_func(x, y)
    print(f"JIT function result: {result_jit.to_numpy()}")

    # Test sjit function
    sjit_func = nb.sjit(simple_function)
    result_sjit = sjit_func(x, y)
    print(f"SJIT function result: {result_sjit.to_numpy()}")

    # Test value_and_grad with jit
    inputs = [x, y]
    loss_jit, grads_jit = nb.value_and_grad(test_function_for_grad, argnums=[0, 1])(
        inputs
    )
    print(
        f"JIT value_and_grad - Loss: {loss_jit.to_numpy()}, Grads: {[g.to_numpy() for g in grads_jit]}"
    )

    # Test value_and_grad with sjit
    sjit_grad_func = nb.sjit(
        lambda inputs: nb.value_and_grad(test_function_for_grad, argnums=[0, 1])(inputs)
    )
    loss_sjit, grads_sjit = sjit_grad_func(inputs)
    print(
        f"SJIT value_and_grad - Loss: {loss_sjit.to_numpy()}, Grads: {[g.to_numpy() for g in grads_sjit]}"
    )


@nb.jit
def train_step_jit(x, y, param, m, v, step, lr):
    """JIT training step"""
    inputs = [x, y]
    loss, grads = nb.value_and_grad(test_function_for_grad, argnums=[0])(inputs)

    # Use the gradient w.r.t. x as if it were a parameter gradient
    grad = grads[0]

    new_param, new_m, new_v = test_adamw_step(param, grad, m, v, step, lr)

    return new_param, new_m, new_v, loss


@nb.sjit
def train_step_sjit(x, y, param, m, v, step, lr):
    """SJIT training step"""
    inputs = [x, y]
    loss, grads = nb.value_and_grad(test_function_for_grad, argnums=[0])(inputs)

    # Use the gradient w.r.t. x as if it were a parameter gradient
    grad = grads[0]

    new_param, new_m, new_v = test_adamw_step(param, grad, m, v, step, lr)

    return new_param, new_m, new_v, loss


def test_training_steps():
    """Test training steps with both jit and sjit"""
    print("\n=== Testing Training Steps ===")

    # Initialize
    x = nb.array([2.0])
    y = nb.array([3.0])
    param = nb.array([1.0])
    m = nb.array([0.0])
    v = nb.array([0.0])

    print(f"Initial param: {param.to_numpy()}")

    # Test JIT training step
    print("\nTesting JIT training step:")
    for step in range(1, 6):
        param_jit, m_jit, v_jit, loss_jit = train_step_jit(
            x, y, param, m, v, step, 0.01
        )
        print(f"Step {step}: param={param_jit.to_numpy()}, loss={loss_jit.to_numpy()}")
        param, m, v = param_jit, m_jit, v_jit

    # Reset for SJIT test
    param = nb.array([1.0])
    m = nb.array([0.0])
    v = nb.array([0.0])

    # Test SJIT training step
    print("\nTesting SJIT training step:")
    for step in range(1, 6):
        param_sjit, m_sjit, v_sjit, loss_sjit = train_step_sjit(
            x, y, param, m, v, step, 0.01
        )
        print(
            f"Step {step}: param={param_sjit.to_numpy()}, loss={loss_sjit.to_numpy()}"
        )
        param, m, v = param_sjit, m_sjit, v_sjit


if __name__ == "__main__":
    test_gradient_computation()
    test_training_steps()
