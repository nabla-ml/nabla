#!/usr/bin/env python3
"""Test JVP (forward-mode autodiff) functionality."""

import nabla as nb
from nabla.core.trafos import jvp


def test_higher_order_jvp():
    """Test higher-order derivatives using nested JVP calls."""
    # print("=== Testing Higher-Order JVP ===")

    def cubic_fn(inputs):
        x = nb.unsqueeze(nb.unsqueeze(inputs[0], [0]), [0])
        x = nb.squeeze(nb.squeeze(x, [0]), [0])
        return [x * x * x]  # f(x) = x³

    x = nb.array([2.0])
    tangent = nb.array([1.0])

    values, first_order = jvp(cubic_fn, [x], [tangent])

    print(values[0].shape)

    def jacobian_fn(inputs):
        x = inputs[0]
        _, tangents = jvp(cubic_fn, [x], [nb.ones(x.shape)])
        return [tangents[0]]

    _, second_order = jvp(jacobian_fn, [x], [tangent])

    print("Values:", values)
    print("First-order derivative:", first_order)
    print("Second-order derivative:", second_order)


if __name__ == "__main__":
    print("Testing JVP (Forward-Mode Autodiff)")
    test_higher_order_jvp()
