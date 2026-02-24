# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

import nabla as nb

try:
    from tests.unit.common import cleanup_caches, tensor_from_jax, to_jax
except ModuleNotFoundError:
    unit_dir = Path(__file__).resolve().parent
    if str(unit_dir) not in sys.path:
        sys.path.insert(0, str(unit_dir))
    from common import cleanup_caches, tensor_from_jax, to_jax


def _jax_model(x, params):
    w1, b1, w2, b2, w_head, b_head = params

    y = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=w1,
        window_strides=(1, 1),
        padding=((1, 1), (1, 1)),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    ) + b1
    y = jax.nn.relu(y)

    y = jax.lax.reduce_window(
        y,
        0.0,
        jax.lax.add,
        window_dimensions=(1, 2, 2, 1),
        window_strides=(1, 2, 2, 1),
        padding=((0, 0), (0, 0), (0, 0), (0, 0)),
    ) / 4.0

    y = jax.lax.conv_general_dilated(
        lhs=y,
        rhs=w2,
        window_strides=(1, 1),
        padding=((1, 1), (1, 1)),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    ) + b2
    y = jax.nn.relu(y)

    y = jax.lax.reduce_window(
        y,
        -jnp.inf,
        jax.lax.max,
        window_dimensions=(1, 2, 2, 1),
        window_strides=(1, 2, 2, 1),
        padding=((0, 0), (0, 0), (0, 0), (0, 0)),
    )

    y = y.reshape((y.shape[0], -1))
    return y @ w_head + b_head


def _nb_model(x, params):
    w1, b1, w2, b2, w_head, b_head = params

    y = nb.conv2d(x, w1, bias=b1, stride=(1, 1), padding=(1, 1, 1, 1))
    y = nb.relu(y)
    y = nb.avg_pool2d(y, kernel_size=(2, 2), stride=(2, 2), padding=0)

    y = nb.conv2d(y, w2, bias=b2, stride=(1, 1), padding=(1, 1, 1, 1))
    y = nb.relu(y)
    y = nb.max_pool2d(y, kernel_size=(2, 2), stride=(2, 2), padding=0)

    y = nb.reshape(y, (int(y.shape[0]), int(y.shape[1] * y.shape[2] * y.shape[3])))
    return nb.matmul(y, w_head) + b_head


class TestCNNToyTraining100:
    def test_full_toy_training_100_steps_lazy_break_each_iter(self):
        cleanup_caches()

        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 16)

        n = 64
        x = jax.random.normal(keys[0], (n, 8, 8, 1), dtype=jnp.float32)

        teacher = [
            jax.random.normal(keys[1], (3, 3, 1, 4), dtype=jnp.float32) * 0.35,
            jax.random.normal(keys[2], (4,), dtype=jnp.float32) * 0.1,
            jax.random.normal(keys[3], (3, 3, 4, 6), dtype=jnp.float32) * 0.30,
            jax.random.normal(keys[4], (6,), dtype=jnp.float32) * 0.1,
            jax.random.normal(keys[5], (24, 1), dtype=jnp.float32) * 0.25,
            jax.random.normal(keys[6], (1,), dtype=jnp.float32) * 0.1,
        ]

        y_clean = _jax_model(x, teacher)
        y = y_clean + 0.01 * jax.random.normal(keys[7], y_clean.shape, dtype=jnp.float32)

        student_jax = [
            teacher[0] + 0.20 * jax.random.normal(keys[8], teacher[0].shape),
            teacher[1] + 0.20 * jax.random.normal(keys[9], teacher[1].shape),
            teacher[2] + 0.20 * jax.random.normal(keys[10], teacher[2].shape),
            teacher[3] + 0.20 * jax.random.normal(keys[11], teacher[3].shape),
            teacher[4] + 0.20 * jax.random.normal(keys[12], teacher[4].shape),
            teacher[5] + 0.20 * jax.random.normal(keys[13], teacher[5].shape),
        ]

        x_nb = tensor_from_jax(x)
        y_nb = tensor_from_jax(y)
        params = [tensor_from_jax(p) for p in student_jax]
        for p in params:
            p.is_traced = True

        def loss_fn(model_params, xb, yb):
            pred = _nb_model(xb, model_params)
            diff = pred - yb
            return nb.mean(diff * diff)

        vg = nb.value_and_grad(loss_fn, argnums=0, realize=False)

        lr = 5e-2
        losses = []

        for _ in range(100):
            loss, grads = vg(params, x_nb, y_nb)
            new_params = [p - lr * g for p, g in zip(params, grads, strict=False)]

            # Mandatory in lazy mode: realize and break trace between iterations.
            nb.realize_all(loss, *new_params)

            losses.append(float(np.asarray(to_jax(loss))))
            params = new_params

        assert np.isfinite(losses).all(), "Training produced non-finite losses"
        assert losses[-1] < losses[0] * 0.25, (
            f"Expected strong convergence in 100 steps, got first={losses[0]:.6f}, last={losses[-1]:.6f}"
        )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([str(Path(__file__)), "-q"]))
