# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import jax
import jax.numpy as jnp
import numpy as np

import nabla as nb
from tests.unit.common import cleanup_caches, tensor_from_jax, to_jax


def _scalar(t):
    return float(np.asarray(to_jax(t)))


def _build_cnn(x, w_conv, b_conv, w_head, b_head):
    h = nb.conv2d(x, w_conv, bias=b_conv, stride=(1, 1), padding=(1, 1, 1, 1))
    h = nb.relu(h)
    h = nb.avg_pool2d(h, kernel_size=(2, 2), stride=(2, 2), padding=0)
    h = nb.reshape(h, (int(h.shape[0]), int(h.shape[1] * h.shape[2] * h.shape[3])))
    return nb.matmul(h, w_head) + b_head


class TestCNNTrainingSmoke:
    def test_cnn_training_loop_breaks_lazy_graph_each_step(self):
        cleanup_caches()

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (16, 8, 8, 3), dtype=jnp.float32)

        # Synthetic regression target from fixed teacher mapping on input mean.
        x_mean = jnp.mean(x, axis=(1, 2))
        teacher_w = jnp.array(
            [[0.7, -0.4], [-0.3, 0.8], [0.5, 0.2]], dtype=jnp.float32
        )
        teacher_b = jnp.array([0.1, -0.2], dtype=jnp.float32)
        y = x_mean @ teacher_w + teacher_b

        x_nb = tensor_from_jax(x)
        y_nb = tensor_from_jax(y)

        w_conv = tensor_from_jax(
            jax.random.normal(jax.random.PRNGKey(1), (3, 3, 3, 4), dtype=jnp.float32)
            * 0.1
        )
        b_conv = tensor_from_jax(jnp.zeros((4,), dtype=jnp.float32))
        w_head = tensor_from_jax(
            jax.random.normal(jax.random.PRNGKey(2), (4 * 4 * 4, 2), dtype=jnp.float32)
            * 0.1
        )
        b_head = tensor_from_jax(jnp.zeros((2,), dtype=jnp.float32))

        params = [w_conv, b_conv, w_head, b_head]
        for p in params:
            p.is_traced = True

        def loss_fn(model_params, xb, yb):
            w1, b1, w2, b2 = model_params
            pred = _build_cnn(xb, w1, b1, w2, b2)
            diff = pred - yb
            return nb.mean(diff * diff)

        vg = nb.value_and_grad(loss_fn, argnums=0, realize=False)

        lr = 5e-2
        losses = []
        first_w_conv_before = np.asarray(to_jax(params[0]))

        for _ in range(10):
            loss, grads = vg(params, x_nb, y_nb)
            new_params = [p - lr * g for p, g in zip(params, grads, strict=False)]

            # Critical for lazy mode: force execution and break graph each step.
            nb.realize_all(loss, *new_params)

            losses.append(_scalar(loss))
            params = new_params

        assert np.isfinite(losses).all()
        assert losses[-1] < losses[0], (
            f"Expected training loss to decrease, got first={losses[0]:.6f}, last={losses[-1]:.6f}"
        )

        first_w_conv_after = np.asarray(to_jax(params[0]))
        conv_update_norm = np.linalg.norm(first_w_conv_after - first_w_conv_before)
        assert conv_update_norm > 0.0, "Conv weights did not update during training"

    def test_cnn_max_pool_backward_smoke(self):
        cleanup_caches()

        x = tensor_from_jax(
            jax.random.normal(jax.random.PRNGKey(11), (4, 6, 6, 3), dtype=jnp.float32)
        )
        y = tensor_from_jax(
            jax.random.normal(jax.random.PRNGKey(12), (4, 2), dtype=jnp.float32)
        )

        w_conv = tensor_from_jax(
            jax.random.normal(jax.random.PRNGKey(13), (3, 3, 3, 5), dtype=jnp.float32)
            * 0.1
        )
        b_conv = tensor_from_jax(jnp.zeros((5,), dtype=jnp.float32))
        w_head = tensor_from_jax(
            jax.random.normal(jax.random.PRNGKey(14), (3 * 3 * 5, 2), dtype=jnp.float32)
            * 0.1
        )
        b_head = tensor_from_jax(jnp.zeros((2,), dtype=jnp.float32))

        params = [w_conv, b_conv, w_head, b_head]
        for p in params:
            p.is_traced = True

        def loss_fn(model_params, xb, yb):
            w1, b1, w2, b2 = model_params
            h = nb.conv2d(xb, w1, bias=b1, stride=(1, 1), padding=(1, 1, 1, 1))
            h = nb.relu(h)
            h = nb.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2), padding=0)
            h = nb.reshape(h, (int(h.shape[0]), int(h.shape[1] * h.shape[2] * h.shape[3])))
            pred = nb.matmul(h, w2) + b2
            diff = pred - yb
            return nb.mean(diff * diff)

        loss, grads = nb.value_and_grad(loss_fn, argnums=0, realize=False)(params, x, y)
        nb.realize_all(loss, *grads)

        assert np.isfinite(_scalar(loss))
        for idx, g in enumerate(grads):
            g_np = np.asarray(to_jax(g))
            assert np.isfinite(g_np).all(), f"Non-finite gradient in param index {idx}"
