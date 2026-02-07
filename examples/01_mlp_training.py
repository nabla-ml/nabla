# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Basic MLP training with PyTorch-style API.

Trains a 2-layer MLP on synthetic data using Nabla's backward() API.
Demonstrates: model definition, SGD optimizer, training loop with .backward().
"""

import math
import nabla as nb
from max.dtype import DType


class MLP:
    """Simple 2-layer MLP with ReLU activation."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.w1 = nb.normal((input_dim, hidden_dim), std=math.sqrt(2.0 / input_dim))
        self.b1 = nb.zeros((hidden_dim,))
        self.w2 = nb.normal((hidden_dim, output_dim), std=math.sqrt(2.0 / hidden_dim))
        self.b2 = nb.zeros((output_dim,))

        for param in [self.w1, self.b1, self.w2, self.b2]:
            param.requires_grad = True
        self.parameters = [self.w1, self.b1, self.w2, self.b2]

    def forward(self, x):
        h = nb.relu(nb.matmul(x, self.w1) + self.b1)
        return nb.matmul(h, self.w2) + self.b2

    def __call__(self, x):
        return self.forward(x)


class SGD:
    """Simple stochastic gradient descent optimizer."""

    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

    def step(self):
        """Update parameters: param = param - lr * grad."""
        updates = []
        for param in self.parameters:
            if param.grad is not None:
                update = param - self.lr * param.grad
                updates.append((param, update))

        if updates:
            nb.realize_all(*[u for _, u in updates])

        for param, update in updates:
            param._impl._buffers = update._impl._buffers
            param._impl._graph_values = []
            param.real = True
            # Detach from graph for next iteration's caching
            param._impl.output_refs = None
            param._impl.output_index = None
            param._impl.is_traced = False
            param.requires_grad = True


def mse_loss(predictions, targets):
    diff = predictions - targets
    return nb.mean(diff * diff)


def train_mlp():
    """Train a simple MLP on synthetic data."""
    print("=" * 60)
    print("PyTorch-style MLP Training with Nabla")
    print("=" * 60)

    input_dim, hidden_dim, output_dim = 10, 20, 5
    num_epochs, learning_rate = 200, 0.1

    model = MLP(input_dim, hidden_dim, output_dim)
    optimizer = SGD(model.parameters, lr=learning_rate)

    print(f"\nModel: {input_dim} -> {hidden_dim} -> {output_dim}, lr={learning_rate}")

    # Synthetic data: predict sum of squared inputs
    X_train = nb.normal((100, input_dim), mean=0.0, std=1.0)
    y_train = nb.reduce_sum(X_train * X_train, axis=1, keepdims=True)
    y_train = nb.broadcast_to(y_train, (100, output_dim))

    print(f"Dataset: {X_train.shape[0]} samples\n")
    print(f"{'Epoch':<8} {'Loss':<12}")
    print("-" * 30)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = mse_loss(predictions, y_train)
        loss_value = loss.realize()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f"{epoch+1:<8} {loss_value.item():<12.6f}")

    print("-" * 30)
    print("✅ Training completed!")

    # Final evaluation
    optimizer.zero_grad()
    final_pred = model(X_train)
    final_loss = mse_loss(final_pred, y_train).realize().item()
    print(f"\nFinal Loss: {final_loss:.6f}")
    print(f"Sample predictions: {final_pred.realize().numpy()[:3, 0]}")
    print(f"Sample targets:     {y_train.numpy()[:3, 0]}")


if __name__ == "__main__":
    train_mlp()
