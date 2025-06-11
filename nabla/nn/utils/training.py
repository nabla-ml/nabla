# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Training utilities for neural networks."""

import nabla as nb


def create_dataset(
    batch_size: int, input_dim: int, seed: int | None = None
) -> tuple[nb.Array, nb.Array]:
    """Create a simple random dataset for testing.

    Args:
        batch_size: Number of samples
        input_dim: Input dimension
        seed: Random seed for reproducibility

    Returns:
        Tuple of (inputs, targets)
    """
    if seed is not None:
        import numpy as np

        np.random.seed(seed)

    x = nb.rand((batch_size, input_dim), lower=-1.0, upper=1.0, dtype=nb.DType.float32)
    # Simple target: sum of inputs
    targets = nb.sum(x, axis=1, keepdims=True)
    return x, targets


def create_sin_dataset(
    batch_size: int = 256, sin_periods: int = 8
) -> tuple[nb.Array, nb.Array]:
    """Create the complex 8-period sin dataset from mlp_train_jit.py.

    Args:
        batch_size: Number of samples to generate
        sin_periods: Number of sin periods in [0, 1] interval

    Returns:
        Tuple of (x, targets) where targets are sin function values
    """
    import numpy as np

    x = nb.rand((batch_size, 1), lower=0.0, upper=1.0, dtype=nb.DType.float32)
    targets = nb.sin(sin_periods * 2.0 * np.pi * x) / 2.0 + 0.5
    return x, targets


def compute_accuracy(
    predictions: nb.Array, targets: nb.Array, threshold: float = 0.5
) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Model predictions
        targets: True labels
        threshold: Classification threshold

    Returns:
        Accuracy as a float between 0 and 1
    """
    pred_labels = predictions > threshold
    target_labels = targets > threshold
    correct = pred_labels == target_labels
    accuracy = nb.mean(correct.astype(nb.DType.float32))
    return accuracy.to_numpy().item()


def compute_correlation(predictions: nb.Array, targets: nb.Array) -> float:
    """Compute Pearson correlation coefficient.

    Args:
        predictions: Model predictions
        targets: True values

    Returns:
        Correlation coefficient as a float
    """
    import numpy as np

    pred_np = predictions.to_numpy().flatten()
    target_np = targets.to_numpy().flatten()

    correlation = np.corrcoef(pred_np, target_np)[0, 1]
    return correlation
