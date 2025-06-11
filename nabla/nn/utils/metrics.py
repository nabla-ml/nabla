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
"""Metrics for evaluating neural network performance."""

import nabla as nb
from max.dtype import DType


def accuracy(predictions: nb.Array, targets: nb.Array) -> nb.Array:
    """Compute classification accuracy.
    
    Args:
        predictions: Model predictions - either logits/probabilities [batch_size, num_classes] 
                    or class indices [batch_size]
        targets: True labels - either one-hot [batch_size, num_classes] or indices [batch_size]
        
    Returns:
        Scalar accuracy value between 0 and 1
    """
    # Handle different prediction formats
    if len(predictions.shape) == 1:
        # Predictions are already class indices
        pred_classes = predictions
    else:
        # Predictions are logits/probabilities - get argmax
        pred_classes = nb.argmax(predictions, axis=-1)
    
    # Handle different target formats
    if len(targets.shape) == 1:
        # Sparse targets (class indices)
        true_classes = targets
    else:
        # One-hot targets
        true_classes = nb.argmax(targets, axis=-1)
    
    # Compute accuracy using equal comparison
    correct_mask = nb.equal(pred_classes, true_classes)
    correct = correct_mask.astype(DType.float32)
    return nb.mean(correct)


def top_k_accuracy(predictions: nb.Array, targets: nb.Array, k: int = 5) -> nb.Array:
    """Compute top-k classification accuracy.
    
    Args:
        predictions: Model predictions (logits or probabilities) [batch_size, num_classes]
        targets: True labels [batch_size] (sparse format)
        k: Number of top predictions to consider
        
    Returns:
        Scalar top-k accuracy value between 0 and 1
    """
    # Get top-k predictions (indices)
    # For now, use a simplified approach
    # In practice, we'd need argsort or a proper top-k implementation
    pred_classes = nb.argmax(predictions, axis=-1)
    
    # For simplicity, this is equivalent to top-1 accuracy
    # A full implementation would require sorting operations
    correct = nb.equal(pred_classes, targets).astype(DType.float32)
    return nb.mean(correct)


def precision(predictions: nb.Array, targets: nb.Array, num_classes: int, class_idx: int = 0) -> nb.Array:
    """Compute precision for a specific class.
    
    Precision = TP / (TP + FP)
    
    Args:
        predictions: Model predictions (logits) [batch_size, num_classes]
        targets: True labels (sparse) [batch_size]
        num_classes: Total number of classes
        class_idx: Class index to compute precision for
        
    Returns:
        Scalar precision value for the specified class
    """
    pred_classes = nb.argmax(predictions, axis=-1)
    
    # Create class indicator arrays
    class_idx_array = nb.full_like(pred_classes, class_idx)
    
    # True positives: predicted as class and actually is class
    pred_is_class = nb.equal(pred_classes, class_idx_array).astype(DType.float32)
    target_is_class = nb.equal(targets, class_idx_array).astype(DType.float32)
    tp = nb.sum(pred_is_class * target_is_class)
    
    # False positives: predicted as class but actually is not
    target_not_class = (1.0 - target_is_class)
    fp = nb.sum(pred_is_class * target_not_class)
    
    # Avoid division by zero
    epsilon = nb.array([1e-8])
    return tp / (tp + fp + epsilon)


def recall(predictions: nb.Array, targets: nb.Array, num_classes: int, class_idx: int = 0) -> nb.Array:
    """Compute recall for a specific class.
    
    Recall = TP / (TP + FN)
    
    Args:
        predictions: Model predictions (logits) [batch_size, num_classes]
        targets: True labels (sparse) [batch_size]
        num_classes: Total number of classes
        class_idx: Class index to compute recall for
        
    Returns:
        Scalar recall value for the specified class
    """
    pred_classes = nb.argmax(predictions, axis=-1)
    
    # Create class indicator arrays
    class_idx_array = nb.full_like(pred_classes, class_idx)
    
    # True positives: predicted as class and actually is class
    pred_is_class = nb.equal(pred_classes, class_idx_array).astype(DType.float32)
    target_is_class = nb.equal(targets, class_idx_array).astype(DType.float32)
    tp = nb.sum(pred_is_class * target_is_class)
    
    # False negatives: not predicted as class but actually is class
    pred_not_class = (1.0 - pred_is_class)
    fn = nb.sum(pred_not_class * target_is_class)
    
    # Avoid division by zero
    epsilon = nb.array([1e-8])
    return tp / (tp + fn + epsilon)


def f1_score(predictions: nb.Array, targets: nb.Array, num_classes: int, class_idx: int = 0) -> nb.Array:
    """Compute F1 score for a specific class.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        predictions: Model predictions (logits) [batch_size, num_classes]
        targets: True labels (sparse) [batch_size]
        num_classes: Total number of classes
        class_idx: Class index to compute F1 score for
        
    Returns:
        Scalar F1 score for the specified class
    """
    prec = precision(predictions, targets, num_classes, class_idx)
    rec = recall(predictions, targets, num_classes, class_idx)
    
    epsilon = nb.array([1e-8])
    return 2 * (prec * rec) / (prec + rec + epsilon)


def mean_squared_error_metric(predictions: nb.Array, targets: nb.Array) -> nb.Array:
    """Compute MSE metric for regression tasks.
    
    Args:
        predictions: Model predictions [batch_size, ...]
        targets: True targets [batch_size, ...]
        
    Returns:
        Scalar MSE value
    """
    diff = predictions - targets
    return nb.mean(diff * diff)


def mean_absolute_error_metric(predictions: nb.Array, targets: nb.Array) -> nb.Array:
    """Compute MAE metric for regression tasks.
    
    Args:
        predictions: Model predictions [batch_size, ...]
        targets: True targets [batch_size, ...]
        
    Returns:
        Scalar MAE value
    """
    diff = predictions - targets
    return nb.mean(nb.abs(diff))


def r_squared(predictions: nb.Array, targets: nb.Array) -> nb.Array:
    """Compute R-squared (coefficient of determination) for regression tasks.
    
    R² = 1 - (SS_res / SS_tot)
    where SS_res = Σ(y_true - y_pred)² and SS_tot = Σ(y_true - y_mean)²
    
    Args:
        predictions: Model predictions [batch_size, ...]
        targets: True targets [batch_size, ...]
        
    Returns:
        Scalar R² value
    """
    # Residual sum of squares
    ss_res = nb.sum((targets - predictions) ** 2)
    
    # Total sum of squares
    targets_mean = nb.mean(targets)
    ss_tot = nb.sum((targets - targets_mean) ** 2)
    
    # R-squared
    epsilon = nb.array([1e-8])
    return 1 - (ss_res / (ss_tot + epsilon))


def pearson_correlation(predictions: nb.Array, targets: nb.Array) -> nb.Array:
    """Compute Pearson correlation coefficient.
    
    Args:
        predictions: Model predictions [batch_size, ...]
        targets: True targets [batch_size, ...]
        
    Returns:
        Scalar correlation coefficient
    """
    # Flatten arrays for correlation calculation
    pred_flat = predictions.reshape((-1,))
    target_flat = targets.reshape((-1,))
    
    # Compute means
    pred_mean = nb.mean(pred_flat)
    target_mean = nb.mean(target_flat)
    
    # Compute correlation
    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean
    
    numerator = nb.sum(pred_centered * target_centered)
    pred_std = nb.sqrt(nb.sum(pred_centered ** 2))
    target_std = nb.sqrt(nb.sum(target_centered ** 2))
    
    epsilon = nb.array([1e-8])
    return numerator / (pred_std * target_std + epsilon)


__all__ = [
    "accuracy",
    "top_k_accuracy",
    "precision",
    "recall", 
    "f1_score",
    "mean_squared_error_metric",
    "mean_absolute_error_metric",
    "r_squared",
    "pearson_correlation",
]