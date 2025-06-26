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
"""Special functions for neural networks."""

from collections.abc import Callable

from ..core.array import Array

# Public API
__all__ = ["softmax", "logsumexp", "where", "cond"]


def logsumexp(arg: Array, axis: int | None = None, keep_dims: bool = False) -> Array:
    """Compute log(sum(exp(x))) in a numerically stable way.

    Args:
        arg: Input array
        axis: Axis along which to compute logsumexp. If None, compute over all elements.
        keep_dims: Whether to keep reduced dimensions

    Returns:
        Array containing logsumexp values
    """
    from .binary import add, sub
    from .reduce import max as array_max
    from .reduce import sum as array_sum
    from .unary import exp, log

    # For numerical stability, subtract the max before exponentiating
    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))

    # Find max along specified axis, keeping dimensions for broadcasting
    x_max = array_max(arg, axes=axis, keep_dims=True)

    # Subtract max and exponentiate
    shifted = sub(arg, x_max)
    exp_shifted = exp(shifted)

    # Sum and take log
    sum_exp = array_sum(exp_shifted, axes=axis, keep_dims=True)
    log_sum_exp = log(sum_exp)

    # Add back the max
    result = add(x_max, log_sum_exp)

    # Remove extra dimensions if not keeping them
    if not keep_dims and axis is not None:
        from .view import squeeze

        axes_to_squeeze = [axis] if isinstance(axis, int) else list(axis)

        for ax in sorted(axes_to_squeeze, reverse=True):
            result = squeeze(result, [ax])  # Pass as list

    return result


def softmax(arg: Array, axis: int = -1) -> Array:
    """Compute softmax function in a numerically stable way.

    Args:
        arg: Input array
        axis: Axis along which to compute softmax

    Returns:
        Array containing softmax probabilities
    """
    from .binary import sub
    from .unary import exp

    # For numerical stability: softmax(x) = exp(x - logsumexp(x))
    log_sum_exp = logsumexp(arg, axis=axis, keep_dims=True)

    # Compute softmax: exp(x - logsumexp(x))
    normalized = sub(arg, log_sum_exp)
    return exp(normalized)


def where(condition: Array, x: Array, y: Array) -> Array:
    """Element-wise selection from x or y based on condition.

    Args:
        condition: Boolean array for selection
        x: Array to select from where condition is True
        y: Array to select from where condition is False

    Returns:
        Array with elements selected from x or y
    """
    from .binary import add, mul
    from .unary import cast, logical_not

    # where(c, x, y) = c * x + (1 - c) * y
    # Convert boolean condition to float for arithmetic
    cond_float = cast(condition, x.dtype)
    inv_cond = cast(logical_not(condition), x.dtype)

    x_part = mul(cond_float, x)
    y_part = mul(inv_cond, y)

    return add(x_part, y_part)


def cond(
    condition: Array, true_fn: Callable, false_fn: Callable, *args, **kwargs
) -> Array:
    """Conditional execution based on a boolean condition.

    Args:
        condition: Boolean array determining which function to execute
        true_fn: Function to execute if condition is True
        false_fn: Function to execute if condition is False
        *args, **kwargs: Arguments passed to the selected function

    Returns:
        Result of the executed function
    """
    from max.dtype import DType

    from .unary import cast

    # Convert condition to boolean if necessary
    bool_condition = cast(condition, DType.bool)

    return where(bool_condition, true_fn(*args, **kwargs), false_fn(*args, **kwargs))
