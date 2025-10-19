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

from ..core.tensor import Tensor

# Public API
__all__ = ["softmax", "logsumexp", "where", "cond"]


def logsumexp(arg: Tensor, axis: int | None = None, keep_dims: bool = False) -> Tensor:
    """Computes the log of the sum of exponentials of input elements.

    This function computes `log(sum(exp(x)))` in a numerically stable way by using
    the identity: `logsumexp(x) = max(x) + log(sum(exp(x - max(x))))`. This
    avoids overflow errors that can occur when `exp(x)` is very large.

    Parameters
    ----------
    arg : Tensor
        The input tensor.
    axis : int | None, optional
        The axis or axes along which to compute the `logsumexp`. If None (the
        default), the operation is performed over all elements of the tensor.
    keep_dims : bool, optional
        If True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast
        correctly against the input tensor. Defaults to False.

    Returns
    -------
    Tensor
        An tensor containing the result of the `logsumexp` operation.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([1.0, 2.0, 3.0])
    >>> nb.logsumexp(x)
    Tensor([3.407606], dtype=float32)

    >>> data = nb.tensor([[1, 2, 3], [4, 5, 6]])
    >>> nb.logsumexp(data, axis=1)
    Tensor([3.407606, 6.407606], dtype=float32)
    """
    from .binary import add, sub
    from .reduce import max as tensor_max
    from .reduce import sum as tensor_sum
    from .unary import exp, log

    # For numerical stability, subtract the max before exponentiating
    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))

    # Find max along specified axis, keeping dimensions for broadcasting
    x_max = tensor_max(arg, axes=axis, keep_dims=True)

    # Subtract max and exponentiate
    shifted = sub(arg, x_max)
    exp_shifted = exp(shifted)

    # Sum and take log
    sum_exp = tensor_sum(exp_shifted, axes=axis, keep_dims=True)
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


def softmax(arg: Tensor, axis: int = -1) -> Tensor:
    """Computes the softmax function for an tensor.

    The softmax function transforms a vector of real numbers into a probability
    distribution. Each element in the output is in the range (0, 1), and the
    elements along the specified axis sum to 1. It is calculated in a
    numerically stable way as `exp(x - logsumexp(x))`.

    Parameters
    ----------
    arg : Tensor
        The input tensor.
    axis : int, optional
        The axis along which the softmax computation is performed. The default
        is -1, which is the last axis.

    Returns
    -------
    Tensor
        An tensor of the same shape as the input, containing the softmax
        probabilities.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([1.0, 2.0, 3.0])
    >>> nb.softmax(x)
    Tensor([0.09003057, 0.24472848, 0.66524094], dtype=float32)

    >>> logits = nb.tensor([[1, 2, 3], [1, 1, 1]])
    >>> nb.softmax(logits, axis=1)
    Tensor([[0.09003057, 0.24472848, 0.66524094],
           [0.33333334, 0.33333334, 0.33333334]], dtype=float32)
    """
    from .binary import sub
    from .unary import exp

    # For numerical stability: softmax(x) = exp(x - logsumexp(x))
    log_sum_exp = logsumexp(arg, axis=axis, keep_dims=True)

    # Compute softmax: exp(x - logsumexp(x))
    normalized = sub(arg, log_sum_exp)
    return exp(normalized)


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Selects elements from two tensors based on a condition.

    This function returns an tensor with elements chosen from `x` where the
    corresponding element in `condition` is True, and from `y` otherwise.
    The function supports broadcasting among the three input tensors.

    Parameters
    ----------
    condition : Tensor
        A boolean tensor. Where True, yield `x`, otherwise yield `y`.
    x : Tensor
        The tensor from which to take values when `condition` is True.
    y : Tensor
        The tensor from which to take values when `condition` is False.

    Returns
    -------
    Tensor
        An tensor with elements from `x` and `y`, depending on `condition`.

    Examples
    --------
    >>> import nabla as nb
    >>> condition = nb.tensor([True, False, True])
    >>> x = nb.tensor([1, 2, 3])
    >>> y = nb.tensor([10, 20, 30])
    >>> nb.where(condition, x, y)
    Tensor([1, 20, 3], dtype=int32)

    Broadcasting example:
    >>> nb.where(nb.tensor([True, False]), nb.tensor(5), nb.tensor([10, 20]))
    Tensor([5, 20], dtype=int32)
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
    condition: Tensor, true_fn: Callable, false_fn: Callable, *args, **kwargs
) -> Tensor:
    """Conditionally executes one of two functions.

    If `condition` is True, `true_fn` is called; otherwise, `false_fn` is
    called. This is a control-flow primitive that allows for conditional
    execution within a computational graph. Unlike `nabla.where`, which
    evaluates both branches, `cond` only executes the selected function.

    Parameters
    ----------
    condition : Tensor
        A scalar boolean tensor that determines which function to execute.
    true_fn : Callable
        The function to be called if `condition` is True.
    false_fn : Callable
        The function to be called if `condition` is False.
    *args
        Positional arguments to be passed to the selected function.
    **kwargs
        Keyword arguments to be passed to the selected function.

    Returns
    -------
    Tensor
        The result of calling either `true_fn` or `false_fn`.

    Examples
    --------
    >>> import nabla as nb
    >>> def f(x):
    ...     return x * 2
    ...
    >>> def g(x):
    ...     return x + 10
    ...
    >>> x = nb.tensor(5)
    >>> # Executes g(x) because the condition is False
    >>> nb.cond(nb.tensor(False), f, g, x)
    Tensor([15], dtype=int32)
    """
    from max.dtype import DType

    from .unary import cast

    # Convert condition to boolean if necessary
    bool_condition = cast(condition, DType.bool)

    return where(bool_condition, true_fn(*args, **kwargs), false_fn(*args, **kwargs))