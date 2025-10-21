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

"""Optimizers for imperative neural network training (PyTorch-like API).

Provides optimizer classes that handle parameter updates after gradients
are computed via backward().

Available optimizers:
- SGD: Stochastic Gradient Descent with optional momentum and weight decay
- Adam: Adaptive Moment Estimation

Examples
--------
>>> from nabla.nn import Module, SGD
>>> model = MyModel()
>>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
>>> for epoch in range(100):
...     pred = model(x)
...     loss = loss_fn(pred, y)
...     loss.backward()
...     optimizer.step()
...     optimizer.zero_grad()
"""

from __future__ import annotations

from typing import Iterator

import nabla as nb
from ..core.tensor import Tensor

__all__ = ["Optimizer", "SGD", "Adam"]


class Optimizer:
    """Base class for all optimizers.
    
    Handles parameter updates after gradients are computed via backward().
    All optimizer implementations should inherit from this class.
    
    Parameters
    ----------
    params : Iterator[Tensor] or list[Tensor]
        Iterator or list of parameters to optimize
        
    Examples
    --------
    >>> from nabla.nn import SGD
    >>> optimizer = SGD(model.parameters(), lr=0.01)
    >>> loss.backward()
    >>> optimizer.step()  # Updates parameters
    >>> optimizer.zero_grad()  # Clears gradients
    """
    
    def __init__(self, params: Iterator[Tensor] | list[Tensor]):
        """
        Initialize optimizer with parameters to optimize.
        
        Args:
            params: Iterator or list of parameters (Tensors) to optimize
        """
        # Convert to list to allow multiple iterations
        self.params = list(params)
    
    def zero_grad(self) -> None:
        """
        Zero gradients for all parameters.
        
        Should be called after each optimization step to clear gradients
        before the next backward pass.
        """
        for param in self.params:
            param.grad = None
    
    def step(self) -> None:
        """
        Perform a single optimization step (parameter update).
        
        Must be implemented by subclasses.
        
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement step() method"
        )
    
    def _update_param_inplace(self, param: Tensor, new_value: Tensor) -> None:
        """
        Update parameter in-place without breaking the computation graph.
        
        This updates the underlying tensor data while preserving the requires_grad
        flag and clearing the gradient.
        
        Args:
            param: Parameter tensor to update
            new_value: New value for the parameter
        """
        # Update the underlying implementation
        param._impl = new_value._impl
        # Keep requires_grad flag
        param.requires_grad = True
        # Clear the gradient
        param.grad = None


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.
    
    Implements SGD with optional momentum and weight decay (L2 regularization).
    
    Parameters
    ----------
    params : Iterator[Tensor] or list[Tensor]
        Parameters to optimize
    lr : float, optional
        Learning rate (default: 0.01)
    momentum : float, optional
        Momentum factor (default: 0.0, no momentum)
    weight_decay : float, optional
        Weight decay (L2 penalty) (default: 0.0, no decay)
        
    Examples
    --------
    >>> from nabla.nn import SGD
    >>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    >>> for data, target in dataloader:
    ...     optimizer.zero_grad()
    ...     output = model(data)
    ...     loss = criterion(output, target)
    ...     loss.backward()
    ...     optimizer.step()
    """
    
    def __init__(
        self, 
        params: Iterator[Tensor] | list[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Initialize momentum buffers if needed
        self.velocity = [None] * len(self.params) if momentum > 0 else None
    
    def step(self) -> None:
        """
        Perform a single SGD update step.
        
        Updates parameters using: param = param - lr * (grad + weight_decay * param)
        With momentum: velocity = momentum * velocity + grad
                      param = param - lr * velocity
        """
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Add weight decay if specified (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param
            
            # Apply momentum if specified
            if self.momentum > 0:
                if self.velocity[i] is None:
                    self.velocity[i] = grad
                else:
                    self.velocity[i] = self.momentum * self.velocity[i] + grad
                grad = self.velocity[i]
            
            # Update parameter: param = param - lr * grad
            new_param = param - self.lr * grad
            self._update_param_inplace(param, new_param)
    
    def __repr__(self) -> str:
        return (f"SGD(lr={self.lr}, momentum={self.momentum}, "
                f"weight_decay={self.weight_decay})")


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).
    
    Implements Adam algorithm with bias correction. Maintains moving averages
    of gradients and their squares for adaptive learning rates.
    
    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 0.001)
        betas: Coefficients for computing running averages of gradient
               and its square (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0.0, no decay)
        
    Example:
        >>> optimizer = Adam(model.parameters(), lr=0.001)
        >>> 
        >>> for data, target in dataloader:
        ...     optimizer.zero_grad()
        ...     output = model(data)
        ...     loss = criterion(output, target)
        ...     loss.backward()
        ...     optimizer.step()
    
    References:
        Adam: A Method for Stochastic Optimization
        Kingma & Ba, ICLR 2015
        https://arxiv.org/abs/1412.6980
    """
    
    def __init__(
        self,
        params: Iterator[Tensor] | list[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moment estimates
        self.m = [None] * len(self.params)  # First moment (mean)
        self.v = [None] * len(self.params)  # Second moment (variance)
        self.t = 0  # Time step
    
    def step(self) -> None:
        """
        Perform a single Adam optimization step.
        
        Updates parameters using adaptive learning rates based on first
        and second moment estimates of gradients.
        """
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Add weight decay if specified
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param
            
            # Initialize moment estimates on first step
            if self.m[i] is None:
                self.m[i] = grad * 0.0  # Initialize with zeros of same shape
                self.v[i] = grad * 0.0
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad * grad)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            
            # Update parameter
            new_param = param - self.lr * m_hat / (nb.sqrt(v_hat) + self.eps)
            self._update_param_inplace(param, new_param)
    
    def __repr__(self) -> str:
        return (f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
                f"eps={self.eps}, weight_decay={self.weight_decay})")
