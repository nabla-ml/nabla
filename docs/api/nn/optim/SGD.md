# SGD

## Signature

```python
nabla.nn.SGD
```

**Source**: `nabla.nn.optim.optimizer`

Stochastic Gradient Descent optimizer.

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

.. code-block:: python

    >>> from nabla.nn import SGD
    >>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    >>> for data, target in dataloader:
    ...     optimizer.zero_grad()
    ...     output = model(data)
    ...     loss = criterion(output, target)
    ...     loss.backward()
    ...     optimizer.step()

