# Optimizer

## Signature

```python
nabla.nn.Optimizer
```

**Source**: `nabla.nn.optim.optimizer`

Base class for all optimizers.

Handles parameter updates after gradients are computed via backward().
All optimizer implementations should inherit from this class.

Parameters
----------
params : Iterator[Tensor] or list[Tensor]
    Iterator or list of parameters to optimize
    
Examples
--------

.. code-block:: python

    >>> from nabla.nn import SGD
    >>> optimizer = SGD(model.parameters(), lr=0.01)
    >>> loss.backward()
    >>> optimizer.step()  # Updates parameters
    >>> optimizer.zero_grad()  # Clears gradients

