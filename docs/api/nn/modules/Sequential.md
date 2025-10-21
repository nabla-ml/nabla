# Sequential

## Signature

```python
nabla.nn.Sequential
```

**Source**: `nabla.nn.modules.container`

Sequential container that applies modules in order.

Modules will be added to the container in the order they are passed
in the constructor. The forward() method automatically chains them.

Parameters
----------
*modules : Module
    Variable number of modules to add sequentially
    
Examples
--------

.. code-block:: python

    >>> from nabla.nn import Sequential, Linear
    >>> model = Sequential(
    ...     Linear(10, 20),
    ...     Linear(20, 10)
    ... )
    >>> output = model(input)  # Automatically applies layers in order

