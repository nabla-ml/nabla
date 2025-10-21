# ModuleList

## Signature

```python
nabla.nn.ModuleList
```

**Source**: `nabla.nn.modules.container`

Container that holds modules in a list.

Like PyTorch's nn.ModuleList - modules are properly registered and can
be indexed, iterated, and appended to. The modules are registered as
submodules so their parameters are collected.

Note: ModuleList does not define forward() - it's a container that you
use within your own modules.

Parameters
----------
*modules : Module
    Variable number of modules to add to the list
    
Examples
--------

.. code-block:: python

    >>> from nabla.nn import ModuleList, Linear
    >>> layers = ModuleList(
    ...     Linear(10, 20),
    ...     Linear(20, 10)
    ... )
    >>> for layer in layers:
    ...     x = layer(x)

