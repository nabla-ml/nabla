# ModuleDict

## Signature

```python
nabla.nn.ModuleDict
```

**Source**: `nabla.nn.modules.container`

Container that holds modules in a dictionary.

Like PyTorch's nn.ModuleDict - modules are properly registered with
string keys. Can be accessed, iterated, and modified like a dict.

Note: ModuleDict does not define forward() - it's a container that you
use within your own modules.

Args:
    modules: Optional dict of modules to initialize with
    
Example:


.. code-block:: python

    >>> components = ModuleDict({
    ...     'encoder': Linear(10, 5),
    ...     'decoder': Linear(5, 10)
    ... })
    >>> encoded = components['encoder'](x)
    >>> decoded = components['decoder'](encoded)

