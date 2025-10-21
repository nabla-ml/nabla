Tensor Creation
===============

.. currentmodule:: nabla

zeros
-----

Description
-----------

Creates an tensor of a given shape filled with zeros.

Parameters
----------
shape : Shape
    The shape of the new tensor, e.g., `(2, 3)` or `(5,)`.
dtype : DType, optional
    The desired data type for the tensor. Defaults to DType.float32.
device : Device, optional
    The device to place the tensor on. Defaults to the CPU.
batch_dims : Shape, optional
    Specifies leading dimensions to be treated as batch dimensions.
    Defaults to an empty tuple.
traced : bool, optional
    Whether the operation should be traced in the graph. Defaults to False.

Returns
-------
Tensor
    An tensor of the specified shape and dtype, filled with zeros.

Examples
--------
.. code-block:: python

    >>> import nabla as nb
    >>> # Create a 2x3 matrix of zeros
    >>> nb.zeros((2, 3), dtype=nb.DType.int32)

Tensor([[0, 0, 0],
       [0, 0, 0]], dtype=int32)

.. autofunction:: nabla.zeros

ones
----

Description
-----------

Creates an tensor of a given shape filled with ones.

Parameters
----------
shape : Shape
    The shape of the new tensor, e.g., `(2, 3)` or `(5,)`.
dtype : DType, optional
    The desired data type for the tensor. Defaults to DType.float32.
device : Device, optional
    The device to place the tensor on. Defaults to the CPU.
batch_dims : Shape, optional
    Specifies leading dimensions to be treated as batch dimensions.
    Defaults to an empty tuple.
traced : bool, optional
    Whether the operation should be traced in the graph. Defaults to False.

Returns
-------
Tensor
    An tensor of the specified shape and dtype, filled with ones.

Examples
--------
.. code-block:: python

    >>> import nabla as nb
    >>> # Create a vector of ones
    >>> nb.ones((4,), dtype=nb.DType.float32)

Tensor([1., 1., 1., 1.], dtype=float32)

.. autofunction:: nabla.ones

rand
----

Description
-----------

Creates an tensor with uniformly distributed random values.

The values are drawn from a continuous uniform distribution over the
interval `[lower, upper)`.

Parameters
----------
shape : Shape
    The shape of the output tensor.
dtype : DType, optional
    The desired data type for the tensor. Defaults to DType.float32.
lower : float, optional
    The lower boundary of the output interval. Defaults to 0.0.
upper : float, optional
    The upper boundary of the output interval. Defaults to 1.0.
device : Device, optional
    The device to place the tensor on. Defaults to the CPU.
seed : int, optional
    The seed for the random number generator. Defaults to 0.
batch_dims : Shape, optional
    Specifies leading dimensions to be treated as batch dimensions.
    Defaults to an empty tuple.
traced : bool, optional
    Whether the operation should be traced in the graph. Defaults to False.

Returns
-------
Tensor
    An tensor of the specified shape filled with random values.

.. autofunction:: nabla.rand
