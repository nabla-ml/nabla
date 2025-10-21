Device Management
=================

.. currentmodule:: nabla

device
------

Description
-----------

Get a device instance based on the provided device name.

Args:
    device_name: Name of the device (e.g., "cpu", "cuda", "mps")

Returns:
    An instance of the corresponding Device class.

.. autofunction:: nabla.device

cpu
---

Description
-----------

Create a CPU device instance.

Returns:
    An instance of the CPU class.

.. autofunction:: nabla.cpu
