# transfer_to

## Signature

```python
nabla.transfer_to(arg: nabla.core.array.Array, device: max._core.driver.Device) -> nabla.core.array.Array
```

## Description

Transfers an array to a different compute device.

This function moves the data of a Nabla array to the specified device
(e.g., from CPU to GPU). If the array is already on the target device,
it is returned unchanged.

Parameters
----------
arg : Array
The input array to transfer.
device : Device
The target device instance (e.g., `nb.Device.cpu()`, `nb.Device.gpu()`).

Returns
-------
Array
A new array residing on the target device.

