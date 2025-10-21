# transfer_to

## Signature

```python
nabla.transfer_to(arg: nabla.core.tensor.Tensor, device: max._core.driver.Device) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.unary`

## Description

Transfers an tensor to a different compute device.

This function moves the data of a Nabla tensor to the specified device
(e.g., from CPU to GPU). If the tensor is already on the target device,
it is returned unchanged.

## Parameters

- **`arg`** (`Tensor`): The input tensor to transfer.

- **`device`** (`Device`): The target device instance (e.g., `nb.Device.cpu()`, `nb.Device.gpu()`).

## Returns

Tensor
    A new tensor residing on the target device.
