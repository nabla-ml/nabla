# device

## Signature

```python
nabla.device(device_name: str) -> max._core.driver.Device
```

**Source**: `nabla.utils.max_interop`

Get a device instance based on the provided device name.

Args:
    device_name: Name of the device (e.g., "cpu", "cuda", "mps")

Returns:
    An instance of the corresponding Device class.

