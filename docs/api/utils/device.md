# Device Management

## `device`

```python
def device(device_name: str) -> max._core.driver.Device:
```
Get a device instance based on the provided device name.

Args:
    device_name: Name of the device (e.g., "cpu", "cuda", "mps")

Returns:
    An instance of the corresponding Device class.

---
## `cpu`

```python
def cpu() -> max._core.driver.Device:
```
Create a CPU device instance.

Returns:
    An instance of the CPU class.

---
## `accelerator`

```python
def accelerator(device_id: int = 0) -> max._core.driver.Device:
```
Create an Accelerator device instance with the specified GPU ID.

Args:
    device_id: GPU ID (default is 0)

Returns:
    An instance of the Accelerator class for the specified GPU.

---
## `accelerator_count`

```python
def accelerator_count() -> int:
```
Get the number of available accelerators (GPUs).

Returns:
    The number of available accelerators.

---
