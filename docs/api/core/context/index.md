# Context & Defaults

## `defaults`

```python
def defaults(dtype: 'DType | None' = None, device: 'Device | None' = None) -> 'tuple[DType, Device]':
```
Get default dtype and device for tensor creation.


---
## `default_device`

```python
def default_device(device: 'Device | graph.DeviceRef') -> 'Generator[None, None, None]':
```
Context manager setting default device.


---
## `default_dtype`

```python
def default_dtype(dtype: 'DType') -> 'Generator[None, None, None]':
```
Context manager setting default dtype.


---
