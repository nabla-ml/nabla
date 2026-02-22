# Custom Kernels

## `call_custom_kernel`

```python
def call_custom_kernel(func_name: 'str', kernel_path: 'Union[str, Path, list[Union[str, Path]]]', values: 'Union[TensorValue, list[TensorValue]]', out_types: 'Union[Any, list[Any]]', device: 'None | DeviceRef' = None, **kwargs: 'Any') -> 'Union[TensorValue, list[TensorValue]]':
```
Helper to invoke a custom Mojo kernel, handling library loading automatically.

**Parameters**

- **`func_name`** – The name of the registered Mojo kernel (e.g. @register("name")).
- **`kernel_path`** – Path(s) to the kernel source file or directory.
- **`values`** – Input TensorValue(s).
- **`out_types`** – Expected output type(s).
- **`device`** – Device to run on (default: CPU).
- **`**kwargs`** – Additional arguments passed to ops.custom.

**Returns**

 – Result TensorValue(s).


---
