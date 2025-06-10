# Array Creation

Functions for creating new arrays with various initialization patterns.

```{toctree}
:maxdepth: 1

creation_arange
creation_arange_like
creation_array
creation_he_normal
creation_he_uniform
creation_lecun_normal
creation_lecun_uniform
creation_ones
creation_ones_like
creation_rand
creation_rand_like
creation_randn
creation_randn_like
creation_xavier_normal
creation_xavier_uniform
creation_zeros
creation_zeros_like
```

## Quick Reference

### `arange`

```python
nabla.arange(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = ()) -> 'Array'
```

Create an array with evenly spaced values.

### `arange_like`

```python
nabla.arange_like(template: 'Array') -> 'Array'
```

Nabla operation: `arange_like`

### `array`

```python
nabla.array(data: 'list | np.ndarray | float | int', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = ()) -> 'Array'
```

Create a new array from data.

### `he_normal`

```python
nabla.he_normal(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = ()) -> 'Array'
```

Nabla operation: `he_normal`

### `he_uniform`

```python
nabla.he_uniform(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = ()) -> 'Array'
```

Nabla operation: `he_uniform`

### `lecun_normal`

```python
nabla.lecun_normal(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = ()) -> 'Array'
```

Nabla operation: `lecun_normal`

### `lecun_uniform`

```python
nabla.lecun_uniform(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = ()) -> 'Array'
```

Nabla operation: `lecun_uniform`

### `ones`

```python
nabla.ones(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = ()) -> 'Array'
```

Create an array filled with ones.

### `ones_like`

```python
nabla.ones_like(template: 'Array') -> 'Array'
```

Create an array of ones with the same shape as input.

### `rand`

```python
nabla.rand(shape: 'Shape', dtype: 'DType' = float32, lower: 'float' = 0.0, upper: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = ()) -> 'Array'
```

Nabla operation: `rand`

### `rand_like`

```python
nabla.rand_like(template: 'Array', lower: 'float' = 0.0, upper: 'float' = 1.0, seed: 'int' = 0) -> 'Array'
```

Nabla operation: `rand_like`

### `randn`

```python
nabla.randn(shape: 'Shape', dtype: 'DType' = float32, mean: 'float' = 0.0, std: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = ()) -> 'Array'
```

Create an array with random values from normal distribution.

### `randn_like`

```python
nabla.randn_like(template: 'Array', mean: 'float' = 0.0, std: 'float' = 1.0, seed: 'int' = 0) -> 'Array'
```

Nabla operation: `randn_like`

### `xavier_normal`

```python
nabla.xavier_normal(shape: 'Shape', dtype: 'DType' = float32, gain: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = ()) -> 'Array'
```

Nabla operation: `xavier_normal`

### `xavier_uniform`

```python
nabla.xavier_uniform(shape: 'Shape', dtype: 'DType' = float32, gain: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = ()) -> 'Array'
```

Nabla operation: `xavier_uniform`

### `zeros`

```python
nabla.zeros(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = ()) -> 'Array'
```

Create an array filled with zeros.

### `zeros_like`

```python
nabla.zeros_like(template: 'Array') -> 'Array'
```

Create a zero array with the same shape as input.

