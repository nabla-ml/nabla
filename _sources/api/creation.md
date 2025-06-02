# Array Creation

Functions for creating new arrays with various initialization patterns.

```{toctree}
:maxdepth: 1
:caption: Functions

creation_arange
creation_array
creation_ones
creation_ones_like
creation_randn
creation_zeros
creation_zeros_like
```

## Quick Reference

### {doc}`arange <creation_arange>`

```python
nabla.arange(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0)) -> 'Array'
```

Create an array with evenly spaced values.

### {doc}`array <creation_array>`

```python
nabla.array(data: 'list | np.ndarray', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0)) -> 'Array'
```

Create a new array from data.

### {doc}`ones <creation_ones>`

```python
nabla.ones(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0)) -> 'Array'
```

Create an array filled with ones.

### {doc}`ones_like <creation_ones_like>`

```python
nabla.ones_like(template: 'Array') -> 'Array'
```

Create an array of ones with the same shape as input.

### {doc}`randn <creation_randn>`

```python
nabla.randn(shape: 'Shape', mean: 'float' = 0.0, std: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0) -> 'Array'
```

Create an array with random values from normal distribution.

### {doc}`zeros <creation_zeros>`

```python
nabla.zeros(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0)) -> 'Array'
```

Create an array filled with zeros.

### {doc}`zeros_like <creation_zeros_like>`

```python
nabla.zeros_like(template: 'Array') -> 'Array'
```

Create a zero array with the same shape as input.

