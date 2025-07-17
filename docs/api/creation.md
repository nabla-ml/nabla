# Array Creation

Functions for creating new arrays with various initialization patterns.

```{toctree}
:maxdepth: 1
:caption: Functions

creation_arange
creation_array
creation_full_like
creation_glorot_uniform
creation_he_normal
creation_he_uniform
creation_lecun_normal
creation_lecun_uniform
creation_ndarange
creation_ndarange_like
creation_ones
creation_ones_like
creation_rand
creation_rand_like
creation_randn
creation_randn_like
creation_triu
creation_xavier_normal
creation_xavier_uniform
creation_zeros
creation_zeros_like
```

## Quick Reference

### {doc}`arange <creation_arange>`

```python
nabla.arange(start: 'int | float', stop: 'int | float | None' = None, step: 'int | float | None' = None, dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), traced: 'bool' = False, batch_dims: 'Shape' = ()) -> 'Array'
```

Create an array with evenly spaced values.

### {doc}`array <creation_array>`

```python
nabla.array(data: 'list | np.ndarray | float | int', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Array'
```

Create a new array from data.

### {doc}`full_like <creation_full_like>`

```python
nabla.full_like(template: 'Array', fill_value: 'float') -> 'Array'
```

Nabla operation: `full_like`

### {doc}`glorot_uniform <creation_glorot_uniform>`

```python
nabla.glorot_uniform(shape: 'Shape', dtype: 'DType' = float32, gain: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Array'
```

Nabla operation: `glorot_uniform`

### {doc}`he_normal <creation_he_normal>`

```python
nabla.he_normal(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Array'
```

Nabla operation: `he_normal`

### {doc}`he_uniform <creation_he_uniform>`

```python
nabla.he_uniform(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Array'
```

Nabla operation: `he_uniform`

### {doc}`lecun_normal <creation_lecun_normal>`

```python
nabla.lecun_normal(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Array'
```

Nabla operation: `lecun_normal`

### {doc}`lecun_uniform <creation_lecun_uniform>`

```python
nabla.lecun_uniform(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Array'
```

Nabla operation: `lecun_uniform`

### {doc}`ndarange <creation_ndarange>`

```python
nabla.ndarange(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Array'
```

Nabla operation: `ndarange`

### {doc}`ndarange_like <creation_ndarange_like>`

```python
nabla.ndarange_like(template: 'Array') -> 'Array'
```

Nabla operation: `ndarange_like`

### {doc}`ones <creation_ones>`

```python
nabla.ones(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Array'
```

Create an array filled with ones.

### {doc}`ones_like <creation_ones_like>`

```python
nabla.ones_like(template: 'Array') -> 'Array'
```

Create an array of ones with the same shape as input.

### {doc}`rand <creation_rand>`

```python
nabla.rand(shape: 'Shape', dtype: 'DType' = float32, lower: 'float' = 0.0, upper: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Array'
```

Nabla operation: `rand`

### {doc}`rand_like <creation_rand_like>`

```python
nabla.rand_like(template: 'Array', lower: 'float' = 0.0, upper: 'float' = 1.0, seed: 'int' = 0) -> 'Array'
```

Nabla operation: `rand_like`

### {doc}`randn <creation_randn>`

```python
nabla.randn(shape: 'Shape', dtype: 'DType' = float32, mean: 'float' = 0.0, std: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Array'
```

Create an array with random values from normal distribution.

### {doc}`randn_like <creation_randn_like>`

```python
nabla.randn_like(template: 'Array', mean: 'float' = 0.0, std: 'float' = 1.0, seed: 'int' = 0) -> 'Array'
```

Nabla operation: `randn_like`

### {doc}`triu <creation_triu>`

```python
nabla.triu(x, k=0)
```

Nabla operation: `triu`

### {doc}`xavier_normal <creation_xavier_normal>`

```python
nabla.xavier_normal(shape: 'Shape', dtype: 'DType' = float32, gain: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Array'
```

Nabla operation: `xavier_normal`

### {doc}`xavier_uniform <creation_xavier_uniform>`

```python
nabla.xavier_uniform(shape: 'Shape', dtype: 'DType' = float32, gain: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Array'
```

Nabla operation: `xavier_uniform`

### {doc}`zeros <creation_zeros>`

```python
nabla.zeros(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Array'
```

Create an array filled with zeros.

### {doc}`zeros_like <creation_zeros_like>`

```python
nabla.zeros_like(template: 'Array') -> 'Array'
```

Create a zero array with the same shape as input.

