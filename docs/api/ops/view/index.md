# View & Shape

## `reshape`

```python
def reshape(x: 'Tensor', shape: 'tuple[int, ...]') -> 'Tensor':
```

---
## `transpose`

```python
def transpose(x: 'Tensor', axis1: 'int', axis2: 'int') -> 'Tensor':
```

---
## `permute`

```python
def permute(x: 'Tensor', order: 'tuple[int, ...]') -> 'Tensor':
```

---
## `unsqueeze`

```python
def unsqueeze(x: 'Tensor', axis: 'int' = 0) -> 'Tensor':
```

---
## `squeeze`

```python
def squeeze(x: 'Tensor', axis: 'int' = 0) -> 'Tensor':
```

---
## `flatten`

```python
def flatten(x: 'Tensor', start_dim: 'int' = 0, end_dim: 'int' = -1) -> 'Tensor':
```
Flatten a range of dimensions into a single dimension using reshape.


---
## `broadcast_to`

```python
def broadcast_to(x: 'Tensor', shape: 'tuple[int, ...]') -> 'Tensor':
```

---
## `concatenate`

```python
def concatenate(tensors: 'Sequence[Tensor]', axis: 'int' = 0) -> 'Tensor':
```

---
## `stack`

```python
def stack(tensors: 'list[Tensor]', axis: 'int' = 0) -> 'Tensor':
```
Stack tensors along a new axis.


---
## `gather`

```python
def gather(x: 'Tensor', indices: 'Tensor', axis: 'int' = 0) -> 'Tensor':
```
Gather elements from x along axis using indices.


---
## `scatter`

```python
def scatter(x: 'Tensor', indices: 'Tensor', updates: 'Tensor', axis: 'int' = 0) -> 'Tensor':
```
Scatter updates into x at indices along axis.


---
## `slice_tensor`

```python
def slice_tensor(x: 'Tensor', start: 'Any', size: 'Any') -> 'Tensor':
```

---
## `slice_update`

```python
def slice_update(x: 'Tensor', update: 'Tensor', start: 'Any', size: 'Any') -> 'Tensor':
```
Update a slice of x with new values.


---
## `moveaxis`

```python
def moveaxis(x: 'Tensor', source: 'int', destination: 'int') -> 'Tensor':
```

---
## `swap_axes`

```python
def swap_axes(x: 'Tensor', axis1: 'int', axis2: 'int') -> 'Tensor':
```

---
## `flip`

```python
def flip(x: 'Tensor', axis: 'int') -> 'Tensor':
```

---
## `pad`

```python
def pad(x: 'Tensor', paddings: 'list[tuple[int, int]]' = None, mode: 'str' = 'constant', value: 'float' = 0.0, **kwargs) -> 'Tensor':
```

---
## `rebind`

```python
def rebind(x: 'Tensor', shape: 'tuple[int, ...]', **kwargs) -> 'Tensor':
```

---
