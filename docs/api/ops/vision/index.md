# Convolution & Pooling

## `conv2d`

```python
def conv2d(x: "'Tensor'", filter: "'Tensor'", *, stride: 'int | tuple[int, int]' = (1, 1), dilation: 'int | tuple[int, int]' = (1, 1), padding: 'int | tuple[int, int] | tuple[int, int, int, int]' = (0, 0, 0, 0), groups: 'int' = 1, bias: "'Tensor | None'" = None, input_layout: 'Any' = None, filter_layout: 'Any' = None) -> "'Tensor'":
```

---
## `conv2d_transpose`

```python
def conv2d_transpose(x: "'Tensor'", filter: "'Tensor'", *, stride: 'int | tuple[int, int]' = (1, 1), dilation: 'int | tuple[int, int]' = (1, 1), padding: 'int | tuple[int, int] | tuple[int, int, int, int]' = (0, 0, 0, 0), output_paddings: 'int | tuple[int, int]' = (0, 0), bias: "'Tensor | None'" = None, input_layout: 'Any' = None, filter_layout: 'Any' = None) -> "'Tensor'":
```

---
## `avg_pool2d`

```python
def avg_pool2d(x: "'Tensor'", *, kernel_size: 'int | tuple[int, int]', stride: 'int | tuple[int, int] | None' = None, padding: 'int | tuple[int, int] | tuple[int, int, int, int]' = 0, dilation: 'int | tuple[int, int]' = (1, 1)) -> "'Tensor'":
```

---
## `max_pool2d`

```python
def max_pool2d(x: "'Tensor'", *, kernel_size: 'int | tuple[int, int]', stride: 'int | tuple[int, int] | None' = None, padding: 'int | tuple[int, int] | tuple[int, int, int, int]' = 0, dilation: 'int | tuple[int, int]' = (1, 1)) -> "'Tensor'":
```

---
