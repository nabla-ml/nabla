# Multi-Output

## `split`

```python
def split(x: 'Tensor', num_splits: 'int', axis: 'int' = 0) -> 'list':
```
Split a tensor into multiple equal chunks along an axis.


---
## `chunk`

```python
def chunk(x: 'Tensor', chunks: 'int', axis: 'int' = 0) -> 'list':
```
Split a tensor into a specified number of chunks.


---
## `unbind`

```python
def unbind(x: 'Tensor', axis: 'int' = 0) -> 'list':
```
Remove a dimension and return list of slices.


---
## `minmax`

```python
def minmax(x: 'Tensor') -> 'dict[str, Tensor]':
```
Return both min and max of a tensor as a dict with 'min' and 'max' keys.


---
