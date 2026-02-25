# Comparison & Logical

## `equal`

```python
def equal(x: 'Tensor', y: 'Tensor | float | int') -> 'Tensor':
```
Return a boolean tensor: ``True`` where ``x == y`` (element-wise).


---
## `not_equal`

```python
def not_equal(x: 'Tensor', y: 'Tensor | float | int') -> 'Tensor':
```
Return a boolean tensor: ``True`` where ``x != y`` (element-wise).


---
## `greater`

```python
def greater(x: 'Tensor', y: 'Tensor | float | int') -> 'Tensor':
```
Return a boolean tensor: ``True`` where ``x > y`` (element-wise).


---
## `greater_equal`

```python
def greater_equal(x: 'Tensor', y: 'Tensor | float | int') -> 'Tensor':
```
Return a boolean tensor: ``True`` where ``x >= y`` (element-wise).


---
## `less`

```python
def less(x: 'Tensor', y: 'Tensor | float | int') -> 'Tensor':
```
Return a boolean tensor: ``True`` where ``x < y`` (element-wise).


---
## `less_equal`

```python
def less_equal(x: 'Tensor', y: 'Tensor | float | int') -> 'Tensor':
```
Return a boolean tensor: ``True`` where ``x <= y`` (element-wise).


---
## `logical_and`

```python
def logical_and(x: 'Tensor', y: 'Tensor') -> 'Tensor':
```
Compute element-wise logical AND. Both inputs are treated as boolean.


---
## `logical_or`

```python
def logical_or(x: 'Tensor', y: 'Tensor') -> 'Tensor':
```
Compute element-wise logical OR. Both inputs are treated as boolean.


---
## `logical_xor`

```python
def logical_xor(x: 'Tensor', y: 'Tensor') -> 'Tensor':
```
Compute element-wise logical XOR. Returns ``True`` where exactly one input is truthy.


---
## `logical_not`

```python
def logical_not(x: 'Tensor') -> 'Tensor':
```
Compute element-wise logical NOT. Returns ``True`` where *x* is falsy.


---
