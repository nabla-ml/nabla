# negate

## Signature

```python
nabla.negate(arg: 'Array') -> 'Array'
```

## Description

Computes the element-wise numerical negative of an array.

This function returns a new array with each element being the negation
of the corresponding element in the input array. It also provides the
implementation for the unary `-` operator on Nabla arrays.

## Parameters

- **`arg`** (`Array`): The input array.

## Returns

- `Array`: An array containing the negated elements.

## Examples

```python
import nabla as nb
x = nb.array([1, -2, 3.5])
nb.negate(x)
Array([-1.,  2., -3.5], dtype=float32)

Using the `-` operator:
-x
Array([-1.,  2., -3.5], dtype=float32)
```
