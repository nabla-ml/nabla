# exp

## Signature

```python
nabla.exp(arg: 'Array') -> 'Array'
```

## Description

Computes the element-wise exponential function (e^x).

This function calculates the base-e exponential of each element in the
input array.

## Parameters

- **`arg`** (`Array`): The input array.

## Returns

- `Array`: An array containing the exponential of each element.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.array([0.0, 1.0, 2.0])
>>> nb.exp(x)
Array([1.       , 2.7182817, 7.389056 ], dtype=float32)
```
