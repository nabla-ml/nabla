# cond

## Signature

```python
nabla.cond(condition: 'Tensor', true_fn: 'Callable', false_fn: 'Callable') -> 'Tensor'
```

## Description

Conditionally executes one of two functions.

If `condition` is True, `true_fn` is called; otherwise, `false_fn` is
called. This is a control-flow primitive that allows for conditional
execution within a computational graph. Unlike `nabla.where`, which
evaluates both branches, `cond` only executes the selected function.

## Parameters

- **`condition`** (`Tensor`): A scalar boolean tensor that determines which function to execute.

- **`true_fn`** (`Callable`): The function to be called if `condition` is True.

- **`false_fn`** (`Callable`): The function to be called if `condition` is False.

- **``** (`*args`): Positional arguments to be passed to the selected function.

- **``** (`**kwargs`): Keyword arguments to be passed to the selected function.

## Returns

- `Tensor`: The result of calling either `true_fn` or `false_fn`.

## Examples

```pycon
>>> import nabla as nb
>>> def f(x):
    return x * 2
...
>>> def g(x):
    return x + 10
...
>>> x = nb.tensor(5)
>>> # Executes g(x) because the condition is False
>>> nb.cond(nb.tensor(False), f, g, x)
Tensor([15], dtype=int32)
```
