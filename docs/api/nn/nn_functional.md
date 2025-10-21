# Functional API (Activations)

## `relu`

```python
def relu(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise Rectified Linear Unit (ReLU) function.

The ReLU function is defined as `max(0, x)`. It is a widely used
activation function in neural networks.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.

**Returns**

`Tensor` – An tensor containing the result of the ReLU operation.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([-2.0, -0.5, 0.0, 1.0, 2.0])
>>> nb.relu(x)
Tensor([0., 0., 0., 1., 2.], dtype=float32)
```

---
## `sigmoid`

```python
def sigmoid(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise sigmoid function.

The sigmoid function, defined as `1 / (1 + exp(-x))`, is a common
activation function that squashes values to the range `(0, 1)`.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.

**Returns**

`Tensor` – An tensor containing the sigmoid of each element.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([-1.0, 0.0, 1.0, 20.0])
>>> nb.sigmoid(x)
Tensor([0.26894143, 0.5       , 0.7310586 , 1.        ], dtype=float32)
```

---
## `tanh`

```python
def tanh(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise hyperbolic tangent of an tensor.

The tanh function is a common activation function in neural networks,
squashing values to the range `[-1, 1]`.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.

**Returns**

`Tensor` – An tensor containing the hyperbolic tangent of each element.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([-1.0, 0.0, 1.0, 20.0])
>>> nb.tanh(x)
Tensor([-0.7615942,  0.       ,  0.7615942,  1.       ], dtype=float32)
```

---
## `softmax`

```python
def softmax(arg: nabla.core.tensor.Tensor, axis: int = -1) -> nabla.core.tensor.Tensor:
```
Computes the softmax function for an tensor.

The softmax function transforms a vector of real numbers into a probability
distribution. Each element in the output is in the range (0, 1), and the
elements along the specified axis sum to 1. It is calculated in a
numerically stable way as `exp(x - logsumexp(x))`.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.
- **`axis`** : `int`, optional, default: `is` – The axis along which the softmax computation is performed. The default
is -1, which is the last axis.

**Returns**

`Tensor` – An tensor of the same shape as the input, containing the softmax
probabilities.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([1.0, 2.0, 3.0])
>>> nb.softmax(x)
Tensor([0.09003057, 0.24472848, 0.66524094], dtype=float32)
```

```python
>>> logits = nb.tensor([[1, 2, 3], [1, 1, 1]])
>>> nb.softmax(logits, axis=1)
Tensor([[0.09003057, 0.24472848, 0.66524094],
       [0.33333334, 0.33333334, 0.33333334]], dtype=float32)
```

---
