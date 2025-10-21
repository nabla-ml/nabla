# Adam

## Signature

```python
nabla.nn.Adam
```

**Source**: `nabla.nn.optim.optimizer`

Adam optimizer (Adaptive Moment Estimation).

Implements Adam algorithm with bias correction. Maintains moving averages
of gradients and their squares for adaptive learning rates.

Args:
    params: Parameters to optimize
    lr: Learning rate (default: 0.001)
    betas: Coefficients for computing running averages of gradient
           and its square (default: (0.9, 0.999))
    eps: Term added to denominator for numerical stability (default: 1e-8)
    weight_decay: Weight decay (L2 penalty) (default: 0.0, no decay)
    
Example:
```python
optimizer = Adam(model.parameters(), lr=0.001)

for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

References:
    Adam: A Method for Stochastic Optimization
    Kingma & Ba, ICLR 2015
    https://arxiv.org/abs/1412.6980

