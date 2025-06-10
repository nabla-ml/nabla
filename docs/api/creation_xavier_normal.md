# xavier_normal

## Signature

```python
nabla.xavier_normal(shape: 'Shape', dtype: 'DType' = float32, gain: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = ()) -> 'Array'
```

## Description

Xavier/Glorot normal initialization for sigmoid/tanh activations.

Samples from normal distribution N(0, std²) where std = gain * sqrt(2 / (fan_in + fan_out))

