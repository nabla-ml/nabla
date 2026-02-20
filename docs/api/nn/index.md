# Neural Networks (nabla.nn)

High-level components for building and training models.

## Modules

Stateful layer modules: `Linear`, `LayerNorm`, `MultiHeadAttention`, `Transformer`, and containers.

```{toctree}
:maxdepth: 1
:hidden:

nn_modules/index
```

## Activations

Activation functions: `relu`, `gelu`, `silu`, `sigmoid`, `softmax`, `tanh`.

```{toctree}
:maxdepth: 1
:hidden:

nn_activations/index
```

## Functional

Stateless layer functions, loss functions (`cross_entropy`, `mse`), and weight initializers.

```{toctree}
:maxdepth: 1
:hidden:

nn_functional/index
```

## Optimizers

Optimization algorithms: `AdamW`, `SGD`, and functional optimizer APIs.

```{toctree}
:maxdepth: 1
:hidden:

nn_optim/index
```

## Fine-tuning

Parameter-efficient fine-tuning: LoRA adapters, NF4 quantization (QLoRA), and checkpoint utilities.

```{toctree}
:maxdepth: 1
:hidden:

nn_finetune/index
```
