# Modules

Stateful, object-oriented layers and containers.

## Base

Base `Module` class providing parameter registration, pytree support, and `state_dict` serialization.

```{toctree}
:maxdepth: 1
:hidden:

base/index
```

## Linear

`Linear` — fully-connected layer with optional bias, supporting batched and distributed inputs.

```{toctree}
:maxdepth: 1
:hidden:

linear/index
```

## Activations

Activation modules: `ReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax`.

```{toctree}
:maxdepth: 1
:hidden:

activations/index
```

## Containers

`Sequential`, `ModuleList`, `ModuleDict` — composable module containers.

```{toctree}
:maxdepth: 1
:hidden:

containers/index
```
