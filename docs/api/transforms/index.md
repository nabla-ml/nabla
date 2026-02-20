# Function Transformations

Purely functional, composable transformations for differentiation, vectorization, compilation, and sharding.

## Differentiation

`grad`, `value_and_grad`, `vjp`, `jvp`, `jacrev`, `jacfwd`, `hessian` — forward and reverse mode AD.

```{toctree}
:maxdepth: 1
:hidden:

differentiation/index
```

## Vectorization

`vmap` — vectorize a function over a batch dimension with zero manual indexing.

```{toctree}
:maxdepth: 1
:hidden:

vectorization/index
```

## Compilation

`jit` — trace and compile a Nabla function to an optimized Mojo/MAX kernel.

```{toctree}
:maxdepth: 1
:hidden:

compilation/index
```

## Distributed

`shard_map` — SPMD partitioning over a `DeviceMesh` for multi-device execution.

```{toctree}
:maxdepth: 1
:hidden:

distributed/index
```
