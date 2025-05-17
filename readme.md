<h1 align="center">NABLA: Differentiable Programming in Mojo</h1>

<p align="center"><em>A Research Preview</em></p>

Nabla brings JIT-accelerated **Automatic Differentiation (AD)** to the Mojo programming language 🔥, a vital technique for gradient-based optimization and physics simulations. Currently, Nabla executes all programs lazily, which - in combination with Mojo's unique memory management capabilities - allows for two quite different programming styles within one framework: functional programming with JAX-like transformations (e.g. vmap, grad, jit) as well as PyTorch-like imperative programming (see examples below).
While Mojo already offers extensive support for NVIDIA/AMD GPUs, Nabla is still limited to CPU execution; our plan is to achieve full GPU integration by Q3 2025. **Note**: Nabla is a research preview - expect further development of the API and rough edges.

<p align="center">
  <a href="https://nablaml.com/docs/get_started">Nabla API</a> •
  <a href="https://docs.modular.com/stable/max/">MAX</a> •
  <a href="https://github.com/nabla-ml/nabla/issues">Report Bug</a>
</p>

## Installation (v25.3)

Get started with Nabla using the Magic package manager.
*(See [Mojo (v25.3) installation guide](https://docs.modular.com/stable/mojo/manual/get-started/).)*

```bash
magic add nabla
```

## Quick Examples

### Imperative programming & explicit graph management

Build trainable neural networks with PyTorch-like syntax:

```python
import nabla

def main():
  # Init params with gradient computation enabled
  weight = nabla.randn((3, 4), DType.float32, requires_grad=True)
  bias = nabla.randn((2, 4), DType.float32, requires_grad=True)
  label = nabla.randn((2, 4), DType.float32)
  input = nabla.randn((2, 3), DType.float32)

  # Compute forward pass (single layer MLP)
  logits = nabla.relu(input @ weight + bias)
  loss =  nabla.sum((logits - label) ** 2)
  print("Loss:", loss)

  # Backward pass to compute gradients
  loss.backward()

  # Update parameters à la SGD
  weight -= 0.01 * weight.grad()
  bias -= 0.01 * bias.grad()
  print("weight:", weight, "bias:", bias)
```

### Functional programming & implicit transformations

Apply JAX-like transformations to pure functions:

```python
import nabla 

def main():
  # Define a simple function
  def foo(args: List[nabla.Array]) -> List[nabla.Array]:
    return List(nabla.sum(nabla.sin(args[0])))

  # create function TRANSFORMATIONS
  # Vectorize the function across the first dimensions
  foo_vmapped = nabla.vmap(foo)

  # first-order derivative transform
  foo_jacobian = nabla.jacrev(foo_vmapped)

  # second-order derivative transform
  foo_hessian = nabla.jacfwd(foo_jacobian)

  # Create input data and compute the hessian
  args = List(nabla.randn((2, 3), DType.float32))
  hessian_output = foo_hessian(args)
  print("Hessian:", hessian_output[0]) # Can you guess the output shape?
```

## Roadmap

Unlike frameworks that retrofit JIT onto eager systems (like PyTorch’s Dynamo), Nabla adopts a slightly different approach: We started this project by building a dynamic compilation system on top of Mojo/MAX first (initially for CPU targets), then added full AD support (forward/reverse modes), and are integrating eager execution after a solid foundation is in place.

Roughly in this order:

- ✅ **Lazy Execution Mode (JIT)**: Compile/optimize program traces.
- ✅ **Custom Ops**: Support custom (differentiable) operations, by defining triplet (maxpr, vjp-rule and jvp-rule).
- ✅ **Core Program Transforms**: Implement vjp, jvp, vmap, backward; Gradient Checkpointing/remat.
- ✅ **Higher-Level Program Transforms**: Implement jacfwd, jacrev, grad.
- 👷 **Enable GPU Execution for Nabla**: **(High Priority, Actively In Progress)**
- 👷 **Eager Execution Mode**: Immediate execution like in PyTorch. (WIP)
- 👷 **Custom Kernels**: Define low-level **CPU/GPU kernels** directly within Mojo.
- 🚧 **Automatic Distributed Execution**: Scale computations across devices (à la pmap/ALPA).
- 🚧 **More ML Primitives**: Expand library core methods with Neural Network modules and models (MLP, Transformer, etc.).
- 🚧 **Python API**: Provide an optional, pip-installable Python package (nabla-py) offering a Pythonic interface to Nabla's core functionalities.
- 💡 **Future Explorations**: Address community needs, advanced features.

## General Status & Caveats (Research Preview)

*   **API Stability:** APIs are subject to change.
*   **Completeness:** Operator coverage is growing but not exhaustive. Feature parity with mature frameworks is not yet achieved.
*   **Performance:** JIT is promising. End-to-end performance tuning is ongoing. GPU performance will be benchmarked once enabled.
*   **Documentation:** Currently basic; will be expanded significantly.
*   **Bugs:** Expect to encounter bugs; please report them!

## Contributing

Contributions welcome! Discuss significant changes in Issues first. Submit PRs for bugs, docs, smaller features.

## License

Nabla is licensed under the [Apache-2.0 license with LLVM Exeptions](https://github.com/nabla-ml/nabla/blob/main/LICENSE).

---

<p align="center" style="margin-top: 3em; margin-bottom: 2em;"><em>Thank you for checking out Nabla!</em></p>

<p align="center">Follow us on <a href="https://twitter.com/nablaml"><strong>X</strong></a> for updates.