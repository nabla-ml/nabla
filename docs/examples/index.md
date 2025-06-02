# Examples Gallery

This section contains practical examples demonstrating Nabla's capabilities across different domains. The gallery includes interactive examples with code and visualizations.

```{toctree}
:maxdepth: 1
:caption: Gallery

../auto_examples/index
```

You can find all the source examples in the [`examples/`](https://github.com/nabla-ml/nabla/tree/main/examples) directory of the GitHub repository.

```{note}
The examples directory includes both beginner-friendly introductions and advanced usage patterns to help you get the most out of Nabla.
```

## Example Categories

### Basic Function Transformations

- **VJP Examples**: Vector-Jacobian products for reverse-mode AD
- **JVP Examples**: Jacobian-vector products for forward-mode AD  
- **VMap Examples**: Vectorization and batching operations

### Neural Networks and Machine Learning

- **MLP Training**: Multi-layer perceptron training from scratch
- **MLP Inference**: Efficient inference with JIT compilation
- **JAX Compatibility**: Comparing Nabla with JAX implementations

### Performance and Compilation

- **JIT Examples**: Just-in-time compilation for acceleration
- **Custom Operations**: Defining custom differentiable operations

### Advanced Use Cases

- **Conditional JIT**: Conditional compilation patterns
- **Custom Kernels**: Integration with Mojo kernels

## Running the Examples

All examples can be run directly:

```bash
cd examples/
python mlp_train.py
python vjp_examples.py
# etc.
```

Or explore them interactively in Jupyter:

```bash
pip install jupyter
jupyter notebook examples/
```
