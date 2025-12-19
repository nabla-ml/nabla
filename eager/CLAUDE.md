# Eager Module: Lazy Eager Execution for MAX

## Overview

The `eager` module implements a **Lazy Eager** execution framework that combines PyTorch's imperative API with JAX's compilation and transformation capabilities, targeting the MAX platform.

**Tagline**: Look imperative, act functional, compile automatically.

---

## Philosophy

**User writes**: Imperative PyTorch-style code
```python
x = Tensor.ones((3, 4))
y = x + x
z = y.sum()
```

**System builds**: Symbolic MAX graph transparently

**System compiles**: On first data access (`.numpy()`, `await`, etc.)

**Result**: No `@jit` decorators needed, but you get compilation benefits.

---

## Module Structure

This module is organized into four logical subdirectories:

### 📁 `core/` - Runtime & State Management
**Contains**: tensor.py, tensor_impl.py, tracing.py, context.py, pytree.py, compute_graph.py, graph_utils.py

**Purpose**: The execution engine and state containers that power lazy eager execution.

**Key mechanisms**:
- Dual-state tensors (unrealized ↔ realized)
- Weakref-based autodiff graph (OutputRefs)
- Global ComputeGraph singleton with epoch tracking
- Shape caching for dynamic batch compilation

**[📖 Read detailed architecture → core/CLAUDE.md](core/CLAUDE.md)**

---

### 📁 `ops/` - Operations Library
**Contains**: operation.py, binary.py, unary.py, creation.py, reduction.py, view.py, multi_output.py, _physical.py

**Purpose**: All tensor operations following a consistent ABC pattern.

**Key mechanisms**:
- Singleton operations with identity-based dispatch
- Automatic metadata propagation (batch_dims, traced, etc.)
- Physical vs logical operation split for vmap
- Multi-output pytree handling

**[📖 Read operation design → ops/CLAUDE.md](ops/CLAUDE.md)**

---

### 📁 `transforms/` - Function Transformations
**Contains**: vmap.py, compile.py

**Purpose**: Higher-order functions that transform user functions.

**Implemented**:
- **vmap**: Automatic vectorization with prefix batch semantics
- **compile**: Computation caching with dynamic dimension support

**Future**: grad, jvp, vjp, scan transforms

**[📖 Read transform internals → transforms/CLAUDE.md](transforms/CLAUDE.md)**

---

### 📁 `sharding/` - Distributed Partitioning
**Contains**: spec.py, propagation.py

**Purpose**: Specify and propagate tensor sharding across device meshes.

**Status**: Infrastructure complete, execution pending MAX multi-device support.

**Key mechanisms**:
- Factor-based sharding propagation (inspired by Shardy/GSPMD)
- Einsum-style operation sharding rules
- Conflict resolution strategies (BASIC vs AGGRESSIVE)

**[📖 Read sharding design → sharding/CLAUDE.md](sharding/CLAUDE.md)**

---

## Key Concepts

### Lazy Eager Execution

**Not JIT**: We build graphs eagerly (no tracing decorators), compile lazily (on data access).

**Benefits**:
- No graph breaks from print/debug statements
- Full graph optimization across operations
- Dynamic shapes via symbolic dimensions
- Deferred compilation until absolutely needed

### Dual-State Tensors

Every tensor can be:
- **Unrealized**: Symbolic MAX graph nodes, no data
- **Realized**: Concrete data in storage, symbolic graph cleared

Shape metadata persists across state transitions, enabling recompilation with different batch sizes.

### The Weakref Trick

Autodiff graph uses weak references to prevent memory leaks while preserving operation provenance. VJP must happen promptly (before GC), which is fine since we walk backwards from loss immediately.

### Batch Dims: Physical vs Logical

For vmap support:
- Physical shape: `(B1, B2, H, W)` - actual tensor shape
- Logical shape: `(H, W)` - what user sees when `batch_dims=2`

Operations automatically preserve batch dimensions, enabling transparent nested vmap.

---

## Design Principles

1. **Separation of Concerns**
   - Tensor: Public API facade
   - TensorImpl: Mutable state container  
   - Operation: Graph construction logic
   - ComputeGraph: Compilation & execution

2. **Immutable Semantics**
   - Operations return NEW tensors with NEW impls
   - Enables functional transformations (vmap, grad)
   - Clean autodiff graph construction

3. **Metadata is First-Class**
   - batch_dims, traced, sharding_spec tracked per tensor
   - Auto-propagates through operations
   - Enables transformation composition

4. **Pytree Everywhere**
   - Multi-output ops return any nested structure
   - Transforms handle dict/list/tuple inputs
   - JAX compatibility

5. **Singletons for Stateless**
   - One operation instance per type globally
   - Fast identity checks
   - Memory efficient

---

## Comparison with Existing Frameworks

| Feature | PyTorch | JAX | Eager Module |
|---------|---------|-----|--------------|
| **Execution** | Eager | Jit-required for perf | Lazy Eager |
| **API Style** | Imperative | Functional | Imperative facade |
| **Compilation** | TorchScript | Always-on | On data access |
| **Transformations** | Limited | Rich (`vmap`, `jvp`, `grad`) | Growing (vmap, compile) |
| **Autodiff** | `.backward()` | `grad()` transform | OutputRefs (VJP planned) |
| **Sharding** | DDP, FSDP | GSPMD (integrated) | Shardy-inspired (infra ready) |
| **Dynamic Shapes** | Via scripting | Limited | First-class (SymbolicDim) |
| **Backend** | PyTorch | XLA | MAX |

---

## Current Status

### ✅ Implemented
- Core lazy eager execution model
- Complete operations library (binary, unary, creation, reduction, view, multi-output)
- vmap transform with nested vmap support
- compile transform with dynamic dimensions
- Sharding specification & propagation (no execution yet)
- Pytree system for nested structures
- OutputRefs autodiff graph infrastructure

### 🚧 In Progress
- Full VJP/backward pass implementation
- Sharded execution (waiting on MAX multi-device)

### 📋 Planned
- grad/vjp/jvp transforms
- Operation fusion optimization
- Graph-level optimizations (CSE, DCE)
- pmap for parallel execution
- scan for stateful loops

---

## Getting Started

### Environment Setup

**Required**: Activate the virtual environment before running any code:

```bash
source venv/bin/activate
python test_*.py
```

### Quick Examples

**Basic Usage**:
```python
import eager

x = eager.Tensor.ones((3, 4))
y = eager.Tensor.arange(12).reshape((3, 4))
z = x + y  # Still symbolic!
result = await z  # Compiles and executes here
```

**Vmap**:
```python
from eager import vmap

def process(x):
    return x * x + 1

batched = vmap(process)
result = batched(eager.Tensor.ones((10, 5)))  # Vectorized!
```

**Compile**:
```python
from eager import compile

@compile(dynamic_dims={0: {0: "batch"}})
def model(x, W, b):
    return (x @ W + b).relu()

# Compiles once, works for any batch size
y1 = model(x_5, W, b)   # Batch size 5
y2 = model(x_10, W, b)  # Batch size 10 (cache hit!)
```

---

## Testing

Run all tests:
```bash
# Vmap tests
python test_vmap.py
python test_vmap_ready.py
python test_vmap_matmul.py

# Compile tests
python test_compile_dynamic.py

# Core tests
python test_eager.py
python test_operation_abc.py

# All others
python test_*.py
```

**Test Coverage**: Comprehensive tests in `test_*.py` cover vmap, compile, operations, and core functionality.

---

## Contributing

### Adding a New Operation

1. Create class inheriting from appropriate ABC in `ops/`
2. Implement `name` property and `maxpr()` method
3. Optionally: `jvp_rule()`, `vjp_rule()`, `sharding_rule()`
4. Create singleton instance and public function
5. Base class handles all metadata propagation automatically

### Adding a New Transform

1. Create file in `transforms/`
2. Define transform function that wraps user function
3. Use pytrees for input/output handling
4. Integrate with GRAPH for compilation if needed
5. Add tests demonstrating composition with other transforms

---

## Architecture FAQs

**Q: Why "Lazy Eager" instead of pure JIT?**  
A: Best of both worlds—imperative debugging + automatic optimization. No decorator soup.

**Q: Why separate Tensor and TensorImpl?**  
A: Enables multi-output ops where multiple Tensors share implementation. Clean state management.

**Q: Why weakrefs in OutputRefs?**  
A: Prevents circular reference memory leaks. VJP happens promptly so intermediates stay alive.

**Q: Why batch_dims instead of JAX's arbitrary axis tracking?**  
A: Prefix semantics are simpler. Batch always at front, operations auto-propagate via max().

**Q: Why separate physical ops from logical ops?**  
A: Users think in logical space, vmap operates in physical space. Clean abstraction boundary.

**Q: Is this production-ready?**  
A: Core execution: yes. Autodiff: infrastructure ready. Sharding: spec done, execution pending.

---

## Future Vision

**Near term** (next 3-6 months):
- Complete VJP/backward implementation
- grad transform for derivatives
- Operation fusion pass
- Sharded execution when MAX multi-device ready

**Medium term** (6-12 months):
- Full autodiff (forward + reverse)
- Advanced transforms (scan, pmap)
- Graph-level optimizations
- Memory planning for large models

**Long term** (1+ years):
- Auto-sharding (no manual annotations)
- Checkpointing for very large models
- Integration with MAX compiler optimizations
- Production deployment tooling

---

## Further Reading

- **[Core Architecture](core/CLAUDE.md)** - Lazy execution, weakrefs, dual state, epoch tracking
- **[Operations Design](ops/CLAUDE.md)** - Singleton pattern, ABC hierarchy, batch propagation
- **[Transforms Guide](transforms/CLAUDE.md)** - Vmap internals, compile caching, dynamic dims
- **[Sharding System](sharding/CLAUDE.md)** - Factor propagation, conflict resolution, future execution

---

## License

Apache 2.0 - See LICENSE file for details.
