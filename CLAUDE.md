# Nabla Project Navigation

> **⚠️ CRITICAL: Before running ANY Python code, activate the virtual environment:**
> ```bash
> source venv/bin/activate
> ```
> **All Python commands, tests, and scripts require this step first.**

---

## Quick Start

**Looking for architecture details?** → [`nabla/CLAUDE.md`](nabla/CLAUDE.md)

**Nabla** is a lazy eager execution framework for MAX, combining PyTorch's imperative API with JAX-inspired transforms (vmap, compile).

When you get 2 GPUs:
```
python -m pytest tests/unit/sharding/test_distributed_gpu.py \
                 tests/unit/debug/test_debug_stress.py \
                 tests/unit/debug/test_sharded_attention.py -v
```

## Project Structure

### 📁 **`nabla/`** - Main Module ⭐
The active tensor computation framework.

**Documentation**: [`nabla/CLAUDE.md`](nabla/CLAUDE.md) - Complete architecture guide

**Submodules with detailed docs**:
- [`core/`](nabla/core/CLAUDE.md) - Runtime engine, dual-state tensors, compute graph
- [`ops/`](nabla/ops/CLAUDE.md) - Operation library with ABC pattern  
- [`transforms/`](nabla/transforms/CLAUDE.md) - vmap, compile transformations
- [`sharding/`](nabla/sharding/CLAUDE.md) - Distributed tensor partitioning

---

### 📁 **`tests/`** - Test Suite
All tests organized by functionality. **Remember: activate venv first!**

**Core & Operations**:
- `test_eager.py` - Basic tensor operations
- `test_operation_abc.py` - Operation ABC pattern
- `test_ops_symbolic.py` - Symbolic dimensions
- `test_refactoring.py` - Architecture validation
- `test_graph_traversal.py` - Graph algorithms

**Transforms**:
- `test_vmap.py` - Basic vmap
- `test_vmap_ready.py` - vmap with batch_dims
- `test_vmap_matmul.py` - vmap + matmul
- `test_vmap_dynamic_batch.py` - vmap with dynamic shapes
- `test_compile_dynamic.py` - compile transform
- `test_dynamic_batch_mlp.py` - End-to-end MLP

**Tracing & Autodiff**:
- `test_tracing.py` - OutputRefs infrastructure
- `test_tracing_infrastructure.py` - Advanced tracing
- `test_verification.py` - Correctness checks

**Sharding**:
- `test_sharding_integration.py` - Sharding propagation

**Run tests**:
```bash
source venv/bin/activate  # ALWAYS DO THIS FIRST

# All tests
python -m pytest tests/

# Specific test
python tests/test_vmap.py

# Category
python -m pytest tests/test_vmap*.py
```

---

### 📁 **`depr/`** - Deprecated Code 🗄️
Old implementations archived here. **Do not use for new development.**

Contains: `depr/nabla/`, `depr/examples/`, `depr/tutorials/`, `depr/tests/`, `depr/docs/`

---

### 📁 **`scripts/`** - Utility Scripts

### 📁 **`assets/`** - Project Assets

---

## Documentation Index

**Start here**: [`nabla/CLAUDE.md`](nabla/CLAUDE.md) - Main architecture overview

**Dive deeper**:
- [`nabla/core/CLAUDE.md`](nabla/core/CLAUDE.md) - Lazy execution, weakrefs, dual-state
- [`nabla/ops/CLAUDE.md`](nabla/ops/CLAUDE.md) - Singleton pattern, batch_dims propagation
- [`nabla/transforms/CLAUDE.md`](nabla/transforms/CLAUDE.md) - Vmap internals, compile caching
- [`nabla/sharding/CLAUDE.md`](nabla/sharding/CLAUDE.md) - Factor propagation, GSPMD-inspired

---

## Quick Example

```bash
# 1. ACTIVATE VENV FIRST (always!)
source venv/bin/activate

# 2. Run Python
python
```

```python
import nabla

# Create tensors (lazy - not computed yet)
x = nabla.Tensor.ones((3, 4))
y = nabla.Tensor.arange(12).reshape((3, 4))
z = x + y

# Force evaluation (compiles + executes)
result = await z
print(result.numpy())
```

---

## FAQ

**Q: I get import errors when running Python**  
A: Did you activate venv? → `source venv/bin/activate`

**Q: Where's the old nabla code?**  
A: In `depr/nabla/`. Current `nabla/` is a complete rewrite.

**Q: Which CLAUDE.md do I read?**  
A: Start with `nabla/CLAUDE.md`, then explore submodules as needed.

**Q: Where are examples?**  
A: Legacy examples in `depr/examples/`. See `tests/` for current usage patterns.

**Q: Tests failing?**  
A: Ensure venv is activated. Run `python -m pytest tests/ -v` for details.

---

## Architecture Summary

```
Imperative Code → Operation Singletons → MAX Graph (lazy) → 
Compile on .numpy()/.await → MLIR Optimization → Executable → Results
```

**Key Innovation**: Build graphs eagerly, compile lazily.

---

**License**: Apache 2.0
