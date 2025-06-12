# Tutorials

Interactive Jupyter notebooks to learn Nabla's features and capabilities through hands-on examples.

```{toctree}
:maxdepth: 2
:caption: Interactive Notebooks

mlp_training
```

## 🎯 Learning Path

Our tutorials are designed to take you from basic concepts to advanced techniques in Nabla. Each tutorial is a fully interactive Jupyter notebook that you can run locally or in your favorite environment.

## 📚 Available Tutorials

### 🔥 MLP Training: Acceleration Modes

**[→ Open Tutorial](mlp_training)**

Learn how to train neural networks efficiently using Nabla's three execution modes:

- **🐢 Eager Mode** - Direct execution, perfect for debugging and development
- **🏃 Dynamic JIT** - Flexible compilation that handles changing input shapes
- **🚀 Static JIT** - Maximum performance optimization for production workloads

**What You'll Learn:**
- Setting up training loops with different acceleration modes
- Performance benchmarking and profiling
- When to use each execution mode
- Memory efficiency considerations

**Dataset:** Complex 8-period sine function approximation
**Difficulty:** Beginner to Intermediate
**Duration:** ~15 minutes

---

## 🚀 Getting Started

To run these tutorials locally:

1. **Install Nabla** (if not already installed):
   ```bash
   pip install nabla-ml
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter notebook tutorials/
   ```

3. **Open any tutorial** and start learning!

## 💡 Contributing

Have an idea for a new tutorial? We'd love to hear from you! Check out our [GitHub repository](https://github.com/nabla-ml/nabla) to learn how to add new tutorials to the collection.