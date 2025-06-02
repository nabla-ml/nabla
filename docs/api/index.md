# API Reference

This page contains the complete API reference for Nabla, organized by functionality.

```{toctree}
:maxdepth: 2
:caption: API Documentation

core
creation
unary
binary
reduction
linalg
manipulation
```

## Quick Reference

### Core Components

- {doc}`core` - Array class and function transformations
- {doc}`creation` - Array creation functions

### Operations

- {doc}`unary` - Element-wise unary operations (sin, cos, exp, etc.)
- {doc}`binary` - Element-wise binary operations (add, multiply, etc.)
- {doc}`reduction` - Reduction operations (sum, mean, etc.)
- {doc}`linalg` - Linear algebra operations
- {doc}`manipulation` - Array view and manipulation operations

## Overview

Nabla provides a comprehensive set of APIs for array operations, function transformations, and automatic differentiation:

- **Core Array**: The fundamental `Array` class and its operations
- **Function Transformations**: Tools like `jit`, `vmap`, `jvp`, and `vjp` for compilation and differentiation
- **Array Operations**: Creation, manipulation, and mathematical operations on arrays