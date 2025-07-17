# API Reference

This page contains the complete API reference for Nabla, organized by functionality.

```{toctree}
:maxdepth: 2
:caption: API Documentation

array
trafos
creation
unary
binary
reduction
linalg
manipulation
```

## Quick Reference

### Core Components

- {doc}`array` - The fundamental Array class with properties and methods
- {doc}`trafos` - Function transformations (jit, vmap, grad, etc.)

### Array Operations

- {doc}`creation` - Array creation functions
- {doc}`unary` - Element-wise unary operations (sin, cos, exp, etc.)
- {doc}`binary` - Element-wise binary operations (add, multiply, etc.)
- {doc}`reduction` - Reduction operations (sum, mean, etc.)
- {doc}`linalg` - Linear algebra operations
- {doc}`manipulation` - Array view and manipulation operations

## Overview

Nabla provides a comprehensive set of APIs for array operations, function transformations, and automatic differentiation:

- **Array Class**: The fundamental `Array` class with its properties, methods, and operator overloading
- **Function Transformations**: Tools like `jit`, `vmap`, `grad`, `vjp`, and `jvp` for compilation and differentiation
- **Array Operations**: Creation, manipulation, and mathematical operations on arrays
