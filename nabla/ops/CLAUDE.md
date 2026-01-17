# Nabla Operations Library

[← Back to Root](../CLAUDE.md)

## The Operation Singleton
Every tensor operation is a stateless singleton inheriting from **[`Operation`](base.py)**.

## Architecture

The `nabla.ops` module is organized into:

*   **`base.py`**: Core `Operation` class and type definitions (`BinaryOperation`, `ReduceOperation`, etc.).
*   **`communication/`**: Distributed collective operations (`all_reduce`, `all_gather`, etc.).
*   **`view/`**: Shape manipulation operations (`reshape`, `transpose`, `slice`, etc.) and batch dimension ops (`batch.py`).
*   **Logical Operations**: Functional files for logical ops (`binary.py`, `unary.py`, `reduction.py`, etc.).

## Operation Categories

| Type | Base Class | Location | Example | Auto-Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Unary** | **[`UnaryOperation`](base.py)** | `unary.py` | `abs`, `neg`, `exp` | Propagates sharding & batch dims 1:1. |
| **Binary** | **[`BinaryOperation`](base.py)** | `binary.py` | `add`, `mul` | Auto-broadcasts shapes, batch dims, and sharding. |
| **Reduction** | **[`ReduceOperation`](base.py)** | `reduction.py` | `sum`, `mean` | Handles axis logic & sharding reduction. |
| **Communication** | **[`CollectiveOperation`](communication/base.py)** | `communication/` | `all_reduce` | Explicit sharding manipulation. |
| **View** | **[`Operation`](base.py)** | `view/` | `reshape` | Manipulate shape/axes without changing data. |
| **Creation** | **[`Operation`](base.py)** | `creation.py` | `ones`, `zeros` | Handles device placement. |

## Developer Guide: Adding a New Op

### 1. The Implementation
Create a new class inheriting from the appropriate base in `nabla/ops/`.

```python
from max.graph import TensorValue, ops
from .base import Operation
from ..core.sharding.propagation import OpShardingRuleTemplate

class MyOp(Operation):
    @property
    def name(self) -> str:
        return "my_op"

    # 1. logical execution (builds MAX graph)
    def maxpr(self, x: TensorValue, y: TensorValue, **kwargs) -> TensorValue:
        return ops.add(x, y) 
    
    # 2. physical propagation (defines sharding)
    def sharding_rule(self, input_shapes, output_shapes, **kwargs):
        # Example: elementwise preservation
        return OpShardingRuleTemplate.parse("... i, ... i -> ... i", input_shapes).instantiate(
            input_shapes, output_shapes
        )
```

### 2. Registration
Instantiate the singleton at the bottom of the file:
```python
my_op = MyOp()
```

### 3. Exposure
Expose a functional API in `nabla/__init__.py` or `nabla/ops/__init__.py`:
```python
def my_op_fn(x, y):
    return binary.my_op(x, y)
```

## The Dispatch Loop
When you call `x + y`, `Operation.__call__` (in `base.py`):
1.  **Infer**: Determines output sharding via `sharding_rule`.
2.  **Reshard**: Aligns inputs to compatible layouts (e.g., auto-resharding for binary ops).
3.  **Execute**: Runs `maxpr` to add nodes to the graph (for each shard).
