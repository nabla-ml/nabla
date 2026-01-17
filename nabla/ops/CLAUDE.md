# Nabla Operations Library

[← Back to Root](../CLAUDE.md)

## The Operation Singleton
Every tensor operation is a stateless singleton inheriting from **[`Operation`](operation.py)**.

## Operation Categories

| Type | Base Class | Example | Auto-Behavior |
| :--- | :--- | :--- | :--- |
| **Unary** | **[`UnaryOperation`](unary.py)** | `abs`, `neg`, `exp` | Propagates sharding & batch dims 1:1. |
| **Binary** | **[`BinaryOperation`](binary.py)** | `add`, `mul`, `max` | Auto-broadcasts shapes, batch dims, and sharding. |
| **Reduction** | **[`ReduceOperation`](reduction.py)** | `sum`, `mean`, `max` | Handles axis logic & sharding reduction. |
| **Creation** | **[`Operation`](creation.py)** | `ones`, `zeros`, `arange` | Handles device placement. |
| **Custom** | **[`CustomOp`](custom_op.py)** | User-defined | Easier API for quick extensions. |

## Developer Guide: Adding a New Op

### 1. The Implementation
Create a new class inheriting from the appropriate base in `nabla/ops/`.

```python
from max.graph import TensorValue, ops
from .operation import Operation
from ..sharding.propagation import OpShardingRuleTemplate

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
        return OpShardingRuleTemplate.parse("... i, ... i -> ... i").instantiate(
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
When you call `x + y`, `Operation.__call__`:
1.  **Infer**: Determines output sharding via `sharding_rule`.
2.  **Reshard**: Aligns inputs to compatible layouts.
3.  **Execute**: Runs `maxpr` to add nodes to the graph.
