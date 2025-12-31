
import sys
from max.graph import Graph, TensorValue, ops
from max.dtype import DType

def test_allreduce_existence():
    print(f"Python version: {sys.version}")
    try:
        from max.graph.ops import allreduce
        print("✅ SUCCESS: max.graph.ops.allreduce is importable!")
        print(f"Docstring: {allreduce.__doc__[:100]}...")
    except ImportError:
        print("❌ FAILURE: max.graph.ops.allreduce NOT found.")
        return

    # Try to verify signature by inspecting
    import inspect
    sig = inspect.signature(allreduce)
    print(f"Signature: {sig}")

if __name__ == "__main__":
    test_allreduce_existence()
