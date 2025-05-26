#!/usr/bin/env python3
"""Debug exactly where in model creation the segfault occurs."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def debug_model_creation():
    """Debug the model creation process step by step."""
    import nabla
    from nabla.core.execution_context import global_execution_context
    from nabla.core.graph_execution import GraphTracer, ModelFactory

    print("Debugging model creation process...")

    global_execution_context.clear()

    # Create the exact scenario that causes segfault
    x = nabla.arange((2, 2))
    y = nabla.randn((2, 2), seed=42)
    z = nabla.mul(x, y)

    print("1. Getting trace...")
    try:
        inputs, trace, cache_key = GraphTracer.get_trace([z])
        print(f"   ✅ Trace obtained successfully")
        print(f"   Inputs: {len(inputs)}")
        print(f"   Trace: {len(trace)} nodes")
        print(f"   Cache key: {cache_key}")
    except Exception as e:
        print(f"   ❌ Trace failed: {e}")
        return False

    print("\n2. Creating model factory...")
    try:

        def create_model():
            print("     Entering ModelFactory.create_model...")
            return ModelFactory.create_model(inputs, trace, [z])

        print("   Calling get_or_create...")
        model = global_execution_context.get_or_create(cache_key, create_model)
        print(f"   ✅ Model created successfully")
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n3. Executing model...")
    try:
        tensor_inputs = [input_node.impl for input_node in inputs]
        model_outputs = model.execute(*tensor_inputs)
        print(f"   ✅ Model executed successfully")
    except Exception as e:
        print(f"   ❌ Model execution failed: {e}")
        return False

    print("\n✅ All steps completed successfully!")
    return True


if __name__ == "__main__":
    success = debug_model_creation()
    if success:
        print("\n🎉 Debug completed successfully!")
    else:
        print("\n❌ Debug failed!")
