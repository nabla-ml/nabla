"""Test the new vjp function with arbitrary function signatures."""


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nabla as nb


def test_vjp_simple():
    """Test VJP with a simple function that takes a single Array."""

    def simple_func(x):
        return x * 2.0

    # Create input
    x = nb.array([1.0, 2.0, 3.0])

    # Test VJP
    outputs, vjp_fn = nb.vjp(simple_func, x)

    print("Test 1: Simple function f(x) = x * 2")
    print(f"Input: {x}")
    print(f"Output: {outputs}")

    # Test gradients
    cotangent = nb.array([1.0, 1.0, 1.0])
    gradient = vjp_fn(cotangent)  # Returns single gradient since single arg

    print(f"Cotangent: {cotangent}")
    print(f"Gradient: {gradient}")  # Single gradient for single argument
    print()


def test_vjp_multiple_args():
    """Test VJP with a function that takes multiple arguments."""

    def multi_arg_func(x, y):
        return x * y + x

    # Create inputs
    x = nb.array([1.0, 2.0])
    y = nb.array([3.0, 4.0])

    # Test VJP
    outputs, vjp_fn = nb.vjp(multi_arg_func, x, y)

    print("Test 2: Multi-argument function f(x, y) = x * y + x")
    print(f"Input x: {x}")
    print(f"Input y: {y}")
    print(f"Output: {outputs}")

    # Test gradients
    cotangent = nb.array([1.0, 1.0])
    gradients = vjp_fn(cotangent)  # Returns tuple of gradients for multiple args

    print(f"Cotangent: {cotangent}")
    print(f"Gradient w.r.t. x: {gradients[0]}")
    print(f"Gradient w.r.t. y: {gradients[1]}")
    print()


def test_vjp_with_kwargs():
    """Test VJP with a function that uses keyword arguments."""

    def kwarg_func(x, scale=2.0):
        return x * scale

    # Create inputs
    x = nb.array([1.0, 2.0])
    scale = nb.array([3.0])

    # Test VJP
    outputs, vjp_fn = nb.vjp(kwarg_func, x, scale=scale)

    print("Test 3: Function with kwargs f(x, scale=2.0) = x * scale")
    print(f"Input x: {x}")
    print(f"Input scale: {scale}")
    print(f"Output: {outputs}")

    # Test gradients
    cotangent = nb.array([1.0, 1.0])
    grad_args, grad_kwargs = vjp_fn(cotangent)

    print(f"Cotangent: {cotangent}")
    print(f"Gradient w.r.t. x: {grad_args[0]}")
    print(f"Gradient w.r.t. scale: {grad_kwargs['scale']}")
    print()


def test_vjp_nested_structure():
    """Test VJP with a function that takes nested structures."""

    def nested_func(data):
        x, y = data
        return x + y

    # Create nested input structure
    x = nb.array([1.0, 2.0])
    y = nb.array([3.0, 4.0])
    data = (x, y)

    # Test VJP
    outputs, vjp_fn = nb.vjp(nested_func, data)

    print("Test 4: Function with nested structure f((x, y)) = x + y")
    print(f"Input data: {data}")
    print(f"Output: {outputs}")

    # Test gradients
    cotangent = nb.array([1.0, 1.0])
    gradient = vjp_fn(cotangent)  # Returns single gradient structure since single arg

    print(f"Cotangent: {cotangent}")
    print(f"Gradient structure: {gradient}")
    print(f"Gradient w.r.t. x: {gradient[0]}")
    print(f"Gradient w.r.t. y: {gradient[1]}")
    print()


if __name__ == "__main__":
    try:
        test_vjp_simple()
        test_vjp_multiple_args()
        test_vjp_with_kwargs()
        test_vjp_nested_structure()
        print("All tests completed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
