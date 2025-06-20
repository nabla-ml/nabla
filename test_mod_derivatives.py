import nabla as nb


# Test VJP for modulo operation
def test_vjp():
    print("Testing VJP for modulo operation...")

    def f(x, y):
        return x % y

    # Test with simple values
    x = nb.array(7.0)
    y = nb.array(3.0)

    print(f"x = {x}, y = {y}")
    print(f"x % y = {f(x, y)}")

    # Get VJP
    result, vjp_fn = nb.vjp(f, x, y)
    print(f"Forward result: {result}")

    # Compute gradients with unit cotangent
    grad_x, grad_y = vjp_fn(nb.array(1.0))
    print(f"VJP gradients: grad_x = {grad_x}, grad_y = {grad_y}")

    # Expected: grad_x should be 1, grad_y should be -floor(x/y) = -floor(7/3) = -2
    print("Expected: grad_x = 1, grad_y = -2")


# Test JVP for modulo operation
def test_jvp():
    print("\nTesting JVP for modulo operation...")

    def f(x, y):
        return x % y

    # Test with simple values
    x = nb.array(7.0)
    y = nb.array(3.0)
    dx = nb.array(1.0)  # tangent for x
    dy = nb.array(1.0)  # tangent for y

    print(f"x = {x}, y = {y}")
    print(f"dx = {dx}, dy = {dy}")

    # Get JVP
    result, jvp_result = nb.jvp(f, (x, y), (dx, dy))
    print(f"Forward result: {result}")
    print(f"JVP result: {jvp_result}")

    # Expected: dx - floor(x/y) * dy = 1 - 2 * 1 = -1
    print("Expected JVP result: -1")


if __name__ == "__main__":
    test_vjp()
    test_jvp()
