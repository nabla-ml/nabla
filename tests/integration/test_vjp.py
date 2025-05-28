import nabla


def test_basic_vjp_transform():
    """Test the new VJPTransform class."""
    print("=== Testing VJPTransform ===")

    def square_fn(inputs: list[nabla.Array]) -> list[nabla.Array]:
        x = inputs[0]
        return [x * x * x]

    # Create input
    x = nabla.array([2.0])
    # print("\nOrignal Function:", nabla.xpr(square_fn, [x]))
    print("Value:", square_fn([x])[0])

    values, jacobian = nabla.vjp(square_fn, [x]) 
    # print("\nJacobian:", nabla.xpr(jacobian, [x]))
    cotangent = [nabla.ones(values[0].shape)] 
    d1 = jacobian(cotangent)
    print(f"First-order derivative: {d1[0]}")

    _, hessian = nabla.vjp(jacobian, [x]) 
    cotangent = [nabla.ones(d1[0].shape)] 
    # print("\nHessian:", nabla.xpr(hessian, [x]))
    d2 = hessian(cotangent)
    print(f"Second-order derivative: {d2[0]}")



def test_basic_vjp_transform_with_xpr_prints():
    """Test the new VJPTransform class."""
    print("=== Testing VJPTransform ===")

    def square_fn(inputs: list[nabla.Array]) -> list[nabla.Array]:
        x = inputs[0]
        return [x * x * x]

    # Create input
    x = nabla.array([2.0])
    print("\nOrignal Function:", nabla.xpr(square_fn, [x]))
    print("Value:", square_fn([x])[0])

    values, jacobian = nabla.vjp(square_fn, [x]) 
    print("\nJacobian:", nabla.xpr(jacobian, [x]))
    cotangent = [nabla.ones(values[0].shape)] 
    d1 = jacobian(cotangent)
    print(f"First-order derivative: {d1[0]}")

    _, hessian = nabla.vjp(jacobian, [x]) 
    cotangent = [nabla.ones(d1[0].shape)] 
    print("\nHessian:", nabla.xpr(hessian, [x]))
    d2 = hessian(cotangent)
    print(f"Second-order derivative: {d2[0]}") 


if __name__ == "__main__":
    
    print("Testing Advanced Nabla Transformations")
    # test_basic_vjp_transform()


    print("\nTesting with xpr prints")
    test_basic_vjp_transform_with_xpr_prints()