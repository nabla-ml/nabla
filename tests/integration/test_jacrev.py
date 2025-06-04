import nabla as nb

if __name__ == "__main__":

    def func(x, y):
        return nb.sin(x * y)

    a = nb.arange((2, 3))
    b = nb.arange((2, 3))

    print("Jacobian:")
    jacobian_fn = nb.jacrev(func, argnums=(0, 1))
    jacobian = jacobian_fn(a, b)
    # print(jacobian)
    print("XPR:", nb.xpr(jacobian_fn, a, b))  # This one gets the correct trace!

    print("\nHessians:")
    hessian_fn = nb.jacrev(jacobian_fn, argnums=(0, 1))
    hessian = hessian_fn(a, b)
    # print(hessian)
    print("XPR:", nb.xpr(hessian_fn, a, b))  # This one gets the correct trace!

    print("\nThird Derivative (Jacobian of Hessian):")
    third_derivative_fn = nb.jacrev(hessian_fn, argnums=(0, 1))
    third_derivative = third_derivative_fn(a, b)
    # print(third_derivative)
    print(
        "XPR:", nb.xpr(third_derivative_fn, a, b)
    )  # This should give the third derivative trace!
