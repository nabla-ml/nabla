import nabla as nb


def my_program(x: list[nb.Array]) -> nb.Array:
    a = x[0] * x[1]
    b = nb.cos(a)
    c = nb.log(x[1])
    y = b * c
    return y


# init input arrays
x1 = nb.array([1.0, 2.0, 3.0])
x2 = nb.array([2.0, 3.0, 4.0])
print("x1:\n", x1)
print("x2:\n", x2)

# init input tangents
x1_tangent = nb.randn_like(x1)
x2_tangent = nb.randn_like(x2)
print("x1_tangent:\n", x1_tangent)
print("x2_tangent:\n", x2_tangent)

# compute the actual jvp
value, value_tangent = nb.jvp(my_program, [x1, x2], [x1_tangent, x2_tangent])
print("value:\n", value)
print("value_tangent:\n", value_tangent)


# compute value and pullback function
value, pullback = nb.vjp(my_program, [x1, x2])
print("value:\n", value)

# init output cotangent
value_cotangent = nb.randn_like(value)
print("value_cotangent:\n", value_cotangent)

# compute the actual vjp
x1_cotangent, x2_cotangent = pullback(value_cotangent)
print("x1_cotangent:\n", x1_cotangent)
print("x2_cotangent:\n", x2_cotangent)


jacrev_fn = nb.jacfwd(my_program)
jacobian = jacrev_fn([x1, x2])
print("jacobian:\n", jacobian)

hessian_fn = nb.jacfwd(jacrev_fn)
hessian = hessian_fn([x1, x2])
print("hessian:\n", hessian)


def my_program(x: tuple[nb.Array, nb.Array]) -> nb.Array:
    a = x[0] * x[1]
    b = nb.cos(a)
    c = nb.log(x[1])
    y = b * c
    return y


print("\n\n--- JACREV ---\n")
jacrev_fn = nb.jacrev(my_program)
# print(nb.xpr(jacrev_fn, [x1, x2]))
jacobian = jacrev_fn((x1, x2))
print("jacobian:\n", jacobian)

hessian_fn = nb.jacrev(jacrev_fn)
# print(nb.xpr(hessian_fn, [x1, x2]))
hessian = hessian_fn((x1, x2))
print("hessian:\n", hessian)
