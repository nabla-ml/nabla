import nabla as nb

# Test basic modulo operation
a = nb.array([7, 8, 9, 10])
b = nb.array([3, 3, 4, 4])

print("a =", a)
print("b =", b)

# Test modulo operation
result = a % b
print("a % b =", result)

# Test with scalars
scalar_result = nb.array(7) % nb.array(3)
print("7 % 3 =", scalar_result)

# Test for boolean generation (the original use case)
test_data = nb.arange((4,)) % 2
print("arange((4,)) % 2 =", test_data)
boolean_data = nb.equal(test_data, nb.array(0))
print("nb.equal(arange((4,)) % 2, 0) =", boolean_data)
print("Type of boolean_data:", type(boolean_data))
